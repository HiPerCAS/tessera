"""Surrogate-driven multi-objective design optimization beyond exhaustive 5x5.

ask/tell loop driving NSGA-II / TPE / QMC / random over a fixed-topology square
GxG grid; logs hypervolume vs #surrogate-evaluations; includes a same-oracle
exhaustive 5x5 reference and an optional D4 symmetry-dedup distinct-eval counter.

Run (from src/):
  python -m tessera.optimize models/best_model.pth \
      --sizes 5 7 9 11 --samplers nsga2 random --seeds 0 1 2 --n-trials 100000
"""
import argparse
import csv
import itertools
import json
import os
import subprocess
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import optuna
from torch_geometric.loader import DataLoader

from torch_geometric.nn import global_max_pool, global_mean_pool

from tessera.config import get_device, load_config, resolve_path
from tessera.graph_builder import build_inference_data
from tessera.model import TSVPhysicsGNN
from tessera.scaler import InputScaler

cfg = load_config()
DEVICE = get_device(cfg)


def _git(*args):
    try:
        return subprocess.check_output(
            ["git", *args], cwd=str(resolve_path(".")),
            stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return None


def decode_arrangement(params, G, n_signals, rng):
    """Deterministic seeded repair to exactly n_signals signal cells."""
    arr = np.array([params[f"arr_{i}"] for i in range(G * G)], dtype=np.int8).reshape(G, G)
    cur = int((arr == 1).sum())
    if cur == n_signals:
        return arr
    flat = arr.flatten()
    if cur > n_signals:
        # flip extra signals to ground in a seeded order
        idx_signal = np.where(flat == 1)[0]
        order = rng.permutation(idx_signal)
        for k in range(cur - n_signals):
            flat[order[k]] = -1
    else:
        idx_ground = np.where(flat == -1)[0]
        order = rng.permutation(idx_ground)
        for k in range(n_signals - cur):
            flat[order[k]] = 1
    return flat.reshape(G, G)


def d4_canonical(arr):
    """Min over the 8 dihedral images, returned as bytes (a stable hash key)."""
    a = np.ascontiguousarray(arr, dtype=np.int8)
    images = []
    for k in range(4):
        r = np.rot90(a, k)
        images.append(np.ascontiguousarray(r).tobytes())
        images.append(np.ascontiguousarray(np.fliplr(r)).tobytes())
    return min(images)


def build_pop_loader(arrs, fixed, scaler, batch_size):
    data = []
    for i, arr in enumerate(arrs):
        d = build_inference_data({**fixed, "arrangement": arr, "id": i})
        scaler.transform(d)
        data.append(d)
    return DataLoader(data, batch_size=batch_size, shuffle=False)


def _db_t(real, imag):
    mag = torch.sqrt(real ** 2 + imag ** 2)
    return 20.0 * torch.log10(mag + 1e-12)


def eval_population(model, loader, device):
    """Returns N x 4 numpy: [avg_s21, max_s11, max_next, max_fext] (dB)."""
    out = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            pn, pe = model(batch.x, batch.edge_index, batch.edge_attr)
            nm = batch.node_mask
            em = batch.edge_mask
            B = int(batch.batch.max().item()) + 1
            sig_batch = batch.batch[nm]
            s21 = _db_t(pn[nm, 0], pn[nm, 1])
            s11 = _db_t(pn[nm, 2], pn[nm, 3])
            avg_s21 = global_mean_pool(s21, sig_batch, size=B)
            max_s11 = global_max_pool(s11, sig_batch, size=B)
            if em.any():
                edge_batch = batch.batch[batch.edge_index[0][em]]
                nxt = _db_t(pe[em, 0], pe[em, 1])
                fxt = _db_t(pe[em, 2], pe[em, 3])
                max_next = global_max_pool(nxt, edge_batch, size=B)
                max_fext = global_max_pool(fxt, edge_batch, size=B)
            else:
                max_next = torch.full((B,), -100.0, device=device)
                max_fext = torch.full((B,), -100.0, device=device)
            r = torch.stack([avg_s21, max_s11, max_next, max_fext], dim=1).cpu().numpy()
            out.append(r)
    if not out:
        return np.zeros((0, 4))
    return np.concatenate(out, axis=0)


def to_loss(arr_4):
    """Convert process_batch_vectorized's [avg_s21, max_s11, max_next, max_fext] (dB)
    to a MINIMIZATION loss tuple [-avg_s21, max_s11, max_next, max_fext].
    """
    out = arr_4.copy().astype(np.float64)
    out[:, 0] = -out[:, 0]
    return out


def make_sampler(name, seed):
    if name == "nsga2":
        return optuna.samplers.NSGAIISampler(seed=seed)
    if name == "tpe":
        return optuna.samplers.TPESampler(seed=seed, multivariate=True)
    if name == "qmc":
        return optuna.samplers.QMCSampler(seed=seed)
    if name == "random":
        return optuna.samplers.RandomSampler(seed=seed)
    if name == "cmaes":
        return optuna.samplers.CmaEsSampler(seed=seed)
    raise SystemExit(f"unknown sampler {name!r}")


def compute_hv(front_loss, ref_pt):
    """Hypervolume on a Pareto front (minimization losses) given a reference point."""
    if front_loss.size == 0:
        return 0.0
    # Try optuna's helper (varies by version); fall back to a tiny in-line implementation.
    try:
        from optuna._hypervolume import compute_hypervolume
        return float(compute_hypervolume(front_loss, ref_pt))
    except Exception:
        try:
            from optuna._hypervolume.hssp import compute_hypervolume as ch2
            return float(ch2(front_loss, ref_pt))
        except Exception:
            # naive WFG-like for small fronts (last-resort fallback)
            return _hv_naive(front_loss, ref_pt)


def _hv_naive(front, ref):
    # Numerical-integration fallback; OK for the small fronts here.
    front = np.asarray(front, dtype=np.float64)
    ref = np.asarray(ref, dtype=np.float64)
    if front.shape[0] == 0:
        return 0.0
    # 4-D: integrate by sorting on first dim
    front = front[front[:, 0].argsort()]
    hv = 0.0
    prev = ref[0]
    for p in front:
        # treat each point as a step; volume = (prev - p[0]) * remaining 3-D HV
        sub = front[front[:, 0] <= p[0]][:, 1:]
        # 3-D HV (sequential reduction)
        sub = sub[sub[:, 0].argsort()]
        h = 0.0; prev1 = ref[1]
        for q in sub:
            sub2 = sub[sub[:, 0] <= q[0]][:, 1:]
            # 2-D HV
            if sub2.size == 0:
                continue
            sub2 = sub2[sub2[:, 0].argsort()]
            h2 = 0.0; prev2 = ref[2]
            for s in sub2:
                h2 += (prev2 - s[0]) * max(0, ref[3] - s[1])
                prev2 = s[0]
            h += (prev1 - q[0]) * h2
            prev1 = q[0]
        hv += (prev - p[0]) * h
        prev = p[0]
    return max(hv, 0.0)


def pareto_front(loss):
    """Return the non-dominated subset of `loss` (minimization)."""
    n = loss.shape[0]
    if n == 0:
        return loss
    keep = np.ones(n, dtype=bool)
    for i in range(n):
        if not keep[i]:
            continue
        for j in range(n):
            if i == j:
                continue
            if np.all(loss[j] <= loss[i]) and np.any(loss[j] < loss[i]):
                keep[i] = False
                break
    return loss[keep]


def fixed_params(cfg_opt):
    fp = cfg_opt.get("fixed_params", {})
    return {
        "radius": float(fp.get("radius", 5e-6)),
        "pitch": float(fp.get("pitch", 60e-6)),
        "height": float(fp.get("height", 100e-6)),
        "liner": float(fp.get("liner", 5e-7)),
        "temperature": float(fp.get("temperature", 300.0)),
        "freq": float(fp.get("freq", 15e9)),
    }


def run_search(model, scaler, sampler_name, n_trials, seed, G, n_signals,
               use_d4, ref_pt, fixed, batch_size, log_every=64):
    rng = np.random.default_rng(seed)
    sampler = make_sampler(sampler_name, seed)
    directions = ["minimize"] * 4   # all four losses minimized
    study = optuna.create_study(directions=directions, sampler=sampler)
    seen = {}
    history = []
    loss_record = []         # cumulative loss for HV
    n_evals = n_distinct = 0
    BATCH = batch_size
    pop_buf, trial_buf, dup_buf = [], [], []

    while n_evals < n_trials:
        step = min(BATCH, n_trials - n_evals)
        trials = []
        arrs = []
        canon_keys = []
        for _ in range(step):
            t = study.ask({f"arr_{i}": optuna.distributions.CategoricalDistribution([-1, 1])
                           for i in range(G * G)})
            a = decode_arrangement(t.params, G, n_signals, rng)
            trials.append(t); arrs.append(a)
            canon_keys.append(d4_canonical(a))
        # decide which to evaluate
        eval_indices = []
        if use_d4:
            for j, k in enumerate(canon_keys):
                if k not in seen:
                    eval_indices.append(j)
        else:
            eval_indices = list(range(len(arrs)))
        # evaluate
        eval_arrs = [arrs[j] for j in eval_indices]
        if eval_arrs:
            loader = build_pop_loader(eval_arrs, fixed, scaler, batch_size)
            r = eval_population(model, loader, DEVICE)
            losses = to_loss(r)
            for jj, j in enumerate(eval_indices):
                seen[canon_keys[j]] = losses[jj]
        # tell every trial (cached if d4-duplicate)
        for j, t in enumerate(trials):
            l = seen[canon_keys[j]]
            study.tell(t, list(l))
            loss_record.append(l)
        n_evals += len(trials)
        n_distinct += len(eval_indices)
        # log HV at controllable granularity
        if n_evals % log_every == 0 or n_evals >= n_trials:
            front = pareto_front(np.asarray(loss_record))
            hv = compute_hv(front, ref_pt)
            history.append({"sampler": sampler_name, "size": G, "seed": seed,
                            "n_evals": n_evals, "n_distinct": n_distinct,
                            "hypervolume": hv, "front_size": int(front.shape[0])})
    final_front = pareto_front(np.asarray(loss_record))
    return history, final_front, n_distinct


def exhaustive_5x5(model, scaler, n_signals, fixed, batch_size, ref_pt):
    """Same-oracle enumeration of C(25, n_signals) 5x5 designs."""
    cells = 25
    if n_signals < 1 or n_signals > cells:
        raise ValueError("bad n_signals")
    losses = []
    indices = list(itertools.combinations(range(cells), n_signals))
    print(f"[exhaustive 5x5] {len(indices):,} designs")
    BUF = batch_size
    arrs = []
    for k, sig in enumerate(indices):
        a = -np.ones(cells, dtype=np.int8)
        a[list(sig)] = 1
        arrs.append(a.reshape(5, 5))
        if len(arrs) == BUF or k == len(indices) - 1:
            loader = build_pop_loader(arrs, fixed, scaler, BUF)
            r = eval_population(model, loader, DEVICE)
            losses.extend(to_loss(r))
            arrs = []
            if (k + 1) % (BUF * 50) == 0:
                print(f"  ...{k+1:,}/{len(indices):,}", flush=True)
    losses = np.asarray(losses)
    front = pareto_front(losses)
    hv = compute_hv(front, ref_pt)
    return front, hv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint", help="finetuned/pretrained checkpoint path or run-id")
    ap.add_argument("--sizes", type=int, nargs="+", default=[5, 7, 9, 11])
    ap.add_argument("--samplers", nargs="+", default=["nsga2", "random"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--n-trials", type=int, default=100000)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--batch-11x11", type=int, default=512)
    ap.add_argument("--signal-fraction", type=float, default=0.48)
    ap.add_argument("--no-d4", action="store_true")
    ap.add_argument("--no-exhaustive", action="store_true")
    ap.add_argument("--exhaustive-only", action="store_true")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--out-suffix", default="")
    args = ap.parse_args()

    cfg_opt = cfg.get("optimization", {})
    fixed = fixed_params(cfg_opt)
    use_d4 = not args.no_d4
    sig_frac = args.signal_fraction
    ts = time.strftime("%Y%m%d_%H%M%S")
    suffix = f"_{args.out_suffix}" if args.out_suffix else ""
    out = Path(args.out_dir) if args.out_dir else resolve_path(
        f"artifacts/analysis/optimization/{ts}{suffix}")
    out.mkdir(parents=True, exist_ok=True)

    # resolve checkpoint via the same logic as evaluate.py
    ckpt_path = Path(args.checkpoint)
    if ckpt_path.is_dir():
        scaler_path = ckpt_path / "input_scaler.pt"
        ckpt_path = ckpt_path / "best_model.pth"
    elif ckpt_path.exists():
        scaler_path = ckpt_path.parent / "input_scaler.pt"
    else:
        rdir = resolve_path("artifacts/runs") / args.checkpoint
        ckpt_path = rdir / "best_model.pth"
        scaler_path = rdir / "input_scaler.pt"
    scaler = InputScaler(); scaler.load(str(scaler_path))
    model = TSVPhysicsGNN().to(DEVICE)
    model.load_state_dict(torch.load(str(ckpt_path), map_location=DEVICE, weights_only=True))
    model.eval()

    all_history = []
    scal_rows = []
    final_fronts = []
    refs = {}
    for G in args.sizes:
        n_signals = max(1, round(sig_frac * G * G))
        batch = args.batch_11x11 if G >= 10 else args.batch

        # 0. fix reference point per size (max-loss over random init batch + 10%)
        rng = np.random.default_rng(0)
        init_arrs = []
        for _ in range(min(args.batch, 4096)):
            a = -np.ones(G * G, dtype=np.int8)
            idx = rng.choice(G * G, n_signals, replace=False)
            a[idx] = 1
            init_arrs.append(a.reshape(G, G))
        loader = build_pop_loader(init_arrs, fixed, scaler, batch)
        init = eval_population(model, loader, DEVICE)
        init_loss = to_loss(init)
        # Reference point: worst observed loss + 10% of the per-dim range.
        # (loss values can be negative -- multiplying by 1.1 silently inverts;
        # an absolute-range margin is sign-safe.)
        rng_loss = (init_loss.max(axis=0) - init_loss.min(axis=0))
        ref_pt = (init_loss.max(axis=0) + 0.1 * np.maximum(rng_loss, 1e-6)).tolist()
        refs[G] = ref_pt
        print(f"[size {G}x{G}] n_signals={n_signals}  ref_pt={ref_pt}", flush=True)

        # 1. exhaustive 5x5 (same oracle)
        if G == 5 and not args.no_exhaustive:
            front5, hv5 = exhaustive_5x5(model, scaler, n_signals, fixed,
                                         batch, ref_pt)
            np.savetxt(out / "exhaustive_5x5_front.csv", front5,
                       header="loss0_negS21,loss1_maxS11,loss2_maxNEXT,loss3_maxFEXT",
                       delimiter=",", comments="")
            (out / "exhaustive_5x5_hv.json").write_text(json.dumps(
                {"hv": float(hv5), "ref_pt": ref_pt,
                 "front_size": int(front5.shape[0])}))
            print(f"[exhaustive 5x5] HV={hv5:.6g}  front={front5.shape[0]}")
            if args.exhaustive_only:
                continue

        if args.exhaustive_only:
            continue

        # 2. sampler comparisons
        for sampler in args.samplers:
            for seed in args.seeds:
                t0 = time.time()
                history, front, n_dist = run_search(
                    model, scaler, sampler, args.n_trials, seed, G, n_signals,
                    use_d4, ref_pt, fixed, batch)
                wall = time.time() - t0
                all_history.extend(history)
                # final scalability row
                final_hv = compute_hv(front, ref_pt)
                scal_rows.append({
                    "size": G, "sampler": sampler, "seed": seed,
                    "n_signals": n_signals,
                    "search_space": "C({},{})".format(G * G, n_signals),
                    "final_hv": float(final_hv),
                    "n_evals": args.n_trials, "n_distinct": int(n_dist),
                    "wall_s": float(wall), "front_size": int(front.shape[0]),
                })
                final_fronts.append({
                    "size": G, "sampler": sampler, "seed": seed,
                    "front": front,
                })
                print(f"[search {sampler:>6} G={G} seed={seed}] HV={final_hv:.6g} "
                      f"front={front.shape[0]}  distinct={n_dist}  wall={wall:.1f}s")

    # write artifacts
    if all_history:
        keys = list(all_history[0].keys())
        with open(out / "hv_history.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader(); w.writerows(all_history)
    if scal_rows:
        with open(out / "scalability.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=sorted({k for r in scal_rows for k in r}))
            w.writeheader(); w.writerows(scal_rows)
    # HV-vs-evals plots
    if all_history:
        import pandas as pd
        df = pd.DataFrame(all_history)
        for G in args.sizes:
            sub = df[df["size"] == G]
            if sub.empty:
                continue
            fig, ax = plt.subplots(figsize=(8, 5))
            for sampler in sub["sampler"].unique():
                ss = sub[sub["sampler"] == sampler]
                ax.plot(ss["n_evals"], ss["hypervolume"], "o-", alpha=0.4,
                        label=f"{sampler}")
            ax.set_xlabel("# surrogate evaluations")
            ax.set_ylabel("hypervolume")
            ax.set_title(f"Pareto HV vs #evals  |  {G}x{G}")
            ax.legend()
            plt.tight_layout()
            plt.savefig(out / f"hv_vs_evals_{G}.png", dpi=150)
            plt.close(fig)
    # save final fronts
    for r in final_fronts:
        np.savetxt(out / f"final_front_{r['size']}_{r['sampler']}_seed{r['seed']}.csv",
                   r["front"],
                   header="loss0_negS21,loss1_maxS11,loss2_maxNEXT,loss3_maxFEXT",
                   delimiter=",", comments="")
    meta = {
        "git_branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "git_sha": _git("rev-parse", "--short", "HEAD"),
        "checkpoint": str(ckpt_path), "scaler": str(scaler_path),
        "device": str(DEVICE), "fixed_params": fixed,
        "sizes": args.sizes, "samplers": args.samplers, "seeds": args.seeds,
        "n_trials": args.n_trials, "batch": args.batch,
        "batch_11x11": args.batch_11x11, "use_d4": use_d4,
        "signal_fraction": sig_frac, "ref_pt_per_size": refs,
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2, default=str))
    print(f"[optimize] -> {out}")


if __name__ == "__main__":
    main()

#!/usr/bin/env bash
# Run an Ansys HFSS <-> Mechanical sign-off on the AEDT machine.
#
# Prerequisites (on THIS machine):
#   - a licensed Ansys AEDT 2026.1 installation
#   - the [signoff] extra:   pip install -e ".[signoff]"   (installs ansys-aedt-core)
#
# Two-step workflow:
#   1) generate the standalone script (anywhere, no Ansys needed):
#          python examples/signoff_generate.py
#   2) run it here (this script), which launches AEDT and drives the loop.
#
# A full HFSS <-> Mechanical loop can take 1-3 hours of wall time.
set -euo pipefail
cd "$(dirname "$0")/.."   # repo root

SCRIPT="${1:-signoff/signoff_optimal_candidate.py}"

if [[ ! -f "$SCRIPT" ]]; then
    echo "Sign-off script not found: $SCRIPT"
    echo "Generate it first:  python examples/signoff_generate.py"
    echo "Or pass a path:     bash examples/run_signoff.sh path/to/script.py"
    exit 1
fi

# Optional: force the output directory (defaults to the script's own directory).
# export TESSERA_SIGNOFF_OUTPUT_DIR="$PWD/signoff/results"

echo "Running sign-off: $SCRIPT"
python "$SCRIPT"

echo
echo "Done. Result JSON -> $(dirname "$SCRIPT")/signoff_result.json"

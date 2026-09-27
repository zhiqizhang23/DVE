#!/bin/bash
# Run all experiments sequentially.
# Output layout: results/<expN>/dve   for *.yaml
#                results/<expN>/nounl for *-nounl.yaml
# Usage: bash run_all.sh [DEVICE]    (default DEVICE=0)

set -e

DEVICE="${1:-0}"

run() {
    local main="$1"; local config="$2"; local result_dir="$3"
    echo ""
    echo "===== Running: $config -> $result_dir ====="
    python "$main" --device "$DEVICE" --result_dir "$result_dir" --config "$config"
}

# T2I experiments (exp1 ~ exp4.3)
run main_t2i.py  configs/exp1.yaml          results/exp1/dve
run main_t2i.py  configs/exp1-nounl.yaml    results/exp1/nounl
run main_t2i.py  configs/exp2.yaml          results/exp2/dve
run main_t2i.py  configs/exp2-nounl.yaml    results/exp2/nounl
run main_t2i.py  configs/exp3.1.yaml        results/exp3.1/dve
run main_t2i.py  configs/exp3.1-nounl.yaml  results/exp3.1/nounl
run main_t2i.py  configs/exp3.2.yaml        results/exp3.2/dve
run main_t2i.py  configs/exp3.2-nounl.yaml  results/exp3.2/nounl
run main_t2i.py  configs/exp3.3.yaml        results/exp3.3/dve
run main_t2i.py  configs/exp3.3-nounl.yaml  results/exp3.3/nounl
run main_t2i.py  configs/exp4.1.yaml        results/exp4.1/dve
run main_t2i.py  configs/exp4.1-nounl.yaml  results/exp4.1/nounl
run main_t2i.py  configs/exp4.2.yaml        results/exp4.2/dve
run main_t2i.py  configs/exp4.2-nounl.yaml  results/exp4.2/nounl
run main_t2i.py  configs/exp4.3.yaml        results/exp4.3/dve
run main_t2i.py  configs/exp4.3-nounl.yaml  results/exp4.3/nounl

# Edit experiment (exp5)
run main_edit.py configs/exp5.yaml          results/exp5/dve
run main_edit.py configs/exp5-nounl.yaml    results/exp5/nounl

echo ""
echo "All done."

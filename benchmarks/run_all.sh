#!/bin/bash
# Run all P4 encode/decode benchmarks and collect summary
set -e

BIN="${1:-./build_review/ab_test}"
RUNS="${2:-3}"
ITERS="${3:-100000}"

echo "============================================================"
echo " TurboPFor C++ vs C Reference — Full Benchmark Suite"
echo " Binary: $BIN  |  Runs: $RUNS  |  Iters: $ITERS"
echo "============================================================"
echo ""

declare -a SUITE_NAMES
declare -a SUITE_RESULTS

run_suite() {
    local name="$1"
    shift
    echo ">>> $name"
    result=$("$BIN" --runs "$RUNS" --iters "$ITERS" "$@" 2>&1)
    summary=$(echo "$result" | grep -E "average diff:|Grand Avg")
    echo "$result" | grep -E "^Avg|^Grand" | tail -1
    echo ""
    SUITE_NAMES+=("$name")
    SUITE_RESULTS+=("$(echo "$summary" | tail -2)")
}

# 32-bit P4 encode/decode (horizontal, n=1..127)
run_suite "p4enc32 / p4d1dec32 (n=1..127)" --all

# 128v32 SIMD (n=128)
run_suite "p4enc128v32 / p4d1dec128v32 (n=128, SIMD)" --simd128

# 256v32 SIMD (n=256)
run_suite "p4enc256v32 / p4d1dec256v32 (n=256, SIMD)" --simd256

# 64-bit P4 encode/decode (horizontal, n=1..127)
run_suite "p4enc64 / p4d1dec64 (n=1..127)" --p64

# 128v64 delta1 SIMD (n=128)
run_suite "p4enc128v64 / p4d1dec128v64 (n=128, SIMD D1)" --simd128v64d1

# 256v64 delta1 (n=256)
run_suite "p4enc256v64 / p4d1dec256v64 (n=256, D1)" --simd256v64d1

# Low-level bitpack/unpack 32-bit
run_suite "bitpack32 (n=1..127)" --bitpack
run_suite "bitunpack32 (n=1..127)" --bitunpack
run_suite "bitunpackd1_32 (n=1..127)" --bitunpackd1

# Low-level bitpack/unpack 64-bit
run_suite "bitpack64 (n=1..127)" --bitpack64
run_suite "bitunpack64 (n=1..127)" --bitunpack64
run_suite "bitunpackd1_64 (n=1..127)" --bitunpackd1_64

echo ""
echo "============================================================"
echo " SUMMARY: C++ vs TurboPFor C Reference"
echo "============================================================"
for i in "${!SUITE_NAMES[@]}"; do
    printf "  %-45s %s\n" "${SUITE_NAMES[$i]}" "$(echo "${SUITE_RESULTS[$i]}" | grep 'average diff' | sed 's/^/  /')"
done

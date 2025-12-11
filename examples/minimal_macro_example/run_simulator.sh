#!/bin/bash
set -e
HEX_ARCH="v73"
if [ -n "$1" ]; then HEX_ARCH="$1"; fi
if [ -z "$DEFAULT_HEXAGON_TOOLS_ROOT" ]; then echo "Set DEFAULT_HEXAGON_TOOLS_ROOT"; exit 1; fi
if [ -z "$HEXAGON_SDK_ROOT" ]; then echo "Set HEXAGON_SDK_ROOT"; exit 1; fi
if [ -z "$DEFAULT_TOOLS_VARIANT" ]; then echo "Set DEFAULT_TOOLS_VARIANT"; exit 1; fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="$SCRIPT_DIR/hexagon_${DEFAULT_TOOLS_VARIANT}_${HEX_ARCH}"
SO_NAME="libminimal_macro_q.so"

mkdir -p "$BUILD_DIR"
# create simple simulation cfgs pointing to tools used by other examples
echo "$DEFAULT_HEXAGON_TOOLS_ROOT/Tools/lib/iss/qtimer.so --csr_base=0xFC900000 --irq_p=1 --freq=19200000 --cnttid=1" > "$BUILD_DIR/q6ss.cfg"
echo "$DEFAULT_HEXAGON_TOOLS_ROOT/Tools/lib/iss/l2vic.so 32 0xab010000" >> "$BUILD_DIR/q6ss.cfg"
echo "$HEXAGON_SDK_ROOT/rtos/qurt/compute${HEX_ARCH}/debugger/lnx64/qurt_model.so" > "$BUILD_DIR/osam.cfg"

$DEFAULT_HEXAGON_TOOLS_ROOT/Tools/bin/hexagon-sim \
  -m${HEX_ARCH}na_1 --simulated_returnval --usefs "$BUILD_DIR" \
  --pmu_statsfile "$BUILD_DIR/pmu_stats.txt" --cosim_file "$BUILD_DIR/q6ss.cfg" \
  --l2tcm_base 0xd800 --rtos "$BUILD_DIR/osam.cfg" \
  "$HEXAGON_SDK_ROOT/rtos/qurt/compute${HEX_ARCH}/sdksim_bin/runelf.pbn" \
  -- "$HEXAGON_SDK_ROOT/libs/run_main_on_hexagon/ship/hexagon_${DEFAULT_TOOLS_VARIANT}_${HEX_ARCH}/run_main_on_hexagon_sim" \
  --"$BUILD_DIR/$SO_NAME" 100

echo "Simulator finished, pmu stats: $BUILD_DIR/pmu_stats.txt"

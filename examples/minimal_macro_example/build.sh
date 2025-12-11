#!/bin/bash
set -e
HEX_ARCH="v73"
if [ -n "$1" ]; then HEX_ARCH="$1"; fi
if [ -z "$HEXAGON_SDK_ROOT" ]; then echo "Set HEXAGON_SDK_ROOT"; exit 1; fi
if [ -z "$DEFAULT_HEXAGON_TOOLS_ROOT" ]; then echo "Set DEFAULT_HEXAGON_TOOLS_ROOT"; exit 1; fi
if [ -z "$DEFAULT_TOOLS_VARIANT" ]; then echo "Set DEFAULT_TOOLS_VARIANT"; exit 1; fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXE_BUILD_DIR="$SCRIPT_DIR/hexagon_${DEFAULT_TOOLS_VARIANT}_${HEX_ARCH}"
mkdir -p "$EXE_BUILD_DIR"
NPU_CC=$DEFAULT_HEXAGON_TOOLS_ROOT/Tools/bin/hexagon-clang

# compile
$NPU_CC -I$HEXAGON_SDK_ROOT/rtos/qurt/compute${HEX_ARCH}/include \
        -I$SCRIPT_DIR/../..//include \
        -m${HEX_ARCH} -G0 -mhvx -mhvx-length=128B -O3 -fPIC -c $SCRIPT_DIR/src/main.c -o $EXE_BUILD_DIR/main.c.obj

# link against libhexkl_macro.a
$NPU_CC -m${HEX_ARCH} -G0 -fpic -shared -o $EXE_BUILD_DIR/libminimal_macro_q.so $EXE_BUILD_DIR/main.c.obj $SCRIPT_DIR/../../lib/hexagon_toolv19_${HEX_ARCH}/libhexkl_macro.a -lc

echo "Built: $EXE_BUILD_DIR/libminimal_macro_q.so"

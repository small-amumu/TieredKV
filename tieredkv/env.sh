# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Zijie Hu
# Put the flat module directories of the library on PYTHONPATH. Usage, from anywhere:
#   source tieredkv/env.sh
# The modules import each other by bare name (import sel14_q, import kalign_opx), as they did on the
# run machines, so every directory below must be on the path.
_TKV_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="$_TKV_ROOT/reference:$_TKV_ROOT/kernels:$_TKV_ROOT/engine:$_TKV_ROOT/bench${PYTHONPATH:+:$PYTHONPATH}"
unset _TKV_ROOT

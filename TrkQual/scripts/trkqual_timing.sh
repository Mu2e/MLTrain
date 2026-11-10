#!/bin/bash
# usage: trkqual_timing.sh MUSING OUTDIR FEATURES NFEATURES REPEATS MODEL...
#
# Builds trkqual_timing.cc against the ONNXRuntime and XGBoost of a Musing (the ones art uses) and
# runs it.  trkqual_check_art.py --timing calls it in a clean environment (env -i), so it sets up
# Mu2e and the Musing itself.  The JSON lines of trkqual_timing go to stdout; everything else to stderr.
[ $# -ge 6 ] || { echo "usage: trkqual_timing.sh MUSING OUTDIR FEATURES NFEATURES REPEATS MODEL..." >&2; exit 2; }
MUSING=$1; OUTDIR=$2; shift 2
SRC=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/trkqual_timing.cc

# set -e only after the setup scripts, which are not written for it
source /cvmfs/mu2e.opensciencegrid.org/setupmu2e-art.sh >&2 || { echo "trkqual_timing.sh: Mu2e setup failed" >&2; exit 1; }
muse setup $MUSING >&2 || { echo "trkqual_timing.sh: muse setup $MUSING failed" >&2; exit 1; }
set -e
VIEW=$SPACK_ENV/.spack-env/view
for f in $VIEW/include/onnxruntime/onnxruntime_cxx_api.h $VIEW/include/xgboost/c_api.h; do
    [ -f $f ] || { echo "trkqual_timing.sh: $f not found in the $MUSING environment" >&2; exit 1; }
done

export OMP_NUM_THREADS=1   # as in the README's art job
g++ -O2 -std=c++17 -I$VIEW/include -o $OUTDIR/trkqual_timing $SRC \
    -L$VIEW/lib64 -Wl,-rpath,$VIEW/lib64 -lonnxruntime -lxgboost >&2
echo "trkqual_timing.sh: built with $(g++ --version | head -1); $(readlink -f $VIEW/lib64/libonnxruntime.so)" >&2
$OUTDIR/trkqual_timing "$OUTDIR" "$@"

import os

def cpu_limit():
    # For some reason, limiting Python to a single thread improve speed by a lot
    os.environ["OMP_NUM_THREADS"] = "1"  # export OMP_NUM_THREADS=4
    os.environ["OPENBLAS_NUM_THREADS"] = "1"  # export OPENBLAS_NUM_THREADS=4
    os.environ["MKL_NUM_THREADS"] = "1"  # export MKL_NUM_THREADS=6
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"  # export VECLIB_MAXIMUM_THREADS=4
    os.environ["NUMEXPR_NUM_THREADS"] = "1"  # export NUMEXPR_NUM_THREADS=6
    os.environ["BLIS_NUM_THREADS"] = "1"
    os.environ["NUMBER_OF_PROCESSORS"] = "1"

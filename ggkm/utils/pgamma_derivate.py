import ctypes
import numpy as np

from pathlib import Path

HERE = Path(__file__).resolve().parent
LIB_PATH = HERE / "libspecfun.so"

_lib = ctypes.CDLL(str(LIB_PATH))
_lib.pgamma_1st_derivative.argtypes = [
    ctypes.c_double,
    ctypes.c_double,
    ctypes.c_double,
]
_lib.pgamma_1st_derivative.restype = ctypes.c_double


def pgamma_shape_derivative(x, shape, scale=1.0):
    return _lib.pgamma_1st_derivative(
        float(x),
        float(shape),
        float(scale),
    )


def pgamma_shape_derivative_vec(u, shape, scale=1.0):
    u = np.asarray(u, dtype=float)

    return np.array(
        [pgamma_shape_derivative(ui, shape, scale) for ui in u],
        dtype=float,
    )

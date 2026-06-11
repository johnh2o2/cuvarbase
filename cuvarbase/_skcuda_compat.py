"""numpy compatibility shim for scikit-cuda.

scikit-cuda 0.5.3 (last release 2019) references numpy aliases that were
removed in numpy 1.24 / 2.x (``np.typeDict``, ``np.float``, ``np.int``,
``np.complex``, ``np.sctypes``). Call :func:`ensure_numpy_aliases` before
``import skcuda`` to restore them; on older numpy versions where the
aliases still exist this is a no-op.
"""
import numpy as np


def ensure_numpy_aliases():
    if not hasattr(np, 'typeDict'):
        np.typeDict = np.sctypeDict

    # Exactly the removed aliases scikit-cuda 0.5.3 references
    for name, alias in (('float', float), ('int', int),
                        ('complex', complex)):
        if name not in np.__dict__:
            setattr(np, name, alias)

    if not hasattr(np, 'sctypes'):
        np.sctypes = {
            'int': [np.int8, np.int16, np.int32, np.int64],
            'uint': [np.uint8, np.uint16, np.uint32, np.uint64],
            'float': [np.float16, np.float32, np.float64],
            'complex': [np.complex64, np.complex128],
            'others': [bool, object, bytes, str, np.void],
        }

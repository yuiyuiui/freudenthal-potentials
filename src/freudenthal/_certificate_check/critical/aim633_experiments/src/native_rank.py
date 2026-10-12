"""Optional compiled sparse modular rank. Falls back to the Python implementation.
Compilation uses the local C++ compiler only; no downloaded code is needed.
"""
from pathlib import Path
import ctypes,subprocess,os
import numpy as np
from scipy import sparse
from algebra import SparseEchelon

_lib=None

def load_lib():
    global _lib
    if _lib is not None:return _lib
    src=Path(__file__).with_suffix('.cpp');lib=src.with_name('_modrank_core.so')
    if not lib.exists() or lib.stat().st_mtime<src.stat().st_mtime:
        subprocess.run(['g++','-O3','-std=c++17','-shared','-fPIC',str(src),'-o',str(lib)],check=True)
    L=ctypes.CDLL(str(lib));L.rank_new.restype=ctypes.c_void_p
    L.rank_new.argtypes=[ctypes.c_int,ctypes.c_int,np.ctypeslib.ndpointer(dtype=np.int32,flags='C_CONTIGUOUS')]
    for name in ['rank_delete','rank_value','rank_max_nnz','rank_ops']:
        getattr(L,name).argtypes=[ctypes.c_void_p]
    L.rank_value.restype=ctypes.c_int;L.rank_max_nnz.restype=ctypes.c_int;L.rank_ops.restype=ctypes.c_int64
    L.rank_add_csr.argtypes=[ctypes.c_void_p,ctypes.c_int,np.ctypeslib.ndpointer(dtype=np.int64,flags='C_CONTIGUOUS'),
                            np.ctypeslib.ndpointer(dtype=np.int32,flags='C_CONTIGUOUS'),np.ctypeslib.ndpointer(dtype=np.int64,flags='C_CONTIGUOUS')]
    _lib=L;return L

class FastRank:
    def __init__(self,ncols,p=1000003,priority=None):
        if not (2<=p<=1000000007):raise ValueError('prime outside native safe range')
        self.ncols=ncols;self.p=p;self.fallback=None;self.handle=None
        order=np.arange(ncols,dtype=np.int32) if priority is None else np.argsort(np.argsort(priority)).astype(np.int32)
        try:
            self.lib=load_lib();self.handle=self.lib.rank_new(ncols,p,order)
        except (OSError,subprocess.CalledProcessError):self.fallback=SparseEchelon(ncols,p,priority)
    def __del__(self):
        if self.handle:self.lib.rank_delete(self.handle);self.handle=None
    @property
    def rank(self):return self.fallback.rank if self.fallback is not None else self.lib.rank_value(self.handle)
    @property
    def max_nnz(self):return self.fallback.max_nnz if self.fallback is not None else self.lib.rank_max_nnz(self.handle)
    def add_matrix_rows(self,A):
        if self.rank==self.ncols:return
        A=A.tocsr();A.sum_duplicates();A.sort_indices()
        if self.fallback is not None:self.fallback.add_matrix_rows(A);return
        ptr=np.ascontiguousarray(A.indptr,dtype=np.int64);idx=np.ascontiguousarray(A.indices,dtype=np.int32);val=np.ascontiguousarray(A.data,dtype=np.int64)
        self.lib.rank_add_csr(self.handle,A.shape[0],ptr,idx,val)
    def add(self,row):
        from algebra import coo_from_rows
        before=self.rank;self.add_matrix_rows(coo_from_rows([row],self.ncols));return self.rank>before

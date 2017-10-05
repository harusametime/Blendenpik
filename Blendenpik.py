'''
Created on 2017/10/05

@author: samejima
'''

import numpy as np
import xalglib
import scipy.sparse as spmat
import ctypes
import time
import math

class Blendenpik(object):
    '''
    classdocs
    '''


    def __init__(self):
        '''
        Constructor
        '''
        
        A = spmat.rand(10000,10000, density = 0.001, format = "csc")
        self.m_tilde = math.ceil(A.shape[0]/1000)*1000 
        self.FHT2D(A)
    
    
    def FHT2D(self, x):
        '''
        FHT2D is FHT for a 2-d array, which is separable to
        the following FHTs for 1-d arrays.
        - First, FHT for each row of the 2-d array.
        - Second, FHT for each column of the 2-d array.
        We use Alglib FHT for 1-d array.
        '''
        n_row = n_col = x.shape[0]
        for r in range(n_row):
            x[r,:] = self.fhtr1d(x[r,:].todense(), n_row)
        for c in range(n_col):
            x[:,c].T = self.fhtr1d(x[:,c].todense(), n_col)
          
    def fhtr1d(self, x, n):
        '''
        This is a wrapper for fhtr1d in ALglib.
        Although xalglib.fhtr1d has python interface, the interface can receive python list.
        This wrapper allows Alglib to receive and return numpy array.
        '''
        _error_msg = ctypes.c_char_p(0)
        __c = ctypes.c_void_p(0)
        __x = xalglib.x_vector(cnt=n, datatype=xalglib.DT_REAL, owner=xalglib.OWN_CALLER, 
                      last_action=0,ptr=xalglib.x_multiptr(p_ptr=x.ctypes.data))
        __n = xalglib.c_ptrint_t(n)
        
        xalglib._lib_alglib.alglib_fhtr1d(
            ctypes.byref(_error_msg), 
            ctypes.byref(__x), 
            ctypes.byref(__n))
        
        INTP = ctypes.POINTER(ctypes.c_double)
        ptr = ctypes.cast(__x.ptr.p_ptr, INTP)
        return np.fromiter(ptr, dtype=np.float, count=n)
                   
if __name__ == '__main__':    
    b = Blendenpik()

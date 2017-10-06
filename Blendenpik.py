'''
Created on 2017/10/05

@author: samejima
'''

import numpy as np
import xalglib
import scipy.sparse as spmat
import scipy.fftpack as fft
import scipy.linalg as lalg
import ctypes
import time
import math
import sparseqr



class Blendenpik(object):
    '''
    classdocs
    '''


    def __init__(self):
        '''
        Constructor
        '''
        
        self.gamma = 1.5 #from 1.5 to 10
        
        A = spmat.rand(20000,10000, density = 0.001, format = "lil")
        self.m_tilde = math.ceil(A.shape[0]/1000)*1000
        
        if self.m_tilde > A.shape[0]:
            zero_mat = spmat.coo_matrix((self.m_tilde-A.shape[0], A.shape[1]))
            M = spmat.bmat([[A], [zero_mat]],format = "lil")
        else:
            M = A
        
        diag_D = np.random.choice(2, self.m_tilde)
        D = spmat.diags(diag_D)
        
        M = self.DCT2D(D*M)
        diag_S = np.random.choice(2, self.m_tilde, p=[1- self.gamma*A.shape[1]/self.m_tilde,self.gamma*A.shape[1]/self.m_tilde])
        S = spmat.diags(diag_S)
        SM = S*M
    
        #Q, R = lalg.qr(SM)
        Q, R = np.linalg.qr(SM)
        print(time.process_time())
        
        
        
        

    def DCT2D(self, x):
        Y = fft.fftn(x.toarray(), shape=x.shape).real
        return Y
        
    def FHT2D(self, x):
        '''
        FHT2D is FHT for a 2-d array, which is separable to
        the following FHTs for 1-d arrays.
        - First, FHT for each row of the 2-d array.
        - Second, FHT for each column of the 2-d array.
        We use Alglib FHT for 1-d array.
        '''
        n_row = x.shape[0]
        n_col = x.shape[1]
        for r in range(n_row):
            #y = self.fhtr1d(x[r,:].todense(), n_col)
            index = x[r,:].nonzero()[1]
            if len(index) > 0:
                x[r,:] = self.fhtr1d(x[r,:].todense(), n_col)
            #print("x", x[r,:])
            
        for c in range(n_col):
            index = x[:,c].nonzero()[1]
            if len(index) > 0:
                x[:,c].T = self.fhtr1d(x[:,c].todense(), n_row)
        
        
        
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

'''
Created on 2017/10/05

@author: samejima
'''

class Blendenpik(object):
    '''
    classdocs
    '''
    
    use_xalglib = False
    use_sparseqr = False
    
    import numpy as np
    import scipy.sparse as spmat
    import scipy.fftpack as fft
    import scipy.linalg as lalg
    import scipy.sparse.linalg as splalg
    import ctypes
    import time
    import math
    
    if use_xalglib:
        import xalglib
    if use_sparseqr:
        import sparseqr


    def __init__(self, A, b, gamma = 1.5):
        '''
        Constructor
        '''
        
        self.A = A
        self.b = b
        self.gamma = gamma #from 1.5 to 10
        
    def solve(self):
        
        self.m_tilde = math.ceil(self.A.shape[0]/1000)*1000
        
        if self.m_tilde > self.A.shape[0]:
            zero_mat = spmat.coo_matrix((self.m_tilde-self.A.shape[0], self.A.shape[1]))
            M = spmat.bmat([[A], [zero_mat]],format = "lil")
        else:
            M = self.A
        
        diag_D = np.random.choice(2, self.m_tilde)
        D = spmat.diags(diag_D)
        
        # Choose from DCT or FHT
        # FHT is recommended by the original paper. But my implementation with alglib
        # is not so fast.
        M = self.DCT2D(D*M)
        
        sampled_rate = min(1, self.gamma*self.A.shape[1]/self.m_tilde)
        sampled_rows = np.random.choice(self.m_tilde, int(sampled_rate * self.m_tilde))
        sampledM = M[sampled_rows,:]
    
        # For QR decomposition, 
        # numpy is the fastest followed by scipy, sparseqr.
        Q, R = np.linalg.qr(sampledM)
        
        z = splalg.lsqr(self.A * np.linalg.inv(R), self.b)[0]
        x = splalg.lsqr(R, z)[0]
        return x
    
    def DCT2D(self, x):
        Y = fft.fftn(x.toarray(), shape=x.shape).real
        return Y
    

    '''
    This code requires Alglib for FHT.
    '''
    if use_xalglib:        
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
            Although xalglib.fhtr1d has python interface, the interface receives only a python list.
            This wrapper allows Alglib to receive and return a numpy array.
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
    
    A = spmat.rand(2000,1000, density = 0.1, format = "lil")
    x_true = np.random.rand(1000)
    b = A * x_true
    blendenpik = Blendenpik(A, b)
    x = blendenpik.solve()
    print("Residual (L2-norm):", np.linalg.norm(x-x_true,ord =2))
    print("Computational time (sec.):",time.process_time())

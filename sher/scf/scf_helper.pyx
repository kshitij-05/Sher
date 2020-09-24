# cython: infer_types=True
import math
import numpy as np
cimport cython
from scipy.special import hyp1f1
DTYPE = np.double
DTYPE_int = np.intc
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)


cpdef make_density(int no_of_electrons,double[:,::1] C):
	n_basis = C.shape[0]
	D=np.zeros((n_basis,n_basis),dtype=DTYPE)
	cdef double[:,::1] D_view = D
	nbasis = C.shape[0]
	cdef Py_ssize_t i,j,m 
	for i in range(nbasis):
		for j in range(nbasis):
			for m in range(int(no_of_electrons/2)):
				D_view[i,j] += C[i,m]*C[j,m]

	return D


cpdef make_fock(double[:,::1] D,double[:,::1] hamil,double[:,:,:,::1] eri):
	n_basis = D.shape[0]
	cdef Py_ssize_t i , j ,k ,l 
	Fock=np.zeros((n_basis,n_basis) , dtype=DTYPE)
	cdef double[:,::1] Fock_view = Fock
	for i in range(n_basis):
		for j in range(n_basis):
			Fock_view[i][j] = hamil[i][j] 
			for k in range(n_basis):
				for l in range(n_basis):
					Fock_view[i][j] += D[k][l]*(  2.0*eri[i][j][k][l] - eri[i][k][j][l]  )
	return Fock




cpdef  get_X(double[:,::1] S):
	nbasis = S.shape[0]
	lambda_b,L_s=np.linalg.eigh(S)
	cdef double[::1] lambda_b_view = lambda_b
	cdef double[:,::1] L_s_view =L_s
	X_temp= np.zeros([nbasis,nbasis] , dtype=DTYPE)
	cdef double[:,::1] X_temp_view = X_temp 
	temp= np.zeros([nbasis,nbasis] , dtype=DTYPE)
	cdef double[:,::1] temp_view =temp
	cdef Py_ssize_t i
	for i in range(nbasis):
		temp_view[i][i]=(lambda_b_view[i])**(-0.5)
	X_temp_view=np.matmul(L_s_view,temp_view)
	X=np.matmul(X_temp_view,L_s.transpose())

	return X

cpdef  deltae(double E,double OLDE):
	return abs(E-OLDE)

cpdef scf_energy(double[:,::1] P,double[:,::1] Hcore,double[:,::1] F):
	N = P.shape[0]
	cdef double Energy=0.0
	cdef Py_ssize_t i,j
	for i in range(N):
		for j in range(N):
			Energy  +=  P[i,j]*(Hcore[i,j]  +  F[i,j])

	return Energy


cpdef make_C(double[:,::1] s_inv_root ,  double[:,::1] Fock):
	fock_ini=np.linalg.multi_dot([np.transpose(s_inv_root),Fock,s_inv_root])
	cdef double[:,::1] fock_ini_view = fock_ini
	E ,C_dash = np.linalg.eigh(fock_ini_view)
	cdef double[:,::1] C_dash_view = C_dash
	C=np.matmul(s_inv_root,C_dash_view)
	return E,C


cpdef rmsd(double[:,::1] D1,double[:,::1] D2):
	cdef double Sum=0.0
	cdef double delta
	n_basis = D1.shape[0]
	cdef int i,j
	for i in range(n_basis):
		for j in range(n_basis):
			Sum += (D2[i][j]-D1[i][j])**2
	delta =np.sqrt(Sum)

	return delta


cpdef double round_up(double V):
	V = np.around(V, decimals=15).tolist()
	return V




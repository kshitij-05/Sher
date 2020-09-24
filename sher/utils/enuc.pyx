# cython: infer_types=True
import numpy as np
cimport cython
import math
DTYPE = np.double
@cython.boundscheck(False)
@cython.wraparound(False)


cdef find_distance(double[::1] a_coordi , double[::1] b_coordi):
	cdef double x_1,x_2,y_1,y_2,z_1,z_2,R_square,R 
	x_1=a_coordi[0]
	x_2=b_coordi[0]
	y_1=a_coordi[1]
	y_2=b_coordi[1]
	z_1=a_coordi[2]
	z_2=b_coordi[2]
	R_square =(x_1-x_2)**2+(y_1-y_2)**2+(z_1-z_2)**2
	R = math.sqrt(R_square)
	return R	


def enuc_calculator(double[::1] atomic_nos, double[:,::1] geom):
	cdef int n = atomic_nos.shape[0]
	cdef double E_nuc=0.0
	cdef int i,j
	cdef double Z_a , Z_b ,R_ab
	for i in range(n):
		for j in range(0,i):
			Z_a=atomic_nos[i]
			Z_b=atomic_nos[j]
			R_ab=find_distance(geom[i],geom[j])
			#print(R_ab)
			E_nuc+=(Z_a*Z_b)/R_ab

	return E_nuc
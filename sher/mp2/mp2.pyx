# cython: infer_types=True
import math
import numpy as np
cimport cython
DTYPE = np.double
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)

def transform_ao2mo(double[:,:,:,::1] two_electron_integrals_4D,double[:,::1] C):
    nbasis = C.shape[0]
    MO2 = np.zeros((nbasis,nbasis,nbasis,nbasis),dtype = DTYPE)
    cdef double[:,:,:, ::1] MO2_view = MO2
    temp = np.zeros((nbasis,nbasis,nbasis,nbasis),dtype = DTYPE)
    cdef double[:,:,:, ::1] temp_view = temp
    temp2 = np.zeros((nbasis,nbasis,nbasis,nbasis),dtype = DTYPE)
    cdef double[:,:,:, ::1] temp2_view = temp2
    temp3= np.zeros((nbasis,nbasis,nbasis,nbasis),dtype = DTYPE)
    cdef double[:,:,:, ::1] temp3_view = temp3

    cdef Py_ssize_t mu,nu,lam,sig ,i,j,k,l
    cdef Py_ssize_t x,y,z,w,v,u
    #Inplace operator not supported: error reminder --> eg [2,4] = [1,2]+[1,2] is not supported#multiply each element
    for mu in range(nbasis):
        for i in range(nbasis):    
            for x in range(0,nbasis):
                for y in range(0,nbasis):
                    for z in range(0,nbasis):
                        temp_view[mu,x,y,z] +=C[i,mu]*two_electron_integrals_4D[i,x,y,z]
        for nu in range(nbasis):
            for j in range(nbasis):
                for w in range(0,nbasis):
                    for v in range(0,nbasis):
                        temp2_view[mu,nu,w,v] += C[j,nu]*temp_view[mu,j,w,v]
            for lam in range(nbasis):
                for k in range(nbasis):
                    for u in range(0,nbasis):
                        temp3_view[mu,nu,lam,u] +=C[k,lam]*temp2_view[mu,nu,k,u]
                for sig in range(nbasis):
                    for l in range(nbasis):
                        MO2_view[mu,nu,lam,sig] +=C[l,sig]*temp3_view[mu,nu,lam,l]
    
    del temp
    del temp2
    del temp3

    return MO2

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)

def space_2_spin_cy(double[:,:,:,::1] MO2 ,int nbasis):
    ints=np.zeros((nbasis*2,nbasis*2,nbasis*2,nbasis*2),dtype = DTYPE)
    cdef double[:,:,:,::1] ints_view = ints
    cdef Py_ssize_t p,q,r,s
    cdef double value1 ,value2

    for p in range(nbasis*2):
        for q in range(nbasis*2):
            for r in range(nbasis*2):
                for s in range(nbasis*2):
                    value1 = MO2[(p)//2][(r)//2][(q)//2][(s)//2] * (p%2 == r%2) * (q%2 == s%2)
                    value2 = MO2[(p)//2][(s)//2][(q)//2][(r)//2] * (p%2 == s%2) * (q%2 == r%2)
                    ints_view[p][q][r][s] = value1 - value2

    #del MO2

    return ints

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)

def mp2_energy(int Nelec,double[:,:,:,::1] ints,double[::1] E):
    cdef double CC = 0.0
    nbasis = E.shape[0]
    cdef Py_ssize_t i,j,a,b
    for i in range(Nelec):
        for j in range(Nelec):
            for a in range(Nelec,nbasis*2):
                for b in range(Nelec,nbasis*2):
                    CC += 0.25*(ints[i][j][a][b]*ints[i][j][a][b])/(E[i//2] + E[j//2] - E[a//2] - E[b//2])

    return CC

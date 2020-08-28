import math
import numpy as np


def transform_ao2mo(nbasis,two_electron_integrals_4D,C,E):
    MO2 = np.zeros((nbasis,nbasis,nbasis,nbasis))
    temp = np.zeros((nbasis,nbasis,nbasis,nbasis))
    temp2 = np.zeros((nbasis,nbasis,nbasis,nbasis))
    temp3= np.zeros((nbasis,nbasis,nbasis,nbasis))
    two_electron_integrals_4D = np.array(two_electron_integrals_4D)
    for mu in range(0,nbasis):
        for i in range(0,nbasis):
            temp[mu,:,:,:] += C[i,mu]*two_electron_integrals_4D[i,:,:,:]
        for nu in range(0,nbasis):
            for j in range(0,nbasis):
                temp2[mu,nu,:,:] += C[j,nu]*temp[mu,j,:,:]
            for lam in range(0,nbasis):
                for k in range(0,nbasis):
                    temp3[mu,nu,lam,:] += C[k,lam]*temp2[mu,nu,k,:]
                #print(temp2)
                for sig in range(0,nbasis):
                    for l in range(0,nbasis):
                        MO2[mu,nu,lam,sig] += C[l,sig]*temp3[mu,nu,lam,l]
    
    del temp
    del temp2
    del temp3
    
    ints=np.zeros((nbasis*2,nbasis*2,nbasis*2,nbasis*2))
    for p in range(0,nbasis*2):
        for q in range(0,nbasis*2):
            for r in range(0,nbasis*2):
                for s in range(0,nbasis*2):
                    value1 = MO2[(p)//2][(r)//2][(q)//2][(s)//2] * (p%2 == r%2) * (q%2 == s%2)
                    value2 = MO2[(p)//2][(s)//2][(q)//2][(r)//2] * (p%2 == s%2) * (q%2 == r%2)
                    ints[p][q][r][s] = value1 - value2

    #del MO2

    fs = np.zeros((nbasis*2))
    for i in range(0,nbasis*2):
        fs[i] = E[i//2]

    fs = np.diag(fs)
    
    return ints, MO2



def mp2_energy(nbasis,Nelec,ints,E):
    CC = 0.0
    for i in range(0,Nelec):
        for j in range(0,Nelec):
            for a in range(Nelec,nbasis*2):
                for b in range(Nelec,nbasis*2):
                    CC += 0.25*(ints[i][j][a][b]*ints[i][j][a][b])/(E[i//2] + E[j//2] - E[a//2] - E[b//2])

    return CC

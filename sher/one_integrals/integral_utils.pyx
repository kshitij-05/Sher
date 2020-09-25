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

cdef double E(int i,int j,int t, double Qx,double a,double b):
	cdef double p ,q
	p = a + b
	q = a*b/p
	cdef double e =2.718281828459045
	if (t < 0) or (t > (i + j)):
	# out of bounds for t
		return 0.0
	elif i == j == t == 0:
	# base case
		return e**(-q*Qx*Qx) # K_AB
	elif j == 0:
	# decrement index i
		return (1.0/(2*p))*E(i-1,j,t-1,Qx,a,b) - \
		(q*Qx/a)*E(i-1,j,t,Qx,a,b) + \
		(t+1)*E(i-1,j,t+1,Qx,a,b)
	else:
	# decrement index j
		return (1.0/(2*p))*E(i,j-1,t-1,Qx,a,b) + \
		(q*Qx/b)*E(i,j-1,t,Qx,a,b) + \
		(t+1)*E(i,j-1,t+1,Qx,a,b)



#----------------------------------------------------
#
#		Overlap Integral functions
#
#---------------------------------------------------

@cython.cdivision(True)

cdef double overlap(double a, int l1,int m1,int n1,double A1,double A2,double A3,\
	double b,int l2,int m2,int n2,double B1,double B2,double B3):
	cdef double S1,S2,S3
	S1 = E(l1,l2,0,A1-B1,a,b) # X
	S2 = E(m1,m2,0,A2-B2,a,b) # Y
	S3 = E(n1,n2,0,A3-B3,a,b) # Z
	return S1*S2*S3*(3.141592653589793/(a+b))**1.5

'''
S1*S2*S3*np.power(np.pi/(a+b),1.5)
'''
@cython.boundscheck(False)
@cython.wraparound(False)

cdef double S(double[::1] aexps, double[::1] acoefs, int[::1] ashell ,double[::1] anorm, double[::1] aorigin,\
	double[::1] bexps, double[::1] bcoefs, int[::1] bshell ,double[::1] bnorm, double[::1] borigin):
	
	cdef int noa_coeffs,nob_coeffs
	noa_coeffs = acoefs.shape[0]
	nob_coeffs = bcoefs.shape[0]
	cdef double s =0.0
	cdef int ia , ib
	cdef double norm1,norm2,coef1,coef2,exp1,exp2,origina1,origina2,origina3,originb1,originb2,originb3
	cdef int shella1,shella2,shella3,shellb1,shellb2,shellb3
	origina1=aorigin[0]
	origina2=aorigin[1]
	origina3=aorigin[2]
	originb1=borigin[0] 
	originb2 =borigin[1]
	originb3 =borigin[2]
	shella1=ashell[0]
	shella2=ashell[1]
	shella3=ashell[2] 
	shellb1=bshell[0] 
	shellb2=bshell[1] 
	shellb3=bshell[2]
	for ia in range(noa_coeffs):
		for ib  in range(nob_coeffs):
			norm1 =anorm[ia]
			norm2 =bnorm[ib]
			coef1 =acoefs[ia]
			coef2 =bcoefs[ib]
			exp1 =aexps[ia]
			exp2 =bexps[ib]

			s += norm1*norm2*coef1*coef2*overlap(exp1,shella1,shella2,shella3,origina1,origina2,origina3,\
				exp2,shellb1,shellb2,shellb3,originb1,originb2,originb3)
	return s

@cython.boundscheck(False)
@cython.wraparound(False)


def S_mat(list exps,list coefs,double[:,::1] origins,int[:,::1] shells,list norms):
	cdef int nbasis = len(exps)
	smat = np.zeros((nbasis,nbasis),dtype = DTYPE)
	cdef double[::1]exp1
	cdef double[::1]exp2
	cdef double[::1]coefs1
	cdef double[::1]coefs2
	cdef double[::1]norm1
	cdef double[::1]norm2
	cdef double[:,::1] smat_view = smat
	cdef double s =0.0
	cdef int i,j
	for i in range(nbasis):
		for j in range(0 , i+1):
			if j==i:
				s = 1.0
			else:
				exp1 = exps[i]
				exp2 = exps[j]
				coefs1 = coefs[i]
				coefs2 = coefs[j]
				norm1= norms[i]
				norm2 = norms[j]
				s = S(exp1,coefs1,shells[i,:],norm1,origins[i,:],exp2,coefs2,shells[j,:],norm2,origins[j,:])
			smat_view[i][j] = smat_view[j][i] = s

	return smat

#-------------------------------------------------
#
#      Kinetic Integrals
#
#-------------------------------------------------

cdef double kinetic(double a,int l1,int m1,int n1,double A1,double A2,double A3,\
	double b,int l2,int m2,int n2,double B1,double B2,double B3):

	cdef double term0 , term1 ,term2
	term0 = b*(2*(l2+m2+n2)+3)*overlap(a,l1,m1,n1,A1,A2,A3,b,l2,m2,n2,B1,B2,B3)
	term1 = -2*(b**2)*(overlap(a,l1,m1,n1,A1,A2,A3,b,l2+2,m2,n2,B1,B2,B3) \
		+ overlap(a,l1,m1,n1,A1,A2,A3,b,l2,m2+2,n2,B1,B2,B3) + overlap(a,l1,m1,n1,A1,A2,A3,b,l2,m2,n2+2,B1,B2,B3))
	term2 = -0.5*(l2*(l2-1)*overlap(a,l1,m1,n1,A1,A2,A3,b,l2-2,m2,n2,B1,B2,B3) + m2*(m2-1)*\
		overlap(a,l1,m1,n1,A1,A2,A3,b,l2,m2-2,n2,B1,B2,B3) \
		+ n2*(n2-1)*overlap(a,l1,m1,n1,A1,A2,A3,b,l2,m2,n2-2,B1,B2,B3))
	return term0+term1+term2

@cython.boundscheck(False)
@cython.wraparound(False)

cdef double T(double[::1] aexps, double[::1] acoefs, int[::1] ashell ,double[::1] anorm, double[::1] aorigin,\
	double[::1] bexps, double[::1] bcoefs, int[::1] bshell ,double[::1] bnorm, double[::1] borigin):
	
	cdef int noa_coeffs,nob_coeffs
	noa_coeffs = acoefs.shape[0]
	nob_coeffs = bcoefs.shape[0]
	cdef double t =0.0
	cdef int ia , ib
	cdef double norm1,norm2,coef1,coef2,exp1,exp2,origina1,origina2,origina3,originb1,originb2,originb3
	cdef int shella1,shella2,shella3,shellb1,shellb2,shellb3
	origina1=aorigin[0]
	origina2=aorigin[1]
	origina3=aorigin[2]
	originb1=borigin[0] 
	originb2 =borigin[1]
	originb3 =borigin[2]
	shella1=ashell[0]
	shella2=ashell[1]
	shella3=ashell[2] 
	shellb1=bshell[0] 
	shellb2=bshell[1] 
	shellb3=bshell[2]
	for ia in range(noa_coeffs):
		for ib  in range(nob_coeffs):
			norm1 =anorm[ia]
			norm2 =bnorm[ib]
			coef1 =acoefs[ia]
			coef2 =bcoefs[ib]
			exp1 =aexps[ia]
			exp2 =bexps[ib]

			t += norm1*norm2*coef1*coef2*kinetic(exp1,shella1,shella2,shella3,origina1,origina2,origina3,\
				exp2,shellb1,shellb2,shellb3,originb1,originb2,originb3)
	return t

@cython.boundscheck(False)
@cython.wraparound(False)

def T_mat(list exps,list coefs,double[:,::1] origins,int[:,::1] shells,list norms):
	cdef int nbasis = len(exps)
	tmat = np.zeros((nbasis,nbasis),dtype = DTYPE)
	cdef double[:,::1] tmat_view = tmat
	cdef double s =0.0
	cdef double[::1]exp1
	cdef double[::1]exp2
	cdef double[::1]coefs1
	cdef double[::1]coefs2
	cdef double[::1]norm1
	cdef double[::1]norm2
	cdef int i,j
	for i in range(nbasis):
		for j in range(0 , i+1):
			exp1 = exps[i]
			exp2 = exps[j]
			coefs1 = coefs[i]
			coefs2 = coefs[j]
			norm1= norms[i]
			norm2 = norms[j]
			s = T(exp1,coefs1,shells[i,:],norm1,origins[i,:],exp2,coefs2,shells[j,:],norm2,origins[j,:])
			tmat_view[i][j] = tmat_view[j][i] = s

	return tmat

#-------------------------------------------------
#
#      Coulomb Integrals
#
#-------------------------------------------------
@cython.cdivision(True)
cdef double gaussian_product_center(double a,double A,double b,double B):
	return (a*A+b*B)/(a+b)


ctypedef unsigned long long Ulong

'''cdef double pochtest(double z,int m1,int m2):
	cdef double s = 1.0
	cdef int i
	for i in range(m1,m2):
		s *=z+i
	return s'''

cdef Ulong fact(int k1 ,int k2):
	cdef Ulong y=1
	cdef int i
	for i in range(k1+1,k2):
		y*=(i+1)
	return y

@cython.cdivision(True)

cdef double smallboys(int n,double T):
	cdef double s=0.0
	cdef double x=-T
	cdef double x10=x**10
	cdef Ulong f10,f20,f30,f40,f50,f60
	f10=3628800									#f10=fact(0,10)
	f20=670442572800							#f20=fact(9,20)
	f30=109027350432000							#f30=fact(19,30)
	f40=3075990524006400						#f40=fact(29,40)
	f50=37276043023296000						#f50=fact(39,50)
	f60=273589847231500800						#f60=fact(49,60)
	cdef int k
	for k in range(53):
		if k<11:
			s+=(x**k)/fact(0,k)/(2*n+2*k+1)
		elif k<21:
			s+=((x**(k-10))/fact(9,k)/(2*n+2*k+1))*x10/f10
		elif k<31:
			s+=((x**(k-20))/fact(19,k)/(2*n+2*k+1))*x10/f10*x10/f20
		elif k<41:
			s+=((x**(k-30))/fact(29,k)/(2*n+2*k+1))*x10/f10*x10/f20*x10/f30
		elif k<51:
			s+=((x**(k-40))/fact(39,k)/(2*n+2*k+1))*x10/f10*x10/f20*x10/f30*x10/f40
		else:
			s+=((x**(k-50))/fact(49,k)/(2*n+2*k+1))*x10/f10*x10/f20*x10/f30*x10/f40*x10/f50

	return s

@cython.cdivision(True)

cdef double test1f1(double a,double b, double x):
	cdef double s=0.0
	cdef double p
	cdef int k,i
	for k in range(60):
		p=1.0
		for i in range(k):
			p*=(a+i)*(x)/(b+i)/(i+1)

		s+=p

	return s

@cython.cdivision(True)

cdef double largeboys(int n,double T):
	return fact(0,fact(0,2*n-1))*(3.141592653589793**0.5)*((T**(2*n+1))**(-0.5))/(2**(n+1))


@cython.cdivision(True)

cdef double boys(int n,double T):
	

	return hyp1f1(n+0.5,n+1.5,-T)/(2.0*n+1.0)
	'''if T<24.000:
	else:
		return largeboys(n,T)'''



cdef double R(int t,int u,int v,int n,double p,double PCx,double PCy,double PCz,double RPC):
	cdef double T , val
	T = p*RPC*RPC
	val = 0.0
	if t == u == v == 0:
		val += ((-2*p)**n)*boys(n,T)
	elif t == u == 0:
		if v > 1:
			val += (v-1)*R(t,u,v-2,n+1,p,PCx,PCy,PCz,RPC)
		val += PCz*R(t,u,v-1,n+1,p,PCx,PCy,PCz,RPC)
	elif t == 0:
		if u > 1:
			val += (u-1)*R(t,u-2,v,n+1,p,PCx,PCy,PCz,RPC)
		val += PCy*R(t,u-1,v,n+1,p,PCx,PCy,PCz,RPC)
	else:
		if t > 1:
			val += (t-1)*R(t-2,u,v,n+1,p,PCx,PCy,PCz,RPC)
		val += PCx*R(t-1,u,v,n+1,p,PCx,PCy,PCz,RPC)
	return val

cdef double normfrobenuis(double a,double b,double c):
	cdef double s=0.0
	s = a**2 + b**2 + c**2
	return s**(0.5)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)

cdef double nuclear_attraction(double a,int l1,int m1,int n1,double A1,double A2,double A3,double b,int l2,int m2,\
	int n2,double B1,double B2,double B3,double C1,double C2,double C3):

	cdef double p ,RPC ,val
	p = a + b
	cdef double RPCdash1,RPCdash2,RPCdash3 	
	RPCdash1 = C1-gaussian_product_center(a,A1,b,B1)
	RPCdash2 = C2-gaussian_product_center(a,A2,b,B2)
	RPCdash3 = C3-gaussian_product_center(a,A3,b,B3)
	RPC = normfrobenuis(RPCdash1,RPCdash2,RPCdash3)
	val = 0.0
	cdef int t,u,v
	cdef double AB1 ,AB2 ,AB3 
	AB1=A1-B1
	AB2=A2-B2
	AB3=A3-B3
	cdef int len1,len2,len3
	len1 =l1+l2+1 
	len2 =m1+m2+1
	len3 =n1+n2+1
	for t in range(len1):
		for u in range(len2):
			for v in range(len3):
				val += E(l1,l2,t,AB1,a,b)*E(m1,m2,u,AB2,a,b)*E(n1,n2,v,AB3,a,b)*R(t,u,v,0,p,-RPCdash1,-RPCdash2,-RPCdash3,RPC)
	val *= (2*3.141592653589793)/p
	return val

@cython.boundscheck(False)
@cython.wraparound(False)

cdef double V(double[::1] aexps, double[::1] acoefs, int[::1] ashell ,double[::1] anorm, double[::1] aorigin,\
	double[::1] bexps, double[::1] bcoefs, int[::1] bshell ,double[::1] bnorm, double[::1] borigin,double[::1] C):
	cdef int noa_coeffs,nob_coeffs
	noa_coeffs = acoefs.shape[0]
	nob_coeffs = bcoefs.shape[0]
	cdef double v =0.0
	cdef int ia , ib
	cdef double norm1,norm2,coef1,coef2,exp1,exp2,origina1,origina2,origina3,originb1,originb2,originb3,C1,C2,C3
	cdef int shella1,shella2,shella3,shellb1,shellb2,shellb3
	origina1=aorigin[0]
	origina2=aorigin[1]
	origina3=aorigin[2]
	originb1=borigin[0] 
	originb2 =borigin[1]
	originb3 =borigin[2]
	shella1=ashell[0]
	shella2=ashell[1]
	shella3=ashell[2] 
	shellb1=bshell[0] 
	shellb2=bshell[1] 
	shellb3=bshell[2]
	C1= C[0]
	C2= C[1]
	C3= C[2]
	for ia in range(noa_coeffs):
		for ib  in range(nob_coeffs):
			norm1 =anorm[ia]
			norm2 =bnorm[ib]
			coef1 =acoefs[ia]
			coef2 =bcoefs[ib]
			exp1 =aexps[ia]
			exp2 =bexps[ib]

			v += norm1*norm2*coef1*coef2*nuclear_attraction(exp1,shella1,shella2,shella3,origina1,origina2,origina3,\
				exp2,shellb1,shellb2,shellb3,originb1,originb2,originb3,C1,C2,C3)
	return v

@cython.boundscheck(False)
@cython.wraparound(False)

def V_mat(list exps,list coefs,double[:,::1] origins,int[:,::1] shells,list norms,double[::1] atomic_nos,double[:,::1] geom):
	cdef int nbasis = len(exps)
	no_of_atoms = atomic_nos.shape[0]
	cdef double[::1]exp1
	cdef double[::1]exp2
	cdef double[::1]coefs1
	cdef double[::1]coefs2
	cdef double[::1]norm1
	cdef double[::1]norm2

	cdef int i,j,k
	P1 = np.zeros((nbasis,nbasis),dtype=DTYPE)
	cdef double[:,::1] P1_view = P1
	cdef double v =0.0
	for i in range(0,nbasis):
		for j in range(0,nbasis):
			for k in range(0,no_of_atoms):
				exp1 = exps[i]
				exp2 = exps[j]
				coefs1 = coefs[i]
				coefs2 = coefs[j]
				norm1= norms[i]
				norm2 = norms[j]
				v += V(exp1,coefs1,shells[i,:],norm1,origins[i,:],exp2,coefs2,shells[j,:],norm2,origins[j,:],geom[k])*atomic_nos[k]
			P1_view[i,j] = -v

	P2 = np.zeros((nbasis, nbasis),dtype=DTYPE)
	cdef double[:,::1] P2_view = P2 
	cdef int m,n
	for m in range(0,nbasis):
		for n in range(0,nbasis):
			if m ==0 and n==0 :
				P2_view[m,n] = P1_view[0,0]
	
			else :
				P2_view[m,n] = -(P1_view[m,n-1] -P1_view[m,n])

	for i in range(0,nbasis):
		P2_view[i,0] = P2_view[0,i]

	return P2
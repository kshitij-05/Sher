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

#----------------------------------------------------
#
#     Two electron integrals
#
#----------------------------------------------------

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
'''
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

	return s'''


@cython.cdivision(True)

cdef double largeboys(int n,double T):
	return fact(0,fact(0,2*n-1))*(3.141592653589793**0.5)*((T**(2*n+1))**(-0.5))/(2**(n+1))


@cython.cdivision(True)

cdef double boys(int n,double T):
	return hyp1f1(n+0.5,n+1.5,-T)/(2.0*n+1.0)
	'''if T<14.60853:
		return smallboys(n,T)
	elif T>=14.60853 and n==0:
		return (3.141592653589793**0.5)*(T**(-0.5))*0.5	
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

cdef double electron_repulsion(double a,int l1,int m1,int n1,double A0,double A1,double A2,\
	double b,int l2,int m2,int n2,double B0,double B1,double B2,\
	double c,int l3,int m3,int n3,double C0,double C1,double C2,\
	double d,int l4,int m4,int n4,double D0,double D1,double D2):
		cdef double p , q , alpha , val ,RPQ
		p = a+b # composite exponent for P (from Gaussians 'a' and 'b')
		q = c+d # composite exponent for Q (from Gaussians 'c' and 'd')
		alpha = p*q/(p+q)
		cdef double P0 = gaussian_product_center(a,A0,b,B0)
		cdef double P1 = gaussian_product_center(a,A1,b,B1)
		cdef double P2 = gaussian_product_center(a,A2,b,B2) 
		cdef double Q0 = gaussian_product_center(c,C0,d,D0)
		cdef double Q1 = gaussian_product_center(c,C1,d,D1)
		cdef double Q2 = gaussian_product_center(c,C2,d,D2)
		cdef double PQ0,PQ1,PQ2
		PQ0=P0-Q0
		PQ1=P1-Q1
		PQ2=P2-Q2
		RPQ = normfrobenuis(PQ0,PQ1,PQ2)
		val = 0.0
		cdef double AB0,AB1,AB2,CD0,CD1,CD2
		AB0=A0-B0
		AB1=A1-B1
		AB2=A2-B2
		CD0=C0-D0
		CD1=C1-D1
		CD2=C2-D2
		cdef int t,u,v,tau,nu,phi
		cdef int len1,len2,len3,len4,len5,len6
		len1 = l1+l2+1
		len2 = m1+m2+1
		len3 = n1+n2+1
		len4 = l3+l4+1
		len5 = m3+m4+1
		len6 = n3+n4+1
		for t in range(len1):
			for u in range(len2):
				for v in range(len3):
					for tau in range(len4):
						for nu in range(len5):
							for phi in range(len6):
								val += E(l1,l2,t,AB0,a,b) * \
										E(m1,m2,u,AB1,a,b) * \
										E(n1,n2,v,AB2,a,b) * \
										E(l3,l4,tau,CD0,c,d) * \
										E(m3,m4,nu ,CD1,c,d) * \
										E(n3,n4,phi,CD2,c,d) * \
										((-1)**(tau+nu+phi)) * \
										R(t+tau,u+nu,v+phi,0,\
											alpha,PQ0,PQ1,PQ2,RPQ)
		val *= 2*(3.141592653589793**2.5)/(p*q*((p+q)**0.5))
		return val



'''
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)

def Eri_1d(double[:,::1] exps,double[:,::1] coefs,double[:,::1] origins,int[:,::1] shells,double[:,::1] norms):
	cdef list unique
	cdef int nbasis
	nbasis = exps.shape[0]
	unique = unique_indices(nbasis)
	length = len(unique)
	Temp_mat = np.zeros((length),dtype = DTYPE)
	cdef double[::1] Temp_mat_view = Temp_mat
	cdef int a,b,c,d,i


	for i in range(length):
		a = unique[i][0]
		b = unique[i][1]
		c = unique[i][2]
		d = unique[i][3]
		Temp_mat_view[i]=ERI(exps[a,:],coefs[a,:],shells[a,:],norms[a,:],origins[a,:],\
			exps[b,:],coefs[b,:],shells[b,:],norms[b,:],origins[b,:],\
			exps[c,:],coefs[c,:],shells[c,:],norms[c,:],origins[c,:],\
			exps[d,:],coefs[d,:],shells[d,:],norms[d,:],origins[d,:])

	
	return Temp_mat'''

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)

cdef int compound_index( int i, int j, int k, int l):
	cdef float ij , kl , ijkl
	if i>j:
		ij=i*(i+1)/2+j
	else:
		ij=j*(j+1)/2+i

	if k>l:
		kl=k*(k+1)/2+l
	else:
		kl=l*(l+1)/2+k

	if ij>kl:
		ijkl=ij*(ij+1)/2+kl
	else:
		ijkl=kl*(kl+1)/2+ij

	return int(ijkl)

@cython.boundscheck(False)
@cython.wraparound(False)

cdef list uniqueindex(int n):
	cdef float maxx = compound_index(n,n,n,n)
	cdef float minn = 0.0
	cdef list ll = []
	cdef int i,j,k,l
	for i in range(n):
		for j in range(n):
			for k in range(n):
				for l in range(n):
					if compound_index(i,j,k,l)==minn:
						ll.append((i,j,k,l))
						minn+=1
					if minn>= maxx:
						break

	return ll


'''
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)

cdef list unlister(list a):
		cdef set b
		cdef list c
		b = set(a)
		c = list(b)
		return c

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)

cdef int compound_index( int i, int j, int k, int l):
	cdef float ij , kl , ijkl
	if i>j:
		ij=i*(i+1)/2+j
	else:
		ij=j*(j+1)/2+i

	if k>l:
		kl=k*(k+1)/2+l
	else:
		kl=l*(l+1)/2+k

	if ij>kl:
		ijkl=ij*(ij+1)/2+kl
	else:
		ijkl=kl*(kl+1)/2+ij

	return int(ijkl)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)

cdef list unique_indices(int nbasis):
		cdef list unlist=[]
		cdef list list1=[]
		cdef list list2
		cdef int i,j,k,l
		for  i in range(nbasis):
			for j in range(nbasis):
				for k in range(nbasis):
					for l in range(nbasis):
						list1.append((i,j,k,l))		
						unlist.append(compound_index(i,j,k,l))

		list2 = unlister(unlist)
		cdef list indices= []
		for i in range(len(list2)):
			indices.append(unlist.index(list2[i]))
		cdef list tei_unique = []
		for i in range(len(indices)):
			tei_unique.append(list1[indices[i]])
		return tei_unique



@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)


cdef double electron_repulsion(double a,int l1,int m1,int n1,double A0,double A1,double A2,\
	double b,int l2,int m2,int n2,double B0,double B1,double B2,\
	double c,int l3,int m3,int n3,double C0,double C1,double C2,\
	double d,int l4,int m4,int n4,double D0,double D1,double D2):
		cdef double p , q , alpha , val 
		p = a+b # composite exponent for P (from Gaussians 'a' and 'b')
		q = c+d # composite exponent for Q (from Gaussians 'c' and 'd')
		alpha = p*q/(p+q)
		cdef double P0 = gaussian_product_center(a,A0,b,B0)
		cdef double P1 = gaussian_product_center(a,A1,b,B1)
		cdef double P2 = gaussian_product_center(a,A2,b,B2) # A and B composite center
		cdef double Q0 = gaussian_product_center(c,C0,d,D0)
		cdef double Q1 = gaussian_product_center(c,C1,d,D1)
		cdef double Q2 = gaussian_product_center(c,C2,d,D2)# C and D composite center
		RPQ = normfrobenuis(P0-Q0,P1-Q1,P2-Q2)
		val = 0.0
		PQ1=P0-Q0
		PQ2=P1-Q1
		PQ3=P2-Q2
		cdef int t,u,v,tau,nu,phi
		for t in range(l1+l2+1):
			for u in range(m1+m2+1):
				for v in range(n1+n2+1):
					for tau in range(l3+l4+1):
						for nu in range(m3+m4+1):
							for phi in range(n3+n4+1):
								val += E(l1,l2,t,A0-B0,a,b) * \
										E(m1,m2,u,A1-B1,a,b) * \
										E(n1,n2,v,A2-B2,a,b) * \
										E(l3,l4,tau,C0-D0,c,d) * \
										E(m3,m4,nu ,C1-D1,c,d) * \
										E(n3,n4,phi,C2-D2,c,d) * \
										((-1)**(tau+nu+phi)) * \
										R(t+tau,u+nu,v+phi,0,\
											alpha,PQ1,PQ2,PQ3,RPQ)
		val *= 2*(3.141592653589793**2.5)/(p*q*((p+q)**0.5))
		return val'''

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)

cdef double ERI(double[::1] aexps, double[::1] acoefs, int[::1] ashell ,double[::1] anorm, double[::1] aorigin,\
	double[::1] bexps, double[::1] bcoefs, int[::1] bshell ,double[::1] bnorm, double[::1] borigin,\
	double[::1] cexps, double[::1] ccoefs, int[::1] cshell ,double[::1] cnorm, double[::1] corigin,\
	double[::1] dexps, double[::1] dcoefs, int[::1] dshell ,double[::1] dnorm, double[::1] dorigin):
		cdef int no_coeffs = acoefs.shape[0]
		cdef double norm1,norm2,norm3,norm4
		cdef double coef1,coef2,coef3,coef4
		cdef double exp1,exp2,exp3,exp4
		cdef int shella1,shella2,shella3,shellb1,shellb2,shellb3,shellc1,shellc2,shellc3,shelld1,shelld2,shelld3
		cdef double origina1,origina2,origina3,originb1,originb2,originb3,originc1,originc2,originc3,origind1,origind2,origind3
		cdef double eri =0.0
		cdef int ja,jb,jc,jd
		shella1,shella2,shella3 = ashell[0],ashell[1],ashell[2]
		shellb1,shellb2,shellb3 = bshell[0],bshell[1],bshell[2]
		shellc1,shellc2,shellc3 = cshell[0],cshell[1],cshell[2]
		shelld1,shelld2,shelld3 = dshell[0],dshell[1],dshell[2]
		origina1,origina2,origina3 = aorigin[0],aorigin[1],aorigin[2]
		originb1,originb2,originb3 = borigin[0],borigin[1],borigin[2]
		originc1,originc2,originc3 = corigin[0],corigin[1],corigin[2]
		origind1,origind2,origind3 = dorigin[0],dorigin[1],dorigin[2]

		for ja in range(no_coeffs):
			norm1,coef1,exp1 = anorm[ja],acoefs[ja],aexps[ja]
			for jb in range(no_coeffs):
				norm2,coef2,exp2 = bnorm[jb],bcoefs[jb],bexps[jb]
				for jc in range(no_coeffs):
					norm3,coef3,exp3 = cnorm[jc],ccoefs[jc],cexps[jc]
					for jd in range(no_coeffs):
						norm4,coef4,exp4 = dnorm[jd],dcoefs[jd],dexps[jd]

						eri += norm1*norm2*norm3*norm4*coef1*coef2*coef3*coef4*\
						electron_repulsion(exp1,shella1,shella2,shella3,origina1,origina2,origina3,\
							exp2,shellb1,shellb2,shellb3,originb1,originb2,originb3,\
							exp3,shellc1,shellc2,shellc3,originc1,originc2,originc3,\
							exp4,shelld1,shelld2,shelld3,origind1,origind2,origind3)

		return eri

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)

def Eri_mat(double[:,::1] exps,double[:,::1] coefs,double[:,::1] origins,int[:,::1] shells,double[:,::1] norms):
	cdef list unique
	cdef int nbasis
	nbasis = exps.shape[0]
	unique = uniqueindex(nbasis)
	length = len(unique)
	Temp_mat = np.zeros((length),dtype = DTYPE)
	cdef double[::1] Temp_mat_view = Temp_mat
	cdef int a,b,c,d,i


	for i in range(length):
		a = unique[i][0]
		b = unique[i][1]
		c = unique[i][2]
		d = unique[i][3]
		Temp_mat_view[i]=ERI(exps[a,:],coefs[a,:],shells[a,:],norms[a,:],origins[a,:],\
			exps[b,:],coefs[b,:],shells[b,:],norms[b,:],origins[b,:],\
			exps[c,:],coefs[c,:],shells[c,:],norms[c,:],origins[c,:],\
			exps[d,:],coefs[d,:],shells[d,:],norms[d,:],origins[d,:])
	
	Twoe_mat = np.zeros((nbasis,nbasis,nbasis,nbasis),dtype = DTYPE)
	cdef double[:,:,:,::1] Twoe_mat_view = Twoe_mat
	
	cdef int m=0
	cdef int j,k,l
	cdef float ij,kl
	for i in range(0,nbasis):
		for j in range(0,i+1):
			for k in range(0,nbasis):
				for l in range(0,k+1):
					ij = i * (i + 1) / 2 + j
					kl = k * (k + 1) / 2 + l
					if ij>=kl:
						Twoe_mat_view[i,j,k,l] = Temp_mat_view[m]
						Twoe_mat_view[i,j,l,k] = Temp_mat_view[m]
						Twoe_mat_view[j,i,k,l] = Temp_mat_view[m]
						Twoe_mat_view[j,i,l,k] = Temp_mat_view[m]
						Twoe_mat_view[k,l,i,j] = Temp_mat_view[m]
						Twoe_mat_view[l,k,i,j] = Temp_mat_view[m]
						Twoe_mat_view[k,l,j,i] = Temp_mat_view[m]
						Twoe_mat_view[l,k,j,i] = Temp_mat_view[m]

						m=m+1

	return Twoe_mat,Temp_mat
import numpy as np  
from scipy.special import hyp1f1
cimport numpy as np
cimport cython
from cpython cimport array
import array
from help_functions import no_of_e



cdef list Basis_attributes_finder(atom):
	cdef dict basis_set_STO3G = {'H': [[[0.3425250914E+01, 0.6239137298E+00, 0.1688554040E+00], [0.1543289673E+00, 0.5353281423E+00, 0.4446345422E+00], (0, 0, 0)]], 
	'He': [[[6.362421394, 1.158922999, 0.3136497915], [0.1543289673, 0.5353281423, 0.4446345422], (0, 0, 0)]],
	'Li': [[[16.11957475, 2.936200663, 0.794650487], [0.1543289673, 0.5353281423, 0.4446345422], (0, 0, 0)],
	[[0.6362897469, 0.1478600533, 0.0480886784], [-0.09996722919, 0.3995128261, 0.7001154689], (0, 0, 0)], 
	[[0.6362897469, 0.1478600533, 0.0480886784], [0.155916275, 0.6076837186, 0.3919573931], (1, 0, 0)], 
	[[0.6362897469, 0.1478600533, 0.0480886784], [0.155916275, 0.6076837186, 0.3919573931], (0, 1, 0)], 
	[[0.6362897469, 0.1478600533, 0.0480886784], [0.155916275, 0.6076837186, 0.3919573931], (0, 0, 1)]],
	'Be': [[[30.16787069, 5.495115306, 1.487192653], [0.1543289673, 0.5353281423, 0.4446345422], (0, 0, 0)],
	[[1.31483311, 0.3055389383, 0.0993707456], [-0.09996722919, 0.3995128261, 0.7001154689], (0, 0, 0)],
	[[1.31483311, 0.3055389383, 0.0993707456], [0.155916275, 0.6076837186, 0.3919573931], (1, 0, 0)],
	[[1.31483311, 0.3055389383, 0.0993707456], [0.155916275, 0.6076837186, 0.3919573931], (0, 1, 0)],
	[[1.31483311, 0.3055389383, 0.0993707456], [0.155916275, 0.6076837186, 0.3919573931], (0, 0, 1)]],
	'B': [[[48.79111318, 8.887362172, 2.40526704], [0.1543289673, 0.5353281423, 0.4446345422], (0, 0, 0)],
	[[2.236956142, 0.5198204999, 0.16906176], [-0.09996722919, 0.3995128261, 0.7001154689], (0, 0, 0)],
	[[2.236956142, 0.5198204999, 0.16906176], [0.155916275, 0.6076837186, 0.3919573931], (1, 0, 0)],
	[[2.236956142, 0.5198204999, 0.16906176], [0.155916275, 0.6076837186, 0.3919573931], (0, 1, 0)],
	[[2.236956142, 0.5198204999, 0.16906176], [0.155916275, 0.6076837186, 0.3919573931], (0, 0, 1)]], 
	'C': [[[71.61683735, 13.04509632, 3.53051216], [0.1543289673, 0.5353281423, 0.4446345422], (0, 0, 0)],
	[[2.941249355, 0.6834830964, 0.2222899159], [-0.09996722919, 0.3995128261, 0.7001154689], (0, 0, 0)],
	[[2.941249355, 0.6834830964, 0.2222899159], [0.155916275, 0.6076837186, 0.3919573931], (1, 0, 0)], 
	[[2.941249355, 0.6834830964, 0.2222899159], [0.155916275, 0.6076837186, 0.3919573931], (0, 1, 0)], 
	[[2.941249355, 0.6834830964, 0.2222899159], [0.155916275, 0.6076837186, 0.3919573931], (0, 0, 1)]],
	'N': [[[99.10616896, 18.05231239, 4.885660238], [0.1543289673, 0.5353281423, 0.4446345422], (0, 0, 0)],
	[[3.780455879, 0.8784966449, 0.2857143744], [-0.09996722919, 0.3995128261, 0.7001154689], (0, 0, 0)], 
	[[3.780455879, 0.8784966449, 0.2857143744], [0.155916275, 0.6076837186, 0.3919573931], (1, 0, 0)],
	[[3.780455879, 0.8784966449, 0.2857143744], [0.155916275, 0.6076837186, 0.3919573931], (0, 1, 0)],
	[[3.780455879, 0.8784966449, 0.2857143744], [0.155916275, 0.6076837186, 0.3919573931], (0, 0, 1)]], 
	'O': [[[130.7093214, 23.80886605, 6.443608313], [0.1543289673, 0.5353281423, 0.4446345422], (0, 0, 0)], 
	[[5.033151319, 1.169596125, 0.38038896], [-0.09996722919, 0.3995128261, 0.7001154689], (0, 0, 0)],
	[[5.033151319, 1.169596125, 0.38038896], [0.155916275, 0.6076837186, 0.3919573931], (1, 0, 0)],
	[[5.033151319, 1.169596125, 0.38038896], [0.155916275, 0.6076837186, 0.3919573931], (0, 1, 0)], 
	[[5.033151319, 1.169596125, 0.38038896], [0.155916275, 0.6076837186, 0.3919573931], (0, 0, 1)]],
	'F': [[[166.679134, 30.36081233, 8.216820672], [0.1543289673, 0.5353281423, 0.4446345422], (0, 0, 0)],
	[[6.464803249, 1.502281245, 0.4885884864], [-0.09996722919, 0.3995128261, 0.7001154689], (0, 0, 0)], 
	[[6.464803249, 1.502281245, 0.4885884864], [0.155916275, 0.6076837186, 0.3919573931], (1, 0, 0)], 
	[[6.464803249, 1.502281245, 0.4885884864], [0.155916275, 0.6076837186, 0.3919573931], (0, 1, 0)], 
	[[6.464803249, 1.502281245, 0.4885884864], [0.155916275, 0.6076837186, 0.3919573931], (0, 0, 1)]], 
	'Ne': [[[207.015607, 37.70815124, 10.20529731], [0.1543289673, 0.5353281423, 0.4446345422], (0, 0, 0)], 
	[[8.24631512, 1.916266291, 0.6232292721], [-0.09996722919, 0.3995128261, 0.7001154689], (0, 0, 0)], 
	[[8.24631512, 1.916266291, 0.6232292721], [0.155916275, 0.6076837186, 0.3919573931], (1, 0, 0)], 
	[[8.24631512, 1.916266291, 0.6232292721], [0.155916275, 0.6076837186, 0.3919573931], (0, 1, 0)], 
	[[8.24631512, 1.916266291, 0.6232292721], [0.155916275, 0.6076837186, 0.3919573931], (0, 0, 1)]], 
	'Na': [[[250.77243, 45.67851117, 12.36238776], [0.1543289673, 0.5353281423, 0.4446345422], (0, 0, 0)], 
	[[12.04019274, 2.797881859, 0.909958017], [-0.09996722919, 0.3995128261, 0.7001154689], (0, 0, 0)], 
	[[12.04019274, 2.797881859, 0.909958017], [0.155916275, 0.6076837186, 0.3919573931], (1, 0, 0)], 
	[[12.04019274, 2.797881859, 0.909958017], [0.155916275, 0.6076837186, 0.3919573931], (0, 1, 0)], 
	[[12.04019274, 2.797881859, 0.909958017], [0.155916275, 0.6076837186, 0.3919573931], (0, 0, 1)], 
	[[1.478740622, 0.4125648801, 0.1614750979], [-0.219620369, 0.2255954336, 0.900398426], (0, 0, 0)], 
	[[1.478740622, 0.4125648801, 0.1614750979], [0.01058760429, 0.5951670053, 0.462001012], (1, 0, 0)], 
	[[1.478740622, 0.4125648801, 0.1614750979], [0.01058760429, 0.5951670053, 0.462001012], (0, 1, 0)], 
	[[1.478740622, 0.4125648801, 0.1614750979], [0.01058760429, 0.5951670053, 0.462001012], (0, 0, 1)]], 
	'Mg': [[[299.2374137, 54.50646845, 14.75157752], [0.1543289673, 0.5353281423, 0.4446345422], (0, 0, 0)], 
	[[15.12182352, 3.513986579, 1.142857498], [-0.09996722919, 0.3995128261, 0.7001154689], (0, 0, 0)], 
	[[15.12182352, 3.513986579, 1.142857498], [0.155916275, 0.6076837186, 0.3919573931], (1, 0, 0)], 
	[[15.12182352, 3.513986579, 1.142857498], [0.155916275, 0.6076837186, 0.3919573931], (0, 1, 0)], 
	[[15.12182352, 3.513986579, 1.142857498], [0.155916275, 0.6076837186, 0.3919573931], (0, 0, 1)], 
	[[1.395448293, 0.3893265318, 0.1523797659], [-0.219620369, 0.2255954336, 0.900398426], (0, 0, 0)], 
	[[1.395448293, 0.3893265318, 0.1523797659], [0.01058760429, 0.5951670053, 0.462001012], (1, 0, 0)], 
	[[1.395448293, 0.3893265318, 0.1523797659], [0.01058760429, 0.5951670053, 0.462001012], (0, 1, 0)], 
	[[1.395448293, 0.3893265318, 0.1523797659], [0.01058760429, 0.5951670053, 0.462001012], (0, 0, 1)]], 
	'Al': [[[351.4214767, 64.01186067, 17.32410761], [0.1543289673, 0.5353281423, 0.4446345422], (0, 0, 0)], 
	[[18.89939621, 4.391813233, 1.42835397], [-0.09996722919, 0.3995128261, 0.7001154689], (0, 0, 0)], 
	[[18.89939621, 4.391813233, 1.42835397], [0.155916275, 0.6076837186, 0.3919573931], (1, 0, 0)], 
	[[18.89939621, 4.391813233, 1.42835397], [0.155916275, 0.6076837186, 0.3919573931], (0, 1, 0)], 
	[[18.89939621, 4.391813233, 1.42835397], [0.155916275, 0.6076837186, 0.3919573931], (0, 0, 1)], 
	[[1.395448293, 0.3893265318, 0.1523797659], [-0.219620369, 0.2255954336, 0.900398426], (0, 0, 0)], 
	[[1.395448293, 0.3893265318, 0.1523797659], [0.01058760429, 0.5951670053, 0.462001012], (1, 0, 0)], 
	[[1.395448293, 0.3893265318, 0.1523797659], [0.01058760429, 0.5951670053, 0.462001012], (0, 1, 0)], 
	[[1.395448293, 0.3893265318, 0.1523797659], [0.01058760429, 0.5951670053, 0.462001012], (0, 0, 1)]], 
	'Si': [[[407.7975514, 74.28083305, 20.10329229], [0.1543289673, 0.5353281423, 0.4446345422], (0, 0, 0)],
	[[23.19365606, 5.389706871, 1.752899952], [-0.09996722919, 0.3995128261, 0.7001154689], (0, 0, 0)], 
	[[23.19365606, 5.389706871, 1.752899952], [0.155916275, 0.6076837186, 0.3919573931], (1, 0, 0)], 
	[[23.19365606, 5.389706871, 1.752899952], [0.155916275, 0.6076837186, 0.3919573931], (0, 1, 0)], 
	[[23.19365606, 5.389706871, 1.752899952], [0.155916275, 0.6076837186, 0.3919573931], (0, 0, 1)], 
	[[1.478740622, 0.4125648801, 0.1614750979], [-0.219620369, 0.2255954336, 0.900398426], (0, 0, 0)], 
	[[1.478740622, 0.4125648801, 0.1614750979], [0.01058760429, 0.5951670053, 0.462001012], (1, 0, 0)], 
	[[1.478740622, 0.4125648801, 0.1614750979], [0.01058760429, 0.5951670053, 0.462001012], (0, 1, 0)], 
	[[1.478740622, 0.4125648801, 0.1614750979], [0.01058760429, 0.5951670053, 0.462001012], (0, 0, 1)]],
	'P': [[[468.3656378, 85.31338559, 23.08913156], [0.1543289673, 0.5353281423, 0.4446345422], (0, 0, 0)], 
	[[28.03263958, 6.514182577, 2.118614352], [-0.09996722919, 0.3995128261, 0.7001154689], (0, 0, 0)], 
	[[28.03263958, 6.514182577, 2.118614352], [0.155916275, 0.6076837186, 0.3919573931], (1, 0, 0)], 
	[[28.03263958, 6.514182577, 2.118614352], [0.155916275, 0.6076837186, 0.3919573931], (0, 1, 0)], 
	[[28.03263958, 6.514182577, 2.118614352], [0.155916275, 0.6076837186, 0.3919573931], (0, 0, 1)], 
	[[1.743103231, 0.4863213771, 0.1903428909], [-0.219620369, 0.2255954336, 0.900398426], (0, 0, 0)], 
	[[1.743103231, 0.4863213771, 0.1903428909], [0.01058760429, 0.5951670053, 0.462001012], (1, 0, 0)], 
	[[1.743103231, 0.4863213771, 0.1903428909], [0.01058760429, 0.5951670053, 0.462001012], (0, 1, 0)], 
	[[1.743103231, 0.4863213771, 0.1903428909], [0.01058760429, 0.5951670053, 0.462001012], (0, 0, 1)]], 
	'S': [[[533.1257359, 97.1095183, 26.28162542], [0.1543289673, 0.5353281423, 0.4446345422], (0, 0, 0)], 
	[[33.32975173, 7.745117521, 2.518952599], [-0.09996722919, 0.3995128261, 0.7001154689], (0, 0, 0)], 
	[[33.32975173, 7.745117521, 2.518952599], [0.155916275, 0.6076837186, 0.3919573931], (1, 0, 0)], 
	[[33.32975173, 7.745117521, 2.518952599], [0.155916275, 0.6076837186, 0.3919573931], (0, 1, 0)], 
	[[33.32975173, 7.745117521, 2.518952599], [0.155916275, 0.6076837186, 0.3919573931], (0, 0, 1)], 
	[[2.029194274, 0.5661400518, 0.2215833792], [-0.219620369, 0.2255954336, 0.900398426], (0, 0, 0)], 
	[[2.029194274, 0.5661400518, 0.2215833792], [0.01058760429, 0.5951670053, 0.462001012], (1, 0, 0)], 
	[[2.029194274, 0.5661400518, 0.2215833792], [0.01058760429, 0.5951670053, 0.462001012], (0, 1, 0)], 
	[[2.029194274, 0.5661400518, 0.2215833792], [0.01058760429, 0.5951670053, 0.462001012], (0, 0, 1)]], 
	'Cl': [[[601.3456136, 109.5358542, 29.64467686], [0.1543289673, 0.5353281423, 0.4446345422], (0, 0, 0)], 
	[[38.96041889, 9.053563477, 2.944499834], [-0.09996722919, 0.3995128261, 0.7001154689], (0, 0, 0)], 
	[[38.96041889, 9.053563477, 2.944499834], [0.155916275, 0.6076837186, 0.3919573931], (1, 0, 0)], 
	[[38.96041889, 9.053563477, 2.944499834], [0.155916275, 0.6076837186, 0.3919573931], (0, 1, 0)], 
	[[38.96041889, 9.053563477, 2.944499834], [0.155916275, 0.6076837186, 0.3919573931], (0, 0, 1)], 
	[[2.129386495, 0.5940934274, 0.232524141], [-0.219620369, 0.2255954336, 0.900398426], (0, 0, 0)], 
	[[2.129386495, 0.5940934274, 0.232524141], [0.01058760429, 0.5951670053, 0.462001012], (1, 0, 0)], 
	[[2.129386495, 0.5940934274, 0.232524141], [0.01058760429, 0.5951670053, 0.462001012], (0, 1, 0)], 
	[[2.129386495, 0.5940934274, 0.232524141], [0.01058760429, 0.5951670053, 0.462001012], (0, 0, 1)]], 
	'Ar': [[[674.4465184, 122.8512753, 33.24834945], [0.1543289673, 0.5353281423, 0.4446345422], (0, 0, 0)], 
	[[45.16424392, 10.495199, 3.413364448], [-0.09996722919, 0.3995128261, 0.7001154689], (0, 0, 0)], 
	[[45.16424392, 10.495199, 3.413364448], [0.155916275, 0.6076837186, 0.3919573931], (1, 0, 0)], 
	[[45.16424392, 10.495199, 3.413364448], [0.155916275, 0.6076837186, 0.3919573931], (0, 1, 0)], 
	[[45.16424392, 10.495199, 3.413364448], [0.155916275, 0.6076837186, 0.3919573931], (0, 0, 1)], 
	[[2.621366518, 0.731354605, 0.2862472356], [-0.219620369, 0.2255954336, 0.900398426], (0, 0, 0)], 
	[[2.621366518, 0.731354605, 0.2862472356], [0.01058760429, 0.5951670053, 0.462001012], (1, 0, 0)], 
	[[2.621366518, 0.731354605, 0.2862472356], [0.01058760429, 0.5951670053, 0.462001012], (0, 1, 0)], 
	[[2.621366518, 0.731354605, 0.2862472356], [0.01058760429, 0.5951670053, 0.462001012], (0, 0, 1)]]}
			
	cdef list attributes = []
	cdef int i
	for i in range(len(basis_set_STO3G[atom])):
			attributes.append(basis_set_STO3G[atom][i])

	return attributes


DTYPE = np.double
ctypedef np.double_t DTYPE_t


cdef class BasisFunction:
		

		cdef public double origin1
		cdef public double origin2
		cdef public double origin3
		cdef public int shell1
		cdef public int shell2
		cdef public int shell3
		cdef public double exps1
		cdef public double exps2
		cdef public double exps3
		cdef public double coefs1
		cdef public double coefs2
		cdef public double coefs3
		cdef public double norm1
		cdef public double norm2
		cdef public double norm3

		def normalize(self):
			cdef int l , m , n , L
			l,m,n = self.shell1, self.shell2, self.shell3
			L = l+m+n
			cdef list exps, coefs
			exps = [self.exps1, self.exps2, self.exps3]
			coefs = [self.coefs1,self.coefs2, self.coefs3]
			cdef np.ndarray[DTYPE_t, ndim=1] norm
			# self.norm is a list of length equal to number primitives
			# normalize primitives first (PGBFs)
			norm = np.sqrt(np.power(2,2*(l+m+n)+1.5)*np.power(exps,l+m+n+1.5)/fact2(2*l-1)/fact2(2*m-1)/fact2(2*n-1)/np.power(np.pi,1.5))
			self.norm1 = norm[0]
			self.norm2 = norm[1]
			self.norm3 = norm[2]
			# now normalize the contracted basis functions (CGBFs)
			# Eq. 1.44 of Valeev integral whitepaper
			cdef double prefactor 
			cdef double N = 0.0
			prefactor = np.power(np.pi,1.5)*fact2(2*l - 1)*fact2(2*m - 1)*fact2(2*n - 1)/np.power(2.0,L)
			cdef int ia , ib
			for ia in range(3):
				for ib in range(3):
					N += norm[ia]*norm[ib]*coefs[ia]*coefs[ib]/np.power(exps[ia] + exps[ib],L+1.5)
			#N = prefactor*N
			N = np.power(N*prefactor,-0.5)
			self.coefs1 = N*self.coefs1
			self.coefs2 = N*self.coefs2
			self.coefs3 = N*self.coefs3





cpdef list orbital_config(list atoms ,list geom):
		cdef list attributes = []
		cdef int i , j
		for i in range(len(atoms)):
			temp_attri = Basis_attributes_finder(atoms[i])

			for j in range(len(temp_attri)):
				temp_attri[j] += [geom[i]]

			attributes += temp_attri
		cdef list orbital_objects = []
		cdef int k
		cdef BasisFunction Orbital
		for k in range(len(attributes)):
			
			Orbital = BasisFunction()
			Orbital.origin1 = attributes[k][3][0]
			Orbital.origin2 = attributes[k][3][1]
			Orbital.origin3 = attributes[k][3][2]
			Orbital.shell1  = attributes[k][2][0]
			Orbital.shell2  = attributes[k][2][1]
			Orbital.shell3  = attributes[k][2][2]
			Orbital.exps1   = attributes[k][0][0]
			Orbital.exps2   = attributes[k][0][1]
			Orbital.exps3   = attributes[k][0][2]
			Orbital.coefs1  = attributes[k][1][0]
			Orbital.coefs2  = attributes[k][1][1]
			Orbital.coefs3  = attributes[k][1][2]

			Orbital.normalize()
			orbital_objects.append(Orbital)

		return orbital_objects





cdef double E(int i,int j,int t, double Qx,double a,double b):
	cdef double p ,q
	p = a + b
	q = a*b/p
	if (t < 0) or (t > (i + j)):
	# out of bounds for t
		return 0.0
	elif i == j == t == 0:
	# base case
		return np.exp(-q*Qx*Qx) # K_AB
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




cdef double overlap(double a, int l1,int m1,int n1,double A1,double A2,double A3,double b,int l2,int m2,int n2,double B1,double B2,double B3):
	S1 = E(l1,l2,0,A1-B1,a,b) # X
	S2 = E(m1,m2,0,A2-B2,a,b) # Y
	S3 = E(n1,n2,0,A3-B3,a,b) # Z
	return S1*S2*S3*np.power(np.pi/(a+b),1.5)




cdef double S(BasisFunction a, BasisFunction b):
	cdef double s =0.0
	cdef int ia , ib
	cdef double ca ,cb
	#cdef list  anorm , aexps , ashell , aorigin , acoefs 
	#cdef list  bnorm , bexps , bshell , borigin , bcoefs
	cdef array.array anorm = array.array('d' , [a.norm1,a.norm2,a.norm3])
	cdef array.array bnorm = array.array('d' , [b.norm1,b.norm2,b.norm3])
	cdef array.array aexps = array.array('d' , [a.exps1,a.exps2,a.exps3])
	cdef array.array bexps = array.array('d' , [b.exps1,b.exps2,b.exps3])
	cdef array.array ashell= array.array('i' , [a.shell1,a.shell2,a.shell3])
	cdef array.array bshell= array.array('i' , [b.shell1,b.shell2,b.shell3])
	cdef array.array aorigin=array.array('d' , [a.origin1,a.origin2,a.origin3])
	cdef array.array borigin=array.array('d' , [b.origin1,b.origin2,b.origin3])
	cdef array.array acoefs= array.array('d' , [a.coefs1,a.coefs2,a.coefs3])
	cdef array.array bcoefs= array.array('d' , [b.coefs1,b.coefs2,b.coefs3])
	for ia , ca in enumerate(acoefs):
		for ib , cb in enumerate(bcoefs):
			s += anorm[ia]*bnorm[ib]*ca*cb*overlap(aexps[ia],ashell[0],ashell[1],ashell[2],aorigin[0],aorigin[1],aorigin[2],\
				bexps[ib],bshell[0],bshell[1],bshell[2],borigin[0],borigin[1],borigin[2])
	return s




cdef int fact2(int n):
	if n <= 1:
		return 1
	else:
		return n*fact2(n-2)




cpdef np.ndarray[DTYPE_t, ndim=2] S_mat(list atoms ,list geom):
	cdef list orbitals =[]
	cdef int nbasis
	orbitals = orbital_config(atoms,geom)
	nbasis = int(len(orbitals))
	cdef np.ndarray[DTYPE_t, ndim=2] overlap_int_matrix = np.zeros((nbasis,nbasis),dtype = DTYPE)
	cdef int i , j
	for i in range(nbasis):
		for j in range(0 , i+1):
			overlap_int_matrix[i][j] =overlap_int_matrix[j][i]  = S(orbitals[i],orbitals[j])

	return overlap_int_matrix


cdef double kinetic(double a,int l1,int m1,int n1,double A1,double A2,double A3,double b,int l2,int m2,int n2,double B1,double B2,double B3):

	cdef double term0 , term1 ,term2
	term0 = b*(2*(l2+m2+n2)+3)*overlap(a,l1,m1,n1,A1,A2,A3,b,l2,m2,n2,B1,B2,B3)
	term1 = -2*np.power(b,2)*(overlap(a,l1,m1,n1,A1,A2,A3,b,l2+2,m2,n2,B1,B2,B3) \
		+ overlap(a,l1,m1,n1,A1,A2,A3,b,l2,m2+2,n2,B1,B2,B3) + overlap(a,l1,m1,n1,A1,A2,A3,b,l2,m2,n2+2,B1,B2,B3))
	term2 = -0.5*(l2*(l2-1)*overlap(a,l1,m1,n1,A1,A2,A3,b,l2-2,m2,n2,B1,B2,B3) + m2*(m2-1)*overlap(a,l1,m1,n1,A1,A2,A3,b,l2,m2-2,n2,B1,B2,B3) \
		+ n2*(n2-1)*overlap(a,l1,m1,n1,A1,A2,A3,b,l2,m2,n2-2,B1,B2,B3))
	return term0+term1+term2



cdef double T(BasisFunction a, BasisFunction b):
		cdef double t =0.0
		cdef int ia , ib
		cdef double ca , cb
		cdef array.array anorm = array.array('d' , [a.norm1,a.norm2,a.norm3])
		cdef array.array bnorm = array.array('d' , [b.norm1,b.norm2,b.norm3])
		cdef array.array aexps = array.array('d' , [a.exps1,a.exps2,a.exps3])
		cdef array.array bexps = array.array('d' , [b.exps1,b.exps2,b.exps3])
		cdef array.array ashell= array.array('i' , [a.shell1,a.shell2,a.shell3])
		cdef array.array bshell= array.array('i' , [b.shell1,b.shell2,b.shell3])
		cdef array.array aorigin=array.array('d' , [a.origin1,a.origin2,a.origin3])
		cdef array.array borigin=array.array('d' , [b.origin1,b.origin2,b.origin3])
		cdef array.array acoefs= array.array('d' , [a.coefs1,a.coefs2,a.coefs3])
		cdef array.array bcoefs= array.array('d' , [b.coefs1,b.coefs2,b.coefs3])
		for ia, ca in enumerate(acoefs):
			for ib, cb in enumerate(bcoefs):
				t += anorm[ia]*bnorm[ib]*ca*cb*kinetic(aexps[ia],ashell[0],ashell[1],ashell[2],aorigin[0],\
					aorigin[1],aorigin[2],bexps[ib],bshell[0],bshell[1],bshell[2],borigin[0],borigin[1],borigin[2])
		return t





cpdef np.ndarray[DTYPE_t, ndim=2] T_mat(list atoms ,list geom):
	cdef list orbitals =[]
	cdef int nbasis
	orbitals = orbital_config(atoms,geom)
	nbasis = int(len(orbitals))
	cdef np.ndarray[DTYPE_t, ndim=2] Kineic_matrix = np.zeros((nbasis,nbasis),dtype = DTYPE)
	cdef int i , j
	for i in range(nbasis):
		for j in range(0 , i+1):
			Kineic_matrix[i][j] = Kineic_matrix[j][i] = T(orbitals[i],orbitals[j])

	return Kineic_matrix


cdef double R(int t,int u,int v,int n,double p,double PCx,double PCy,double PCz,double RPC):
	cdef double T , val
	T = p*RPC*RPC
	val = 0.0
	if t == u == v == 0:
		val += np.power(-2*p,n)*boys(n,T)
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


cdef double boys(int n,double T):
	return hyp1f1(n+0.5,n+1.5,-T)/(2.0*n+1.0)

cdef double gaussian_product_center(double a,double A,double b,double B):
	return (a*A+b*B)/(a+b)



cdef double nuclear_attraction(double a,int l1,int m1,int n1,double A1,double A2,double A3,double b,int l2,int m2,\
	int n2,double B1,double B2,double B3,double C1,double C2,double C3):

	cdef double p ,RPC ,val
	p = a + b
	cdef list C = [C1,C2,C3]
	cdef list P = [gaussian_product_center(a,A1,b,B1),gaussian_product_center(a,A2,b,B2),gaussian_product_center(a,A3,b,B3)]
	cdef list RPCdash = [C[0]-P[0],C[1]-P[1],C[2]-P[2]]
	RPC = np.linalg.norm(RPCdash)
	val = 0.0
	cdef int t,u,v
	cdef double AB1 ,AB2 ,AB3 ,PC1 , PC2 , PC3
	AB1=A1-B1
	AB2=A2-B2
	AB3=A3-B3
	PC1=P[0]-C[0]
	PC2=P[1]-C[1]
	PC3=P[2]-C[2]
	for t in range(int(l1+l2+1)):
		for u in range(int(m1+m2+1)):
			for v in range(int(n1+n2+1)):
				val += E(l1,l2,t,AB1,a,b)*E(m1,m2,u,AB2,a,b)*E(n1,n2,v,AB3,a,b)*R(t,u,v,0,p,PC1,PC2,PC3,RPC)
	val *= 2*np.pi/p
	return val

cdef double V(BasisFunction a, BasisFunction b,list C):
	cdef double v = 0.0
	cdef int ia , ib
	cdef double ca , cb
	cdef array.array anorm = array.array('d' , [a.norm1,a.norm2,a.norm3])
	cdef array.array bnorm = array.array('d' , [b.norm1,b.norm2,b.norm3])
	cdef array.array aexps = array.array('d' , [a.exps1,a.exps2,a.exps3])
	cdef array.array bexps = array.array('d' , [b.exps1,b.exps2,b.exps3])
	cdef array.array ashell= array.array('i' , [a.shell1,a.shell2,a.shell3])
	cdef array.array bshell= array.array('i' , [b.shell1,b.shell2,b.shell3])
	cdef array.array aorigin=array.array('d' , [a.origin1,a.origin2,a.origin3])
	cdef array.array borigin=array.array('d' , [b.origin1,b.origin2,b.origin3])
	cdef array.array acoefs= array.array('d' , [a.coefs1,a.coefs2,a.coefs3])
	cdef array.array bcoefs= array.array('d' , [b.coefs1,b.coefs2,b.coefs3])
	for ia, ca in enumerate(acoefs):
		for ib, cb in enumerate(bcoefs):
			v += anorm[ia]*bnorm[ib]*ca*cb*nuclear_attraction(aexps[ia],ashell[0],ashell[1],ashell[2],aorigin[0],\
				aorigin[1],aorigin[2],bexps[ib],bshell[0],bshell[1],bshell[2],borigin[0],borigin[1],borigin[2],C[0],C[1],C[2])
	return v


cpdef np.ndarray[DTYPE_t, ndim=2] V_mat(list atoms ,list geom):
	cdef list orbitals = orbital_config(atoms,geom)
	cdef int nbasis ,i ,j ,k
	nbasis = int(len(orbitals))
	cdef np.ndarray[DTYPE_t, ndim=2] Potential_matrix = np.zeros((nbasis,nbasis),dtype=DTYPE)
	cdef double v =0.0
	for i in range(nbasis):
		for j in range(nbasis):
			for k in range(len(atoms)):
				v += V(orbitals[i], orbitals[j],geom[k])*int(no_of_e(atoms[k]))
			Potential_matrix[i][j] = -v

	cdef np.ndarray[DTYPE_t, ndim=2] Potential_matrix_out = np.zeros((nbasis, nbasis),dtype=DTYPE)

	for i in range(nbasis):
		for j in range(nbasis):
			if i ==0 and j==0 :
				Potential_matrix_out[i][j] = Potential_matrix[0][0]
	
			else :
				Potential_matrix_out[i][j] = -(Potential_matrix[i][j-1] -Potential_matrix[i][j])

	for i in range(nbasis):
		Potential_matrix_out[i][0] = Potential_matrix_out[0][i]

	return Potential_matrix_out

cpdef double electron_repulsion(double a,int l1,int m1,int n1,double A0,double A1,double A2,\
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
		RPQdash = [P0-Q0,P1-Q1,P2-Q2]
		RPQ = np.linalg.norm(RPQdash)
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
										np.power(-1,tau+nu+phi) * \
										R(t+tau,u+nu,v+phi,0,\
											alpha,P0-Q0,P1-Q1,P2-Q2,RPQ)
		val *= 2*np.power(np.pi,2.5)/(p*q*np.sqrt(p+q))
		return val



cpdef double ERI(BasisFunction a, BasisFunction b,BasisFunction c, BasisFunction d):
		'''Evaluates overlap between two contracted Gaussians
		Returns float.
		Arguments:
		a: contracted Gaussian 'a', BasisFunction object
		b: contracted Gaussian 'b', BasisFunction object
		c: contracted Gaussian 'b', BasisFunction object
		d: contracted Gaussian 'b', BasisFunction object
		'''
		cdef double eri =0.0
		cdef int ja,jb,jc,jd
		cdef double ca , cb , cc ,cd
		cdef array.array anorm = array.array('d' , [a.norm1,a.norm2,a.norm3])
		cdef array.array bnorm = array.array('d' , [b.norm1,b.norm2,b.norm3])
		cdef array.array aexps = array.array('d' , [a.exps1,a.exps2,a.exps3])
		cdef array.array bexps = array.array('d' , [b.exps1,b.exps2,b.exps3])
		cdef array.array acoefs= array.array('d' , [a.coefs1,a.coefs2,a.coefs3])
		cdef array.array bcoefs= array.array('d' , [b.coefs1,b.coefs2,b.coefs3])
		cdef array.array cnorm = array.array('d' , [c.norm1,c.norm2,c.norm3])
		cdef array.array dnorm = array.array('d' , [d.norm1,d.norm2,d.norm3])
		cdef array.array cexps = array.array('d' , [c.exps1,c.exps2,c.exps3])
		cdef array.array dexps = array.array('d' , [d.exps1,d.exps2,d.exps3])
		cdef array.array ccoefs= array.array('d' , [c.coefs1,c.coefs2,c.coefs3])
		cdef array.array dcoefs= array.array('d' , [d.coefs1,d.coefs2,d.coefs3])
		for ja in range(3):
			for jb in range(3):
				for jc in range(3):
					for jd in range(3):
						eri += anorm[ja]*bnorm[jb]*cnorm[jc]*dnorm[jd]*\
								acoefs[ja]*bcoefs[jb]*ccoefs[jc]*dcoefs[jd]*\
								electron_repulsion(aexps[ja],a.shell1,a.shell2,a.shell3,a.origin1,a.origin2,a.origin3,\
								bexps[jb],b.shell1,b.shell2,b.shell3,b.origin1,b.origin2,b.origin3,\
								cexps[jc],c.shell1,c.shell2,c.shell3,c.origin1,c.origin2,c.origin3,\
								dexps[jd],d.shell1,d.shell2,d.shell3,d.origin1,d.origin2,d.origin3)

		return eri



cdef list unlister(list a):
		cdef set b
		cdef list c
		b = set(a)
		c = list(b)
		return c


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

cdef list unique_indices(int nbasis):
		cdef list unlist =[] 
		cdef list list1 =[]
		cdef list list2 =[]
		cdef int i,j,k,l
		for  i in range(nbasis):
			for j in range(nbasis):
				for k in range(nbasis):
					for l in range(nbasis):
						list1.append((i,j,k,l))		

		for  i in range(nbasis):
			for j in range(nbasis):
				for k in range(nbasis):
					for l in range(nbasis):
						unlist.append(compound_index(i,j,k,l))

		list2 = unlister(unlist)
		cdef list indices= []
		for i in range(len(list2)):
			indices.append(unlist.index(list2[i]))
		cdef list tei_unique = []
		for i in range(len(indices)):
			tei_unique.append(list1[indices[i]])
		return tei_unique



def Eri_mat(list atoms ,list geom, double[:,::1] Overlap_mat):
	cdef list orbitals =[]
	cdef int nbasis
	orbitals = orbital_config(atoms,geom)
	nbasis = len(orbitals)
	cdef int i
	cdef list unique_indices_list
	unique_indices_list = unique_indices(nbasis)
	cdef np.ndarray[DTYPE_t, ndim=4] Temp_mat = np.zeros((nbasis,nbasis,nbasis,nbasis),dtype = DTYPE)
	cdef list temp_eri_1D = []
	cdef int a,b,c,d
	for i in range(len(unique_indices_list)):
		a = unique_indices_list[i][0]
		b = unique_indices_list[i][1]
		c = unique_indices_list[i][2]
		d = unique_indices_list[i][3]
		#print(Overlap_mat[a][b],Overlap_mat[c][d])
		#if Overlap_mat[a][b]==0.0 or Overlap_mat[c][d]==0.0:
			#print(Overlap_mat[a][b],Overlap_mat[c][d])
		#	eri_temp = 0.0
		#else:
		eri_temp = ERI(orbitals[a],orbitals[b],orbitals[c],orbitals[d])
		temp_eri_1D.append([a,b,c,d,eri_temp])
	cdef int x , y , z, w
	
	for i in range(len(temp_eri_1D)):
		x=temp_eri_1D[i][0]
		y=temp_eri_1D[i][1]
		z=temp_eri_1D[i][2]
		w=temp_eri_1D[i][3]
		Temp_mat[x][y][z][w]=temp_eri_1D[i][4]
		Temp_mat[y][x][z][w]=temp_eri_1D[i][4]
		Temp_mat[x][y][w][z]=temp_eri_1D[i][4]
		Temp_mat[y][x][w][z]=temp_eri_1D[i][4]
		Temp_mat[z][w][x][y]=temp_eri_1D[i][4]
		Temp_mat[w][z][x][y]=temp_eri_1D[i][4]
		Temp_mat[z][w][y][x]=temp_eri_1D[i][4]
		Temp_mat[w][z][y][x]=temp_eri_1D[i][4]
	
	return Temp_mat

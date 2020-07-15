import numpy as np  
from scipy.special import hyp1f1
from help_functions import compound_index

def no_of_e(element):
  symbol = [
            'H','He',
            'Li','Be','B','C','N','O','F','Ne',
            'Na','Mg','Al','Si','P','S','Cl','Ar',
            'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe',
            'Co', 'Ni', 'Cu', 'Zn',
            'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr',
            'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru',
            'Rh', 'Pd', 'Ag', 'Cd',
            'In', 'Sn', 'Sb', 'Te', 'I', 'Xe',
            'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm',  'Eu',
            'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu',
            'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg',
            'Tl','Pb','Bi','Po','At','Rn']
  u=symbol.index(element)+1
  return u

def Basis_attributes_finder(atom):
	
	basis_set_STO3G = {'H' :  [[[0.3425250914E+01,  0.6239137298E+00,  0.1688554040E+00],[0.1543289673E+00,  0.5353281423E+00,  0.4446345422E+00],(0,0,0)]] ,

	                   'C' :  [[[0.7161683735E+02,  0.1304509632E+02,  0.3530512160E+01],[0.1543289673E+00,  0.5353281423E+00,  0.4446345422E+00],(0,0,0)],
	     	                   [[0.2941249355E+01,  0.6834830964E+00,  0.2222899159E+00],[-0.9996722919E-01, 0.3995128261E+00,  0.7001154689E+00],(0,0,0)],
	     	                   [[0.2941249355E+01,  0.6834830964E+00,  0.2222899159E+00],[0.1559162750E+00,  0.6076837186E+00,  0.3919573931E+00],(1,0,0)],
	     	                   [[0.2941249355E+01,  0.6834830964E+00,  0.2222899159E+00],[0.1559162750E+00,  0.6076837186E+00,  0.3919573931E+00],(0,1,0)],
	                           [[0.2941249355E+01,  0.6834830964E+00,  0.2222899159E+00],[0.1559162750E+00,  0.6076837186E+00,  0.3919573931E+00],(0,0,1)]],

	                   'O' :  [[[0.1307093214E+03,  0.2380886605E+02,  0.6443608313E+01],[0.1543289673E+00,  0.5353281423E+00,  0.4446345422E+00],(0,0,0)],
	     	                   [[0.5033151319E+01,  0.1169596125E+01,  0.3803889600E+00],[-0.9996722919E-01, 0.3995128261E+00,  0.7001154689E+00],(0,0,0)],
	     	                   [[0.5033151319E+01,  0.1169596125E+01,  0.3803889600E+00],[0.1559162750E+00,  0.6076837186E+00,  0.3919573931E+00],(1,0,0)],
	     	                   [[0.5033151319E+01,  0.1169596125E+01,  0.3803889600E+00],[0.1559162750E+00,  0.6076837186E+00,  0.3919573931E+00],(0,1,0)],
	     	                   [[0.5033151319E+01,  0.1169596125E+01,  0.3803889600E+00],[0.1559162750E+00,  0.6076837186E+00,  0.3919573931E+00],(0,0,1)]], 

	     	           'N' :  [[[0.9910616896E+02,  0.1805231239E+02,  0.4885660238E+01],[0.1543289673E+00,  0.5353281423E+00,  0.4446345422E+00],(0,0,0)],
	     	           		   [[0.3780455879E+01,  0.8784966449E+00,  0.2857143744E+00],[-0.9996722919E-01, 0.3995128261E+00,  0.7001154689E+00],(0,0,0)],
	     	           		   [[0.3780455879E+01,  0.8784966449E+00,  0.2857143744E+00],[0.1559162750E+00,  0.6076837186E+00,  0.3919573931E+00],(1,0,0)],
	     	           		   [[0.3780455879E+01,  0.8784966449E+00,  0.2857143744E+00],[0.1559162750E+00,  0.6076837186E+00,  0.3919573931E+00],(0,1,0)],
	     	           		   [[0.3780455879E+01,  0.8784966449E+00,  0.2857143744E+00],[0.1559162750E+00,  0.6076837186E+00,  0.3919573931E+00],(0,0,1)]]}
	attributes = []                         
	for i in range(len(basis_set_STO3G[atom])):
			attributes.append(basis_set_STO3G[atom][i])

	return attributes

def round_up(V):
  V = np.around(V, decimals=15).tolist()
  return V

def orbital_config(atoms , geom):
	attributes = []
	for i in range(len(atoms)):
		attributes += Basis_attributes_finder(atoms[i])
		length_attri = len(attributes)
		if i ==0 :
			for j in range(len(Basis_attributes_finder(atoms[i]))):
				attributes[j] += [geom[i]]
		else :
			for j in range(len(Basis_attributes_finder(atoms[i]))):
				attributes[length_attri+j-1] += [geom[i]]		

	orbital_objects = []
	for i in range(len(attributes)):
		orbital_objects.append(BasisFunction( attributes[i][3] ,attributes[i][2], attributes[i][0] ,attributes[i][1]))
		

	return orbital_objects



def E(i,j,t,Qx,a,b):
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
            return (1/(2*p))*E(i-1,j,t-1,Qx,a,b) - \
            (q*Qx/a)*E(i-1,j,t,Qx,a,b) + \
            (t+1)*E(i-1,j,t+1,Qx,a,b)
        else:
        # decrement index j
            return (1/(2*p))*E(i,j-1,t-1,Qx,a,b) + \
            (q*Qx/b)*E(i,j-1,t,Qx,a,b) + \
            (t+1)*E(i,j-1,t+1,Qx,a,b)


def overlap(a,lmn1,A,b,lmn2,B):
    l1,m1,n1 = lmn1 # shell angular momentum on Gaussian 'a'
    l2,m2,n2 = lmn2 # shell angular momentum on Gaussian 'b'
    S1 = E(l1,l2,0,A[0]-B[0],a,b) # X
    S2 = E(m1,m2,0,A[1]-B[1],a,b) # Y
    S3 = E(n1,n2,0,A[2]-B[2],a,b) # Z
    return S1*S2*S3*np.power(np.pi/(a+b),1.5)




def S(a,b):
    s = 0.0
    for ia, ca in enumerate(a.coefs):
          for ib, cb in enumerate(b.coefs):
              s += a.norm[ia]*b.norm[ib]*ca*cb*overlap(a.exps[ia],a.shell,a.origin,b.exps[ib],b.shell,b.origin)
    return s




def fact2(n):
  if n <= 1:
     return 1
  else:
     return n*fact2(n-2)




class BasisFunction(object):

      def __init__(self,origin=[ 0.0 ,0.0 ,0.0 ],shell=(0,0,0),exps=[],coefs=[]):
            self.origin = np.asarray(origin)
            self.shell = shell
            self.exps = exps
            self.coefs = coefs
            self.norm = None
            self.normalize()


      def normalize(self):
            l,m,n = self.shell
            L = l+m+n
            # self.norm is a list of length equal to number primitives
            # normalize primitives first (PGBFs)
            self.norm = np.sqrt(np.power(2,2*(l+m+n)+1.5)*np.power(self.exps,l+m+n+1.5)/fact2(2*l-1)/fact2(2*m-1)/fact2(2*n-1)/np.power(np.pi,1.5))
            # now normalize the contracted basis functions (CGBFs)
            # Eq. 1.44 of Valeev integral whitepaper
            prefactor = np.power(np.pi,1.5)*fact2(2*l - 1)*fact2(2*m - 1)*fact2(2*n - 1)/np.power(2.0,L)
            N = 0.0
            num_exps = len(self.exps)
            for ia in range(num_exps):
               for ib in range(num_exps):
                   N += self.norm[ia]*self.norm[ib]*self.coefs[ia]*self.coefs[ib]/np.power(self.exps[ia] + self.exps[ib],L+1.5)
            N *= prefactor
            N = np.power(N,-0.5)
            for ia in range(num_exps):
                  self.coefs[ia] *= N

def S_mat(atoms , geom):
	orbitals = orbital_config(atoms,geom)
	nbasis = int(len(orbitals))
	overlap_int_matrix = np.zeros((nbasis,nbasis))
	for i in range(nbasis):
		for j in range(0 , i+1):
			overlap_int_matrix[i][j] =overlap_int_matrix[j][i]  = S(orbitals[i],orbitals[j])

	return overlap_int_matrix


def kinetic(a,lmn1,A,b,lmn2,B):
		l1,m1,n1 = lmn1
		l2,m2,n2 = lmn2
		term0 = b*(2*(l2+m2+n2)+3)*overlap(a,(l1,m1,n1),A,b,(l2,m2,n2),B)
		term1 = -2*np.power(b,2)*(overlap(a,(l1,m1,n1),A,b,(l2+2,m2,n2),B) + overlap(a,(l1,m1,n1),A,b,(l2,m2+2,n2),B) + overlap(a,(l1,m1,n1),A,b,(l2,m2,n2+2),B))
		term2 = -0.5*(l2*(l2-1)*overlap(a,(l1,m1,n1),A,b,(l2-2,m2,n2),B) + m2*(m2-1)*overlap(a,(l1,m1,n1),A,b,(l2,m2-2,n2),B) + n2*(n2-1)*overlap(a,(l1,m1,n1),A,b,(l2,m2,n2-2),B))
		return term0+term1+term2



def T(a,b):
		t = 0.0
		for ia, ca in enumerate(a.coefs):
			for ib, cb in enumerate(b.coefs):
				t += a.norm[ia]*b.norm[ib]*ca*cb*kinetic(a.exps[ia],a.shell,a.origin,b.exps[ib],b.shell,b.origin)
		return t

def T_mat(atoms , geom):
	orbitals = orbital_config(atoms,geom)
	nbasis = int(len(orbitals))
	Kineic_matrix = np.zeros((nbasis,nbasis))
	for i in range(nbasis):
		for j in range(0 , i+1):
			Kineic_matrix[i][j] = Kineic_matrix[j][i] = T(orbitals[i],orbitals[j])

	return Kineic_matrix





def R(t,u,v,n,p,PCx,PCy,PCz,RPC):
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


def boys(n,T):
	return hyp1f1(n+0.5,n+1.5,-T)/(2.0*n+1.0)

def gaussian_product_center(a,A,b,B):
	return (a*A+b*B)/(a+b)



def nuclear_attraction(a,lmn1,A,b,lmn2,B,C):

	l1,m1,n1 = lmn1
	l2,m2,n2 = lmn2
	p = a + b
	P = gaussian_product_center(a,A,b,B) # Gaussian composite center
	RPC = np.linalg.norm(P-C)
	val = 0.0
	for t in range(l1+l2+1):
		for u in range(m1+m2+1):
			for v in range(n1+n2+1):
				val += E(l1,l2,t,A[0]-B[0],a,b)*E(m1,m2,u,A[1]-B[1],a,b)*E(n1,n2,v,A[2]-B[2],a,b)*R(t,u,v,0,p,P[0]-C[0],P[1]-C[1],P[2]-C[2],RPC)
	val *= 2*np.pi/p
	return val

def V(a,b,C):
	v = 0.0
	for ia, ca in enumerate(a.coefs):
		for ib, cb in enumerate(b.coefs):
			v += a.norm[ia]*b.norm[ib]*ca*cb*nuclear_attraction(a.exps[ia],a.shell,a.origin,b.exps[ib],b.shell,b.origin,C)
	return v




def V_mat(atoms , geom):
	orbitals = orbital_config(atoms,geom)
	nbasis = int(len(orbitals))
	Potential_matrix = np.zeros((nbasis,nbasis))
	v = 0
	for i in range(nbasis):
		for j in range(nbasis):
			for k in range(len(atoms)):
				v += V(orbitals[i], orbitals[j],geom[k])*no_of_e(atoms[k])
			Potential_matrix[i][j] = -v

	Potential_matrix_out = np.zeros((nbasis, nbasis))

	for i in range(nbasis):
		for j in range(nbasis):
			if i ==0 and j==0 :
				Potential_matrix_out[i][j] = Potential_matrix[0][0]
	
			else :
				Potential_matrix_out[i][j] = -(Potential_matrix[i][j-1] -Potential_matrix[i][j])

	for i in range(nbasis):
		Potential_matrix_out[i][0] = Potential_matrix_out[0][i]

	return Potential_matrix_out


def electron_repulsion(a,lmn1,A,b,lmn2,B,c,lmn3,C,d,lmn4,D):
		''' Evaluates kinetic energy integral between two Gaussians
		Returns a float.
		a,b,c,d: orbital exponent on Gaussian 'a','b','c','d'
		lmn1,lmn2
		lmn3,lmn4: int tuple containing orbital angular momentum
		for Gaussian 'a','b','c','d', respectively
		A,B,C,D: list containing origin of Gaussian 'a','b','c','d'
		'''
		l1,m1,n1 = lmn1
		l2,m2,n2 = lmn2
		l3,m3,n3 = lmn3
		l4,m4,n4 = lmn4
		p = a+b # composite exponent for P (from Gaussians 'a' and 'b')
		q = c+d # composite exponent for Q (from Gaussians 'c' and 'd')
		alpha = p*q/(p+q)
		P = gaussian_product_center(a,A,b,B) # A and B composite center
		Q = gaussian_product_center(c,C,d,D) # C and D composite center
		RPQ = np.linalg.norm(P-Q)
		val = 0.0
		for t in range(l1+l2+1):
			for u in range(m1+m2+1):
				for v in range(n1+n2+1):
					for tau in range(l3+l4+1):
						for nu in range(m3+m4+1):
							for phi in range(n3+n4+1):
								val += E(l1,l2,t,A[0]-B[0],a,b) * \
										E(m1,m2,u,A[1]-B[1],a,b) * \
										E(n1,n2,v,A[2]-B[2],a,b) * \
										E(l3,l4,tau,C[0]-D[0],c,d) * \
										E(m3,m4,nu ,C[1]-D[1],c,d) * \
										E(n3,n4,phi,C[2]-D[2],c,d) * \
										np.power(-1,tau+nu+phi) * \
										R(t+tau,u+nu,v+phi,0,\
											alpha,P[0]-Q[0],P[1]-Q[1],P[2]-Q[2],RPQ)
		val *= 2*np.power(np.pi,2.5)/(p*q*np.sqrt(p+q))
		return val



def ERI(a,b,c,d):
		'''Evaluates overlap between two contracted Gaussians
		Returns float.
		Arguments:
		a: contracted Gaussian 'a', BasisFunction object
		b: contracted Gaussian 'b', BasisFunction object
		c: contracted Gaussian 'b', BasisFunction object
		d: contracted Gaussian 'b', BasisFunction object
		'''
		eri = 0.0
		for ja, ca in enumerate(a.coefs):
			for jb, cb in enumerate(b.coefs):
				for jc, cc in enumerate(c.coefs):
					for jd, cd in enumerate(d.coefs):
						eri += a.norm[ja]*b.norm[jb]*c.norm[jc]*d.norm[jd]*\
								ca*cb*cc*cd*\
								electron_repulsion(a.exps[ja],a.shell,a.origin,\
								b.exps[jb],b.shell,b.origin,\
								c.exps[jc],c.shell,c.origin,\
								d.exps[jd],d.shell,d.origin)
		return eri





def unique_indices(lit):


		tei = []
		for  i in range(len(lit)):
		  for j in range(len(lit)):
		    for k in range(len(lit)):
		      for l in range(len(lit)):
		        tei.append((lit[i],lit[j],lit[k],lit[l]))

		temp = []




		for i in range(len(temp)):
			print(temp[i])

		unlist = []
		for  i in range(len(lit)):
		  for j in range(len(lit)):
		    for k in range(len(lit)):
		      for l in range(len(lit)):
		      	unlist.append(compound_index(i,j,k,l))

		#for i in range(len(unlist)):
		#	print(unlist[i])


		list2 = set(unlist)

		list3 = list(list2)

		indices = []
		for i in range(len(list3)):
			indices.append(unlist.index(list3[i]))


		tei_unique = []
		#print(indices)
		#print(len(indices))
		for i in range(len(indices)):
			tei_unique.append(tei[indices[i]])
		return tei_unique


def Eri_mat(atoms , geom, S_matrix):
	orbitals = orbital_config(atoms,geom)
	unique_indices_list = unique_indices(orbitals)
	#print(len(unique_indices_list))
	nbasis = int(len(orbitals))
	Temp_mat = np.zeros((nbasis,nbasis,nbasis,nbasis))
	temp_eri_1D = []
	for i in range(len(unique_indices_list)):
		eri_temp = ERI(unique_indices_list[i][0],unique_indices_list[i][1],unique_indices_list[i][2],unique_indices_list[i][3])
		temp_eri_1D.append([orbitals.index(unique_indices_list[i][0])]+[orbitals.index(unique_indices_list[i][1])] \
			+[orbitals.index(unique_indices_list[i][2])]+[orbitals.index(unique_indices_list[i][3])] +[round_up(eri_temp)])

	#for i in range(len(temp_eri_1D)):
	#	print(temp_eri_1D[i])

	for i in range(len(temp_eri_1D)):
		x=int(temp_eri_1D[i][0])  
		y=int(temp_eri_1D[i][1]) 
		z=int(temp_eri_1D[i][2]) 
		w=int(temp_eri_1D[i][3]) 
		Temp_mat[x][y][z][w]=float(temp_eri_1D[i][4])
		Temp_mat[y][x][z][w]=float(temp_eri_1D[i][4])
		Temp_mat[x][y][w][z]=float(temp_eri_1D[i][4])
		Temp_mat[y][x][w][z]=float(temp_eri_1D[i][4])
		Temp_mat[z][w][x][y]=float(temp_eri_1D[i][4])
		Temp_mat[w][z][x][y]=float(temp_eri_1D[i][4])
		Temp_mat[z][w][y][x]=float(temp_eri_1D[i][4])
		Temp_mat[w][z][y][x]=float(temp_eri_1D[i][4])
	return Temp_mat




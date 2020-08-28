# cython: language_level=3, boundscheck=False
import numpy as np
import itertools
cimport numpy as np
cimport cython
#########################################################
#Function to define no of Electron
#On Entry 
#element--> name of the element
#On Exit
#u--> atomic no
########################################################

cpdef int no_of_e(element):
  cdef list symbol = ['H','He', 'Li','Be','B','C','N','O','F','Ne','Na','Mg','Al','Si','P','S','Cl','Ar','K', 'Ca', 'Sc',
                 'Ti', 'V', 'Cr', 'Mn', 'Fe','Co', 'Ni', 'Cu', 'Zn','Ga', 'Ge', 'As', 'Se', 'Br', 'Kr','Rb', 'Sr', 'Y', 'Zr',
                 'Nb', 'Mo', 'Tc', 'Ru','Rh', 'Pd', 'Ag', 'Cd','In', 'Sn', 'Sb', 'Te', 'I', 'Xe','Cs', 'Ba', 'La', 'Ce', 'Pr',
                 'Nd', 'Pm', 'Sm',  'Eu','Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu','Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt',
                  'Au', 'Hg','Tl','Pb','Bi','Po','At','Rn']
  cdef int u 
  u=symbol.index(element)+1
  return u
#########################################################
# Calculate the distance between two atoms
#On Entry 
#a--> position of first atom, in list containing x,y and z
#b--> position of second atom
#On return
#R--> distance between two atoms
##########################################################
cpdef double find_distance(list a, list b):
      cdef double x_1 , x_2 , y_1 , y_2 , z_1 , z_2
      x_1=a[0]
      x_2=b[0]
      y_1=a[1]
      y_2=b[1]
      z_1=a[2]
      z_2=b[2]

      R_square =(x_1-x_2)**2+(y_1-y_2)**2+(z_1-z_2)**2
      R=np.sqrt(R_square)

      return R


#####################################################
# The compound index code 
#
#on Input 
#i,j,k,l --> four integers
#
#On Return 
#ijkl-->gives the compound index of I,J,K,L
#
#####################################################
cpdef int compound_index( int i, int j, int k, int l):
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

#########################################################
#This function calculate X=S^-1/2
#
#On input
#S--> overlap matrix dimension (nbasis,nbasis)
#
#On output
#X--> the transformation matrix dimension (nbasis,nbasis)
##########################################################
DTYPE = np.double
ctypedef np.double_t DTYPE_t

def  get_X(double[:,::1] S):
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
    
cpdef int no_of_electrons(list atoms):
                cdef list symbol = ['H','He', 'Li','Be','B','C','N','O','F','Ne','Na','Mg','Al','Si','P','S','Cl','Ar','K', 'Ca', 'Sc',
                 'Ti', 'V', 'Cr', 'Mn', 'Fe','Co', 'Ni', 'Cu', 'Zn','Ga', 'Ge', 'As', 'Se', 'Br', 'Kr','Rb', 'Sr', 'Y', 'Zr',
                 'Nb', 'Mo', 'Tc', 'Ru','Rh', 'Pd', 'Ag', 'Cd','In', 'Sn', 'Sb', 'Te', 'I', 'Xe','Cs', 'Ba', 'La', 'Ce', 'Pr',
                 'Nd', 'Pm', 'Sm',  'Eu','Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu','Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt',
                  'Au', 'Hg','Tl','Pb','Bi','Po','At','Rn']

                cdef int mole_elec=0
                for i in range(len(atoms)):
                     mole_elec+= symbol.index(atoms[i])+1

                return mole_elec

cpdef double Masscalculator(atoms):
                cdef dict Atomic_mass_dictionary = {'H':'1.008','He':'4.002','Li':'6.938','Be':'9.012','B':'10.806','C':'12.0096','N':'14.00643','O':'15.99903','F':'18.998','Ne':'20.1797','Na':'22.989',
'Mg':'24.304','Al':'26.981','Si':'28.084','P':'30.973','S':'32.059','Cl':'35.446','Ar':'39.948','K':'39.0983', 'Ca':'40.078', 'Sc':'44.955', 'Ti':'47.867',
'V':'50.9415','Cr':'51.9961', 'Mn':'54.938', 'Fe':'55.845','Co':'58.933', 'Ni':'58.6934', 'Cu':'63.546', 'Zn':'65.38','Ga':'69.723', 'Ge':'72.630',
'As':'74.921', 'Se':'78.971', 'Br':'79.901', 'Kr':'83.798','Rb':'85.4678', 'Sr':'87.62', 'Y':'88.905', 'Zr':'91.224', 'Nb':'92.906', 'Mo':'95.95', 
'Tc':'98', 'Ru':'101.07','Rh':'102.905', 'Pd':'106.42', 'Ag':'107.8682', 'Cd':'112.414','In':'114.818', 'Sn':'118.710', 'Sb':'121.760', 'Te':'127.60', 'I':'126.904'
, 'Xe':'131.293','Cs':'132.905', 'Ba':'137.327', 'La':'138.905 ', 'Ce':'140.116', 'Pr':'140.907 ', 'Nd':'144.242', 'Pm':'145', 'Sm':'150.36',  'Eu':'151.964',
'Gd':'157.25', 'Tb':'158.925', 'Dy':'162.500', 'Ho':'164.930 ', 'Er':'167.259', 'Tm':'168.934 ', 'Yb':'173.054', 'Lu':'174.9668','Hf':'178.49', 'Ta':'180.947 ',
'W':'183.84', 'Re':'186.207', 'Os':'190.23', 'Ir':'192.217', 'Pt':'195.084', 'Au':'196.966 ', 'Hg':'200.592','Tl':'204.382','Pb':'207.2','Bi':'208.980 ','Po':'209','At':'210','Rn':'222'}
                cdef double mass=0.0
                mass=Atomic_mass_dictionary[atoms]
                return mass



cpdef double round_up(double V):
  V = np.around(V, decimals=15).tolist()
  return V




def make_fock(double[:,::1] D,double[:,::1] hamil,double[:,:,:,::1] eri):
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


def make_C(double[:,::1] s_inv_root ,  double[:,::1] Fock):
      fock_ini=np.linalg.multi_dot([np.transpose(s_inv_root),Fock,s_inv_root])
      cdef double[:,::1] fock_ini_view = fock_ini
      E ,C_dash = np.linalg.eigh(fock_ini_view)
      cdef double[:,::1] C_dash_view = C_dash
      C=np.matmul(s_inv_root,C_dash_view)
      return E,C




def make_density(int no_of_electrons,double[:,::1] C):
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


def rmsd(double[:,::1] D1,double[:,::1] D2):
      cdef double Sum=0.0
      cdef double delta
      n_basis = D1.shape[0]
      cdef int i,j
      for i in range(n_basis):
          for j in range(n_basis):
               Sum += (D2[i][j]-D1[i][j])**2
      delta =np.sqrt(Sum)

      return delta


cpdef double del_energy(double E1,double E2):
       cdef double Del
       Del= abs(E2 - E1)
       return Del       

def scf_energy(double[:,::1] P,double[:,::1] Hcore,double[:,::1] F):
      N = P.shape[0]
      cdef double Energy=0.0
      cdef Py_ssize_t i,j
      for i in range(N):
        for j in range(N):
            Energy  +=  P[i,j]*(Hcore[i,j]  +  F[i,j])

      return Energy

#def enuc_calculator(atoms,geom):
#       indices = []
#       #stuff = atoms
#       #for L in range(0, len(stuff)+1):
#       for i in range(len(atoms)):
#            indices.append(i)
#       atomic_numbers = atomic_number(atoms)
#       #print(atomic_numbers)
#       combs=[]
#       for subset in itertools.combinations(indices,2):
#            combs.append(subset)


       #print(combs)
#       dist_wrt_combs = []
#       for i in range(len(combs)):
#            dist_wrt_combs.append(find_distance(geom[combs[i][0]],geom[combs[i][1]]))
#
#
 #      print(dist_wrt_combs)
#
 #      enuc=0
  #     for i in range(len(dist_wrt_combs)):
   #           enuc += ((8.9875517923*(10**9))*(atomic_numbers[combs[i][0]]*atomic_numbers[combs[i][1]])*((1.60217662*(10**(-19)))**2))/dist_wrt_combs[i]
    #          #print(atomic_numbers[combs[i][0]])
     #         #print(atomic_numbers[combs[i][1]])
#
 #      return enuc

cpdef double enuc_calculator(list atoms,list geom):
      cdef double E_nuc=0.0
      cdef double R_ab=0.0
      cdef int i,j,Z_a,Z_b 
      for i in range(len(atoms)):
          for j in range(0,i):
               Z_a=no_of_e(atoms[i])
               Z_b=no_of_e(atoms[j])
               R_ab=find_distance(geom[i],geom[j])
               #print(R_ab)
               E_nuc+=(Z_a*Z_b)/R_ab

      return E_nuc


def fprime(double[:,::1] X,double[:,::1] F):
    return np.dot(np.transpose(X),np.dot(F,X))



def  deltae(double E,double OLDE):
    return abs(E-OLDE)





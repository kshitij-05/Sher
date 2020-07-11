#Read the geometry
import numpy as np
from math import sqrt
import itertools
from scipy import linalg as LA
#########################################################
#Function to define no of Electron
#On Entry 
#element--> name of the element
#On Exit
#u--> atomic no
########################################################

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
##########################################
# Calculate the distance between two atoms
#On Entry 
#a--> position of first atom, in list containing x,y and z
#b--> position of second atom
#On return
#R--> distance between two atoms
##########################################################
def find_distance(a,b):
 x_1=float(a[0])
 x_2=float(b[0])
 y_1=float(a[1])
 y_2=float(b[1])
 z_1=float(a[2])
 z_2=float(b[2])
 R_square =(x_1-x_2)**2+(y_1-y_2)**2+(z_1-z_2)**2
 R=sqrt(R_square)
 
 return R
##########################################################
#This function reads the 1 electron Hamiltonian from the disk
#
#On Entry
#file_name--> Name of the file to be read from
#basis--> Dimension of the basis set
#
#On Exit
#A-->Numpy array with the 1 electron Hamiltonian elements(nbasis,nbasis)
##########################################################
def file_read_1e(file_name,nbasis):

 #open the file
 input_file=open(file_name)   #open the file
 #read the file using readline to file_content
 file_content=input_file.readlines()# read the content
 #close the file
 input_file.close() 
 A=np.zeros([nbasis,nbasis])

 for line in file_content:
    V_line=line.rstrip()
    V_line=V_line.split() 
    i=int(V_line[0])-1
    j=int(V_line[1])-1
    A[i][j]=float(V_line[2])
    A[j][i]=float(V_line[2])
 return A
 #########################################################
#This function reads the 2 electron Hamiltonian from the disk
#
#On Entry
#file_name--> Name of the file to be read from
#basis--> Dimension of the basis set
#
#On Exit
#twoe-->Numpy array with the 2 electron integrals (nbasis,nbasis,nbasis,nbasis)
##########################################################
def read_2_e(file,nbasis):

#read the file
 input_file=open(file)   #open the file

 file_content=input_file.readlines()# read the content

 input_file.close()            # close the file

#print(file_content)
 twoe_index=[]
 twoe_value=[]

 for line in file_content:
    V_line=line.rstrip()
    V_line=line.split()
    i=int(V_line[0])
    j=int(V_line[1])# change from Dirac to Muliken Ordering
    k=int(V_line[2])
    l=int(V_line[3])
    ijkl=compound_index(i,j,k,l)
    twoe_index.append(ijkl)
    twoe_value.append(V_line[4])
    
 #define the 4 d array
 twoe=np.zeros([nbasis,nbasis,nbasis,nbasis])

 for i in range(nbasis):
   for j in range(nbasis):
     for k in range(nbasis):
       for l in range(nbasis):
           ijkl=compound_index(i+1,j+1,k+1,l+1)
           if ijkl in twoe_index:
                ind=twoe_index.index(ijkl)
                twoe[i,j,k,l]=float(twoe_value[ind])
                
           
 return twoe
 
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
def compound_index(i,j,k,l):

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

  return ijkl

#########################################################
#This function calculate X=S^-1/2
#
#On input
#S--> overlap matrix dimension (nbasis,nbasis)
#
#On output
#X--> the transformation matrix dimension (nbasis,nbasis)
##########################################################
def get_X(S,nbasis):
   lambda_b,L_s=np.linalg.eig(S)
   X=np.zeros([nbasis,nbasis])
   X_temp=np.zeros([nbasis,nbasis])
   temp=np.zeros([nbasis,nbasis])
   for i in range(nbasis):
       temp[i][i]=(lambda_b[i])**(-0.5)
   X_temp=np.matmul(L_s,temp)
   X=np.matmul(X_temp,L_s.transpose())
   
   return X
    
def no_of_electrons(atoms):
                symbol = ['H','He', 'Li','Be','B','C','N','O','F','Ne','Na','Mg','Al','Si','P','S','Cl','Ar','K', 'Ca', 'Sc',
                 'Ti', 'V', 'Cr', 'Mn', 'Fe','Co', 'Ni', 'Cu', 'Zn','Ga', 'Ge', 'As', 'Se', 'Br', 'Kr','Rb', 'Sr', 'Y', 'Zr',
                 'Nb', 'Mo', 'Tc', 'Ru','Rh', 'Pd', 'Ag', 'Cd','In', 'Sn', 'Sb', 'Te', 'I', 'Xe','Cs', 'Ba', 'La', 'Ce', 'Pr',
                 'Nd', 'Pm', 'Sm',  'Eu','Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu','Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt',
                  'Au', 'Hg','Tl','Pb','Bi','Po','At','Rn']

                sum=0
                for i in range(len(atoms)):
                     sum+= symbol.index(atoms[i])+1

                return sum

def atomic_number(atoms):
            atomic_numbers = []
            symbol = ['H','He', 'Li','Be','B','C','N','O','F','Ne','Na','Mg','Al','Si','P','S','Cl','Ar','K', 'Ca', 'Sc',
                 'Ti', 'V', 'Cr', 'Mn', 'Fe','Co', 'Ni', 'Cu', 'Zn','Ga', 'Ge', 'As', 'Se', 'Br', 'Kr','Rb', 'Sr', 'Y', 'Zr',
                 'Nb', 'Mo', 'Tc', 'Ru','Rh', 'Pd', 'Ag', 'Cd','In', 'Sn', 'Sb', 'Te', 'I', 'Xe','Cs', 'Ba', 'La', 'Ce', 'Pr',
                 'Nd', 'Pm', 'Sm',  'Eu','Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu','Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt',
                  'Au', 'Hg','Tl','Pb','Bi','Po','At','Rn']
            for i in range(len(atoms)):
                 atomic_numbers.append(symbol.index(atoms[i])+1)
            return atomic_numbers

            

def Masscalculator(atoms):
                Atomic_mass_dictionary = {'H':'1.008','He':'4.002','Li':'6.938','Be':'9.012','B':'10.806','C':'12.0096','N':'14.00643','O':'15.99903','F':'18.998','Ne':'20.1797','Na':'22.989',
'Mg':'24.304','Al':'26.981','Si':'28.084','P':'30.973','S':'32.059','Cl':'35.446','Ar':'39.948','K':'39.0983', 'Ca':'40.078', 'Sc':'44.955', 'Ti':'47.867',
'V':'50.9415','Cr':'51.9961', 'Mn':'54.938', 'Fe':'55.845','Co':'58.933', 'Ni':'58.6934', 'Cu':'63.546', 'Zn':'65.38','Ga':'69.723', 'Ge':'72.630',
'As':'74.921', 'Se':'78.971', 'Br':'79.901', 'Kr':'83.798','Rb':'85.4678', 'Sr':'87.62', 'Y':'88.905', 'Zr':'91.224', 'Nb':'92.906', 'Mo':'95.95', 
'Tc':'98', 'Ru':'101.07','Rh':'102.905', 'Pd':'106.42', 'Ag':'107.8682', 'Cd':'112.414','In':'114.818', 'Sn':'118.710', 'Sb':'121.760', 'Te':'127.60', 'I':'126.904'
, 'Xe':'131.293','Cs':'132.905', 'Ba':'137.327', 'La':'138.905 ', 'Ce':'140.116', 'Pr':'140.907 ', 'Nd':'144.242', 'Pm':'145', 'Sm':'150.36',  'Eu':'151.964',
'Gd':'157.25', 'Tb':'158.925', 'Dy':'162.500', 'Ho':'164.930 ', 'Er':'167.259', 'Tm':'168.934 ', 'Yb':'173.054', 'Lu':'174.9668','Hf':'178.49', 'Ta':'180.947 ',
'W':'183.84', 'Re':'186.207', 'Os':'190.23', 'Ir':'192.217', 'Pt':'195.084', 'Au':'196.966 ', 'Hg':'200.592','Tl':'204.382','Pb':'207.2','Bi':'208.980 ','Po':'209','At':'210','Rn':'222'}

                mass=Atomic_mass_dictionary[atoms]
                return mass



def round_up(V):
  V = np.around(V, decimals=10).tolist()
  return V



def sym_matrix(temp):

  for i in range(len(temp)):
        temp[i]=[float(x) for x in temp[i]]

  #print(temp_s)
  mat_lower=np.zeros((n_basis,n_basis))
  for i in range(len(temp)):
         mat_lower[int(temp[i][0])-1][int(temp[i][1])-1]=temp[i][2]

  mat_upper=np.zeros((n_basis,n_basis))
  for i in range(len(mat_lower)):
         for j in range(len(mat_lower)):
             if j>i:
                mat_upper[i][j]=mat_lower[j][i]

    #print(s_mat_upper)
  result=np.zeros((n_basis,n_basis))
  result = [[mat_lower[i][j] + mat_upper[i][j]  for j in range(len(mat_lower))] for i in range(len(mat_lower))]
   
  return result



def make_fock(n_basis,D,hamil,eri):
  Fock=np.zeros((n_basis,n_basis))
  for i in range(n_basis):
     for j in range(n_basis):
        Fock[i][j] = hamil[i][j] 
        for k in range(n_basis):
             for l in range(n_basis):
                 Fock[i][j] += D[k][l]*(  2.0*eri[i][j][k][l] - eri[i][k][j][l]  )
  return Fock



def make_c_0(s_inv_root,Fock):
      fock_ini=np.linalg.multi_dot([np.transpose(s_inv_root),Fock,s_inv_root])
      fock_ini = round_up(fock_ini)
      # As Our Fock Matrix is a real-symmetric Hermitian       
      E_0 ,C_0_dash = LA.eigh(fock_ini)                    
      E_0 = np.around(E_0, decimals=10).tolist()
      C_0=np.matmul(s_inv_root,C_0_dash)
      C_0 = round_up(C_0)
      return C_0




def make_density(n_basis,no_of_electrons,C):
     D=np.zeros((n_basis,n_basis))

     for i in range(len(C)):
         for j in range(len(C)):
            for m in range(int(no_of_electrons/2)):
                D[i][j] += C[i][m]*C[j][m]

     return D



def rmsd(n_basis, D1, D2):
      Sum=0
      delta=0
      for i in range(n_basis):
          for j in range(n_basis):
               Sum += (D2[i][j]-D1[i][j])**2
      delta =np.sqrt(Sum)

      return delta


def del_energy(E1, E2):
       Del= abs(E2 - E1)
       return Del       


def scf_energy(N,D,H,F):
      Energy=0
      for i in range(N):
        for j in range(N):
            Energy  +=  D[i][j]*(H[i][j]  +  F[i][j])

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

def enuc_calculator(atoms,geom):
      E_nuc=0.0
      for i in range(len(atoms)):
          for j in range(0,i):
               Z_a=no_of_e(atoms[i])
               Z_b=no_of_e(atoms[j])
               R_ab=find_distance(geom[i],geom[j])
               #print(R_ab)
               E_nuc+=(Z_a*Z_b)/R_ab

      return E_nuc

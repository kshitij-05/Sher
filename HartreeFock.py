#########################################################################################################################################
######################                                                                                                ###################
######################                         This Program is for Hartree Fock SCF calculaions                       ###################
######################                                                                                                ###################
#########################################################################################################################################



import numpy as np
from help_functions import no_of_e,find_distance,get_X,make_density,rmsd,del_energy,enuc_calculator,atomic_number,scf_energy,rmsd,del_energy,make_density,make_fock,make_c_0,fprime,makedensity,deltae,deltap,currentenergy
from scipy import linalg as LA
import itertools
from integrals_functions import Basis_attributes_finder , E , overlap , S , fact2 , BasisFunction , orbital_config ,S_mat , T_mat , V_mat , Eri_mat
import time

T1 = time.time()

input_x_file = open(input(),'r')
content = input_x_file.readlines()
input_x_file.close()
input_file = []
for line in content:
    v_line=line.strip()
    if len(v_line)>0:
        input_file.append(v_line.split())

#print(input_file)


Type_of_computation = input_file[0][0]
#print(Type_of_computation)

basis_set = input_file[1][0]
#print(basis_set)

no_of_atoms = input_file[2][0]
#print(no_of_atoms)

no_of_electrons = int(input_file[3][0])
#print(no_of_electrons)

for i in range(4):
	input_file.pop(0)
geom_file = input_file
#print(geom_file)

Atoms = []
for i in range(len(geom_file)):
	Atoms.append(geom_file[i][0])

#print(Atoms)
geom_raw = geom_file
for i in range(len(geom_file)):
	geom_raw[i].pop(0)

for i in range(len(geom_raw)):
        geom_raw[i]=[float(x) for x in geom_raw[i]]

#print(geom_raw)


##############################################################
########      Nuclear-Nuclear repulsion energy         #######
##############################################################


#U = kQ1Q2/r12

enuc = enuc_calculator(Atoms,geom_raw)
 
#print(enuc)


###############################################################
############   One electron integrals Computation      ########
###############################################################

Overlap_matrix = S_mat(Atoms,geom_raw)
Overlap_matrix = np.around(Overlap_matrix, decimals=15).tolist() 

nbasis = int(len(Overlap_matrix))

#print(np.array(Overlap_matrix))

Kinetic_energy_matrix = T_mat(Atoms , geom_raw)

#print(np.array(Kinetic_energy_matrix))

Potential_energy_matrix = V_mat(Atoms , geom_raw)

#print(Potential_energy_matrix)

Two_electron_integrals_4d = Eri_mat(Atoms  , geom_raw , Overlap_matrix)

Two_electron_integrals_4d = np.around(Two_electron_integrals_4d, decimals=15).tolist()

#print(Two_electron_integrals_4d)

###########################################################
###              Build Core Hamiltonian                 ###
###########################################################


core_hamil = [[Kinetic_energy_matrix[i][j] + Potential_energy_matrix[i][j]  for j in range(nbasis)] for i in range(nbasis)]
core_hamil = np.real(core_hamil)  
#print(np.array(core_hamil))

##########################################################
#####              First  iteration                   ####
##########################################################


s_inv_root= get_X(Overlap_matrix,nbasis)
fock_ini_prime=np.matmul(core_hamil, s_inv_root)
fock_ini = np.matmul(np.conj(s_inv_root.transpose()),fock_ini_prime)
E_0 ,C_0_dash = LA.eigh(fock_ini)                    
E_0 = np.around(E_0, decimals=15).tolist()
C_0=np.matmul(s_inv_root,C_0_dash)

D_0=np.zeros((nbasis,nbasis))

for i in range(len(C_0)):
    for j in range(len(C_0)):
      for m in range(int(no_of_electrons/2)):
           D_0[i][j] += C_0[i][m]*C_0[j][m]

D_0 = np.around(D_0, decimals=14).tolist() 

#print(np.array(D_0))
Energy_0=  scf_energy(nbasis,D_0,core_hamil,fock_ini) 
#print(Energy_0)

'''
#####################################################################
###           DIIS extrapolation for the SCF procedure            ###
#####################################################################
C= C_0
D= D_0
F= core_hamil
E= Energy_0
delta_energy = 1
rmsd_d = 1

Output_file = []

Fock_matrices = []
Fock_matrices.append(F)

Density_matrices = []
Density_matrices.append(D)
Nerror = 6

for i in range(Nerror):
		new_C = make_c_0(s_inv_root, F)
		new_D = make_density(nbasis, no_of_electrons, new_C)
		new_D = np.around(new_D, decimals=14).tolist() 
		new_F = make_fock(nbasis,new_D,core_hamil,Two_electron_integrals_4d)
		new_E = scf_energy(nbasis,new_D,core_hamil,new_F)
		#print(new_E)
		Fock_matrices.append(new_F)
		Density_matrices.append(new_D)
		delta_energy= del_energy(E,new_E)
		#print(delta_energy)
		rmsd_d = rmsd(nbasis,D,new_D)
		#print(rmsd_d)
		Output_file.append([new_E+enuc , delta_energy , rmsd_d])
		#print(str(new_E)+str('    ')+str(delta_energy)+str('    ')+ str(rmsd_d))
		C = new_C
		D = new_D
		F = new_F
		E = new_E

ErrorSet = []

for i in range(Nerror):
	ei_mat= ((np.dot(Fock_matrices[i],np.dot(Density_matrices[i],Overlap_matrix)) - \
	np.dot(Overlap_matrix,np.dot(Density_matrices[i],Fock_matrices[i]))))
	ErrorSet.append(ei_mat)
	#print(ei_mat)


Bmat = np.zeros((Nerror+1,Nerror+1))
ZeroVec = np.zeros((Nerror+1))
ZeroVec[-1] = -1.0
for a in range(0,Nerror):
	for b in range(0,a+1):
		Bmat[a,b] = Bmat[b,a] = np.trace(np.dot(ErrorSet[a].T,ErrorSet[b]))
		Bmat[a,Nerror] = Bmat[Nerror,a] = -1.0

#print(ErrorSet[0])

#print(Bmat)
coeff = np.linalg.solve(Bmat,ZeroVec)
#print(coeff)

Fock_new = np.zeros((nbasis,nbasis))
for i in range(len(coeff)):
	Fock_new += coeff[i]*Fock_matrices[i]

######################################################################
###                       SFC Procedure                          #####
######################################################################

F= Fock_new
E= Energy_0
delta_energy = 1
rmsd_d = 1

while delta_energy >  0.0000000001 and rmsd_d > 0.0000000001 :
			new_C = make_c_0(s_inv_root, F)
			new_D = make_density(nbasis, no_of_electrons, new_C)
			new_D = np.around(new_D, decimals=14).tolist()
			new_F = make_fock(nbasis,new_D,core_hamil,Two_electron_integrals_4d)
			new_E = scf_energy(nbasis,new_D,core_hamil,new_F)
			#print(new_E)
			delta_energy= del_energy(E,new_E)
			#print(delta_energy)
			rmsd_d = rmsd(nbasis,D,new_D)
			#print(rmsd_d)
			Output_file.append([new_E+enuc , delta_energy , rmsd_d])
			#print(str(new_E)+str('    ')+str(delta_energy)+str('    ')+ str(rmsd_d))
			C = new_C
			D = new_D
			F = new_F
			E = new_E




Total_HF_energy = E + enuc

print(np.array(Output_file))

print('Total Hartree Fock Energy is {} Hartrees'.format(Total_HF_energy))

'''




##   Trial SCF procedure with DIIS ###

C= C_0
D= D_0
F= core_hamil
E= Energy_0
delta_energy = 1
rmsd_d = 1
Fock_matrices = []
ErrorSet = []
OLDE = 0.0 
Nerror = 6
convergence = 10**(-15)



for i in range(0 , 100):
	Fock = make_fock(nbasis,D,core_hamil,Two_electron_integrals_4d)
	if i > 0 :
		error = ((np.dot(Fock,np.dot(D,Overlap_matrix)) - np.dot(Overlap_matrix,np.dot(D,Fock))))
		if len(ErrorSet) < Nerror :
			Fock_matrices.append(Fock)
			ErrorSet.append(error)

		elif len(ErrorSet) >= Nerror:
			del Fock_matrices[0]
			del ErrorSet[0]
			Fock_matrices.append(Fock) 
			ErrorSet.append(error)

	NErr = len(ErrorSet)
	if NErr >= 2:
			Bmat = np.zeros((NErr+1,NErr+1))
			ZeroVec = np.zeros((NErr+1))
			ZeroVec[-1] = -1.0
			for a in range(0,NErr):
				for b in range(0,a+1):
					Bmat[a,b] = Bmat[b,a] = np.trace(np.dot(ErrorSet[a].T,ErrorSet[b]))
					Bmat[a,NErr] = Bmat[NErr,a] = -1.0

			try:
				coeff = np.linalg.solve(Bmat,ZeroVec)
			except np.linalg.linalg.LinAlgError as err:
				if 'Singular matrix' in err.message:
					print ('\tSingular B matrix, turing off DIIS')
					
			else:
				Fock = 0.0
				for i in range(0,len(coeff)-1):
					Fock += coeff[i]*Fock_matrices[i]

	Fprime = fprime(s_inv_root,Fock)
	E,Cprime = np.linalg.eigh(Fprime)
	C = np.dot(s_inv_root,Cprime)
	OLDP = D
	D = make_density(nbasis,no_of_electrons,C)
	DELTA = deltae(currentenergy(D,core_hamil,Fock,nbasis),OLDE)
	print ("Current Energy: {0:.12f}".format(currentenergy(D,core_hamil,Fock,nbasis)+enuc),"Hartree",'\t',"Delta: {0:.12f}".format(DELTA)	)
	OLDE = currentenergy(D,core_hamil,Fock,nbasis)
	if DELTA < convergence:
		print ("NUMBER ITERATIONS: ",j)
		break
	if j == 119:
		print ("SCF not converged!")
		break			
	EN = currentenergy(D,core_hamil,Fock,nbasis)




T2 = time.time()
print('Time taken for convergence is {} seconds'.format(T2 - T1))

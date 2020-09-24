import numpy as np
import time
import sys
from utils.utils import no_of_electrons
from utils.enuc import enuc_calculator
from utils.basisfunctions1 import orbital_config
from one_integrals.integral_utils import S_mat , T_mat ,V_mat
from two_integrals.two_integrals_utils import Eri_mat
from scf.scf import scf_iteration
#from one_integrals.integral_utils import S

inFile = sys.argv[1]
with open(inFile,'r') as i:
	content = i.readlines()
input_file =[]
for line in content:
	v_line=line.strip()
	if len(v_line)>0:
		input_file.append(v_line.split())

#print(input_file)
'''
print("\t#------------------------------------------------------#\n",
      "\t#                                                      #\n",
      "\t#               New Prototype for HF SCF               #\n",
      "\t#                                                      #\n",
      "\t#------------------------------------------------------#\n")'''

Level_of_theory = input_file[0][0]

basis_set = input_file[0][1]

charge, multiplicity = input_file[1]

#print(Level_of_theory , basis_set ,charge , multiplicity)


for i in range(2):
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

#print(Atoms)

geom = np.array(geom_raw)


def scf_hf(Atoms , geom):
	no_of_e , atomic_nos = no_of_electrons(Atoms)
	for i in range(len(atomic_nos)):
		atomic_nos[i]=float(atomic_nos[i])
	atomic_nos = np.array(atomic_nos)

	#---------------------------------------
	#     ENUC
	#--------------------------------------

	t1 = time.time()
	enuc = enuc_calculator(atomic_nos,geom)
	t2 = time.time()
	print(t2-t1)
	#----------------------------------------
	#    Orbitals
	#----------------------------------------
	exps,coefs,origins,shells,norms = orbital_config(Atoms , geom_raw)
	#print(exps)
	#-----------------------------------------
	#       Overlap integrals
	#-----------------------------------------
	t1 = time.time()
	Overlap_mat = S_mat(exps,coefs,origins,shells,norms)
	nbasis = Overlap_mat.shape[0]
	t2 = time.time()
	print('{} seconds for overlap integrals'.format(t2-t1))
	#print(Overlap_mat)

	#----------------------------------------
	#     Kinetic integrals
	#----------------------------------------
	t1 = time.time()
	Kinetic_mat = T_mat(exps,coefs,origins,shells,norms)
	t2 = time.time()
	print('{} seconds for kinetic integrals'.format(t2-t1))
	#print(Kinetic_mat)

	#---------------------------------------
	#   Coulomb integrals
	#---------------------------------------
	#print(atomic_nos.shape[0])
	#print(R(0,0,1,0,1.2322342343,2.23234234523,3.344324234,1.1231432423,2.3234224234))
	t1 = time.time()
	Potential_mat = V_mat(exps,coefs,origins,shells,norms,atomic_nos,geom)
	t2 = time.time()
	print('{} seconds for coulomb integrals'.format(t2-t1))
	#print(Potential_mat)
	'''for i in range(7):
		for j in range(7):
			if Potential_mat[i,j]>0.0000000001 or Potential_mat[i,j]<-0.0000000001:
				print(Potential_mat[i,j])'''
	core_hamil = [[Kinetic_mat[i][j] + Potential_mat[i][j]  for j in range(nbasis)] for i in range(nbasis)]
	core_hamil = np.real(core_hamil)
	#print(core_hamil)
	#----------------------------------------
	#  Two elec integrals 
	#----------------------------------------
	t1 = time.time()
	twoe,eri = Eri_mat(exps,coefs,origins,shells,norms)
	t2 = time.time()
	print('{} seconds for two elec integrals'.format(t2-t1))



	EN, E, C, P, F = scf_iteration(10**(-12),enuc,no_of_e,nbasis,Overlap_mat,core_hamil,twoe,False,True)

	#print(twoe)



t1 = time.time()
SCF = scf_hf(Atoms,geom)
t2 = time.time()
print('Total time :',t2-t1)
import numpy as np
from sher.scf.scf_helper import make_density,make_fock,get_X,deltae,scf_energy,make_C,rmsd,round_up

def scf_iteration(convergence,ENUC,Nelec,dim,S,Hcore,twoe,printops,do_DIIS):

	S_minhalf = get_X(S)
	#if do_DIIS == True:
		#print("DIIS acceleration ON")
	#elif do_DIIS == False:
	#	print ("DIIS acceleration OFF")
	P = np.zeros((dim,dim)) # P is density matrix, set intially to zero.
	OLDE = 0.0 
	G = np.zeros((dim,dim)) # The G matrix is used to make the Fock matrix
	num_e = 6
	ErrorSet = []
	FockSet = []
	for j in range(0,120):
		F = make_fock(P,Hcore,twoe)
		if do_DIIS == True:
			if j > 0:
				error = ((np.dot(F,np.dot(P,S)) - np.dot(S,np.dot(P,F))))
				if len(ErrorSet) < num_e:
					FockSet.append(F)
					ErrorSet.append(error)
				elif len(ErrorSet) >= num_e:
					del FockSet[0]
					del ErrorSet[0]
					FockSet.append(F) 
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
						do_DIIS = False
				else:
					F = 0.0
					for i in range(0,len(coeff)-1):
						F += coeff[i]*FockSet[i]

		E,C = make_C(S_minhalf,F)        # C back to AO basis
		#D = make_density(nbasis,no_of_electrons,C)
		OLDP = P
		P = make_density(Nelec,C)
		# test for convergence. if meets criteria, exit loop and calculate properties of interest
		rmsd_ = rmsd(OLDP , P)
		DELTA = deltae(scf_energy(P,Hcore,F),OLDE)
		if printops == True:
			print ("E: {0:.15f}".format(round_up(scf_energy(P,Hcore,F)+ENUC)),"a.u.",'\t',"Del E: {0:.15f}".format(round_up(DELTA)),'\t',"RMSD :{0:.15f}".format(round_up(rmsd_)))
		OLDE = scf_energy(P,Hcore,F)
		if rmsd_ < convergence:
			#print ("NUMBER ITERATIONS: ",j , '\n')
			break
		if j == 119:
			#print ("SCF not converged!")
			break
		EN = scf_energy(P,Hcore,F)
	print ("TOTAL E(SCF) = ", EN+ENUC)
	#print "C = \n", C
	return EN, E, C, P, F
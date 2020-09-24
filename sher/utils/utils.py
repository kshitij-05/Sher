

def no_of_electrons(atoms):
	symbol = ['H','He', 'Li','Be','B','C','N','O','F','Ne','Na','Mg','Al','Si','P','S','Cl','Ar','K', 'Ca', 'Sc',
	 'Ti', 'V', 'Cr', 'Mn', 'Fe','Co', 'Ni', 'Cu', 'Zn','Ga', 'Ge', 'As', 'Se', 'Br', 'Kr','Rb', 'Sr', 'Y', 'Zr',
	 'Nb', 'Mo', 'Tc', 'Ru','Rh', 'Pd', 'Ag', 'Cd','In', 'Sn', 'Sb', 'Te', 'I', 'Xe','Cs', 'Ba', 'La', 'Ce', 'Pr',
	 'Nd', 'Pm', 'Sm',  'Eu','Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu','Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt',
	 'Au', 'Hg','Tl','Pb','Bi','Po','At','Rn']
	
	mole_elec=0
	atomic_nos =[]
	for i in range(len(atoms)):
		mole_elec+= symbol.index(atoms[i])+1
		atomic_nos.append(symbol.index(atoms[i])+1)

	return mole_elec, atomic_nos
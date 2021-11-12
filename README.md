# Sher

This is a program that computes one and two electron integrals over gaussian functions.
It can also perform Hartree fock and MP2 energy calculation

## Installation

Step 1. Download Sher :  $ git clone https://github.com/kshitij-05/Sher.git  \
Step 2. Compile Cython files,  \
          To compile go to directories one,two_integrals,utils and scf and run $ python3 setup.py build_ext --inplace\
Step 3. Run test.\
          To run a test job run $ python3 sher.py h2o.inp in terminal  \

### Example input file

h2o.inp\
hf cc-pvdz\
0 1\
O      0.000000000000  -0.143225816552   0.000000000000\
H      1.638036840407   1.136548822547  -0.000000000000\
H      -1.638036840407   1.136548822547  -0.00000000000

### Example output

0.000378600002 seconds taken for overlap integrals calculation\
0.000315600002 seconds taken for kinetic integrals calculation\
0.001334799999 seconds taken for coulomb integrals calculation\
0.026611800000 seconds for two elec integrals calculation\
E: -117.839710793896586 a.u.     Del E: 125.842077855707359      RMSD :2.550261117524990\
E: -70.284216167831161 a.u.      Del E: 47.555494626065425       RMSD :1.826673131222464\
E: -74.576672702967613 a.u.      Del E: 4.292456535136452        RMSD :0.403889826715792\
E: -75.105709829987916 a.u.      Del E: 0.529037127020302        RMSD :0.088003718396659\
E: -74.954655961996622 a.u.      Del E: 0.151053867991294        RMSD :0.020519849404759\
E: -74.938944421788975 a.u.      Del E: 0.015711540207647        RMSD :0.012108639682051\
E: -74.942105960574722 a.u.      Del E: 0.003161538785747        RMSD :0.000460862236915\
E: -74.942079991069633 a.u.      Del E: 0.000025969505089        RMSD :0.000001081074359\
E: -74.942079954715680 a.u.      Del E: 0.000000036353953        RMSD :0.000000052668297\
E: -74.942079954042612 a.u.      Del E: 0.000000000673069        RMSD :0.000000000336078\
E: -74.942079954042612 a.u.      Del E: 0.000000000000000        RMSD :0.000000000000080\
TOTAL E(SCF) =  -74.94207995404261  Hartrees\
0.006673700002 seconds for scf procedure\
Total time taken : 0.037254000002576504


# SingleOrigin

A Python module for crystallographic analysis of S/TEM data.
Main module is for atomic column position finding intended for structural analysis of high resolution scanning transmission electron microscope images of crystalline lattices. The module uses 2D Gaussian fitting and automatically accounts for intensity overlap between closely spaced columns based on image morphology. Atom columns are initially found by registering a projected reference lattice to the image based on a CIF of the the imaged structure (or a similar one).

Incorporates a number of different methods to visualize the atom column positions:
1) Diretly plotting the fitted positions.
2) Plotting displacement vectors from the reference to fitted positions.
3) Calcualting vector pair correlation functions (vPCFs) for various sublattices to see the distributions of projected bond vectors in real space. Analagous to a pair distribution function, as obtained from X-ray or neutron "total scattering" experiments, but retaining the orientation information available in the image. See the original vPCF paper in APL Materials: https://aip.scitation.org/doi/10.1063/5.0058928
4) Plot inter-atom column distances (or distance deviations from the reference lattice) corresponding to a vector (or vectors) in a vPCF.

Secondary module(s) for diffraction and 4D STEM analysis is under development. Currently operational methods are:
-Strain mapping is implimented in the ReciprocalLattice class.
-Reciprocal lattice measurements from single diffraction patterns 4D STEM datasets or HRSTEM FFTs can be made using the ReciprocalLattice class. Superlattice may also be measured.

Installation:
It is recommended to install by pip in a clean Python environment (https://docs.python.org/3/library/venv.html). SingleOrigin may be used in Spyder or Jupyter notebooks/lab. To install, activate your environment and run the following in a command line prompt if you downloaded the tarball:

pip install "pathtofile/SingleOrigin-2.5.3.tar.gz"

OR from PyPI:

pip install SingleOrigin

See example scripts and Jupyter-lab notebooks at: https://github.com/sdfunni/SingleOrigin 

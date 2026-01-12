# SingleOrigin

A Python module for analysis of multiple S/TEM modalities: atomic resolution images, 4D STEM and EELS.

## **<ins>Features incorporated into all modules</ins>**:

### - <ins>Numpy / Pandas-based data strctures</ins>. Easy results extraction for:
- Subsequent custom analysis
- Custom plotting/figure building

### - <ins>Streamlined</ins>. Class for each processing pipline / data type: 
- Stores data and results (as class attributes)
- Built-in processing methods (as class methods)
- Reduced user inputs/programming because methods already know where to find most required information

### - <ins>Fast</ins>:
- Efficient coding
- Parallel CPU processing for major iterative tasks (e.g. fitting 1000s of atom columns)

## **<ins>Module descriptions</ins>**:

**<ins>Atomic resolution image analysis module</ins>** intended for structural analysis of high resolution scanning transmission electron microscope images of crystalline lattices. The module uses 2D Gaussian fitting and automatically accounts for intensity overlap between closely spaced columns based on image morphology. Atom columns are initially found by registering a projected reference lattice to the image based on a CIF of the imaged structure (or a similar one).

Incorporates a number of different methods to visualize the atom column positions:
- Directly plotting the fitted positions.
- Plotting displacement vectors from the reference to fitted positions.
- Calculating vector pair correlation functions (vPCFs) for various sublattices to see the distributions of projected bond vectors in real space. Analogous to a pair distribution function, as obtained from X-ray or neutron "total scattering" experiments, but retaining the orientation information available in the image. See the original vPCF paper in APL Materials: https://aip.scitation.org/doi/10.1063/5.0058928
- Plot inter-atom column distances (or distance deviations from the reference lattice) corresponding to a vector (or vectors) in a vPCF.

**<ins>Diffraction / 4D STEM analysis module</ins>.** Currently operational methods are:
- Basic virtual detectors and other utility functions.
- Strain mapping is implemented in the ReciprocalLattice class (direct or cepstrum).
- Imaginary exit-wave cepstrum for polarization measurement from diffraction data.
- Reciprocal lattice measurements from single diffraction patterns, 4D STEM datasets or HRSTEM FFTs can be made using the ReciprocalLattice class. Superlattice may also be measured.

**<ins>EELS module</ins>.** Manipulation and quantification of EELS, especially EELS spectrum images. *Quantitative elemental analysis* using model fitting.

## **Installation:**

It is recommended to install by pip in a clean Python environment (https://docs.python.org/3/library/venv.html). SingleOrigin may be used in Spyder or Jupyter notebooks/lab. To install, activate your environment and run the following in a command line prompt if you downloaded the tarball:

pip install "pathtofile/SingleOrigin-3.0.0.tar.gz"

OR build from Github:

pip install git+https://github.com/sdfunni/SingleOrigin.git

OR from PyPI:

pip install SingleOrigin

See example scripts and Jupyter-lab notebooks at: https://github.com/sdfunni/SingleOrigin 

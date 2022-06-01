# SingleOrigin

This is a package for atomic column position finding intended for high resolution scanning transmission electron microscope images.

Work flow:

UnitCell(cif_path) : Import a .cif file of the structure and create a unit cell structural projection for the applicable zone axis.

Apply methods UnitCell.transform_basis(a2, a3, za) and UnitCell.project_uc_2d(proj_axis = 0) to create zone axis projection.

import_image("image_path") : Load image (.tif, .png, Velox, or Gatan DM formats supported by built-in function). 

AtomicColumnLattice(image, UnitCell, resolution=0.8) : initializes AtomicColumnLattice class object (acl). This will contain the image, crystallographic data & atom column position data. Class methods are used for fitting atom columns, plotting functions, etc. "resolution" argument is the probe width in pm. Used for calculating filter values during "fit_atom_columns" step.

acl.fft_get_basis_vect(a1_order=1, a2_order=1, sigma=2) : User picks basis vectors from FFT to scale and orient reference lattice. Ensure correct orientation relative to image (especially for polar structures). Basis vector order values (a1_order, a2_order)  are the FFT spot order picked, i.e. may be 2 or 4 (for spots such as 002 or 440) if missing reflections due to the space group.

acl.define_reference_lattice() : user picks appropriate atom column to register lattice to image.

acl.fit_atom_columns(local_thresh_factor=0.95, diff_filter='auto', grouping_filter='auto) : fits the atom columns. "diff_filter" applys a Laplacian of Gaussian filterto differentiate close columns for masking. "grouping_filter" applys a Gaussian blurring filter for grouping masked regions so that close columns are fit simultaneously. Both methods are set to "auto" by default and the algorithm calculates filter values based on the estimated pixel size and specified resoultion. Fitting data saved in a Pandas DataFrame under the class attribute "at_cols".

acl.refine_reference_lattice() : refines the reference lattice based on the located positions. Prints reference lattice distortions to the console.

See https://github.com/sdfunni/SingleOrigin for example images and scripts.


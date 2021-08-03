# SingleOrigin

(Caffenated) Atomic column position finding for high resolution scanning transmission electron microscope images.

Work flow:
1) Create a unit cell structural projection for a specified zone axis based on data imported from a .cif. Automatically merges atoms with the same projected position into one column. Optional function to merge columns at a specified proximity cutt-off.
 
2) User picks reciprocal basis vectors from the FFT of the image. Function transforms these into rough image basis vectors.

3) User picks an origin atom column from the image. If this column type is not at the (0,0) position of the projected unit cell, an origin offset must be specified in crystallographic fractional coordinates. From this position, and previously calculated basis vectors, a reference lattice is generated. Watershed segmentation is used to find and mask peaks in the image. The closest peak to each atomic column in the reference lattice is fitted with a 2D Gaussian function. All data is stored in a Pandas DataFrame.

4) Option to refine the reference lattice based based on some or a subset of the fitted positions (e.g. refine based on a specific sub lattice.)

***See code comments for more details.

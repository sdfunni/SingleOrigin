"""SingleOrigin is a module for atomic column position finding intended for
    high resolution scanning transmission electron microscope images.
    Copyright (C) 2023  Stephen D. Funni

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see https://www.gnu.org/licenses"""


import os
import copy
import warnings
import io
import json
from contextlib import redirect_stdout

import numpy as np
from numpy.linalg import norm

import pandas as pd

from matplotlib import pyplot as plt

from scipy.signal import convolve2d
from scipy.optimize import minimize
from scipy.ndimage.morphology import (
    binary_fill_holes,
    binary_erosion,
    binary_dilation,
)
from scipy.ndimage import (
    label,
    find_objects,
    map_coordinates,
    gaussian_filter,
    gaussian_laplace,
    maximum_filter,
)

from PyQt5.QtWidgets import QFileDialog as qfd

import imageio

import hyperspy.api as hs

from ncempy.io.dm import dmReader
from ncempy.io.ser import serReader
from ncempy.io.emdVelox import fileEMDVelox
from ncempy.io.emd import emdReader

from skimage.segmentation import watershed
# from cv2 import watershed
from skimage.measure import (moments, moments_central)

from tifffile import imwrite
import time

# %%


"""Error message string(s)"""
no_mask_error = (
    "Float division by zero during moment estimation. No image intensity "
    + "present to calculate moment. If raised during Guassian fitting of atom "
    + "columns, this means that no pixel region was found for fitting of at "
    + "least one atom column. This situation may result from: \n"
    + "1) Too high a 'local_thresh_factor' value resulting in no pixels "
    + "remaining for some atom columns. Check 'self.fit_masks' to see if some "
    + "atom columns do not have mask regions. \n "
    + "2) Due to mask regions splitting an atom column as a result of too "
    + "small or too large a value used for 'grouping_filter' or 'diff_filter'."
    + " Check 'self.group_masks' and 'self.fit_masks' to see if this may be "
    + "occuring. Too small a 'fitting_filter' value may cause noise peaks to "
    + "be detected, splitting an atom column. Too large a value for either "
    + "filter may cause low intensity peaks to be ignored by the watershed "
    + "algorithm. Masks may  be checked with the 'show_masks()' method. \n"
    + "3) Reference lattice may extend to regions of the image without "
    + "detectable atom columns. Use or alter an roi_mask to restrict the "
    + "reference lattice to avoid these parts of the image."
)
# %%
"""General Crystallographic computations"""


def metric_tensor(a, b, c, alpha, beta, gamma):
    """Calculate the metric tensor for a lattice.

    Parameters
    ----------
    a, b, c : ints or floats
        Basis vector magnitudes

    alpha, beta, gamma : floats
        Basis vector angles in degrees

    Returns
    -------
    g : 3x3 ndarray
        The metric tensor

    """

    [alpha, beta, gamma] = np.radians([alpha, beta, gamma])

    g = np.array(
        [[a**2, a*b*np.cos(gamma), a*c*np.cos(beta)],
         [b*a*np.cos(gamma), b**2, b*c*np.cos(alpha)],
         [c*a*np.cos(beta), c*b*np.cos(alpha), c**2]]
    )

    g[abs(g) <= 1e-10] = 0

    return g


def bond_length(at1, at2, g):
    """Calculate distance between two lattice points.

    Parameters
    ----------
    at1, at2 : array_like of shape (1,3) or (3,)
        Lattice point vectors in fractional coordinates

    g : 3x3 ndarray
        The metric tensor

    Returns
    -------
    d : float
        Distance between the points in real units

    """

    at1 = np.array(at1)
    at2 = np.array(at2)
    at12 = at2 - at1
    d = np.sqrt(at12 @ g @ at12.T).item()

    return d


def bond_angle(pos1, pos2, pos3, g):
    """Calculate the angle between two bond two interatomic bonds using the
        dot product. The vertex is at the second atomic position.

    Parameters
    ----------
    pos1, pos2, pos3: array_like of shape (1,3n or (n,)
        Three atomic positions in fractional coordiantes. "pos2" is the
        vertex of the angle; "pos1" and "pos3" are the end points of the
        bonds.

    g : nxn ndarray
        The metric tensor. If atomic positions are in real Cartesian
        coordinates, give the identitiy matrix.

    Returns
    -------
    theta : float
        Angle in degrees

    """

    vec1 = np.array(pos1) - np.array(pos2)
    vec2 = np.array(pos3) - np.array(pos2)
    p_q = np.array([vec1, vec2])

    [[pp, pq], [qp, qq]] = np.array(p_q @ g @ p_q.T)
    theta = np.degrees(np.arccos(pq/(pp**0.5 * qq**0.5)))

    return theta


def absolute_angle_bt_vectors(vec1, vec2, g):
    """Calculate the angle between two vectors using the dot product.

    Parameters
    ----------
    vec1, vec2, : array_like of shape (1,n) or (n,)
        The two vectors.

    g : nxn ndarray
        The metric tensor. If vectors are in real Cartesian
        coordinates, give the identitiy matrix.

    Returns
    -------
    theta : float
        Angle in degrees

    """

    p_q = np.array([vec1, vec2])

    [[pp, pq], [qp, qq]] = np.array(p_q @ g @ p_q.T)
    theta = np.degrees(np.arccos(pq/(pp**0.5 * qq**0.5)))

    return theta


def rotation_angle_bt_vectors(vec1, vec2, trans_mat=None):
    """Calculate the rotation angle from one vector to a second vector. This
        results in a rotation angle with sign giving the rotation direction
        assuming a right-handed system. Note that the order of the vectors
        matters. For rotation in image coordinates, sign of the result should
        be flipped due to the fact that "y" coordinates increases going "down"
        the image.

    Parameters
    ----------
    vec1, vec2, : array_like of shape (1,2) or (2,)
        The two vectors.

    trans_mat : 2x2 ndarray
        The transformation matrix from the vector coordinates to
        Cartesian coordinates, if not already in a Cartesian system. If equal
        to None, a Cartesian system is assumed.
        Default: None

    Returns
    -------
    theta : float
        Rotation angle in degrees.

    """

    if trans_mat is not None:
        vec1 = trans_mat @ vec1
        vec2 = trans_mat @ vec2

    theta = np.degrees(np.arctan2(vec2[1], vec2[0])
                       - np.arctan2(vec1[1], vec1[0]))

    return theta


def IntPlSpc(hkl, g):
    """Calculate the spacing of a set of lattice planes

    Parameters
    ----------
   hkl : array_like of ints of shape (1,3) or (3,)
        Miller indices of the lattice plane

    g : 3x3 ndarray
        The metric tensor

    Returns
    -------
    d_hkl : float
        Inter-planar spacing

    """

    hkl = np.array(hkl)
    d_hkl = (hkl @ np.linalg.inv(g) @ hkl.T)**-0.5

    return d_hkl


def IntPlAng(hkl_1, hkl_2, g):
    """Calculate the spacing of a set of lattice planes

    Parameters
    ----------
    hkl_1 ,hkl_2 : array_like of ints of shape (1,3) or (3,)
        Miller indices of the lattice planes

    g : 3x3 ndarray
        The metric tensor

    Returns
    -------
    theta : float
        Inter-planar angle

    """

    p_q = np.array([hkl_1, hkl_2])
    [[pp, pq], [qp, qq]] = np.array(p_q @ np.linalg.inv(g) @ p_q.T)

    theta = np.degrees(np.arccos(
        np.round(pq/(pp**0.5 * qq**0.5),
                 decimals=10)
    ))

    return theta


def TwoTheta(hkl, g, wavelength):
    """Calculate two theta

    Parameters
    ----------
    hkl : array_like of ints of shape (1,3) or (3,)
         Miller indices of the lattice plane

    g : 3x3 ndarray
        The metric tensor

    wavelength : wavelength of the incident radiation in meters

    Returns
    -------
    two_theta : float
        Bragg diffraciton angle

    """

    hkl = np.array(hkl)
    d_hkl = IntPlSpc(hkl, g) * 1e-10
    two_theta = np.degrees(2 * np.arcsin(wavelength / (2 * d_hkl)))

    return two_theta


def elec_wavelength(V=200e3):
    """Electron wavelength as a function of accelerating voltage

    Parameters
    ----------
    V : int or float
         Accelerating voltage

    Returns
    -------
    wavelength : float
        Electron wavelength in meters

    """

    m_e = 9.109e-31  # electron mass (kg)
    e = 1.602e-19  # elementary charge (C)
    c = 2.997e8  # speed of light (m/s)
    h = 6.626e-34  # Plank's constant (Nms)

    wavelength = h/(2*m_e*e*V*(1+e*V/(2*m_e*c**2)))**.5

    return wavelength


# %%
"""Directory and file functions"""


def select_folder(path=None):
    """Select a folder in dialog box and return path

    Parameters
    ----------
    path : str or None
        The path to the desired folder. This option is available so this
        function can be retained in a script when the user does not always
        want to deal with clicking in a dialog box. If the path is passed
        in the function call the dialog box will not open, but the path
        can easily be commented out to activate the interactive functionality.

    Returns
    -------
    path : str
        The path to the selected folder.

    """
    if not path:
        print('Select folder')
        path = qfd.getExistingDirectory()

    return path


# %%
"""General image functions"""


def load_image(
        path=None,
        display_image=True,
        images_from_stack='all',
        load_dset=0,
        return_path=False,
        norm_image=True,
        full_metadata=False,
):
    """Select image from 'Open File' dialog box, import and (optionally) plot

    Parameters
    ----------
    path : str or None
        The location of the image to load or the path to the folder containing
        the desired image. If only a directory is given, the "Open file"
        dialog box will still open allowing you to select an image file.

    display_image : bool
        If True, plots image (or first image if a series is imported).
        Default: True

    images_from_stack : None or 'all' or int or list-like
        If file at path contains a stack of images, this argument controls
        importing some or all of the images.
        Default: None: import only the first image of the stack.
        'all' : import all images as a 3d numpy array.

    load_dset : str, or int, or list of strings
        If more than one dataset in the file, the title of the desired
        dataset (for Velox .emd files) or the dataset number (for all other
        filetypes). For .emd files with multiple datasets, passing a list will
        result in all matching datasets being returned in a dictionary with
        "name" : numpy.ndarray as the key : value pairs.

    full_metadata : bool
        For emd files ONLY, whether to load the entire metadata as nested
        dictionaries using JSON. If False, loads standard metadata using
        ncempy reader (including pixel size). If True, all metadata available
        in the file is loaded. It is a lot of metadata!
        Default: False

    Returns
    -------
    image : ndarray
        The imported image

    metadata : dict
        The metadata available in the original file

    """

    if path is None:
        path, _ = qfd.getOpenFileName(
            caption='Select an image to load...',
            filter="Images (*.png *.jpg *.tif *.dm4 *.dm3 *.emd *.ser)"
        )

        print(f'path to imported image: {path}')

    elif path[-4:] in ['.dm4', '.dm3', '.emd', '.ser', '.tif', '.png', '.jpg']:
        pass

    else:
        path, _ = qfd.getOpenFileName(
            caption='Select an image to load...',
            directory=path,
            filter="Images (*.png *.jpg *.tif *.dm4 *.dm3 *.emd *.ser)"
        )

    if ((type(load_dset) == int) | (type(load_dset) == str)) & (load_dset != 'all'):
        load_dset = [load_dset]

    if (type(load_dset) == list) & (len(load_dset) > 1) & (path[-3:] != 'emd'):
        raise Exception(
            'Loading multiple datasets is only implimented for .emd files. '
            + 'Specify a single dataset for this file or pass None to load '
            + 'the default dataset.'
        )

    if path[-3:] in ['dm4', 'dm3']:

        dm_file = dmReader(path)
        images = {'image': dm_file['data']}
        metadata = {'image':
                    {key: val for key, val in dm_file.items() if key != 'data'}
                    }

    elif path[-3:] == 'emd':
        # Load emd files using hyperspy (extracts dataset names for files with
        # multiple datasets)

        emd = hs.load(path)
        if type(emd) != list:
            dsets = [emd.metadata.General.title]
            # images = {dsets[0]: np.array(emd)}
            # print('Loaded only dataset: ', dsets[0])
            # dset_ind = 0
        else:
            dsets = [emd[i].metadata.General.title for i in range(len(emd))]
            print('Datasets found: ', ', '.join(dsets))

            # Get list of all datasets to load if "all" specified
            if load_dset == 'all':
                load_dset = dsets

            # Otherwise get list of requested datasets that are in the file
            else:
                num_requested = len(load_dset)
                load_dset = [dset_ for dset_ in load_dset
                             if dset_ in dsets]
                num_avail = len(load_dset)

                if len(load_dset) == 0:
                    raise Exception('No matching dataset(s) found to load.')
                elif num_requested > num_avail:
                    print(
                        'Some datasets not available. Loading: ',
                        *load_dset
                    )

        images = {}
        metadata = {}
        for dset_ in load_dset:
            dset_ind = np.argwhere(np.array(dsets) == dset_).item()
            images[dset_] = np.array(emd[dset_ind])

            # Change DPC vector images from complex type to an image stack
            if images[dset_].dtype == 'complex64':
                images[dset_] = np.stack([np.real(images[dset_]),
                                          np.imag(images[dset_])])

            # Get metadata using ncempy (because it loads more informative
            # metadata and allows loading everyting using JSON)
            try:
                trap = io.StringIO()
                with redirect_stdout(trap):  # To suppress printing
                    emd_file = emdReader(path, dsetNum=dset_ind)

                # image = emd_file['data']
                metadata[dset_] = {
                    key: val for key, val in emd_file.items() if key != 'data'
                }

            except IndexError as ie:
                raise ie
            except TypeError:

                try:
                    # Need to remove EDS datasets from the list and get the
                    # correct index as spectra are not seen by ncempy functions
                    dset_label = dsets[dset_ind]
                    dsets = [i for i in dsets if i != 'EDS']
                    dset_ind = np.argwhere(
                        np.array(dsets) == dset_label).item()

                    emd_vel = fileEMDVelox(path)
                    if full_metadata is False:
                        _, metadata[dset_] = emd_vel.get_dataset(dset_ind)

                    elif full_metadata is True:
                        group = emd_vel.list_data[dset_ind]
                        tempMetaData = group['Metadata'][:, 0]
                        validMetaDataIndex = np.where(tempMetaData > 0)
                        metaData = tempMetaData[validMetaDataIndex].tobytes()
                        # Interpret as UTF-8 encoded characters, load as JSON
                        metadata[dset_] = json.loads(
                            metaData.decode('utf-8', 'ignore')
                        )

                except IndexError as ie:
                    raise ie
                except:
                    raise Exception('Unknown file type.')

        metadata['imageType'] = dsets[dset_ind]

    elif path[-3:] == 'ser':
        ser_file = serReader(path, dsetNum=load_dset)
        images = {load_dset: ser_file['data']}
        metadata = {load_dset: {
            key: val for key, val in ser_file.items() if key != 'data'
        }}

    else:
        images = {'image': imageio.volread(path)}
        metadata = {'image': images['image'].meta}

    for key in images.keys():
        h, w = images[key].shape[-2:]

        if images_from_stack == 'all':
            pass
        elif (type(images_from_stack) == list
              or type(images_from_stack) == int):
            images[key] = images[key][images_from_stack, :, :]
        else:
            raise Exception('"images_from_stack" must be "all", an int, or '
                            + 'a list of ints.')

        # Norm the image(s)
        if norm_image:
            images[key] = image_norm(images[key])

        # Make image dimensions even length
        if len(images[key].shape) == 2:
            images[key] = images[key][:int((h//2)*2), :int((w//2)*2)]
            image_ = images[key]

        if len(images[key].shape) == 3:
            images[key] = images[key][:, :int((h//2)*2), :int((w//2)*2)]
            image_ = images[key][0, :, :]

        if display_image is True:
            fig, axs = plt.subplots()
            axs.imshow(image_, cmap='gray')
            axs.set_xticks([])
            axs.set_yticks([])

    # If only one image, extract from dictionaries
    if len(images) == 1:
        key = list(images.keys())[0]
        images = images[key]
        metadata = metadata[key]

    if return_path:
        return images, metadata, path
    else:
        return images, metadata


"""
*** Old version of load_image... retain until any bugs fixed in new one.
"""
# def load_image(
#         path=None,
#         display_image=True,
#         images_from_stack=None,
#         dset=None,
#         return_path=False,
#         norm_image=True,
#         full_metadata=False,
# ):
#     """Select image from 'Open File' dialog box, import and (optionally) plot

#     Parameters
#     ----------
#     path : str or None
#         The location of the image to load or the path to the folder containing
#         the desired image. If only a directory is given, the "Open file"
#         dialog box will still open allowing you to select an image file.

#     display_image : bool
#         If True, plots image (or first image if a series is imported).
#         Default: True

#     images_from_stack : None or 'all' or int or list-like
#         If file at path contains a stack of images, this argument controls
#         importing some or all of the images.
#         Default: None: import only the first image of the stack.
#         'all' : import all images as a 3d numpy array.

#     dset : str or int
#         If more than one dataset in the file, the title or index of the desired
#         dataset (for Velox .emd files) or the dataset number for all other
#         filetypes.

#     full_metadata : bool
#         For emd files ONLY, whether to load the entire metadata as nested
#         dictionaries using JSON. If False, loads standard metadata using
#         ncempy reader (including pixel size). If True, all metadata available
#         in the file is loaded. It is a lot of metadata!
#         Default: False

#     Returns
#     -------
#     image : ndarray
#         The imported image

#     metadata : dict
#         The metadata available in the original file

#     """

#     if path is None:
#         path, _ = qfd.getOpenFileName(
#             caption='Select an image to load...',
#             filter="Images (*.png *.jpg *.tif *.dm4 *.dm3 *.emd *.ser)"
#         )

#         print(f'path to imported image: {path}')

#     elif path[-4:] in ['.dm4', '.dm3', '.emd', '.ser', '.tif', '.png', '.jpg']:
#         pass

#     else:
#         path, _ = qfd.getOpenFileName(
#             caption='Select an image to load...',
#             directory=path,
#             filter="Images (*.png *.jpg *.tif *.dm4 *.dm3 *.emd *.ser)"
#         )

#     if path[-3:] in ['dm4', 'dm3']:

#         dm_file = dmReader(path)
#         image = (dm_file['data'])
#         metadata = {key: val for key, val in dm_file.items() if key != 'data'}

#     elif path[-3:] == 'emd':
#         # Load emd files using hyperspy (extracts dataset names for files with
#         # multiple datasets)

#         emd = hs.load(path)
#         if type(emd) != list:
#             dsets = [emd.metadata.General.title]
#             image = np.array(emd)
#             print('Loaded only dataset: ', dsets[0])
#             dset_ind = 0
#         else:
#             dsets = [emd[i].metadata.General.title for i in range(len(emd))]
#             print('Datasets found: ', ', '.join(dsets))

#             # if dset == 'all':
#             #     dsets_to_load = dsets
#             # else:
#             #     dsets_to_load = np.isin(dsets, dset)

#             # if len(dsets_to_load) == 0:
#             #     raise Exception('No matching dataset(s) found to load.')

#             if type(dset) == str and np.isin(dset, dsets).item():
#                 dset_ind = np.argwhere(np.array(dsets) == dset).item()
#             elif type(dset) == int:
#                 dset_ind = dset
#             # If dset not specified try to load the HAADF
#             elif np.isin('HAADF', dsets).item():
#                 dset_ind = np.argwhere(np.array(dsets) == 'HAADF').item()
#             # Otherwise import the last dataset
#             else:
#                 dset_ind = len(dsets) - 1

#             print(f'{dsets[dset_ind]} image loaded.')

#             image = np.array(emd[dset_ind])

#         # Change DPC vector images from complex type to an image stack
#         if image.dtype == 'complex64':
#             image = np.stack([np.real(image), np.imag(image)])

#         # Get metadata using ncempy (because it loads more informative metadata
#         # and allows loading everyting using JSON)
#         try:
#             trap = io.StringIO()
#             with redirect_stdout(trap):  # To suppress printing from emdReader
#                 emd_file = emdReader(path, dsetNum=dset_ind)

#             # image = emd_file['data']
#             metadata = {
#                 key: val for key, val in emd_file.items() if key != 'data'
#             }

#         except IndexError as ie:
#             raise ie
#         except TypeError:

#             try:
#                 # Need to remove EDS datasets from the list and get the correct
#                 # index as spectra are not seen by ncempy functions
#                 dset_label = dsets[dset_ind]
#                 dsets = [i for i in dsets if i != 'EDS']
#                 dset_ind = np.argwhere(np.array(dsets) == dset_label).item()

#                 emd = fileEMDVelox(path)
#                 if full_metadata is False:
#                     _, metadata = emd.get_dataset(dset_ind)

#                 elif full_metadata is True:
#                     group = emd.list_data[dset_ind]
#                     tempMetaData = group['Metadata'][:, 0]
#                     validMetaDataIndex = np.where(tempMetaData > 0)
#                     metaData = tempMetaData[validMetaDataIndex].tobytes()
#                     # Interpret as UTF-8 encoded characters and load as JSON
#                     metadata = json.loads(metaData.decode('utf-8', 'ignore'))

#             except IndexError as ie:
#                 raise ie
#             except:
#                 raise Exception('Unknown file type.')

#         metadata['imageType'] = dsets[dset_ind]

#     elif path[-3:] == 'ser':
#         ser_file = serReader(path, dsetNum=dset)
#         image = ser_file['data']
#         metadata = {
#             key: val for key, val in ser_file.items() if key != 'data'
#         }

#     else:
#         image = imageio.volread(path)
#         metadata = image.meta

#     h, w = image.shape[-2:]

#     if images_from_stack is None and len(image.shape) == 3:
#         image = image[0, :, :]

#     elif images_from_stack == 'all':
#         pass
#     elif (type(images_from_stack) == list
#           or type(images_from_stack) == int):
#         image = image[images_from_stack, :, :]

#     # Norm the image(s)
#     if norm_image:
#         image = image_norm(image)

#     # Make image dimensions even length
#     if len(image.shape) == 2:
#         image = image[:int((h//2)*2), :int((w//2)*2)]
#         image_ = image

#     if len(image.shape) == 3:
#         image = image[:, :int((h//2)*2), :int((w//2)*2)]
#         image_ = image[0, :, :]

#     if display_image is True:
#         fig, axs = plt.subplots()
#         axs.imshow(image_, cmap='gray')
#         axs.set_xticks([])
#         axs.set_yticks([])

#     if return_path:
#         return image, metadata, path
#     else:
#         return image, metadata


def image_norm(image):
    """Norm an image to 0-1

    Parameters
    ----------
    image : ndarray
         Input image as an ndarray

    Returns
    -------
    image_normed : ndarray with the same shape as 'image'
        The normed image

    """

    [min_, max_] = [np.min(image), np.max(image)]
    image_normed = (image - min_)/(max_ - min_)

    return image_normed


def write_image_array_to_tif(image, filename, folder=None, bits=16):
    """Save an ndarray as an 8 or 16 bit TIFF image.

    Parameters
    ----------
    image : ndarray
        Input image as an ndarray

    folder : str or None
        Directory in which to save the image. If None, select file dialog box
        will be opened.
    filename : str
        The file name to use for the saved image.

    bits : int
        The number of bits to use for the saved .tif. Must be 8, 16.

    Returns
    -------
    None

    """

    image = image_norm(image)
    if folder is None:
        folder = qfd.getExistingDirectory()
    if (filename[-4] != '.'):
        filename += '.tif'

    if bits == 8:
        dtype = np.uint8
    elif bits == 16:
        dtype = np.uint16
    else:
        raise Exception('"bits" must be 8 or 16')

    imwrite(
        os.path.join(folder, filename),
        (image*(2**bits-1)).astype(dtype),
        photometric='minisblack'
    )


def img_equ_ellip(image):
    """Calculate the equivalent ellipse

    Parameters
    ----------
    image : ndarray
         Input image as an ndarray

    Returns
    -------
    eigvals : squared magnitudes of the major and minor semi-axes, in that
        order
    eigvects : matrix containing the unit vectors of the major and minor
    semi-axes (in that order) as row vectors
    x0, y0 : coordinates of the ellipse center

    """

    M = moments(image, order=1)
    mu = moments_central(image, order=2)

    try:
        [x0, y0] = [M[0, 1]/M[0, 0], M[1, 0]/M[0, 0]]

        [u20, u11, u02] = [
            mu[2, 0]/M[0, 0],
            mu[1, 1]/M[0, 0],
            mu[0, 2]/M[0, 0]
        ]

        cov = np.array(
            [[u20, u11],
             [u11, u02]]
        )

    except ZeroDivisionError as err:
        raise ZeroDivisionError(no_mask_error) from err

    try:
        eigvals, eigvects = np.linalg.eig(cov)
    except np.linalg.LinAlgError as err:
        raise ArithmeticError(no_mask_error) from err

    # Exchange vector components so each column vector is [x, y]:
    eigvects = np.flipud(eigvects)

    if eigvects[0, 0] < 0:
        eigvects[:, 0] *= -1
    if eigvects[0, 1] < 0:
        eigvects[:, 1] *= -1

    ind_sort = np.flip(np.argsort(eigvals))  # Sort large to small

    eigvals = np.abs(np.take_along_axis(
        eigvals,
        ind_sort,
        0
    ))

    eigvects = np.array([eigvects[:, ind] for ind in ind_sort])

    return eigvals, eigvects, x0, y0


def img_ellip_param(image):
    """Find parameters of the equivalent ellipse

    Calls img_equ_ellip and transforms result to a more intuitive form

    Parameters
    ----------
    image : ndarray
         Input image as an ndarray

    Returns
    -------
    x0, y0 : coordinates of the ellipse center

    eccen : eccentricity of the ellipse (standard mathmatical definition)

    theta : rotation angle of the major semi-axis relative to horizontal
        in degrees (positive is counterclockwise)

    sig_1 : magnitude of the major semi-axis

    sig_2 : magnitude of the major semi-axis

    """

    eigvals, eigvects, x0, y0 = img_equ_ellip(image)
    major = np.argmax(eigvals)
    minor = np.argmin(eigvals)
    sig_1 = np.sqrt(eigvals[major])
    sig_2 = np.sqrt(eigvals[minor])
    eccen = np.sqrt(1-eigvals[minor]/eigvals[major])
    theta = np.degrees(np.arcsin(
        np.cross(np.array([1, 0]),
                 eigvects[major]).item()
    ))

    return x0, y0, eccen, theta, sig_1, sig_2


def gaussian_2d(x, y, x0, y0, sig_1, sig_2, ang, A=1, I_o=0):
    """Sample a 2D, ellpitical Gaussian function.

    Samples a specified 2D Gaussian function at an array of points.

    Parameters
    ----------
    x, y : ndarrays, must have the same shape
        They x and y coordinates of each sampling point. If given arrays
        generated by numpy.mgrid or numpy.meshgrid, will return an image
        of the Gaussian.

    x0, y0 : center of the Gaussian

    sig_1 : sigma of the major axis

    sig_ratio : ratio of sigma major to sigma minor

    ang : rotation angle of the major axis from horizontal

    A : Peak amplitude of the Gaussian

    I_o : Constant background value

    Returns
    -------
    I : the value of the function at the specified points. Will have the same
        shape as x, y inputs

    """

    ang = np.radians(-ang)  # negative due to inverted y axis in python
    I = I_o + A*np.exp(-1/2*(
        ((np.cos(ang) * (x - x0) + np.sin(ang) * (y - y0)) / sig_1)**2
        + ((-np.sin(ang) * (x - x0) + np.cos(ang) * (y - y0)) / sig_2)**2))

    return I


def LoG_2d(x, y, x0, y0, sig, A=1, I_o=0):
    """Sample a round, 2D Laplacian of Gaussian function.

    Samples a 2D Laplacian of Gaussian function at an array of points.

    Parameters
    ----------
    x, y : ndarrays, must have the same shape
        They x and y coordinates of each sampling point. If given arrays
        generated by numpy.mgrid or numpy.meshgrid, will return an image
        of the Gaussian.
    x0, y0 : center of the Gaussian
    sig : variance of the
    A : Peak amplitude of the Gaussian
    I_o : Constant background value

    Returns
    -------
    I : the value of the function at the specified points. Will have the same
        shape as x, y inputs

    """

    I = I_o + (-A*np.exp(-((x - x0)**2 + (y - y0)**2) / (2*sig**2))
               * (((x - x0)**2 + (y - y0)**2) / (sig**4) - 2/sig**2))

    return I


def gaussian_ellip_ss(p0, x, y, z, masks=None):
    """Sum of squares for a Gaussian function.

    Takes a parameter vector, coordinates, and corresponding data values;
    returns the sum of squares of the residuals.

    Parameters
    ----------
    p0 : array_like with shape (n,7)
        n = number of peaks to fit
        Array containing the Gaussian function parameter vector(s):
            [x0, y0, sig_maj, sig_rat, ang, A, I_o]

    x, y : 1D array_like, must have the same shape
        The flattened arrays of x and y coordinates of image pixels

    z : 1D array_like, must have the same shape as x and y
        The flattened array of image values at the x, y coordinates

    masks : 2d array_like of size (n, m)
        n = number of peaks to fit
        m = number of unmasked pixels
        The flattened masks for each peak. Each of the "n" rows is 1 where the
        mask for the corresponding peak and 0 elsewhere.

    Returns
    -------
    r_sum_sqrd : float
        The sum of the squares of the residuals

    """

    if p0.shape[0] > 7:
        I0 = p0[-1]
        p0_ = p0[:-1].reshape((-1, 6))
        x0, y0, sig_maj, sig_rat, ang, A = np.split(p0_, 6, axis=1)
    else:
        x0, y0, sig_maj, sig_rat, ang, A, I0 = p0

    sig_min = sig_maj/sig_rat

    # Sum the functions for each peak:
    model = np.sum(A*np.exp(-1/2*(((np.cos(ang) * (x - x0)
                                    + np.sin(ang) * (y - y0))
                                   / sig_maj)**2
                                  + ((-np.sin(ang) * (x - x0)
                                      + np.cos(ang) * (y - y0))
                                     / sig_min)**2)),
                   axis=0) + I0

    # Subtract from data to get residuals:
    R = z - model
    r_sum_sqrd = (R @ R.T).flatten()

    return r_sum_sqrd


def gaussian_circ_ss(p0, x, y, z, masks=None):
    """Sum of squares for a Gaussian function.

    Takes a parameter vector, coordinates, and corresponding data values;
    returns the sum of squares of the residuals.

    Parameters
    ----------
    p0 : array_like with shape (n,7)
        n = number of peaks to fit
        Array containing the Gaussian function parameter vector(s):
            [x0, y0, sig_maj, sig_rat, ang, A, I_o]

    x, y : 1D array_like, must have the same shape
        The flattened arrays of x and y coordinates of image pixels

    z : 1D array_like, must have the same shape as x and y
        The flattened array of image values at the x, y coordinates

    masks : 2d array_like of size (n, m)
        n = number of peaks to fit
        m = number of unmasked pixels
        The flattened masks for each peak. Each of the "n" rows is 1 where the
        mask for the corresponding peak and 0 elsewhere.

    Returns
    -------
    r_sum_sqrd : float
        The sum of the squares of the residuals

    """

    if p0.shape[0] > 5:
        I0 = p0[-1]
        p0_ = p0[:-1].reshape((-1, 4))
        x0, y0, sig, A = np.split(p0_, 4, axis=1)
    else:
        x0, y0, sig, A, I0 = p0

    # Sum the functions for each peak:
    model = np.sum(A*np.exp(-1/2*(
        (((x - x0)) / sig)**2
        + (((y - y0)) / sig)**2)),
        axis=0) + I0

    # Subtract from data to get residuals:
    R = z - model
    r_sum_sqrd = (R @ R.T).flatten()

    return r_sum_sqrd


def fit_gaussian_ellip(
        data,
        p0,
        masks=None,
        method='BFGS',
        bounds=None
):
    """Fit an elliptical 2D Gaussain function to data.

    Fits a 2D, elliptical Gaussian to an image. Intensity values equal to zero
    are ignored.

    Parameters
    ----------
    data : ndarray
        Image containing a Gaussian peak

    p0 : array_like with shape (n*6 + 1,)
        Initial guess for the n-Gaussian parameter vector where each peak
        has 6 independent parameters (x0, y0, sig_maj, sig_ratio, ang, A) the
        whole region has a constant background (I_0) which is the last item in
        the array.

    masks : 2d array_like of size (n, m)
        n = number of peaks to fit
        m = number of unmasked pixels
        The flattened masks for each peak. Each of the "n" rows is 1 where the
        mask for the corresponding peak and 0 elsewhere.

    method : str, the minimization solver name
        Supported solvers are: ''BFGS', 'L-BFGS-B', 'Powell', 'trust-constr'.
        Default: 'BFGS'

    bounds : list of two-tuples of length 7*n or None
        The bounds for Gaussian fitting parameters. Only works with methods
        that accept bounds (e.g. 'L-BFGS-B', but not 'BFGS'). Otherwise must
        be set to None.
        Each two-tuple is in the form: (upper, lower).
        Order of bounds must be: [x0, y0, sig_maj, sig_ratio, ang, A, I_0] * n
        Default: None

    Returns
    -------
    params : 1D array
        The best fit parameter vector found by least squares

    """

    num_gauss = int(np.ceil(p0.shape[0]/7))
    img_shape = data.shape

    I0 = p0[-1]
    p0_ = p0[:-1].reshape((num_gauss, 6))

    p0_[:, 4] *= -1

    y, x = np.indices(img_shape)
    z = data.flatten()

    unmasked_data = np.nonzero(z)
    z = np.take(z, unmasked_data)
    x = np.take(x.flatten(), unmasked_data)
    y = np.take(y.flatten(), unmasked_data)

    x0y0 = p0_[:, :2]
    if masks is None:
        image = np.zeros(img_shape)
        image[y, x] = z
        masks_labeled, _ = label(image)

    elif (type(masks) == imageio.core.util.Array
          or type(masks) == np.ndarray):
        masks_labeled, _ = label(masks)

    masks_to_peaks = map_coordinates(
        masks_labeled,
        np.flipud(x0y0.T),
        order=0
    ).astype(int)

    masks_labeled = np.take(masks_labeled, unmasked_data).flatten()

    masks_labeled = np.array(
        [np.where(masks_labeled == mask_num, 1, 0)
         for i, mask_num in enumerate(masks_to_peaks)]
    )

    p0_ = np.append(p0_.flatten(), I0)

    with warnings.catch_warnings():
        warnings.filterwarnings(
            'ignore',
            category=UserWarning,
            lineno=182
        )

        warnings.filterwarnings(
            'ignore',
            category=RuntimeWarning,
            lineno=579
        )

    params = minimize(
        gaussian_ellip_ss,
        p0_,
        args=(x, y, z, masks_labeled),
        bounds=bounds,
        method=method
    ).x

    params = np.concatenate(
        (params[:-1].reshape((-1, 6)),
         np.ones((num_gauss, 1))*params[-1]),
        axis=1
    )

    params[:, 4] *= -1
    params[:, 4] = ((params[:, 4] + np.pi/2) % np.pi) - np.pi/2

    return params


def fit_gaussian_circ(
        data,
        p0,
        masks=None,
        method='BFGS',
        bounds=None
):
    """Fit an elliptical 2D Gaussain function to data.

    Fits a 2D, elliptical Gaussian to an image. Intensity values equal to zero
    are ignored.

    Parameters
    ----------
    data : ndarray
        Image containing a Gaussian peak

    p0 : array_like with shape (n, 5)
        Initial guess for the n-Gaussian parameter vector where each peak
        has 4 independent parameters (x0, y0, sig, A) the
        whole region has a constant background (I_0).

    masks : 2d array_like of size (n, m)
        n = number of peaks to fit
        m = number of unmasked pixels
        The flattened masks for each peak. Each of the "n" rows is 1 where the
        mask for the corresponding peak and 0 elsewhere.

    method : str, the minimization solver name
        Supported solvers are: 'BFGS', 'L-BFGS-B', 'Powell', 'trust-constr'.
        Default: 'BFGS'

    bounds : list of two-tuples of length 7*n or None
        The bounds for Gaussian fitting parameters. Only works with methods
        that accept bounds (e.g. 'L-BFGS-B', but not 'BFGS'). Otherwise must
        be set to None.
        Each two-tuple is in the form: (upper, lower).
        Order of bounds must be: [x0, y0, sig, A, I_0] * n
        Default: None

    Returns
    -------
    params : 1D array
        The best fit parameter vector found by least squares

    """

    num_gauss = int(np.ceil(p0.shape[0]/5))
    img_shape = data.shape
    p0.shape
    I0 = p0[-1]
    p0_ = p0[:-1].reshape((num_gauss, 4))

    y, x = np.indices(img_shape)
    z = data.flatten()

    unmasked_data = np.nonzero(z)
    z = np.take(z, unmasked_data)
    x = np.take(x.flatten(), unmasked_data)
    y = np.take(y.flatten(), unmasked_data)

    x0y0 = p0_[:, :2]
    if masks is None:
        image = np.zeros(img_shape)
        image[y, x] = z
        masks_labeled, _ = label(image)

    elif (type(masks) == imageio.core.util.Array
          or type(masks) == np.ndarray):
        masks_labeled, _ = label(masks)

    masks_to_peaks = map_coordinates(
        masks_labeled,
        np.flipud(x0y0.T),
        order=0
    ).astype(int)

    masks_labeled = np.take(
        masks_labeled,
        unmasked_data
    ).flatten()

    masks_labeled = np.array(
        [np.where(masks_labeled == mask_num, 1, 0)
         for i, mask_num in enumerate(masks_to_peaks)]
    )

    p0_ = np.append(p0_.flatten(), I0)

    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=UserWarning,
                                lineno=182)
        warnings.filterwarnings('ignore', category=RuntimeWarning,
                                lineno=579)

    params = minimize(
        gaussian_circ_ss,
        p0_,
        args=(x, y, z, masks_labeled),
        bounds=bounds,
        method=method
    ).x

    params = np.concatenate(
        (params[:-1].reshape((-1, 4)),
         np.ones((num_gauss, 1))*params[-1]),
        axis=1
    )

    # Add dummy columns for sigma ratio and rotation angle:
    params = np.insert(
        params,
        3,
        np.array([1, 0]*num_gauss).reshape(-1, 2).T,
        axis=1
    )

    return params


def pcf_radial(
        dr,
        coords,
        total_area=None
):
    """Calculate a radial pair correlation function from 2D or 3D data.

    Parameters
    ----------
    dr : int or float
        The step size for binning distances

    coords : array_like with shape (n, d)
        The x, y, z coordinates of each point. n is the number of points,
        d is the number of dimensions (i.e. 2 or 3)

    total_area : area or volume of region containing the points. Estimate
        made if not given.

    Returns
    -------
    pcf : 1D array
        Histogram of point-point distances with bin size dr

    """

    if total_area is None:
        diag_vect = np.max(coords, axis=0) - np.min(coords, axis=0)
        total_area = diag_vect @ np.ones(diag_vect.shape)

    n = coords.shape[0]
    rho = n / total_area

    vects = np.array([coords - i for i in coords])

    dist = np.hypot(vects[:, :, 0], vects[:, :, 1])
    bins = (dist/dr).astype(int)

    r = np.arange(0, np.max(dist), dr)
    A_sh = np.array([np.pi * (r_**2 - (r_ - dr)**2) for r_ in r])

    hist = np.bincount(bins.flatten())
    hist[0] = 0

    pcf = hist / (n * rho * A_sh)

    return pcf


def v_pcf(
        xlim,
        ylim,
        coords1,
        coords2=None,
        d=0.05,
        area=None,
        method='weighted'
):
    """
    Get a 2D pair (or pair-pair) correlation function for a dataset.

    Parameters
    ----------
    xlim, ylim : 2-tuple of floats or ints
        The limits of the vPDF along each dimension. Must include 0 in both
        x and y. The limits will determine the size of the vPCF array and the
        time required to calculate.
    coords1 : array of shape (n, 2)
        The (x,y) coordinates of the data points.
    coords2 : None or array of shape (n, 2)
        If dcoords 2 is None, a vPCF is calculated for coords1 with respect to
        itself. If a second data array is passed as coords2, a pair-pair vPCF
        is found i.e. the vPCF of coords1 data with respect to coords2. coords1
        and coords2 do not necessarily have to have the same number of data
        points.
        Default: None
    d : scalar
        The pixel size of the vPCF in the same units as the coords1/coords2.
    area : scalar
        The area containing the data points. Used to calculate the density
        for normalizing the vPCF values. If None, the rectangle containing
        the extreme points in coords1 is taken as the area. This may be wrong
        if the data does not come from a retangular area or the rectangle has
        been rotated relative to the cartesian axes.
        Default: None
    method : 'bin' or 'weighted'
        The method to use for calculating the v_pcf. If 'bin', uses a direct
        histogram binning function in two dimensions. If 'weighted',
        linearly divides the count for each data point among the 2x2 nearest
        neighbor pixels. Examples:
            1: A point exactly at the center of a pixel will have its full
            weight placed in that pixel and none in any others.
            2: A point at the common corner of 4 pixels will have 1/4 weight
            assigned to each.
        Discussion: 'bin' is about 4x faster in execution while 'weight' is
        more quantitatively correct. Practically, the results will be very
        similar. 'bin' should only be preferred if the function must be called
        many times and speed is critical. This option may be removed in a
        future version in favor of always using the weighted method.
        Default: 'weight'

    Returns
    -------
    v_pcf : ndarray of shape (int((ylim[1]-ylim[0])/d),
                              int((xlim[1]-xlim[0])/d))
        The vPCF.
    origin : array-like of shape (2,)
        The x, y coordinates of the vPCF origin, given that the y axis points
        down.

    """

    if ((xlim[0] > 0) or (ylim[0] > 0) or (xlim[1] < 0) or (ylim[1] < 0)):
        raise Exception(
            "x and y limits must include the origin, i.e. (0,0)"
        )

    if area is None:
        area = (np.max(coords1[:, 0])
                - np.min(coords1[:, 0])) * \
            (np.max(coords1[:, 1]) - np.min(coords1[:, 1]))

    # Get the point-to-point vectors
    if coords2 is None:
        coords2 = coords1

        # Skip 0 length vectors for a partial vPCF
        vects = np.array([
            np.delete(coords1, i, axis=0) - xy for i, xy in enumerate(coords2)
        ])
    else:
        # Keep all vectors for a pair-pair vPCF
        vects = np.array([coords1 - i for i in coords2])

    x_ = vects[:, :, 0].flatten()
    y_ = vects[:, :, 1].flatten()

    n_sq = coords1.shape[0] * coords2.shape[0]
    denominator = n_sq / area

    # Get bin spacing
    xedges = np.arange(xlim[0], xlim[1], d)
    yedges = np.arange(ylim[0], ylim[1], d)

    # Find edge closest to 0 and shift edges so (0,0) is exactly at the center
    # of a pixel
    x_min_ind = np.argmin(np.abs(xedges))
    y_min_ind = np.argmin(np.abs(yedges))
    xedges -= xedges[x_min_ind] + d/2
    yedges -= yedges[y_min_ind] + d/2

    if method == 'bin':
        # Bin into 2D histogram
        H, _, _ = np.histogram2d(
            y_,
            x_,
            bins=[yedges, xedges]
        )

    elif method == 'weighted':
        xcents = xedges[:-1] + d/2
        ycents = yedges[:-1] + d/2

        H = np.zeros((ycents.shape[0], xcents.shape[0]))

        # Round values down to nearest pixel
        xF = np.floor(x_/d) * d
        yF = np.floor(y_/d) * d

        # Get x and y weights for the floor pixels
        xFw = 1 - np.abs(x_/d % 1)
        yFw = 1 - np.abs(y_/d % 1)

        # Weighted histogram for x & y floor pixels
        H += np.histogram2d(
            yF, xF,
            bins=[yedges, xedges],
            weights=xFw * yFw
        )[0]

        # Weighted histogram for x ceiling & y floor pixels
        H += np.histogram2d(
            yF, xF + d,
            bins=[yedges, xedges],
            weights=(1 - xFw) * yFw
        )[0]

        # Weighted histogram for x floor & y ceiling pixels
        H += np.histogram2d(
            yF + d, xF,
            bins=[yedges, xedges],
            weights=xFw * (1 - yFw)
        )[0]

        # Weighted histogram for x & y ceiling pixels
        H += np.histogram2d(
            yF + d, xF + d,
            bins=[yedges, xedges],
            weights=(1 - xFw) * (1 - yFw)
        )[0]

    else:
        raise Exception(
            "'method' must be either 'bin' or 'weighted'"
        )

    # Flip so y axis is positive going up
    H = np.flipud(H)

    # Find the origin
    origin = np.array([
        np.argwhere(np.isclose(xedges, -d/2)).item(),
        yedges.shape[0] - np.argwhere(np.isclose(yedges, -d/2)).item() - 2
    ])

    H[origin[1], origin[0]] = 0

    v_pcf = H/(denominator * d**2)  # Normalize vPCF by number density

    return v_pcf, origin


def detect_peaks(
        image,
        min_dist=4,
        thresh=0
):
    """Detect peaks in an image using a maximum filter with a minimum
    separation distance and threshold.

    Parameters
    ----------
    image : 2D array_like
        The image to be analyzed.

    min_dist : int or float
        The minimum distance allowed between detected peaks. Used to create
        a circular neighborhood kernel for peak detection.
        Default: 4

    thresh : int or float
        The minimum image value that should be considered a peak. Used to
        remove low intensity background noise peaks.
        Default: 0

    Returns
    -------
    peaks : 2D array_like with shape: image.shape
        Array with 1 indicating peak pixels and 0 elsewhere.

    """

    kern_rad = int(np.floor(min_dist/2))
    if min_dist < 1:
        min_dist = 1
    size = 2*kern_rad + 1
    neighborhood = np.array(
        [1 if np.hypot(i - kern_rad, j - kern_rad) <= min_dist/2
         else 0
         for j in range(size) for i in range(size)]
    ).reshape((size, size))

    # im_std = std_local(image, int(min_dist/2))
    # max_, min_ = [np.max(im_std), np.min(im_std)]
    # thresh = 0.2*(max_ - min_) + min_

    peaks = (maximum_filter(image, footprint=neighborhood) == image
             ) * (image > thresh)

    return peaks.astype(int)


def disp_vect_sum_squares(p0, xy, M):
    """Objective function for 'fit_lattice()'.

    Parameters
    ----------
    p0 : list-like of shape (6,)
        The current basis guess of the form: [a1x, a1y, a2x, a2y, x0, y0].
        Where [a1x, a1y] is the first basis vector, [a2x, a2y] is the second
        basis vector and [x0, y0] is the origin.
    xy : array-like of shape (n, 2)
        The array of measured [x, y] lattice coordinates.
    M : array-like of shape (n, 2)
        The array of fractional coorinates or reciprocal lattice indice
        corresponding to the xy coordinates. Rows correspond to [u, v] or
        [h, k] coordinates depending on whether the data is in real or
        reciproal space.

    Returns
    -------
    sum_sq : scalar
        The sum of squared errors given p0.

    """

    dir_struct_matrix = p0[:4].reshape((2, 2))
    origin = p0[4:]

    err_xy = xy - M @ dir_struct_matrix - origin
    sum_sq = np.sum(err_xy**2)

    return sum_sq


def fit_lattice(p0, xy, M, fix_origin=False):
    """Find the best fit of a rigid lattice to a set of points.

    Parameters
    ----------
    p0 : list-like of shape (6,)
        The initial basis guess of the form: [a1x, a1y, a2x, a2y, x0, y0].
        Where [a1x, a1y] is the first basis vector, [a2x, a2y] is the second
        basis vector and [x0, y0] is the origin.
    xy : array-like of shape (n, 2)
        The array of measured [x, y] lattice coordinates.
    M : array-like of shape (n, 2)
        The array of fractional coorinates or reciprocal lattice indices
        corresponding to the xy coordinates. Rows correspond to [u, v] or
        [h, k] coordinates depending on whether the data is in real or
        reciproal space.
    fix_origin : bool
        Whether to fix the origin (if True) or allow it to be refined
        (if False). Generally, should be false unless data is from an FFT,
        then the origin is known and should be fixed.

    Returns
    -------
    params : list-like of shape (6,)
        The refined basis, using the same form as p0.

    """

    p0 = np.array(p0)
    x0, y0 = p0[4:]

    if fix_origin is True:
        bounds = [(None, None)] * 4 + [(x0, x0), (y0, y0)]
    else:
        bounds = [(None, None)] * 6

    params = minimize(
        disp_vect_sum_squares,
        p0,
        bounds=bounds,
        args=(xy, M),
        method='L-BFGS-B',
    ).x

    return params


def watershed_segment(
        image,
        sigma=None,
        buffer=0,
        local_thresh_factor=0.95,
        watershed_line=True,
        min_dist=4
):
    """Segment an image using the Watershed algorithm.

    Parameters
    ----------
    image : 2D array_like
        The image to be segmented.

    sigma : int or float
        The Laplacian of Gaussian sigma value to use for peak sharpening. If
        None, no filtering is applied.
        Default: None

    buffer : int
        The border within which peaks are ignored.
        Default: 0

    local_thresh_factor : float
        Removes background from each segmented region by thresholding.
        Threshold value determined by finding the maximum value of edge pixels
        in the segmented region and multipling this value by the
        local_thresh_factor value. The filtered image is used for this
        calculation.
        Default 0.95.

    watershed_line : bool
        Seperate segmented regions by one pixel.
        Default: True.

    min_dist : int or float
        The minimum distance allowed between detected peaks. Used to create
        a circular neighborhood kernel for peak detection.
        Default: 4.

    Returns
    -------
    masks : 2D array with same shape as image

    num_masks : int
        The number of masks

    slices : List of image slices which contain each region

    peaks : DataFrame with the coordinates and corresponding mask label for
        each peak not outside the buffer

    """

    img_der = copy.deepcopy(image)
    [h, w] = image.shape

    if type(sigma) in (int, float, tuple):
        img_der = image_norm(-gaussian_laplace(img_der, sigma))

    local_max = detect_peaks(img_der, min_dist=min_dist)

    # Remove peaks at edges
    frame_edge = np.zeros((h, w), dtype=np.int64)
    frame_edge[1:-1, 1:-1] = 1
    local_max *= frame_edge
    local_max = label(local_max)[0]

    masks = watershed(-img_der, local_max, watershed_line=watershed_line)
    # img_der = (image_norm(-img_der) * (2**8 - 1)).astype(np.uint8)
    # img_der_3 = np.zeros((h, w, 3), dtype=np.uint8)
    # img_der_3[:, :, 0] = img_der
    # img_der_3[:, :, 1] = img_der
    # img_der_3[:, :, 2] = img_der

    # masks = np.squeeze(watershed(img_der_3, copy.deepcopy(local_max)))

    # masks[masks <= 0] = 0

    slices = find_objects(masks)
    num_masks = int(np.max(masks))

    """Refine masks with local_thresh_factor"""
    if local_thresh_factor > 0:
        masks_ref = np.zeros(image.shape)

        for i in range(0, num_masks):
            mask_sl = np.where(masks[slices[i][0], slices[i][1]] == i+1, 1, 0)
            img_der_sl = img_der[slices[i][0], slices[i][1]]
            edge = mask_sl - binary_erosion(mask_sl)
            thresh = np.max(edge * img_der_sl) * (local_thresh_factor)
            mask_sl = np.where(mask_sl*img_der_sl >= thresh, i+1, 0)
            masks_ref[slices[i][0], slices[i][1]] += mask_sl

        masks = masks_ref

    _, peak_xy = np.unique(local_max, return_index=True)

    peak_xy = np.fliplr(np.array(np.unravel_index(
        peak_xy,
        local_max.shape
    )).T[1:, :])

    peaks = pd.DataFrame.from_dict({
        'x': list(peak_xy[:, 0]),
        'y': list(peak_xy[:, 1]),
        'max': image[peak_xy[:, 1], peak_xy[:, 0]],
        'label': [i+1 for i in range(num_masks)]
    })

    peaks = peaks[
        ((peaks.x >= buffer) &
         (peaks.x <= w - buffer) &
         (peaks.y >= buffer) &
         (peaks.y <= h - buffer))
    ]

    peaks = peaks.reset_index(drop=True)

    return masks, num_masks, slices, peaks


def band_pass_filter(
        shape,
        high_pass=5,
        low_pass=None,
        filter_edge_smoothing=0
):
    """Create a high and/or low pass filter.

    Parameters
    ----------
    shape : two-tuple
        The image shape for which to make the mask.

    high_pass : int or None
        The number of reciprocal pixels below which to block with the high
        pass filter.
        Default: 5

    low_pass : int or None
        The number of reciprocal pixels above which to block with the low pass
        filter.
        Default: None

    filter_edge_smoothing : int
        Gaussian blur sigma used to smooth the hard edge of the band pass
        filter in reciprocal pixels.
        Default 0.

    Returns
    -------
    masks : 2D array with same shape as image

    num_masks : int
        The number of masks

    slices : List of image slices which contain each region

    peaks : DataFrame with the coordinates and corresponding mask label for
        each peak not outside the buffer

    """

    f_freq_1d = [
        np.fft.fftfreq(shape[0], 1),
        np.fft.fftfreq(shape[1], 1)
    ]

    max_dim = np.argmax(shape)
    fft_pixel_size = np.array(
        [f_freq_1d[0][1] - f_freq_1d[0][0],
         f_freq_1d[1][1] - f_freq_1d[1][0]]
    )

    if not high_pass:
        high_pass = 0
    else:
        high_pass *= fft_pixel_size[max_dim]
    if not low_pass:
        low_pass = np.inf
    else:
        low_pass *= fft_pixel_size[max_dim]

    fft_freq_abs = norm(np.array(
        np.meshgrid(f_freq_1d[1], f_freq_1d[0])), axis=0
    )

    mask = np.where(
        ((fft_freq_abs >= high_pass) &
         (fft_freq_abs <= low_pass)),
        1, 0
    ).astype(float)

    if filter_edge_smoothing:
        mask = image_norm(gaussian_filter(
            mask,
            sigma=filter_edge_smoothing,
            truncate=4*filter_edge_smoothing
        ))

    return mask


def get_phase_from_com(
        comx,
        comy,
        theta,
        flip=True,
        high_low_filter=False,
        filter_params={
            'beam_energy': 200e3,
            'conv_semi_angle': 18,
            'pixel_size': 0.01,
            'high_pass': 0.05,
            'low_pass': 0.85,
            'edge_smoothing': 0.01
        }
):
    """Reconstruct phase from center of mass shift components.

    *** Citations...

    Parameters
    ----------
    com_xy : ndarray of shape (h,w,2)
        The center of mass shift component images as a stack.

    theta : scalar
        Rotation angle in degrees between real and reciprocal space.

    flip : bool
        Whether to transpose x and y axes.
        Default: False

    high_low_filter : bool
        Whether to perform high and/or low pass filtering as defined in the
        filter_params argument.
        Default: False

    filter_params : dict
        Dictionary of parameters used for calculating the high and/or low pass
        filters:
            {'beam_energy' : 200e3,    # electron-volts
             'conv_semi_angle' : 18,   # mrads
             'pixel_size' : 0.01,      # nm
             'high_pass' : 0.05,       # fraction of aperture passband
             'low_pass' : 0.85,        # fraction of aperture passband
             'edge_smoothing' : 0.01}  # fraction of aperture passband

    Returns
    -------
    phase : 2D array with shape (h,w)
        The reconstructed phase of the sample transmission function.

    """

    f_freq_1d_y = np.fft.fftfreq(comx.shape[0], 1)
    f_freq_1d_x = np.fft.fftfreq(comx.shape[1], 1)
    h, w = comx.shape

    if high_low_filter:
        lambda_ = elec_wavelength(filter_params['beam_energy']) * 1e9  # in nm
        h, w = comx.shape
        min_dim = np.argmin((h, w))
        fft_pixelSize = 1 / (np.array(comx.shape)
                             * filter_params['pixel_size'])  # nm^-1

        apeture_cutoff = 2*np.sin(2 * filter_params['conv_semi_angle']/1000
                                  )/lambda_
        low_pass = (filter_params['low_pass']*apeture_cutoff
                    / fft_pixelSize[min_dim])
        high_pass = (filter_params['high_pass']*apeture_cutoff
                     / fft_pixelSize[min_dim])  # pixels

        sigma = (filter_params['edge_smoothing'] * apeture_cutoff
                 / fft_pixelSize[min_dim])

        mask = band_pass_filter((h, w), high_pass=high_pass,
                                low_pass=low_pass,
                                filter_edge_smoothing=sigma)

    else:
        mask = np.ones((h, w))

    # theta = np.radians(theta)
    if not flip:
        comx_ = comx*np.cos(theta) - comy*np.sin(theta)
        comy_ = comx*np.sin(theta) + comy*np.cos(theta)
    if flip:
        comx_ = comx*np.cos(theta) + comy*np.sin(theta)
        comy_ = comx*np.sin(theta) - comy*np.cos(theta)

    k_p = np.array(np.meshgrid(f_freq_1d_x, f_freq_1d_y))

    _ = np.seterr(divide='ignore')
    denominator = 1/((k_p[0] + 1j*k_p[1]))
    denominator[0, 0] = 0
    _ = np.seterr(divide='warn')

    complex_image = np.fft.ifft2((mask * (np.fft.fft2(comy_)
                                          + 1j*np.fft.fft2(comx_)))
                                 * denominator)

    # phase = (np.real(complex_image)**2 + np.imag(complex_image)**2)**0.5
    phase = - np.real(complex_image)
    return phase


def get_charge_density_image(comx, comy, theta):
    """Get charge density image.

    Parameters
    ----------
    comx, comy : ndarrays of shape (h,w)
        The center of mass shift component images.

    theta : scalar
        Rotation angle in degrees between horizontal scan direction and
        detector orientation (i.e. the comx vector).

    Returns
    -------
    charge_density : 2D array with shape (h,w)
        The sample charge density convolved with the probe intensity function.

    """

    theta = np.radians(theta)
    comx_ = comx*np.cos(theta) - comy*np.sin(theta)
    comy_ = comx*np.sin(theta) + comy*np.cos(theta)

    charge_density = np.gradient(comx_, axis=1) + np.gradient(comy_, axis=0)

    return charge_density


def fast_rotate_90deg(image, angle):
    """Rotate images by multiples of 90 degrees. Faster than
    scipy.ndimage.rotate().

    Parameters
    ----------
    image : ndarray of shape (h,w)
        The image.

    angle : scalar
        Rotation angle in degrees. Must be a multiple of 90.

    Returns
    -------
    rotated_image : 2D array
        The image rotated by the specified angle.

    """

    angle = angle % 360
    if angle == 90:
        image_ = np.flipud(image.T)
    elif angle == 180:
        image_ = np.flipud(np.fliplr(image))
    elif angle == 270:
        image_ = np.fliplr(image.T)
    elif angle == 0:
        image_ = image
    else:
        raise Exception('Argument "angle" must be a multiple of 90 degrees')

    return image_


def fft_amplitude_area(
        image,
        xy_fft,
        r,
        blur,
        thresh=0.5,
        fill_holes=True,
        buffer=10
):
    """Create mask based on Bragg spot filtering (via FFT) of image.

    Parameters
    ----------
    image : ndarray of shape (h,w)
        The image.

    xy_fft : ndarray

        The Bragg spot coordinates in the FFT. Must be shape (n,2).
    r : int, float or list-like of ints or floats
        The radius (or radii) of Bragg spot pass filters. If a sclar, the same
        radius is applied to all Bragg spots. If list-like, must be of shape
        (n,).

    blur : int or float
        The gaussian sigma used to blur the amplitude image.

    thresh : float
        The relative threshold for creating the mask from the blured amplitude
        image.
        Default: 0.5

    fill_holes : bool
        If true, interior holes in the mask are filled.
        Default: True

    buffer : int
        Number of pixels to erode from the mask after binarization. When used
        as a mask for restricting atom column detection, this prevents
        searching too close to the edge of the area.
        Default: 10

    Returns
    -------
    mask : 2D array
        The final amplitude mask.

    """

    if not (type(r) == int) | (type(r) == float):
        if xy_fft.shape[0] != r.shape[0]:
            raise Exception("If'r' is not an int or float, its length "
                            "must match the first dimension of xy_fft.")

    fft = np.fft.fftshift(np.fft.fft2(image))
    mask = np.zeros(fft.shape)
    xy = np.mgrid[:mask.shape[0], : mask.shape[1]]
    xy = np.array([xy[1], xy[0]]).transpose((1, 2, 0))
    if (type(r) == int) | (type(r) == float):
        for xy_ in xy_fft:
            mask += np.where(norm(xy - xy_, axis=2) <= r, 1, 0)

    else:
        for i, xy_ in enumerate(xy_fft):
            mask += np.where(norm(xy - xy_, axis=2) <= r[i], 1, 0)

    amplitude = np.real(np.abs(np.fft.ifft2(np.fft.fftshift(fft * mask))))
    # amplitude = np.real(np.abs(image_complex))
    amplitude = image_norm(gaussian_filter(amplitude, sigma=blur))
    mask = np.where(amplitude > thresh, 1, 0)
    if fill_holes:
        mask = binary_fill_holes(mask)
    mask = np.where(mask, 1, 0)
    mask = binary_dilation(mask, iterations=buffer)

    return mask


def std_local(image, r):
    """Get local standard deviation of an image
    Parameters
    ----------
    image : ndarray of shape (h,w)
        The image.

    r : int
        Kernel radius. STD is calculated in a square kernel of size 2*r + 1.

    Returns
    -------
    std : 2D array
        The image local standard deviation of the image.
    """

    im = np.array(image, dtype=float)
    im2 = im**2
    ones = np.ones(im.shape)

    kernel = np.ones((2*r+1, 2*r+1))
    s = convolve2d(im, kernel, mode="same")
    s2 = convolve2d(im2, kernel, mode="same")
    ns = convolve2d(ones, kernel, mode="same")
    var = (s2/ns - (s/ns)**2)
    var = np.where(var < 0, 0, var)

    return var ** 0.5


def binary_find_largest_rectangle(array):
    """Gets the slice object of the largest rectangle of 1s in a 2D binary
    array. Modified version. Original by Andrew G. Clark

    Parameters
    ----------
    array : ndarray of shape (h,w)
        The binary image.

    Returns
    -------
    xlim : list-like of length 2
        The x limits (columns) of the largest rectangle.

    ylim : list-like of length 2
        The y limits (columns) of the largest rectangle.

    sl : numpy slice object
        The slice object which crops the image to the largest rectangle.


    The MIT License (MIT)

    Copyright (c) 2020 Andrew G. Clark

    Permission is hereby granted, free of charge, to any person obtaining a
    copy of this software and associated documentation files (the "Software"),
    to deal in the Software without restriction, including without limitation
    the rights to use, copy, modify, merge, publish, distribute, sublicense,
    and/or sell copies of the Software, and to permit persons to whom the
    Software is furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in
    all copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
    THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
    FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
    DEALINGS IN THE SOFTWARE."""

    # first get the sums of successive vertical pixels
    vert_sums = (np.zeros_like(array)).astype('float')
    vert_sums[0] = array[0]
    for i in range(1, len(array)):
        vert_sums[i] = (vert_sums[i-1] + array[i]) * array[i]

    # declare some variables for keeping track of the largest rectangle
    max_area = -1
    pos_at_max_area = (0, 0)
    height_at_max_area = -1
    x_end = 0

    # go through each row of vertical sums and find the largest rectangle
    for i in range(len(vert_sums)):
        positions = []  # a stack
        heights = []  # a stack
        for j in range(len(vert_sums[i])):
            h = vert_sums[i][j]
            if len(positions) == 0 or h > heights[-1]:
                heights.append(h)
                positions.append(j)
            elif h < heights[-1]:
                while len(heights) > 0 and h < heights[-1]:
                    h_tmp = heights.pop(-1)
                    pos_tmp = positions.pop(-1)
                    area_tmp = h_tmp * (j - pos_tmp)
                    if area_tmp > max_area:
                        max_area = area_tmp
                        # this is the bottom left
                        pos_at_max_area = (pos_tmp, i)
                        height_at_max_area = h_tmp
                        x_end = j
                heights.append(h)
                positions.append(pos_tmp)
        while len(heights) > 0:
            h_tmp = heights.pop(-1)
            pos_tmp = positions.pop(-1)
            area_tmp = h_tmp * (j - pos_tmp)
            if area_tmp > max_area:
                max_area = area_tmp
                pos_at_max_area = (pos_tmp, i)  # this is the bottom left
                height_at_max_area = h_tmp
                x_end = j

    top_left = (int(pos_at_max_area[0]), int(pos_at_max_area[1]
                                             - height_at_max_area) + 1)
    width = int(x_end - pos_at_max_area[0])
    height = int(height_at_max_area - 1)
    xlim = [top_left[0], top_left[0] + width]
    ylim = [top_left[1], top_left[1] + height]
    sl = np.s_[ylim[0]:ylim[1], xlim[0]:xlim[1]]

    return xlim, ylim, sl


def binary_find_smallest_rectangle(array):
    """
    Get the smallest rectangle (with horizontal and vertical sides) that
    contains an entire ROI defined by a binary array.

    This is useful for cropping to the smallest area without losing any useful
    data. Unless the region is already a rectangle with horiaontal and vertical
    sides, there will be remaining areas that are not part of the ROI. If only
    ROI area is desired in the final rectangle, use
    "binary_find_largest_rectangle."


    Parameters
    ----------
    array : ndarray of shape (h,w)
        The binary image.

    Returns
    -------
    xlim : list-like of length 2
        The x limits (columns) of the smallest rectangle.

    ylim : list-like of length 2
        The y limits (columns) of the smallest rectangle.

    sl : numpy slice object
        The slice object which crops the image to the smallest rectangle.
    """

    xinds = np.where(np.sum(array.astype(int), axis=1) > 0, 1, 0
                     ).reshape((-1, 1))
    yinds = np.where(np.sum(array.astype(int), axis=0) > 0, 1, 0
                     ).reshape((1, -1))

    newroi = (xinds @ yinds).astype(bool)

    xlim, ylim, sl = binary_find_largest_rectangle(newroi)

    return xlim, ylim, sl


def fft_square(image,
               hanning_window=False
               ):
    """Gets FFT with equal x & y pixel sizes

    Parameters
    ----------
    image : 2D array
        The image.

    hanning_window : bool
        Whether to apply a hanning window to the image before taking the FFT.
        Default: False

    Returns
    -------
    fft : ndarray
        FFT  of image after cropping to largest possible square image.

    """

    h, w = image.shape
    m = (min(h, w) // 2) * 2
    U = int(m/2)
    image_square = copy.deepcopy(image[int(h/2)-U: int(h/2)+U,
                                       int(w/2)-U: int(w/2)+U])
    if hanning_window:
        hann = np.outer(np.hanning(m), np.hanning(m))
        image_square *= hann

    fft = image_norm(np.abs(np.fft.fftshift(np.fft.fft2(image_square))))

    return fft


def get_fft_pixel_size(image, pixel_size):
    """Gets FFT pixel size for a real-space image.

    Parameters
    ----------
    image : 2D array
        The image. If not square, the smaller dimension will be used.

    pixel_size : scalar
        The pixel size of the real space image.

    Returns
    -------
    reciprocal_pixel_size : scalar
        Size of a pixel in an FFT of the image (assuming the image is first
        cropped to the largest square).

    """
    h, w = image.shape
    m = (min(h, w) // 2) * 2

    reciprocal_pixel_size = (pixel_size * m) ** -1

    return reciprocal_pixel_size

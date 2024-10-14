import os
import io
import json
from contextlib import redirect_stdout
import warnings

import numpy as np

import pandas as pd

import imageio
import hyperspy.api as hs
from h5py import File

import matplotlib.pyplot as plt

from ncempy.io.dm import dmReader
from ncempy.io.ser import serReader
from ncempy.io.emdVelox import fileEMDVelox
from ncempy.io.emd import emdReader

from empad2 import (
    load_calibration_data,
    load_background,
    load_dataset,
)

from SingleOrigin.system import select_file
from SingleOrigin.image import image_norm

pkg_dir, _ = os.path.split(__file__)

# %%

# TODO : update load_iamge to use my emdVelox readers


def load_image(
        path=None,
        display_image=True,
        images_from_stack='all',
        dsets_to_load='all',
        return_path=False,
        norm_image=True,
        full_metadata=False,
        bin_dims=None,
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

    dsets_to_load : str, or int, or list of strings
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

    bin_dims : tuple or list or None
        Binning factor for each dimension in the loaded dataset. If None, no
        binning applied.
        Default: None.

    Returns
    -------
    image : ndarray
        The imported image

    metadata : dict
        The metadata available in the original file

    """

    if path is None:
        path = select_file(
            message='Select an image to load...',
            ftypes=['.png', '.jpg', '.tif', '.dm4', '.dm3', '.emd', '.ser'],
        )

        print(f'path to imported image: {path}')

    elif path[-4:] in ['.dm4', '.dm3', '.emd', '.ser', '.tif', '.png', '.jpg']:
        pass

    else:
        path = select_file(
            message='Select a valid image to load...',
            ftypes=['.png', '.jpg', '.tif', '.dm4', '.dm3', '.emd', '.ser'],
        )

    # if ((type(dsets_to_load) is int) | (type(dsets_to_load) is str)) \
    #         & (dsets_to_load != 'all'):

    #     dsets_to_load = [dsets_to_load]

    if (type(dsets_to_load) is list) & (len(dsets_to_load) > 1) \
            & (path[-3:] != 'emd'):

        raise Exception(
            'Loading multiple datasets is only implimented for .emd files. '
            + 'Specify a single dataset for this file or pass None to load '
            + 'the default dataset.'
        )

    elements = pd.read_csv(
        os.path.join(pkg_dir, 'Element_table.txt')).sym.tolist()

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
        if type(emd) is not list:
            dsets = np.array([emd.metadata.General.title])

            if dsets_to_load == 'all':
                dsets_to_load = dsets

        else:
            dsets = np.array([emd[i].metadata.General.title
                              for i in range(len(emd))])

            print('Datasets found: ', ', '.join(dsets))

            # Get list of all datasets to load if "all" specified.
            # Remove EDS datasets because they cause problems.
            # This still allows loading of elemental maps from EDS
            if dsets_to_load == 'all':
                dsets_to_load = dsets[dsets != 'EDS']
                dsets_to_load_inds = [i for i, label in enumerate(dsets)
                                      if label != 'EDS']

            # Otherwise get list of requested datasets that are in the file
            else:
                num_requested = len(dsets_to_load)

                dsets_to_load = [dset_ for dset_ in dsets_to_load
                                 if dset_ in dsets]

                dsets_to_load_inds = [np.argwhere(dsets == dset_).item()
                                      for dset_ in dsets_to_load]

                num_avail = len(dsets_to_load)

                if len(dsets_to_load) == 0:
                    raise Exception('No matching dataset(s) found to load.')
                elif num_requested > num_avail:
                    print(
                        'Some datasets not available. Loading: ',
                        *dsets_to_load
                    )

        images = {}
        metadata = {}
        for i, dset_ in enumerate(dsets_to_load):
            if type(emd) is not list:
                images[dset_] = np.array(emd)
                dset_ind = 0
            else:
                dset_ind = dsets_to_load_inds[i]
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

                metadata[dset_] = {
                    key: val for key, val in emd_file.items() if key != 'data'
                }

            except IndexError as ie:
                raise ie
            except TypeError:

                try:
                    # Need to remove EDS datasets from the list and get the
                    # correct index as spectra are not seen by ncempy functions

                    emd_vel = fileEMDVelox(path)
                    if full_metadata is False:
                        _, metadata[dset_] = emd_vel.get_dataset(i)

                    elif full_metadata is True:
                        group = emd_vel.list_data[i]
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
                    warnings.warn(
                        'Missing metadata. Try "full_metadata=True" to see ' +
                        'available details.'
                    )
                    metadata[dset_] = {'pixelSize': [1, 1],
                                       'pixelUnit': ['pixel', 'pixel']}

        metadata['imageType'] = dsets[dset_ind]

    elif path[-3:] == 'ser':
        ser_file = serReader(path)
        images = {0: ser_file['data']}
        metadata = {
            0: {key: val for key, val in ser_file.items() if key != 'data'}
        }

    else:
        images = {'image': imageio.volread(path)}
        # metadata = {'image': images['image'].meta}
        metadata = {'image': None}

    for key in images.keys():
        h, w = images[key].shape[-2:]

        if ((images_from_stack == 'all') or (len(images[key].shape) <= 2)):
            pass
        elif (type(images_from_stack) is list
              or type(images_from_stack) is int):
            images[key] = images[key][images_from_stack, :, :]
        else:
            raise Exception('"images_from_stack" must be "all", an int, or '
                            + 'a list of ints.')

        # Norm the image(s), if specified, unless they are EDS spectra
        if norm_image and not any(key == el for el in elements):
            images[key] = image_norm(images[key])

        if len(images[key].shape) == 2:
            # images[key] = images[key][:int((h//2)*2), :int((w//2)*2)]
            image_ = images[key]

        if len(images[key].shape) == 3:
            # images[key] = images[key][:, :int((h//2)*2), :int((w//2)*2)]
            image_ = images[key][0, :, :]

        if display_image is True:
            fig, axs = plt.subplots()
            axs.imshow(image_, cmap='gray')
            axs.set_xticks([])
            axs.set_yticks([])

        if (type(bin_dims) in [tuple, list]):
            if (len(bin_dims) != len(images[key].shape)):
                raise Exception(
                    'If binning, must specify binning factor for each ' +
                    'dimension in the dataset and must be valid for all ' +
                    'loaded datasets.')

            dims = list(images[key].shape)
            new_dims = np.array([
                [dims[i]//bin_dims[i], bin_dims[i]] for i in range(len(dims))
            ]).ravel()

            images[key] = images[key].reshape(new_dims).mean(
                axis=tuple([i for i in range(1, len(new_dims), 2)]))

    # If only one image, extract from dictionaries
    if len(images) == 1:
        key = list(images.keys())[0]
        images = images[key]
        metadata = metadata[key]

    if return_path:
        return images, metadata, path
    else:
        return images, metadata


def emdVelox(path, sum_haadf_frames=True):
    """Read Velox emd files containing elemental maps and import as a dict.
    Works for newest version of Velox (early 2024); ncempy reader and hyperspy
    currently DO NOT work for this version.

    Parameters
    ----------
    path : str
         Path to the file.
    sum_haadf_frames : bool
        Whether to sum the stack of HAADF images acquired during the SI scan.
        Default: True

    Returns
    -------
    dstes : dict of numpy arrays
        Dictionary with label:image key:value pairs. Labels are 'HAADF' or
        element corresponding to each map.

    """

    f = File(path)

    filekeys = f.keys()
    dset_labels = {}

    # For SI datassets in Velox Version 10
    if 'Displays' in filekeys:

        imdisp = f['Displays']['ImageDisplay']

        for k in imdisp:
            d = eval(imdisp[k][0].decode().replace(r'\/', '/'))
            title = d['id']
            if ((title == 'ColorMix') | (title == 'CM')):
                continue

            refcode = d['data']

            datacodedict = eval(f[refcode][0].decode())

            dset_labels[title] = datacodedict['dataPath']

    # For regular images and SI Velox Version 11
    elif 'Presentation' in filekeys:
        imdisp = f['Presentation/Displays/ImageDisplay']

        for k in imdisp:
            d = eval(imdisp[k][0].decode().replace(r'\/', '/'))
            title = d['display']['label']
            if title == 'ColorMix':
                continue

            dset_labels[title] = d['dataPath'].replace('\\', '')

    dsets = {}
    for i, (k, v) in enumerate(dset_labels.items()):
        if sum_haadf_frames:
            # print(np.array(f[v]['Data'].dtype))
            dsets[k] = np.mean(np.array(f[v]['Data']), axis=2)
        else:
            dsets[k] = (np.array(f[v]['Data']).transpose((2, 0, 1))).squeeze()
        if i == 0:
            metadata = np.array(
                f[v]['Metadata'][:, 0]
            ).tobytes().decode().split('\x00')[0]

            metadata = json.loads(metadata)

    if 'Spectrum' in list(f['Data'].keys()):
        code = 'Data/Spectrum/' + list(f['Data/Spectrum'].keys())[0]

        counts = np.array(f[code]['Data']).squeeze()

        for k, v in metadata['Detectors'].items():
            if v['DetectorType'] == 'AnalyticalDetector':
                datastart = float(v['BeginEnergy'])  # Cut off cata below this
                binstart = float(v['OffsetEnergy'])
                binsize = float(v['Dispersion'])
                binedges = np.arange(
                    binstart,
                    binstart + binsize * (counts.shape[0] + 1),
                    binsize
                )
                bincenters = np.arange(
                    binstart + binsize/2,
                    binstart + binsize/2 + binsize * counts.shape[0],
                    binsize
                )
                break

        datastartind = np.argmin(np.abs(bincenters - datastart))

        counts[:datastartind] = 0

        dsets['Spectrum'] = {
            'Counts': counts,
            'eV_cent': bincenters,
            'eV_bins': binedges,
        }

    return dsets, metadata


# def emdVelox_v2(
#         path,
#         SI_sum_haadf_frames=True,
#         SI_sum_spectrum=False,
# ):
#     """Read Velox emd files containing elemental maps and import as a dict.
#     Works for newest version of Velox (as of early 2024).

#     Parameters
#     ----------
#     path : str
#          Path to the file.

#     SI_sum_haadf_frames : bool
#         Whether to sum the stack of HAADF images acquired during the SI scan.
#         Default: True

#     SI_sum_spectrum : bool
#         Whether to sum the EDS spectrum image data into a single spectrum. This
#         has no effect on previously generated elemental maps.

#     Returns
#     -------
#     dstes : dict of numpy arrays
#         Dictionary with label:image key:value pairs. Labels are 'HAADF' or
#         element corresponding to each map.

#     metadata : dict
#         The Velox metadata as a nested dictionary.

#     """

#     f = File(path)
#     filekeys = f.keys()

#     dsets = hs.load(path)
#     dsets = {signal.metadata.General.title: signal.data for signal in dsets}
#     dsets = {k.replace(' ()', ''): v for k, v in dsets.items()}

#     if 'DCFI' in dsets.keys():
#         dsets['DCFI'] = np.sum(dsets['DCFI'], axis=0)

#     # For SI datassets in Velox Version 10
#     if 'Displays' in filekeys:
#         imdisp = f['Displays/ImageDisplay']

#     # For regular images and SI Velox Version 11
#     elif 'Presentation' in filekeys:
#         imdisp = f['Presentation/Displays/ImageDisplay']

#     for k in imdisp:
#         d = eval(imdisp[k][0].decode().replace(r'\/', '/'))
#         title = d['display']['label']
#         if title == 'ColorMix':
#             continue

#         dset_label = d['dataPath'].replace('\\', '')
#         break

#     metadata = np.array(
#         f[dset_label]['Metadata'][:, 0]
#     ).tobytes().decode().split('\x00')[0]

#     metadata = json.loads(metadata)

#     if 'EDS' in dsets.keys():
#         for k, v in metadata['Detectors'].items():
#             if v['DetectorType'] == 'AnalyticalDetector':
#                 datastart = float(v['BeginEnergy'])  # Cut off cata below this
#                 binstart = float(v['OffsetEnergy'])
#                 binsize = float(v['Dispersion'])
#                 binedges = np.arange(
#                     binstart,
#                     binstart + binsize * (dsets['EDS'].shape[-1] + 1),
#                     binsize
#                 )
#                 bincenters = np.arange(
#                     binstart + binsize/2,
#                     binstart + binsize/2 + binsize * dsets['EDS'].shape[-1],
#                     binsize
#                 )
#                 break

#         datastartind = np.argmin(np.abs(bincenters - datastart))

#         dsets['EDS'][..., :datastartind] = 0

#         dsets['EDS_eV'] = {'eV_cent': bincenters, 'eV_bins': binedges}

#         if SI_sum_spectrum:
#             dsets['EDS'] = np.sum(dsets['EDS'], axis=(0, 1))

#     return dsets, metadata


def load_empad2_data(
    dataPath,
    bkgdPath,
    sensor='andromeda',
    scan_dims=None,
    bkgd_dims=None,
    bin_scan=None,
    bin_detector=None
):
    """
    Parameters
    ----------
    dataPath: str
        Path to EMPAD2 data file.
    bkgdPath: str
        Path to EMPAD2 background scan file.
    sensor: str
        The Cornell microscope where the data was collected.
        'andromeda' or 'cryo-titan'.
    scan_dims: 1- or 2-tuple or None
        The real space dimensions of the dataset. For time series without
        scanning, pass a 1-tuple of the number of frames. If None, assumes
        a square scan.
    bin_scan: 2-tuple, scalar or None
        The real-space binning factor to use.
    bin_detector: 2-tuple, scalar or None
        The reciprocal-space binning factor to use.

    Returns
    -------
    data : ndarray
        The empad data as a numpy array

    """

    if scan_dims is None:
        try:
            dims = str(dataPath.parts[-1])[:-4].split('_')[1:]

        except AttributeError:
            dims = dataPath.split('/')[-1][:-4].split('_')[1:]

        scan_dims = (int(dims[0][1:]), int(dims[1][1:]))

    cal = load_calibration_data(sensor)
    bkgd = load_background(bkgdPath, scan_size=bkgd_dims, calibration_data=cal)
    data = load_dataset(
        filepath=dataPath,
        background=bkgd,
        calibration_data=cal,
        scan_size=scan_dims,
    ).data

    # TODO : Use my data binning function
    if bin_scan is not None:
        if np.isin(type(bin_scan), [int, float]).item():
            bin_scan = (bin_scan,)

        dims = list(data.shape)
        new_dims = np.array([
            dims[0]/bin_scan[0], bin_scan[0],
            dims[1]/bin_scan[-1], bin_scan[-1],
            dims[2], dims[3]
        ]).astype(int)
        data = data.reshape(new_dims).mean(axis=(1, 3))

    if bin_detector is not None:
        if np.isin(type(bin_detector), [int, float]).item():
            bin_detector = (bin_detector, bin_detector)

        dims = list(data.shape)
        new_dims = np.array([
            dims[0], dims[1],
            dims[2]/bin_detector[0], bin_detector[0],
            dims[2]/bin_detector[-1], bin_detector[-1],
        ]).astype(int)
        data = data.reshape(new_dims).mean(axis=(3, 5))

    data = np.flip(data, axis=0)

    return data


def load_empad_data(
    dataPath,
    scan_dims=None,
    bin_scan=None,
    bin_detector=None
):
    """
    Parameters
    ----------
    dataPath: str
        Path to EMPAD2 data file.
    scan_dims: tuple or None
        The real space dimensions of the dataset. If None, assumes a square
        scan. If no scanning, a 1-tuple will arange the data in to an array
        of shape(n, 128, 128).
    bin_scan: 2-tuple, scalar or None
        The real-space binning factor to use.
    bin_detector: 2-tuple, scalar or None
        The reciprocal-space binning factor to use.

    Returns
    -------
    data : ndarray
        The empad data as a numpy array

    """

    with open(dataPath, 'rb') as file:
        data = np.fromfile(file, np.float32)

    num_frames = data.size/(128 * 130)

    if scan_dims is None:
        try:
            dims = str(dataPath.parts[-1])[:-4].split('_')[1:]
        except AttributeError:
            dims = dataPath.split('/')[-1][:-4].split('_')[1:]

        scan_dims = (int(dims[1][1:]), int(dims[0][1:]))

    data = np.reshape(
        data,
        (scan_dims[0], scan_dims[1], 130, 128),
        order='C'
    )[:, :, 0:128, :]

    # TODO : Use my data binning function
    if bin_scan is not None:
        if np.isin(type(bin_scan), [int, float]).item():
            bin_scan = (bin_scan,)

        dims = list(data.shape)
        new_dims = np.array([
            dims[0]/bin_scan[0], bin_scan[0],
            dims[1]/bin_scan[-1], bin_scan[-1],
            dims[2], dims[3]
        ]).astype(int)
        data = data.reshape(new_dims).mean(axis=(1, 3))

    if bin_detector is not None:
        if np.isin(type(bin_detector), [int, float]).item():
            bin_detector = (bin_detector, bin_detector)

        dims = list(data.shape)
        new_dims = np.array([
            dims[0], dims[1],
            dims[2]/bin_detector[0], bin_detector[0],
            dims[2]/bin_detector[-1], bin_detector[-1],
        ]).astype(int)
        data = data.reshape(new_dims).mean(axis=(3, 5))

    data = np.squeeze(data)

    minval = np.min(data)
    data = data - minval + 0.1

    return data

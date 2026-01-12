"""Module for reading various types of S/TEM data."""

import os
import io
import json
from contextlib import redirect_stdout
import warnings
from pathlib import Path

import numpy as np

import pandas as pd

import imageio
import hyperspy.api as hs
from h5py import File

from ncempy.io.dm import dmReader
from ncempy.io.ser import serReader
from ncempy.io.emdVelox import fileEMDVelox
from ncempy.io.emd import emdReader

from SingleOrigin.utils.system import select_file, check_package_installation
from SingleOrigin.utils.image import image_norm, bin_data
from SingleOrigin.utils.plot import quickplot

if check_package_installation('empad2'):
    from empad2 import (
        load_calibration_data,
        load_background,
        load_dataset,
    )

pkg_dir = Path(__file__).parents[1]

# %%

# TODO : update load_iamge to use my emdVelox readers


def load_image(
        path=None,
        plot=True,
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

    plot : bool
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

    return_path : bool
        Whether to return the path of the selected file.

    norm_image : bool
        Whether to norm the image contrast to the range 0 to 1.

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

    path : str
        Path of the selected file.

    """

    if isinstance(path, str):
        path = Path(path)

    if path is None:
        print('select the file')
        path = select_file(
            message='Select an image to load...',
            # ftypes=['.png', '.jpg', '.tif', '.dm4', '.dm3', '.emd', '.ser'],
        )

        print(f'path to imported image: {path}')

    elif path.parts[-1][-4:] in [
            '.dm4', '.dm3', '.emd', '.ser', '.tif', '.png', '.jpg']:
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
            & (path.parts[-1][-3:] != 'emd'):

        raise Exception(
            'Loading multiple datasets is only implimented for .emd files. '
            + 'Specify a single dataset for this file or pass None to load '
            + 'the default dataset.'
        )

    elements = pd.read_csv(
        os.path.join(pkg_dir, 'Element_table.txt')).sym.tolist()

    if path.parts[-1][-3:] in ['dm4', 'dm3']:

        dm_file = dmReader(path)
        images = {'image': dm_file['data']}
        metadata = {'image':
                    {key: val for key, val in dm_file.items() if key != 'data'}
                    }

    elif path.parts[-1][-3:] == 'emd':
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

    elif path.parts[-1][-3:] == 'ser':
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

        if plot is True:
            quickplot(image_, cmap='gray')

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


def emdVelox(
        path,
        sum_haadf_frames=True,
        load_SI=True,
        SIframe_range=[0, None],
):
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

    load_SI : bool
        For a SI dataset, whether to load the 3D SI data. Otherwise only loads
        elemental maps and selected spectra saved in the file by the Velox
        software.

    SIframe_range : 2-list
        The start and stop frames to integrate when loading the 3D SI dataset.
        To stop at the end, pass None as the second value.
        Default: [0, None]

    Returns
    -------
    dsets : dict of numpy arrays
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
            if "%" in title:
                continue

            dset_labels[title] = d['dataPath'].replace('\\', '')

    dsets = {}

    # check for multiple DCFI datasets
    dcfi = []
    for k in list(dset_labels.keys()):
        dcfi += ['DCFI' in k]
        # dset_labels[new_key] = dset_labels.pop(k)

    # Get number of original datasets
    if any(dcfi):
        original = ~np.array(dcfi)
        num_original = np.sum(original)

    for i, (k, v) in enumerate(dset_labels.items()):
        # Handle DCFI labeling
        if 'DCFI' in k:
            if num_original == 1:
                k = 'DCFI'
            elif num_original > 1:
                # Find matching original
                origkeys = [k for i, k in enumerate(dset_labels.keys())
                            if original[i]]
                match = [key in k for key in origkeys]
                origkey = origkeys[np.argwhere(match).item()]

                k = " ".join(['DCFI', origkey])

        if sum_haadf_frames:
            dsets[k] = np.mean(np.array(f[v]['Data']), axis=2)
        else:
            dsets[k] = (np.array(f[v]['Data']).transpose((2, 0, 1))).squeeze()

        if i == 0:
            metadata = np.array(
                f[v]['Metadata'][:, 0]
            ).tobytes().decode().split('\x00')[0]

            metadata = json.loads(metadata)
            try:
                metadata['pixelSize'] = \
                    float(metadata['BinaryResult']['PixelSize']['width'])
                metadata['pixelUnit'] = metadata['BinaryResult']['PixelUnitX']

                if metadata['pixelUnit'] == 'm':
                    if (metadata['pixelSize']
                            * int(metadata['Scan']['ScanSize']['width'])) > 5e-6:

                        metadata['pixelSize'] *= 1e6
                        metadata['pixelUnit'] = 'um'
                    else:
                        metadata['pixelSize'] *= 1e9
                        metadata['pixelUnit'] = 'nm'
            except KeyError:
                metadata['pixelSize'] = None
                metadata['pixelUnit'] = None

    if load_SI and 'SpectrumImage' in list(f['Data'].keys()):
        print('SI found.')

        if 'SpectrumImage' in list(f['Data'].keys()):
            # Get information for handling energy axis
            for k, v in metadata['Detectors'].items():
                if v['DetectorType'] == 'AnalyticalDetector':
                    # Cut off cata below this
                    datastart = float(v['BeginEnergy'])
                    binstart = float(v['OffsetEnergy'])
                    binsize = float(v['Dispersion'])
                    datastartind = int((datastart - binstart) // binsize)

                    break

        if 'Spectrum' in list(f['Data'].keys()):
            code = 'Data/Spectrum/' + list(f['Data/Spectrum'].keys())[0]

            counts = np.array(f[code]['Data']).squeeze()

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

            counts[:datastartind] = 0

            dsets['Spectrum'] = {
                'Counts': counts,
                'eV_cent': bincenters,
                'eV_bins': binedges,
                'BeginEnergy': datastart,
                'OffsetEnergy': binstart,
                'Binsize': binsize,
            }

        si = np.array(hs.load(
            path, select_type='spectrum_image', sum_frames=True,
            first_frame=SIframe_range[0],
            last_frame=SIframe_range[1]))
        # return si

        si[..., :datastartind] = 0

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

        dsets['SI'] = {
            'SI': si,
            'eV_cent': bincenters,
            'eV_bins': binedges,
            'BeginEnergy': datastart,
            'OffsetEnergy': binstart,
            'Binsize': binsize,
        }

    if load_SI and 'EelsSpectrumImage' in list(f['Data'].keys()):
        imdisp = f['Data']['EelsSpectrumImage']
        eels_labels = {}
        eels_segments = []
        eVs = []
        refkey = list(f['Features']['SIFeature'].keys())[0]

        xref = eval(f['Features']['SIFeature'][refkey][0].decode().replace(
            r'\/', '/'))['eels']['eelsDetectors']

        eels_labels = {}
        for d in xref:
            ind = d['index']
            refcode = d['spectrumImage']
            datacodedict = eval(f[refcode][0].decode())
            eels_labels[ind] = datacodedict['dataPath']

        for j, key in eels_labels.items():
            if j == 0:
                eelsmeta = np.array(
                    imdisp[key]['Metadata'][:, 0]
                ).tobytes().decode().split('\x00')[0]
                eelsmeta = json.loads(eelsmeta)

                dispersion = float(eelsmeta[
                    'CustomProperties'
                ]['AnalyticalDetector[EELS].SpectrumDispersion[0]']['value'])
                offsets = []
                for i in range(5):
                    strip = eelsmeta[
                        'CustomProperties'
                    ][f'AnalyticalDetector[EELS].SpectrumOffsetEnergy[{i}]']
                    offsets += [float(strip['value'])]

                dwelltimes = []
                for i in range(5):
                    strip = eelsmeta[
                        'CustomProperties'
                    ][f'AnalyticalDetector[EELS].DetectorExposureTime[{i}]']
                    dwelltimes += [float(strip['value'])]

                eVs = np.array([np.arange(2048) * dispersion + offsets[i]
                                for i in range(5)
                                ])

            eels_segments += [np.array(f[key]['Data']).transpose((2, 0, 1))]

        dsets['EELS_SI'] = {
            'SI': np.array(eels_segments),
            'eV': eVs,
            'DwellTimes': dwelltimes,
            # 'eV_bins': binedges,
            # 'BeginEnergy': datastart,
            # 'OffsetEnergy': binstart,
            'BinSize': dispersion,
        }
        metadata['eels'] = eelsmeta

    return dsets, metadata


def load_empad2_data(
    dataPath,
    bkgdPath,
    sensor='andromeda',
    bin_scan=None,
    bin_detector=None,
):
    """
    Load an EMPAD2 dataset. This function wraps functions from the empad2
    Python module, automatically detecting the scan size.

    Parameters
    ----------
    dataPath: str
        Path to EMPAD2 data file.

    bkgdPath: str
        Path to EMPAD2 background scan file.

    sensor: str
        The Cornell microscope where the data was collected.
        'andromeda' or 'cryo-titan'.

    bin_scan: 2-tuple, scalar or None
        The real-space binning factor to use.

    bin_detector: 2-tuple, scalar or None
        The reciprocal-space binning factor to use.

    Returns
    -------
    data : ndarray
        The EMPAD2 data as a numpy array.

    """

    if not check_package_installation('empad2'):
        raise Exception(
            'Please install emapd2 module to use this function.'
        )

    # Get the dataset scan dims:
    try:
        dims = str(dataPath.parts[-1])[:-4].split('_')[1:]

    except AttributeError:
        dims = dataPath.split('/')[-1][:-4].split('_')[1:]

    scan_dims = tuple([int(dims[i][1:]) for i in range(len(dims))])
    if len(scan_dims) == 1:
        scan_dims = scan_dims + (1,)

    # Get the background scan dims:
    try:
        dims = str(bkgdPath.parts[-1])[:-4].split('_')[1:]

    except AttributeError:
        dims = bkgdPath.split('/')[-1][:-4].split('_')[1:]

    bkgd_dims = tuple([int(dims[i][1:]) for i in range(len(dims))])
    if len(bkgd_dims) == 1:
        bkgd_dims = bkgd_dims + (1,)

    cal = load_calibration_data(sensor)
    bkgd = load_background(bkgdPath, scan_size=bkgd_dims, calibration_data=cal)
    data = load_dataset(
        filepath=dataPath,
        background=bkgd,
        calibration_data=cal,
        scan_size=scan_dims,
    ).data

    if bin_scan is None:
        bin_scan = np.array([1, 1])
    if bin_detector is None:
        bin_detector = np.array([1, 1])
    binning = np.concatenate([bin_scan, bin_detector])

    if np.max(binning) > 1:
        data = bin_data(data, binning)

    data = np.flip(data, axis=0)

    return data


def load_empad_data(
    dataPath,
    bin_scan=None,
    bin_detector=None,
    swapped_scan_axes=False,
):
    """
    Parameters
    ----------
    dataPath: str
        Path to EMPAD data file.

    bin_scan: 2-list, scalar or None
        The real-space binning factor to use.

    bin_detector: 2-list, scalar or None
        The reciprocal-space binning factor to use.

    swapped_scan_axes : bool
        If true, switches the x & y scan dimensions read from the file name
        which is used to correctly shape the scan when loading the dataset.
        Only needed for non-square scans. If is  not shapped correctly,
        switch this variable.
        Default: False

    Returns
    -------
    data : ndarray
        The empad data as a numpy array

    """

    with open(dataPath, 'rb') as file:
        data = np.fromfile(file, np.float32)

    # Get the dataset scan dims:
    try:
        dims = str(dataPath.parts[-1])[:-4].split('_')[1:]

    except AttributeError:
        dims = dataPath.split('/')[-1][:-4].split('_')[1:]

    scan_dims = tuple([int(dims[i][1:]) for i in range(len(dims))])
    if not swapped_scan_axes:
        scan_dims = tuple(np.flip(scan_dims))

    print(scan_dims)
    if len(scan_dims) == 1:
        scan_dims = scan_dims + (1,)

    data = np.reshape(
        data,
        (*scan_dims, 130, 128),
        order='C'
    )[..., 0:128, :]

    if bin_scan is None:
        bin_scan = np.array([1, 1])
    if bin_detector is None:
        bin_detector = np.array([1, 1])
    binning = np.concatenate([bin_scan, bin_detector])

    if np.max(binning) > 1:
        data = bin_data(data, binning)

    return data


def load_empad_h5(path):
    """
    Parameters
    ----------
    path: str
        Path to EMPAD data file.

    Returns
    -------
    data : ndarray
        The empad data as a numpy array.

    """
    file = File(path, "r")

    data = np.array(file['datacube_root']['datacube']['data'])

    return data

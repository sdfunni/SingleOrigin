import os
import sys
import importlib

from pathlib import Path

from PyQt5.QtWidgets import QApplication, QWidget, QFileDialog
from PyQt5.QtWidgets import QFileDialog as qfd

import h5py

# %%
"""Directory and file functions"""


def select_folder():
    """Select a folder in dialog box and return path

    Parameters
    ----------
    None.

    Returns
    -------
    path : str
        The path to the selected folder.

    """

    print('Select folder...')
    app = QApplication(sys.argv)
    widget = QWidget()
    path = Path(qfd.getExistingDirectory(widget))

    return path


def select_file(folder_path=None, message=None, ftypes=None):
    """Select a folder in dialog box and return path

    Parameters
    ----------
    folder_path : str or None
        The path to the desired folder in which you want to select a file.
        If None, dialog will open in the current working directory and user
        must navigate to the desired folder.

    ftypes : str, list of strings or None
        File type(s) to show. Other file types will be hidden. If None, all
        types will be shown.

    Returns
    -------
    path : str
        The path of the selected file.

    """

    cwd = os.getcwd()
    if folder_path is not None:
        os.chdir(folder_path)

    if type(message) is not str:
        print('Select file...')
    else:
        print(message)

    if ftypes is not None:
        if type(ftypes) is str:
            ftypes = [ftypes]
        ftypes = ['*' + ftype for ftype in ftypes]
        ftypes = f'({", ".join(ftypes)})'

    app = QApplication(sys.argv)
    widget = QWidget()
    path = Path(qfd.getOpenFileName(
        widget,
        filter=ftypes,
    )[0])

    os.chdir(cwd)

    return path


def show_h5_tree(path):
    """Print full tree structure of an h5 file to the console.

    Parameters
    ----------
    path : str
        The path to the h5-type file. It does not necessarily need to have a
        .h5 file extension so long as it is readable by h5py.File().

    Returns
    -------
    None.


    """

    def h5_tree(val, pre='', out=""):
        length = len(val)
        for key, val in val.items():
            length -= 1
            if length == 0:  # the last item
                if type(val) == h5py._hl.group.Group:
                    out += pre + '└── ' + key + "\n"
                    out = h5_tree(val, pre+'    ', out)
                else:
                    out += pre + '└── ' + key + f' {val.shape}\n'
            else:
                if type(val) == h5py._hl.group.Group:
                    out += pre + '├── ' + key + "\n"
                    out = h5_tree(val, pre+'│   ', out)
                else:
                    out += pre + '├── ' + key + f' {val.shape}\n'
        return out

    file = h5py.File(path, "r")
    structure = h5_tree(file)
    print(structure)


def str_to_dict(s):
    """Convert a dictionary-like string to an actual dictionary

    Parameters
    ----------
    s : str
         The dictionary in string form.

    Returns
    -------
    d : dict
        The string converted to a Python dictionary.

    """

    s = s[1:-1]
    kvs = s.split(', ')
    kvs = [kv.split(': ') for kv in kvs]
    d = {k: v for [k, v] in kvs}

    return d


def check_package_installation(name):

    try:
        importlib.import_module(name)
        result = True
    except ImportError:
        result = False

    return result

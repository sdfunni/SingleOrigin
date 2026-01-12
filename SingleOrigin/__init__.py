"""
SingleOrigin is a module for analysis of multimodal scanning transmission
electron microscope data.
Copyright (C) 2025  Stephen D. Funni

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see https://www.gnu.org/licenses
"""


__version__ = "3.0.0"

# General modules
from SingleOrigin.utils.system import *
from SingleOrigin.utils.read import *
from SingleOrigin.utils.crystalmath import *
from SingleOrigin.utils.mathfn import *
from SingleOrigin.utils.image import *
from SingleOrigin.utils.fourier import *
from SingleOrigin.utils.lattice import *
from SingleOrigin.utils.peakfit import *
from SingleOrigin.utils.plot import *
from SingleOrigin.utils.ndstem import *

from SingleOrigin.cell.cell_transform import UnitCell

from SingleOrigin.hrimage.find_atom_columns import (
    HRImage,
    AtomicColumnLattice
)
from SingleOrigin.hrimage.pcf import *
from SingleOrigin.hrimage.gpa import *

from SingleOrigin.measure_lattice import ReciprocalLattice

from SingleOrigin.eels.utils import *
from SingleOrigin.eels.eels import *

from SingleOrigin.eds import *

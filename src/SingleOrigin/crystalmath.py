import numpy as np
from numpy import exp, sin, cos

# %%


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
        [[a**2, a*b*cos(gamma), a*c*cos(beta)],
         [b*a*cos(gamma), b**2, b*c*cos(alpha)],
         [c*a*cos(beta), c*b*cos(alpha), c**2]]
    )

    g[np.abs(g) <= 1e-10] = 0

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
        Inter-planar spacing in Angstroms

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
        Inter-planar angle in degrees

    """

    p_q = np.array([hkl_1, hkl_2])
    [[pp, pq], [qp, qq]] = np.array(p_q @ np.linalg.inv(g) @ p_q.T)

    theta = np.degrees(np.arccos(
        np.round(pq/(pp**0.5 * qq**0.5),
                 decimals=10)
    ))

    return theta


def get_astar_2d_matrix(g1, g2, g):
    """
    Get the a reciprocal basis matrix for two g_hkl vectors

    Parameters
    ----------
    g1, g2 : array_like of ints of shape (1,3) or (3,)
        Miller indices of the lattice planes
    g : 3x3 ndarray
        The metric tensor

    Returns
    -------
    a_star_2d : 2x2 array
        The projected 2d reciprocal basis vector matrix (as row vectors).

    """

    a1_star = 1/IntPlSpc(g1, g)
    a2_star = 1/IntPlSpc(g2, g)
    alpha_star = np.radians(IntPlAng(g1, g2, g))

    a_star_2d = np.array([
        [a1_star, 0],
        [a2_star*cos(alpha_star), a2_star*sin(alpha_star)]
    ])
    a_star_2d[np.abs(a_star_2d) <= 1e-10] = 0

    return a_star_2d


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


def elec_wavelength(V):
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

    m_0 = 9.109e-31  # electron rest mass (kg)
    e = 1.602e-19  # elementary charge (C)
    c = 2.997e8  # speed of light (m/s)
    h = 6.626e-34  # Plank's constant (Nms)

    wavelength = h/(2*m_0*e*V*(1+e*V/(2*m_0*c**2)))**.5

    return wavelength


def elec_velocity(V):
    """Electron veolcity as a function of accelerating voltage

    Parameters
    ----------
    V : int or float
         Accelerating voltage in V.

    Returns
    -------
    wavelength : float
        Electron veolcity in m/s.

    """

    m_0 = 9.109e-31  # electron rest mass (kg)
    e = 1.602e-19  # elementary charge (C)
    c = 2.997e8  # speed of light (m/s)

    v = c * (1 - 1/(1+e*V/(m_0*c**2))**2)**0.5

    return v

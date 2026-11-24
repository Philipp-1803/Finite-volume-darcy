"""Cell-centered TPFA for Darcy flow on a uniform cubic grid."""

from dataclasses import dataclass

import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve


@dataclass
class Data:
    N: int = 4
    a: float = 0.0
    b: float = 1.0
    K: float = 1.0
    p_left: float = 1.0
    p_right: float = 0.0

    u_down: float = 2.0
    u_up: float = 0.0


def source(x, y, z):
    return 8 * np.sin(2 * np.pi * x) * np.sin(2 * np.pi * z)


def assemble_system(data, h, t_interior, t_boundary):
    n = data.N
    A = lil_matrix((n**3, n**3), dtype=float)
    b_vec = np.zeros(n**3)
    centers = data.a + (np.arange(n) + 0.5) * h
    directions = ((0, 0, -1), (0, 0, 1), (0, -1, 0), (0, 1, 0), (-1, 0, 0), (1, 0, 0))

    for cell in range(n**3):
        iz, remainder = divmod(cell, n**2)
        iy, ix = divmod(remainder, n)
        b_vec[cell] = h**3 * source(centers[ix], centers[iy], centers[iz])

        for dz, dy, dx in directions:
            jz, jy, jx = iz + dz, iy + dy, ix + dx
            if 0 <= jz < n and 0 <= jy < n and 0 <= jx < n:
                neighbor = (jz * n + jy) * n + jx
                A[cell, cell] += t_interior
                A[cell, neighbor] -= t_interior
            elif dx:
                pressure = data.p_left if dx < 0 else data.p_right
                A[cell, cell] += t_boundary
                b_vec[cell] += t_boundary * pressure
            elif dy:
                outward_flux = data.u_down if dy < 0 else data.u_up
                b_vec[cell] -= h**2 * outward_flux

    return A.tocsr(), b_vec


def compute_fluxes(data, h, pressuregrid):

    n = data.N
    fluxgrid_x = np.zeros((n, n, n + 1))
    fluxgrid_y = np.zeros((n, n + 1, n))
    fluxgrid_z = np.zeros((n + 1, n, n))
    scale = data.K / h

    fluxgrid_x[:, :, 0] = 2 * scale * (data.p_left - pressuregrid[:, :, 0])
    fluxgrid_x[:, :, 1:-1] = scale * (pressuregrid[:, :, :-1] - pressuregrid[:, :, 1:])
    fluxgrid_x[:, :, -1] = 2 * scale * (pressuregrid[:, :, -1] - data.p_right)

    fluxgrid_y[:, 0, :] = -data.u_down
    fluxgrid_y[:, 1:-1, :] = scale * (pressuregrid[:, :-1, :] - pressuregrid[:, 1:, :])
    fluxgrid_y[:, -1, :] = data.u_up

    fluxgrid_z[1:-1, :, :] = scale * (pressuregrid[:-1, :, :] - pressuregrid[1:, :, :])
    return fluxgrid_x, fluxgrid_y, fluxgrid_z


def solve_3d(data):
    h = (data.b - data.a) / data.N
    t_interior = data.K * h

    A, b_vec = assemble_system(data, h, t_interior, 2 * t_interior)
    pressuregrid = spsolve(A, b_vec).reshape(data.N, data.N, data.N)
    fluxgrid_x, fluxgrid_y, fluxgrid_z = compute_fluxes(data, h, pressuregrid)

    return pressuregrid, fluxgrid_x, fluxgrid_y, fluxgrid_z

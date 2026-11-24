import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve

from dataclasses import dataclass


@dataclass
class Data:
    N: int = 4
    a: float = 0.0
    b: float = 1.0
    K: float = 1.0
    p_left: float = 1.0
    p_right: float = 0.0


directions = {"left": -1, "right": 1}


def assemble_system(data, t_interior, t_boundary):
    boundary_pressure = {"left": data.p_left, "right": data.p_right}
    A = lil_matrix((data.N, data.N), dtype=float)
    b_vec = np.zeros(data.N)

    for ix in range(data.N):
        for direction, dix in directions.items():
            ix_neighbor = ix + dix

            if 0 <= ix_neighbor < data.N:
                A[ix, ix] += t_interior
                A[ix, ix_neighbor] -= t_interior
            else:
                A[ix, ix] += t_boundary
                b_vec[ix] += t_boundary * boundary_pressure[direction]

    return A.tocsr(), b_vec


def compute_fluxes(data, pressure, t_interior, t_boundary):
    fluxgrid_x = np.zeros(data.N + 1)

    fluxgrid_x[0] = t_boundary * (data.p_left - pressure[0])
    fluxgrid_x[data.N] = t_boundary * (pressure[-1] - data.p_right)

    fluxgrid_x[1:-1] = t_interior * (
        pressure[:-1] - pressure[1:]
    )  # start bei index -1, also alles bis auf den letzten eintrag minus alles ab dem ersten Index

    return fluxgrid_x


def solve_1d(data):
    h = (data.b - data.a) / data.N
    t_interior = data.K / h
    t_boundary = 2 * data.K / h

    A, b_vec = assemble_system(data, t_interior, t_boundary)

    pressure = spsolve(A, b_vec)

    fluxgrid_x = compute_fluxes(data, pressure, t_interior, t_boundary)

    return pressure, fluxgrid_x

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


def assemble_mixed_system(data, t_interior, t_boundary):
    number_fluxes = data.N + 1
    number_unknowns = number_fluxes + data.N
    A = lil_matrix((number_unknowns, number_unknowns), dtype=float)
    b_vec = np.zeros(number_unknowns)

    # left
    A[0, 0] = 1.0
    A[0, number_fluxes] = t_boundary
    b_vec[0] = t_boundary * data.p_left

    # middle
    for ix in range(1, data.N):
        A[ix, ix] = 1.0
        A[ix, number_fluxes + ix - 1] = -t_interior
        A[ix, number_fluxes + ix] = t_interior

    # right
    A[data.N, data.N] = 1.0
    A[data.N, number_fluxes + data.N - 1] = -t_boundary
    b_vec[data.N] = -t_boundary * data.p_right

    # Condition div(u)=0
    for ix in range(data.N):
        row = number_fluxes + ix
        A[row, ix] = -1.0
        A[row, ix + 1] = 1.0

    return A.tocsr(), b_vec


def solve_1d_without_postprocessing(data):
    h = (data.b - data.a) / data.N
    t_interior = data.K / h
    t_boundary = 2 * data.K / h

    A, b_vec = assemble_mixed_system(data, t_interior, t_boundary)
    w = spsolve(A, b_vec)

    number_fluxes = data.N + 1
    fluxgrid_x = w[:number_fluxes]
    pressure = w[number_fluxes:]
    return pressure, fluxgrid_x

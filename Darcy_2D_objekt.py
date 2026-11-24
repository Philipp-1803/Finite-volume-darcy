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
    u_down: float = 2.0
    u_up: float = 0.0


directions = {
    "left": (0, -1),
    "right": (0, 1),
    "down": (-1, 0),
    "up": (1, 0),
}


def assemble_system(data, h, t_interior, t_boundary):
    boundarypressure = {"left": data.p_left, "right": data.p_right}
    boundaryflux = {"down": data.u_down, "up": data.u_up}
    number_cells = data.N**2
    A = lil_matrix((number_cells, number_cells), dtype=float)
    b_vec = np.zeros(number_cells)
    punkte = np.zeros((number_cells, 2))

    for k in range(number_cells):
        j, i = divmod(k, data.N)
        punkte[k] = (data.a + (i + 0.5) * h, data.a + (j + 0.5) * h)

        for direction, (dj, di) in directions.items():
            j_neighbor = j + dj
            i_neighbor = i + di

            if 0 <= j_neighbor < data.N and 0 <= i_neighbor < data.N:
                neighbor = j_neighbor * data.N + i_neighbor
                A[k, k] += t_interior
                A[k, neighbor] -= t_interior

            elif direction in boundarypressure:
                A[k, k] += t_boundary
                b_vec[k] += t_boundary * boundarypressure[direction]

            elif direction in boundaryflux:
                if direction == "down":
                    b_vec[k] -= h * boundaryflux[direction]
                if direction == "up":
                    b_vec[k] += 0

    return A.tocsr(), b_vec


def compute_fluxes(data, h, pressuregrid):
    fluxgrid_x = np.zeros((data.N, data.N + 1))
    fluxgrid_y = np.zeros((data.N + 1, data.N))

    fluxgrid_x[:, 0] = 2 * data.K / h * (data.p_left - pressuregrid[:, 0])
    fluxgrid_x[:, 1:-1] = data.K / h * (pressuregrid[:, :-1] - pressuregrid[:, 1:])
    fluxgrid_x[:, -1] = 2 * data.K / h * (pressuregrid[:, -1] - data.p_right)

    fluxgrid_y[0, :] = -data.u_down
    fluxgrid_y[1:-1, :] = data.K / h * (pressuregrid[:-1, :] - pressuregrid[1:, :])
    fluxgrid_y[-1, :] = data.u_up

    return fluxgrid_x, fluxgrid_y


def solve_2d(data):
    h = (data.b - data.a) / data.N
    t_interior = data.K
    t_boundary = 2 * data.K

    A, b_vec = assemble_system(data, h, t_interior, t_boundary)

    pressure = spsolve(A, b_vec)
    pressuregrid = pressure.reshape(data.N, data.N)

    fluxgrid_x, fluxgrid_y = compute_fluxes(data, h, pressuregrid)

    return pressuregrid, fluxgrid_x, fluxgrid_y

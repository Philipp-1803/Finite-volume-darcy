import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve

from dataclasses import dataclass

directions = {
    "left": (0, -1),
    "right": (0, 1),
    "down": (-1, 0),
    "up": (1, 0),
}


@dataclass
class Darcy2D:
    N: int = 4
    a: float = 0.0
    b: float = 1.0
    K: float = 1.0
    p_left: float = 1.0
    p_right: float = 0.0
    u_down: float = 2.0
    u_up: float = 0.0

    def transmissibilities(self):
        h = (self.b - self.a) / self.N
        t_interior = self.K / h
        t_boundary = 2 * self.K / h
        return h, t_interior, t_boundary

    def assemble_system(self, h, t_interior, t_boundary):
        boundarypressure = {"left": self.p_left, "right": self.p_right}
        boundaryflux = {"down": self.u_down, "up": self.u_up}
        number_cells = self.N**2
        A = lil_matrix((number_cells, number_cells), dtype=float)
        b_vec = np.zeros(number_cells)
        punkte = np.zeros((number_cells, 2))

        for k in range(number_cells):
            j, i = divmod(k, self.N)
            punkte[k] = (self.a + (i + 0.5) * h, self.a + (j + 0.5) * h)

            for direction, (dj, di) in directions.items():
                j_neighbor = j + dj
                i_neighbor = i + di

                if 0 <= j_neighbor < self.N and 0 <= i_neighbor < self.N:
                    neighbor = j_neighbor * self.N + i_neighbor
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

    def compute_fluxes(self, h, pressuregrid):
        fluxgrid_x = np.zeros((self.N, self.N + 1))
        fluxgrid_y = np.zeros((self.N + 1, self.N))

        fluxgrid_x[:, 0] = 2 * self.K / h * (self.p_left - pressuregrid[:, 0])
        fluxgrid_x[:, 1:-1] = self.K / h * (pressuregrid[:, :-1] - pressuregrid[:, 1:])
        fluxgrid_x[:, -1] = 2 * self.K / h * (pressuregrid[:, -1] - self.p_right)

        fluxgrid_y[0, :] = -self.u_down
        fluxgrid_y[1:-1, :] = self.K / h * (pressuregrid[:-1, :] - pressuregrid[1:, :])
        fluxgrid_y[-1, :] = self.u_up

        return fluxgrid_x, fluxgrid_y

    def solve_2d(self):
        h, t_interior, t_boundary = self.transmissibilities()

        A, b_vec = self.assemble_system(h, t_interior, t_boundary)

        pressure = spsolve(A, b_vec)
        pressuregrid = pressure.reshape(self.N, self.N)

        fluxgrid_x, fluxgrid_y = self.compute_fluxes(h, pressuregrid)

        return pressuregrid, fluxgrid_x, fluxgrid_y


if __name__ == "__main__":
    problem = Darcy2D(N=4)
    pressure, fluxgrid_x, fluxgrid_y = problem.solve_2d()

    print("Pressure:", pressure)
    print("Flux in x-Direction:", fluxgrid_x)
    print("Flux in y-Direction:", fluxgrid_y)

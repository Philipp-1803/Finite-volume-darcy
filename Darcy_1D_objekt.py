import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve

from dataclasses import dataclass

directions = {"left": -1, "right": 1}


@dataclass
class Darcy1D:
    N: int = 4
    a: float = 0.0
    b: float = 1.0
    K: float = 1.0
    p_left: float = 1.0
    p_right: float = 0.0

    def transmissibilities(self):
        h = (self.b - self.a) / self.N
        t_interior = self.K / h
        t_boundary = 2 * self.K / h
        return t_interior, t_boundary

    def assemble_system(self, t_interior, t_boundary):
        boundary_pressure = {"left": self.p_left, "right": self.p_right}
        A = lil_matrix((self.N, self.N), dtype=float)
        b_vec = np.zeros(self.N)

        for ix in range(self.N):
            for direction, dix in directions.items():
                ix_neighbor = ix + dix

                if 0 <= ix_neighbor < self.N:
                    A[ix, ix] += t_interior
                    A[ix, ix_neighbor] -= t_interior
                else:
                    A[ix, ix] += t_boundary
                    b_vec[ix] += t_boundary * boundary_pressure[direction]

        return A.tocsr(), b_vec

    def compute_fluxes(self, pressure, t_interior, t_boundary):
        fluxgrid_x = np.zeros(self.N + 1)

        fluxgrid_x[0] = t_boundary * (self.p_left - pressure[0])
        fluxgrid_x[self.N] = t_boundary * (pressure[-1] - self.p_right)

        fluxgrid_x[1:-1] = t_interior * (
            pressure[:-1] - pressure[1:]
        )  # start bei index -1, also alles bis auf den letzten eintrag minus alles ab dem ersten Index

        return fluxgrid_x

    def solve_1d(self):
        t_interior, t_boundary = self.transmissibilities()

        A, b_vec = self.assemble_system(t_interior, t_boundary)

        pressure = spsolve(A, b_vec)

        fluxgrid_x = self.compute_fluxes(pressure, t_interior, t_boundary)

        return pressure, fluxgrid_x


if __name__ == "__main__":
    problem = Darcy1D(N=4)
    pressure, fluxgrid_x = problem.solve_1d()

    print("Pressure:", pressure)
    print("Flux:", fluxgrid_x)

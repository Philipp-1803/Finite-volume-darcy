"""Cell-centered TPFA for Darcy flow on a uniform cubic grid."""

from dataclasses import dataclass

import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve


@dataclass
class Darcy3D:
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
        t_interior = self.K * h
        return h, t_interior

    def source(self, x, y, z):
        return 8 * np.sin(2 * np.pi * x) * np.sin(2 * np.pi * z)

    def assemble_system(self, h, t_interior, t_boundary):
        n = self.N
        A = lil_matrix((n**3, n**3), dtype=float)
        b_vec = np.zeros(n**3)
        centers = self.a + (np.arange(n) + 0.5) * h
        directions = (
            (0, 0, -1),
            (0, 0, 1),
            (0, -1, 0),
            (0, 1, 0),
            (-1, 0, 0),
            (1, 0, 0),
        )

        for cell in range(n**3):
            iz, remainder = divmod(cell, n**2)
            iy, ix = divmod(remainder, n)
            b_vec[cell] = h**3 * self.source(centers[ix], centers[iy], centers[iz])

            for dz, dy, dx in directions:
                jz, jy, jx = iz + dz, iy + dy, ix + dx
                if 0 <= jz < n and 0 <= jy < n and 0 <= jx < n:
                    neighbor = (jz * n + jy) * n + jx
                    A[cell, cell] += t_interior
                    A[cell, neighbor] -= t_interior
                elif dx:
                    pressure = self.p_left if dx < 0 else self.p_right
                    A[cell, cell] += t_boundary
                    b_vec[cell] += t_boundary * pressure
                elif dy:
                    outward_flux = self.u_down if dy < 0 else self.u_up
                    b_vec[cell] -= h**2 * outward_flux

        return A.tocsr(), b_vec

    def compute_fluxes(self, h, pressuregrid):
        n = self.N
        fluxgrid_x = np.zeros((n, n, n + 1))
        fluxgrid_y = np.zeros((n, n + 1, n))
        fluxgrid_z = np.zeros((n + 1, n, n))
        scale = self.K / h

        fluxgrid_x[:, :, 0] = 2 * scale * (self.p_left - pressuregrid[:, :, 0])
        fluxgrid_x[:, :, 1:-1] = scale * (
            pressuregrid[:, :, :-1] - pressuregrid[:, :, 1:]
        )
        fluxgrid_x[:, :, -1] = 2 * scale * (pressuregrid[:, :, -1] - self.p_right)

        fluxgrid_y[:, 0, :] = -self.u_down
        fluxgrid_y[:, 1:-1, :] = scale * (
            pressuregrid[:, :-1, :] - pressuregrid[:, 1:, :]
        )
        fluxgrid_y[:, -1, :] = self.u_up

        fluxgrid_z[1:-1, :, :] = scale * (
            pressuregrid[:-1, :, :] - pressuregrid[1:, :, :]
        )

        return fluxgrid_x, fluxgrid_y, fluxgrid_z

    def solve_3d(self):
        h, t_interior = self.transmissibilities()

        A, b_vec = self.assemble_system(h, t_interior, 2 * t_interior)
        pressuregrid = spsolve(A, b_vec).reshape(self.N, self.N, self.N)
        fluxgrid_x, fluxgrid_y, fluxgrid_z = self.compute_fluxes(h, pressuregrid)

        return pressuregrid, fluxgrid_x, fluxgrid_y, fluxgrid_z


if __name__ == "__main__":
    problem = Darcy3D(N=4)
    pressure, fluxgrid_x, fluxgrid_y, fluxgrid_z = problem.solve_3d()

    print("Pressure:", pressure)
    print("Flux in x-Direction:", fluxgrid_x)
    print("Flux in y-Direction:", fluxgrid_y)
    print("Flux in z-Direction:", fluxgrid_z)

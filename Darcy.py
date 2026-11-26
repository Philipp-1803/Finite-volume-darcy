import Darcy_1D_objekt
import Darcy_2D_objekt
import Darcy_3D_objekt
import Darcy_1D_objekt_without_postprocessing


def experiment_1d():
    problem = Darcy_1D_objekt.Darcy1D(N=4)
    pressure, fluxgrid_x = problem.solve_1d()

    print("Pressure", pressure)
    print("Flux", fluxgrid_x)


def experiment_2d():
    problem = Darcy_2D_objekt.Darcy2D(N=4)
    pressuregrid, fluxgrid_x, fluxgrid_y = problem.solve_2d()

    print("Pressure", pressuregrid)
    print("Flux_x", fluxgrid_x)
    print("Flux_y", fluxgrid_y)


def experiment_3d():
    problem = Darcy_3D_objekt.Darcy3D(N=4)
    pressuregrid, fluxgrid_x, fluxgrid_y, fluxgrid_z = problem.solve_3d()

    print("Pressure", pressuregrid)
    print("Flux_x", fluxgrid_x)
    print("Flux_y", fluxgrid_y)
    print("Flux_z", fluxgrid_z)


if __name__ == "__main__":
    experiment_1d()
    experiment_2d()
    experiment_3d()

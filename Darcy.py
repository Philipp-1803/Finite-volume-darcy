import Darcy_1D_objekt
import Darcy_2D_objekt
import Darcy_3D_objekt
import Darcy_1D_objekt_without_postprocessing


def experiment_1d():
    data = Darcy_1D_objekt.Data()
    pressure, fluxgrid_x = Darcy_1D_objekt.solve_1d(data)

    print("Pressure", pressure)
    print("Flux", fluxgrid_x)


def experiment_1d_without_postprocessing():
    data = Darcy_1D_objekt_without_postprocessing.Data()
    pressure, fluxgrid_x = (
        Darcy_1D_objekt_without_postprocessing.solve_1d_without_postprocessing(data)
    )

    print("Pressure", pressure)
    print("Flux", fluxgrid_x)


def experiment_2d():
    data = Darcy_2D_objekt.Data()
    pressuregrid, fluxgrid_x, fluxgrid_y = Darcy_2D_objekt.solve_2d(data)

    print("Pressure", pressuregrid)
    print("Flux_x", fluxgrid_x)
    print("Flux_y", fluxgrid_y)


def experiment_3d():
    data = Darcy_3D_objekt.Data()
    pressuregrid, fluxgrid_x, fluxgrid_y, fluxgrid_z = Darcy_3D_objekt.solve_3d(data)

    print("Pressure", pressuregrid)
    print("Flux_x", fluxgrid_x)
    print("Flux_y", fluxgrid_y)
    print("Flux_z", fluxgrid_z)


if __name__ == "__main__":
    experiment_1d()
    experiment_1d_without_postprocessing()

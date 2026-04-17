"""Minimal end-to-end example: load a STEP file, sample point clouds, run CADMap."""

from cadmap.conversion.occ import SolidConvertor, add_geom_to_solid, create_diffsurf_of_solid
from cadmap.sampling.pointcloud import sample_points_from_solid
from cadmap.deformation.fitting import CADMapOptimizer
from cadmap.visualization.plots import write_step

STEP_FILE = "my_part.step"
DEVICE = "cpu"
N_EPOCHS = 500


def main():
    solid = SolidConvertor(STEP_FILE).convert()
    add_geom_to_solid(solid)
    create_diffsurf_of_solid(solid, device=DEVICE)

    cad_pts = sample_points_from_solid(solid, n_pts=500)

    # Replace with your actual segmented point-cloud patches
    targets = cad_pts

    optimizer = CADMapOptimizer(solid, targets=targets, device=DEVICE, n_epochs=N_EPOCHS)
    optimizer.run()

    result = optimizer.export_occ_solid()
    write_step(result, "output.step")
    print("Saved deformed solid to output.step")


if __name__ == "__main__":
    main()

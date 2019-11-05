import autofit as af
import autolens as al
from test_autolens.integration.tests import runner

test_type = "lens_only"
test_name = "lens_light"
data_type = "lens_light_dev_vaucouleurs"
data_resolution = "LSST"


def make_pipeline(name, phase_folders, optimizer_class=af.MultiNest):
    phase1 = al.PhaseInterferometer(
        phase_name="phase_1",
        phase_folders=phase_folders,
        real_space_shape_2d=(20, 20),
        real_space_pixel_scales=0.2,
        galaxies=dict(lens=al.GalaxyModel(redshift=0.5, sersic=al.lp.EllipticalSersic)),
        optimizer_class=optimizer_class,
    )

    phase1.optimizer.const_efficiency_mode = True
    phase1.optimizer.n_live_points = 40
    phase1.optimizer.sampling_efficiency = 0.8

    return al.PipelineInterferometer(name, phase1)


if __name__ == "__main__":
    import sys

    runner.run(sys.modules[__name__])

import autofit as af
from autolens.model.galaxy import galaxy_model as gm
from autolens.model.profiles import light_profiles as lp
from autolens.pipeline import pipeline as pl
from autolens.pipeline.phase import phase_imaging
from test.integration.tests.lens_only import runner

test_type = "lens_only"
test_name = "lens_x1_galaxy"
data_type = "lens_only_dev_vaucouleurs"
data_resolution = "LSST"


def make_pipeline(
        name,
        phase_folders,
        optimizer_class=af.MultiNest
):
    phase1 = phase_imaging.LensPlanePhase(
        phase_name="phase_1",
        phase_folders=phase_folders,
        lens_galaxies=dict(
            lens=gm.GalaxyModel(redshift=0.5, sersic=lp.EllipticalSersic)
        ),
        optimizer_class=optimizer_class,
    )

    phase1.optimizer.const_efficiency_mode = True
    phase1.optimizer.n_live_points = 40
    phase1.optimizer.sampling_efficiency = 0.8

    return pl.PipelineImaging(
        name,
        phase1
    )


if __name__ == "__main__":
    runner.run(
        make_pipeline
    )

import autofit as af
from autolens.model.galaxy import galaxy_model as gm
from autolens.pipeline.phase import phase_imaging
from autolens.pipeline import pipeline as pl
from autolens.model.profiles import light_profiles as lp
from test.integration.tests import runner

test_type = "model_mapper"
test_name = "link_variable_float_to_next_phase"
data_type = "lens_light_dev_vaucouleurs"
data_resolution = "LSST"


def make_pipeline(name, phase_folders, optimizer_class=af.MultiNest):

    phase1 = phase_imaging.PhaseImaging(
        phase_name="phase_1",
        phase_folders=phase_folders,
        galaxies=dict(lens=gm.GalaxyModel(light=lp.EllipticalSersic)),
        optimizer_class=optimizer_class,
    )

    phase1.optimizer.const_efficiency_mode = True
    phase1.optimizer.n_live_points = 20
    phase1.optimizer.sampling_efficiency = 0.8

    class MMPhase2(phase_imaging.PhaseImaging):
        def pass_priors(self, results):

            self.galaxies.lens.light.centre = results.from_phase(
                "phase_1"
            ).variable.galaxies.lens.light.centre

            self.galaxies.lens.light.axis_ratio = results.from_phase(
                "phase_1"
            ).variable.galaxies.lens.light.axis_ratio

    phase2 = MMPhase2(
        phase_name="phase_2",
        phase_folders=phase_folders,
        galaxies=dict(lens=gm.GalaxyModel(light=lp.EllipticalSersic)),
        optimizer_class=optimizer_class,
    )

    phase2.optimizer.const_efficiency_mode = True
    phase2.optimizer.n_live_points = 20
    phase2.optimizer.sampling_efficiency = 0.8

    return pl.PipelineImaging(name, phase1, phase2)


if __name__ == "__main__":
    import sys

    runner.run(sys.modules[__name__])

import autofit as af
from autolens.model.galaxy import galaxy_model as gm
from autolens.model.profiles import light_profiles as lp
from autolens.model.profiles import mass_profiles as mp
from autolens.model.inversion import pixelizations as pix
from autolens.model.inversion import regularization as reg
from autolens.pipeline import pipeline as pl
from autolens.pipeline.phase import phase_imaging
from test.integration.tests import runner

test_type = "full_pipeline"
test_name = "hyper_with_lens_light"
data_type = "lens_and_source_smooth"
data_resolution = "LSST"


def make_pipeline(
    name,
    phase_folders,
    pipeline_pixelization=pix.VoronoiBrightnessImage,
    pipeline_regularization=reg.AdaptiveBrightness,
    optimizer_class=af.MultiNest,
):

    phase1 = phase_imaging.PhaseImaging(
        phase_name="phase_1_lens_sersic",
        phase_folders=phase_folders,
        galaxies=dict(lens=gm.GalaxyModel(redshift=0.5, light=lp.EllipticalSersic)),
        optimizer_class=optimizer_class,
    )

    phase1.optimizer.const_efficiency_mode = True
    phase1.optimizer.n_live_points = 30
    phase1.optimizer.sampling_efficiency = 0.3

    phase1 = phase1.extend_with_multiple_hyper_phases(hyper_galaxy=True)

    class LensSubtractedPhase(phase_imaging.PhaseImaging):
        def pass_priors(self, results):

            ## Lens Light Sersic -> Sersic ##

            self.galaxies.lens.light = results.from_phase(
                "phase_1_lens_sersic"
            ).constant.galaxies.lens.light

            ## Lens Mass, Move centre priors to centre of lens light ###

            self.galaxies.lens.mass.centre = (
                results.from_phase("phase_1_lens_sersic")
                .variable_absolute(a=0.1)
                .galaxies.lens.light.centre
            )

            ## Set all hyper_galaxy-galaxies if feature is turned on ##

            self.galaxies.lens.hyper_galaxy = (
                results.last.hyper_combined.constant.galaxies.lens.hyper_galaxy
            )

    phase2 = LensSubtractedPhase(
        phase_name="phase_2_lens_sie_source_sersic",
        phase_folders=phase_folders,
        galaxies=dict(
            lens=gm.GalaxyModel(
                redshift=0.5,
                light=lp.EllipticalSersic,
                mass=mp.EllipticalIsothermal,
                shear=mp.ExternalShear,
            ),
            source=gm.GalaxyModel(redshift=1.0, light=lp.EllipticalSersic),
        ),
        optimizer_class=optimizer_class,
    )

    phase2.optimizer.const_efficiency_mode = False
    phase2.optimizer.n_live_points = 50
    phase2.optimizer.sampling_efficiency = 0.3

    phase2 = phase2.extend_with_multiple_hyper_phases(
        hyper_galaxy=True, include_background_sky=True, include_background_noise=True
    )

    class LensSourcePhase(phase_imaging.PhaseImaging):
        def pass_priors(self, results):

            ## Lens Light, Sersic -> Sersic ###

            self.galaxies.lens.light = results.from_phase(
                "phase_1_lens_sersic"
            ).variable.galaxies.lens.light

            ## Lens Mass, SIE -> SIE, Shear -> Shear ###

            self.galaxies.lens.mass = results.from_phase(
                "phase_2_lens_sie_source_sersic"
            ).variable.galaxies.lens.mass

            self.galaxies.lens.shear = results.from_phase(
                "phase_2_lens_sie_source_sersic"
            ).variable.galaxies.lens.shear

            ### Source Light, Sersic -> Sersic ###

            self.galaxies.source = results.from_phase(
                "phase_2_lens_sie_source_sersic"
            ).variable.galaxies.source

            ## Set all hyper_galaxy-galaxies if feature is turned on ##

            self.galaxies.lens.hyper_galaxy = (
                results.last.hyper_combined.constant.galaxies.lens.hyper_galaxy
            )

    phase3 = LensSourcePhase(
        phase_name="phase_3_lens_sersic_sie_source_sersic",
        phase_folders=phase_folders,
        galaxies=dict(
            lens=gm.GalaxyModel(
                redshift=0.5,
                light=lp.EllipticalSersic,
                mass=mp.EllipticalIsothermal,
                shear=mp.ExternalShear,
            ),
            source=gm.GalaxyModel(redshift=1.0, light=lp.EllipticalSersic),
        ),
        optimizer_class=optimizer_class,
    )

    phase3.optimizer.const_efficiency_mode = True
    phase3.optimizer.n_live_points = 75
    phase3.optimizer.sampling_efficiency = 0.3

    phase3 = phase3.extend_with_multiple_hyper_phases(hyper_galaxy=True)

    class InversionPhase(phase_imaging.PhaseImaging):
        def pass_priors(self, results):

            ## Lens Light & Mass, Sersic -> Sersic, SIE -> SIE, Shear -> Shear ###

            self.galaxies.lens = results.from_phase(
                "phase_3_lens_sersic_sie_source_sersic"
            ).constant.galaxies.lens

            ## Set all hyper_galaxy-galaxies if feature is turned on ##

            self.galaxies.lens.hyper_galaxy = (
                results.last.hyper_combined.constant.galaxies.lens.hyper_galaxy
            )

    phase4 = InversionPhase(
        phase_name="phase_4_initialize_magnification_inversion",
        phase_folders=phase_folders,
        galaxies=dict(
            lens=gm.GalaxyModel(
                redshift=0.5,
                light=lp.EllipticalSersic,
                mass=mp.EllipticalIsothermal,
                shear=mp.ExternalShear,
            ),
            source=gm.GalaxyModel(
                redshift=1.0,
                pixelization=pix.VoronoiMagnification,
                regularization=reg.Constant,
            ),
        ),
        optimizer_class=optimizer_class,
    )

    phase4.optimizer.const_efficiency_mode = True
    phase4.optimizer.n_live_points = 20
    phase4.optimizer.sampling_efficiency = 0.8

    phase4 = phase4.extend_with_multiple_hyper_phases(
        hyper_galaxy=True, inversion=False
    )

    class InversionPhase(phase_imaging.PhaseImaging):
        def pass_priors(self, results):

            ## Lens Light & Mass, Sersic -> Sersic, SIE -> SIE, Shear -> Shear ###

            self.galaxies.lens = results.from_phase(
                "phase_3_lens_sersic_sie_source_sersic"
            ).variable.galaxies.lens

            ### Source Inversion, Inv -> Inv ###

            self.galaxies.source = results.from_phase(
                "phase_4_initialize_magnification_inversion"
            ).constant.galaxies.source

            ## Set all hyper_galaxy-galaxies if feature is turned on ##

            self.galaxies.lens.hyper_galaxy = (
                results.last.hyper_combined.constant.galaxies.lens.hyper_galaxy
            )

    phase5 = InversionPhase(
        phase_name="phase_5_lens_sersic_sie_source_magnification_inversion",
        phase_folders=phase_folders,
        galaxies=dict(
            lens=gm.GalaxyModel(
                redshift=0.5,
                light=lp.EllipticalSersic,
                mass=mp.EllipticalIsothermal,
                shear=mp.ExternalShear,
            ),
            source=gm.GalaxyModel(
                redshift=1.0,
                pixelization=pipeline_pixelization,
                regularization=pipeline_regularization,
            ),
        ),
        optimizer_class=optimizer_class,
    )

    phase5.optimizer.const_efficiency_mode = True
    phase5.optimizer.n_live_points = 75
    phase5.optimizer.sampling_efficiency = 0.2

    phase5 = phase5.extend_with_multiple_hyper_phases(
        hyper_galaxy=True,
        include_background_sky=True,
        include_background_noise=True,
        inversion=False,
    )

    class InversionPhase(phase_imaging.PhaseImaging):
        def pass_priors(self, results):

            ## Lens Light & Mass, Sersic -> Sersic, SIE -> SIE, Shear -> Shear ###

            self.galaxies.lens = results.from_phase(
                "phase_5_lens_sersic_sie_source_magnification_inversion"
            ).constant.galaxies.lens

            ## Set all hyper_galaxy-galaxies if feature is turned on ##

            self.galaxies.lens.hyper_galaxy = (
                results.last.hyper_combined.constant.galaxies.lens.hyper_galaxy
            )

    phase6 = InversionPhase(
        phase_name="phase_6_initialize_inversion",
        phase_folders=phase_folders,
        galaxies=dict(
            lens=gm.GalaxyModel(
                redshift=0.5,
                light=lp.EllipticalSersic,
                mass=mp.EllipticalIsothermal,
                shear=mp.ExternalShear,
            ),
            source=gm.GalaxyModel(
                redshift=1.0,
                pixelization=pipeline_pixelization,
                regularization=pipeline_regularization,
            ),
        ),
        optimizer_class=optimizer_class,
    )

    phase6.optimizer.const_efficiency_mode = True
    phase6.optimizer.n_live_points = 20
    phase6.optimizer.sampling_efficiency = 0.8

    phase6 = phase6.extend_with_multiple_hyper_phases(
        hyper_galaxy=True,
        include_background_sky=True,
        include_background_noise=True,
        inversion=True,
    )

    class InversionPhase(phase_imaging.PhaseImaging):
        def pass_priors(self, results):

            ## Lens Light & Mass, Sersic -> Sersic, SIE -> SIE, Shear -> Shear ###

            self.galaxies.lens = results.from_phase(
                "phase_7_lens_sersic_sie_source_inversion"
            ).variable.galaxies.lens

            ### Source Inversion, Inv -> Inv ###

            self.galaxies.source = results.from_phase(
                "phase_6_initialize_inversion"
            ).hyper_combined.constant.galaxies.source

            ## Set all hyper_galaxy-galaxies if feature is turned on ##

            self.galaxies.lens.hyper_galaxy = (
                results.last.hyper_combined.constant.galaxies.lens.hyper_galaxy
            )

            self.galaxies.source.hyper_galaxy = (
                results.last.hyper_combined.constant.galaxies.source.hyper_galaxy
            )

    phase7 = InversionPhase(
        phase_name="phase_7_lens_sersic_sie_source_inversion",
        phase_folders=phase_folders,
        galaxies=dict(
            lens=gm.GalaxyModel(
                redshift=0.5,
                light=lp.EllipticalSersic,
                mass=mp.EllipticalIsothermal,
                shear=mp.ExternalShear,
            ),
            source=gm.GalaxyModel(
                redshift=1.0,
                pixelization=pipeline_pixelization,
                regularization=pipeline_regularization,
            ),
        ),
        optimizer_class=optimizer_class,
    )

    phase7.optimizer.const_efficiency_mode = True
    phase7.optimizer.n_live_points = 75
    phase7.optimizer.sampling_efficiency = 0.2

    phase7 = phase7.extend_with_multiple_hyper_phases(
        hyper_galaxy=True,
        include_background_sky=True,
        include_background_noise=True,
        inversion=True,
    )

    return pl.PipelineImaging(
        name, phase1, phase2, phase3, phase4, phase5, phase6, phase7, hyper_mode=False
    )


if __name__ == "__main__":
    import sys

    runner.run(sys.modules[__name__])

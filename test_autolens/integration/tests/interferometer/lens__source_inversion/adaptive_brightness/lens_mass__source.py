import autofit as af
import autolens as al
from test_autolens.integration.tests.interferometer import runner

test_type = "lens__source_inversion"
test_name = "lens_mass__source_adaptive_brightness"
data_name = "lens_sie__source_smooth"
instrument = "sma"


def make_pipeline(name, phase_folders, real_space_mask, search=af.PySwarmsGlobal()):

    phase1 = al.PhaseInterferometer(
        phase_name="phase_1",
        phase_folders=phase_folders,
        galaxies=dict(
            lens=al.GalaxyModel(redshift=0.5, mass=al.mp.EllipticalIsothermal),
            source=al.GalaxyModel(redshift=1.0, light=al.lp.EllipticalSersic),
        ),
        real_space_mask=real_space_mask,
        search=search,
    )

    phase2 = al.PhaseInterferometer(
        phase_name="phase_2_weighted_regularization",
        phase_folders=phase_folders,
        galaxies=dict(
            lens=al.GalaxyModel(
                redshift=0.5,
                mass=phase1.result.instance.galaxies.lens.mass,
                shear=phase1.result.instance.galaxies.lens.shear,
            ),
            source=al.GalaxyModel(
                redshift=1.0,
                pixelization=al.pix.VoronoiBrightnessImage,
                regularization=al.reg.AdaptiveBrightness,
            ),
        ),
        real_space_mask=real_space_mask,
        search=search,
    )

    phase2.search.const_efficiency_mode = True
    phase2.search.n_live_points = 40
    phase2.search.sampling_efficiency = 0.8

    phase3 = al.PhaseInterferometer(
        phase_name="phase_3",
        phase_folders=phase_folders,
        galaxies=dict(
            lens=al.GalaxyModel(
                redshift=0.5,
                mass=phase1.model.galaxies.lens.mass,
                shear=phase1.model.galaxies.lens.shear,
            ),
            source=al.GalaxyModel(
                redshift=1.0,
                pixelization=phase2.result.instance.galaxies.source.pixelization,
                regularization=phase2.result.instance.galaxies.source.regularization,
            ),
        ),
        real_space_mask=real_space_mask,
        search=search,
    )

    phase3.search.const_efficiency_mode = True
    phase3.search.n_live_points = 40
    phase3.search.sampling_efficiency = 0.8

    phase4 = al.PhaseInterferometer(
        phase_name="phase_4_weighted_regularization",
        phase_folders=phase_folders,
        galaxies=dict(
            lens=al.GalaxyModel(
                redshift=0.5,
                mass=phase3.result.instance.galaxies.lens.mass,
                shear=phase3.result.instance.galaxies.lens.shear,
            ),
            source=al.GalaxyModel(
                redshift=1.0,
                pixelization=phase2.model.galaxies.source.pixelization,
                regularization=phase2.model.galaxies.source.pixelization,
            ),
        ),
        real_space_mask=real_space_mask,
        search=search,
    )

    phase4.search.const_efficiency_mode = True
    phase4.search.n_live_points = 40
    phase4.search.sampling_efficiency = 0.8

    return al.PipelineDataset(name, phase1, phase2, phase3, phase4)


if __name__ == "__main__":
    import sys

    runner.run(sys.modules[__name__])

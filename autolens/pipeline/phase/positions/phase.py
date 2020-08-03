import autofit as af
from astropy import cosmology as cosmo
from autogalaxy.pipeline.phase import abstract
from autogalaxy.pipeline.phase.imaging.phase import PhaseAttributes as AgPhaseAttributes
from autolens.pipeline.phase.settings import PhaseSettingsPositions
from autolens.pipeline.phase.positions.analysis import Analysis
from autolens.pipeline.phase.positions.result import Result


class PhasePositions(abstract.AbstractPhase):

    galaxies = af.PhaseProperty("galaxies")

    Analysis = Analysis
    Result = Result

    @af.convert_paths
    def __init__(
        self,
        paths,
        *,
        search,
        solver,
        galaxies=None,
        settings=PhaseSettingsPositions(),
        cosmology=cosmo.Planck15,
    ):

        """

        A phase in an lens pipeline. Uses the set non_linear search to try to fit models and hyper_galaxies
        passed to it.

        Parameters
        ----------
        search: class
            The class of a non_linear search
        sub_size: int
            The side length of the subgrid
        """

        super().__init__(
            paths,
            search=search,
            settings=settings,
            galaxies=galaxies,
            cosmology=cosmology,
        )

        self.solver = solver

    def make_phase_attributes(self, analysis):
        return PhaseAttributes(cosmology=self.cosmology)

    def make_analysis(self, positions, imaging=None, results=None):
        """
        Create an lens object. Also calls the prior passing and masked_imaging modifying functions to allow child
        classes to change the behaviour of the phase.

        Parameters
        ----------
        positions
        mask: Mask
            The default masks passed in by the pipeline
        dataset: im.Imaging
            An masked_imaging that has been masked
        results: autofit.tools.pipeline.ResultsCollection
            The result from the previous phase

        Returns
        -------
        lens : Analysis
            An lens object that the non-linear search calls to determine the fit of a set of values
        """

        self.output_phase_info()

        analysis = self.Analysis(
            positions=positions,
            solver=self.solver,
            imaging=imaging,
            cosmology=self.cosmology,
            image_path=self.search.paths.image_path,
            results=results,
        )

        return analysis

    def output_phase_info(self):

        file_phase_info = "{}/{}".format(self.search.paths.output_path, "phase.info")

        with open(file_phase_info, "w") as phase_info:
            phase_info.write("Optimizer = {} \n".format(type(self.search).__name__))
            phase_info.write(
                "Positions Threshold = {} \n".format(self.settings.positions_threshold)
            )
            phase_info.write("Cosmology = {} \n".format(self.cosmology))

            phase_info.close()


class PhaseAttributes(AgPhaseAttributes):
    def __init__(self, cosmology):
        super().__init__(cosmology=cosmology)

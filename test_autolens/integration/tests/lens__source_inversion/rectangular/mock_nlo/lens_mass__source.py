from test_autolens.integration.tests.lens__source_inversion.rectangular import lens_mass__source
from test_autolens.integration.tests.runner import run_a_mock


class TestCase:
    def _test_lens_mass__source(self):
        run_a_mock(lens_mass__source)

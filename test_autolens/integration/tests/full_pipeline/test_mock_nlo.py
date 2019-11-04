from test_autolens.integration.tests.full_pipeline import hyper_no_lens_light_bg
from test_autolens.integration.tests.full_pipeline import hyper_with_lens_light
from test_autolens.integration.tests.full_pipeline import hyper_with_lens_light_bg
from test_autolens.integration.tests.full_pipeline import (
    hyper_with_lens_light_bg_new_api,
)
from test_autolens.integration.tests.runner import run_a_mock


class TestCase:

    def _test_hyper_no_lens_light_bg(self):
        run_a_mock(hyper_no_lens_light_bg)

    def _test_hyper_with_lens_light(self):
        run_a_mock(hyper_with_lens_light)

    def _test_hyper_with_lens_light_bg(self):
        run_a_mock(hyper_with_lens_light_bg)

    def _test_hyper_with_lens_light_bg_new_api(self):
        run_a_mock(hyper_with_lens_light_bg_new_api)

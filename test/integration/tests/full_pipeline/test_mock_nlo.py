from test import decomposed_lens_light_and_dark
from test import hyper_no_lens_light_bg
from test import hyper_with_lens_light
from test import hyper_with_lens_light_bg
from test import hyper_with_lens_light_bg_new_api
from test import run_a_mock


class TestCase:
    def _test_decomposed_lens_light_and_dark(self):
        run_a_mock(decomposed_lens_light_and_dark)

    def _test_hyper_no_lens_light_bg(self):
        run_a_mock(hyper_no_lens_light_bg)

    def _test_hyper_with_lens_light(self):
        run_a_mock(hyper_with_lens_light)

    def _test_hyper_with_lens_light_bg(self):
        run_a_mock(hyper_with_lens_light_bg)

    def _test_hyper_with_lens_light_bg_new_api(self):
        run_a_mock(hyper_with_lens_light_bg_new_api)

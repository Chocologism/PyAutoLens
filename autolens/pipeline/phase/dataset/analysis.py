from autolens.lens import ray_tracing


class Analysis:
    def plane_for_instance(self, instance):
        raise NotImplementedError()

    def tracer_for_instance(self, instance):

        return ray_tracing.Tracer.from_galaxies(
            galaxies=instance.galaxies, cosmology=self.cosmology
        )

    def stochastic_log_evidences_for_instance(self, instance, samples=100):
        raise NotImplementedError()

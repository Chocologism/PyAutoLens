from src import exc


class Tracer(object):

    def __init__(self, lens_galaxies, source_galaxies, masked_image):
        """The ray-tracing calculations, defined by a lensing system with just one image-plane and source-plane.

        This has no associated cosmology, thus all calculations are performed in arc seconds and galaxies do not need
        known redshift measurements. For computational efficiency, it is recommend this ray-tracing class is used for
        lens modeling, provided cosmological information is not necessary.

        Parameters
        ----------
        lens_galaxies : [Galaxy]
            The list of lens galaxies in the image-plane.
        source_galaxies : [Galaxy]
            The list of source galaxies in the source-plane.
        masked_image : MaskedImage
            An image that has been masked for efficiency
        """
        self.image_plane = Plane(lens_galaxies, masked_image, compute_deflections=True)

        source_plane_grids = self.image_plane.trace_to_next_plane()

        self.source_plane = Plane(source_galaxies, source_plane_grids, compute_deflections=False)

    def generate_image_of_galaxy_light_profiles(self, mapping):
        """Generate the image of the galaxies over the entire ray trace."""
        return self.image_plane.generate_image_of_galaxy_light_profiles(mapping
                                                                        ) + self.source_plane.generate_image_of_galaxy_light_profiles(
            mapping)

    def generate_blurring_image_of_galaxy_light_profiles(self):
        """Generate the image of all galaxy light profiles in the blurring regions of the image."""
        return self.image_plane.generate_blurring_image_of_galaxy_light_profiles(
        ) + self.source_plane.generate_blurring_image_of_galaxy_light_profiles()

    def generate_pixelization_matrices_of_source_galaxy(self, mapping):
        return self.source_plane.generate_pixelization_matrices_of_galaxy(mapping)


class Plane(object):

    def __init__(self, galaxies, masked_image, compute_deflections=True):
        """

        Represents a plane, which is a set of galaxies and grids at a given redshift in the lens ray-tracing
        calculation.

        The image-plane coordinates are defined on the observed image's uniform regular grid_coords. Calculating its
        model images from its light profiles exploits this uniformity to perform more efficient and precise calculations
        via an iterative sub-griding approach.

        The light profiles of galaxies at higher redshifts (and therefore in different lens-planes) can be assigned to
        the ImagePlane. This occurs when:

        1) The efficiency and precision offered by computing the light profile on a uniform grid_coords is preferred and
        won't lead noticeable inaccuracy. For example, computing the light profile of the main lens galaxy, ignoring
        minor lensing effects due to a low mass foreground substructure.

        2) When evaluating the light profile in its lens-plane is inaccurate. For example, when modeling the
        point-source images of a lensed quasar, effects like micro-lensing mean lens-plane modeling will be inaccurate.


        Parameters
        ----------
        galaxies : [Galaxy]
            The galaxies in the plane.
        masked_image: MaskedImage
            An image that has been masked for efficiency
        """

        self.galaxies = galaxies
        self.masked_image = masked_image
        if compute_deflections:
            self.deflections = self.masked_image.deflection_grids_for_galaxies(self.galaxies)

    def trace_to_next_plane(self):
        """Trace the grids to the next plane.

        NOTE : This does not work for multi-plane lensing, which requires one to use the previous plane's deflection
        angles to perform the tracing. I guess we'll ultimately call this class 'LensPlanes' and have it as a list.
        """
        return self.masked_image.traced_grids_for_deflections(self.deflections)

    def generate_image_of_galaxy_light_profiles(self, mapping):
        """Generate the image of the galaxies in this plane."""
        return self.masked_image.sub.intensities_via_grid(self.galaxies, mapping)

    def generate_blurring_image_of_galaxy_light_profiles(self):
        """Generate the image of the galaxies in this plane."""
        return self.masked_image.blurring.intensities_via_grid(self.galaxies)

    def generate_pixelization_matrices_of_galaxy(self, mapping):

        pixelized_galaxies = list(filter(lambda galaxy: galaxy.has_pixelization, self.galaxies))

        if len(pixelized_galaxies) == 0:
            return None
        if len(pixelized_galaxies) == 1:
            return pixelized_galaxies[0].pixelization.inversion_from_pix_grids(self.masked_image,
                                                                               self.masked_image.sub, mapping)
        elif len(pixelized_galaxies) > 1:
            raise exc.PixelizationException('The number of galaxies with pixelizations in one plane is above 1')

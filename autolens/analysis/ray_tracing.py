from autolens import exc
from autolens.imaging import mask
from astropy import constants
from functools import wraps
import math
import numpy as np


class AbstractTracer(object):

    @property
    def all_planes(self):
        raise NotImplementedError()

    @property
    def has_galaxy_with_light_profile(self):
        return any(list(map(lambda galaxy : galaxy.has_light_profile, self.galaxies)))

    @property
    def has_galaxy_with_pixelization(self):
        return any(list(map(lambda galaxy : galaxy.has_pixelization, self.galaxies)))

    @property
    def has_grid_mappers(self):
        return isinstance(self.all_planes[0].grids.image, mask.GridMapper)

    @property
    def has_hyper_galaxy(self):
        return any(list(map(lambda galaxy : galaxy.has_hyper_galaxy, self.galaxies)))

    @property
    def image_plane_image(self):
        return sum(self.image_plane_images_of_planes)

    @property
    def image_plane_images_of_planes(self):
        return [plane.image_plane_image for plane in self.all_planes]

    @property
    def image_plane_images_of_galaxies(self):
        """
        Returns
        -------
        image_plane_lens_galaxy_images: [ndarray]
            An masked_image for each galaxy in this ray tracer
        """
        return [galaxy_image for plane in self.all_planes for galaxy_image in plane.image_plane_images_of_galaxies]

    @property
    def image_plane_blurring_image(self):
        return sum(self.image_plane_blurring_images_of_planes)

    @property
    def image_plane_blurring_images_of_planes(self):
        return [plane.image_plane_blurring_image for plane in self.all_planes]

    @property
    def image_plane_blurring_images_of_galaxies(self):
        """
        Returns
        -------
        image_plane_lens_galaxy_images: [ndarray]
            An masked_image for each galaxy in this ray tracer
        """
        return [galaxy_blurring_image for plane in self.all_planes for galaxy_blurring_image
                in plane.image_plane_blurring_images_of_galaxies]

    def plane_images_of_planes(self, shape=(30, 30)):
        return [plane.plane_image(shape) for plane in self.all_planes]

    @property
    def image_grids_of_planes(self):
        return [plane.grids.image for plane in self.all_planes]

    @property
    def xticks_of_planes(self):
        return [plane.xticks_from_image_grid for plane in self.all_planes]

    @property
    def yticks_of_planes(self):
        return [plane.yticks_from_image_grid for plane in self.all_planes]

    @property
    def hyper_galaxies(self):
        return [hyper_galaxy for plane in self.all_planes for hyper_galaxy in
                plane.hyper_galaxies]

    @property
    def galaxies(self):
        return [galaxy for plane in self.all_planes for galaxy in plane.galaxies]

    @property
    def all_with_hyper_galaxies(self):
        return len(list(filter(None, self.hyper_galaxies))) == len(self.galaxies)


class TracerImagePlane(AbstractTracer):

    @property
    def all_planes(self):
        return [self.image_plane]

    def __init__(self, lens_galaxies, image_grids):
        """The ray-tracing calculations, defined by a lensing system with just one image-plane. This doesn't actually \
        perform any ray-tracing / lensing calculations, and is used purely for light-profile fitting of objects in \
        the image-plane (e.g. the lens galaxy)

        By default, this has no associated cosmology, thus all calculations are performed in arc seconds and galaxies \
        do not need input redshifts. For computational efficiency, it is recommend this ray-tracing class is used for \
        lens modeling, provided cosmological information is not necessary.

        If a cosmology is supplied, the plane's angular diameter distances, conversion factors, etc. will be computed.

        Parameters
        ----------
        lens_galaxies : [Galaxy]
            The list of lens galaxies in the masked_image-plane.
        image_grids : mask.GridCollection
            The masked_image-plane coordinate grids where ray-tracing calculation are performed, (this includes the
            masked_image-grid, sub-grid, blurring-grid, etc.).
        """
        if not lens_galaxies:
            raise exc.RayTracingException('No lens galaxies have been input into the Tracer')

        self.image_plane = Plane(lens_galaxies, image_grids, compute_deflections=True)


class TracerImageSourcePlanes(TracerImagePlane):

    @property
    def all_planes(self):
        return [self.image_plane, self.source_plane]

    def __init__(self, lens_galaxies, source_galaxies, image_grids, cosmology=None):
        """The ray-tracing calculations, defined by a lensing system with just one masked_image-plane and source-plane.

        By default, this has no associated cosmology, thus all calculations are performed in arc seconds and galaxies \
        do not need input redshifts. For computational efficiency, it is recommend this ray-tracing class is used for \
        lens modeling, provided cosmological information is not necessary.

        If a cosmology is supplied, the plane's angular diameter distances, conversion factors, etc. will be computed.

        Parameters
        ----------
        lens_galaxies : [Galaxy]
            The list of lens galaxies in the masked_image-plane.
        source_galaxies : [Galaxy]
            The list of source galaxies in the source-plane.
        image_grids : mask.GridCollection
            The masked_image-plane coordinate grids where ray-tracing calculation are performed, (this includes the
            masked_image-grid, sub-grid, blurring-grid, etc.).
        cosmology : astropy.cosmology.Planck15
            The cosmology of the ray-tracing calculation.
        """

        super(TracerImageSourcePlanes, self).__init__(lens_galaxies, image_grids)

        if not source_galaxies:
            raise exc.RayTracingException('No source galaxies have been input into the Tracer (TracerImageSourcePlanes)')

        if cosmology is not None:
            self.geometry = TracerGeometry(redshifts=[lens_galaxies[0].redshift, source_galaxies[0].redshift],
                                           cosmology=cosmology)
        else:
            self.geometry = None

        source_plane_grids = self.image_plane.trace_to_next_plane()

        self.source_plane = Plane(source_galaxies, source_plane_grids, compute_deflections=False)

    def reconstructors_from_source_plane(self, borders):
        return self.source_plane.reconstructor_from_plane(borders)


class AbstractTracerMulti(AbstractTracer):

    @property
    def all_planes(self):
        return self.planes

    def __init__(self, galaxies, cosmology):
        """The ray-tracing calculations, defined by a lensing system with just one masked_image-plane and source-plane.

        This has no associated cosmology, thus all calculations are performed in arc seconds and galaxies do not need
        known redshift measurements. For computational efficiency, it is recommend this ray-tracing class is used for
        lens modeling, provided cosmological information is not necessary.

        Parameters
        ----------
        galaxies : [Galaxy]
            The list of galaxies in the ray-tracing calculation.
        image_grids : mask.GridCollection
            The masked_image-plane coordinate grids where ray-tracing calculation are performed, (this includes the
            masked_image-grid, sub-grid, blurring-grid, etc.).
        cosmology : astropy.cosmology
            The cosmology of the ray-tracing calculation.
        """

        if not galaxies:
            raise exc.RayTracingException('No galaxies have been input into the Tracer (TracerImageSourcePlanes)')

        self.galaxies_redshift_order = sorted(galaxies, key=lambda galaxy: galaxy.redshift, reverse=False)

        # Ideally we'd extract the planes_red_Shfit order from the list above. However, I dont know how to extract it
        # Using a list of class attributes so make a list of redshifts for now.

        galaxy_redshifts = list(map(lambda galaxy: galaxy.redshift, self.galaxies_redshift_order))
        self.planes_redshift_order = [redshift for i, redshift in enumerate(galaxy_redshifts)
                                      if redshift not in galaxy_redshifts[:i]]
        self.geometry = TracerGeometry(redshifts=self.planes_redshift_order, cosmology=cosmology)

        # TODO : Idea is to get a list of all galaxies in each plane - can you clean up the logic below?

        self.planes_galaxies = []

        for (plane_index, plane_redshift) in enumerate(self.planes_redshift_order):
            self.planes_galaxies.append(list(map(lambda galaxy:
                                                 galaxy if galaxy.redshift == plane_redshift else None,
                                                 self.galaxies_redshift_order)))
            self.planes_galaxies[plane_index] = list(filter(None, self.planes_galaxies[plane_index]))


class TracerMulti(AbstractTracerMulti):

    def __init__(self, galaxies, cosmology, image_grids):
        """The ray-tracing calculations, defined by a lensing system with just one masked_image-plane and source-plane.

        This has no associated cosmology, thus all calculations are performed in arc seconds and galaxies do not need
        known redshift measurements. For computational efficiency, it is recommend this ray-tracing class is used for
        lens modeling, provided cosmological information is not necessary.

        Parameters
        ----------
        galaxies : [Galaxy]
            The list of galaxies in the ray-tracing calculation.
        image_grids : mask.GridCollection
            The masked_image-plane coordinate grids where ray-tracing calculation are performed, (this includes the
            masked_image-grid, sub-grid, blurring-grid, etc.).
        cosmology : astropy.cosmology
            The cosmology of the ray-tracing calculation.
        """

        if not galaxies:
            raise exc.RayTracingException('No galaxies have been input into the Tracer (TracerImageSourcePlanes)')

        super(TracerMulti, self).__init__(galaxies, cosmology)

        self.planes = []

        for plane_index in range(0, len(self.planes_redshift_order)):

            if plane_index < len(self.planes_redshift_order) - 1:
                compute_deflections = True
            elif plane_index == len(self.planes_redshift_order) - 1:
                compute_deflections = False
            else:
                raise exc.RayTracingException('A galaxy was not correctly allocated its previous / next redshifts')

            new_grid = image_grids

            if plane_index > 0:
                for previous_plane_index in range(plane_index):
                    scaling_factor = self.geometry.scaling_factor(plane_i=previous_plane_index,
                                                                  plane_j=plane_index)

                    def scale(grid):
                        return np.multiply(scaling_factor, grid)

                    scaled_deflections = self.planes[previous_plane_index].deflections. \
                        apply_function(scale)

                    def subtract_scaled_deflections(grid, scaled_deflection):
                        return np.subtract(grid, scaled_deflection)

                    new_grid = new_grid.map_function(subtract_scaled_deflections, scaled_deflections)

            self.planes.append(Plane(galaxies=self.planes_galaxies[plane_index], grids=new_grid,
                                     compute_deflections=compute_deflections))

    def reconstructors_from_planes(self, borders):
        return list(map(lambda plane: plane.reconstructor_from_plane(borders), self.planes))


class TracerGeometry(object):

    def __init__(self, redshifts, cosmology):
        """The geometry of a ray-tracing grid comprising an masked_image-plane and source-plane.

        This sets up the angular diameter distances between each plane and the Earth, and between one another. \
        The critical density of the lens plane is also computed.

        Parameters
        ----------
        cosmology : astropy.cosmology.Planck15
            The cosmology of the ray-tracing calculation.
        """
        self.cosmology = cosmology
        self.redshifts = redshifts
        self.final_plane = len(self.redshifts) - 1
        self.ang_to_final_plane = self.ang_to_earth(plane_i=self.final_plane)

    def arcsec_per_kpc(self, plane_i):
        return self.cosmology.arcsec_per_kpc_proper(z=self.redshifts[plane_i]).value

    def kpc_per_arcsec(self, plane_i):
        return 1.0 / self.cosmology.arcsec_per_kpc_proper(z=self.redshifts[plane_i]).value

    def ang_to_earth(self, plane_i):
        return self.cosmology.angular_diameter_distance(self.redshifts[plane_i]).to('kpc').value

    def ang_between_planes(self, plane_i, plane_j):
        return self.cosmology.angular_diameter_distance_z1z2(self.redshifts[plane_i], self.redshifts[plane_j]). \
            to('kpc').value

    @property
    def constant_kpc(self):
        # noinspection PyUnresolvedReferences
        return constants.c.to('kpc / s').value ** 2.0 / (4 * math.pi * constants.G.to('kpc3 / M_sun s2').value)

    def critical_density_kpc(self, plane_i, plane_j):
        return self.constant_kpc * self.ang_to_earth(plane_j) / \
               (self.ang_between_planes(plane_i, plane_j) * self.ang_to_earth(plane_i))

    def critical_density_arcsec(self, plane_i, plane_j):
        return self.critical_density_kpc(plane_i, plane_j) * self.kpc_per_arcsec(plane_i) ** 2.0

    def scaling_factor(self, plane_i, plane_j):
        return (self.ang_between_planes(plane_i, plane_j) * self.ang_to_final_plane) / (
                self.ang_to_earth(plane_j) * self.ang_between_planes(plane_i, self.final_plane))


class Plane(object):

    def __init__(self, galaxies, grids, compute_deflections=True):
        """

        Represents a plane, which is a set of galaxies and grids at a given redshift in the lens ray-tracing
        calculation.

        The masked_image-plane coordinates are defined on the observed masked_image's uniform regular grid_coords.
        Calculating its model images from its light profiles exploits this uniformity to perform more efficient and
        precise calculations via an iterative sub-griding approach.

        The light profiles of galaxies at higher redshifts (and therefore in different lens-planes) can be assigned to
        the ImagePlane. This occurs when:

        1) The efficiency and precision offered by computing the light profile on a uniform grid_coords is preferred and
        won't lead noticeable inaccuracy. For example, computing the light profile of the main lens galaxy, ignoring
        minor lensing effects due to a low mass foreground substructure.

        2) When evaluating the light profile in its lens-plane is inaccurate. For example, when modeling the
        point-source images of a lensed quasar, effects like micro-lensing mean lens-plane modeling will be inaccurate.


        Parameters ---------- galaxies : [Galaxy] The galaxies in the plane. grids :
        mask.GridCollection The grids of (x,y) coordinates in the plane, including the masked_image grid_coords,
        sub-grid_coords, blurring, grid_coords, etc.
        """
        self.galaxies = galaxies
        self.grids = grids

        if compute_deflections:
            def calculate_deflections(grid):
                return sum(map(lambda galaxy: galaxy.deflections_from_grid(grid), galaxies))

            self.deflections = self.grids.apply_function(calculate_deflections)

    def trace_to_next_plane(self):
        """Trace the grids to the next plane.

        NOTE : This does not work for multi-plane lensing, which requires one to use the previous plane's deflection
        angles to perform the tracing. I guess we'll ultimately call this class 'LensPlanes' and have it as a list.
        """
        return self.grids.map_function(np.subtract, self.deflections)

    @property
    def image_plane_image(self):
        return sum(self.image_plane_images_of_galaxies)

    @property
    def image_plane_images_of_galaxies(self):
        """
        Returns
        -------
        image_plane_lens_galaxy_images: [ndarray]
            A list of images of galaxies in this plane
        """
        return [self.image_plane_image_from_galaxy(galaxy) for galaxy in self.galaxies]

    def image_plane_image_from_galaxy(self, galaxy):
        """
        Parameters
        ----------
        galaxy: Galaxy
            An individual galaxy, assumed to be in this plane

        Returns
        -------
        galaxy_image: ndarray
            An array describing the intensity of light coming from the galaxy embedded in this plane
        """
        return intensities_from_grid(self.grids.sub, [galaxy])

    @property
    def image_plane_blurring_image(self):
        return sum(self.image_plane_blurring_images_of_galaxies)

    @property
    def image_plane_blurring_images_of_galaxies(self):
        """
        Returns
        -------
        image_plane_lens_galaxy_images: [ndarray]
            A list of images of galaxies in this plane
        """
        return [self.image_plane_blurring_image_from_galaxy(galaxy) for galaxy in self.galaxies]

    def image_plane_blurring_image_from_galaxy(self, galaxy):
        """
        Parameters
        ----------
        galaxy: Galaxy
            An individual galaxy, assumed to be in this plane

        Returns
        -------
        galaxy_image: ndarray
            An array describing the intensity of light coming from the galaxy embedded in this plane
        """
        return intensities_from_grid(self.grids.blurring, [galaxy])

    def plane_image(self, shape=(30, 30)):
        return sum(self.plane_images_of_galaxies(shape))

    def plane_images_of_galaxies(self, shape=(30, 30)):
        """
        Returns
        -------
        image_plane_lens_galaxy_images: [ndarray]
            A list of images of galaxies in this plane
        """
        plane_grid = uniform_grid_from_lensed_grid(self.grids.image, shape)
        return [self.plane_image_from_galaxy(plane_grid, galaxy) for galaxy in self.galaxies]

    def plane_image_from_galaxy(self, plane_grid, galaxy):
        """
        Parameters
        ----------
        plane_grid : ndarray
            A uniform / regular grid of coordinates.
        galaxy: Galaxy
            An individual galaxy, assumed to be in this plane

        Returns
        -------
        galaxy_image: ndarray
            An array describing the intensity of light coming from the galaxy embedded in this plane
        """
        return intensities_from_grid(plane_grid, [galaxy])

    @property
    def xticks_from_image_grid(self):
        return np.around(np.linspace(np.amin(self.grids.image[:,0]), np.amax(self.grids.image[:,0]), 4), 2)

    @property
    def yticks_from_image_grid(self):
        return np.around(np.linspace(np.amin(self.grids.image[:,1]), np.amax(self.grids.image[:,1]), 4), 2)

    @property
    def hyper_galaxies(self):
        return list(filter(None.__ne__, [galaxy.hyper_galaxy for galaxy in self.galaxies]))

    def reconstructor_from_plane(self, borders):

        pixelized_galaxies = list(filter(lambda galaxy: galaxy.has_pixelization, self.galaxies))

        if len(pixelized_galaxies) == 0:
            return None
        if len(pixelized_galaxies) == 1:
            return pixelized_galaxies[0].pixelization.reconstructor_from_pixelization_and_grids(self.grids, borders)
        elif len(pixelized_galaxies) > 1:
            raise exc.PixelizationException('The number of galaxies with pixelizations in one plane is above 1')


class TracerImageSourcePlanesPositions(AbstractTracer):

    @property
    def all_planes(self):
        return [self.image_plane, self.source_plane]

    def __init__(self, lens_galaxies, positions, cosmology=None):
        """The ray-tracing calculations, defined by a lensing system with just one masked_image-plane and source-plane.

        By default, this has no associated cosmology, thus all calculations are performed in arc seconds and galaxies \
        do not need input redshifts. For computational efficiency, it is recommend this ray-tracing class is used for \
        lens modeling, provided cosmological information is not necessary.

        If a cosmology is supplied, the plane's angular diameter distances, conversion factors, etc. will be computed.

        Parameters
        ----------
        lens_galaxies : [Galaxy]
            The list of lens galaxies in the masked_image-plane.
        source_galaxies : [Galaxy]
            The list of source galaxies in the source-plane.
        image_grids : mask.GridCollection
            The masked_image-plane coordinate grids where ray-tracing calculation are performed, (this includes the
            masked_image-grid, sub-grid, blurring-grid, etc.).
        cosmology : astropy.cosmology.Planck15
            The cosmology of the ray-tracing calculation.
        """

        # if cosmology is not None:
        #     self.geometry = TracerGeometry(redshifts=[lens_galaxies[0].redshift, source_galaxies[0].redshift],
        #                                    cosmology=cosmology)
        # else:
        #     self.geometry = None

        self.image_plane = PlanePositions(lens_galaxies, positions, compute_deflections=True)

        source_plane_grids = self.image_plane.trace_to_next_plane()

        self.source_plane = PlanePositions(None, source_plane_grids, compute_deflections=False)


class TracerMultiPositions(AbstractTracerMulti):

    def __init__(self, galaxies, cosmology, positions):
        """The ray-tracing calculations, defined by a lensing system with just one masked_image-plane and source-plane.

        This has no associated cosmology, thus all calculations are performed in arc seconds and galaxies do not need
        known redshift measurements. For computational efficiency, it is recommend this ray-tracing class is used for
        lens modeling, provided cosmological information is not necessary.

        Parameters
        ----------
        galaxies : [Galaxy]
            The list of galaxies in the ray-tracing calculation.
        image_grids : mask.GridCollection
            The masked_image-plane coordinate grids where ray-tracing calculation are performed, (this includes the
            masked_image-grid, sub-grid, blurring-grid, etc.).
        cosmology : astropy.cosmology
            The cosmology of the ray-tracing calculation.
        """

        if not galaxies:
            raise exc.RayTracingException('No galaxies have been input into the Tracer (TracerImageSourcePlanes)')

        super(TracerMultiPositions, self).__init__(galaxies, cosmology)

        self.planes = []

        for plane_index in range(0, len(self.planes_redshift_order)):

            if plane_index < len(self.planes_redshift_order) - 1:
                compute_deflections = True
            elif plane_index == len(self.planes_redshift_order) - 1:
                compute_deflections = False
            else:
                raise exc.RayTracingException('A galaxy was not correctly allocated its previous / next redshifts')

            new_positions = positions

            if plane_index > 0:
                for previous_plane_index in range(plane_index):

                    scaling_factor = self.geometry.scaling_factor(plane_i=previous_plane_index, plane_j=plane_index)
                    scaled_deflections = list(map(lambda deflections :
                                                  np.multiply(scaling_factor, deflections),
                                                  self.planes[previous_plane_index].deflections))

                    new_positions = list(map(lambda positions, deflections :
                                             np.subtract(positions, deflections), new_positions, scaled_deflections))

            self.planes.append(PlanePositions(galaxies=self.planes_galaxies[plane_index], positions=new_positions,
                                     compute_deflections=compute_deflections))


class PlanePositions(object):

    def __init__(self, galaxies, positions, compute_deflections=True):
        """

        Represents a plane, which is a set of galaxies and grids at a given redshift in the lens ray-tracing
        calculation.

        The masked_image-plane coordinates are defined on the observed masked_image's uniform regular grid_coords.
        Calculating its model images from its light profiles exploits this uniformity to perform more efficient and
        precise calculations via an iterative sub-griding approach.

        The light profiles of galaxies at higher redshifts (and therefore in different lens-planes) can be assigned to
        the ImagePlane. This occurs when:

        1) The efficiency and precision offered by computing the light profile on a uniform grid_coords is preferred and
        won't lead noticeable inaccuracy. For example, computing the light profile of the main lens galaxy, ignoring
        minor lensing effects due to a low mass foreground substructure.

        2) When evaluating the light profile in its lens-plane is inaccurate. For example, when modeling the
        point-source images of a lensed quasar, effects like micro-lensing mean lens-plane modeling will be inaccurate.


        Parameters ---------- galaxies : [Galaxy] The galaxies in the plane. grids :
        mask.GridCollection The grids of (x,y) coordinates in the plane, including the masked_image grid_coords,
        sub-grid_coords, blurring, grid_coords, etc.
        """
        self.galaxies = galaxies
        self.positions = positions

        if compute_deflections:
            def calculate_deflections(positions):
                    return sum(map(lambda galaxy: galaxy.deflections_from_grid(positions), galaxies))

            self.deflections = list(map(lambda positions : calculate_deflections(positions), self.positions))

    def trace_to_next_plane(self):
        """Trace the grids to the next plane.

        NOTE : This does not work for multi-plane lensing, which requires one to use the previous plane's deflection
        angles to perform the tracing. I guess we'll ultimately call this class 'LensPlanes' and have it as a list.
        """
        return list(map(lambda positions, deflections : np.subtract(positions, deflections),
                        self.positions, self.deflections))


def sub_to_image_grid(func):
    """
    Wrap the function in a function that may perform two operations on the quantities (intensities, surface_density,
    potential, deflections) computed in the *galaxy* and *profile* modules.

    1) If the grid is a sub-grid (mask.SubGrid), rebin values to the image-grid by taking the mean of each set of \
    sub-gridded values.

    2) If the grid is a GridMapper, returned the grid mapped to 2d.

    Parameters
    ----------
    func : (profiles, *args, **kwargs) -> Object
        A function that requires transformed coordinates

    Returns
    -------
        A function that can except cartesian or transformed coordinates
    """

    @wraps(func)
    def wrapper(grid, galaxies, *args, **kwargs):
        """

        Parameters
        ----------
        profile : GeometryProfile
            The profiles that owns the function
        grid : ndarray
            PlaneCoordinates in either cartesian or profiles coordinate system
        args
        kwargs

        Returns
        -------
            A value or coordinate in the same coordinate system as those passed in.
        """

        result = func(grid, galaxies, *args, *kwargs)

        if isinstance(grid, mask.SubGrid):
            return grid.sub_data_to_image(result)
        else:
            return result

    return wrapper

@sub_to_image_grid
def intensities_from_grid(grid, galaxies):
    return sum(map(lambda g: g.intensities_from_grid(grid), galaxies))

@sub_to_image_grid
def surface_density_from_grid(grid, galaxies):
    return sum(map(lambda g: g.surface_density_from_grid(grid), galaxies))

@sub_to_image_grid
def potential_from_grid(grid, galaxies):
    return sum(map(lambda g: g.potential_from_grid(grid), galaxies))

def deflections_from_grid(grid, galaxies):
    return sum(map(lambda galaxy: galaxy.deflections_from_grid(grid), galaxies))

def deflections_from_grid_collection(grid_collection, galaxies):
    return grid_collection.apply_function(lambda grid: deflections_from_grid(grid, galaxies))

def traced_collection_for_deflections(grids, deflections):
    def subtract_scaled_deflections(grid, scaled_deflection):
        return np.subtract(grid, scaled_deflection)

    result = grids.map_function(subtract_scaled_deflections, deflections)

    return result

def uniform_grid_from_lensed_grid(grid, shape):

    x_min = np.amin(grid[:, 0])
    x_max = np.amax(grid[:, 0])
    y_min = np.amin(grid[:, 1])
    y_max = np.amax(grid[:, 1])

    x_pixel_scale = ((x_max - x_min) / shape[0])
    y_pixel_scale = ((y_max - y_min) / shape[1])

    x_grid = np.linspace(x_min + (x_pixel_scale/2.0), x_max - (x_pixel_scale/2.0), shape[0])
    y_grid = np.linspace(y_min + (y_pixel_scale/2.0), y_max - (y_pixel_scale/2.0), shape[1])

    source_plane_grid = np.zeros((shape[0]*shape[1], 2))

    i = 0
    for x in range(shape[0]):
        for y in range(shape[1]):
            source_plane_grid[i] = np.array([x_grid[x], y_grid[y]])
            i += 1

    return source_plane_grid
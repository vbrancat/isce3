import h5py
import numpy as np
from isce3.io import optimize_chunk_size
from nisar.workflows.h5_prep import set_get_geo_info
from nisar.workflows.helpers import get_cfg_freq_pols

from .InSAR_base_writer import InSARBaseWriter
from .InSAR_HDF5_optimizer_config import get_InSAR_output_options
from .InSAR_L2_writer import L2InSARWriter
from .InSAR_products_info import InSARProductsInfo
from .product_paths import GUNWGroupsPaths
from .RIFG_writer import RIFGWriter
from .RUNW_writer import RUNWWriter
from .units import Units


class GUNWWriter(RUNWWriter, RIFGWriter, L2InSARWriter):
    """
    Writer class for GUNW product inherent from both the RUNWWriter,
    RIFGWriter, and the L2InSARWriter
    """
    def __init__(self, **kwds):
        """
        Constructor for GUNW writer class
        """
        hdf5_opt_config, kwds = get_InSAR_output_options(kwds, 'GUNW')

        super().__init__(**kwds)

        # HDF5 IO optimizer configuration
        self.hdf5_optimizer_config = hdf5_opt_config

        # group paths are GUNW group paths
        self.group_paths = GUNWGroupsPaths()

        # GUNW product information
        self.product_info = InSARProductsInfo.GUNW()

    def save_to_hdf5(self):
        """
        Save to HDF5
        """
        L2InSARWriter.save_to_hdf5(self)

    def add_root_attrs(self):
        """
        add root attributes
        """
        InSARBaseWriter.add_root_attrs(self)

        self.attrs["title"] = np.bytes_("NISAR L2 GUNW Product")
        self.attrs["reference_document"] = \
            np.bytes_("D-102272 NISAR NASA SDS Product Specification"
                       " L2 Geocoded Unwrapped Interferogram")

        ctype = h5py.h5t.py_create(np.complex64)
        ctype.commit(self["/"].id, np.bytes_("complex64"))

    def add_radar_grid_cubes(self):
        """
        Add the radar grid cubes
        """
        L2InSARWriter.add_radar_grid_cubes(self)

        ## Add the radar grid cubes of solid eath tide phase for along-track and along-slant range.
        proc_cfg = self.cfg["processing"]
        tropo_cfg = proc_cfg['troposphere_delay']
        radar_grid_cubes_geogrid = proc_cfg["radar_grid_cubes"]["geogrid"]
        radar_grid_cubes_heights = proc_cfg["radar_grid_cubes"]["heights"]

        radar_grid = self[self.group_paths.RadarGridPath]
        descrs = ["Solid Earth tides phase along slant range direction",
                  'Solid Earth tides phase in along-track direction']
        product_names = ['slantRangeSolidEarthTidesPhase']

        # Add the troposphere datasets to the radarGrid cube
        if tropo_cfg['enabled']:
            for delay_type in ['wet', 'hydrostatic', 'comb']:
                if tropo_cfg[f'enable_{delay_type}_product']:
                    descrs.append(f"{delay_type.capitalize()} component "
                                  "of the troposphere phase screen")
                    if delay_type == 'comb':
                        product_names.append(f'combinedTroposphericPhaseScreen')
                    else:
                        product_names.append(f'{delay_type}TroposphericPhaseScreen')

        cube_shape = [len(radar_grid_cubes_heights),
                      radar_grid_cubes_geogrid.length,
                      radar_grid_cubes_geogrid.width]

        # Retrieve the x, y, and z coordinates from the radargrid cube
        # Since the radargrid cube has been added, it is safe to
        # access those coordinates here.
        xds = radar_grid['xCoordinates']
        yds = radar_grid['yCoordinates']
        zds = radar_grid['heightAboveEllipsoid']

        # Include options for compression for dataset creation
        create_dataset_kwargs = {}
        if self.hdf5_optimizer_config.compression_enabled:
            if self.hdf5_optimizer_config.compression_type is not None:
                create_dataset_kwargs['compression'] = \
                    self.hdf5_optimizer_config.compression_type
            if self.hdf5_optimizer_config.compression_level is not None:
                create_dataset_kwargs['compression_opts'] = \
                    self.hdf5_optimizer_config.compression_level
            # Add shuffle filter options
            create_dataset_kwargs['shuffle'] = \
                self.hdf5_optimizer_config.shuffle_filter

        for product_name, descr in zip(product_names,descrs):
            if product_name not in radar_grid:
                if self.hdf5_optimizer_config.chunk_size is not None:
                    ds_chunk_size = \
                        optimize_chunk_size(
                            (1,
                            self.hdf5_optimizer_config.chunk_size[0],
                            self.hdf5_optimizer_config.chunk_size[1]),
                            cube_shape)
                    create_dataset_kwargs['chunks'] = ds_chunk_size

                ds = radar_grid.require_dataset(name=product_name,
                            shape=cube_shape,
                            dtype=np.float64,
                            **create_dataset_kwargs)

                ds.attrs['_FillValue'] = np.nan
                ds.attrs['description'] = np.bytes_(descr)
                ds.attrs['units'] = Units.radian
                ds.attrs['grid_mapping'] = np.bytes_('projection')
                ds.dims[0].attach_scale(zds)
                ds.dims[1].attach_scale(yds)
                ds.dims[2].attach_scale(xds)

    def add_algorithms_to_procinfo_group(self):
        """
        Add the algorithms to processingInformation group
        """
        RUNWWriter.add_algorithms_to_procinfo_group(self)
        L2InSARWriter.add_geocoding_to_algo_group(self)

    def add_parameters_to_procinfo_group(self):
        """
        Add parameters group to processingInformation/parameters group
        """
        RUNWWriter.add_parameters_to_procinfo_group(self)

        # the unwrappedInterfergram group under the processingInformation/parameters
        # group is copied from the RUNW product, but the name in RUNW product is
        # 'interferogram', while in GUNW its name is 'unwrappedInterferogram'. Here
        # is to rename the interfegram group name to unwrappedInterferogram group name
        old_igram_group_name = \
            f"{self.group_paths.ParametersPath}/interferogram"
        new_igram_group_name = \
            f"{self.group_paths.ParametersPath}/unwrappedInterferogram"
        self.move(old_igram_group_name, new_igram_group_name)

        for freq, *_ in get_cfg_freq_pols(self.cfg):
            number_of_azimuth_looks = \
                self[f'{new_igram_group_name}/frequency{freq}/numberOfAzimuthLooks']
            number_of_slant_range_looks = \
                self[f'{new_igram_group_name}/frequency{freq}/numberOfRangeLooks']
            number_of_azimuth_looks.attrs['description'] = \
                np.bytes_('Number of looks applied in the'
                          ' along-track direction to form the'
                          ' unwrapped interferogram')
            number_of_slant_range_looks.attrs['description'] = \
                np.bytes_('Number of looks applied in the'
                          ' slant range direction to form the'
                          ' unwrapped interferogram')

        # the wrappedInterfergram group under the processingInformation/parameters
        # group is copied from the RIFG product, but the name in RIFG product is
        # 'interferogram', while in GUNW its name is 'wrappedInterferogram'. Here
        # is to rename the interfegram group name to wrappedInterferogram group name
        RIFGWriter.add_interferogram_to_procinfo_params_group(self)
        new_igram_group_name = \
            f"{self.group_paths.ParametersPath}/wrappedInterferogram"
        self.move(old_igram_group_name, new_igram_group_name)

        L2InSARWriter.add_geocoding_to_procinfo_params_group(self)

        # Update the descriptions of the reference and secondary
        for rslc_name in ['reference', 'secondary']:
            rslc = self[self.group_paths.ParametersPath][rslc_name]
            rslc['referenceTerrainHeight'].attrs['description'] = \
                np.bytes_("Reference Terrain Height as a function of"
                           f" map coordinates for {rslc_name} RSLC")
            rslc['referenceTerrainHeight'].attrs['units'] = \
                Units.meter

    def add_grids_to_hdf5(self):
        """
        Add grids to HDF5
        """
        L2InSARWriter.add_grids_to_hdf5(self)

        pcfg = self.cfg["processing"]
        geogrids = pcfg["geocode"]["geogrids"]
        wrapped_igram_geogrids = pcfg["geocode"]["wrapped_igram_geogrids"]

        grids_val = np.bytes_("projection")

        # Only add the common fields such as list of polarizations, pixel offsets, and center frequency
        for freq, pol_list, _ in get_cfg_freq_pols(self.cfg):
            # Create the swath group
            grids_freq_group_name = (
                f"{self.group_paths.GridsPath}/frequency{freq}"
            )
            grids_freq_group = self.require_group(grids_freq_group_name)

            # Create the pixeloffsets group
            offset_group_name = f"{grids_freq_group_name}/pixelOffsets"
            self.require_group(offset_group_name)

            rslc_freq_group = self.ref_h5py_file_obj[
                f"{self.ref_rslc.SwathPath}/frequency{freq}"
            ]

            unwrapped_geogrids = geogrids[freq]
            wrapped_geogrids = wrapped_igram_geogrids[freq]

            # shape of the unwrapped phase
            unwrapped_shape = (
                unwrapped_geogrids.length,
                unwrapped_geogrids.width,
            )

            # shape of the wrapped interferogram
            wrapped_shape = (
                wrapped_geogrids.length,
                wrapped_geogrids.width,
            )

            unwrapped_group_name = \
                f"{grids_freq_group_name}/unwrappedInterferogram"

            wrapped_group_name = \
                f"{grids_freq_group_name}/wrappedInterferogram"

            pixeloffsets_group_name = \
                f"{grids_freq_group_name}/pixelOffsets"

            # Create the mask layer for each group
            for ds_group_name, ds_geogrid in zip([unwrapped_group_name,
                                                  wrapped_group_name,
                                                  pixeloffsets_group_name],
                                                 [unwrapped_geogrids,
                                                  wrapped_geogrids,
                                                  unwrapped_geogrids]):

                ds_group = self.require_group(ds_group_name)

                # set the geo information for the mask
                yds, xds = set_get_geo_info(
                    self,
                    ds_group_name,
                    ds_geogrid,
                )

                self._create_2d_dataset(
                    ds_group,
                    "mask",
                    (ds_geogrid.length,
                     ds_geogrid.width),
                    np.uint8,
                    ("Combination of water mask and a mask of subswaths of valid samples"
                     " in the reference RSLC and geometrically-coregistered secondary RSLC."
                     " Each pixel value is a three-digit number:"
                     " the most significant digit represents the water flag of that pixel in the reference RSLC,"
                     " where 1 is water and 0 is non-water;"
                     " the second digit represents the subswath number of that pixel in the reference RSLC;"
                     " the least-significant digit represents the subswath number of that pixel in the secondary RSLC."
                     " A value of '0' in either subswath digit indicates an invalid sample in the corresponding RSLC"),
                    grid_mapping=grids_val,
                    xds=xds,
                    yds=yds,
                    fill_value=255,
                )
            ds_group['mask'].attrs['valid_min'] = 0
            ds_group['mask'].attrs['percentage_water'] = 0.0

            for pol in pol_list:
                unwrapped_pol_name = f"{unwrapped_group_name}/{pol}"
                unwrapped_pol_group = self.require_group(unwrapped_pol_name)

                yds, xds = set_get_geo_info(
                    self,
                    unwrapped_pol_name,
                    unwrapped_geogrids,
                )

                #unwrapped dataset parameters as tuples in the following
                #order: dataset name, data type, description, and units
                unwrapped_ds_params = [
                    ("coherenceMagnitude", np.float32,
                     f"Coherence magnitude between {pol} layers",
                     Units.unitless),
                    ("connectedComponents", np.uint16,
                     f"Connected components for {pol} layer",
                     Units.unitless),
                    ("ionospherePhaseScreen", np.float32,
                     "Ionosphere phase screen",
                     Units.radian),
                    ("ionospherePhaseScreenUncertainty", np.float32,
                     "Uncertainty of the ionosphere phase screen",
                     "radians"),
                    ("unwrappedPhase", np.float32,
                    f"Unwrapped interferogram between {pol} layers",
                     Units.radian),
                ]

                for ds_param in unwrapped_ds_params:
                    ds_name, ds_datatype, ds_description, ds_unit\
                        = ds_param
                    self._create_2d_dataset(
                        unwrapped_pol_group,
                        ds_name,
                        unwrapped_shape,
                        ds_datatype,
                        ds_description,
                        ds_unit,
                        grids_val,
                        xds=xds,
                        yds=yds,
                    )

                wrapped_pol_name = f"{wrapped_group_name}/{pol}"
                wrapped_pol_group = self.require_group(wrapped_pol_name)

                yds, xds = set_get_geo_info(
                    self,
                    wrapped_pol_name,
                    wrapped_geogrids,
                )

                #wrapped dataset parameters as tuples in the following
                #order: the dataset name,data type, description, and units
                wrapped_ds_params = [
                    ("coherenceMagnitude", np.float32,
                     f"Coherence magnitude between {pol} layers",
                     Units.unitless),
                    ("wrappedInterferogram", np.complex64,
                     f"Complex wrapped interferogram between {pol} layers",
                     Units.unitless),
                ]

                for ds_param in wrapped_ds_params:
                    ds_name, ds_datatype, ds_description, ds_unit\
                        = ds_param
                    self._create_2d_dataset(
                        wrapped_pol_group,
                        ds_name,
                        wrapped_shape,
                        ds_datatype,
                        ds_description,
                        ds_unit,
                        grids_val,
                        xds=xds,
                        yds=yds,
                    )

                pixeloffsets_pol_name = f"{pixeloffsets_group_name}/{pol}"
                pixeloffsets_pol_group = self.require_group(
                    pixeloffsets_pol_name
                )

                yds, xds = set_get_geo_info(
                    self,
                    pixeloffsets_pol_name,
                    unwrapped_geogrids,
                )

                # pixel offsets dataset parameters as tuples in the following
                # order: dataset name,data type, description, and units
                pixel_offsets_ds_params = [
                    ("alongTrackOffset", np.float32,
                     "Along-track offset",
                     Units.meter),
                    ("correlationSurfacePeak", np.float32,
                     "Normalized cross-correlation surface peak",
                     Units.unitless),
                    ("slantRangeOffset", np.float32,
                     "Slant range offset",
                     Units.meter),
                ]

                for ds_param in pixel_offsets_ds_params:
                    ds_name, ds_datatype, ds_description, ds_unit\
                        = ds_param
                    self._create_2d_dataset(
                        pixeloffsets_pol_group,
                        ds_name,
                        unwrapped_shape,
                        ds_datatype,
                        ds_description,
                        ds_unit,
                        grids_val,
                        xds=xds,
                        yds=yds,
                    )
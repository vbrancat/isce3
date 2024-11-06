import os

import numpy as np
from isce3.core import LUT2d
from nisar.workflows import geo2rdr, rdr2geo
from nisar.workflows.h5_prep import add_geolocation_grid_cubes_to_hdf5
from nisar.workflows.helpers import (get_cfg_freq_pols, get_offset_radar_grid,
                                     get_pixel_offsets_dataset_shape,
                                     get_pixel_offsets_params)

from .dataset_params import DatasetParams, add_dataset_and_attrs
from .InSAR_base_writer import InSARBaseWriter
from .product_paths import L1GroupsPaths
from .units import Units
from .utils import (extract_datetime_from_string, generate_insar_subswath_mask,
                    get_geolocation_grid_cube_obj)


class L1InSARWriter(InSARBaseWriter):
    """
    InSAR Level 1 products writer inherit from the InSARBaseWriter
    The Level 1 products include ROFF, RUNW, and RIFG.

    Attributes
    ----------
    igram_range_looks : int
        range looks for the interferogram
    igram_azimuth_looks : int
        azimuth looks for the interferogram
    """
    def __init__(self, **kwds):
        """
        Constructor for InSAR L1 products (RIFG, RUNW, and ROFF).
        """
        super().__init__(**kwds)

        # Level 1 product group path
        self.group_paths = L1GroupsPaths()

        # Range and azimuth looks that will be performed on the interfergoram
        self.igram_range_looks = 1
        self.igram_azimuth_looks = 1

    def save_to_hdf5(self):
        """
        Write the attributes and groups to the HDF5
        """
        super().save_to_hdf5()

        self.add_geolocation_grid_cubes()
        self.add_swaths_to_hdf5()

    def add_geolocation_grid_cubes(self):
        """
        Add the geolocation grid cubes
        """
        # Pull the heights and espg from the radar_grid_cubes group
        # in the runconfig
        radar_grid_cfg = self.cfg["processing"]["radar_grid_cubes"]
        heights = np.array(radar_grid_cfg["heights"]).astype(np.float64)
        epsg = radar_grid_cfg["output_epsg"]

        # Retrieve the group
        geolocationGrid_path = self.group_paths.GeolocationGridPath

        # Pull the radar frequency
        cube_freq = "A" if "A" in self.freq_pols else "B"
        grid_doppler = LUT2d()
        native_doppler = self.ref_rslc.getDopplerCentroid(
            frequency=cube_freq
        )
        native_doppler.bounds_error = False
        geo2rdr_params = dict(threshold_geo2rdr=1e-8,
                              numiter_geo2rdr=50,
                              delta_range=10)

        geolocation_radargrid = get_geolocation_grid_cube_obj(self.cfg)

        # Add geolocation grid cubes to hdf5
        if self.hdf5_optimizer_config.chunk_size is None:
            chunk_size = None
        else:
            chunk_size = (1,
                          self.hdf5_optimizer_config.chunk_size[0],
                          self.hdf5_optimizer_config.chunk_size[1])

        add_geolocation_grid_cubes_to_hdf5(
            self,
            geolocationGrid_path,
            geolocation_radargrid,
            heights,
            self.ref_orbit,
            native_doppler,
            grid_doppler,
            epsg,
            chunk_size=chunk_size,
            compression_enabled=\
                self.hdf5_optimizer_config.compression_enabled,
            compression_type=\
                self.hdf5_optimizer_config.compression_type,
            compression_level=\
                self.hdf5_optimizer_config.compression_level,
            shuffle_filter=\
                self.hdf5_optimizer_config.shuffle_filter,
            **geo2rdr_params,
        )

        geolocationGrid_group = self.require_group(geolocationGrid_path)
        # Add baseline to the geolocation grid
        self.add_baseline_info_to_cubes(geolocationGrid_group,
                                        geolocation_radargrid,
                                        is_geogrid=False)


        geolocation_grid_group = self[geolocationGrid_path]

        geolocation_grid_group['epsg'][...] = \
            geolocation_grid_group['epsg'][()].astype(np.uint32)
        geolocation_grid_group['epsg'].attrs['description'] = \
            np.bytes_("EPSG code corresponding to the coordinate system"
                       " used for representing the geolocation grid")
        geolocation_grid_group['losUnitVectorX'].attrs['units'] = Units.unitless
        geolocation_grid_group['losUnitVectorY'].attrs['units'] = Units.unitless

        geolocation_grid_group['alongTrackUnitVectorX'].attrs['units'] = \
            Units.unitless
        geolocation_grid_group['alongTrackUnitVectorY'].attrs['units'] = \
            Units.unitless
        geolocation_grid_group['heightAboveEllipsoid'][...] = \
            geolocation_grid_group['heightAboveEllipsoid'][()].astype(np.float64)

        zero_dopp_time_units = geolocation_grid_group['zeroDopplerTime'].attrs['units']
        zero_dopp_time_units = extract_datetime_from_string(str(zero_dopp_time_units),
                                                            'seconds since ')
        if zero_dopp_time_units is not None:
            geolocation_grid_group['zeroDopplerTime'].attrs['units']\
                = np.bytes_(zero_dopp_time_units)

    def add_algorithms_to_procinfo_group(self):
        """
        Add the algorithms group to the processingInformation group
        """
        super().add_algorithms_to_procinfo_group()
        self.add_coregistration_to_algo_group()

    def add_interferogram_to_procinfo_params_group(self):
        """
        Add the interferogram group to "processingInformation/parameters group"
        """
        proc_cfg_crossmul = self.cfg["processing"]["crossmul"]
        range_filter = proc_cfg_crossmul["common_band_range_filter"]
        azimuth_filter = proc_cfg_crossmul["common_band_azimuth_filter"]

        flatten = proc_cfg_crossmul["flatten"]

        interferogram_ds_params = [
            DatasetParams(
                "commonBandRangeFilterApplied",
                np.bytes_(str(range_filter)),
                (
                    "Flag to indicate if common band range filter has been"
                    " applied"
                ),
            ),
            DatasetParams(
                "commonBandAzimuthFilterApplied",
                np.bytes_(str(azimuth_filter)),
                (
                    "Flag to indicate if common band azimuth filter has been"
                    " applied"
                ),
            ),
            DatasetParams(
                "ellipsoidalFlatteningApplied",
                np.bytes_(str(flatten)),
                (
                    "Flag to indicate if the interferometric phase has been "
                    "flattened with respect to a zero height ellipsoid"
                ),
            ),
            DatasetParams(
                "topographicFlatteningApplied",
                np.bytes_(str(flatten)),
                (
                    "Flag to indicate if the interferometric phase has been "
                    "flattened with respect to topographic height using a DEM"
                ),
            ),
            DatasetParams(
                "numberOfRangeLooks",
                np.uint32(self.igram_range_looks),
                (
                    "Number of looks applied in the slant range direction to"
                    " form the wrapped interferogram"
                ),
                {
                    "units": Units.unitless,
                },
            ),
            DatasetParams(
                "numberOfAzimuthLooks",
                np.uint32(self.igram_azimuth_looks),
                (
                    "Number of looks applied in the along-track direction to"
                    " form the wrapped interferogram"
                ),
                {
                    "units": Units.unitless,
                },
            ),
        ]

        for freq, *_ in get_cfg_freq_pols(self.cfg):
            bandwidth_group_path = f"{self.ref_rslc.SwathPath}/frequency{freq}"
            bandwidth_group = self.ref_h5py_file_obj[bandwidth_group_path]

            igram_group_name = \
                f"{self.group_paths.ParametersPath}/interferogram/frequency{freq}"
            igram_group = self.require_group(igram_group_name)

            # TODO: the azimuthBandwidth and rangeBandwidth are placeholders heres,
            # and copied from the bandpassed RSLC data.
            # those should be updated in the crossmul module.
            bandwidth_group.copy(
                "processedAzimuthBandwidth",
                igram_group,
                "azimuthBandwidth",
            )
            igram_group['azimuthBandwidth'].attrs['description'] = \
                np.bytes_("Processed azimuth bandwidth for frequency " + \
                           f"{freq} interferometric layers")
            igram_group['azimuthBandwidth'].attrs['units'] = Units.hertz

            bandwidth_group.copy(
                "processedRangeBandwidth",
                igram_group,
                "rangeBandwidth",
            )
            igram_group['rangeBandwidth'].attrs['description'] = \
                np.bytes_("Processed slant range bandwidth for frequency " + \
                           f"{freq} interferometric layers")
            igram_group['rangeBandwidth'].attrs['units'] = Units.hertz

            for ds_param in interferogram_ds_params:
                add_dataset_and_attrs(igram_group, ds_param)

    def add_parameters_to_procinfo_group(self):
        """
        Add the parameters group to the "processingInformation" group
        """
        super().add_parameters_to_procinfo_group()
        self.add_interferogram_to_procinfo_params_group()
        self.add_pixeloffsets_to_procinfo_params_group()

    def _get_interferogram_dataset_shape(self, freq : str, pol : str):
        """
        Get the interfergraom dataset shape at a given frequency and polarization

        Parameters
        ---------
        freq: str
            frequency ('A' or 'B')
        pol : str
            polarization ('HH', 'HV', 'VH', or 'VV')

        Returns
        ----------
        igram_shape : tuple
             interfergraom shape
        """
        # get the RSLC lines and columns
        slc_dset = \
            self.ref_h5py_file_obj[
                f"{self.ref_rslc.SwathPath}/frequency{freq}/{pol}"]
        slc_lines, slc_cols = slc_dset.shape

        # shape of the interferogram product
        igram_shape = (slc_lines // self.igram_azimuth_looks,
                       slc_cols // self.igram_range_looks)

        return igram_shape


    def _add_datasets_to_pixel_offset_group(self):
        """
        Add datasets to pixel offsets group
        """
        pcfg = self.cfg['processing']
        for freq, pol_list, _ in get_cfg_freq_pols(self.cfg):
            # create the swath group
            swaths_freq_group_name = \
                f"{self.group_paths.SwathsPath}/frequency{freq}"

            # get the shape of offset product
            off_shape = get_pixel_offsets_dataset_shape(self.cfg, freq)

            # add the interferogram and pixelOffsets groups to the polarization group
            for pol in pol_list:

                offset_pol_group_name = (
                    f"{swaths_freq_group_name}/pixelOffsets/{pol}"
                )
                offset_pol_group = self.require_group(offset_pol_group_name)

                # pixelOffsets datasets
                pixel_offsets_ds_params = [
                    (
                        "alongTrackOffset",
                        "Along-track offset",
                        Units.meter,
                    ),
                    (
                        "correlationSurfacePeak",
                        "Normalized correlation surface peak",
                        Units.unitless,
                    ),
                    (
                        "slantRangeOffset",
                        "Slant range offset",
                        Units.meter,
                    ),
                ]

                for pixel_offsets_ds_param in pixel_offsets_ds_params:
                    ds_name, ds_description, ds_unit = pixel_offsets_ds_param
                    self._create_2d_dataset(
                        offset_pol_group,
                        ds_name,
                        off_shape,
                        np.float32,
                        ds_description,
                        units=ds_unit,
                    )

    def add_pixel_offsets_to_swaths_group(self):
        """
        Add pixel offsets product to swaths group
        """
        is_roff,  margin, rg_start, az_start,\
        rg_skip, az_skip, rg_search, az_search,\
        rg_chip, az_chip, _ = get_pixel_offsets_params(self.cfg)

        for freq, pol_list, _ in get_cfg_freq_pols(self.cfg):
            # Create the swath group
            swaths_freq_group_name = \
                f"{self.group_paths.SwathsPath}/frequency{freq}"

            swaths_freq_group = self.require_group(swaths_freq_group_name)

            # center frequency and sub swaths groups of the RSLC
            rslc_swaths_group = \
                self.ref_h5py_file_obj[f"{self.ref_rslc.SwathPath}"]

            rslc_freq_group = self.ref_h5py_file_obj[
                f"{self.ref_rslc.SwathPath}/frequency{freq}"
            ]

            # Update the offset radar grid
            rslc_radar_grid = self.ref_rslc.getRadarGrid(freq)
            off_radargrid = get_offset_radar_grid(self.cfg, rslc_radar_grid)

            # shape of offset product
            off_length, off_width = off_radargrid.length, off_radargrid.width

            # add the slantRange, zeroDopplerTime, and their spacings to pixel offset group
            # where the starting range/sensing start of the offsets radar grid
            # is at the center of the matching window
            offset_slant_range = np.array(
                [off_radargrid.starting_range +
                 i*off_radargrid.range_pixel_spacing
                 for i in range(off_radargrid.width)])
            offset_zero_doppler_time = np.array(
                [off_radargrid.sensing_start +
                 i/off_radargrid.prf
                 for i in range(off_radargrid.length)])

            zero_dopp_time_units = \
                rslc_swaths_group["zeroDopplerTime"].attrs['units']
            time_str = extract_datetime_from_string(str(zero_dopp_time_units),
                                                    'seconds since ')
            if time_str is not None:
                zero_dopp_time_units = time_str


            ds_offsets_params = [
                DatasetParams(
                    "slantRange",
                    offset_slant_range,
                    "Slant range vector",
                    {'units': Units.meter},
                ),
                DatasetParams(
                    "zeroDopplerTime",
                    offset_zero_doppler_time,
                    "Zero Doppler azimuth time since UTC epoch vector",
                    {'units': zero_dopp_time_units},
                ),
                DatasetParams(
                    "zeroDopplerTimeSpacing",
                    1.0/off_radargrid.prf,
                    "Along-track spacing of the offset grid",
                    {'units': Units.second},
                ),
                DatasetParams(
                    "slantRangeSpacing",
                    off_radargrid.range_pixel_spacing,
                    "Slant range spacing of the offset grid",
                    {'units': Units.meter},
                ),
                DatasetParams(
                    "sceneCenterAlongTrackSpacing",
                    rslc_freq_group["sceneCenterAlongTrackSpacing"][()]
                    * az_skip,
                    (
                        "Nominal along-track spacing in meters between"
                        " consecutive lines near mid-swath of the product images"
                    ),
                    {"units": Units.meter},
                ),
                DatasetParams(
                    "sceneCenterGroundRangeSpacing",
                    rslc_freq_group["sceneCenterGroundRangeSpacing"][()]
                    * rg_skip,
                    (
                        "Nominal ground range spacing in meters between"
                        " consecutive pixels near mid-swath of the product images"
                    ),
                    {"units": Units.meter},
                ),
            ]
            offset_group_name = f"{swaths_freq_group_name}/pixelOffsets"
            offset_group = self.require_group(offset_group_name)
            for ds_param in ds_offsets_params:
                add_dataset_and_attrs(offset_group, ds_param)

            # add the digital elevation model layer to the pixelOffsets group
            self._create_2d_dataset(offset_group,
                                    'digitalElevationModel',
                                    shape=(off_length, off_width),
                                    dtype=np.float32,
                                    description=("Digital Elevation Model (DEM) in radar coordinates."
                                                 " This dataset is produced using Copernicus WorldDEM-30"
                                                 " Copyright DLR e.V. 2010-2014 and Copyright Airbus Defence and"
                                                 " Space GmbH 2014-2018 provided under COPERNICUS by the European Union and ESA;"
                                                 " all rights reserved. This dataset is generated by referencing the"
                                                 " Copernicus DEM elevations to the WGS84 ellipsoid and"
                                                 " projecting them onto a range/Doppler grid"),
                                    units=Units.meter)

            # temporarily add stats attributes to the 'digitalElevationModel' dataset
            # TODO: remove this placeholder for setting stats values
            # to 0.0 once the actual values are being computed.
            for attr in ['mean_value', 'min_value',
                         'max_value', 'sample_stddev']:
                offset_group['digitalElevationModel'].attrs[attr] = 0.0

            # add the subswath mask layer to the pixel offset group
            self._create_2d_dataset(offset_group,
                                    'mask',
                                    shape=(off_length, off_width),
                                    dtype=np.uint8,
                                    description=("Mask indicating the subswaths of valid samples in the reference RSLC"
                                                 " and geometrically-coregistered secondary RSLC."
                                                 " Each pixel value is a two-digit number:"
                                                 " the least significant digit represents the"
                                                 " subswath number of that pixel in the secondary RSLC,"
                                                 " and the most significant digit represents"
                                                 " the subswath number of that pixel in the reference RSLC."
                                                 " A value of '0' in either digit indicates an invalid sample"
                                                 " in the corresponding RSLC"),
                                    fill_value=255)
            offset_group['mask'].attrs['long_name'] = np.bytes_("Valid samples subswath mask")
            offset_group['mask'].attrs['valid_min'] = 0

            range_offset_path = \
                os.path.join( self.topo_path,
                                f'geo2rdr/freq{freq}/range.off')
            azimuth_offset_path = \
                os.path.join( self.topo_path,
                                f'geo2rdr/freq{freq}/azimuth.off')

            # If there are no offset products, run the rdr2geo and
            # geo2rdr to generate the offsets products
            if ((not os.path.exists(range_offset_path)) or
                (not os.path.exists(azimuth_offset_path))):
                rdr2geo.run(self.cfg)
                geo2rdr.run(self.cfg)

            # get the nearest neighbor slant range and azimuth index in the RSLC radar grid
            # to generate subswath mask
            rg_idx = np.round([rslc_radar_grid.slant_range_index(rg)
                               for rg in offset_slant_range])
            az_idx = np.round([rslc_radar_grid.azimuth_index(az)
                               for az in offset_zero_doppler_time])

            offset_group['mask'][...] = \
                generate_insar_subswath_mask(self.ref_rslc,
                                             self.sec_rslc,
                                             range_offset_path,
                                             azimuth_offset_path,
                                             freq,
                                             az_idx,
                                             rg_idx)

        # add the datasets to pixel offsets group
        self._add_datasets_to_pixel_offset_group()

    def add_interferogram_to_swaths_group(self):
        """
        Add the interferogram group to the swaths group
        """
        pcfg = self.cfg['processing']
        for freq, pol_list, _ in get_cfg_freq_pols(self.cfg):
            # Create the swath group
            swaths_freq_group_name = (
                f"{self.group_paths.SwathsPath}/frequency{freq}"
            )
            swaths_freq_group = self.require_group(swaths_freq_group_name)

            # center frequency and sub swaths groups of the RSLC
            rslc_swaths_group = \
                self.ref_h5py_file_obj[f"{self.ref_rslc.SwathPath}"]

            rslc_freq_group = self.ref_h5py_file_obj[
                f"{self.ref_rslc.SwathPath}/frequency{freq}"
            ]

            # shape of the interferogram product
            igram_shape = self._get_interferogram_dataset_shape(freq,
                                                                pol_list[0])

            # add the slantRange, zeroDopplerTime, and their spacings to inteferogram group

            rslc_radar_grid = self.ref_rslc.getRadarGrid(freq)

            # multilook the radar grid of the reference RSLC to
            # get the radar grid of the interferogram
            igram_radargrid = rslc_radar_grid.multilook(
                self.igram_azimuth_looks,
                self.igram_range_looks)

            # compute the slant range and zero doppler time vector
            # for the interferogram
            igram_slant_range = np.array(
                [igram_radargrid.starting_range +
                 i*igram_radargrid.range_pixel_spacing
                 for i in range(igram_radargrid.width)])
            igram_zero_doppler_time = np.array(
                [igram_radargrid.sensing_start +
                 i/igram_radargrid.prf
                 for i in range(igram_radargrid.length)])

            zero_dopp_time_units = \
                rslc_swaths_group["zeroDopplerTime"].attrs['units']
            time_str = extract_datetime_from_string(str(zero_dopp_time_units),
                                                    'seconds since ')
            if time_str is not None:
                zero_dopp_time_units = time_str

            ds_igram_params = [
                DatasetParams(
                    "slantRange",
                    igram_slant_range,
                    "Slant range vector",
                    {'units': Units.meter},
                ),
                DatasetParams(
                    "zeroDopplerTime",
                    igram_zero_doppler_time,
                    "Zero Doppler azimuth time since UTC epoch vector",
                    {'units': zero_dopp_time_units},
                ),
                DatasetParams(
                    "zeroDopplerTimeSpacing",
                    1.0/igram_radargrid.prf,
                    (
                        "Time interval in the along-track direction for raster"
                        " layers. This is same as the spacing between"
                        " consecutive entries in the zeroDopplerTime array"
                    ),
                    {'units': Units.second},
                ),
                DatasetParams(
                    "slantRangeSpacing",
                    igram_radargrid.range_pixel_spacing,
                    (
                        "Slant range spacing of grid. Same as difference"
                        " between consecutive samples in slantRange array"
                    ),
                    {'units': Units.meter},
                ),
                DatasetParams(
                    "sceneCenterAlongTrackSpacing",
                    rslc_freq_group["sceneCenterAlongTrackSpacing"][()]
                    * self.igram_azimuth_looks,
                    (
                        "Nominal along-track spacing in meters "
                        "between consecutive lines near mid-swath of the product images"
                    ),
                    {"units": Units.meter},
                ),
                DatasetParams(
                    "sceneCenterGroundRangeSpacing",
                    rslc_freq_group["sceneCenterGroundRangeSpacing"][()]
                    * self.igram_range_looks,
                    (
                        "Nominal ground range spacing in meters between "
                        "consecutive pixels near mid-swath of the product images"
                    ),
                    {"units": Units.meter},
                ),
            ]

            igram_group_name = f"{swaths_freq_group_name}/interferogram"
            igram_group = self.require_group(igram_group_name)
            for ds_param in ds_igram_params:
                add_dataset_and_attrs(igram_group, ds_param)

            # add the digital elevation model layer to the interferogram group
            self._create_2d_dataset(igram_group,
                                    'digitalElevationModel',
                                    shape=igram_shape,
                                    dtype=np.float32,
                                    description=("Digital Elevation Model (DEM) in radar coordinates."
                                                 " This dataset is produced using Copernicus WorldDEM-30"
                                                 " Copyright DLR e.V. 2010-2014 and Copyright Airbus Defence and"
                                                 " Space GmbH 2014-2018 provided under COPERNICUS by the European Union and ESA;"
                                                 " all rights reserved. This dataset is generated by referencing the"
                                                 " Copernicus DEM elevations to the WGS84 ellipsoid and"
                                                 " projecting them onto a range/Doppler grid"),
                                    units=Units.meter)

            # temporarily add stats attributes to the 'digitalElevationModel' dataset
            # TODO: remove this placeholder for setting stats values
            # to 0.0 once the actual values are being computed.
            for attr in ['mean_value', 'min_value',
                         'max_value', 'sample_stddev']:
                igram_group['digitalElevationModel'].attrs[attr] = 0.0

            # add the subswath mask layer to the interferogram group
            self._create_2d_dataset(igram_group,
                                    'mask',
                                    shape=igram_shape,
                                    dtype=np.uint8,
                                    description=("Mask indicating the subswaths of valid samples in the reference RSLC"
                                                 " and geometrically-coregistered secondary RSLC."
                                                 " Each pixel value is a two-digit number:"
                                                 " the least significant digit represents the"
                                                 " subswath number of that pixel in the secondary RSLC,"
                                                 " and the most significant digit represents"
                                                 " the subswath number of that pixel in the reference RSLC."
                                                 " A value of '0' in either digit indicates an invalid sample"
                                                 " in the corresponding RSLC"),
                                    fill_value=255)
            igram_group['mask'].attrs['valid_min'] = 0
            igram_group['mask'].attrs['long_name'] = np.bytes_("Valid samples subswath mask")

            range_offset_path = \
                os.path.join(self.topo_path,
                                f'geo2rdr/freq{freq}/range.off')
            azimuth_offset_path = \
                os.path.join(self.topo_path,
                                f'geo2rdr/freq{freq}/azimuth.off')

            # If there are no offset products, run the rdr2geo and
            # geo2rdr to generate the offsets products
            if ((not os.path.exists(range_offset_path)) or
                (not os.path.exists(azimuth_offset_path))):
                rdr2geo.run(self.cfg)
                geo2rdr.run(self.cfg)

            # get the nearest neighbor slant range and azimuth index in the RSLC radar grid
            # to generate subswath mask
            rg_idx = np.round([rslc_radar_grid.slant_range_index(rg)
                               for rg in igram_slant_range])
            az_idx = np.round([rslc_radar_grid.azimuth_index(az)
                               for az in igram_zero_doppler_time])

            igram_group['mask'][...] = \
                generate_insar_subswath_mask(self.ref_rslc,
                                             self.sec_rslc,
                                             range_offset_path,
                                             azimuth_offset_path,
                                             freq,
                                             az_idx,
                                             rg_idx)

            # add the interferogram and pixelOffsets groups to the polarization group
            for pol in pol_list:
                igram_pol_group_name = (
                    f"{swaths_freq_group_name}/interferogram/{pol}"
                )
                igram_pol_group = self.require_group(igram_pol_group_name)

                # Interferogram datasets
                igram_ds_params = [
                    (
                        "coherenceMagnitude",
                        np.float32,
                        f"Coherence magnitude between {pol} layers",
                        Units.unitless,
                    ),
                ]

                for igram_ds_param in igram_ds_params:
                    ds_name, ds_dtype, ds_description, ds_unit = igram_ds_param
                    self._create_2d_dataset(
                        igram_pol_group,
                        ds_name,
                        igram_shape,
                        ds_dtype,
                        ds_description,
                        units=ds_unit,
                    )

    def add_swaths_to_hdf5(self):
        """
        Add Swaths to the HDF5
        """
        self.require_group(self.group_paths.SwathsPath)

        # Add the common datasetst to the swaths group
        for freq, pol_list, _ in get_cfg_freq_pols(self.cfg):
            # Create the swath group
            swaths_freq_group_name = (
                f"{self.group_paths.SwathsPath}/frequency{freq}"
            )
            swaths_freq_group = self.require_group(swaths_freq_group_name)

            list_of_pols = DatasetParams(
                "listOfPolarizations",
                np.bytes_(pol_list),
                f"List of processed polarization layers with frequency {freq}",
            )
            add_dataset_and_attrs(swaths_freq_group, list_of_pols)

            rslc_freq_group = self.ref_h5py_file_obj[
                f"{self.ref_rslc.SwathPath}/frequency{freq}"]
            rslc_freq_group.copy(
                "processedCenterFrequency",
                swaths_freq_group,
                "centerFrequency",
            )

            # Add the description and units
            cfreq = swaths_freq_group["centerFrequency"]
            cfreq.attrs['description'] = np.bytes_("Center frequency of"
                                                    " the processed image in hertz")
            cfreq.attrs['units'] = Units.hertz

        self.add_pixel_offsets_to_swaths_group()

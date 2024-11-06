#!/usr/bin/env python3

import os
import time

import h5py
import journal
import numpy as np

import isce3
from isce3.core import crop_external_orbit
from isce3.core.rdr_geo_block_generator import block_generator
from isce3.core.types import (truncate_mantissa, read_c4_dataset_as_c8,
                              to_complex32)
from isce3.io import HDF5OptimizedReader, optimize_chunk_size, compute_page_size

from nisar.products.readers import SLC
from nisar.products.readers.orbit import load_orbit_from_xml
from nisar.workflows.compute_stats import compute_stats_complex_data
from nisar.workflows.h5_prep import (add_radar_grid_cubes_to_hdf5,
                                     prep_gslc_dataset)
from nisar.workflows.geocode_corrections import AzSrgCorrections
from nisar.workflows.gslc_runconfig import GSLCRunConfig
from nisar.workflows.yaml_argparse import YamlArgparse
from nisar.products.writers import GslcWriter
from nisar.workflows.helpers import validate_fs_page_size

def run(cfg):
    '''
    run geocodeSlc according to parameters in cfg dict
    '''
    # pull parameters from cfg
    input_hdf5 = cfg['input_file_group']['input_file_path']
    output_hdf5 = cfg['product_path_group']['sas_output_file']
    freq_pols = cfg['processing']['input_subset']['list_of_frequencies']
    geogrids = cfg['processing']['geocode']['geogrids']
    radar_grid_cubes_geogrid = cfg['processing']['radar_grid_cubes']['geogrid']
    radar_grid_cubes_heights = cfg['processing']['radar_grid_cubes']['heights']
    dem_file = cfg['dynamic_ancillary_file_group']['dem_file']
    tec_file = cfg['dynamic_ancillary_file_group']['tec_file']
    orbit_file = cfg["dynamic_ancillary_file_group"]['orbit_file']
    threshold_geo2rdr = cfg['processing']['geo2rdr']['threshold']
    iteration_geo2rdr = cfg['processing']['geo2rdr']['maxiter']
    columns_per_block = cfg['processing']['blocksize']['x']
    lines_per_block = cfg['processing']['blocksize']['y']
    flatten = cfg['processing']['flatten']

    output_options = cfg['output']
    geogrid_expansion_threshold = 100
    apply_data_driven_correction = cfg['dynamic_ancillary_file_group']['reference_gslc'] is not None

    output_dir = os.path.dirname(os.path.abspath(output_hdf5))
    os.makedirs(output_dir, exist_ok=True)

    # init parameters shared by frequency A and B
    slc = SLC(hdf5file=input_hdf5)

    # if provided, load an external orbit from the runconfig file;
    # otherwise, load the orbit from the RSLC metadata.
    orbit = slc.getOrbit()
    if orbit_file is not None:
        # slc will get first radar grid whose frequency is available.
        # orbit has not frequency dependency.
        external_orbit = load_orbit_from_xml(orbit_file,
                                             slc.getRadarGrid().ref_epoch)

        # Apply 2 mins of padding before / after sensing period when cropping
        # the external orbit.
        # 2 mins of margin is based on the number of IMAGEN TEC samples required for
        # TEC computation, with few more safety margins for possible needs in the future.
        #
        # `7` in the line below is came from the default value for `npad` in
        # `crop_external_orbit()`. See:
        #.../isce3/python/isce3/core/crop_external_orbit.py
        npad = max(int(120.0 / external_orbit.spacing),
                   7)
        orbit = crop_external_orbit(external_orbit, orbit,
                                    npad=npad)

    dem_raster = isce3.io.Raster(dem_file)
    epsg = dem_raster.get_epsg()
    proj = isce3.core.make_projection(epsg)
    ellipsoid = proj.ellipsoid

    # Doppler of the image grid (Zero for NISAR)
    image_grid_doppler = isce3.core.LUT2d()

    info_channel = journal.info("gslc_array.run")
    info_channel.log("starting geocode SLC")
    warning_channel = journal.warning("gslc_array.run")

    fs_dict = dict(fs_strategy=output_options['fs_strategy'],
                   fs_persist=True)

    if output_options['fs_strategy'] == 'page':
        # Decide what page size to use based on geogrid shape.
        # If user provides the page size, then the workflow takes that value.
        # Otherwise, the workflow decides the pagesize automatically.

        geogrid_freq_and_size = ((geogrid[0], geogrid[1].length * geogrid[1].width) for
                                 geogrid in geogrids.items())

        smallest_freq = min(geogrid_freq_and_size, key=lambda x: x[1])[0]

        output_gslc_shape = (geogrids[smallest_freq].length,
                             geogrids[smallest_freq].width)
        optimal_chunk_size = optimize_chunk_size(output_options['chunk_size'],
                                                 output_gslc_shape)

        # Populate `fs_page_size` in file space argument dict.
        # Determine the page size. Use the value provided by the user if available;
        # otherwise, automatically calculate it based on the memory footprint of the chunk.
        if output_options['fs_page_size']:
            validate_fs_page_size(output_options['fs_page_size'], optimal_chunk_size)
            fs_dict['fs_page_size'] = output_options['fs_page_size']
        else:
            gslc_dtype = cfg['output']['data_type'].split('_')[0]
            gslc_dtype_size = np.dtype(gslc_dtype).itemsize
            chunk_memory_footprint = np.prod(output_options['chunk_size']) * gslc_dtype_size
            fs_dict['fs_page_size'] = compute_page_size(chunk_memory_footprint)

    elif output_options['fs_strategy'] != 'page' and output_options['fs_page_size'] is not None:
            warning_channel.log('fs_page_size is relevant only when '
                                'fs_strategy is page. Ignoring the page size provided by user.')

    t_all = time.perf_counter()
    with h5py.File(output_hdf5, 'w', **fs_dict) as dst_h5, \
            HDF5OptimizedReader(name=input_hdf5, mode='r', libver='latest', swmr=True) as src_h5:

        prep_gslc_dataset(cfg, 'GSLC', dst_h5)
        for freq, pol_list in freq_pols.items():
            root_ds = f'/science/LSAR/GSLC/grids/frequency{freq}'
            radar_grid = slc.getRadarGrid(freq)
            geo_grid = geogrids[freq]

            # get doppler centroid
            native_doppler = slc.getDopplerCentroid(frequency=freq)

            # get azimuth and slant range geocoding corrections
            az_srg_corrections = AzSrgCorrections(cfg, slc, freq, orbit)
            az_correction = az_srg_corrections.az_correction_lut
            srg_correction = az_srg_corrections.slant_range_correction_lut

            # write corrections to HDF5
            proc_info_path = "/science/LSAR/GSLC/metadata/processingInformation"
            proc_info_group = dst_h5.require_group(proc_info_path)
            az_srg_corrections.write_corrections_hdf5(proc_info_group)

            # get subswaths for current freq SLC from its Swath
            sub_swaths = isce3.product.Swath(input_hdf5, freq).sub_swaths()

            # initialize source/rslc and destination/gslc datasets
            rslc_datasets = []
            gslc_datasets = []
            for polarization in pol_list:
                # check the datatype of RSLC
                is_complex32 = slc.is_dataset_complex32(freq, polarization)

                # path and dataset to rdr SLC data in HDF5
                rslc_ds_path = slc.slcPath(freq, polarization)
                rslc_datasets.append(src_h5[rslc_ds_path])

                # path and dataset to geo SLC data in HDF5
                dataset_path = f'/{root_ds}/{polarization}'
                gslc_datasets.append(dst_h5[dataset_path])

            # retrieve dataset to geo SLC mask in HDF5
            mask_dataset_path = f'/{root_ds}/mask'
            mask_dataset = dst_h5[mask_dataset_path]

            # loop over geogrid blocks skipping those without radar data
            # where block_generator skips blocks where no radar data is found
            for (rdr_blk_slice, geo_blk_slice, geo_blk_shape, blk_geo_grid) in \
                 block_generator(geo_grid, radar_grid, orbit, dem_raster,
                                 lines_per_block, columns_per_block,
                                 geogrid_expansion_threshold):

                # unpack block parameters
                az_first = rdr_blk_slice[0].start
                rg_first = rdr_blk_slice[1].start

                # init input/rslc and output/gslc blocks for each polarization
                gslc_data_blks = []
                rslc_data_blks = []
                for rslc_dataset in rslc_datasets:
                    # extract RSLC data block/array
                    if is_complex32:
                        rslc_data_blks.append(
                            read_c4_dataset_as_c8(rslc_dataset, rdr_blk_slice))
                    else:
                        rslc_data_blks.append(rslc_dataset[rdr_blk_slice])

                    # prepare zero'd GSLC data block/array
                    gslc_data_blks.append(
                        np.zeros(geo_blk_shape, dtype=np.complex64))

                # init geocoded mask block/array with 255 as invalid value
                mask_data_blk = np.full(geo_blk_shape, 255, dtype=np.ubyte)

                # run geocodeSlc
                isce3.geocode.geocode_slc(gslc_data_blks, rslc_data_blks,
                                          dem_raster, radar_grid, blk_geo_grid,
                                          orbit, native_doppler,
                                          image_grid_doppler, ellipsoid,
                                          threshold_geo2rdr,
                                          iteration_geo2rdr,
                                          sliced_radargrid=radar_grid,                                           
                                          mask_block=mask_data_blk,
                                          first_azimuth_line=az_first,
                                          first_range_sample=rg_first,
                                          flatten=flatten,
                                          az_time_correction=az_correction,
                                          srange_correction=srg_correction,
                                          subswaths=sub_swaths)

                # write geocoded blocks to respective HDF5 datasets
                for gslc_dataset, gslc_data_blk in zip(gslc_datasets,
                                                       gslc_data_blks):
                    # only convert/modify output if type not 'complex64'
                    # do nothing if type is 'complex64'
                    output_type = cfg['output']['data_type']
                    if output_type == 'complex32':
                        gslc_data_blk = to_complex32(gslc_data_blk)
                    if output_type == 'complex64_zero_mantissa':
                        # use default nonzero_mantissa_bits = 10 below
                        truncate_mantissa(gslc_data_blk)

                    # write to GSLC block HDF5
                    gslc_dataset.write_direct(gslc_data_blk,
                                              dest_sel=geo_blk_slice)

                # write to mask block HDF5
                mask_dataset.write_direct(mask_data_blk, dest_sel=geo_blk_slice)

            # loop over polarizations and compute statistics
            for gslc_dataset in gslc_datasets:
                gslc_raster = isce3.io.Raster(f"IH5:::ID={gslc_dataset.id.id}".encode("utf-8"), update=True)
                compute_stats_complex_data(gslc_raster, gslc_dataset)

        cube_geogrid = isce3.product.GeoGridParameters(
            start_x=radar_grid_cubes_geogrid.start_x,
            start_y=radar_grid_cubes_geogrid.start_y,
            spacing_x=radar_grid_cubes_geogrid.spacing_x,
            spacing_y=radar_grid_cubes_geogrid.spacing_y,
            width=int(radar_grid_cubes_geogrid.width),
            length=int(radar_grid_cubes_geogrid.length),
            epsg=radar_grid_cubes_geogrid.epsg)

        cube_group_name = '/science/LSAR/GSLC/metadata/radarGrid'

        # if available use frequency A to get radar grid and native doppler
        # else use frequency B
        cube_freq = "A" if "A" in freq_pols else "B"
        cube_rdr_grid = slc.getRadarGrid(cube_freq)
        cube_native_doppler = slc.getDopplerCentroid(frequency=cube_freq)
        cube_native_doppler.bounds_error = False
        # The native-Doppler LUT bounds error is turned off to
        # computer cubes values outside radar-grid boundaries
        add_radar_grid_cubes_to_hdf5(dst_h5, cube_group_name,
                                     cube_geogrid, radar_grid_cubes_heights,
                                     cube_rdr_grid, orbit, cube_native_doppler,
                                     image_grid_doppler, threshold_geo2rdr,
                                     iteration_geo2rdr,
                                     chunk_size=(1,) + tuple(output_options['chunk_size']),
                                     compression_enabled=output_options['compression_enabled'],
                                     compression_type='gzip',
                                     compression_level=output_options['compression_level'],
                                     shuffle_filter=output_options['shuffle'])

    t_all_elapsed = time.perf_counter() - t_all
    info_channel.log(f"successfully ran geocode SLC in {t_all_elapsed:.3f} seconds")


if __name__ == "__main__":
    yaml_parser = YamlArgparse()
    args = yaml_parser.parse()
    gslc_runconfig = GSLCRunConfig(args)

    sas_output_file = gslc_runconfig.cfg[
        'product_path_group']['sas_output_file']

    if os.path.isfile(sas_output_file):
        os.remove(sas_output_file)

    run(gslc_runconfig.cfg)

    with GslcWriter(runconfig=gslc_runconfig) as gslc_obj:
        gslc_obj.populate_metadata()

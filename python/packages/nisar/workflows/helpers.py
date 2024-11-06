'''
collection of useful functions used across workflows
'''

import datetime
import os
import pathlib
from collections import defaultdict
from dataclasses import dataclass, field
import json
import h5py
import isce3
import journal
import numpy as np
from isce3.core.resample_block_generators import get_blocks
from isce3.io.gdal.gdal_raster import GDALRaster
from isce3.product import RadarGridParameters
from nisar.products.readers import SLC
from nisar.workflows.get_product_geometry import \
    get_geolocation_grid as compute_geogrid_geometry
from nisar.workflows.h5_prep import get_off_params
from osgeo import gdal, gdal_array
from pathlib import Path


class JsonNumpyEncoder(json.JSONEncoder):
    """
    A thin wrapper around JSONEncoder w/ augmented default method to support
    various numpy array and complex data types
    """

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)

        elif isinstance(obj, np.floating):
            return float(obj)

        elif isinstance(obj, (complex, np.complexfloating)):
            return {'real': obj.real, 'imag': obj.imag}

        elif isinstance(obj, np.ndarray):
            return obj.tolist()

        elif isinstance(obj, np.bool_):
            return bool(obj)

        return super().default(obj)


def build_uniform_quantizer_lut_l0b(
        nbits: int, bad_val: float = 0.0, twos_complement: bool = True
) -> np.ndarray:
    """Build uniform quantizer LUT used in place of BFPQLUT in L0B product"""
    # get size of BFPQLUT to be power of 2!
    nbits_pow2 = 2 ** int(np.ceil(np.log2(nbits)))
    size_lut = 2 ** nbits_pow2
    len_decoder_lut = 2**nbits
    len_decoder_lut_h = len_decoder_lut // 2
    bfpq_uq = np.full(size_lut, bad_val, dtype='f4')
    # 2s complement sign representation of unsigned integer
    bfpq_uq[:len_decoder_lut_h] = np.arange(0.5, len_decoder_lut_h, dtype='f4')
    if twos_complement:
        bfpq_uq[:-len_decoder_lut_h - 1:-1] = -bfpq_uq[:len_decoder_lut_h]
    else:  # signed magnitude representation
        size_lut_h = size_lut // 2
        bfpq_uq[size_lut_h: size_lut_h + len_decoder_lut_h] = (
            -bfpq_uq[:len_decoder_lut_h])
    return bfpq_uq


def slice_gen(n_smp: int, n_smp_blk: int) -> slice:
    """slice generator.

    Parameters
    ----------
    n_smp : int
        Total number of samples
    n_smp_blk : int
        Number of samples per full block

    Yields
    ------
    slice
        slice object for each block.
        The last block can be partial and have less
        number of samples than `n_smp_blk`!

    """
    n_blk = int(np.ceil(n_smp / n_smp_blk))
    for n in range(n_blk):
        i_start = n * n_smp_blk
        i_stop = min(n_smp, i_start + n_smp_blk)
        yield slice(i_start, i_stop)


def deep_update(original, update, flag_none_is_valid=True):
    '''
    update default runconfig key with user supplied dict
    https://stackoverflow.com/questions/3232943/update-value-of-a-nested-dictionary-of-varying-depth

    If `flag_none_is_valid` is `True`, then an empty field in a user-supplied
    runconfig will be treated the same as if that field was omitted entirely.
    Otherwise, if the field is blank (i.e., `None`) it would override the default value
    with None.
    '''
    for key, val in update.items():
        if isinstance(val, dict) and original.get(key) is not None:
            # Only call deep_update() if `original[key] is not empty
            original[key] = deep_update(original.get(key, {}), val,
                flag_none_is_valid)
        elif (flag_none_is_valid or val is not None):
            # Update `original[key]` with val if
            # 1. The flag `flag_none_is_valid` is enabled:
            #    In this case, `None` is considered a valid value
            #    and therefore we don't need to check if `val` is None
            # 2. Update `original` if `val` is not `None``
            original[key] = val

    # return updated original
    return original


def autovivified_dict():
    '''
    Use autovivification to create nested dictionaries.
    https://en.wikipedia.org/wiki/Autovivification
    defaultdict creates any items you try to access if they don't exist yet.
    defaultdict only performs this for a single level.
    https://stackoverflow.com/a/5900634
    The recursion extends this behavior and allows the creation of additional levels.
    https://stackoverflow.com/a/22455426
    '''
    return defaultdict(autovivified_dict)


WORKFLOW_SCRIPTS_DIR = os.path.dirname(os.path.realpath(__file__))


def check_write_dir(dst_path: str):
    '''
    Raise error if given path does not exist or not writeable.
    '''
    if not dst_path:
        dst_path = '.'

    error_channel = journal.error('helpers.check_write_dir')

    # check if scratch path exists
    dst_path_ok = os.path.isdir(dst_path)

    if not dst_path_ok:
        try:
            os.makedirs(dst_path, exist_ok=True)
        except OSError:
            err_str = f"Unable to create {dst_path}"
            error_channel.log(err_str)
            raise OSError(err_str)

    # check if path writeable
    write_ok = os.access(dst_path, os.W_OK)
    if not write_ok:
        err_str = f"{dst_path} scratch directory lacks write permission."
        error_channel.log(err_str)
        raise PermissionError(err_str)


def check_dem(dem_path: str):
    '''
    Raise error if DEM is not system file, netCDF, nor S3.
    '''
    error_channel = journal.error('helpers.check_dem')

    try:
        gdal.Open(dem_path)
    except:
        err_str = f'{dem_path} cannot be opened by GDAL'
        error_channel.log(err_str)
        raise ValueError(err_str)


def check_radargrid_orbit_tec(radar_grid, orbit, tec_path):
    '''
    Check if the input orbit and TEC files' temporal coverage is enough.
    Raise RuntimeError when the coverage is not enough.

    Parameters
    ----------
    radar_grid: isce3.product.RadarGridParameters
        Radar grid of the input RSLC
    orbit: isce3.core.Orbit
        Orbit data provided
    tec: str
        path to the IMAGEN TEC data

    Raises
    ------
    ApplicationError: Raised by `journal.error` instance When
        the temporal coverage of orbit and / or TEC file is not sufficient.
    '''

    error_channel = journal.error('helpers.check_radargrid_orbit_tec')
    info_channel = journal.info('helpers.check_radargrid_orbit_tec')

    radargrid_ref_epoch = datetime.datetime.fromisoformat(radar_grid.ref_epoch.isoformat_usec())
    sensing_start = radargrid_ref_epoch + datetime.timedelta(seconds=radar_grid.sensing_start)
    sensing_stop = radargrid_ref_epoch + datetime.timedelta(seconds=radar_grid.sensing_stop)

    orbit_start = datetime.datetime.fromisoformat(orbit.start_datetime.isoformat_usec())
    orbit_end = datetime.datetime.fromisoformat(orbit.end_datetime.isoformat_usec())

    # Compute the paddings of orbit and TEC w.r.t. radar grid
    orbit_margin_start = (sensing_start - orbit_start).total_seconds()
    orbit_margin_end = (orbit_end - sensing_stop).total_seconds()

    margin_info_msg = (f'Orbit margin before radar sensing start : {orbit_margin_start} seconds\n'
                       f'Orbit margin after radar sensing stop   : {orbit_margin_end} seconds\n')

    if not tec_path:
        info_channel.log('IMAGEN TEC was not provided. '
                         'Checking the orbit data and sensing start / stop.')

        info_channel.log(margin_info_msg)

        if orbit_margin_start < 0.0:
            error_channel.log('Not enough input orbit data at the radar sensing start.')
        if orbit_margin_end < 0.0:
            error_channel.log('Not enough input orbit data at the radar sensing end.')

    else:
        # Load timing information from IMAGEN TEC and check with orbit and sensing
        with open(tec_path, 'r') as jin:
            imagen_dict = json.load(jin)
            num_utc = len(imagen_dict['utc'])
            tec_start = datetime.datetime.fromisoformat(imagen_dict['utc'][0])
            tec_end = datetime.datetime.fromisoformat(imagen_dict['utc'][-1])

        tec_margin_start = (sensing_start - tec_start).total_seconds()
        tec_margin_end = (tec_end - sensing_stop).total_seconds()

        # Compute the half the TEC spacing, which is required when computing
        # azimuth TEC gradient. Note the timing grid for TEC gradient is
        # shifted by half of the TEC spacing
        minimum_margin_sec = (tec_end - tec_start).total_seconds() / (num_utc -1) / 2

        margin_info_msg += (f'IMAGEN TEC margin before radar sensing start : {tec_margin_start} seconds\n'
                            f'IMAGEN TEC margin after radar sensing stop   : {tec_margin_end} seconds\n'
                            f'Minimum required margin                : {minimum_margin_sec} seconds\n')

        info_channel.log(margin_info_msg)

        # Check if the margin looks okay when TEC is provided

        if orbit_margin_start < minimum_margin_sec:
            error_channel.log('Input orbit\'s margin before radar sensing start is not enough '
                            f'({orbit_margin_start} < {minimum_margin_sec})')

        if orbit_margin_end < minimum_margin_sec:
            error_channel.log('Input orbit\'s margin after radar sensing stop is not enough '
                            f'({orbit_margin_end} < {minimum_margin_sec})')

        if tec_margin_start < minimum_margin_sec:
            error_channel.log('IMAGEN TEC margin before radar sensing start is not enough '
                            f'({tec_margin_start} < {minimum_margin_sec})')

        if tec_margin_end < minimum_margin_sec:
            error_channel.log(f'IMAGEN TEC margin after radar sensing stop is not enough '
                            f'({tec_margin_end} < {minimum_margin_sec})')


def check_log_dir_writable(log_file_path: str):
    '''
    Check to see if destination directory of log file path is writable.
    Raise error if directory lacks write permission.
    '''
    error_channel = journal.error('helpers.check_log_dir_writeable')

    dest_dir, _ = os.path.split(log_file_path)

    # get current working directory if no directory in run_config_path
    if not dest_dir:
        dest_dir = os.getcwd()

    if not os.access(dest_dir, os.W_OK):
        err_str = f"No write permission to {dest_dir}"
        error_channel.log(err_str)
        raise PermissionError(err_str)


def check_mode_directory_tree(parent_dir: str, mode: str, frequency_list: list, pols: dict = {}):
    '''
    Checks existence parent directory and sub-directories.
    Sub-directories made from mode sub_dir + frequency_list.
    Expected directory tree:
    outdir/
    └── mode/
        └── freq(A,B)
            └── (HH, HV, VH, VV)
    '''
    error_channel = journal.error('helpers.check_directory_tree')

    parent_dir = pathlib.Path(parent_dir)

    # check if parent is a directory
    if not parent_dir.is_dir():
        err_str = f"{str(parent_dir)} not a valid path"
        error_channel.log(err_str)
        raise NotADirectoryError(err_str)

    # check if mode-directory exists
    mode_dir = parent_dir / f'{mode}'
    if not mode_dir.is_dir():
        err_str = f"{str(mode_dir)} not a valid path"
        error_channel.log(err_str)
        raise NotADirectoryError(err_str)

    # check number frequencies
    n_frequencies = len(frequency_list)
    if n_frequencies not in [1, 2]:
        err_str = f"{n_frequencies} is an invalid number of frequencies. Only 1 or 2 frequencies allowed"
        error_channel.log(err_str)
        raise ValueError(err_str)

    for freq in frequency_list:
        # check if frequency allowed
        if freq not in ['A', 'B']:
            err_str = f"frequency {freq} not valid. Only [A, B] allowed."
            error_channel.log(err_str)
            raise ValueError(err_str)

        # check if mode-directory exists
        freq_dir = mode_dir / f'freq{freq}'
        if not freq_dir.is_dir():
            err_str = f"{str(freq_dir)} not a valid path"
            error_channel.log(err_str)
            raise NotADirectoryError(err_str)

        # if no polarizations given continue to check next frequency
        if not pols:
            continue

        # check if frequency in polarization dict
        if freq not in pols:
            err_str = f"No key in polarization dict for frequency: {freq}"
            error_channel.log(err_str)
            raise KeyError(err_str)

        # check if polarization directory exists
        for pol in pols[freq]:
            pol_dir = freq_dir / pol
            if not pol_dir.is_dir():
                err_str = f"{str(pol_dir)} not a valid path"
                error_channel.log(err_str)
                raise NotADirectoryError(err_str)


def check_hdf5_freq_pols(h5_path: str, freq_pols: dict):
    '''
    Check if frequency (keys) and polarizations (items) exist in HDF5
    Expected HDF5 structure:
    swath or grid group/
    └── freq(A,B) group
        └── (HH, HV, VH, VV) dataset
    '''
    error_channel = journal.error('helpers.check_hdf5_freq_pols')

    # attempt to open HDF5
    try:
        h5_obj = h5py.File(h5_path, 'r', libver='latest', swmr=True)
    except:
        err_str = f"h5py unable to open {h5_path}"
        error_channel.log(err_str)
        raise ValueError(err_str)

    # use with to ensure h5_obj closes
    with h5_obj:
        product_type = h5_obj['/science/LSAR/identification/productType'][()].decode('UTF-8')
        if product_type.startswith('G'):
            grid_type = 'grids'
        else:
            grid_type = 'swaths'
        grid_path = f'/science/LSAR/{product_type}/{grid_type}'

        # get swath/grid group from hdf5
        grid_group = h5_obj[grid_path]

        # check if frequencies in group
        for freq in freq_pols:
            freq_str = f"frequency{freq}"
            if freq_str not in grid_group:
                err_str = f"{freq} not found in swath/grid group of {h5_path}"
                error_channel.log(err_str)
                raise ValueError(err_str)

            # get frequency group from swath/grid group
            freq_group = grid_group[freq_str]
            if 'interferogram' in freq_group:
                freq_group = freq_group['interferogram']

            # check if polarizations in group
            for pol in freq_pols[freq]:
                if pol not in freq_group:
                    err_str = f"{pol} not found in {freq} group of swath/grid group of {h5_path}"
                    error_channel.log(err_str)
                    raise ValueError(err_str)


def copy_raster(infile, freq, pol,
                lines_per_block, outfile, file_type="ENVI"):
    '''
    Copy RSLC dataset to GDAL format and convert real and
    imaginary parts from float16 to float32

    Parameters
    ----------
    infile: str
        Path to RSLC HDF5
    freq: str
        RSLC frequency band to process ('A' or 'B')
    pol: str
        RSLC polarization to process
    outfile: str
        Output filename
    file_type: str
        GDAL-friendly file format
    '''

    # Open RSLC HDF5 file dataset and check if complex32
    rslc = SLC(hdf5file=infile)
    is_complex32 = rslc.is_dataset_complex32(freq, pol)
    hdf5_ds = rslc.getSlcDataset(freq, pol)

    # Get RSLC dimension through GDAL
    gdal_ds = gdal.Open(f'HDF5:{infile}:/{rslc.slcPath(freq, pol)}')
    rslc_length, rslc_width = gdal_ds.RasterYSize, gdal_ds.RasterXSize

    # Create output file
    driver = gdal.GetDriverByName(file_type)
    out_ds = driver.Create(outfile, rslc_width, rslc_length,
                           1, gdal.GDT_CFloat32)

    # Start block processing
    lines_per_block = min(rslc_length, lines_per_block)
    num_blocks = int(np.ceil(rslc_length / lines_per_block))

    # Iterate over blocks to convert and write
    for block in range(num_blocks):
        line_start = block * lines_per_block

        # Check for last block and compute block length accordingly
        if block == num_blocks - 1:
            block_length = rslc_length - line_start
        else:
            block_length = lines_per_block

        # Read a block of data from RSLC and convert real and imag part to float32
        s = np.s_[line_start:line_start + block_length, :]
        if is_complex32:
            data_block = isce3.core.types.read_c4_dataset_as_c8(hdf5_ds, s)
        else:
            data_block = hdf5_ds[s]

        # Write to GDAL raster
        out_ds.GetRasterBand(1).WriteArray(data_block[0:block_length],
                                           yoff=line_start, xoff=0)
    out_ds.FlushCache()


def complex_raster_path_from_h5(slc, freq, pol, hdf5_path, lines_per_block,
                                c32_output_path):
    '''
    Get path for io.raster based on raster datatype. If datatype is not
    complex64,convert and save to temporary file. Raster object generated here
    to avoid potential artifacts caused by copying for Raster objects.

    Parameters
    ----------
    slc: nisar.products.readers.SLC
        RSLC object
    freq: str
        RSLC frequency band to process ('A' or 'B')
    pol: str
        RSLC polarization to process
    hdf5_path: str
        Source HDF5 file
    lines_per_block: int
        Lines per block to be converted and written to complex32 (if needed)
    c32_output_path: str
        GDAL-friendly file format

    Returns
    -------
    raster_path: str
        isce3.io.Raster-friendly path to raster dataset
    file_path: str
        File containing raster dataset. Differs from raster_path if when output
        is HDF5
    '''
    if slc.is_dataset_complex32(freq, pol):
        # If SLC dataset is complex32 HDF5, convert to complex64, write to
        # ENVI raster, and return path ENVI raster
        copy_raster(hdf5_path, freq, pol, lines_per_block,
                    c32_output_path, file_type='ENVI')
        raster_path = c32_output_path
        file_path = c32_output_path
    else:
        # If SLC dataset is complex64 HDF5, return GDAL path to HDF5 dataset
        slc_h5_path = f'/{slc.SwathPath}/frequency{freq}/{pol}'
        raster_path = f'HDF5:{hdf5_path}:{slc_h5_path}'
        file_path = hdf5_path

    return raster_path, file_path

def get_pixel_offsets_params(cfg : dict):
    """
    Get the pixel offsets parameters from the runconfig dictionary

    Parameters
    ----------
    cfg : dict
        InSAR runconfig dictionray

    Returns
    ----------
    is_roff : boolean
        Offset product or not
    margin : int
        Margin
    rg_start : int
        Start range
    az_start : int
        Start azimuth
    rg_skip : int
        Pixels skiped across range
    az_skip : int
        Pixels skiped across the azimth
    rg_search : int
        Window size across range
    az_search : int
        Window size across azimuth
    rg_chip : int
        Fine window size across range
    az_chip : int
        Fine window size across azimuth
    ovs_factor : int
        Oversampling factor
    """
    proc_cfg = cfg["processing"]

    # pull the offset parameters
    is_roff = proc_cfg["offsets_product"]["enabled"]
    (margin, rg_gross, az_gross,
        rg_start, az_start,
        rg_skip, az_skip, ovs_factor) = \
            [get_off_params(proc_cfg, param, is_roff)
            for param in ["margin", "gross_offset_range",
                        "gross_offset_azimuth",
                        "start_pixel_range","start_pixel_azimuth",
                        "skip_range", "skip_azimuth",
                        "correlation_surface_oversampling_factor"]]

    rg_search, az_search, rg_chip, az_chip = \
        [get_off_params(proc_cfg, param, is_roff,
                        pattern="layer",
                        get_min=True,) for param in \
                            ["half_search_range",
                                "half_search_azimuth",
                                "window_range",
                                "window_azimuth"]]
    # Adjust margin
    margin = max(margin, np.abs(rg_gross), np.abs(az_gross))

    # Compute slant range/azimuth vectors of offset grids
    if rg_start is None:
        rg_start = margin + rg_search
    if az_start is None:
        az_start = margin + az_search

    return (is_roff,  margin, rg_start, az_start,
            rg_skip, az_skip, rg_search, az_search,
            rg_chip, az_chip, ovs_factor)

def get_pixel_offsets_dataset_shape(cfg : dict, freq : str):
    """
    Get the pixel offsets dataset shape at a given frequency

    Parameters
    ---------
    cfg : dict
        InSAR runconfig dictionary
    freq: str
        frequency ('A' or 'B')

    Returns
    ----------
    tuple
        (off_length, off_width):
    """
    proc_cfg = cfg["processing"]
    is_roff,  margin, _, _,\
    rg_skip, az_skip, rg_search, az_search,\
    rg_chip, az_chip, _ = get_pixel_offsets_params(cfg)

    ref_h5_slc_file = cfg["input_file_group"]["reference_rslc_file"]
    ref_rslc = SLC(hdf5file=ref_h5_slc_file)

    radar_grid = ref_rslc.getRadarGrid(freq)
    slc_lines, slc_cols = (radar_grid.length, radar_grid.width)

    off_length = get_off_params(proc_cfg, "offset_length", is_roff)
    off_width = get_off_params(proc_cfg, "offset_width", is_roff)
    if off_length is None:
        margin_az = 2 * margin + 2 * az_search + az_chip
        off_length = (slc_lines - margin_az) // az_skip
    if off_width is None:
        margin_rg = 2 * margin + 2 * rg_search + rg_chip
        off_width = (slc_cols - margin_rg) // rg_skip

    # shape of offset product
    return (off_length, off_width)

def get_offset_radar_grid(cfg, radar_grid_slc):
    ''' Create radar grid object for offset datasets

    Parameters
    ----------
    cfg : dict
        Dictionary containing processing parameters
    radar_grid_slc : SLC
        Object containing SLC properties
    '''
    proc_cfg = cfg["processing"]
    is_roff,  margin, rg_start, az_start,\
    rg_skip, az_skip, rg_search, az_search,\
    rg_chip, az_chip, _ = get_pixel_offsets_params(cfg)

    slc_lines, slc_cols = (radar_grid_slc.length, radar_grid_slc.width)

    off_length = get_off_params(proc_cfg, "offset_length", is_roff)
    off_width = get_off_params(proc_cfg, "offset_width", is_roff)

    if off_length is None:
        margin_az = 2 * margin + 2 * az_search + az_chip
        off_length = (slc_lines - margin_az) // az_skip
    if off_width is None:
        margin_rg = 2 * margin + 2 * rg_search + rg_chip
        off_width = (slc_cols - margin_rg) // rg_skip

    # the starting range/sensing start of the pixel offsets radar grid
    # to be at the center of the matching window
    offset_starting_range = radar_grid_slc.starting_range + \
                            (rg_start + rg_chip//2)\
                            * radar_grid_slc.range_pixel_spacing
    offset_sensing_start = radar_grid_slc.sensing_start + \
                           (az_start + az_chip//2)\
                           / radar_grid_slc.prf
    # Range spacing for offsets
    offset_range_spacing = radar_grid_slc.range_pixel_spacing * rg_skip
    offset_prf = radar_grid_slc.prf / az_skip

    # Create offset radar grid
    radar_grid = isce3.product.RadarGridParameters(offset_sensing_start,
                                                   radar_grid_slc.wavelength,
                                                   offset_prf,
                                                   offset_starting_range,
                                                   offset_range_spacing,
                                                   radar_grid_slc.lookside,
                                                   off_length,
                                                   off_width,
                                                   radar_grid_slc.ref_epoch)
    return radar_grid


def get_cfg_freq_pols(cfg):
    '''
    Generator of frequencies and polarizations for offset processing. Special
    attention given if single co-pol required.

    Parameters
    ----------
    cfg: dict
        RunConfig containing frequencies, polarizations, and fine resample
        settings

    Yields
    ------
    freq: ['A', 'B']
        Frequency for current
    pol_list: list
        List of polarizations associated with current frequency as dictated the
        the runconfig
    pol: list
        List of polarizations associated with current frequency. Maybe single
        co-pol if single co-pol offset processing flag is True.
    '''
    # Extract frequencies and polarizations to process
    freq_pols = cfg['processing']['input_subset']['list_of_frequencies']

    # Loop over items in freq_pols dict
    for freq, pol_list in freq_pols.items():
        # Yield for single co-pol for offset
        if cfg['processing']['process_single_co_pol_offset']:
            # init empty list to be populated only with co-pol channels
            pol =[]
            # For quad-pol data, priority to HH.
            if 'HH' in pol_list:
                pol = ['HH']
            elif 'VV' in pol_list:
                pol = ['VV']
            yield freq, pol_list, pol
        # Yield whatever is pol_list
        else:
            yield freq, pol_list, pol_list


def get_ground_track_velocity_product(ref_rslc : SLC,
                                      slant_range : np.ndarray,
                                      zero_doppler_time : np.ndarray,
                                      dem_file : str,
                                      output_dir: str):
    """
    Generate the ground track velocity product in a radar grid
    that has the same wavelength, look side, and reference
    epoch as the frequency A radar grid of the reference RSLC
    but with different slant range and zero doppler time.

    Parameters
    ----------
    ref_rslc : SLC object
        The SLC object of the reference RSLC
    slant_range: np.ndarray
        Slant range of the pixel offsets product
    zero_doppler_time: np.ndarray
        Zero doppler time of the pixel offsets product
    dem_file : str
        The DEM file
    output_dir : str
        The output directory

    Returns
    ----------
    ground_track_velocity_file : str
        ground track velocity output file
    """
    # NOTE: the prod_geometry_args dataclass is defined here
    # to avoid the usage of the parser command line
    @dataclass
    class GroundtrackVelocityGenerationParams:
        """
        Parameters to generate the ground track velocity.
        Definition of each parameter can be found in the
        get_product_geometry.py
        """
        threshold_rdr2geo = None
        num_iter_rdr2geo = None
        extra_iter_rdr2geo = None
        threshold_geo2rdr = None
        num_iter_geo2rdr = None
        delta_range_geo2rdr = None
        threshold_geo2rdr = 1e-8
        num_iter_geo2rdr = 50
        delta_range_geo2rdr = 10.0
        dem_interp_method = None
        output_dir = None
        dem_file = None
        epsg = None
        # Only the ground track velocity will be generated
        flag_interpolated_dem = False
        flag_coordinate_x = False
        flag_coordinate_y = False
        flag_incidence_angle = False
        flag_los = False
        flag_along_track = False
        flag_elevation_angle = False
        flag_ground_track_velocity = True

    args = GroundtrackVelocityGenerationParams()
    args.dem_file = dem_file
    args.output_dir = output_dir

    # Create the radar grid of pixel offsets product
    radar_grid = ref_rslc.getRadarGrid()
    zero_doppler_starting_time = zero_doppler_time[0]
    prf = 1.0 / (zero_doppler_time[1] - zero_doppler_time[0])
    starting_range = slant_range[0]
    range_spacing = slant_range[1] - slant_range[0]

    pixel_offsets_radar_grid = \
        RadarGridParameters(zero_doppler_starting_time,
                            radar_grid.wavelength,
                            prf,
                            starting_range,
                            range_spacing,
                            radar_grid.lookside,
                            len(zero_doppler_time),
                            len(slant_range),
                            radar_grid.ref_epoch)

    ground_track_velocity_file = f'{args.output_dir}/groundTrackVelocity.tif'
    compute_geogrid_geometry(ref_rslc, args,
                             pixel_offsets_radar_grid)

    return ground_track_velocity_file


def validate_fs_page_size(fs_page_size, chunks, itemsize=8):
    """
    Issue a warning if it seems like the page size is poorly chosen.

    Parameters
    ----------
    fs_page_size : int
        HDF5 file space page size in bytes
    chunks : tuple[int, ...]
        Chunk dimensions
    itemsize : int, optional
        Number of bytes per pixel.  Defaults to 8, which corresponds to the
        widest type available in a NISAR product.
    """
    warn = journal.warning('helpers.validate_fs_page_size').log
    # Limits gleaned from HDF5 sources (only min is documented).
    min_size = 512
    max_size = 1024**3
    if not (min_size <= fs_page_size <= max_size):
        warn(f"File space page size not in interval [{min_size}, {max_size}]")

    # HDF5 docs say powers of two work best for FAPL page size.  Assume same
    # holds true for FCPL page size.
    if (fs_page_size <= 0) or (fs_page_size & (fs_page_size - 1) != 0):
        warn("File space page size is not a positive power of two.")
    # If we're to retrieve a chunk of data and its metadata in a single read
    # (e.g., AWS s3 request) the page size must be bigger than the chunk.  Not
    # sure how much storage is required for HDF5 metadata.
    if not (fs_page_size > np.prod(chunks) * itemsize):
        warn("File space page size is not larger than a chunk of data.")


def _as_np_bytes_if_needed(val):
    '''
    If type str encountered, convert and return as np.string_. Otherwise return
    as is.
    '''
    val = np.bytes_(val) if isinstance(val, str) else val
    return val


@dataclass
class HDF5DatasetParams:
    '''
    Convenience dataclass for passing parameters to be written to h5py.Dataset
    '''
    # Dataset name
    name: str
    # Data to be stored in Dataset
    value: object
    # Description attribute of Dataset
    description: str
    # Other attributes to be written to Dataset
    attr_dict: dict = field(default_factory=dict)


def add_dataset_and_attrs(group, meta_item):
    '''Write dataset parameters stored in HDF5DatasetParams object to h5py group.

    Parameters
    ----------
    group: h5py.Group
        h5py group where dataset and associated parameters are to be written
    meta_item: HDF5DatasetParams
        HDF5DatasetParams dataclass object containing dataset parameters
    '''
    # Ensure it is clear to write by deleting pre-existing Dataset
    if meta_item.name in group:
        del group[meta_item.name]

    # Convert to be written dataset value, if necessary
    val = _as_np_bytes_if_needed(meta_item.value)
    try:
        if val is None:
            # Assume NaN is valid dataset value if None is provided
            group[meta_item.name] = np.nan
        else:
            group[meta_item.name] = val
    except TypeError as exc:
        raise TypeError(f'unable to write {meta_item.name}') from exc

    # Write data and attributes
    val_ds = group[meta_item.name]
    desc = _as_np_bytes_if_needed(meta_item.description)
    val_ds.attrs['description'] = desc
    for key, val in meta_item.attr_dict.items():
        val_ds.attrs[key] = _as_np_bytes_if_needed(val)


def sum_gdal_rasters(filepath1, filepath2, out_filepath, data_type=np.float64,
                     driver_name='ENVI', row_blocks=2048, col_blocks=2048,
                     invalid_value=np.nan):
    """
    Sum 2 GDAL memory-mappable rasters block by block and save the result
    to an output file in a GDAL-friendly format

    Parameters
    ----------
    filepath1: str | os.PathLike
        Path to the first GDAL memory-mappable raster
    filepath2: str | os.PathLike
        Path to the second GDAL memory-mappable raster
    out_filepath: str | os.PathLike
        Path to the output file
    data_type: np.dtype, optional
        Data type of the input and output files (default: np.float64)
    driver_name: str, optional
        Name of the GDAL driver for the output file (default: ENVI)
    row_blocks: int, optional
        Number of rows in each block (default: 2048)
    col_blocks: int, optional
        Number of columns in each block (default: 2048)
    invalid_value: float | numpy.nan, optional
        The invalid value of the rasters. When this value is encountered
        in either raster, it will be regarded as invalid and masked over the
        output data. (default: np.nan)
    """
    error_channel = journal.error('helpers.sum_gdal_rasters').log

    if not isinstance(row_blocks, int) or not isinstance(col_blocks, int):
        error_channel('row_block and col_blocks must be integers')

    # Open the GDAL rasters and get shape
    in_ds1 = gdal.Open(os.fspath(filepath1))
    in_ds2 = gdal.Open(os.fspath(filepath2))
    length1, width1 = in_ds1.RasterYSize, in_ds1.RasterXSize
    length2, width2 = in_ds2.RasterYSize, in_ds2.RasterXSize

    # Check that rasters have the same shape
    if (length1 != length2) or (width1 != width2):
       error_channel(f'Input raster files do not have the same number of '
                     f'rows ({length1}, {length2}) or columns ({width1},{width2}')

    # Check we only have one raster band for both input rasters
    if (in_ds1.RasterCount != 1) or (in_ds2.RasterCount != 1):
        error_channel("Input raster files must be single-band rasters")

    # Check that the data-type specified by the user matches the data type
    # of the GDAL input raster files
    gdal_dtype = gdal_array.NumericTypeCodeToGDALTypeCode(data_type)

    for ds in [in_ds1, in_ds2]:
        raster_dtype = ds.GetRasterBand(1).DataType
        if raster_dtype != gdal_dtype:
            error_channel(f"input file has unexpected GDAL datatype {raster_dtype};"
                          f" expected {gdal_dtype}")

    # Initialize reader objects for filepath1 and filepath2
    file1_reader = np.memmap(
            filename=filepath1,
            shape=(length1, width1),
            dtype=data_type,
            mode='r',
        )

    file2_reader = np.memmap(
        filename=filepath2,
        shape=(length1, width1),
        dtype=data_type,
        mode='r',
    )

    # Initialize output writer
    out_file_writer = GDALRaster.create_dataset_file(
        filepath=Path(out_filepath),
        dtype=data_type,
        shape=(length1, width2),
        num_bands=1,
        driver_name=driver_name,
    )

    # Process blocks
    for out_block_slice in get_blocks(
            block_max_shape=(row_blocks, col_blocks),
            grid_shape=(length1, width1),
            quiet=True,
    ):
        file1_block = file1_reader[out_block_slice].astype(data_type, copy=False)
        file2_block = file2_reader[out_block_slice].astype(data_type, copy=False)

        # Use numpy masked arrays to handle sums where one of the element is a fill_value
        masked_file1_block = np.ma.masked_equal(file1_block, invalid_value)
        masked_file2_block = np.ma.masked_equal(file2_block, invalid_value)

        # Perform the sum, with the rule that if either element is masked (invalid), the result will be the fill_value
        sum_result_masked = np.ma.where(masked_file1_block.mask | masked_file2_block.mask,
                                        invalid_value, masked_file1_block + masked_file2_block)

        # Convert the masked array to a normal array, filling masked elements with the fill value
        sum_result = sum_result_masked.filled(invalid_value)

        # Write the result to the output file
        out_file_writer[out_block_slice] = sum_result

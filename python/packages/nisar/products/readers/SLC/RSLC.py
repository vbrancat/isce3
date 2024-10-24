# -*- coding: utf-8 -*-
from __future__ import annotations

import h5py
import journal
import logging
import pyre
import re

from nisar.noise import NoiseEquivalentBackscatterProduct
from isce3.core import DateTime
from isce3.core.types import ComplexFloat16Decoder, is_complex32

from .SLCBase import SLCBase

PRODUCT = 'RSLC'

log = logging.getLogger("nisar.products.readers.RSLC")


class RSLC(SLCBase, family='nisar.productreader.rslc'):
    """
    Class for parsing NISAR RSLC products into ISCE3 structures.
    """

    productValidationType = pyre.properties.str(default=PRODUCT)
    productValidationType.doc = 'Validation tag to ensure correct product type'

    _ProductType = pyre.properties.str(default=PRODUCT)
    _ProductType.doc = 'The type of the product.'

    @property
    def ProductPath(self) -> str:
        # The product group name should be "RSLC" per the spec. However, early
        # sample products used "SLC" instead, and identification.productType is
        # not reliable, either. We maintain compatibility with both options.
        with h5py.File(self.filename, 'r', libver='latest', swmr=True) as f:
            g = f[self.RootPath]
            if "RSLC" in g:
                return f"{g.name}/RSLC"
            elif "SLC" in g:
                return f"{g.name}/SLC"
        raise RuntimeError("HDF5 file missing 'RSLC' or 'SLC' product group.")

    def imageDatasetPath(self, frequency: str, polarization: str) -> str:
        # implementation of GenericProduct.imageDatasetPath
        data_path = f"{self.SwathPath}/frequency{frequency}/{polarization}"
        return data_path

    def getSlcDatasetAsNativeComplex(
        self,
        frequency: str,
        polarization: str,
    ) -> h5py.Dataset | ComplexFloat16Decoder:
        """
        Get an SLC raster layer as a complex64 or complex128 dataset.

        Return the SLC dataset corresponding to a given frequency sub-band and
        polarization from the HDF5 file as a complex64 (i.e. pairs of 32-bit floats)
        or complex128 (i.e. pairs of 64-bit floats) dataset. If the data was stored as
        complex32 (i.e. pairs of 16-bit floats), it will be lazily converted to
        complex64 when accessed.

        Parameters
        ----------
        frequency : "A" or "B"
            The frequency letter, either "A" or "B".
        polarization: str
            The polarization term associated with the data array.
            One of "HH", "HV", "VH", "VV", "LH", "LV", "RH", "RV".

        Returns
        -------
        h5py.Dataset or isce3.core.types.ComplexFloat16Decoder
            The HDF5 dataset, possibly wrapped in a decoder layer that handles
            converting from half precision complex values to single precision.
        """
        dataset = self.getSlcDataset(frequency, polarization)

        if is_complex32(dataset):
            return ComplexFloat16Decoder(dataset)
        else:
            return dataset

    def getProductLevel(self):
        """
        Returns the product level
        """
        return "L1"

    def is_dataset_complex32(self, freq: str, pol: str) -> bool:
        """
        Determine if RSLC raster is of data type complex32

        Parameters
        ----------
        freq : "A" or "B"
            The frequency letter, either "A" or "B".
        pol: str
            The polarization term associated with the data array.
            One of "HH", "HV", "VH", "VV", "LH", "LV", "RH", "RV".
        """
        # Set error channel
        error_channel = journal.error('SLC.is_dataset_complex32')

        with h5py.File(self.filename, 'r', libver='latest', swmr=True) as h:
            freq_path = f'/{self.SwathPath}/frequency{freq}'
            if freq_path not in h:
                err_str = f'Frequency {freq} not found in SLC'
                error_channel.log(err_str)
                raise LookupError(err_str)

            slc_path = self.slcPath(freq, pol)
            if slc_path not in h:
                err_str = f'Polarization {pol} for frequency {freq} not found in SLC'
                error_channel.log(err_str)
                raise LookupError(err_str)

            return is_complex32(h[slc_path])


    def getNoiseEquivalentBackscatter(self, frequency=None, pol=None):
        '''
        Extract noise equivalent backscatter product for a particular
        frequency band and TxRx polarization.  It's conceptually the same as
        as noise equivalent sigma zero (NESZ) but agnostic with respect to the
        area normalization convention.

        Parameters
        ----------
        frequency : str, optional
            Frequency band such as 'A', 'B'.
            Default is the very first one in lexicographical order.
        pol : str, optional
            TxRx polarization such as 'HH', 'HV', etc.
            Default is the first co-pol in frequency if `frequency`
            otherwise the very first co-pol in very first frequency
            band. If no co-pol, the first cross-pol product will
            be picked.

        Returns
        -------
        nisar.noise.NoiseEquivalentBackscatterProduct

        '''
        # set frequency and pol
        if frequency is None:
            frequency = self._getFirstFrequency()
        if pol is None:
            pols = self.polarizations[frequency]
            co_pol = [p for p in pols if p[0] == p[1] or p[0] in ('L', 'R')]
            if len(co_pol) == 0:  # no co-pol
                pol = pols[0]
            else:  # there exists a co-pol
                pol = co_pol[0]

        # Save typing...
        cal, freq  = self.CalibrationInformationPath, f'frequency{frequency}'

        # Set paths relative to cal group. Support three product spec versions.
        # Keys correspond to the first tag of the NISAR PIX repo that implements
        # the given data layout (though v0.0.0 doesn't exist and is just
        # shorthand for the first ever version).
        layouts = {
            "v1.2.0": {
                "noise": _h5join(cal, freq, "noiseEquivalentBackscatter", pol),
                "time": _h5join(cal, freq, "noiseEquivalentBackscatter",
                    "zeroDopplerTime"),
                "range": _h5join(cal, freq, "noiseEquivalentBackscatter",
                    "slantRange"),
            },
            "v1.0.0": {
                "noise": _h5join(cal, freq, "nes0", pol),
                "time": _h5join(cal, freq, "nes0", "zeroDopplerTime"),
                "range": _h5join(cal, freq, "nes0", "slantRange"),
            },
            "v0.0.0": {
                "noise": _h5join(cal, freq, pol, "nes0"),
                "time": _h5join(cal, "zeroDopplerTime"),
                "range": _h5join(cal, "slantRange"),
            },
        }

        # parse all fields for NESZ
        with h5py.File(self.filename, 'r', libver='latest', swmr=True) as fid:
            for version, paths in layouts.items():
                if paths["noise"] in fid:
                    break
                log.warning("Couldn't find noise with RSLC schema "
                    f"corresponding to tag {version} of NISAR_PIX.")
            else:
                raise IOError("Could not find noise layer in RSLC file.")

            noise = fid[paths["noise"]][:]
            sr = fid[paths["range"]][:]
            azt_dset = fid[paths["time"]]
            azt = azt_dset[:]
            units = azt_dset.attrs['units'].decode()

        # datetime UTC pattern to look for in units to get epoch
        dt_pat = re.compile(
            '[0-9]{4}-[0-9]{2}-[0-9]{2}[T ][0-9]{2}:[0-9]{2}:[0-9]{2}(?:[.][0-9]{0,9})?'
        )
        matches = dt_pat.findall(units)
        if len(matches) != 1:
            raise RuntimeError(
                f"missing epoch in zeroDopplerTime units attribute: {units!r}"
            )
        utc_str = matches[0]
        epoch = DateTime(utc_str)
        # build and return noise product
        return NoiseEquivalentBackscatterProduct(noise, sr, azt, epoch,
            frequency, pol)


def _h5join(*paths: str) -> str:
    """Join two paths to be used in HDF5"""
    # avoid repeated path separators
    return "/".join(path.rstrip("/") for path in paths)

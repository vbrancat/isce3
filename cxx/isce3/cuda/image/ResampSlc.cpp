#include "ResampSlc.h"

#include <algorithm>
#include <iostream>
#include <chrono>
#include <cmath>

// isce3::core
#include <isce3/core/Constants.h>
#include <isce3/core/LUT2d.h>

#include <isce3/image/Tile.h>

#include <isce3/cuda/core/gpuInterpolator.h>

#include "gpuResampSlc.h"

using isce3::io::Raster;

// Alternative generic resamp entry point: use filenames to internally create rasters
void isce3::cuda::image::ResampSlc::
resamp(const std::string & inputFilename,          // filename of input SLC
       const std::string & outputFilename,         // filename of output resampled SLC
       const std::string & rgOffsetFilename,       // filename of range offsets
       const std::string & azOffsetFilename,       // filename of azimuth offsets
       int inputBand, bool flatten, int rowBuffer,
       int chipSize)
{
    // Make input rasters
    Raster inputSlc(inputFilename, GA_ReadOnly);
    Raster rgOffsetRaster(rgOffsetFilename, GA_ReadOnly);
    Raster azOffsetRaster(azOffsetFilename, GA_ReadOnly);

    // Make output raster; geometry defined by offset rasters
    const int outLength = rgOffsetRaster.length();
    const int outWidth = rgOffsetRaster.width();
    Raster outputSlc(outputFilename, outWidth, outLength, 1, GDT_CFloat32, "ISCE");

    // Call generic resamp
    resamp(inputSlc, outputSlc, rgOffsetRaster, azOffsetRaster, inputBand, flatten,
           rowBuffer, chipSize);
}

// Generic resamp entry point from externally created rasters
void isce3::cuda::image::ResampSlc::
resamp(isce3::io::Raster & inputSlc, isce3::io::Raster & outputSlc,
       isce3::io::Raster & rgOffsetRaster, isce3::io::Raster & azOffsetRaster,
       int inputBand, bool flatten, int rowBuffer,
       int chipSize)
{
    // Set the band number for input SLC
    _inputBand = inputBand;
    // Cache width of SLC image
    const int inLength = inputSlc.length();
    const int inWidth = inputSlc.width();
    // Cache output length and width from offset images
    const int outLength = rgOffsetRaster.length();
    const int outWidth = rgOffsetRaster.width();

    // Check if reference data is available
    if (flatten && !this->haveRefData()) {
        std::string error_msg{"Unable to flatten; reference data not provided."};
        throw isce3::except::InvalidArgument(ISCE_SRCINFO(), error_msg);
    }

    // initialize interpolator
    isce3::cuda::core::gpuSinc2dInterpolator<thrust::complex<float>> interp(chipSize-1, isce3::core::SINC_SUB);

    // Determine number of tiles needed to process image
    const int nTiles = _computeNumberOfTiles(outLength, _linesPerTile);
    std::cout <<
        "GPU resampling using " << nTiles << " tiles of " << _linesPerTile
        << " lines per tile\n";
    // Start timer
    auto timerStart = std::chrono::steady_clock::now();

    // For each full tile of _linesPerTile lines...
    for (int tileCount = 0; tileCount < nTiles; tileCount++) {

        // Make a tile for representing input SLC data
        Tile_t tile;
        tile.width(inWidth);
        // Set its line index bounds (line number in output image)
        tile.rowStart(tileCount * _linesPerTile);
        if (tileCount == (nTiles - 1)) {
            tile.rowEnd(outLength);
        } else {
            tile.rowEnd(tile.rowStart() + _linesPerTile);
        }

        // Initialize offsets tiles
        isce3::image::Tile<float> azOffTile, rgOffTile;
        _initializeOffsetTiles(tile, azOffsetRaster, rgOffsetRaster,
                               azOffTile, rgOffTile, outWidth);

        // Get corresponding image indices
        std::cout << "Reading in image data for tile " << tileCount << std::endl;
        _initializeTile(tile, inputSlc, azOffTile, outLength, rowBuffer, chipSize/2);

        // Perform interpolation
        std::cout << "Interpolating tile " << tileCount << std::endl;
        gpuTransformTile(tile, outputSlc, rgOffTile, azOffTile, _rgCarrier, _azCarrier,
                _dopplerLUT, interp, inWidth, inLength, this->startingRange(),
                this->rangePixelSpacing(), this->sensingStart(), this->prf(),
                this->wavelength(), this->refStartingRange(),
                this->refRangePixelSpacing(), this->refWavelength(), flatten,
                chipSize, _invalid_value);

    }

    // Print out timing information and reset
    auto timerEnd = std::chrono::steady_clock::now();
    const double elapsed = 1.0e-3 * std::chrono::duration_cast<std::chrono::milliseconds>(
        timerEnd - timerStart).count();
    std::cout << "Elapsed processing time: " << elapsed << " sec" << "\n";
}

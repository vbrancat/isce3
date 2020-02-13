// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// Source Author: Bryan Riel
// Copyright 2017-2018

#pragma once

// std
#include <string>
#include <algorithm>
#include <locale>
#include <map>

#include <isce/core/Constants.h>
#include <isce/core/LookSide.h>
#include <isce/io/IH5.h>
#include <isce/product/Metadata.h>
#include <isce/product/Swath.h>

// Declarations
namespace isce {
    namespace product {
        class Product;
    }
}

// Product class declaration
class isce::product::Product {

    public:
        /** Constructor from IH5File object. */
        Product(isce::io::IH5File &);

        /** Constructor with Metadata and Swath map. */
        inline Product(const Metadata &, const std::map<char, isce::product::Swath> &);

        /** Get a read-only reference to the metadata */
        inline const Metadata & metadata() const { return _metadata; }
        /** Get a reference to the metadata. */
        inline Metadata & metadata() { return _metadata; }

        /** Get a read-only reference to a swath */
        inline const Swath & swath(char freq) const { return _swaths.at(freq); }
        /** Get a reference to a swath */
        inline Swath & swath(char freq) { return _swaths[freq]; }
        /** Set a swath */
        inline void swath(const Swath & s, char freq) { _swaths[freq] = s; }

        /** Get the look direction */
        inline isce::core::LookSide lookSide() const { return _lookSide; }
        /** Set look direction using enum */
        inline void lookSide(isce::core::LookSide side) { _lookSide = side; }
        /** Set look direction from a string */
        inline void lookSide(const std::string &);

        /** Get the filename of the HDF5 file. */
        inline std::string filename() const { return _filename; }

    private:
        isce::product::Metadata _metadata;
        std::map<char, isce::product::Swath> _swaths;
        std::string _filename;
        isce::core::LookSide _lookSide;
};

/** @param[in] meta Metadata object
  * @param[in] swaths Map of Swath objects per frequency */
isce::product::Product::
Product(const Metadata & meta, const std::map<char, isce::product::Swath> & swaths) :
    _metadata(meta), _swaths(swaths) {}

/** @param[in] look String representation of look side */
void
isce::product::Product::
lookSide(const std::string & inputLook) {
    _lookSide = isce::core::parseLookSide(inputLook);
}

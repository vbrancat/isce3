#cython: language_level=3
#
# Author: Bryan V. Riel
# Copyright 2017-2019
#

from DateTime cimport DateTime
from RadarGridParameters cimport RadarGridParameters
#from pygeometry cimport _parseDirection

cdef class pyRadarGridParameters:
    """
    Cython wrapper for isce::product::RadarGridParameters.
    """
    cdef RadarGridParameters * c_radargrid
    cdef bool __owner

    def __cinit__(self,
                  pySwath swath=None,
                  side = None):
        cdef Direction _side
        if swath is not None:
            if side is None:
                raise AttributeError("Invalid look direction.")
            _side = _parseDirection(side)
            self.c_radargrid = new RadarGridParameters(
                deref(swath.c_swath), _side)
            self.__owner = True
        else:
            self.c_radargrid = new RadarGridParameters()
            self.__owner = True
        return

    def __dealloc__(self):
        if self.__owner:
            del self.c_radargrid

    @staticmethod
    cdef cbind(RadarGridParameters radarGrid):
        """
        Creates a new pyRadarGridParameters instance from a C++ RadarGridParameters instance.
        """
        new_grid = pyRadarGridParameters()
        del new_grid.c_radargrid
        new_grid.c_radargrid = new RadarGridParameters(radarGrid)
        new_grid.__owner = True
        return new_grid
               
    @property
    def lookSide(self):
        cdef int n = self.c_radargrid.lookSide()
        return n

    @property
    def refEpoch(self):
        cdef DateTime date = self.c_radargrid.refEpoch()
        return pyDateTime.cbind(date)

    @property
    def sensingStart(self):
        return self.c_radargrid.sensingStart()

    @property
    def wavelength(self):
        return self.c_radargrid.wavelength()

    @property
    def prf(self):
        return self.c_radargrid.prf()

    @property
    def startingRange(self):
        return self.c_radargrid.startingRange()

    @property
    def rangePixelSpacing(self):
        return self.c_radargrid.rangePixelSpacing()

    @property
    def length(self):
        return self.c_radargrid.length()

    @property
    def width(self):
        return self.c_radargrid.width()

    @property
    def size(self):
        return self.c_radargrid.size()

    @property
    def sensingStop(self):
        return self.c_radargrid.sensingStop()

    @property
    def sensingMid(self):
        return self.c_radargrid.sensingMid()

    def sensingTime(self, int line):
        return self.c_radargrid.sensingTime(line)

    def sensingDateTime(self, int line):
        cdef DateTime date = self.c_radargrid.sensingDateTime(line)
        return pyDateTime.cbind(date)

    @property
    def endingRange(self):
        return self.c_radargrid.endingRange()

    @property
    def midRange(self):
        return self.c_radargrid.midRange()

    def slantRange(self, int column):
        return self.c_radargrid.slantRange(column)

    def multilook(self, int az_looks, int rg_looks):
        cdef RadarGridParameters radar_grid = self.c_radargrid.multilook(az_looks, rg_looks)
        return pyRadarGridParameters.cbind(radar_grid)

# end of file 

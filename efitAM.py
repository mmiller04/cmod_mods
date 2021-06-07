### modified efitAR script using francesco's get geqdsk function in lyman_data

import numpy as np
from lyman_data import get_geqdsk_cmod
from scipy import interpolate
from omfit_classes.omfit_mds import OMFITmdsValue

class efit:

    def __init__(self, shotnumber, time, efit_id='EFIT01'):

        self.shotnumber = int(shotnumber)
        self.time = time*1e3
        self.get_geqdsk(efit_id)

    def get_geqdsk(self, efit_id):

        self.efit_id = efit_id

        geqdsk = get_geqdsk_cmod(self.shotnumber,self.time) # eqdsk file only gotten for 1 time

#        self.limiter = geqdsk['LIM']
        self.psi = geqdsk['PSIRZ']
        self.r = OMFITmdsValue(server='cmod',shot=self.shotnumber,treename='analysis',TDI='\EFIT_GEQDSK:R').data()
#        self.r = geqdsk['R']
        self.z = OMFITmdsValue(server='cmod',shot=self.shotnumber,treename='analysis',TDI='\EFIT_GEQDSK:Z').data()
#        self.z = geqdsk['Z']
#        self.times = geqdsk['GTIME']
        self.psi_boundary = geqdsk['SIBRY']
        self.psi_mag_axis = geqdsk['SIMAG']

        self.psin = np.divide(self.psi - self.psi_mag_axis, self.psi_boundary - self.psi_mag_axis)

#        self.rr = np.tile(self.r, (len(self.z), 1))
#        self.zz = np.transpose(np.tile(self.z, (len(self.r), 1)))

        return None


    def rz2Psi(self, rzArray):

        #intepolate the psi grid
        psiInt = interpolate.interp2d(self.r, self.z, self.psin)

        psiArr = np.zeros(len(rzArray))

        for i in range(len(rzArray)):
            currR = rzArray[i,0]
            currZ = rzArray[i,1]

            psiArr[i] = psiInt(currR,currZ)

        return psiArr





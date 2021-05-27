### modified efitAR script using francesco's get geqdsk function in lyman_data

import numpy as np
from lyman_data import get_geqdsk_cmod

class efit:

	def __init__(self, shotnumber, time, efit_id='EFIT01'):

		self.shotnumber = int(shotnumber)
		self.get_geqdsk(efit_id)

	def get_geqdsk(self, efit_id):

		self.efit_id = efit_id

		geqdsk = get_geqdsk_cmod(self.shotnumber,self.time)

		self.limiter = geqdsk['LIM']
		self.psi = geqdsk['PSIRZ']
		self.r = geqdsk['R']
		self.z = geqdsk['Z']
		self.times = geqdsk['GTIME']
		self.psi_boundary = geqdsk['SSIBRY']
		self.psi_mag_axis = geqdsk['SSIMAG']

		self.psin = []
		for i in range(len(self.psi)):
			self.psin.append(np.divide(self.psi[i] - self.psi_mag_axis[i], self.psi_boundary[i] - self.psi_mag_axis[i]))

		self.rr = np.tile(self.r, (len(self.z), 1))
		self.zz = np.transpose(np.tile(self.z, (len(self.r), 1)))

		return None
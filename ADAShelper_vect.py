
import sys
sys.path.insert(1, '/fusion/projects/diagnostics/llama/PythonDataTools/Aurora')
import aurora  as au
import numpy as np
import scipy
import scipy.interpolate

h = 6.626*10**(-34) #J-s
c = 2.9979*10**8 #m/s

class ADAS:


	def __init__(self, wavelength):

		"""
		scb_ion = load_ion_data(imp = 'H',atomdat_dir = '/fusion/projects/diagnostics/llama/PythonDataTools/HTPD/atomAI/strahl_atomdat/newdat_d3d/',plot = False)

		logne2, logTe2,S = scb_ion['scd']  # ionization

		self.ionFunc= scipy.interpolate.interp2d(logne2,logTe2,S[0],kind = 'cubic')
		"""
		
		filename = 'pec12#h_pju#h0.dat'
		path = au.get_adas_file_loc(filename, filetype='adf15')
		pec_dict = au.read_adf15(path)
		pec_recom = pec_dict[wavelength]['recom']
		pec_exc = pec_dict[wavelength]['excit']

		self.recomFunc = pec_recom

		self.excFunc = pec_exc

		self.wvl = wavelength
		

		scb_ion2 = au.get_atom_data('H')

		self.ionFunc= scb_ion2
		

	def ionFuncHelp(self,ne,te):
		#ne and te in cm^(-3) and te eV

		#returns ionization cross section in m^3/s
		#temp = 10**(self.ionFunc(np.log10(ne), np.log10(te))-6)#-6 converts from cm^3/s to m^3/s

		#returns cm^3/s

		temp = au.interp_atom_prof(self.ionFunc['scd'],np.log10(ne),np.log10(te),x_multiply = False)[:,0]

		return temp*1e-6 #convert from cm^3/s to m^3/s
	
	def ion(self,ne,te,neErr, teErr):
		"""
		Inputs:

		ne,- cm^3-float
		te-in eV

		Output:
		ionization cross section - m^3/s
		"""
		#print(self.ionFunc(np.log10(ne), np.log10(te))[0])

		ion = self.ionFuncHelp(ne,te) 

		nete = genPair(ne,te,neErr,teErr).reshape((-1,2))

		x0 = nete[:,0]
		x1 = nete[:,1]


		errIon = self.ionFuncHelp(x0,x1)


		errIon = errIon.reshape((ne.shape[0],-1)).T-ion
		#estimate error by taking the mean of the extremes
		errIonF = (errIon.max(axis=0)+errIon.min(axis=0))/2

		err = ion*np.log(10)*errIonF
		return ion,err
	

	def recom(self,ne,te):

		return 10**self.recomFunc.ev(np.log10(ne), np.log10(te))

	def excit(self,ne,te):

		return 10**self.excFunc.ev(np.log10(ne), np.log10(te))

	"""Inputs:
	emiss - in W/cm^3-float
	ne,ni - cm^3-flot
	te-in eV

	Output:
	Ionization Rate - float - in m^-3s^-1
	"""
	def calcIonRate(self,emiss,ne,ni,te,emissErr=None,neErr=None,niErr=None,teErr=None):

		if np.all(emissErr ==None) or np.all(neErr ==None) or np.all(niErr ==None) or np.all(teErr == None):
			emissErr = np.zeros(ne.shape[0])
			neErr = np.zeros(ne.shape[0])
			niErr = np.zeros(ne.shape[0])
			teErr = np.zeros(ne.shape[0])
		
		#for some reason errors come in as weird mds objects only some of the time
		neErr = neErr.astype(float)
		teErr = teErr.astype(float)
		niErr = niErr.astype(float)

		ion,ionErr = self.ion(ne,te,neErr,teErr)


		n0,n0Err = self.calcNDens(emiss,ne,ni,te,emissErr,neErr,niErr,teErr)
		#print('n0: '+str(n0))
		#print('n0 :'+str(n0Err/n0))

		
		#print(ion)
		#print(ionErr)

		err = np.sqrt((n0Err/n0)**2+(neErr/ne)**2+(ionErr/ion)**2)
		#print('ion :'+str(ionErr/ion))

		#print(err)

		ion = n0*ne*10**12*ion

		#remove all nan entries
		err[np.isnan(err)] = 0

		err = ion*err

		return ion,err




	"""
	Inputs:
	emiss - in W/cm^3-float
	ne,ni - cm^3-flot
	te-in eV

	Output:
	dens - float - in cm^-3
	"""
	def calcNDens(self,emiss,ne,ni,te,emissErr=None,neErr=None,niErr=None,teErr=None):

		if np.all(emissErr ==None) or np.all(neErr ==None) or np.all(niErr ==None) or np.all(teErr == None):
			emissErr = np.zeros(ne.shape[0])
			neErr = np.zeros(ne.shape[0])
			niErr = np.zeros(ne.shape[0])
			teErr = np.zeros(ne.shape[0])
		#for some reason errors come in as weird mds objects only some of the time
		neErr = neErr.astype(float)
		teErr = teErr.astype(float)
		niErr = niErr.astype(float)

		# we will assume that there is no error in the ADAS coefficients
		# but there is error from ne and te through the ADAS coefficient

		excCoef = self.excit(ne,te)
		recomCoef = self.recom(ne,te)



		#1.986*10**(-15) is photons * Angstroms (from wavelength)/plancks constant * speed of light
		#so the emiss*wvl/1.98*10**-15  where emiss in W/cm^3 is of units photon/cm^3/s
		dens = emiss*self.wvl/(1.986449*10**(-15)*(ne*excCoef+ni*recomCoef))






		# Now calculating error, we can't do easy error propogation for coefficients
		# since there is not a close form result
		#We need all combos of err in ne and te
		nete = genPair(ne,te,neErr,teErr).reshape((-1,2))



		x0 = nete[:,0]
		x1 = nete[:,1]


		#calculate excitation coefficient for each pair and subtract off value
		errExC = self.excit(x0,x1).reshape((ne.shape[0],-1)).T-excCoef 
		#estimate error by taking the mean of the extremes
		errExC = (errExC.max(axis=0)+errExC.min(axis=0))/2

		#repeat for recombination coefficient
		errReC = self.recom(x0,x1).reshape((ne.shape[0],-1)).T-recomCoef 
		errReC = (errReC.max(axis=0)+errReC.min(axis=0))/2


		#Now we add the errors by propogating the error
		#first the denominator
		"""
		print('\n')
		print('Dens Err')
		print(neErr)	
		print(type(neErr))
		print(type(float(neErr)))
		print(neErr/ne)
		"""
		denErr = (ne*excCoef)**2*((errExC/excCoef)**2+(neErr/ne)**2)
		denErr += (ni*recomCoef)**2*((errReC/recomCoef)**2+(niErr/ni)**2)
		denErr = np.sqrt(denErr)

		den = (ne*excCoef+ni*recomCoef)
		"""
		print((denErr/den)**2)

		print((emissErr/emiss)**2)
		print(emissErr)
		print(emiss)
		"""

		#numerator error is just the emiss

		err = np.sqrt((denErr/den)**2+(emissErr/emiss)**2)*dens



		return dens,err


		"""Inputs:
		n0 - ground state density - cm^(-3)
		ne,ni - cm^-3-flot
		te-in eV

		Output:
		emissivity - float - W/cm^-3
		"""
	def calcEmiss(self,n0,ne,ni,te):

		


		excCoef = self.excFunc(scipy.log10(ne),scipy.log10(te))
		recomCoef = self.recomFunc(scipy.log10(ne),scipy.log10(te))

		#constant is planck's constant times speed of light in Joules -Angstroms

		#dens = emiss*self.wvl/(1.986449*10**(-15)*(ne*excCoef))
		emiss = (1.986449*10**(-15)*n0*(ne*excCoef+ni*recomCoef))/self.wvl


		return emiss

def genPair(x,y,xErr,yErr):

	xHigh = np.tile(x+xErr,2)
	xLow = np.tile(x-xErr,2)
	yHigh = np.tile(y+yErr,2)
	yLow = np.tile(y-yErr,2)

	temp = np.column_stack((np.concatenate((xHigh,xLow)),np.concatenate((yHigh,yLow))))

	return temp.reshape((x.shape[0],4,-1),order = 'F')

def estErr(l,val):
	err = np.diagonal(l).reshape((-1,4))

	err = (err.T-val) #transpose to subtract ion

	return (err.max(axis=0)+err.min(axis=0))/2



if __name__ == "__main__":
	adas = ADAS(1215.2)

	ne = 5e14  # cm ^-3

	te = 1500 #eV

	emiss = 3e20 #ph/m^3/s
	emiss = emiss*(4*np.pi)*10**(-6)*1.642*10**(-18) #convert ot W/cm^3
	l = 20


	ne = 1.9e13
	te = 50
	emiss = 0.046
	emiss = np.full(l,emiss)*(np.arange(l))

	ne = np.full(l,ne)*np.arange(l)
	te = np.full(l,te)*np.arange(l)

	emissE = emiss*0.2
	neE = ne*0.2
	teE = te*0.2


	print(adas.calcIonRate(emiss,ne,ne,te,emissErr=emissE,neErr=neE,niErr=neE,teErr=teE))
	print(adas.calcNDens(emiss,ne,ne,te,emissErr=emissE,neErr=neE,niErr=neE,teErr=teE))
	#print(adas.calcIonRate(emiss,ne,ne,te))
	#print(adas.calcNDens(emiss,ne,ne,te))

	

	""""
	adas = ADAS(1215.2)

	ne = 1*10**14

	te = 50

	emiss = 0.1

	#These should give an ionization arate of 1 e23 m^-3s^-1
	#and a neutral density of 1e16 m^-3
	
	print(type(ne))
	print(type(te))
	
	print(adas.calcIonRate(emiss,ne,ne,te))
	print(adas.calcNDens(emiss,ne,ne,te))
	"""

	"""
	sys.path.insert(1, '/fusion/projects/diagnostics/llama/PythonDataTools/WindowDict')
	import LyaDictFunc as ldf

	data = ldf.load_dict('/fusion/projects/diagnostics/llama/PythonDataTools/WindowDict/Dicts/180916_P3400_AMRNOV')

	cDict = data['180916_P3400_AMRNOV']

	psi = cDict['ne']['psi']

	iS = np.argmin(np.abs(psi-1.02))
	iE = np.argmin(np.abs(psi-1.035))

	psi = psi[iS:iE]

	for i in range(iE-2,iE):
		print('\n')
		print('nDens '+str(cDict['nDens'][i]/1e16))
		print('err nDens '+str(cDict['err_nDens'][i]/1e16))
		cNe = cDict['ne']['ne'][i]*10**-6
		cNeE = cDict ['ne']['err_ne'][i]*10**-6
		cTe = cDict['te']['te'][i]
		cTeE = cDict ['te']['err_te'][i]

		print(cNe/1e18)
		print(cTe)

		cEmiss = cDict['map_emiss'][i] *(4*np.pi)*10**(-6)*1.642*10**(-18)
		cEErr = cDict['err_map_emiss'][i] *(4*np.pi)*10**(-6)*1.642*10**(-18)

		adas.calcNDens(cEmiss,cNe,cNe,cTe,emissErr = cEErr,neErr = cNeE,niErr = cNeE,teErr = cTeE)
	
	"""

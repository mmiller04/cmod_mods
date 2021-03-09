import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from scipy.stats.mstats import mquantiles

import sys
#sys.path.insert(1, '/fusion/projects/diagnostics/llama/PythonDataTools')
#import efitAR as EFIT
#import gadata
#sys.path.insert(1, '/fusion/projects/diagnostics/llama/PythonDataTools/LLAMA_tomo')
import LLAMA_tomography5_Window as tomo


import scipy.interpolate
from scipy import optimize
import math

import MDSplus as mds
conn = mds.Connection('alcdata.psfc.mit.edu:8000')


def loadEFIT(shotN,dDict):
	print(dDict['efitID'])
	try:
	
		efit = EFIT.efit(shotN, efit_id = str(dDict['efitID']))
	except:
		efit = EFIT.efit(shotN, efit_id = str(dDict['efitID'])[2:8]) #for python 3 b proceeds and causes errors
	print('loaded')
	dDict['efit'] = efit

	return dDict

"""
loads TTools fits, emissivity, and EFIT
window averages the brightness for ELM sync windows before
performing inversion
smoothT is the averaging window if not ELM syncing
"""
def loadShot(shotN, dDict, window = False, smoothT=10):

	#convert to numpy list if not one

	shotN = np.asarray(shotN)

	for i in range(len(shotN)):
		subDict = {}


		cShot = shotN[i,0]
		filename = shotN[i,1]


		subDict['shotN'] = int(cShot)
		#subDict['current'] = shotN[i,2]
		#subDict['gasPuff'] = shotN[i,3]


		try:
			loadTTFit(cShot,filename,subDict)
			print('loaded Toms Tools fit')
		except:
			print("no Tom's Tools fit")

		try:
			loadEFIT(cShot,subDict)
			print('loaded EFIT')
		except:
			print("error with EFIT load")

		
		if window ==True:
			loadBrightEmissWindow(cShot,subDict)
			print('loaded Window Inversion')
		else:
			loadBrightEmiss(cShot,smoothT,subDict)
			print('loaded Inversion')


		dDict[str(cShot)+'_'+filename] = subDict




	return dDict


def loadTTFit(shotN,filename,dDict):

	shotN = int(shotN)


	neR = gadata.gadata('.'+filename+':NEDATR',shotN,tree = 'PROFDB_PED')
	neRAux = gadata.gadata('.'+filename+':NEDATR:YAUX',shotN,tree = 'PROFDB_PED')

	nePsi = gadata.gadata('.'+filename+':NEDATPSI',shotN,tree = 'PROFDB_PED')
	nePsiAux = gadata.gadata('.'+filename+':NEDATPSI:YAUX',shotN,tree = 'PROFDB_PED')

	print('arrays equal?')

	print(np.all(neRAux.zdata == nePsiAux.zdata))
	print(np.all(neR.zdata ==nePsi.zdata))




	subDict = {}

	subDict['ne'] = neR.zdata*1e20
	subDict['err_ne'] = neR.zerr*1e20
	subDict['rMid'] = neR.xdata
	subDict['err_rMid'] = neR.xerr
	subDict['psi'] = nePsi.xdata
	subDict['err_psi'] = nePsi.xerr
	subDict['t'] = neRAux.zdata[:,1]
	subDict['eqT'] = neRAux.zdata[:,2]
	subDict['RTS'] = neRAux.zdata[:,3]
	subDict['ZTS'] = neRAux.zdata[:,4]





	neFit = gadata.gadata('.'+filename+':NETANHPSI:FIT_COEF',shotN,tree = 'PROFDB_PED')

	subDict['top'] = neFit.zdata[2]*1e20
	subDict['fitParam'] = neFit.zdata

	neFit = gadata.gadata('.'+filename+':NETANHPSI:FITDOC',shotN,tree = 'PROFDB_PED')



	neFit = gadata.gadata('.'+filename+':NETANHPSI',shotN,tree = 'PROFDB_PED')

	subDict['fit'] = neFit.zdata*1e20
	subDict['fitPsi'] = neFit.xdata

	neFitR = gadata.gadata('.'+filename+':NETANHR',shotN,tree = 'PROFDB_PED')

	subDict['fitRDat'] = neFitR.zdata*1e20
	subDict['fitR'] = neFitR.xdata

	neTime = gadata.gadata('.'+filename+':TWINDOWS',shotN,tree = 'PROFDB_PED')
	print('TWINDOWS:')
	print(neTime.zdata)
	subDict['tWindow'] = np.asarray(neTime.zdata)


	dDict['ne'] = subDict

	subDictT = {}

	teR = gadata.gadata('.'+filename+':TEDATR',shotN,tree = 'PROFDB_PED')
	teRAux = gadata.gadata('.'+filename+':TEDATR:YAUX',shotN,tree = 'PROFDB_PED')

	tePsi = gadata.gadata('.'+filename+':TEDATPSI',shotN,tree = 'PROFDB_PED')
	tePsiAux = gadata.gadata('.'+filename+':TEDATPSI:YAUX',shotN,tree = 'PROFDB_PED')

	tePsiAuxDOC = gadata.gadata('.'+filename+':TEDATR:YAUX_DOC',shotN,tree = 'PROFDB_PED')

	print('arrays equal?')
	print(np.all(teRAux.zdata == tePsiAux.zdata))
	print(np.all(teR.zdata ==tePsi.zdata))

	subDictT['te'] = teR.zdata*1000
	subDictT['err_te'] = teR.zerr*1000
	subDictT['rMid'] = teR.xdata
	subDictT['err_rMid'] = teR.xerr
	subDictT['psi'] = tePsi.xdata
	subDictT['err_psi'] = tePsi.xerr
	subDictT['t'] = teRAux.zdata[:,1]
	subDictT['eqT'] = teRAux.zdata[:,2]
	subDictT['RTS'] = teRAux.zdata[:,3]
	subDictT['ZTS'] = teRAux.zdata[:,4]

	teFit = gadata.gadata('.'+filename+':TETANHPSI:FIT_COEF',shotN,tree = 'PROFDB_PED')

	subDictT['top'] = teFit.zdata[2]*1000
	subDict['fitParam'] = teFit.zdata

	teFit = gadata.gadata('.'+filename+':NETANHPSI:FITDOC',shotN,tree = 'PROFDB_PED')


	teFit = gadata.gadata('.'+filename+':TETANHPSI',shotN,tree = 'PROFDB_PED')

	subDictT['fit'] = teFit.zdata*1000
	subDictT['fitPsi'] = teFit.xdata

	teFitR = gadata.gadata('.'+filename+':TETANHR',shotN,tree = 'PROFDB_PED')

	subDictT['fitRDat'] = teFit.zdata*1000
	subDictT['fitR'] = teFitR.xdata

	dDict['te'] = subDictT


	conn.openTree('PROFDB_PED',shotN)
	dDict['efitID'] = str(conn.get('.'+filename+':EFITTREE'))

	"""
	print(teR.xdata)
	print(teR.xunits)
	print(teR.ydata)
	print(teR.yunits)
	print(teR.zdata)
	print(teR.zunits)
	print(teR.zdata.shape)
	
	"""

	return dDict

def loadBrightEmiss(shotN,smoothT,shotDict,fileLoc):

	smoothTTxt = str(smoothT)+'msSmooth'

	# tomoDict     = np.load(   fileLoc+'LLAMA_'+str(int(shotN))+'.npz',allow_pickle=True)
	tomoDict = tomo.tomoReturn(shotN)


	for i, key in enumerate(tomoDict):
		shotDict[key] = tomoDict[key]

	shotDict['smoothT'] = smoothT
	return shotDict


def loadBrightEmissWindow(shotN,shotDict):

	print('test')

	tomoDict = tomo.tomoWindow(shotDict['ne']['tWindow'])


	for i, key in enumerate(tomoDict):
		shotDict[key] = tomoDict[key]

	shotDict['smoothT'] = 0 #means infinite
	return shotDict

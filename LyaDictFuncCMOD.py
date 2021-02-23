import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from scipy.stats.mstats import mquantiles

import sys
# sys.path.insert(1, '/fusion/projects/diagnostics/llama/PythonDataTools')

import ADAShelper as ADAS
import DEGAShelper as DEGAS
import efitAR as EFIT
import gadata

import scipy.interpolate as interpolate
from scipy import optimize
import math

import MDSplus as mds
conn = mds.Connection('alcdata.psfc.mit.edu:8000')

import pullData as lD

#for fitting 
from scipy.optimize import curve_fit
from scipy import asarray as ar,exp
from scipy import stats

import scipy.special as sp

import pylab as plb

import h5py


chargeE = 1.602e-19

def save_dict(dDict, filename):

	hf = h5py.File(filename,'w')

	for i,key in enumerate(dDict):

		shotDict = dDict[key]

		cGroup=hf.create_group(key)

		for j,subKey in enumerate(shotDict):

			if subKey =='ne' or subKey == 'te':

				subGroup=cGroup.create_group(subKey)

				for k,subKeyTT in enumerate(shotDict[subKey]):


					subGroup.create_dataset(subKeyTT,data = shotDict[subKey][subKeyTT])


			elif subKey == 'efit':
				continue

			else:
				
				
				cGroup.create_dataset(subKey,data = shotDict[subKey])

	hf.close()

	return 0 




def load_dict(filename):
	hf = h5py.File(filename,'r')

	#loading into here
	dDict = {}


	for i,key in enumerate(hf):

		subDict = {}

		fileSubD = hf[key]


		for j,subKey in enumerate(fileSubD):

			if subKey =='ne' or subKey == 'te':

				

				TTsubD = {}

				for k,subKeyTT in enumerate(fileSubD[subKey]):

					TTsubD[subKeyTT] = np.array(fileSubD[subKey][subKeyTT])

				subDict[subKey] = TTsubD


			else:


				subDict[subKey] = np.array(fileSubD[subKey])


		#you can't save efit objects so load it after the fact
		try:
			print(subDict['efitID'])
			lD.loadEFIT(key[:6],subDict)


			dDict[key] = subDict
		except:
			print('no EFIT stored')
			dDict[key] = subDict

	
	hf.close()
	
	return dDict



"""
This doesn't really average anymore just pulls the correct
window from emiss

The averaging is occuring at the inversion step. The brightness is first
averaged and then inverted. This occurs in the loadDict call.
"""


def AveEmiss(dDict,tStart, tEnd):
	tvec = dDict['time']
	quant = dDict['emiss']
	R_tomo = dDict['radial_grid']
	nr = len(R_tomo)//2

	tMean = (tStart+tEnd)/2



	iMean = np.argmin(np.abs(tvec-tMean))




	print('time mean: '+str(tvec[iMean]))


	print('desired window: '+str(tStart) +' '+str(tEnd))


	aveQuant = quant[iMean,nr:]
	aveErr = dDict['emiss_err'][iMean,nr:]

	return aveQuant,aveErr





def windowAveQuant(tag,cDict,tree = None):

	quant = gadata.gadata(tag,cDict['shotN'],tree = tree)


	if quant.xunits !='ms':
		print('ERORR: Expected ms got '+str(quant.xunits))

	nWindows = cDict['ne']['tWindow'].shape[-1]

	AveQuant = np.zeros(nWindows)

	for i in range(nWindows):


		cWindow = cDict['ne']['tWindow'][:,i]

		tStart = cWindow[0]
		tEnd = cWindow[1]



		iStart = np.argmin(np.abs(quant.xdata-tStart))
		iEnd = np.argmin(np.abs(quant.xdata-tEnd))
		AveQuant[i] = np.mean(quant.zdata[iStart:iEnd])

	#removing NaN values which come from iStart = iEnd
	ave = np.mean(AveQuant[np.logical_not(np.isnan(AveQuant))])
	print(tag+' is: '+ str(ave)+ ' '+quant.zunits)
	return ave

#recursive function for finding psi point on the line between rz1 and rz2

def psi2rz(efitInterp,rz1,rz2,psiVal, numP = 50, tol =0.001):

	if efitInterp(rz1[0],rz1[1])>psiVal or efitInterp(rz2[0],rz2[1])< psiVal:
		print('not in provided range')
		print(efitInterp(rz1[0],rz1[1]))
		print(efitInterp(rz2[0],rz2[1]))
		print(psiVal)

		return np.array([np.inf,np.inf])

	else:
		#create a linear array
		rZArray = np.column_stack((np.linspace(rz1[0],rz2[0],numP),np.linspace(rz1[1],rz2[1],numP)))

		#find the corresponding psi value for the points
		psi = EFIT.rz2PsiQuick(efitInterp,rZArray)

		#Find the closest entry
		iMin = np.argmin(np.abs(psi-psiVal))

		#if we are within tolerance break
		if (psi[iMin]-psiVal) < tol:
				#print('actual psi :' +str(efitInterp(rZArray[iMin][0],rZArray[iMin][1])))
				#print('desired psi :' +str(psiVal))

				return rZArray[iMin]

		#if not run again on closests points
		else:
			

			highI = np.argmax(psi>psiVal)
			lowI = np.nonzero(psi<psiVal)[-1][-1]


			return psi2rz(efitInterp,rZArray[lowI],rZArray[highI],psiVal, numP = numP,tol = tol)

#finds the average r,z, coordinate of psi in the tWindows in shotDict
#largely relies on psi2rz above, this is costly to run. 
def AvePsiR(shotDict, rz1,rz2,psiVal, numP = 50, tol =0.001):

	nWindows = shotDict['ne']['tWindow'].shape[-1]
	efit = shotDict['efit']
	efitT = efit.times


	#using regular python list as we will be appending
	#regularly python lists is fairly fast for appending

	resRZ = []
	#need to find number of windows first
	for j in range(nWindows):


		cWindow = shotDict['ne']['tWindow'][:,j]
		tStart = cWindow[0]
		tEnd = cWindow[1]
		
		#this confusinging gives the time closest to tStart which is >tStart
		#iSt =np.argmin(np.abs(efitT - tStart))
		iSt =np.argmin(np.abs(efitT < tStart))
		#this gives time closest to tEnd, strictly <tEnd
		iE = np.argmin(efitT<tEnd)


		"""
		print('time start: '+str(efitT[iSt]))
		print('time end: '+str(efitT[iE]))
		print('desired window: '+str(tStart) +' '+str(tEnd))
		"""
		cEfitT = efitT[iSt:iE] #this is empty if iSt<iE
		#print(cEfitT)
		for i in range(len(cEfitT)):
			#print('subloop')
			psin = efit.get_psin(cEfitT[i])

			#intepolate the psi grid
			psiInt = interpolate.interp2d(efit.r, efit.z, psin)

			rz = psi2rz(psiInt,rz1,rz2,psiVal,numP, tol)

			resRZ.append([rz[0],rz[1]])

	#print(resRZ)
	result = np.asarray(resRZ)


	return np.mean(result,axis = 0),np.std(result,axis=0)

def TT2LLAMAhelper(currDict,rz1,rz2):

	efit = currDict['efit']
	efitT = currDict['ne']['eqT']

	ne = currDict['ne']['ne']
	nePsi = currDict['ne']['psi']


	mapTTrz = np.zeros(shape = (2,len(nePsi)))


	#there are only a few efit times used in ne and te
	#so we will only interpolate each efit once to save time
	uEfitT = np.unique(currDict['ne']['eqT'])

	for i in range(len(uEfitT)):
		cEfitT = uEfitT[i]
		psin = efit.get_psin(cEfitT)

		#intepolate the psi grid
		psiInt = interpolate.interp2d(efit.r, efit.z, psin)


		#first we find all the points which use this efitTime
		indexL = np.where(efitT==cEfitT)[0]

		for j in range(len(indexL)):
			cI = indexL[j]

			psiVal = nePsi[cI]
			print('searching for: '+str(psiVal))

			llamarz = psi2rz(psiInt,rz1,rz2,psiVal, tol = 0.001)

			mapTTrz[:,cI] = llamarz

	return mapTTrz

def TT2LLAMAhelpErr(currDict,rz1,rz2,nePsi):
	efit = currDict['efit']
	efitT = currDict['ne']['eqT']

	ne = currDict['ne']['ne']

	mapTTrz = np.zeros(shape = (2,len(nePsi)))
	mapTTrzErr = np.zeros(shape = (2,len(nePsi)))
	

	for i in range(len(nePsi)):
		cPsi = nePsi[i]
		mapTTrz[:,i],mapTTrzErr[:,i] = AvePsiR(currDict, rz1,rz2,cPsi)


	return mapTTrz,mapTTrzErr

def mapTT2LLAMA(currDict,FWHM = 0.08):



	#We want to project onto where LLAMA data is so this is an estimate
	R_tomo = currDict['radial_grid']
	z_tomo = np.interp(R_tomo, currDict['R_tg'],currDict['Z_tg'] )

	RZemissPts = np.column_stack((R_tomo,z_tomo))

	nr = len(R_tomo)//2


	rz1 = RZemissPts[nr+1]
	rz2 = RZemissPts[-1] #this is a bit of an estimation  here

	mapTTrz,err = TT2LLAMAhelpErr(currDict,rz1,rz2,currDict['ne']['psi'])

	#there is error from the TT fit as well which we can map
	mapTTrzH,errH = TT2LLAMAhelpErr(currDict,rz1,rz2,currDict['ne']['psi']+currDict['ne']['err_psi'])
	mapTTrzL,errL = TT2LLAMAhelpErr(currDict,rz1,rz2,currDict['ne']['psi']-currDict['ne']['err_psi'])


	

	rErr = np.abs(np.column_stack((mapTTrzH[0,:],mapTTrzL[0,:]))-np.column_stack((mapTTrz[0,:],mapTTrz[0,:])))
	zErr = np.abs(np.column_stack((mapTTrzH[1,:],mapTTrzL[1,:]))-np.column_stack((mapTTrz[1,:],mapTTrz[1,:])))


	#there may be som inf points which could cause some errors, so we remove them
	errTT = np.zeros(shape = (2,len(rErr)))
	for i in range(len(rErr)):
		errTT[0,i] = np.mean(rErr[i,:][np.isfinite(rErr[i,:])])
		errTT[1,i] = np.mean(zErr[i,:][np.isfinite(zErr[i,:])])



	

	totErr = np.sqrt(errTT**2+err**2)
	"""
	plt.ioff()
	f,a = plt.subplots(5,1)
	a[0].errorbar(mapTTrz[0,:],currDict['ne']['ne'],xerr = currDict['ne']['err_rMid'],yerr = currDict['ne']['err_ne'])
	a[1].errorbar(mapTTrz[0,:],currDict['ne']['ne'],xerr = errTT[0,:],yerr = currDict['ne']['err_ne'])
	a[2].errorbar(mapTTrz[0,:],currDict['ne']['ne'],xerr =totErr[0,:],yerr = currDict['ne']['err_ne'])

	"""

	#checking rMid
	rz1 = np.asarray([1.8,0])
	rz2 = np.asarray([2.5,0])

	mapTTrzH,errH = TT2LLAMAhelpErr(currDict,rz1,rz2,currDict['ne']['psi']+currDict['ne']['err_psi'])
	mapTTrzL,errL = TT2LLAMAhelpErr(currDict,rz1,rz2,currDict['ne']['psi']-currDict['ne']['err_psi'])

	rMid = currDict['ne']['rMid']
	rErr = np.mean(np.abs(np.column_stack((mapTTrzH[0,:],mapTTrzL[0,:]))-np.column_stack((rMid,rMid))),axis = 1)
	#zErr = np.mean(np.abs(np.column_stack((mapTTrzH[1,:],mapTTrzL[1,:]))-np.column_stack((mapTTrz[1,:],mapTTrz[1,:]))),axis = 1)
	"""
	a[3].errorbar(rMid,currDict['ne']['ne'],xerr = rErr,yerr = currDict['ne']['err_ne'])
	a[4].errorbar(rMid,currDict['ne']['ne'],xerr = currDict['ne']['err_rMid'],yerr = currDict['ne']['err_ne'])

	plt.show()
	"""
	"""

	#now calculating the error by perturbing Z by 1/2 FWHM

	Hrz1 = rz1+np.asarray([0,FWHM/2])
	Hrz2 = rz2+np.asarray([0,FWHM/2])

	Lrz1 = rz1-np.asarray([0,FWHM/2])
	Lrz2 = rz2-np.asarray([0,FWHM/2])

	HmapTTrz =  TT2LLAMAhelper(currDict,Hrz1,Hrz2)
	LmapTTrz = TT2LLAMAhelper(currDict,Lrz1,Lrz2)

	#taking the average of errors
	rErr = np.abs(np.mean(np.column_stack((HmapTTrz[0,:],LmapTTrz[0,:])),axis = 1) - mapTTrz[0,:])
	zErr = np.abs(np.mean(np.column_stack((HmapTTrz[1,:],LmapTTrz[1,:])),axis = 1) - mapTTrz[1,:])
	err = np.column_stack((rErr,zErr))

	"""
	return mapTTrz,totErr







def meTTHelper(currDict,RZemissPts,emiss,efit,nr):

	ne = currDict['ne']['ne']
	mapEmiss = np.zeros(len(ne))

	for i in range(len(ne)):

		efitT = currDict['ne']['eqT'][i]

		psiVal = efit.rz2Psi(RZemissPts[nr:],efitT)



		emissInt = interpolate.interp1d(psiVal,emiss,fill_value = 'extrapolate')

		mapEmiss[i] = emissInt(currDict['ne']['psi'][i])

	return mapEmiss





def mapEmissTT(shotDict,emiss,FWHM=0.08):

	currDict = shotDict
	efit = currDict['efit']
	R_tomo = currDict['radial_grid']
	z_tomo = np.interp(R_tomo, currDict['R_tg'],currDict['Z_tg'] )

	ne = currDict['ne']['ne']

	nr = len(R_tomo)//2

	RZemissPts = np.column_stack((R_tomo,z_tomo))



	if (np.any(currDict['ne']['psi'] != currDict['te']['psi'])) or (np.any(currDict['ne']['rMid'] != currDict['te']['rMid'])):
		print('Error: Ne and Te arrays differ')


	mapEmiss = meTTHelper(currDict,RZemissPts,emiss,efit,nr)
	mapEmissErrInv = meTTHelper(currDict,RZemissPts,currDict['aveEmissErr'],efit,nr)

	#Now calculating Error
	#we will perturb the z coordinate up and down by FWHM/2

	z_tomoH = np.interp(R_tomo, currDict['R_tg'],currDict['Z_tg'] + FWHM/2)
	z_tomoL = np.interp(R_tomo, currDict['R_tg'],currDict['Z_tg'] - FWHM/2)


	HRZemissPts = np.column_stack((R_tomo,z_tomoH))
	LRZemissPts = np.column_stack((R_tomo,z_tomoL))

	mapEmissErr = np.zeros(shape=(len(ne),2))

	mapEmissErr[:,0] = meTTHelper(currDict,HRZemissPts,emiss,efit,nr)
	mapEmissErr[:,1] = meTTHelper(currDict,LRZemissPts,emiss,efit,nr)

	"""
	plt.ioff()
	f,a = plt.subplots(3,1)
	a[0].plot(currDict['ne']['psi'],mapEmissErr[:,0],c = 'r')
	a[0].plot(currDict['ne']['psi'],mapEmissErr[:,1],c = 'g')
	a[0].plot(currDict['ne']['psi'],mapEmiss,c = 'b')
	a[0].set_xlim([0.8,1.2])
	a[0].set_ylim([0,np.amax(mapEmiss)*1.2])
	"""

	#We will take the average of the errors and the difference from the calculated value
	#divide by sqrt(3) assuming a square distribution of errors 
	mapEmissErr =np.mean(np.abs(mapEmissErr - np.column_stack((mapEmiss,mapEmiss))),axis = 1)/np.sqrt(3)
	mapEmissErr = np.sqrt(mapEmissErr**2+mapEmissErrInv**2)


	#a[0].errorbar(currDict['ne']['psi'],mapEmiss,yerr = mapEmissErr)


	tEmiss = emiss[1:]
	emissG = np.zeros(len(tEmiss))
	r = R_tomo[nr+1:]
	pmR = 0.02


	for i in range(len(tEmiss)):


		cR = r[i]

				#< 0 condition is actually error in extrpolation in emissInterp
		if cR ==np.inf or math.isnan(cR):
			continue

		rStart = cR-pmR
		rStop = cR+pmR

		iS = np.argmin(np.abs(r-rStart))
		iE = np.argmin(np.abs(r-rStop))

		cDat = tEmiss[iS:iE]
		x = r[iS:iE]


		if np.any(x==np.inf):
			continue


		coef = np.polyfit(x,cDat,1)

		poly1d_fn = np.poly1d(coef)

		#a.plot(x,poly1d_fn(x),label = 'fit',color = 'b')




		emissG[i] = coef[0] 


	emissG*= 0.01
	emissG = np.abs(emissG)
	localApproxErr = meTTHelper(currDict,RZemissPts[1:],emissG,efit,nr)
	err2 = np.sqrt(localApproxErr**2+mapEmissErrInv**2)
	"""
	a[1].errorbar(currDict['ne']['psi'],mapEmiss,yerr = err2)
	a[1].set_xlim([0.8,1.2])
	a[1].set_ylim([0,np.amax(mapEmiss)*1.2])

	a[2].plot(currDict['ne']['psi'],localApproxErr,c = 'r')
	a[2].plot(currDict['ne']['psi'],mapEmissErrInv,c = 'b')
	a[2].plot(currDict['ne']['psi'],mapEmissErr,c = 'g')

	a[2].set_xlim([0.8,1.2])
	a[2].set_ylim([0,np.amax(mapEmissErr)*1.2])

	plt.show()
	"""

	#We ad in quadreture to the error in the inversion
	#mapEmissErr = np.sqrt(mapEmissErr**2+currDict['aEmissErr']**2)

	return mapEmiss,err2


def calcIonizationRZ(shotDict, ADASobj):

	ne = shotDict['ne']['ne']
	te = shotDict['te']['te']
	neErr = shotDict['ne']['err_ne']
	teErr = shotDict['te']['err_te']

	#for some reason Tom's Tools gives different lengths of arrays for 
	#so we will map te onto the ne points

	teInt = interpolate.interp1d(shotDict['TS_lya_rz'][0,:],te,fill_value = 'extrapolate')
	teIntErr = interpolate.interp1d(shotDict['TS_lya_rz'][0,:],teErr,fill_value = 'extrapolate')

	#we are only interpreting on r so we are assuming changes in z are small
	#need to check if this is true - AR Jun 23 2020
	R_tomo = shotDict['radial_grid']
	nr = len(R_tomo)//2

	emissInterp = interpolate.interp1d(R_tomo[nr:],shotDict['aveEmiss'],\
		fill_value = 'extrapolate')
	emissInterpErr = interpolate.interp1d(R_tomo[nr:],shotDict['aveEmissErr'],\
		fill_value = 'extrapolate')

	res = np.zeros(len(ne))
	err = np.zeros(len(ne))

	for i in range(len(ne)):

		cR = shotDict['TS_lya_rz'][0,i]
		cEmiss = emissInterp(cR)*(4*np.pi)*10**(-6)*1.642*10**(-18)
		cEErr = emissInterpErr(cR)*(4*np.pi)*10**(-6)*1.642*10**(-18)
		cTe = teInt(cR)
		cTeE =teIntErr(cR)


		#< 0 condition is actually error in extrpolation in emissInterp
		if cEmiss < 0 or cR ==np.inf or math.isnan(cTe):
			continue

		cNe = ne[i]*10**(-6)#convert to cm^(-3)
		cNeE = neErr[i]*10**(-6)#convert to cm^(-3)
		


		cNi = cNe
		cNiE = cNeE
		"""
		print('emiss err')
		print(cR)
		print(cEErr)
		print(cEmiss)

		#import pdb
		#pdb.set_trace()
		"""
		
		 


		#result is in m^{-3}s^-1
		res[i],err[i] = ADASobj.calcIonRate(cEmiss,cNe,cNe,cTe,emissErr = cEErr,neErr = cNeE,niErr = cNiE,teErr = cTeE)
		if math.isnan(res[i]):
			print('isnan')
			print(cEmiss)
			print(cNe)
			print(cTe)
	"""
	
	plt.ioff()
	f,a = plt.subplots(2,1)
	a[0].errorbar(shotDict['TS_lya_rz'][0,:],res,xerr =shotDict['err_TS_lya_rz'][0:] ,yerr = err)
	a[1].errorbar(shotDict['TS_lya_rz'][0,:],emissInterp(shotDict['TS_lya_rz'][0,:]) ,yerr = emissInterpErr(shotDict['TS_lya_rz'][0,:]))
	a[1].errorbar(R_tomo[nr:],shotDict['aveEmiss'] ,yerr = shotDict['aveEmissErr'])

	plt.show()
	"""

	return res,err

def calcNeutIon(shotDict,RATEobj,key):


	ne = shotDict['ne']['ne']
	te = shotDict['te']['te']
	neErr = shotDict['ne']['err_ne']
	teErr = shotDict['te']['err_te']

	#for some reason Tom's Tools gives different lengths of arrays for 
	#so we will map te onto the ne points

	teInt = interpolate.interp1d(shotDict['te']['psi'],te,fill_value = 'extrapolate')
	teIntErr = interpolate.interp1d(shotDict['te']['psi'],teErr,fill_value = 'extrapolate')

	res = np.zeros(len(ne))
	err = np.zeros(len(ne))
	for i in range(len(ne)):
		cPsi = shotDict['ne']['psi'][i]

		#converting to W/cm^3 so multiply by energy of photon
		cEmiss = shotDict['map_emiss'][i] *(4*np.pi)*10**(-6)*1.642*10**(-18)
		cEErr = shotDict['err_map_emiss'][i] *(4*np.pi)*10**(-6)*1.642*10**(-18)

		#< 0 condition is actually error in extrpolation in mappEmiss
		if cPsi<0.8 or cEmiss < 0 :
			continue

		#converting to W/cm^3 so multiply by energy of photon


		cNe = ne[i]*10**(-6)#convert to cm^(-3)
		cNeE = neErr[i]*10**(-6)#convert to cm^(-3)
		cTe = teInt(cPsi)
		cTeE =teIntErr(cPsi)


		cNi = cNe
		cNiE = cNeE

		if key == 'ion':

			#result is in m^{-3}s^-1
			res[i],err[i] = RATEobj.calcIonRate(cEmiss,cNe,cNe,cTe,emissErr = cEErr,neErr = cNeE,niErr = cNiE,teErr = cTeE)
		else:
			res[i],err[i] = RATEobj.calcNDens(cEmiss,cNe,cNe,cTe,emissErr = cEErr,neErr = cNeE,niErr = cNiE,teErr = cTeE)
			#result is in m^{-3}s^-1
			res[i]*=10**6
			err[i]*=10**6

	return res,err


def calcNeutDensRZ(shotDict,ADASobj):

	ne = shotDict['ne']['ne']
	te = shotDict['te']['te']
	neErr = shotDict['ne']['err_ne']
	teErr = shotDict['te']['err_te']

	#for some reason Tom's Tools gives different lengths of arrays for 
	#so we will map te onto the ne points

	teInt = interpolate.interp1d(shotDict['TS_lya_rz'][0,:],te,fill_value = 'extrapolate')
	teIntErr = interpolate.interp1d(shotDict['TS_lya_rz'][0,:],teErr,fill_value = 'extrapolate')

	#we are only interpreting on r so we are assuming changes in z are small
	#need to check if this is true - AR Jun 23 2020
	R_tomo = shotDict['radial_grid']
	nr = len(R_tomo)//2

	emissInterp = interpolate.interp1d(R_tomo[nr:],shotDict['aveEmiss'],\
		fill_value = 'extrapolate')
	emissInterpErr = interpolate.interp1d(R_tomo[nr:],shotDict['aveEmissErr'],\
		fill_value = 'extrapolate')

	res = np.zeros(len(ne))
	err = np.zeros(len(ne))

	for i in range(len(ne)):

		cR = shotDict['TS_lya_rz'][0,i]
		cEmiss = emissInterp(cR)*(4*np.pi)*10**(-6)*1.642*10**(-18)
		cEErr = emissInterpErr(cR)*(4*np.pi)*10**(-6)*1.642*10**(-18)
		cTe = teInt(cR)
		cTeE =teIntErr(cR)


		#< 0 condition is actually error in extrpolation in mappEmiss
		if cR ==np.inf or cEmiss < 0 or math.isnan(cTe):
			continue

		cNe = ne[i]*10**(-6)#convert to cm^(-3)
		cNeE = neErr[i]*10**(-6)#convert to cm^(-3)
		


		cNi = cNe
		cNiE = cNeE




		res[i],err[i] = ADASobj.calcNDens(cEmiss,cNe,cNe,cTe,emissErr = cEErr,neErr = cNeE,niErr = cNiE,teErr = cTeE)

	#result is in m^{-3}s^-1
	res*=10**6
	err*=10**6
	return res,err



def shotDict(shotList,ADASfile):

	"""loads TTools EFIT and emissivity data then loads relevant physics parameters to
	the dictionar

	Input shotN-numpy array - [[shotN,TTools filename,label]]
	y"""
	
	dDict = {}

	lD.loadShot(shotList,dDict,window = True)
	print(dDict.keys())



	for i,key in enumerate(dDict):
		print('currDict: '+str(key))


		currDict = dDict[key]

		nWindows = currDict['ne']['tWindow'].shape[-1]



		#we need the shape of the emissivity array
		nr = len(currDict['radial_grid'])//2

		# add on the number of windows we are looking at
		eShape = np.append(currDict['emiss'][-1,nr:].shape,nWindows)
		print('emiss Shape')
		print(eShape)
		aEmiss = np.zeros(shape = (eShape))
		aEmissErr = np.zeros(shape = (eShape))
		
		#take the averges over the ELM windows

		gPAve = windowAveQuant('gasa_cal',currDict)
		currAve =  windowAveQuant('Ip',currDict)
		dTopAve = windowAveQuant('.PROFILE_FITS.TANHFIT.DENSITY:PED',currDict,tree = 'ELECTRONS')
		neBarAve = windowAveQuant('density',currDict)
		pInjA = windowAveQuant('pinj',currDict)


		aEmiss =np.ndarray.flatten(currDict['emiss'])[nr:]
		aEmissErr =np.ndarray.flatten(currDict['emiss_err'])[nr:]

		currDict['aveEmiss']=aEmiss
		currDict['aveEmissErr']=aEmissErr


		mapE,mapEErr = mapEmissTT(dDict[key],aEmiss)

		#load them up
		currDict['map_emiss'] = mapE
		currDict['err_map_emiss'] = mapEErr
		currDict['densTop'] = dTopAve
		currDict['gasPuff']=gPAve
		currDict['neBar'] = neBarAve
		currDict['current']=currAve
		currDict['pInj'] = pInjA
		"""
		TTrz ,TTrzErr= mapTT2LLAMA(dDict[key])
		currDict['TS_lya_rz'] = TTrz
		currDict['err_TS_lya_rz'] = TTrzErr
		"""

		#addELMf(currDict,key)



	
		
		#calculate the things related to mapEmiss and aEmiss

		ionR,ionErr = calcNeutIon(currDict,ADASfile,'ion')
		
		currDict['ion'] = ionR
		currDict['err_ion'] = ionErr
		"""
		currDict['ionRS'],currDict['err_ionRS'] = calcIonizationRZ(currDict,ADASfile)

		"""
		nDens,nErr = calcNeutIon(currDict,ADASfile,'dens')
		currDict['nDens'] = nDens
		currDict['err_nDens'] = nErr


	return dDict



if __name__ == "__main__":
	main()
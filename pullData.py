import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from scipy.stats.mstats import mquantiles

import sys
import efitAM as EFIT
import kinetic_profiles as kp

import LLAMA_tomography5_Window as tomo


import scipy.interpolate
from scipy import optimize
import math

import MDSplus as mds
conn = mds.Connection('alcdata.psfc.mit.edu:8000')


def loadEFIT(shotN,dDict):
    # print(dDict['efitID'])
    # try:  
    efit_time = (dDict['ne']['tWindow'][0] + dDict['ne']['tWindow'][1])/2
    efit = EFIT.efit(shotN,efit_time)
    # except:
    #   efit = EFIT.efit(shotN, efit_id = str(dDict['efitID'])[2:8]) #for python 3 b proceeds and causes errors
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

        cShot = shotN[i]
#        filename = shotN[i,1]


        subDict['shotN'] = int(cShot)
        #subDict['current'] = shotN[i,2]
        #subDict['gasPuff'] = shotN[i,3]


        try:
            loadKineticFitsWindow(cShot,subDict)
            print('loaded kinetic fits')
        except:
            print("no kinetic fits")

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


        dDict[str(cShot)] = subDict




    return dDict


def loadKineticFitsWindow(shotN,dDict):

    shotN = int(shotN)

    profs = kp.cmoddata(shotN)
    profs.ne_Te_data()

    subDict = {}

    subDict['ne'] = profs.ne_data
    subDict['err_ne'] = profs.ne_err
    subDict['rMid'] = profs.Rmid_ne
    subDict['err_rMid'] = profs.Rmid_ne_err
    subDict['psi'] = profs.psin_ne
    subDict['err_psi'] = profs.psin_ne_err

    # subDict['t'] = neRAux.zdata[:,1]
    # subDict['eqT'] = neRAux.zdata[:,2]
    # subDict['RTS'] = neRAux.zdata[:,3]
    # subDict['ZTS'] = neRAux.zdata[:,4]

    subDict['TS_inds'] = profs.ne_TS_inds
    subDict['SP_inds'] = profs.ne_SP_inds

    profs.ne_Te_fits()

    # subDict['top'] = neFit.zdata[2]*1e20
    subDict['fitParam'] = profs.c_ne
    subDict['fit'] = profs.res_fit_ne
    subDict['fitPsi'] = profs.psin_ne
    # subDict['fitRDat'] = neFitR.zdata*1e20
    # subDict['fitR'] = neFitR.xdata

    print('TWINDOWS:')
    print(str(profs.TS_tmin)+':'+str(profs.TS_tmax))
    subDict['tWindow'] = [[profs.TS_tmin,profs.TS_tmax]]

    dDict['ne'] = subDict


    subDictT = {}

    subDictT['te'] = profs.Te_data
    subDictT['err_te'] = profs.Te_err
    subDictT['rMid'] = profs.Rmid_Te
    subDictT['err_rMid'] = profs.Rmid_Te_err
    subDictT['psi'] = profs.psin_Te
    subDictT['err_psi'] = profs.psin_Te_err

    # subDictT['t'] = neRAux.zdata[:,1]
    # subDictT['eqT'] = neRAux.zdata[:,2]
    # subDictT['RTS'] = neRAux.zdata[:,3]
    # subDictT['ZTS'] = neRAux.zdata[:,4]

    subDictT['TS_inds'] = profs.Te_TS_inds
    subDictT['SP_inds'] = profs.Te_SP_inds

    profs.ne_Te_fits()

    # subDictT['top'] = neFit.zdata[2]*1e20
    subDictT['fitParam'] = profs.c_Te
    subDictT['fit'] = profs.res_fit_Te
    subDictT['fitPsi'] = profs.psin_Te
    # subDictT['fitRDat'] = neFitR.zdata*1e20
    # subDictT['fitR'] = neFitR.xdata

    print('TWINDOWS:')
    print(str(profs.TS_tmin)+':'+str(profs.TS_tmax))
    subDictT['tWindow'] = [[profs.TS_tmin,profs.TS_tmax]]

    dDict['te'] = subDictT

    # conn.openTree('PROFDB_PED',shotN)
    # dDict['efitID'] = str(conn.get('.'+filename+':EFITTREE'))

    return dDict


def loadBrightEmissWindow(shotN,shotDict):

    tomoDict = tomo.tomoCMOD(shotN,shotDict['ne']['tWindow'])

    for i, key in enumerate(tomoDict):
        shotDict[key] = tomoDict[key]

    # shotDict['smoothT'] = 0 #means infinite

    return shotDict




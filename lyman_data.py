'''
Methods to load and plot C-Mod Ly-alpha data.

sciortino, August 2020
'''
import matplotlib.pyplot as plt
plt.ion()
import numpy as np
import xarray
from scipy.interpolate import interp1d
from omfit_classes import omfit_eqdsk, omfit_mds
import shutil,os, scipy
from IPython import embed

from scipy.constants import Boltzmann as kB, e as q_electron

import sys
sys.path.insert(1,'/home/millerma/Aurora')
import aurora


def get_vol_avg_pressure(shot,time,rhop,ne,Te):
    ''' Calculate volume-averaged pressure given some ne,Te radial profiles.

    ne must be in cm^-3 units and Te in eV.
    '''
    # find volume-averaged pressure    
    p_Pa = (ne*1e6) * (Te*q_electron)
    p_atm = p_Pa/101325.  # conversion factor between Pa and atm

    # load geqdsk dictionary
    geqdsk = get_geqdsk_cmod(shot,time*1e3)
    
    # find volume average within LCFS
    indLCFS = np.argmin(np.abs(rhop-1.0))
    p_Pa_vol_avg = aurora.vol_average(p_Pa[:indLCFS], rhop[:indLCFS], method='omfit',geqdsk = geqdsk)[-1]
    #p_atm_vol_avg = p_Pa_vol_avg/101325.

    return p_Pa_vol_avg


def get_geqdsk_cmod(shot,time):
    ''' Get a geqdsk file in omfit_eqdsk format by loading it from disk, if available, 
    or from MDS+ otherwise.  

    time must be in ms!
    '''
    time = np.floor(time)   # TODO: better to do this outside!!
    file_name=f'g{shot}.{str(int(time)).zfill(5)}'
    gfiles_loc = '/home/millerma/EFIT/gfiles/'
    
    def fetch_and_move():
        geqdsk = omfit_eqdsk.OMFITgeqdsk('').from_mdsplus(
            device='CMOD',shot=shot, time=time, SNAPfile='ANALYSIS',
            fail_if_out_of_range=True,time_diff_warning_threshold=20
        )
        geqdsk.save(raw=True)
        #from IPython import embed
        #embed()
        shutil.move(file_name, gfiles_loc+file_name)
    
    if os.path.exists(gfiles_loc + file_name):
        # fetch local g-file if available
        try:
            geqdsk = omfit_eqdsk.OMFITgeqdsk(gfiles_loc + file_name)
            #print(f'Fetched g-file for shot {shot}')
            kk = geqdsk.keys()  # quick test
        except:
            geqdsk = fetch_and_move()
    else:
        geqdsk = fetch_and_move()
        
    return geqdsk

    
def get_Greenwald_frac(shot, tmin,tmax, roa, ne, Ip_MA, a_m=0.22):
    ''' Calculate Greenwald density fraction by normalizing volume-averaged density.

    INPUTS
    ------
    shot : int, shot number
    tmin and tmax: floats, time window (in [s]) to fetch equilibrium.
    ne: 1D array-like, expected as time-independent. Units of 1e20 m^-3.
    Ip_MA: float, plasma current in MA. 
    a_m : minor radius in units of [m]. Default of 0.69 is for C-Mod. 

    OUTPUTS
    -------
    n_by_nG : float
        Greenwald density fraction, defined with volume-averaged density.
    '''
    tmin *= 1000.  # change to ms
    tmax *= 1000.  # change to ms
    time = (tmax+tmin)/2.
    geqdsk = get_geqdsk_cmod(shot,time)
    
    # find volume average within LCFS
    rhop = aurora.rad_coord_transform(roa,'r/a','rhop', geqdsk)

    indLCFS = np.argmin(np.abs(rhop-1.0))
    n_volavg = aurora.vol_average(ne[:indLCFS], rhop[:indLCFS], geqdsk=geqdsk)[-1]

    # Greenwald density
    n_Gw = Ip_MA/(np.pi * a_m**2)   # units of 1e20 m^-3, same as densities above

    # Greenwald fraction:
    f_gw = n_volavg/n_Gw

    return f_gw


    
                    
def get_CMOD_gas_fueling(shot, tmin=None, tmax=None, get_rate=False, plot=False):
    ''' Load injected gas amounts and give a grand total in Torr-l 
    Translated from gas_input2_ninja.dat scope. 

    tmin and tmax must be in units of [s]. If given, a mean result over that time range 
    will be returned; otherwise, both the time base and the gas time series will be returned. 

    If get_rate==True, the time derivative of the total injected gas amount is output, in units of [Torr-l/s]
    '''
    
    _c_side = smooth(omfit_mds.OMFITmdsValue(server='CMOD',shot=shot,treename='cmod',TDI='\\plen_cside').data()[0,:],31)
    _t = omfit_mds.OMFITmdsValue(server='CMOD',shot=shot,treename='cmod',TDI='dim_of(\\plen_cside)').data()
    _b_sideu = smooth(omfit_mds.OMFITmdsValue(server='CMOD',shot=shot,treename='cmod',TDI='\\plen_bsideu').data()[0,:],31)
    _b_top = smooth(omfit_mds.OMFITmdsValue(server='CMOD',shot=shot,treename='cmod',TDI='\\plen_btop').data()[0,:],31)

    plen_bot_time = omfit_mds.OMFITmdsValue(server='CMOD',shot=shot,treename='edge',TDI='\edge::gas_ninja.plen_bot').dim_of(0)
    plen_bot = smooth(omfit_mds.OMFITmdsValue(server='CMOD',shot=shot,treename='edge',TDI='\edge::gas_ninja.plen_bot').data()[0,:],31)

    # only work with quantities within [0,2]s interval
    ind0 = np.argmin(np.abs(_t)); ind1 = np.argmin(np.abs(_t-2.0))

    time = _t[ind0:ind1]
    c_side = _c_side[ind0:ind1]
    b_sideu = _b_sideu[ind0:ind1]
    b_top = _b_top[ind0:ind1]

    # ninja system is on a different time base than the other measurements
    ninja2 = interp1d(plen_bot_time, plen_bot, bounds_error=False)(time)
    
    gas_tot = c_side + b_sideu + b_top + ninja2

    if get_rate:
        gas_tot = smooth(np.gradient(smooth(gas_tot,31), time),31)  # additional smoothing (x2)
        
    if plot:
        fig,ax = plt.subplots()
        ax.plot(time, gas_tot, label='total')
        ax.plot(time, c_side if not get_rate else np.gradient(c_side,time), label='c-side')
        ax.plot(time, b_sideu if not get_rate else np.gradient(b_sideu,time), label='b-side u')
        ax.plot(time, b_top if not get_rate else np.gradient(b_top,time), label='b-top')
        ax.plot(time, ninja2 if not get_rate else np.gradient(ninja2,time), label='ninja2')
        ax.legend(loc='best').set_draggable(True)
        ax.set_xlabel('time [s]')
        ax.set_ylabel('Total injected gas [Torr-l]' if not get_rate else 'Total injected gas rate [Torr-l/s]')
        
    if (tmin is not None) and (tmax is not None):
        itmin = np.argmin(np.abs(time-tmin))
        itmax = np.argmin(np.abs(time-tmax))
        gas_tot = np.mean(gas_tot[itmin:itmax])
        return gas_tot
    else:
        return time, gas_tot

    
def get_Lya_data(shot=1080416024, systems=['LYMID'], plot=True):
    ''' Get Ly-alpha data for C-Mod from any (or all) of the systems:
    ['LYMID','WB1LY','WB4LY','LLY','BPLY']
    '''

    bdata = {} # BRIGHT node
    edata = {} # EMISS node

    if systems=='all':
        systems=['LYMID','WB1LY','WB4LY','LLY','BPLY']
        
    if plot:
        fig, ax = plt.subplots(1,2, figsize=(13,8))
        ls = ['-','--','-.',':','--']

    for ss,system in enumerate(systems):
        fetched_0=True; fetched_1=True
        try:
            bdata[system] = fetch_bright(shot, system)
            
            if plot:
                for ch in np.arange(bdata[system].values.shape[1]):
                    ax[0].plot(bdata[system].time, bdata[system].values[:,ch],
                               label=system+', '+str(ch), ls=ls[ss])
        except:
            print('Could not fetch C-Mod Ly-alpha BRIGHT data from system '+system)
            fetched_0=False
            pass
        
        try:
            edata[system] = fetch_emiss(shot, system)
            if plot:
                for ch in np.arange(edata[system].values.shape[1]):
                    ax[1].plot(edata[system].time, edata[system].values[:,ch],
                               label=system+', '+str(ch), ls=ls[ss])
        except:
            print('Could not fetch C-Mod Ly-alpha EMISS data from system '+system)
            fetched_1=False
            pass
        
        if plot:
            ax[0].set_xlabel('time [s]'); ax[1].set_xlabel('time [s]')
            if fetched_0: ax[0].set_ylabel(r'Brightness [$'+str(bdata[system].units)+'$]')
            if fetched_1: ax[1].set_ylabel(r'Emissivity [$'+str(edata[system].units)+'$]')
            ax[0].legend(); ax[1].legend()
            
    return bdata,edata


def fetch_bright(shot,system):

    _bdata = {}
    node = omfit_mds.OMFITmdsValue(server='CMOD', shot=shot, treename='SPECTROSCOPY',
                         TDI='\\SPECTROSCOPY::TOP.BOLOMETER.RESULTS.DIODE.'+\
                             '{:s}:BRIGHT'.format(system))
    _bdata = xarray.DataArray(
        node.data(), coords={'time':node.dim_of(1),'R':node.dim_of(0)},
        dims=['time','R'],
        attrs={'units': node.units()})

    return _bdata


def fetch_emiss(shot,system):

    _edata = {}
    node = omfit_mds.OMFITmdsValue(server='CMOD', shot=shot, treename='SPECTROSCOPY',
                         TDI='\\SPECTROSCOPY::TOP.BOLOMETER.RESULTS.DIODE.'+\
                             '{:s}:EMISS'.format(system))
    _edata = xarray.DataArray(
        node.data(), coords={'time':node.dim_of(1),'R':node.dim_of(0)},
        dims=['time','R'],
        attrs={'units': node.units()})

    #print('Emissivity units: ' , node.units())

    return _edata


def get_CMOD_1D_geom(shot,time):

    # right gap
    tmp = omfit_mds.OMFITmdsValue(server='CMOD', treename='ANALYSIS', shot=shot,
                        TDI='\\ANALYSIS::TOP.EFIT.RESULTS.A_EQDSK.ORIGHT')
    time_vec = tmp.dim_of(0)
    _gap_R = tmp.data()
    gap_R = _gap_R[time_vec.searchsorted(time)-1]

    # R location of LFS LCFS
    tmp = omfit_mds.OMFITmdsValue(server='CMOD', treename='ANALYSIS', shot=shot,
                        TDI='\\ANALYSIS::TOP.EFIT.RESULTS.G_EQDSK.RBBBS')
    time_vec = tmp.dim_of(0)
    _rbbbs = tmp.data()*1e2 # m --> cm
    rbbbs = _rbbbs[:,time_vec.searchsorted(time)-1]

    Rsep = np.max(rbbbs)

    return Rsep,gap_R




def plot_emiss(edata, shot, time, ax=None):
    ''' Plot emissivity profile '''

    # get Rsep and gap
    Rsep, gap = get_CMOD_1D_geom(shot,time)
    Rwall = Rsep+gap
    print('Rwall,Rsep,gap:',Rwall, Rsep,gap)

    if ax is None:
        fig,ax = plt.subplots()

    tidx = np.argmin(np.abs(edata.time.values - time))
    ax.plot(edata.R.values, edata.values[tidx,:], '.-')  #*100 - Rwall

    ax.set_ylabel(r'emissivity [${:}$]'.format(edata.units))
    ax.set_xlabel(r'R [cm]')
    return ax


def plot_bright(bdata, shot, time,ax=None):
    ''' Plot brightness over chords profile '''

    # get Rsep and gap
    Rsep, gap = get_CMOD_1D_geom(shot,time)
    Rwall = Rsep+gap
    print('Rwall,Rsep,gap:',Rwall, Rsep,gap)

    if ax is None:
        fig,ax = plt.subplots()

    tidx = bdata.time.values.searchsorted(time)-1
    mask = np.nonzero(bdata.values[tidx,:])[0]
    ax.plot(bdata.R.values[mask], bdata.values[tidx,mask], '.-')  #*100-Rwall

    ax.set_ylabel(r'brightness [${:}$]'.format(bdata.units))
    ax.set_xlabel(r'R [cm]')

    return ax

def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

def get_P_ohmic(shot):
    ''' Get Ohmic power

    Translated/adapted from scopes:
    _vsurf =  deriv(smooth1d(\ANALYSIS::EFIT_SSIBRY,2))*$2pi ;
    _ip=abs(\ANALYSIS::EFIT_AEQDSK:CPASMA);
    _li = \analysis::efit_aeqdsk:ali;
    _L = _li*6.28*67.*1.e-9;
    _vi = _L*deriv(smooth1d(_ip,2));
    _poh=_ip*(_vsurf-_vi)/1.e6
    '''
    from omfit_classes.omfit_mds import OMFITmdsValue
    # psi at the edge:
    ssibry_node = OMFITmdsValue(server='CMOD', shot=shot, treename='ANALYSIS',TDI='\\analysis::efit_ssibry')
    time = ssibry_node.dim_of(0)
    ssibry = ssibry_node.data()
    
    # total voltage associated with magnetic flux inside LCFS
    vsurf = np.gradient(smooth(ssibry,5),time) * 2 * np.pi

    # calculated plasma current
    ip_node= OMFITmdsValue(server='CMOD', shot=shot, treename='ANALYSIS',TDI='\\analysis::EFIT_AEQDSK:CPASMA')
    ip = np.abs(ip_node.data())

    # internal inductance
    li = OMFITmdsValue(server='CMOD', shot=shot, treename='ANALYSIS',TDI='\\analysis::EFIT_AEQDSK:ali').data()

    R_cm = 67.0 # value chosen/fixed in scopes
    L = li*2.*np.pi*R_cm*1e-9  # total inductance (nH)
    
    vi = L * np.gradient(smooth(ip,2),time)   # induced voltage
    
    P_oh = ip * (vsurf - vi)/1e6 # P=IV   #MW
    return time, P_oh

    
def get_CMOD_var(var,shot, tmin=None, tmax=None, plot=False):
    ''' Get tree variable for a CMOD shot. If a time window is given, the value averaged over that window is returned,
    or else the time series is given.  See list below for acceptable input variables.
    '''
    from omfit_classes.omfit_mds import OMFITmdsValue

    if var=='Bt':
        node = OMFITmdsValue(server='CMOD', shot=shot, treename='magnetics',TDI='\\magnetics::Bt')
    elif var=='Bp':
        # use Bpolav, average poloidal B field --> see definition in Silvagni NF 2020
        node = OMFITmdsValue(server='CMOD',shot=shot,treename='analysis', TDI='\EFIT_AEQDSK:bpolav')
    elif var=='Ip':
        node = OMFITmdsValue(server='CMOD', shot=shot, treename='magnetics',TDI='\\magnetics::Ip')
    elif var=='nebar':
        node = OMFITmdsValue(server='CMOD', shot=shot, treename='electrons',TDI='\\electrons::top.tci.results:nl_04')
    elif var=='P_RF':
        node = OMFITmdsValue(server='CMOD', shot=shot, treename='RF',TDI='\\RF::RF_power_net')
    elif var=='P_ohmic':
        node = OMFITmdsValue(server='CMOD', shot=shot, treename='RF',TDI='\\RF::RF_power_net')
    elif var=='P_rad':
        node = OMFITmdsValue(server='CMOD', shot=shot, treename='spectroscopy',TDI='\\spectroscopy::top.bolometer:twopi_diode') # kW
    elif var=='p_D2':
        node = OMFITmdsValue(server='CMOD', shot=shot, treename='EDGE',TDI='\\EDGE::TOP.GAS.RATIOMATIC.F_SIDE')  # mTorr
    elif var=='p_E_BOT_MKS':
        node = OMFITmdsValue(server='CMOD', shot=shot, treename='EDGE',TDI='\\EDGE::E_BOT_MKS')  # mTorr   #lower divertor
    elif var=='p_B_BOT_MKS':
        node = OMFITmdsValue(server='CMOD', shot=shot, treename='EDGE',TDI='\\EDGE::B_BOT_MKS')     # mTorr  # lower divertor
    elif var=='p_F_CRYO_MKS':
        node = OMFITmdsValue(server='CMOD', shot=shot, treename='EDGE',TDI='\\EDGE::F_CRYO_MKS')     # mTorr, only post 2006
    elif var=='q95':
        node = OMFITmdsValue(server='CMOD',shot=shot,treename='analysis', TDI='\EFIT_AEQDSK:qpsib')
    elif var=='Wmhd':
        node = OMFITmdsValue(server='CMOD',shot=shot,treename='analysis', TDI='\EFIT_AEQDSK:wplasm')
    elif var=='areao':
        node = OMFITmdsValue(server='CMOD',shot=shot,treename='analysis', TDI='\EFIT_AEQDSK:areao')
    elif var=='betat':
        node = OMFITmdsValue(server='CMOD',shot=shot,treename='analysis', TDI='\EFIT_AEQDSK:betat')
    elif var=='betap':
        node = OMFITmdsValue(server='CMOD',shot=shot,treename='analysis', TDI='\EFIT_AEQDSK:betap')
    elif var=='P_oh':
        t,data = get_P_ohmic(shot)   # accurate routine to estimate Ohmic power
    else:
        raise ValueError('Variable '+var+' was not recognized!')

    if var not in ['P_oh']: 
        data = node.data()
        t = node.dim_of(0)

        if var=='p_E_BOT_MKS' or var=='p_B_BOT_MKS' or var=='p_F_CRYO_MKS':  # anomalies in data storage
            data = data[0,:]
    
    if var=='P_rad':
        # From B.Granetz's matlab scripts: factor of 4.5 from cross-calibration with 2pi_foil during flattop
        # NB: Bob's scripts mention that this is likely not accurate when p_rad (uncalibrated) <= 0.5 MW
        data *= 4.5
        # data from the twopi_diode is output in kW. Change to MW for consistency
        data /= 1e3
        
    if plot:
        plt.figure()
        plt.plot(t,data)
        plt.xlabel('time [s]')
        plt.ylabel(var)

    if tmin is not None and tmax is not None:
        tidx0 = np.argmin(np.abs(t - tmin))
        tidx1 = np.argmin(np.abs(t - tmax))
        return np.mean(data[tidx0:tidx1])
    else:
        return t,data

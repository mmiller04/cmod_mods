### script to gather and prcoess ne and Te from ETS and ASP

import numpy as np
from matplotlib import pyplot as plt
import asp_probes as probes
import thomson as ts
from scipy.optimize import curve_fit
import tanh_fitting as tanh
import fit_2D as fit

class cmoddata:

    def __init__(self,shot):

        self.shot = shot
        self.time_plunge = 1.0


    def ne_Te_data(self):

        ## get Thomson data

        _out = ts.get_thomson_data(self.shot,self.time_plunge)
        p_ne_ETS, p_Te_ETS = _out

        # TS
        min_ETS_X = p_Te_ETS.X.min() if p_Te_ETS.X.min() < p_ne_ETS.X.min() else p_ne_ETS.X.min()
        max_ETS_X = p_Te_ETS.X.max() if p_Te_ETS.X.max() > p_ne_ETS.X.max() else p_ne_ETS.X.max()
        res_ETS_X = np.linspace(min_ETS_X,max_ETS_X,100)

        try: # sometimes get underflow error when errorbars put in 
            _out = tanh.super_fit(p_Te_ETS.X[:,0],p_Te_ETS.y,vals_unc=p_Te_ETS.err_y,x_out=res_ETS_X)
        except:
            _out = tanh.super_fit(p_Te_ETS.X[:,0],p_Te_ETS.y,x_out=res_ETS_X)
        Te_ETS_fit, c_Te = _out

        try:
            _out = tanh.super_fit(p_ne_ETS.X[:,0],p_ne_ETS.y,vals_unc=p_ne_ETS.err_y,x_out=res_ETS_X)
        except:
            _out = tanh.super_fit(p_ne_ETS.X[:,0],p_ne_ETS.y,x_out=res_ETS_X)
        ne_ETS_fit, c_ne = _out
        
        self.Te_ETS_fit = Te_ETS_fit
        self.ne_ETS_fit = ne_ETS_fit
        self.res_ETS_X = res_ETS_X

        # calculate Te at LCFS from 2pt model
        Te_lcfs_eV = fit.Teu_2pt_model(self.shot,p_Te_ETS.t_min,p_Te_ETS.t_max,ne_ETS_fit,Te_ETS_fit,res_ETS_X)
        print('Te_LCFS', Te_lcfs_eV)

        # shift profiles independently
        _out = fit.shift_profs([1],res_ETS_X,Te_ETS_fit[None,:],Te_LCFS=Te_lcfs_eV)
        rho_ETS, xSep_TS = _out
        if xSep_TS == 1:
            print('TS shift failed')

        rho_ne_ETS = p_ne_ETS.X.transpose() + (1 - xSep_TS) 
        rho_Te_ETS = p_Te_ETS.X.transpose() + (1 - xSep_TS) 

        self.xSep_TS = xSep_TS

        ## try to get probe data

        # guess time for probe plunge
        time_plunge = 1.0

        try:
            _out = probes.get_clean_asp_data(self.shot,self.time_plunge)
            rho, rho_unc, ne_prof, ne_unc_prof, Te_prof, Te_unc_prof, a, b, ax = _out
            print('Probes fetched')
    
            # try to fit probes for shift

            # remove bad ASP_Te values (usually happens for rho > 1.02)
            Te_cut = 1.02
            probe_fit = False # if False, probes will not be fit (and not be shifted)

            while not probe_fit:
                rho = rho[np.where(rho<Te_cut)]
                rho_unc = rho_unc[np.where(rho<Te_cut)]
                Te_prof = Te_prof[np.where(rho<Te_cut)]
                Te_unc_prof = Te_unc_prof[np.where(rho<Te_cut)]
                ne_prof = ne_prof[np.where(rho<Te_cut)]
                ne_unc_prof = ne_unc_prof[np.where(rho<Te_cut)]
       
                # try to fit probes for shift
                try_again = 'n' # exits while loop if not set to 'y'
                try:
                    _out = fit_probes(rho,Te_prof,ne_prof,plot=True)
                    rho_ASP_fit,Te_ASP_fit,ne_ASP_fit = _out
                    self.rho_ASP_fit,self.Te_ASP_fit,self.ne_ASP_fit = rho_ASP_fit,Te_ASP_fit,ne_ASP_fit
                    print('Probes fit successful')

                    probe_fit_good = input('Probe fit good? (y/n) ')

                    if probe_fit_good == 'y':
                        probe_fit = True
                    else:
                        try_again = input('Try again? (y/n) ')

                except:
                    print('Probe fit failed')
                    try_again = input('Try again? (y/n) ')
                    
                if try_again == 'n':
                    probe_fit = True
                else:
                    Te_cut -= 0.005
                    if Te_cut <= np.min(rho):
                        print('Te_cut too small')
                        probe_fit = True # exit while loop without fit
                        self.rho_ASP_fit,self.Te_ASP_fit,self.ne_ASP_fit = np.array([]),np.array([]),np.array([])     

            # try to shift probes
            _out = fit.shift_profs([1],rho,Te_ASP_fit[None,:]*1e-3,Te_LCFS=Te_lcfs_eV)
            rho_ASP, xSep_SP = _out
            if xSep_SP == 1:
                print('Probe shift failed')
            self.xSep_SP = xSep_SP

            plot_probes(rho,Te_prof,Te_ASP_fit,ne_prof,ne_ASP_fit) # change to plot also TS
            probes_good = input('Keep probes? (y/n) ')

            if probes_good == 'n':
                rho_ASP,rho_unc,ne_prof,ne_unc_prof,Te_prof,Te_unc_prof = np.array([]),np.array([]),np.array([]),np.array([]),np.array([]),np.array([])
                self.rho_ASP_fit,self.Te_ASP_fit,self.ne_ASP_fit = np.array([]),np.array([]),np.array([])
                self.xSep_SP = 0

        except: 
            print('Probe fetch failed')
            rho_ASP,rho_unc,ne_prof,ne_unc_prof,Te_prof,Te_unc_prof = np.array([]),np.array([]),np.array([]),np.array([]),np.array([]),np.array([])
            self.rho_ASP_fit,self.Te_ASP_fit,self.ne_ASP_fit = np.array([]),np.array([]),np.array([])
            self.xSep_SP = 0

        # concatenate shifted profiles
        rho_ne_combined = np.hstack((rho_ne_ETS[0],rho_ASP))
        rho_ne_err_combined = np.hstack((p_ne_ETS.err_X[:,0],rho_unc))
        rho_Te_combined = np.hstack((rho_Te_ETS[0],rho_ASP))
        rho_Te_err_combined = np.hstack((p_Te_ETS.err_X[:,0],rho_unc))
        ne_combined = np.hstack((p_ne_ETS.y*1e20,ne_prof))
        ne_err_combined = np.hstack((p_ne_ETS.err_y*1e20,ne_unc_prof))
        Te_combined = np.hstack((p_Te_ETS.y*1e3,Te_prof))
        Te_err_combined = np.hstack((p_Te_ETS.err_y*1e3,Te_unc_prof))

        # do this only if one time slice
#        rho_ne_combined = rho_ne_combined[0]
        ne_sorted_inds = np.argsort(rho_ne_combined)

        self.rho_ne = rho_ne_combined[ne_sorted_inds] # poloidal flux (rhop = sqrt(psi_norm))
        self.rho_ne_err = rho_ne_err_combined[ne_sorted_inds]
        self.psin_ne = self.rho_ne**2
        psin_err0 = (self.rho_ne-self.rho_ne_err/2)**2
        psin_err1 = (self.rho_ne+self.rho_ne_err/2)**2
        self.psin_ne_err = psin_err1-psin_err0
        self.ne_data = ne_combined[ne_sorted_inds]
        self.ne_err = ne_err_combined[ne_sorted_inds]

        # do this only if one time slice
#        rho_Te_combined = rho_Te_combined[0]
        Te_sorted_inds = np.argsort(rho_Te_combined)
        
        self.rho_Te = rho_Te_combined[Te_sorted_inds] # poloidal flux (rhop = sqrt(psi_norm))
        self.rho_Te_err = rho_Te_err_combined[Te_sorted_inds]
        self.psin_Te = self.rho_Te**2
        psin_err0 = (self.rho_Te-self.rho_Te_err/2)**2
        psin_err1 = (self.rho_Te+self.rho_Te_err/2)**2
        self.psin_Te_err = psin_err1-psin_err0
        self.Te_data = Te_combined[Te_sorted_inds]
        self.Te_err = Te_err_combined[Te_sorted_inds]

        # get indices corresponding to ETS and ASP

        num_ne_TS = len(p_ne_ETS.y)
        self.ne_TS_inds = np.where(ne_sorted_inds < num_ne_TS)[0]
        self.ne_SP_inds = np.where(ne_sorted_inds >= num_ne_TS)[0]

        num_Te_TS = len(p_Te_ETS.y)
        self.Te_TS_inds = np.where(Te_sorted_inds < num_Te_TS)[0]
        self.Te_SP_inds = np.where(Te_sorted_inds >= num_Te_TS)[0]

        # store also in terms of big R

        from lyman_data import get_geqdsk_cmod
        import aurora

        geqdsk = get_geqdsk_cmod(self.shot,self.time_plunge*1e3)
        self.Rmid_ne = aurora.rad_coord_transform(self.rho_ne,'rhop','Rmid',geqdsk)
        self.Rmid_ne_err = aurora.rad_coord_transform(self.rho_ne_err,'rhop','Rmid',geqdsk)
        self.Rmid_Te = aurora.rad_coord_transform(self.rho_Te,'rhop','Rmid',geqdsk)
        self.Rmid_Te_err = aurora.rad_coord_transform(self.rho_Te_err,'rhop','Rmid',geqdsk)

        self.Rmid_LCFS = aurora.rad_coord_transform(1,'rhop','Rmid',geqdsk)


        # store time of TS: +/- 0.2 time of plunge
        self.TS_tmin = p_ne_ETS.t_min
        self.TS_tmax = p_ne_ETS.t_max

        return None
        
    
    def ne_Te_fits(self):

        # fit profiles using tanh
        _out = tanh.super_fit(self.psin_Te,self.Te_data)
        self.res_fit_Te, self.c_Te = _out
        _out = tanh.super_fit(self.psin_ne,self.ne_data/1e20)
        self.res_fit_ne, self.c_ne = _out
        self.res_fit_ne = self.res_fit_ne*1e20

        return None

    def plot_ne_Te(self):

        fig,ax = plt.subplots(2,sharex=True)
        ax[0].errorbar(self.psin_Te,self.Te_data,self.Te_err,fmt='.')
        ax[0].plot(self.psin_Te,self.res_fit_Te,'--')
        ax[1].errorbar(self.psin_ne,self.ne_data,self.ne_err,fmt='.')
        ax[1].plot(self.psin_ne,self.res_fit_ne,'--')
        plt.show()

        fig,ax = plt.subplots(2,sharex=True)
        ax[0].plot(self.rho_Te[self.Te_SP_inds]-(1-self.xSep_SP),self.Te_data[self.Te_SP_inds],'o')
        ax[0].plot(self.rho_ASP_fit,self.Te_ASP_fit,'--')
        ax[0].plot(self.rho_Te[self.Te_TS_inds]-(1-self.xSep_TS),self.Te_data[self.Te_TS_inds],'o')
        ax[0].plot(self.res_ETS_X,self.Te_ETS_fit*1e3,'--')

        ax[1].plot(self.rho_ne[self.ne_SP_inds]-(1-self.xSep_SP),self.ne_data[self.ne_SP_inds],'o')
        ax[1].plot(self.rho_ASP_fit,self.ne_ASP_fit,'--')
        ax[1].plot(self.rho_ne[self.ne_TS_inds]-(1-self.xSep_TS),self.ne_data[self.ne_TS_inds],'o')
        ax[1].plot(self.res_ETS_X,self.ne_ETS_fit*1e20,'--')
        plt.show()

        return None



def fit_probes(rho,Te_prof,ne_prof,plot=False):
    
    # SP
    def probe_func(x,a,k,b):
        return a*np.exp(-k*x)+b

    _out = curve_fit(probe_func,rho-1,Te_prof)
    popt_Te,pcov_Te = _out
    Te_ASP_fit = probe_func(rho-1,*popt_Te)

    _out = curve_fit(probe_func,rho-1,ne_prof/1e19)
    popt_ne,pcov_ne = _out
    ne_ASP_fit = probe_func(rho-1,*popt_ne)*1e19

    plot_probes(rho,Te_prof,Te_ASP_fit,ne_prof,ne_ASP_fit)

    return rho,Te_ASP_fit,ne_ASP_fit


def plot_probes(rho,Te,Te_fit,ne,ne_fit):

    fig,ax = plt.subplots(2,sharex=True)
    ax[0].plot(rho,Te,'o')
    ax[0].plot(rho,Te_fit,'--')
    ax[1].plot(rho,ne,'o')
    ax[1].plot(rho,ne_fit,'--')
    plt.show()

    return None





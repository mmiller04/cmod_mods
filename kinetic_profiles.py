### script to gather and prcoess ne and Te from ETS and ASP

import numpy as np
import asp_probes as probes
import tanh_fitting as tanh
import fit_2D as fit

class cmoddata:

    def __init__(self,shot):

        self.shot = shot
        self.time_plunge = 1.0


    def ne_Te_data(self):

        # guess time for probe plunge
        time_plunge = 1.0

        _out = probes.get_clean_asp_data(self.shot,self.time_plunge)
        rho, rho_unc, ne_prof, ne_unc_prof, Te_prof, Te_unc_prof, p_ne_ETS, p_Te_ETS, ax = _out

        # remove bad ASP_Te values (usually happens for rho > 1.02)
        rho = rho[np.where(rho<1.02)]
        rho_unc = rho[np.where(rho<1.02)]
        Te_prof = Te_prof[np.where(rho<1.02)]
        Te_unc_prof = Te_unc_prof[np.where(rho<1.02)]
        ne_prof = ne_prof[np.where(rho<1.02)]
        ne_unc_prof = ne_unc_prof[np.where(rho<1.02)]

        # calculate Te at LCFS from 2pt model
        Te_lcfs_eV = fit.Teu_2pt_model(self.shot,p_Te_ETS.t_min,p_Te_ETS.t_max,p_ne_ETS.y,p_Te_ETS.y,p_Te_ETS.X[:,0])

        # shift profiles independently
        _out = fit.shift_profs([1],p_Te_ETS.X[:,0],p_Te_ETS.y[None,:],Te_LCFS=Te_lcfs_eV)
        rho_Te_ETS, xSep = _out
        rho_ne_ETS = p_ne_ETS.X[:,0] + (1 - xSep)

        _out = fit.shift_profs([1],rho,Te_prof[None,:]*1e-3,Te_LCFS=Te_lcfs_eV)
        rho_ASP, xSep = _out

        # concatenate shifted profiles
        rho_ne_combined = np.hstack((rho_ne_ETS[0],rho_ASP[0]))
        rho_ne_err_combined = np.hstack((p_ne_ETS.err_X[:,0],rho_unc))
        rho_Te_combined = np.hstack((rho_Te_ETS[0],rho_ASP[0]))
        rho_Te_err_combined = np.hstack((p_Te_ETS.err_X[:,0],rho_unc))
        ne_combined = np.hstack((p_ne_ETS.y*1e20,ne_prof))
        ne_err_combined = np.hstack((p_ne_ETS.err_y*1e20,ne_unc_prof))
        Te_combined = np.hstack((p_Te_ETS.y*1e3,Te_prof))
        Te_err_combined = np.hstack((p_Te_ETS.err_y*1e20,ne_unc_prof))


        ne_sorted_inds = np.argsort(rho_ne_combined)

        self.rho_ne = rho_ne_combined[ne_sorted_inds] # poloidal flux (rhop = sqrt(psi_norm))
        self.rho_ne_err = rho_ne_err_combined[ne_sorted_inds]
        self.psin_ne = self.rho_ne**2
        psin_err0 = (self.rho_ne-self.rho_ne_err/2)**2
        psin_err1 = (self.rho_ne+self.rho_ne_err/2)**2
        self.psin_ne_err = psin_err1-psin_err0
        self.ne_data = ne_combined[ne_sorted_inds]
        self.ne_err = ne_err_combined[ne_sorted_inds]

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
        self.ne_TS_inds = np.argsort(ne_sorted_inds[:num_ne_TS])
        self.ne_SP_inds = np.argsort(ne_sorted_inds[num_ne_TS:])

        num_Te_TS = len(p_Te_ETS.y)
        self.Te_TS_inds = np.argsort(Te_sorted_inds[:num_Te_TS])
        self.Te_SP_inds = np.argsort(Te_sorted_inds[num_Te_TS:])

        # store also in terms of big R

        from lyman_data import get_geqdsk_cmod
        import aurora

        geqdsk = get_geqdsk_cmod(self.shot,self.time_plunge*1e3)
        self.Rmid_ne = aurora.rad_coord_transform(self.rho_ne,'rhop','Rmid',geqdsk)
        self.Rmid_ne_err = aurora.rad_coord_transform(self.rho_ne_err,'rhop','Rmid',geqdsk)
        self.Rmid_Te = aurora.rad_coord_transform(self.rho_Te,'rhop','Rmid',geqdsk)
        self.Rmid_Te_err = aurora.rad_coord_transform(self.rho_Te_err,'rhop','Rmid',geqdsk)

        # store time of TS: +/- 0.2 time of plunge
        self.TS_tmin = p_ne_ETS.t_min
        self.TS_tmax = p_ne_ETS.t_max

        return None


    def ne_Te_fits(self):

        # fit profiles using tanh
        _out = tanh.super_fit(self.psin_Te,self.Te_data)
        self.res_fit_Te, self.c_Te = _out
        _out = tanh.super_fit(self.psin_ne,self.ne_data)
        self.res_fit_ne, self.c_ne = _out

        return None






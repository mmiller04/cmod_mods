
import sys
sys.path.append('/home/sciortino/atomAI')
from utils import *
from adas_atomic_rates import *

sys.path.append('/home/millerma/Aurora')
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

        logne22, logTe22,S2 = scb_ion2['scd']

        self.ionFunc= scipy.interpolate.interp2d(logne22,logTe22,S2[0],kind = 'cubic')
        


    
    def ion(self,ne,te,neErr, teErr):
        """
        Inputs:

        ne,- cm^3-float
        te-in eV

        Output:
        ionization cross section - m^3/s
        """
        #print(self.ionFunc(np.log10(ne), np.log10(te))[0])

        ion = 10**(self.ionFunc(np.log10(ne), np.log10(te))-6) #-6 converts from cm^3/s to m^3/s

        nete = genPair(ne,te,neErr,teErr)


        errIonF = [10**(self.ionFunc(np.log10(x[0]),np.log10(x[1]))-6)-ion for x in nete]
        errIonF = (np.amax(errIonF)+np.amin(errIonF))/2


        err = ion*np.log(10)*errIonF
        return ion,err
    

    def recom(self,ne,te):

        return 10**self.recomFunc(np.log10(ne), np.log10(te))

    def excit(self,ne,te):

        return 10**self.excFunc(np.log10(ne), np.log10(te))

    """Inputs:
    emiss - in W/cm^3-float
    ne,ni - cm^3-flot
    te-in eV

    Output:
    Ionization Rate - float - in m^-3s^-1
    """
    def calcIonRate(self,emiss,ne,ni,te,emissErr=0.0,neErr=0.0,niErr=0.0,teErr=0.0):
        
        #for some reason errors come in as weird mds objects only some of the time
        neErr = float(neErr)
        teErr = float(teErr)
        niErr = float(niErr)


        n0,n0Err = self.calcNDens(emiss,ne,ni,te,emissErr,neErr,niErr,teErr)
        #print('n0: '+str(n0))
        #print('n0 :'+str(n0Err/n0))

        ion,ionErr = self.ion(ne,te,neErr,teErr)
        #print(ion)
        #print(ionErr)

        err = np.sqrt((n0Err/n0)**2+(neErr/ne)**2+(ionErr/ion)**2)
        #print('ion :'+str(ionErr/ion))

        #print(err)

        ion = n0*ne*10**12*ion

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
    def calcNDens(self,emiss,ne,ni,te,emissErr=0.0,neErr=0.0,niErr=0.0,teErr=0.0):

        #for some reason errors come in as weird mds objects only some of the time
        neErr = float(neErr)
        teErr = float(teErr)
        niErr = float(niErr)

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
        nete = genPair(ne,te,neErr,teErr)

        #We will take the min and max difference from value and calculate the mean
        errExC  = [self.excit(x[0],x[1])-excCoef for x in nete]
        errExC = (np.amax(errExC)+np.amin(errExC))/2

        errReC  = [self.recom(x[0],x[1])-recomCoef for x in nete]
        errReC =(np.amax(errReC)+np.amin(errReC))/2



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
    return np.column_stack((np.concatenate((np.tile(x+xErr,2),np.tile(x-xErr,2))),np.tile([y+yErr,y-yErr],2)))




if __name__ == "__main__":
    
    """
    plt.ioff()
    atom_data = load_ion_data(imp = 'H')
    #plt.show()


    ne_mean = 1e20


    logne2, logTe2,S = atom_data['scd']  # ionization
    
    nelen = len(logne2)


    logne2 = np.tile(logne2,len(logTe2))
    print(logne2)
    logTe2 = np.repeat(logTe2,nelen)

    S = np.ndarray.flatten(S)

    print(logTe2)

    print(np.ndarray.flatten(S))

    print(logTe2.shape)
    print(logne2.shape)
    print(S.shape)

    
    
    #S = interp1d(logne2, S, kind='cubic', bounds_error=False)(np.log10(ne_mean)-6)
    f = scipy.interpolate.interp2d(logne2,logTe2,S[0])



    print(10**f(np.log10(2*10**14),np.log10(50))*2*10**16*2*10**20)
    
    print('right factor: '+str(10**f(np.log10(2*10**14),np.log10(50))))
    

    """
    
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

    import pdb
    pdb.set_trace()

    


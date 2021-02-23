### Adapted for CMOD

import numpy as np 
import LyaDictFunc as pde
#import emissCompare as ec
import matplotlib.pyplot as plt

import sys
sys.path.insert(1, '/fusion/projects/diagnostics/llama/PythonDataTools')
import ADAShelper as ADAS
import gadata



def CombineDict(dList,saveName):
	out = {}


	for i in range(len(dList)):
		key = str(dList[i,0])+'_'+dList[i,1]

		print('\n')
		print('\n')
		print(key)


		ELMDict = pde.load_dict('Dicts/'+key)
		out[key] = ELMDict[key]

	pde.save_dict(out, saveName)

	return 0

def modifyDict(dictList):
	ADASfile = ADAS.ADAS(1215.2)

	for i in range(len(dictList)):

		cItem = np.array([shotList[i]])
		key = str(cItem[0,0])+'_'+cItem[0,1]
		print('\n')
		print(key)
		cDict = pde.load_dict('Dicts/'+key)[key]

		import pdb
		pdb.set_trace()



		filename = cItem[0,1]
		shotN = int(cItem[0,0])

		#fluxRS,BC,r,fluxErr = pde.FluxProf(cDict)
		
		ttrz,err = pde.mapTT2LLAMA(cDict)
		cDict['TS_lya_rz'] = ttrz
		cDict['err_TS_lya_rz'] = err


		pde.save_dict({key:cDict},'Dicts/'+key)
		

		

		

	return 0 


def main():

	ADASfile = ADAS.ADAS(1215.2)

	
	
	# shotList = np.array([[180907,'P3500_AMRNOV'],[180913,'P3800_AMRNOV'],\
	# 	[180912,'P3700_AMRNOV'],[180914,'P3400_AMRNOV'],\
	# 	[180915,'P3400_AMRNOV'],[180916,'P3400_AMRNOV'],[180913,'P2700_AMRNOV'],[180914,'P2600_AMRNOV']])


	# shotList = np.array([[184313,'P3750_TFDay2'],[184313,'P3100_TFDay2'],[184314,'P2900_TFDay2'],\
	# 	[184314,'P3700_TFDay2'],[184315,'P3100_TFDay2'],[184315,'P3800_TFDay2'],[184368,'P3800_TFDay2'],\
	# 	[184368,'P3100_TFDay2']])
	# shotList = np.array([[184315,'P3800_TFDay2'],[184368,'P3800_TFDay2'],\
	# 	[184368,'P3100_TFDay2']])

	# shotList = np.array([[184370,'P3000_TFDAY2'],[184313,'P3100_TFDay2'],\
	# 	[184374,'P3050_TFDAY2'],[184375,'P3250_TFDAY2']])

	# shotList = np.array([[183031,'P3000_REFSEP'],[184366,'P3000_REFSEP']])

	# shotList = np.array([[183031,'P3000_REFSEP']])

	shotList = np.array([[1070710003]])

	for i in range(len(shotList)):

		cItem = np.array([shotList[i]])
		print(cItem)
		key = str(cItem[0,0])+'_'+cItem[0,1]
		dDict = pde.shotDict(cItem,ADASfile)


		#ec.tomoCheck(dDict)


		pde.save_dict(dDict,'Dicts/'+key)


	return 0

if __name__ == "__main__":

	
	main()
	"""
	shotList = np.array([[184313,'P1950_TFDay2'],[184313,'P3100_TFDay2'],[184314,'P2900_TFDay2'],\
		[184315,'P3100_TFDay2'],[184368,'P3100_TFDay2']])
	shotList = np.array([[184313,'P1950_TFDay2'],[184313,'P3100_TFDay2'],\
		[184315,'P3100_TFDay2'],[184314,'P3700_TFDay2']])
	
	shotList = np.array([[184313,'P1950_TFDay2'],[184313,'P3750_TFDay2'],\
		[184314,'P3700_TFDay2'],[184315,'P3800_TFDay2']])
	
	shotList = np.array([[180907,'P3500_AMRNOV'],[184307,'P2600_TFDay2'],[184309,'P2900_TFDay2']])
	
	"""
	# shotList = np.array([[183031,'P3000_REFSEP'],[184366,'P3000_REFSEP']])

	#CombineDict(shotList,'RefComp')

	

import csv
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn import preprocessing

data = pd.read_csv(filepath_or_buffer='../../Covid High Flow300 Test.csv', header=1, dtype=str)
#print(data.dtypes)
#print(list(data.columns.values))

yesNoList = ['DM','HTN','Prior MI (Stent CABG)','CHF (HFrEF)','CHF (HFpEF)',
'OSA','Sick Contact hospitalized?','Chronic Lung Disease (COPD etc)','Immunne suppressed / Transplant',
'Liver Cirrhosis','ESRD on HD', 'Active Smoker / Vaping', 'Fomer smoker / vaper', 'Active pregnancy',
'CPAP / BiPAP','Intubation','Decadron','Remdesivir','Convalesent Plasma', 'CPAP / BiPAP.1', 'Intubation.1',
'CPAP / BiPAP.2','Intubation.2','Dead in 30 days','Crash > 4 days', 'Zero to 60 L oxygen.2']
convertDict = {'Age':float, 'Zip code':float, 'BMI':float, 'Weight (kg)':float, 'SBP':float, 'DBP':float, 'P':float, 'R':float,
               'Rom Air Sat':float, 'Temp (F)':float, 'Zero to 60 L oxygen':float,
               'Oxyen sat on highest oxygen':float, 'HGBA1c (+/1 100 days)':float, 'WBC':float,
               'HGB':float, 'Creat':float, 'ALT':float, 'TBILI':float, 'CRP': float, 'DDimer': float,
	       'Ferritin':float, 'LDH':float, 'SBP.1':float, 'DBP.1': float, 'P.1':float, 'R.1':float,
               'Temp (F).1':float, 'Zero to 60 L oxygen.1':float, 'Oxyen sat on highest oxygen.1':float,
               'WBC.1':float, 'HGB.1':float, 'Creat.1':float, 'ALT.1':float, 'TBILI.1':float,
               'CRP.1':float, 'DDimer.1':float, 'Ferritin.1':float, 'LDH.1':float, 'DM':bool, 'HTN':bool,
	       'Prior MI (Stent CABG)':bool, 'Chronic Lung Disease (COPD etc)':bool, 'ESRD on HD':bool,
	       'CPAP / BiPAP':bool, 'Intubation':bool, 'Decadron':bool, 'Remdesivir':bool, 'Convalesent Plasma':bool,
	       'CPAP / BiPAP.1':bool, 'Intubation.1':bool
               }

for i in yesNoList:
    data[i] = data[i].map({'yes':True, 'no':False, 'Yes':True, 'No':False, 'yes ':True, 'no ':False})
data['Sex'] = data['Sex'].map({'M': False, 'F': True})
data = data.astype(convertDict)

inputColumns = ['Age','Sex','Zip code','DM','HTN','Prior MI (Stent CABG)','CHF (HFrEF)','CHF (HFpEF)',
                'OSA', 'Sick Contact hospitalized?', 'Chronic Lung Disease (COPD etc)',
                'Immunne suppressed / Transplant', 'Liver Cirrhosis','ESRD on HD','Active Smoker / Vaping',
                'Fomer smoker / vaper','Active pregnancy', 'BMI', 'Weight (kg)','SBP','DBP','P', 'R',
                'Rom Air Sat', 'Temp (F)', 'Zero to 60 L oxygen', 'Oxyen sat on highest oxygen',
                'CPAP / BiPAP', 'Intubation', 'HGBA1c (+/1 100 days)','WBC','HGB','Creat','ALT','TBILI',
                'CRP','DDimer', 'Ferritin', 'LDH', 'Decadron', 'Remdesivir', 'Convalesent Plasma', 'SBP.1',
                'DBP.1', 'P.1', 'R.1', 'Temp (F).1','Zero to 60 L oxygen.1', 'Oxyen sat on highest oxygen.1',
		'CPAP / BiPAP.1', 'Intubation.1', 'WBC.1', 'HGB.1', 'Creat.1', 'ALT.1', 'TBILI.1', 'CRP.1',
		'DDimer.1', 'Ferritin.1', 'LDH.1']
'''

inputColumns = ['Age','Sex','Zip code','DM','HTN','Prior MI (Stent CABG)','CHF (HFrEF)','CHF (HFpEF)',
                'OSA', 'Sick Contact hospitalized?', 'Chronic Lung Disease (COPD etc)',
                'Immunne suppressed / Transplant', 'Liver Cirrhosis','ESRD on HD','Active Smoker / Vaping',
                'Fomer smoker / vaper','Active pregnancy', 'BMI', 'Weight (kg)','SBP','DBP','P', 'R',
                'Rom Air Sat', 'Temp (F)', 'Zero to 60 L oxygen', 'Oxyen sat on highest oxygen',
                'CPAP / BiPAP', 'Intubation', 'HGBA1c (+/1 100 days)','WBC','HGB','Creat','ALT','TBILI',
                'CRP','DDimer', 'Ferritin', 'LDH', 'Decadron', 'Remdesivir', 'Convalesent Plasma']
'''
#outputColumn = 'Oxygen>30'
outputColumn = 'Zero to 60 L oxygen.2'



inputData = data[inputColumns]
outputData = data[outputColumn]

externalData = []

externalData.append(input("Enter Age: "))
sex = input("Enter Sex (M or F): ")
if sex == "F" or sex == "f": externalData.append(True)
elif sex == "M" or sex == "m": externalData.append(False)
else: sys.exit("invalid input")
externalData.append(input("Enter Zip Code: "))
externalData.append(str(input("Enter DM: ")))
externalData.append(str(input("Enter HTN: ")))
externalData.append(str(input("Enter Prior MI (Stent, CABG): ")))
externalData.append(str(input("Enter CHF (HFrEF,< 45): ")))
externalData.append(str(input("Enter CHF (HFpEF,>= 45): ")))
externalData.append(str(input("Enter OSA: ")))
externalData.append(str(input("Enter Contact Hospitalized: ")))
externalData.append(str(input("Enter Chronic Lung Disease: ")))
externalData.append(str(input("Enter Immune Suppressed/ Transplant: ")))
externalData.append(str(input("Enter Liver Cirrhosis: ")))
externalData.append(str(input("Enter ESRD on HD: ")))
externalData.append(str(input("Enter Active Smoker/Vaper: ")))
externalData.append(str(input("Enter Former Smoker/Vaper: ")))
externalData.append(str(input("Enter Active Pregnancy: ")))
externalData.append(input("Enter BMI: "))
externalData.append((input("Enter Weight (kg): ")))
externalData.append((input("Enter SBP: ")))
externalData.append((input("Enter DBP: ")))
externalData.append((input("Enter P: ")))
externalData.append((input("Enter R: ")))
externalData.append((input("Enter Room Air Sat: ")))
externalData.append((input("Enter Temp (f): ")))
externalData.append((input("Enter 0-60L Oxygen: ")))
externalData.append((input("Enter Oxygen Sat: ")))
externalData.append(str(input("Enter CPAP/BiPAP: ")))
externalData.append(str(input("Enter Intubation: ")))
externalData.append((input("Enter HGBA1C: ")))
externalData.append((input("Enter WBC: ")))
externalData.append((input("Enter HGB: ")))
externalData.append((input("Enter Creat: ")))
externalData.append((input("Enter ALT: ")))
externalData.append((input("Enter TBILI: ")))
externalData.append((input("Enter CRP: ")))
externalData.append((input("Enter DDIMER: ")))
externalData.append((input("Enter Ferritin: ")))
externalData.append((input("Enter LDH: ")))
externalData.append(str(input("Enter Decadron: ")))
externalData.append(str(input("Enter Remdesivir: ")))
externalData.append(str(input("Enter Convalesent Plasma: ")))
externalData.append((input("Enter SBP 48: ")))
externalData.append((input("Enter DBP 48: ")))
externalData.append((input("Enter P 48: ")))
externalData.append((input("Enter R 48: ")))
externalData.append((input("Enter Temp (f) 48: ")))
externalData.append((input("Enter 0-60L Oxygen 48: ")))
externalData.append((input("Enter Oxygen Sat 48: ")))
externalData.append(str(input("Enter CPAP/BiPAP 48: ")))
externalData.append(str(input("Enter Intubation 48: ")))
externalData.append((input("Enter WBC 48: ")))
externalData.append((input("Enter HGB 48: ")))
externalData.append((input("Enter Creat 48: ")))
externalData.append((input("Enter ALT 48: ")))
externalData.append((input("Enter TBILI 48: ")))
externalData.append((input("Enter CRP 48: ")))
externalData.append((input("Enter DDIMER 48: ")))
externalData.append((input("Enter Ferritin 48: ")))
externalData.append((input("Enter LDH 48: ")))

print(externalData)

def strConvert(string):
	if isinstance(string, str):
		if string == "yes" or string == "Yes" or string == "true" or string == "True":
			return True
		elif string == "no" or string == "No" or string == "false" or string == "False":
			return False
		elif string == "":
			return
		else:
			sys.exit("invalid input")
	elif isinstance(string, float):
		if str(string)[-2:] == ".0":
			return int(string)
		else:
			return float(string)
	else:
		return string
externalData = map(strConvert, externalData)

manualData = [46,False,95822,False,False,False,False,False,False,False,False,False,False,False,False,False,False,28.87,102,139,71,117,24,75,103.3,6,91,False,False,6.2,14.4,10.9,0.86,37,0.8,14.5,0.57,652,375,True,True,False,115,65,78,28,99.9,60,97,True,False,12.5,10.5,0.62,31,0.4,17,0.97,1111,684]
print(manualData)

print((externalData))
	
df2 = pd.DataFrame(np.array([manualData]),)
dat = xgb.DMatrix(df2)

bst2 = xgb.Booster()  # init model

bst2.load_model('../covid 48 v1 models/covid 48 v1 Seed2.model')  
labels_predict2 = bst2.predict(dat)


print("Model #2: " + str(labels_predict2))





bst8 = xgb.Booster()  # init model

bst8.load_model('../covid 48 v1 models/covid 48 v1 Seed8.model')  

labels_predict8 = bst8.predict(dat)

print("Model #8: " + str(labels_predict8))





bst24 = xgb.Booster()  # init model

bst24.load_model('../covid 48 v1 models/covid 48 v1 Seed24.model')  

labels_predict24 = bst24.predict(dat)


print("Model #24: " + str(labels_predict24))





bst64 = xgb.Booster()  # init model

bst64.load_model('../covid 48 v1 models/covid 48 v1 Seed64.model')  

labels_predict64 = bst64.predict(dat)


print("Model #64: " + str(labels_predict64))





bst256 = xgb.Booster()  # init model

bst256.load_model('../covid 48 v1 models/covid 48 v1 Seed256.model')  

labels_predict256 = bst256.predict(dat)


print("Model #256: " + str(labels_predict256))

oxygen = []

if labels_predict256[0].round() + labels_predict8[0].round() + labels_predict24[0].round() + labels_predict64[0].round() + labels_predict256[0].round() > 2:
	oxygen.append(1)
else:
	oxygen.append(0)
print(oxygen)

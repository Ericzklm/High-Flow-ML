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

data = pd.read_csv(filepath_or_buffer='../../Covid High Flow300 Fix.csv', header=1, dtype=str)
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

#Load in our models from file

casesToTest = 302 #specify how many cases we want to test
dat = xgb.DMatrix(inputData.head(casesToTest))
labels = outputData.head(casesToTest)



bst2 = xgb.Booster()  # init model

bst2.load_model('../covid 48 v1 models/covid 48 v1 Seed2.model')  
labels_predict2 = bst2.predict(dat)


#print(labels_predict2)





bst8 = xgb.Booster()  # init model

bst8.load_model('../covid 48 v1 models/covid 48 v1 Seed8.model')  

labels_predict8 = bst8.predict(dat)


#print(labels_predict8)





bst24 = xgb.Booster()  # init model

bst24.load_model('../covid 48 v1 models/covid 48 v1 Seed24.model')  

labels_predict24 = bst24.predict(dat)


#print(labels_predict24)





bst64 = xgb.Booster()  # init model

bst64.load_model('../covid 48 v1 models/covid 48 v1 Seed64.model')  

labels_predict64 = bst64.predict(dat)


#print(labels_predict64)





bst256 = xgb.Booster()  # init model

bst256.load_model('../covid 48 v1 models/covid 48 v1 Seed256.model')  

labels_predict256 = bst256.predict(dat)


#print(labels_predict256)





#specify which patients (which rows) to check the results of

patientNum = np.arange(0, casesToTest, 1).tolist()
oxygen = []
output = []

for i in range(len(labels_predict2)):
	if patientNum.count(i) > 0:
		if labels_predict256[i].round() + labels_predict8[i].round() + labels_predict24[i].round() + labels_predict64[i].round() + labels_predict256[i].round() > 2:
			oxygen.append(1)
		else:
			oxygen.append(0)
		output.append(outputData[i])

print("Accuracy: " + str(accuracy_score(output, oxygen)) + "\n")
print(classification_report(output, oxygen))

truePositive = 0
trueNegative = 0
falsePositive = 0
falseNegative = 0

for i in range(len(output)):
  if output[i] == True and oxygen[i] == 1:
    truePositive = truePositive + 1
  elif output[i] == False and oxygen[i] == 0:
    trueNegative = trueNegative + 1
  elif output[i] == True and oxygen[i] == 0:
    falseNegative = falseNegative + 1
  elif output[i] == False and oxygen[i] == 1:
    falsePositive = falsePositive + 1

print("\n\t\tActual")
print("Predicted\tTrue\tFalse")
print("True\t\t" + str(truePositive) + "\t" + str(falsePositive))
print("False\t\t" + str(falseNegative) + "\t" + str(trueNegative))

print("\nTrue Positives: " + str(truePositive))
print("True Negatives: " + str(trueNegative))
print("False Negatives: " + str(falseNegative))
print("False Positives: " + str(falsePositive))
print("Sensitivity: " + str(truePositive / (truePositive + falseNegative)))
print("Specificity: " + str(trueNegative / (trueNegative + falsePositive)))
print("Positive Predictive Value: " + str(truePositive / (truePositive + falsePositive)))
print("Negative Predictive Value: " + str(falsePositive / (falsePositive + truePositive)))
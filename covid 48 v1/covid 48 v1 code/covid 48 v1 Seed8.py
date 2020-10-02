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
outputColumn = 'Zero to 60 L oxygen.2'

inputData = data[inputColumns]
outputData = data[outputColumn]

inputTrain, inputTest, outputTrain, outputTest = train_test_split(inputData, outputData, test_size = .3, random_state = 8)

trainMatrix = xgb.DMatrix(inputTrain, label=outputTrain, feature_names=inputColumns[:60])
testMatrix = xgb.DMatrix(inputTest, label=outputTest, feature_names=inputColumns[:60])
params = {'max_depth':5, 'eta':0.004, 'subsample':1.0, 'min_child_weight':1.0, 'reg_lambda':0.0, 'reg_alpha':0.0, 'objective':'binary:logistic', 'eval_metric': 'error'}
model = xgb.train(params, trainMatrix, 1000, evals=[(testMatrix, "Test")], early_stopping_rounds=200)

param_grid = {'eta':[.3,.25,.2,.15,0.1,.075,0.05,0.01,0.005,0.001], 'max_depth':np.arange(1,10,1).tolist(), 'subsample':np.arange(1,0.1,-0.1).tolist(), 'colsample_bytree':np.arange(1,0.1,-0.1).tolist(), 'min_child_weight':np.arange(1,100,5).tolist()}
#param_grid = {'eta':[.3,.25,.2,.15,0.1], 'max_depth':np.arange(1,10,5).tolist(), 'subsample':np.arange(1,0.1,-0.5).tolist(), 'colsample_bytree':np.arange(1,0.1,-0.5).tolist(), 'min_child_weight':np.arange(1,100,50).tolist()}
#Save the best results
bestParams = {}
lowestError = 2048

for max_depth in param_grid['max_depth']:
    for eta in param_grid['eta']:
        for subsample in param_grid['subsample']:
            for colsample_bytree in param_grid['colsample_bytree']:
                for min_child_weight in param_grid['min_child_weight']:
                    cvResults = xgb.cv({'max_depth':max_depth, 'eta':eta, 'subsample':subsample, 'colsample_bytree':colsample_bytree, 'min_child_weight':min_child_weight, 'objective':'binary:logistic', 'eval_metric': 'error'}, trainMatrix, num_boost_round=600, seed=2, nfold=5, early_stopping_rounds=125)
                    if abs(cvResults['test-{}-mean'.format('error')]).min() < lowestError:
                        lowestError = abs(cvResults['test-{}-mean'.format('error')]).min()
                        bestParams = {'max_depth':max_depth, 'eta':eta, 'subsample':subsample, 'colsample_bytree':colsample_bytree, 'min_child_weight':min_child_weight, 'objective':'binary:logistic', 'eval_metric': 'error'}
                    #print(str(abs(cvResults['test-{}-mean'.format('error')]).min()) + ' , ' + str(lowestError))
print(bestParams)
print(lowestError)

model = xgb.train(bestParams, trainMatrix, 5000, evals=[(testMatrix, "Test")], early_stopping_rounds=1000)

model.save_model('../covid 48 v1 models/covid 48 v1 Seed8.model')

outputTrainPredict = model.predict(trainMatrix)
outputTestPredict = model.predict(testMatrix)

print(bestParams)

print("\nSpirit Model:")
print("\nTraining Accuracy: " + str(accuracy_score(outputTrain, outputTrainPredict.round())))
print("Testing Accuracy: " + str(accuracy_score(outputTest, outputTestPredict.round())))

truePositive = 0
trueNegative = 0
falsePositive = 0
falseNegative = 0


for i in range(len(outputTest)):
  if outputTest.values[i] == True and outputTestPredict.round()[i] == 1:
    truePositive = truePositive + 1
  elif outputTest.values[i] == False and outputTestPredict.round()[i] == 0:
    trueNegative = trueNegative + 1
  elif outputTest.values[i] == True and outputTestPredict.round()[i] == 0:
    falseNegative = falseNegative + 1
  elif outputTest.values[i] == False and outputTestPredict.round()[i] == 1:
    falsePositive = falsePositive + 1

print("\n\t\tActual")
print("Predicted\tTrue\tFalse")
print("True\t\t" + str(truePositive) + "\t" + str(falsePositive))
print("False\t\t" + str(falseNegative) + "\t" + str(trueNegative))

print("\nTrue Positives: " + str(truePositive))
print("True Negatives: " + str(trueNegative))
print("False Negatives (Type II error): " + str(falseNegative))
print("False Positives (Type I error): " + str(falsePositive))
print("Sensitivity: " + str(truePositive / (truePositive + falseNegative)))
print("Specificity: " + str(trueNegative / (trueNegative + falsePositive)))
print("Positive Predicted Rate: " + str(truePositive / (truePositive + falsePositive)))
print("Negative Predicted Rate: " + str(trueNegative / (trueNegative + falseNegative)) + "\n")

fig, ax = plt.subplots(figsize=(15,12))
xgb.plot_importance(model, ax=ax)
plt.show()



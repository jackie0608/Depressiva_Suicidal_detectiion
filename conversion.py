import numpy as np
from functools import reduce
import operator
import pandas as pd
import csv
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

import matplotlib.pyplot as plt


def tosentiment(problist):
    sentiment = []
    for p in problist:
        if p >= 0.5:
            sentiment.append(1)
        else:
            sentiment.append(0)
    return sentiment
    
#data Definition
# gptresults
f = open("gptresults", "r")
data = f.read()

#cleaningStep, remove tensor, and all other things
elementsToRemove= ['\n','[tensor(','device=',"'cuda:0')",'[', ']', '"']

cleanData = data
for el in elementsToRemove:
    cleanData = cleanData.replace(el,'')
cleanData = cleanData.replace(' ',',')
 
#convert to numeric using np.matrix
numericData_matrix = np.matrix(cleanData)
numericData_array = np.array(numericData_matrix)
gptmeasure = reduce(operator.concat, numericData_array)
gptmeasure = gptmeasure[0::3]
'''
map(float, gptmeasure)
print(gptmeasure)
print(len(gptmeasure))
print(type(gptmeasure[0]))
'''

# albert/bert/xlnet results
f1 = open("albertresults", "r")
f2 = open("bertresults", "r")
f3 = open("xlnetresults", "r")
dataalbert = f1.read()
databert = f2.read()
dataxlnet = f3.read()

#cleaningStep, remove tensor, and all other things
elementsToRemove= ['\r','\n','[', ']', ' ', '"']


cleanDataalbert = dataalbert
cleanDatabert = databert
cleanDataxlnet = dataxlnet
for el in elementsToRemove:
    cleanDataalbert = cleanDataalbert.replace(el,'')
    cleanDatabert = cleanDatabert.replace(el,'')
    cleanDataxlnet = cleanDataxlnet.replace(el,'')

numericData_matrix_albert = np.matrix(cleanDataalbert)
numericData_array_albert = np.array(numericData_matrix_albert)
albertmeasure = reduce(operator.concat, numericData_array_albert)
map(float, albertmeasure)

numericData_matrix_bert = np.matrix(cleanDatabert)
numericData_array_bert = np.array(numericData_matrix_bert)
bertmeasure = reduce(operator.concat, numericData_array_bert)
map(float, bertmeasure)

numericData_matrix_xlnet = np.matrix(cleanDataxlnet)
numericData_array_xlnet = np.array(numericData_matrix_xlnet)
xlnetmeasure = reduce(operator.concat, numericData_array_xlnet)
map(float, xlnetmeasure)


# t5results
f = open("t5results", "r")
t5data = f.read()

#cleaningStep, remove tensor, and all other things
elementsToRemove= ['\n','[tensor(','device=',"'cuda:0')",'[', ']', '"']

t5cleanData = t5data
for el in elementsToRemove:
    t5cleanData = t5cleanData.replace(el,'')
 
#convert to numeric using np.matrix
numericData_matrix_t5 = np.matrix(t5cleanData)
numericData_array_t5 = np.array(numericData_matrix_t5)
t5measure = reduce(operator.concat, numericData_array_t5)
map(float, t5measure)



# load real label
df = pd.read_csv(open('data/dev.tsv', errors="ignore"), delimiter="\t", quoting=csv.QUOTE_NONE,encoding='windows-1252',error_bad_lines=False)
realmeasure = df["label"].tolist()
map(float, realmeasure)



#gptmeasure = tosentiment(gptmeasure)
#albertmeasure = tosentiment(albertmeasure)
#bertmeasure = tosentiment(bertmeasure)
#xlnetmeasure = tosentiment(xlnetmeasure)

x_values = []
for i in range(len(xlnetmeasure)):
    new = []
    new.append(gptmeasure[i])
    new.append(albertmeasure[i])
    new.append(bertmeasure[i])
    new.append(xlnetmeasure[i])
    new.append(t5measure[i])
    x_values.append(new)
y_values = realmeasure
print(x_values[0])

# we use linear regression as a base!!! ** sometimes misunderstood **
reg = LinearRegression().fit(x_values, y_values)

print("weight of models output:",end = '')
print(reg.coef_)
print("interept:",end = '')
print(reg.intercept_)


TP = 0
FP = 0
TN = 0
FN = 0

for i in range(len(y_values)):
    predictions = reg.predict([x_values[i]])
    predictions = round(float(predictions[0]))
    #print(predictions)

    if predictions == 1 and y_values[i] == 1:
        TP += 1
    elif predictions == 0 and y_values[i] == 1:
        FN += 1
    elif predictions == 0 and y_values[i] == 0:
        TN += 1
    else:
        FP += 1

  
precision = TP/(TP+FP)
recall = TP/(TP+FN)
F1 = 2*(precision*recall)/(precision+recall)  
print("precision: " + str(precision))
print("recall: " + str(recall))
print("F1-score: " + str(F1))
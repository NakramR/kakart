import petitchatbase as pcb
import thebestpredictionsherenotintheotherone as bestpredictions
import predictions as worstpredictions
import numpy as np
import pandas as pd
import time

#pcb.maxuserid = '100'
#pcb.maxuserid = '10000'
pcb.maxuserid = '1000000000'

pcb.debugWithTimer("initializing data")
pcb.initData(pcb.maxuserid)

pcb.debugWithTimer("splitting train/test orders")

train, test = pcb.trainAndTestForValidation()

#print(usersInTest)


## predictions

pcb.debugWithTimer("generating random prediction")
#p1 = bestpredictions.generateRandomPrediction()

pcb.debugWithTimer("generating freq threshold prediction")
p2 = bestpredictions.predictOverFrequencyThreshold(0.3)

pcb.debugWithTimer("generating decision tree prediction")
p3 = worstpredictions.generateDecisionTreePrediction(train, test)

# pcb.debugWithTimer("generating linear regression prediction")
# p4 = worstpredictions.generateLinearRegressionPrediction(train, test)
#
# pcb.debugWithTimer("generating xgboost prediction")
# p5 = worstpredictions.generateXGBoostPrediction(train, test)
#

predictionToSaveFull = p3





pcb.debugWithTimer("creating CSV")
predictionToSaveFull = predictionToSaveFull[predictionToSaveFull['ordered'] == True]
predictionToSaveTestOnly  = predictionToSaveFull[predictionToSaveFull['user_id'].isin(pcb.usersInTest['user_id'].values.tolist())].groupby('user_id')['product_id'].apply(list)
#print(predictionDF)

pcb.debugWithTimer("formatting CSV")
csvToSave = pd.DataFrame(columns={'user_id', 'predictions'})
csvToSave['predictions'] = predictionToSaveTestOnly
csvToSave['user_id'] = predictionToSaveTestOnly.keys()
csvToSave['productsx'] = csvToSave['predictions'].map(lambda x: ' '.join(str(xx) for xx in x))

## add users that have no predictions

emptyUsers = []
for user in pcb.usersInTest['user_id']:
    if user not in predictionToSaveTestOnly.keys():
        emptyUsers.append( {'user_id': user, 'productsx' : 'None'})

if len(emptyUsers) > 0:
    csvToSave = csvToSave.append(emptyUsers)

#add order_ids to the CSV

csvToSave = pd.merge(pcb.usersInTest, csvToSave, on='user_id')

csvToSave['products'] = csvToSave['productsx'] # this is to make sure order_id is put in the CSV before products, as they are serialized in the order they are created in the dataframe

pcb.debugWithTimer("saving CSV")
csvWithCorrectColumns = pd.DataFrame()
csvWithCorrectColumns['order_id'] = csvToSave['order_id']
csvWithCorrectColumns['products'] = csvToSave['products']
csvWithCorrectColumns.reindex(columns=['order_id', 'products']).to_csv('data\\myprediction1.csv', index=False, header=True, columns={'order_id', 'products'})
#csvWithCorrectColumns.to_csv('data\\myprediction1.csv', index=False, header=True, columns={'order_id', 'products'})


pcb.debugWithTimer("generating score estimate")
usercount = 0
sumf1 = 0.0
myprediction = predictionToSaveFull.groupby('user_id')['product_id'].apply(list)
for index, x in pcb.truthPerUser.iteritems():

    if index in train['user_id'].values:
        continue

    usercount = usercount +1

    #if myprediction.__contains__(index) == True:
    if index in myprediction:
        xx = pcb.eval_fun(pcb.truthPerUser[index],myprediction[index])
        sumf1 = sumf1 + xx[2]

if usercount != 0:
    sumf1 = sumf1/usercount
    print(sumf1)
else:
    print("No user, no predictions. Pbbbbbt")

pcb.debugWithTimer("done")
total = time.perf_counter() - pcb.start
print("total time ", end='')
print(total)
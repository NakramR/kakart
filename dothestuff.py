import petitchatbase as pcb
import thebestpredictionsherenotintheotherone as bestpredictions
import predictions as predictions
import numpy as np
import pandas as pd
import time
import random
import sys

#pcb.maxuserid = '10'
pcb.maxuserid = '100'
#pcb.maxuserid = '10000'
#pcb.maxuserid = '1000000000'

if sys.argv[1]:
    pcb.maxuserid = str(sys.argv[1])

random.seed(42)

pcb.debugWithTimer("initializing data")
pcb.initData(pcb.maxuserid)

pcb.debugWithTimer("splitting train/test orders")

pcb.train, pcb.test = pcb.trainAndTestForValidation()
pcb.train, pcb.test = pcb.balancedTrainAndTestForValidation()

#print(usersInTest)


## predictions

# pcb.debugWithTimer("generating random prediction")
# p1 = bestpredictions.generateRandomPrediction()

# pcb.debugWithTimer("generating freq threshold prediction")
# p2 = bestpredictions.predictOverFrequencyThreshold(0.35)
#
# pcb.debugWithTimer("generating decision tree prediction")
# p3 = predictions.generateDecisionTreePrediction(pcb.train, pcb.test)
#
# pcb.debugWithTimer("generating linear regression prediction")
# p4 = predictions.generateLinearRegressionPrediction(pcb.train, pcb.test)
#
# pcb.debugWithTimer("generating xgboost prediction")
# p5 = predictions.generateXGBoostPrediction(pcb.train, pcb.test)

# pcb.debugWithTimer("generating xgboost prediction")
# p6 = bestpredictions.sLogistic(pcb.train, pcb.test)

#print(pcb.train['user_id'].unique())

pcb.debugWithTimer("generating myFirstNN prediction")
p7 = bestpredictions.myFirstNN(pcb.train, pcb.test)

# for i in range(5, 8): #threshold between 0.25 and 0.4 is where the good stuff is, possibly
#      pcb.debugWithTimer("generating freq threshold prediction + " + str(i*0.05))
#      p2 = bestpredictions.predictOverFrequencyThreshold(i*0.05)
#      pcb.debugWithTimer("scoring p2: "+ str(i*0.05))
#      pcb.scorePrediction(p2)


#pcb.scorePrediction(p2)

# pcb.debugWithTimer("scoring p3: ")
# pcb.scorePrediction(p3)
# pcb.debugWithTimer("scoring p4: ")
# pcb.scorePrediction(p4)
# pcb.debugWithTimer("scoring p5: ")
# pcb.scorePrediction(p5)
# pcb.debugWithTimer("scoring p6: ")
# pcb.scorePrediction(p6)
pcb.debugWithTimer("scoring p7: ")
pcb.scorePrediction(p7)

predictionToSaveFull = p7

pcb.debugWithTimer("creating CSV")
predictionToSaveFull = predictionToSaveFull[predictionToSaveFull['predy'] == True]
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
pcb.scorePrediction(predictionToSaveFull)
# usercount = 0
# sumf1 = 0.0
# myprediction = predictionToSaveFull.groupby('user_id')['product_id'].apply(list)
# for index, x in pcb.truthPerUser.iteritems():
#
#     if index in pcb.train['user_id'].values:
#         continue
#
#     usercount = usercount +1
#
#     #if myprediction.__contains__(index) == True:
#     if index in myprediction:
#         xx = pcb.eval_fun(pcb.truthPerUser[index],myprediction[index])
#         sumf1 = sumf1 + xx[2]
#
# if usercount != 0:
#     sumf1 = sumf1/usercount
#     print(sumf1)
# else:
#     print("No user, no predictions. Pbbbbbt")

pcb.debugWithTimer("done")
total = time.perf_counter() - pcb.start
print("total time ", end='')
print(total)
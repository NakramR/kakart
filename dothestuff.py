import petitchatbase as pcb
import thebestpredictionsherenotintheotherone as bestpredictions
import predictions as predictions
import moreNN
import numpy as np
import pandas as pd
import time
import random
import sys
import matplotlib.pyplot as plt

pcb.debugWithTimer("donewithImport")
#pcb.maxuserid = '10'
pcb.maxuserid = '100'
#pcb.maxuserid = '10000'
#pcb.maxuserid = '1000000000'


if sys.argv[1]:
    pcb.maxuserid = str(sys.argv[1])
    if pcb.maxuserid == 'all':
        pcb.maxuserid = '1000000000'

random.seed(42)

pcb.debugWithTimer("initializing data")
pcb.initData(pcb.maxuserid)

pcb.debugWithTimer("splitting train/test orders")

pcb.train, pcb.test = pcb.trainAndTestForValidation()

# pcb.trainidx, pcb.stestidx, pcb.testidx = pcb.balancedTrainAndTestDFIDXForValidation()
#
# pcb.train = pcb.userProductStats.loc[pcb.trainidx]
# pcb.test = pcb.userProductStats.loc[pcb.stestidx]
#
pcb.train, pcb.test = pcb.balancedTrainAndTestForValidation()



#print(usersInTest)

lastPrediction = []
## predictions

# pcb.debugWithTimer("generating random prediction")
# p1 = bestpredictions.generateRandomPrediction()
# lastPrediction = p1

# pcb.debugWithTimer("generating freq threshold prediction")
# p2 = bestpredictions.predictOverFrequencyThreshold(0.35)
# pcb.debugWithTimer("scoring p2: ")
# pcb.scorePrediction(p2)
# # 10k f1:0.314086319171    full: f1:0.305183197346
# lastPrediction = p2

#
# pcb.debugWithTimer("generating decision tree prediction")
# p3 = predictions.generateDecisionTreePrediction(pcb.train, pcb.test)
# lastPrediction = p3
#
# pcb.debugWithTimer("generating linear regression prediction")
# p4 = predictions.generateLinearRegressionPrediction(pcb.train, pcb.test)
# lastPrediction = p4
#
# pcb.debugWithTimer("generating xgboost prediction")
# p5 = predictions.generateXGBoostPrediction(pcb.train, pcb.test)
# pcb.scorePrediction(p5)
# lastPrediction = p5
# # 10k f1:0.331689111983







# pcb.debugWithTimer("generating stephan's logistic prediction")
# p6 = bestpredictions.sLogistic(pcb.train, pcb.test)
# pcb.debugWithTimer("scoring p6: ")
# pcb.scorePrediction(p6)
# # 10k f1: 0.314483286208    full: f1:0.303212998498
# lastPrediction = p6

#print(pcb.train['user_id'].unique())

# pcb.debugWithTimer("generating myFirstNN prediction")
# p7 = bestpredictions.myFirstNN(pcb.train, pcb.test)
# pcb.debugWithTimer("scoring p7: ")
# pcb.scorePrediction(p7)
# lastPrediction = p7

# pcb.debugWithTimer("generating mySecondNN prediction")
# p8 = bestpredictions.mySecondNN(pcb.train, pcb.test)
# pcb.debugWithTimer("scoring p7: ")
# pcb.scorePrediction(p8)
# lastPrediction = p8


# pcb.debugWithTimer("generating TF prediction")
# p9 = predictions.predictFirstTime(pcb.train, pcb.test)
# lastPrediction = p9
# pcb.debugWithTimer("scoring p9: ")
# pcb.scorePrediction(p9)
# lastPrediction = p9

# pcb.debugWithTimer("generating LSTM prediction")
# p10 = predictions.lstm(pcb.train, pcb.test)
# pcb.debugWithTimer("scoring p10: ")
# pcb.scorePrediction(p10)
# lastPrediction = p10

#
# pcb.debugWithTimer("generating thirdNN prediction")
# p11 = bestpredictions.myThirdNN(pcb.train, pcb.test)
# pcb.debugWithTimer("scoring p11: ")
# pcb.scorePrediction(p11)
# lastPrediction = p11

pcb.debugWithTimer("generating generateXGBoostPredictionLeChat prediction")

p12 = bestpredictions.generateXGBoostPredictionLeChat(pcb.train, pcb.test, depth=5, estimators=80, learning_rate=0.1)
pcb.debugWithTimer("***scoring p12:")
pcb.scorePrediction(p12)
lastPrediction = p12

pcb.debugWithTimer("generating fourthNN prediction")
p13 = moreNN.myFourthNN(pcb.train, pcb.test, True)
pcb.debugWithTimer("scoring p13: ")
#pcb.scorePrediction(p13)
lastPrediction = p13




#combined = combinePredictions([p2, p5, p7])





predictionToSaveFull = lastPrediction

if predictionToSaveFull is None:
    exit(-1)
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
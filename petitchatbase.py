import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn
import random
from sklearn.model_selection import *
from sqlalchemy import create_engine
import sklearn
import time
import os.path
from sklearn.metrics import f1_score
from collections import Counter
import sys
from ast import literal_eval


random.seed(42)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

debug = True

maxuserid = '10' # will be overriden

start = time.perf_counter()
lasttime = start

products = [] # pd.read_csv('data\\products.csv')
prod_prior = [] #pd.read_csv('data\\order_products__prior.csv')
prod_train = [] #pd.read_csv('data\\order_products__train.csv')
orders = [] #pd.read_csv('data\\orders.csv')
userProductStats = []
truth = []
truthperuser = []
usersInTest = []
uniqueproductperusercache = []

train = []
test = []
holdout = []
trainidx = []
holdoutidx = []
testidx = []




# ideas:
# - if an order happens on the same day as another, probably isn't for something that was in the other order
# - if an item is ordered every time for a user, it's likely to be included
# - if an item has a high reorder percentage for all users after first inclusion, inclusion more likely?
# - if an item has a low reorder percentage for all users after first inclusion, inclusion not likely?



### functions here functions here functions here functions here functions here functions here functions here
### functions here functions here functions here functions here functions here functions here functions here
### functions here functions here functions here functions here functions here functions here functions here
### functions here functions here functions here functions here functions here functions here functions here
### functions here functions here functions here functions here functions here functions here functions here
### functions here functions here functions here functions here functions here functions here functions here
### functions here functions here functions here functions here functions here functions here functions here
### functions here functions here functions here functions here functions here functions here functions here
### functions here functions here functions here functions here functions here functions here functions here
### functions here functions here functions here functions here functions here functions here functions here

# for a given user_id, compute which of the products in the orders had previously been ordered, and which had been immediately been preorder


def debugWithTimer(message):
    global lasttime, debug
    if debug == True:
        print( '[ %s seconds ] ' % round(time.perf_counter() - lasttime,3) )
        print(message + "... ", end='', flush=True )
        lasttime = time.perf_counter()

def are_we_running_in_debug_mode():
    gettrace = getattr(sys, 'gettrace', None)
    if gettrace is None:
        print("can't find anything")
        return False # don't know, really
    elif gettrace():
        print("we're running in debug mode")
        return True
    else:
        print("we're not running in debug mode")
        return False

# because sklearn's has its own random seed.
def deterministic_train_holdout_split(list, test_size):
    random.shuffle(list)
    cutoff = int(len(list)*test_size)
    holdout = list[:cutoff]
    train = list[cutoff:]

    return train, holdout


def initData(maxusers):
    global prod_prior, prod_train, orders, userProductStats, truth, truthperuser, usersInTest, cacheStore

    files = { 'prod_prior'       : 'SELECT orders.user_id, prod_prior.* FROM prod_prior LEFT JOIN orders ON orders.order_id = prod_prior.order_id WHERE user_id <'
            , 'prod_train'       : 'SELECT orders.user_id, prod_train.* FROM prod_train LEFT JOIN orders ON orders.order_id = prod_train.order_id WHERE user_id <'
            , 'orders'          : 'SELECT * FROM orders_e WHERE user_id <'
            , 'truth'           : 'SELECT orders.user_id, prod_train.product_id, TRUE AS ordered FROM prod_train LEFT JOIN orders ON orders.order_id = prod_train.order_id WHERE user_id <'
            , 'usersInTest'     : "SELECT user_id, order_id FROM orders WHERE eval_set = 'test' AND user_id <"
            , 'products'        : "SELECT * FROM products_e"
            , 'userProductStats': '__THIS_IS_REPLACED_BY_A_FUNCTION_CALL__'
             }

    cacheStore = pd.HDFStore('data/cache/store' + str(maxuserid) +'.h5' )

    for file, query in files.items():
        reload = True
        if cacheStore.__contains__(file):
            cachedVar = cacheStore[file]
            exec("global " + file + ";" + file + ' = cachedVar')
            reload = False
            # temporary
            if file == 'products' and not('ordertoreorderfreq' in cachedVar.columns):
                reload = True

        if reload:
            if file == 'userProductStats':
                x = getUserProductStats(maxusers)
                userProductStats = x
            else:
                if query[-1] == '<':
                    query = query + ' ' + maxusers
                x = pd.read_sql(query, postgresconnection)
                exec( "global " + file + ";" + file + ' = x')

            cacheStore[file] = x

    cacheStore.close()

    truthperuser = truth.groupby('user_id')['product_id'].apply(list)


def eval_fun(labels, preds):
    rr = (np.intersect1d(labels, preds))
    precision = np.float(len(rr)) / len(preds)
    recall = np.float(len(rr)) / len(labels)
    try:
        f1 = 2 * precision * recall / (precision + recall)
    except ZeroDivisionError:
        return (precision, recall, 0.0)
    return (precision, recall, f1)

def scorePrediction(predictionperitem):
    global truthperuser, train, uniqueproductperusercache, truth
    usercount = 0
    sumf1 = 0.0
    sumf1x = 0.0

    if predictionperitem is None:
        return 0,0

    debugWithTimer("preparing score")
#    uniquetrainusers = truth['user_id'].unique() #truth contains all train results

    myprediction = predictionperitem[predictionperitem['predy'] == True]
    myprediction = myprediction.groupby('user_id')['product_id'].apply(list)

    if ( len(uniqueproductperusercache) == 0 ):
        uniqueproductperuser = userProductStats.groupby('user_id')['product_id'].unique() #past products only
        uniqueproductperusercache = uniqueproductperuser
    else:
        uniqueproductperuser = uniqueproductperusercache

    usercount = len(predictionperitem['user_id'].unique()) -len(myprediction) # count people who have no prediction


    print('Prediction frequency: %s ' % Counter(predictionperitem['predy']))

    debugWithTimer("iterating")
    for index, x in truthperuser.iteritems():

        # if index in uniquetrainusers:
        #     continue

        if index in myprediction:
            usercount = usercount + 1
            #eval_fun
            #xx = eval_fun(truthperuser[index], myprediction[index])
            #sumf1 = sumf1 + xx[2]

            # get the full product list, including entirely new products that were not present in training data
            fulluserprods = set().union(list(uniqueproductperuser[index]),list(myprediction[index]))

            # get a boolean match between truth & full product list
            # get a boolean match between prediction & full product list
            bPred = list(i in myprediction[index] for i in fulluserprods)
            bTruth = list(i in truthperuser[index] for i in fulluserprods)

            sumf1x = sumf1x + sklearn.metrics.f1_score(bTruth, bPred)

    if usercount != 0:
        sumf1 = sumf1 / usercount
        sumf1x = sumf1x / usercount
        print(" Scoring sklearn.f1:", end='')
        print(sumf1x)
    else:
        print("No user, no predictions. Pbbbbbt")

    return sumf1, sumf1x

def addPercentages(user_id):
    global orders, allproductorders, userproducts
    orderIdsByUser = orders[orders["user_id"] == user_id].sort_values(by='order_number').order_id
    allProducts = []

    for order_id in orderIdsByUser :
        products = allproductorders[allproductorders["order_id"]== order_id].product_id
        commonProducts = set(products).intersection(allProducts)

        if ( len(products) == 0 ):
            print(user_id)
            percentage = -1
        else:
            percentage = len(commonProducts)/len(products)

        allProducts.extend(products)

        orders.loc[orders.order_id == order_id, 'priorpercent'] = percentage

    userproducts[user_id] = allProducts

def calculateAverageProductOrders(row, *args):
    #print(row['orderfrequency'])
    threshold = args[0]

    # assume we're splitting by 5
    if threshold > int(row['totaluserorders'])+5:
        return row['orderfreqlast'+str(threshold-5)]

    nbpdorders = sum(int(x >= int(row['totaluserorders'])-threshold) for x in row['order_containing_product'])
    if threshold > int(row['totaluserorders']):
        nbpdorders = float(nbpdorders)/int(row['totaluserorders'])
    else:
        nbpdorders = float(nbpdorders) / threshold
    return nbpdorders

def getUserProductStats(maxuser):

    step = 0 #hijack this if problems
    saveIntermediarySteps = False

    for i in range(9,-1,-1):
        filepath = 'data/cache/userProductStats' + str(maxuser) + '.step' + str(i) + '.csv'
        if os.path.isfile(filepath):
            debugWithTimer("loading csv cached step")
            d = pd.read_csv(filepath)
            step = i + 1
            debugWithTimer("converting string back to arrays")
            d['order_containing_product'] = d['order_containing_product'].apply(
                literal_eval)  # convert the string representation of arrays back to arrays
            break

    print('Reloading From Step' + str(step))

    if step == 0:
        query = "SELECT u.*, prod_train.reordered FROM userproducttable2 u LEFT JOIN prod_train on u.eval_order_id = prod_train.order_id AND u.product_id = prod_train.product_id WHERE user_id < " + maxuser
        d = pd.read_sql(query, postgresconnection)
        if saveIntermediarySteps == True:
            d.to_csv('data/cache/userProductStats' + str(maxuser) + '.step'+ str(step) + ".csv")
        step = step + 1

    if step == 1:
        # IMPUTATION
        debugWithTimer("imputation")
        missingValues(d)
        # d["order_days_since_prior_product_order"].fillna(0, inplace=True)
        d["dayfrequency"].fillna(0, inplace=True)
        d["reordered"].fillna(0, inplace=True)

        if saveIntermediarySteps == True:
            debugWithTimer("saving CSV")
            d.to_csv('data/cache/userProductStats' + str(maxuser) + '.step'+ str(step) + ".csv")
        step = step + 1

    if step == 2:
        debugWithTimer("converting string array to array")
        d['order_containing_product'] = d['product_order_sequence'].apply(lambda x: list(map(int, x.split(','))))

        if saveIntermediarySteps == True:
            debugWithTimer("saving CSV")
            d.to_csv('data/cache/userProductStats' + str(maxuser) + '.step'+ str(step) + ".csv")
        step = step + 1

    if step == 3:
        debugWithTimer("sorting order array")
        d['order_containing_product'].apply(lambda x: x.sort())

        if saveIntermediarySteps == True:
            debugWithTimer("saving CSV")
            d.to_csv('data/cache/userProductStats' + str(maxuser) + '.step'+ str(step) + ".csv")
        step = step + 1

    if step == 4:
        debugWithTimer("orderfreqoverratio")
        d['orderfreqoverratio'] = d.apply(
            lambda row: 0 if row['dayfrequency'] == 0 else float(row['days_without_product_order']) / row[
                'dayfrequency']
            , axis=1)  # days without product order expressed as as a ratio of the usual order frequency.

        if saveIntermediarySteps == True:
            debugWithTimer("saving CSV")
            d.to_csv('data/cache/userProductStats' + str(maxuser) + '.step'+ str(step) + ".csv")
        step = step + 1

    if step == 5:
        debugWithTimer("step 5 ")
        for i in range(1, 5):
            debugWithTimer("step i" + str(i))
            d['orderfreqlast' + str(i * 5)] = d.apply(calculateAverageProductOrders, args=(i * 5,), axis=1)

        if saveIntermediarySteps == True:
            debugWithTimer("saving CSV")
            d.to_csv('data/cache/userProductStats' + str(maxuser) + '.step'+ str(step) + ".csv")
        step = step + 1

    if step == 6:
        debugWithTimer("step 6 ")
        for i in range(5, 10):
            debugWithTimer("step i" + str(i))
            d['orderfreqlast' + str(i * 5)] = d.apply(calculateAverageProductOrders, args=(i * 5,), axis=1)

        if saveIntermediarySteps == True:
            debugWithTimer("saving CSV")
            d.to_csv('data/cache/userProductStats' + str(maxuser) + '.step'+ str(step) + ".csv")
        step = step + 1

    if step == 7:
        debugWithTimer("step 7 ")
        for i in range(10, 15):
            debugWithTimer("step i" + str(i))
            d['orderfreqlast' + str(i * 5)] = d.apply(calculateAverageProductOrders, args=(i * 5,), axis=1)

        if saveIntermediarySteps == True:
            debugWithTimer("saving CSV")
            d.to_csv('data/cache/userProductStats' + str(maxuser) + '.step'+ str(step) + ".csv")
        step = step + 1

    if step == 8:
        debugWithTimer("step 8 ")
        for i in range(15, 20):
            debugWithTimer("step i" + str(i))
            d['orderfreqlast' + str(i * 5)] = d.apply(calculateAverageProductOrders, args=(i * 5,), axis=1)

        if saveIntermediarySteps == True:
            debugWithTimer("saving CSV")
            d.to_csv('data/cache/userProductStats' + str(maxuser) + '.step'+ str(step) + ".csv")
        step = step + 1

    # x = list(map(int,d['product_order_sequence'][0].split(',')))
    # x.sort()

    return d

# TODO: create features: user groups (preferred products)
# TODO: create features: user groups (purchase frequency)
# TODO: create features: user groups (order frequency)
# TODO: create features: product groups (product groups (throughout aisles/departments) by how frequently they are reordered)
# TODO: create features: seasonal products reordering vs order day/ per product

def missingValues(df):
    total = df.isnull().sum().sort_values(ascending=False)
    percent = (df.isnull().sum() / df.isnull().count()).sort_values(ascending=False)
    missing_data = pd.concat([total, percent], axis=1, keys=['Total Missing', 'Percent'])
    print(missing_data)

def trainAndTestForValidation():
    originalTrain = userProductStats[userProductStats['testortrain'] != 'test']
    originalTest = userProductStats[userProductStats['testortrain'] == 'test']
    print('originalTrain : ', originalTrain.shape, ' originalTest ', originalTest.shape)

    uniqueusers = originalTrain['user_id'].unique()

    trainUsers, testUsers = deterministic_train_holdout_split(uniqueusers, test_size=0.2)

    train = originalTrain[originalTrain['user_id'].isin(trainUsers)]
    test = originalTrain[~originalTrain['user_id'].isin(trainUsers)]

    train = train.sample(frac=1) #shuffle
    test = pd.concat([test, originalTest])
    train = train.drop(['testortrain'], axis=1)
    test = test.drop(['testortrain'], axis=1)
    print('train : ', train.shape, ' test ', test.shape)
    return train, test

def balancedTrainAndTestForValidation(mostrecent=100):

    ups = userProductStats.merge(products[['numorders', 'numreorders', 'numusers', 'reordersperuser', 'ordertoreorderfreq', 'product_id',
                     'aisle_id', 'department_id']], on='product_id', suffixes=('', '_'))

    originalTrain = ups[ups['testortrain'] != 'test']
    test = ups[ups['testortrain'] == 'test']

    print('originalTrain : ', originalTrain.shape, ' originalTest ', test.shape)

    # split users in holdout and train
    uniqueTrainUsers = originalTrain['user_id'].unique()
    trainUsers, holdoutUsers = deterministic_train_holdout_split(uniqueTrainUsers, test_size=0.2)

    train = originalTrain[originalTrain['user_id'].isin(trainUsers)]
    holdout = originalTrain[originalTrain['user_id'].isin(holdoutUsers)]

    # get the same number of negative samples as positive samples
    numPositive = len(train[train['reordered'] == 1])
    negIndices = list(train[train['reordered'] == 0].index)
    random.shuffle(negIndices)

    indicesToKeep = negIndices[:numPositive]
    indicesToKeep.extend(list(train[train['reordered'] == 1].index))

    balancedTrain = train.loc[indicesToKeep]

    # shuffle them to avoid all 0s followed by all 1s
    train = balancedTrain.sample(frac=1)

    train = train.drop(['testortrain'], axis=1)
    holdout = holdout.drop(['testortrain'], axis=1)
    test = test.drop(['testortrain'], axis=1)

    print('train : ', train.shape, ' test ', test.shape, ' holdout ', holdout.shape)
    return train, holdout, test

def balancedTrainAndTestDFIDXForValidation(mostrecent=100):

    tt = userProductStats[['testortrain','reordered','user_id']]

    originalTrain = tt[tt['testortrain'] != 'test']
    originalTest = tt[tt['testortrain'] == 'test']
    print('originalTrain : ', originalTrain.shape, ' originalTest ', originalTest.shape)

    uniqueusers = originalTrain['user_id'].unique()

    trainUsers, testUsers = deterministic_train_holdout_split(uniqueusers, test_size=0.2)

    train = originalTrain[originalTrain['user_id'].isin(trainUsers)]
    holdout = originalTrain[~originalTrain['user_id'].isin(trainUsers)]

    #shuffle train
    train = train.sample(frac=1)

    # maintain equal balance of positive/negative results
    posIndices = train[train['reordered'] == 1].index
    numPositive = len(posIndices)
    negIndices = list(train[train['reordered'] == 0].index)
    random.shuffle(negIndices)

    indicesToKeep = negIndices[:numPositive]
    indicesToKeep.extend(list(posIndices))

    trainIndices = indicesToKeep
    holdoutIndices = holdout.index
    testIndices = originalTest.index

    return trainIndices, holdoutIndices, testIndices
    # train2 = train.loc[indicesToKeep]
    #
    # train = train2.sample(frac=1)
    # test = pd.concat([test, originalTest])
    # train = train.drop(['testortrain'], axis=1)
    # test = test.drop(['testortrain'], axis=1)
    # print('train : ', train.shape, ' test ', test.shape)
    # return train, test

def trainAndTestForSubmission():
    originalTrain = userProductStats[userProductStats['testortrain'] != 'test']
    originalTest = userProductStats[userProductStats['testortrain'] == 'test']
    print('originalTrain : ', originalTrain.shape, ' originalTest ', originalTest.shape)

    return originalTrain, originalTest

def isInPriorRun(definition):
    priorScores = loadPriorRuns()
    if priorScores is None:
        return False

    if maxuserid + 'Ç' + str(definition) + "\n" in priorScores:
        return True
    else:
        return False

def loadPriorRuns():
    priorScores = {}
    with open('priorRuns.txt', 'r') as f:
        line = f.readline()
        while line != '':
            x = line.split('Å')
            priorScores[x[1]] = x[0]
            line = f.readline()
    return priorScores

def saveRun(definition, score):
    with open('priorRuns.txt', 'a') as f:
        f.write("{:.5f}".format(score) + 'Å' + maxuserid + 'Ç' + str(definition) + "\n")


### END OF FUNCTIONS END OF FUNCTIONS END OF FUNCTIONS END OF FUNCTIONS END OF FUNCTIONS END OF FUNCTIONS
### END OF FUNCTIONS END OF FUNCTIONS END OF FUNCTIONS END OF FUNCTIONS END OF FUNCTIONS END OF FUNCTIONS
### END OF FUNCTIONS END OF FUNCTIONS END OF FUNCTIONS END OF FUNCTIONS END OF FUNCTIONS END OF FUNCTIONS
### END OF FUNCTIONS END OF FUNCTIONS END OF FUNCTIONS END OF FUNCTIONS END OF FUNCTIONS END OF FUNCTIONS
### END OF FUNCTIONS END OF FUNCTIONS END OF FUNCTIONS END OF FUNCTIONS END OF FUNCTIONS END OF FUNCTIONS
### END OF FUNCTIONS END OF FUNCTIONS END OF FUNCTIONS END OF FUNCTIONS END OF FUNCTIONS END OF FUNCTIONS
### END OF FUNCTIONS END OF FUNCTIONS END OF FUNCTIONS END OF FUNCTIONS END OF FUNCTIONS END OF FUNCTIONS
### END OF FUNCTIONS END OF FUNCTIONS END OF FUNCTIONS END OF FUNCTIONS END OF FUNCTIONS END OF FUNCTIONS
### END OF FUNCTIONS END OF FUNCTIONS END OF FUNCTIONS END OF FUNCTIONS END OF FUNCTIONS END OF FUNCTIONS
### END OF FUNCTIONS END OF FUNCTIONS END OF FUNCTIONS END OF FUNCTIONS END OF FUNCTIONS END OF FUNCTIONS
### END OF FUNCTIONS END OF FUNCTIONS END OF FUNCTIONS END OF FUNCTIONS END OF FUNCTIONS END OF FUNCTIONS
### END OF FUNCTIONS END OF FUNCTIONS END OF FUNCTIONS END OF FUNCTIONS END OF FUNCTIONS END OF FUNCTIONS


debugWithTimer("setting up postgres connection... ")
postgresconnection = create_engine('postgresql://stephan:saipass@192.168.1.5:5432/kakart')

debugWithTimer("reading CSVs")

aisles = pd.read_csv('data\\aisles.csv')
departments = pd.read_csv('data\\departments.csv')

debugWithTimer("done reading CSVs")











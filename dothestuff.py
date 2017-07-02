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
import thebestpredictionsherenotintheotherone as bestpredictions

random.seed(42)
pd.set_option('display.float_format', lambda x: '%.3f' % x)


debug = True
def debugWithTimer(message):
    global lasttime, debug
    if debug == True:
        print( round(time.perf_counter() - lasttime,3) )
        print(message + "... ", end='', flush=True )
        lasttime = time.perf_counter()

start = time.perf_counter()
lasttime = start

#maxuserid = '10'
#maxuserid = '1000'
maxuserid = '1000000000'



debugWithTimer("setting up postgres connection... ")
postgresconnection = create_engine('postgresql://stephan:saipass@192.168.1.5:5432/kakart')

debugWithTimer("reading CSVs")

aisles = pd.read_csv('data\\aisles.csv')
products = pd.read_csv('data\\products.csv')
departments = pd.read_csv('data\\departments.csv')

prod_prior = [] #pd.read_csv('data\\order_products__prior.csv')
prod_train = [] #pd.read_csv('data\\order_products__train.csv')
orders = [] #pd.read_csv('data\\orders.csv')
userproductstats = []


## Save to postgres
#aisles.to_sql("aisles", engine)
#products.to_sql("products", postgresconnection)
#departments.to_sql("departments", postgresconnection)
#prod_prior.to_sql("prod_prior", postgresconnection, chunksize=1024, if_exists='append')
#prod_train.to_sql("prod_train", postgresconnection, chunksize=1024)
#orders.to_sql("orders", postgresconnection, chunksize=1024)


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


def initData(maxusers):
    global prod_prior, prod_train, orders, userproductstats

    if os.path.isfile('data\\cache\\prod_prior' + maxusers + '.csv'):
        prod_prior = pd.read_csv('data\\cache\\prod_prior' + maxusers + '.csv')
        prod_train = pd.read_csv('data\\cache\\prod_train' + maxusers + '.csv')
        orders = pd.read_csv('data\\cache\\orders' + maxusers + '.csv')
    else:
        prod_prior = pd.read_sql('SELECT orders.user_id, prod_prior.* FROM prod_prior LEFT JOIN orders ON orders.order_id = prod_prior.order_id WHERE user_id < ' + maxusers, postgresconnection)
        prod_train = pd.read_sql('SELECT orders.user_id, prod_train.* FROM prod_train LEFT JOIN orders ON orders.order_id = prod_train.order_id WHERE user_id < ' + maxusers, postgresconnection)
        orders = pd.read_sql('SELECT * FROM orders WHERE user_id < ' + maxusers, postgresconnection)
        prod_prior.to_csv('data\\cache\\prod_prior' + maxusers + '.csv')
        prod_train.to_csv('data\\cache\\prod_train' + maxusers + '.csv')
        orders.to_csv('data\\cache\\orders' + maxusers + '.csv')

    if os.path.isfile('data\\cache\\userproductstats' + maxusers + '.csv'):
        userproductstats = pd.read_csv('data\\cache\\userproductstats' + maxusers + '.csv')
    else:
        userproductstats = getUserProductStats(maxusers)
        userproductstats.to_csv('data\\cache\\userproductstats' + maxusers + '.csv')




def eval_fun(labels, preds):
    rr = (np.intersect1d(labels, preds))
    precision = np.float(len(rr)) / len(preds)
    recall = np.float(len(rr)) / len(labels)
    try:
        f1 = 2 * precision * recall / (precision + recall)
    except ZeroDivisionError:
        return (precision, recall, 0.0)
    return (precision, recall, f1)


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

        #orders[(orders["order_id"] == order_id)]["priorpercent"] = percentage

    userproducts[user_id] = allProducts
    #orders[order_id]['priorpercent'] = 0.98
    #orders[order_id]['immediatepriorpercent'] = 0.98
    #user[user_id][product_id].frequency = 0.75

def getUserProductStats(maxuser):
    query = "SELECT * FROM userproducttable WHERE user_id < " + maxuser

    d = pd.read_sql(query, postgresconnection)

    d = pd.merge(d, prod_train[['product_id', 'user_id','reordered']], on=['user_id', 'product_id'], how='left')

    # IMPUTATION
    print('IMPUTATION')
    missingValues(d)
    # d["order_days_since_prior_product_order"].fillna(0, inplace=True)
    d["dayfrequency"].fillna(0, inplace=True)

    return d

def missingValues(df):
    total = df.isnull().sum().sort_values(ascending=False)
    percent = (df.isnull().sum() / df.isnull().count()).sort_values(ascending=False)
    missing_data = pd.concat([total, percent], axis=1, keys=['Total Missing', 'Percent'])
    print(missing_data)


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


## split the orders in each subcategory prior/train/test

debugWithTimer("initializing data")
initData(maxuserid)
orders["priorpercent"] = np.nan
orders["immediatepriorpercent"] = np.nan

debugWithTimer("splitting train/test orders")

trainorders = orders[orders['eval_set'] == 'train']

train, test = train_test_split(trainorders, test_size = 0.2)




# dataperuser = pd.DataFrame(data=orders.user_id.unique(), columns={'user_id'})
#
# plt.clf()
# seaborn.countplot(data=orders, x='order_dow')
# plt.show()
#
#
# plt.clf()
# seaborn.countplot(data=orders, x='order_hour_of_day')
# plt.show()
#
# plt.clf()
# seaborn.countplot(data=train, x='order_number')
# plt.show()

#orderextra = pd.read_csv('data\\ordersextra.csv')



debugWithTimer("reading SQL truth")
#userpriorproducts = pd.read_sql('SELECT DISTINCT prod_prior.product_id, orders.user_id FROM prod_prior LEFT JOIN orders ON orders.order_id = prod_prior.order_id WHERE user_id < ' + maxuserid, postgresconnection)
truth = pd.read_sql('SELECT orders.user_id, prod_train.product_id, TRUE AS ordered FROM prod_train LEFT JOIN orders ON orders.order_id = prod_train.order_id WHERE user_id < ' + maxuserid, postgresconnection)
#print(truth)

truthperuser = truth.groupby('user_id')['product_id'].apply(list)
#print(truthperuser)

debugWithTimer("getting test users")
usersInTest = pd.read_sql("SELECT user_id, order_id FROM orders WHERE eval_set = 'test' AND user_id < " + maxuserid, postgresconnection)
#print(usersInTest)


## predictions

debugWithTimer("generating random prediction")
#p1 = bestpredictions.generateRandomPrediction()


debugWithTimer("generating freq threshold prediction")
p2 = bestpredictions.predictOverFrequencyThreshold(0.3)






predictionToSaveFull = p2



debugWithTimer("creating CSV")
predictionToSaveFull = predictionToSaveFull[predictionToSaveFull['ordered'] == True]
predictionToSaveTestOnly  = predictionToSaveFull[predictionToSaveFull['user_id'].isin(usersInTest['user_id'].values.tolist())].groupby('user_id')['product_id'].apply(list)
#print(predictionDF)

debugWithTimer("formatting CSV")
csvToSave = pd.DataFrame(columns={'user_id', 'predictions'})
csvToSave['predictions'] = predictionToSaveTestOnly
csvToSave['user_id'] = predictionToSaveTestOnly.keys()
csvToSave['productsx'] = csvToSave['predictions'].map(lambda x: ' '.join(str(xx) for xx in x))

## add users that have no predictions

emptyUsers = []
for user in usersInTest['user_id']:
    if user not in predictionToSaveTestOnly.keys():
        emptyUsers.append( {'user_id': user, 'products' : 'None'})

if len(emptyUsers) > 0:
    csvToSave = csvToSave.append(emptyUsers)

#add order_ids to the CSV

csvToSave = pd.merge(usersInTest, csvToSave, on='user_id')

csvToSave['products'] = csvToSave['productsx'] # this is to make sure order_id is put in the CSV before products, as they are serialized in the order they are created in the dataframe

debugWithTimer("saving CSV")
csvToSave.to_csv('data\\myprediction1.csv', index=False, header=True, columns={'order_id', 'products'})

debugWithTimer("generating score estimate")
usercount = 0
sumf1 = 0.0
myprediction = predictionToSaveFull.groupby('user_id')['product_id'].apply(list)
for index, x in truthperuser.iteritems():
    usercount = usercount +1

    if myprediction.__contains__(index) == True:
        xx = eval_fun(truthperuser[index],myprediction[index])
        sumf1 = sumf1 + xx[2]

sumf1 = sumf1/usercount
print(sumf1)

debugWithTimer("done")

# plt.clf()
# seaborn.jointplot(x='order_number', y='priorpercent', data=orders)
# plt.show()
#
# print(train.count())
# print(test.count())
#
# print('blah')







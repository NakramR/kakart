import os.path
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
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostRegressor
from sklearn.cross_validation import cross_val_score

random.seed(42)

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
maxuserid = '10'
#maxuserid = '1000'
#maxuserid = '1000000000'



debugWithTimer("setting up postgres connection... ")
postgresconnection = create_engine('postgresql://stephan:saipass@192.168.1.5:5432/kakart')

debugWithTimer("reading CSVs")

aisles = pd.read_csv('data\\aisles.csv')
products = pd.read_csv('data\\products.csv')
departments = pd.read_csv('data\\departments.csv')

prod_prior = [] #pd.read_csv('data\\order_products__prior.csv')
prod_train = [] #pd.read_csv('data\\order_products__train.csv')
orders = [] #pd.read_csv('data\\orders.csv')
userProductStats = []


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




def initData(maxusers):
    global prod_prior, prod_train, orders, userProductStats

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

    if os.path.isfile('data\\cache\\userProductStats' + maxusers + '.csv'):
        userproductstats = pd.read_csv('data\\cache\\userProductStats' + maxusers + '.csv')
    else:
        userproductstats = getUserProductStats(maxusers)
        userproductstats.to_csv('data\\cache\\userProductStats' + maxusers + '.csv')




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
    query = """ SELECT DISTINCT user_id, product_id, numproductorders, totaluserorders , firstproductorder , lastproductorder
, CAST(numproductorders AS float)/((totaluserorders-firstproductorder )+1) AS frequency
, (SELECT eval_set FROM orders WHERE eval_set <> 'prior' AND orders.user_id = pustats.user_id ) AS testortrain
FROM
(
SELECT
p.order_id, p.product_id, user_id, order_number
, COUNT(*) OVER (PARTITION BY user_id, p.product_id) AS numproductorders
, MAX(order_number) OVER (PARTITION BY user_id) AS totaluserorders
, MIN(order_number) OVER (PARTITION BY user_id, p.product_id) AS firstproductorder
, MAX(order_number) OVER (PARTITION BY user_id, p.product_id) AS lastproductorder
FROM prod_prior p
LEFT JOIN orders ON orders.order_id = p.order_id
--LEFT JOIN products ON p.product_id = products.product_id
WHERE user_id < """ + maxuser + """
ORDER BY user_id, order_number
) AS pustats
ORDER BY user_id, frequency DESC, product_id"""

    d = pd.read_sql(query, postgresconnection)

    d = pd.merge(d, prod_train[['product_id', 'user_id','reordered']], on=['user_id', 'product_id'], how='left')
    d["reordered"].fillna(0, inplace=True)
    return d



def generateDecisionTreePrediction():
    print('Decision tree...')
    estimator = DecisionTreeClassifier(max_leaf_nodes=10, random_state=0)
    train = userProductStats[userProductStats['testortrain'] == 'train']
    test = userProductStats[userProductStats['testortrain'] == 'test']

    train = train.drop(['testortrain'], axis=1)
    test = test.drop(['testortrain'], axis=1)

    print('train : ', len(train), ' test ', test.shape)
    X_train = train.drop(['reordered'], axis=1)
    y_train = train['reordered']

    estimator.fit(X_train, y_train)

    #y_pred = cross_val_score(estimator=estimator, X=X_train, y=y_train, cv=3, n_jobs=1) #returns 3 results

    y_pred = estimator.predict(test.drop(['reordered'], axis=1))
    df = pd.DataFrame(columns=('user_id', 'product_id', 'ordered'))
    df['user_id'] = test['user_id']
    df['product_id'] = test['product_id']
    df['ordered'] = y_pred

    return df


def generateDecisionTreeByUserPrediction():
    print('Decision tree...')
    estimator = DecisionTreeClassifier(max_leaf_nodes=7, random_state=0)

    prediction = []
    for name, group in userProductStats.groupby('user_id'):
        X_train = group.drop(['reordered'], axis=1)
        y_train = group['reordered']
        estimator.fit(X_train, y_train)
        y_pred = estimator.predict(X_train)
        prediction.extend(y_pred)
    #uniqueUserId = X_train.user_id.unique()
    print('pred : ', len(prediction), ' shape ', userProductStats.shape)

    df = pd.DataFrame(columns=('user_id', 'product_id', 'ordered'))
    df['user_id'] = userProductStats['user_id']
    df['product_id'] = userProductStats['product_id']
    df['ordered'] = prediction

    return df


#groupby('user_id')

def generateRandomPrediction():
    randpred = pd.DataFrame(columns=('user_id', 'product_id', 'ordered'))
    debugWithTimer('reading distinct user products')
    userpriorproducts = userProductStats

    #userpriorproducts = pd.read_sql('SELECT DISTINCT prod_prior.product_id, orders.user_id FROM prod_prior LEFT JOIN orders ON orders.order_id = prod_prior.order_id WHERE user_id < ' + maxuserid, postgresconnection)

    #print(userpriorproducts.describe())

    debugWithTimer('iterating over prior products')
    temp = []
    for index, x in userpriorproducts.iterrows():

        newline = { 'user_id': x['user_id'], 'product_id': x['product_id'], 'ordered': (random.random() > 0.5) }
        temp.append(newline)


    randpred = randpred.append(temp)

    return randpred




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


debugWithTimer("generating random prediction")
X_train = userProductStats.drop(['reordered'], axis=1)
y_train = userProductStats['reordered']

#p1 = generateRandomPrediction()
p1 = generateDecisionTreePrediction()
#print(p1)

debugWithTimer("reading SQL truth")
#userpriorproducts = pd.read_sql('SELECT DISTINCT prod_prior.product_id, orders.user_id FROM prod_prior LEFT JOIN orders ON orders.order_id = prod_prior.order_id WHERE user_id < ' + maxuserid, postgresconnection)
truth = pd.read_sql('SELECT orders.user_id, prod_train.product_id, TRUE AS ordered FROM prod_train LEFT JOIN orders ON orders.order_id = prod_train.order_id WHERE user_id < ' + maxuserid, postgresconnection)
#print(truth)

truthperuser = truth.groupby('user_id')['product_id'].apply(list)
#print(truthperuser)

myprediction = p1[p1['ordered'] == True].groupby('user_id')['product_id'].apply(list)
#print(myprediction)

debugWithTimer("getting test users")
usersInTest = pd.read_sql("SELECT user_id, order_id FROM orders WHERE eval_set = 'test' AND user_id < " + maxuserid, postgresconnection)
#print(usersInTest)

debugWithTimer("creating CSV")
predictionToSave = p1[p1['ordered'] == True]
xxx1 = predictionToSave[predictionToSave['user_id'].isin(usersInTest['user_id'].values.tolist())]
xxx2 = xxx1.groupby('user_id')['product_id']
xxx3 = xxx2.apply(list)
predictionDF = xxx3
#print(predictionDF)

debugWithTimer("formatting CSV")
csvToSave = pd.DataFrame(columns={'user_id', 'predictions'})
csvToSave['predictions'] = xxx3
csvToSave['user_id'] = xxx3.keys()
csvToSave['productsx'] = csvToSave['predictions'].map(lambda x: ' '.join(str(xx) for xx in x))

## add users that have no predictions

emptyUsers = []
for user in usersInTest['user_id']:
    if user not in xxx3.keys():
        emptyUsers.append( {'user_id': user, 'productsx' : 'None'})

if len(emptyUsers) > 0:
    csvToSave = csvToSave.append(emptyUsers)

#add order_ids to the CSV

csvToSave = pd.merge(usersInTest, csvToSave, on='user_id')

csvToSave['products'] = csvToSave['productsx']

debugWithTimer("saving CSV")
csvToSave.to_csv('data\\myprediction1.csv', index=False, header=True, columns={'order_id', 'products'})

debugWithTimer("generating score estimate")
usercount = 0
sumf1 = 0.0
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




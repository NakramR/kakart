import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn
import random
from sklearn.model_selection import *


aisles = pd.read_csv('data\\aisles.csv')
products = pd.read_csv('data\\products.csv')
departments = pd.read_csv('data\\departments.csv')

prod_prior = pd.read_csv('data\\order_products__prior.csv')
#prod_prior = pd.read_csv('data\\order_products__train.csv')
prod_train = pd.read_csv('data\\order_products__train.csv')

allproductorders = pd.concat([prod_prior, prod_train])

orders = pd.read_csv('data\\orders.csv')

samplesubmission = pd.read_csv('data\\sample_submission.csv')

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

orders["priorpercent"] = np.nan
orders["immediatepriorpercent"] = np.nan
# for a given user_id, compute which of the products in the orders had previously been ordered, and which had been immediately been preorder
def addPercentages(user_id):
    global orders, allproductorders
    orderIdsByUser = orders[orders["user_id"] == user_id].sort_values(by='order_number').order_id
    allProducts = []

    for order_id in orderIdsByUser :
        products = allproductorders[allproductorders["order_id"]== order_id].product_id
        commonProducts = set(products).intersection(allProducts)

        if ( len(products) == 0 ):
            print("0")
            percentage = -1
        else:
            percentage = len(commonProducts)/len(products)

        allProducts.extend(products)

        orders.loc[orders.order_id == order_id, 'priorpercent'] = percentage
        #orders[(orders["order_id"] == order_id)]["priorpercent"] = percentage


    #orders[order_id]['priorpercent'] = 0.98
    #orders[order_id]['immediatepriorpercent'] = 0.98

    #user[user_id][product_id].frequency = 0.75



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


c = 0
for user_id in orders.user_id:
    c = c + 1
    if c%10 == 0:
        print(c)
    addPercentages(user_id)

orders.to_csv('data\\ordersextra.csv')

seaborn.jointplot(x='order_number', y='priorpercent', data=orders)

print(train.count())
print(test.count())

print('blah')







import pandas as pd
import random
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
import petitchatbase as pcb
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import tensorflow as tf
from collections import Counter

random.seed(42)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

#pcb.maxuserid = '10'
#pcb.maxuserid = '1000'
#pcb.maxuserid = '1000000000'


def generateDecisionTreePrediction(train, test):
    print('\n##################\nDecision tree\n##################')
    estimator = DecisionTreeClassifier(max_leaf_nodes=10, random_state=0)

    X_train = train.drop(['reordered'], axis=1)
    y_train = train['reordered']

    estimator.fit(X_train, y_train)
    #y_pred = cross_val_score(estimator=estimator, X=X_train, y=y_train, cv=3, n_jobs=1) #returns 3 results

    y_pred = estimator.predict(test.drop(['reordered'], axis=1))
    df = pd.DataFrame(columns=('user_id', 'product_id', 'predy'))
    df['user_id'] = test['user_id']
    df['product_id'] = test['product_id']
    df['predy'] = y_pred
    return df


def generateLogisticRegressionPrediction(train, test):
    print('\n##################\nLogistic Regression\n##################')
    estimator = LogisticRegression(class_weight='balanced')

    X_train = train.drop(['reordered'], axis=1)
    y_train = train['reordered']

    estimator.fit(X_train, y_train)
    y_pred = estimator.predict(test.drop(['reordered'], axis=1))

    df = pd.DataFrame(columns=('user_id', 'product_id', 'predy'))
    df['user_id'] = test['user_id']
    df['product_id'] = test['product_id']
    df['predy'] = y_pred
    return df


def generateLinearRegressionPrediction(train, test):
    print('\n##################\nLinear Regression\n##################')
    estimator = LinearRegression()

    X_train = train.drop(['reordered'], axis=1)
    y_train = train['reordered']

    estimator.fit(X_train, y_train)
    estimator.score(X_train, y_train)
    # Equation coefficient and Intercept
    print('Coefficient: \n', estimator.coef_)
    print('Intercept: \n', estimator.intercept_)
    y_pred = estimator.predict(test.drop(['reordered'], axis=1))

    df = pd.DataFrame(columns=('user_id', 'product_id', 'predy'))
    df['user_id'] = test['user_id']
    df['product_id'] = test['product_id']
    df['predy'] = y_pred
    return df



def generateXGBoostPrediction(train, test):
    print('\n##################\nXGBoost\n##################')
    estimator = XGBClassifier()

    X_train = train.drop(['reordered'], axis=1)
    y_train = train['reordered']

    estimator.fit(X_train, y_train)
    y_pred = estimator.predict(test.drop(['reordered'], axis=1))

    df = pd.DataFrame(columns=('user_id', 'product_id', 'predy'))
    df['user_id'] = test['user_id']
    df['product_id'] = test['product_id']
    df['predy'] = y_pred
    return df


def predictFirstTime(train, test):
    features = ['orderfrequency']
    tf.set_random_seed(42)
    x_train = train[features]
    y_train = np.column_stack(
        (list(float(not (i)) for i in train['reordered']),
         list(float(i) for i in train['reordered'])))

    noProducts = len(pcb.products)
    inputPlaceholder = tf.placeholder('float', [None, noProducts], name='myWonderfullInput')
    truthYPlaceholder = tf.placeholder('float', [None, 1], name="mylabels")
    for userId, group in train.groupby('user_id'):
        input = np.zeros(noProducts)
        output = np.zeros(noProducts)
        for prodId in group['product_id']:
            input[prodId] = group[group['product_id'] == prodId]['orderfrequency']
        for prodId in pcb.truthperuser[userId]:
            if prodId not in group['product_id'].values:
                output[prodId] = 1
        print('Counter input : %s, \n output %s' % ( Counter(input), Counter(output)))

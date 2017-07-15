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
    inputValues = []
    outputValues = []
    for userId, group in train.groupby('user_id'):
        inputPerUser = np.zeros(noProducts)
        outputPerUser = np.zeros(noProducts)
        for prodId in group['product_id']:
            inputPerUser[prodId] = group[group['product_id'] == prodId]['orderfrequency']
        inputValues.append(inputPerUser)
        for prodId in pcb.truthperuser[userId]:
            if prodId not in group['product_id'].values:
                outputPerUser[prodId] = 1
        outputValues.append(outputPerUser)
        # print('Counter input : %s, \n output %s' % ( Counter(inputPerUser), Counter(outputPerUser)))
        print('.', end="", flush=True)

    hiddenLayerSizes = [1000]


    inputPlaceholder = tf.placeholder('float', [None, noProducts], name='myWonderfullInput')
    truthYPlaceholder = tf.placeholder('float', [None, noProducts], name="mylabels")
    previousLayer = inputPlaceholder
    previousLayerSize = noProducts
    count = 0
    for layerSize in hiddenLayerSizes:
        count = count+1
        w = tf.Variable(tf.random_normal([previousLayerSize, layerSize]), name="w" + str(count))
        b = tf.Variable(tf.random_normal([layerSize]), name="b" + str(count))

        preact = tf.add(tf.matmul(previousLayer, w), b, name="preactivation" + str(count))
        act = tf.nn.relu(preact, name="relu" + str(count))
        # with tf.name_scope('dropout'):
        #     layer = tf.nn.dropout(act, neuronDropoutRate, name='dropout'+str(count))

        previousLayer = act
        previousLayerSize = layerSize


    w = tf.Variable(tf.random_normal([previousLayerSize, noProducts]), name="w" + str(count))
    preact = tf.matmul(previousLayer, w, name="activation" + str(count))
    # act = tf.nn.softmax(preact)
    output = preact
    lossFunction = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=truthYPlaceholder),
                                  name="xent")
    prediction = tf.nn.sigmoid(output)

    rmse = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(prediction, truthYPlaceholder))))
    correct = tf.reduce_mean(tf.square(tf.subtract(prediction, truthYPlaceholder)), name="correct")
    accuracy = tf.reduce_mean(tf.cast(correct, 'float'), name="accuracyMeasure")

    train_step = tf.train.AdamOptimizer(learning_rate=0.01).minimize(lossFunction)


    with tf.Session() as s:
        tf.set_random_seed(42)
        tf.global_variables_initializer().run()

        batchSize = 100
        curStep = 1

        for curStep in range(1,int(len(outputValues)/batchSize)+1):
            batch_x = inputValues[(curStep-1)*batchSize:(curStep)*batchSize]
            batch_y = outputValues[(curStep-1)*batchSize:(curStep)*batchSize]

            feed_dict = {inputPlaceholder: batch_x, truthYPlaceholder: batch_y}
            train_step.run(feed_dict=feed_dict)

            if (curStep % 10 == 0):
                acc, r, err = s.run([accuracy, rmse, lossFunction], feed_dict=feed_dict)
                print('Accuracy: %s, rmse : %s error:%s ' % (str(acc), r, str(err)))

        # print('Accuracy: %s, rmse : %s error:%s ' % (str(acc), r, str(err)))

    # o = prediction.eval(feed_dict={input:x_test})

    df = pd.DataFrame(columns=('user_id', 'product_id', 'predy'))
    df['user_id'] = test['user_id']
    df['product_id'] = test['product_id']
    df['predy'] = o

    _, f1score = pcb.scorePrediction(df)

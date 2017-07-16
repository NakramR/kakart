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
from sklearn.calibration import CalibratedClassifierCV
from tensorflow.contrib import layers

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
    features = ['orderfrequency', 'dayfrequency', 'days_without_product_order', 'department_id', 'aisle_id',
                 'eval_days_since_prior_order', 'numproductorders', 'totaluserorders', 'user_id', 'product_id']
    param = {}
    #param['booster'] = 'gbtree'
    param['objective'] = 'binary:logistic'
    # param["eval_metric"] = "error"
    # param['eta'] = 0.3
    # param['gamma'] = 0
    param['max_depth'] = 4
    param['n_estimators'] =80
    param['learning_rate'] = 0.1
    # param['min_child_weight'] = 1
    # param['max_delta_step'] = 0
    #param['subsample'] = 1
    # param['colsample_bytree'] = 1
    # param['silent'] = 1
    # param['seed'] = 0
    #param['base_score'] = 0.4

    X_train = train[features]
    test = test[features]

    y_train = train['reordered']

    estimator = XGBClassifier()
    estimator.set_params(**param)
    metLearn = CalibratedClassifierCV(estimator, method='sigmoid', cv=5)
    metLearn.fit(X_train, y_train)
    y_pred = metLearn.predict(test)

    # estimator.fit(X_train, y_train)
    # y_pred = estimator.predict(test)
    print('Predict counter : %s' % (Counter(y_pred)))


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

    for prodid in pcb.products['product_id']:
        tf.reset_default_graph()

        # get truth values per product

        outputValuesPerProduct = list([ i[int(prodid)]] for i in outputValues)

        inputPlaceholder = tf.placeholder('float', [None, noProducts], name='myWonderfullInput')
        truthYPlaceholder = tf.placeholder('float', [None,1], name="mylabels")
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


        w = tf.Variable(tf.random_normal([previousLayerSize, 1]), name="w" + str(count))
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

        acc = 0
        r = 0
        err = 0
        with tf.Session() as s:
            tf.set_random_seed(42)
            tf.global_variables_initializer().run()

            batchSize = 100
            curStep = 1

            feed_dict = {}
            for curStep in range(1,int(len(outputValuesPerProduct)/batchSize)+2):
                batch_x = inputValues[(curStep-1)*batchSize:(curStep)*batchSize]
                batch_y = outputValuesPerProduct[(curStep-1)*batchSize:(curStep)*batchSize]

                feed_dict = {inputPlaceholder: batch_x, truthYPlaceholder: batch_y}
                train_step.run(feed_dict=feed_dict)

                if (curStep % 10 == 0):
                    acc, r, err = s.run([accuracy, rmse, lossFunction], feed_dict=feed_dict)
                    print('Accuracy: %s, rmse : %s error:%s ' % (str(acc), r, str(err)))

            acc, r, err = s.run([accuracy, rmse, lossFunction], feed_dict=feed_dict)
            print('Accuracy for product %s: %s, rmse : %s error:%s ' % (prodid, str(acc), r, str(err)))

    o = prediction.eval(feed_dict={input:test})

    df = pd.DataFrame(columns=('user_id', 'product_id', 'predy'))
    df['user_id'] = test['user_id']
    df['product_id'] = test['product_id']
    df['predy'] = o

    _, f1score = pcb.scorePrediction(df)

def lstm(train, test):
    batchsize = 50
    SEQLEN = 30
    ALPHASIZE =  len(pcb.products)# number of products
    CELLSIZE = 512
    NLAYERS = 2
    Xd = tf.placeholder(tf.uint8, [None, None,1 ])#batchsize, seqlen,
    #X = tf.one_hot(Xd, ALPHASIZE, 1.0, 0.0)
    X = Xd

    Yd = tf.placeholder(tf.uint8, [None, None,1])
    #Y_ = tf.one_hot(Yd, ALPHASIZE, 1.0, 0.0)
    Y_ = Yd

    Hin = tf.placeholder(tf.float32, [None, CELLSIZE * NLAYERS])

    cell = tf.contrib.rnn.GRUCell(CELLSIZE)
    mcell = tf.contrib.rnn.MultiRNNCell([cell] * NLAYERS, state_is_tuple = False)
    Hr, H = tf.nn.dynamic_rnn(mcell, X, initial_state=Hin)

    Hf = tf.reshape(Hr, [-1, CELLSIZE])
    Ylogits = layers.linear(Hf, ALPHASIZE)
    Y = tf.nn.softmax(Ylogits)
    Yp = tf.argmax(Y, 1)
    Yp = tf.reshape(Yp, [batchsize, -1 ])

    loss = tf.nn.softmax_cross_entropy_with_logits(logits = Ylogits, labels =Y_)
    train_step = tf.train.AdamOptimizer(1e-3).minimize(loss)

    with tf.Session() as sess:
        curStep = 1

        for curStep in range(1, int(len(train) / batchsize) + 1):
            inH = np.zeros([batchsize, CELLSIZE * NLAYERS])

            batch_x = train[(curStep - 1) * batchsize:(curStep) * batchsize]
            batch_y = train[(curStep - 1) * batchsize:(curStep) * batchsize]

            data = {X: batch_x, Y_: batch_y, Hin : inH}
            _, y, outH= sess.run([train_step,Yp, H, ],feed_dict=data)
            inH = outH


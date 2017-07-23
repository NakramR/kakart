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
    ALPHASIZE =  1 # len(pcb.products)# number of products
    CELLSIZE = 512
    NLAYERS = 2

    pkeep = tf.placeholder(tf.float32)
    lr = tf.placeholder(tf.float32)

    X = tf.placeholder(tf.float32, [None, None, 1], name='X')  # [ BATCHSIZE, SEQLEN ]
    # expected outputs = same sequence shifted by 1 since we are trying to predict the next character
    Y_ = tf.placeholder(tf.float32, [None, None, 1], name='Y_')  # [ BATCHSIZE, SEQLEN ]
    # input state
    Hin = tf.placeholder(tf.float32, [None, CELLSIZE * NLAYERS], name='Hin')  # [ BATCHSIZE, INTERNALSIZE * NLAYERS]

    # using a NLAYERS=3 layers of GRU cells, unrolled SEQLEN=30 times
    # dynamic_rnn infers SEQLEN from the size of the inputs Xo

    # How to properly apply dropout in RNNs: see README.md
    cells = [tf.contrib.rnn.GRUCell(CELLSIZE) for _ in range(NLAYERS)]
    # "naive dropout" implementation
    dropcells = [tf.contrib.rnn.DropoutWrapper(cell, input_keep_prob=pkeep) for cell in cells]
    multicell = tf.contrib.rnn.MultiRNNCell(dropcells, state_is_tuple=False)
    multicell = tf.contrib.rnn.DropoutWrapper(multicell, output_keep_prob=pkeep)  # dropout for the softmax layer

    Yr, H = tf.nn.dynamic_rnn(multicell, X, dtype=tf.float32, initial_state=Hin)
    # Yr: [ BATCHSIZE, SEQLEN, INTERNALSIZE ]
    H = tf.identity(H, name='H')  # just to give it a name

    # Softmax layer implementation:
    # Flatten the first two dimension of the output [ BATCHSIZE, SEQLEN, ALPHASIZE ] => [ BATCHSIZE x SEQLEN, ALPHASIZE ]
    # then apply softmax readout layer. This way, the weights and biases are shared across unrolled time steps.
    # From the readout point of view, a value coming from a sequence time step or a minibatch item is the same thing.

    Yflat = tf.reshape(Yr, [-1, CELLSIZE])  # [ BATCHSIZE x SEQLEN, INTERNALSIZE ]
    Ylogits = layers.linear(Yflat, ALPHASIZE)  # [ BATCHSIZE x SEQLEN, ALPHASIZE ]
    Yflat_ = tf.reshape(Y_, [-1, ALPHASIZE])  # [ BATCHSIZE x SEQLEN, ALPHASIZE ]
    loss = tf.nn.softmax_cross_entropy_with_logits(logits=Ylogits, labels=Yflat_)  # [ BATCHSIZE x SEQLEN ]
    loss = tf.reshape(loss, [batchsize, -1])  # [ BATCHSIZE, SEQLEN ]

    Yo = tf.nn.sigmoid(Ylogits, name='Yo')  # [ BATCHSIZE x SEQLEN, ALPHASIZE ]
    Y = Yo  # [ BATCHSIZE x SEQLEN ]

    Y = tf.reshape(Y, [batchsize, -1], name="Y")  # [ BATCHSIZE, SEQLEN ]
    train_step = tf.train.AdamOptimizer(lr).minimize(loss)

    # stats for display
    seqloss = tf.reduce_mean(loss, 1)
    batchloss = tf.reduce_mean(seqloss)
    #accuracy = tf.reduce_mean(tf.cast(tf.equal(Y_, Y), tf.float32))

    accuracy = tf.reduce_mean(tf.abs(tf.add(Y, -Y_)), name="correct")

    loss_summary = tf.summary.scalar("batch_loss", batchloss)
    acc_summary = tf.summary.scalar("batch_accuracy", accuracy)
    summaries = tf.summary.merge([loss_summary, acc_summary])

    columnNames = ["D-" + str(i) for i in range(SEQLEN)]

    batch_x = pd.DataFrame(columns=columnNames)
    batch_y = pd.DataFrame(columns=columnNames)

    with tf.Session() as sess:
        curStep = 1
        tf.global_variables_initializer().run()

        for curStep in range(1, int(len(train) / batchsize) + 1):
            inH = np.zeros([batchsize, CELLSIZE * NLAYERS])

            batch = train[(curStep - 1) * batchsize:(curStep) * batchsize]

            (batch_x, batch_y) = createBatchArray(batch, SEQLEN)

            # for i, row in batch.iterrows():
            #
            #     x = createOrderBitfield(row, SEQLEN, 'x')
            #     y = createOrderBitfield(row, SEQLEN, 'y')
            #     batch_x.loc[len(batch_x)-1] = list(x)
            #     batch_y.loc[len(batch_y)-1] = list(y)

                # zeze = batch_x.apply(createOrderBitfield, args=(SEQLEN,'x'), axis=1)
                # rere = batch_x.apply(createOrderBitfield, args=(SEQLEN,'y'), axis=1)


            data = {X: batch_x, Y_: batch_y, Hin : inH, lr: 0.001, pkeep:1.0}
            _, y, outH= sess.run([train_step,Y, H, ],feed_dict=data)
            inH = outH

            if (curStep % 10 == 0):
                acc, batcherr, seqerr = sess.run([accuracy, batchloss, seqloss], feed_dict=data)
                print('Accuracy: %s, batchloss : %s seqless:%s ' % (str(acc), str(batcherr), str(seqerr)))

        acc, batcherr, seqerr = sess.run([accuracy, batchloss, seqloss], feed_dict=data)
        print('Accuracy: %s, batchloss : %s seqless:%s ' % (str(acc), str(batcherr), str(seqerr)))

        test_batch, _ = createBatchArray(test, SEQLEN)

        inH = np.zeros([batchsize, CELLSIZE * NLAYERS])
        data = {X: batch_x, Hin : inH}
        o = Y.eval(feed_dict=data)

    df = pd.DataFrame(columns=('user_id', 'product_id', 'predy'))
    df['user_id'] = test['user_id']
    df['product_id'] = test['product_id']
    df['predy'] = o

    _, f1score = pcb.scorePrediction(df)



def createOrderBitfield(row, *args):
    SEQLEN = args[0]
    x = np.zeros([SEQLEN,1])
    r = row['order_containing_product']
    total = row['totaluserorders']
    total = min(total, SEQLEN)

    if ( args[1] == 'y'): #looking for the output, shift everything by one, and add 'reordered'
        x[list((i + SEQLEN - total - 2) for i in r[-(SEQLEN-1):]), 0] = 1
        x[SEQLEN-1,0] = row['reordered']
    else:
        x[list((i + SEQLEN - total - 1) for i in r[-SEQLEN:]), 0] = 1

    return x

def createBatchArray(batch, SEQLEN):
    x = np.zeros((len(batch),SEQLEN,1))
    y = np.zeros((len(batch),SEQLEN,1))

    count = 0
    for i, row in batch.iterrows():
        r = row['order_containing_product']
        total = row['totaluserorders']
        total = max(total, SEQLEN)

        for ii in r:
            indexFromEnd = total-ii
            if indexFromEnd < 0:
                continue
            x[count,indexFromEnd,0] = 1
            if indexFromEnd > 0:
                y[count,indexFromEnd-1,0] = 1
        y[count,SEQLEN-1,0] = row['reordered']
        count = count+1

    return (x,y)

    return x

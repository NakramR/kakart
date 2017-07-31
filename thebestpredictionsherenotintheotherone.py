import petitchatbase as pcb
import pandas as pd
import random
import sklearn
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
import tensorflow as tf
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sys
import operator
import math
import matplotlib as pyplot
import xgboost
from xgboost import XGBClassifier
from collections import Counter
from sklearn.calibration import CalibratedClassifierCV
import matplotlib.pyplot as plt


def generateRandomPrediction():
    randpred = pd.DataFrame(columns=('user_id', 'product_id', 'predy'))
    pcb.debugWithTimer('reading distinct user products')
    userpriorproducts = pcb.userProductStats

    pcb.debugWithTimer('iterating over prior products')
    temp = []
    for index, x in userpriorproducts.iterrows():
        newline = { 'user_id': x['user_id'], 'product_id': x['product_id'], 'predy': (random.random() > 0.5) }
        temp.append(newline)

    randpred = randpred.append(temp)

    return randpred

def predictOverFrequencyThreshold(threshold):
    userpriorproducts = pcb.userProductStats

    userpriorproducts['predy'] = userpriorproducts['orderfrequency'] > threshold

    #userpriorproducts['predy'] = userpriorproducts.query("orderfrequency > " + str(threshold) + " or (dayfrequency > days_without_product_order + eval_days_since_prior_order)")

    return userpriorproducts

def sLogistic(train, test):
    print('\n##################\nStephan''s Logistic\n##################')

    #X_train = train.drop(['reordered'], axis=1)
    features = ['orderfrequency', 'dayfrequency', 'department_id', 'aisle_id', 'days_without_product_order','eval_days_since_prior_order']
    #features = ['orderfrequency']
    X_train = train[features]
    Y_train = train['reordered']

    kfold= sklearn.model_selection.KFold(n_splits=10,random_state=42)
    model = LogisticRegression(class_weight='balanced')
    scoring = 'accuracy'
    results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)

    print("Accuracy: %.3f (%.3f)" % (results.mean(), results.std()))
    print(results)
    model.fit( X_train, Y_train)

    y_pred = model.predict(test[features])
    #y_pred = model.predict(test.drop(['reordered'], axis=1))

    df = pd.DataFrame(columns=('user_id', 'product_id', 'predy'))
    df['user_id'] = test['user_id']
    df['product_id'] = test['product_id']
    df['predy'] = y_pred
    return df

def generateXGBoostPredictionLeChat(train, holdout, test, depth=4, estimators=80, learning_rate=0.1):
    print('\n##################\nXGBoost\n##################')
    features = ['orderfrequency', 'dayfrequency', 'department_id', 'aisle_id', 'days_without_product_order','eval_days_since_prior_order',
                   'numproductorders', 'totaluserorders','day_number_of_last_product_order', 'eval_order_dow', 'orderfreqoverratio', 'orderfreqlast5', 'orderfreqlast10',
       'orderfreqlast15', 'reordersperuser', 'ordertoreorderfreq']

    param = {}
    #param['booster'] = 'gbtree'
    param['objective'] = 'binary:logistic'
    # param['objective'] = 'multi:softprob'
    # param["eval_metric"] = "error"
    # param['eta'] = 0.3
    # param['gamma'] = 0
    param['max_depth'] = depth
    param['n_estimators'] =estimators
    param['learning_rate'] = learning_rate
    # param['min_child_weight'] = 1
    # param['max_delta_step'] = 0
    #param['subsample'] = 1
    # param['colsample_bytree'] = 1
    # param['silent'] = 1
    # param['seed'] = 0
    #param['base_score'] = 0.4

    X_train = train[features]
    #test = test[features]

    y_train = train['reordered']

    estimator = XGBClassifier()
    estimator.set_params(**param)
    metLearn = CalibratedClassifierCV(estimator, method='sigmoid', cv=5)
    metLearn.fit(X_train, y_train)

#    secLearn = RandomizedSearchCV(estimator, n_jobs = 3, cv=5 )

    testNoID = test[features]

    yPred = metLearn.predict(testNoID)
    yPredHoldout = metLearn.predict_proba(holdout[features])
    yPredHoldout = list(el[1] for el in yPredHoldout)

    # estimator.fit(X_train, y_train)
    # xgboost.plot_importance(estimator, height=0.2)
    # plt.show()

    # estimator.fit(X_train, y_train)
    # y_pred = estimator.predict(test)
    print('Predict counter : %s, holdout %s' % (Counter(yPred), Counter(yPredHoldout)))


    df = pd.DataFrame(columns=('user_id', 'product_id', 'predy'))
    df['user_id'] = test['user_id']
    df['product_id'] = test['product_id']
    df['predy'] = yPred

    threshold = 0.7
    dfHoldout = pd.DataFrame(columns=('user_id', 'product_id', 'predy'))
    dfHoldout['user_id'] = holdout['user_id']
    dfHoldout['product_id'] = holdout['product_id']
    dfHoldout['predy'] = list(int(el >= threshold) for el in yPredHoldout)
    dfHoldout['floaty'] = yPredHoldout
    dfHoldout.to_csv('data/results/xgboost' + pcb.maxuserid + '.csv', index=False)

    return dfHoldout, df

def makeHyperParamString(hiddenLayerSizes, dropoutRate, numFeatures, optimizer, learningrate, lossFunction, extra):

    s = ''
    if pcb.maxuserid == '1000000000':
        s = s + 'uall'
    elif  int(pcb.maxuserid) > 1000:
        s = s + 'u' + str(int(pcb.maxuserid)/1000) + 'k'
    else:
        s = s + 'u' + str(pcb.maxuserid)
    s = s + "-feat" + str(numFeatures) + "("

    for layerSize in hiddenLayerSizes:
        s = s + str(layerSize) + '-'
    s = s + '2)'
    s = s + '-dropout' + str(dropoutRate)
    s = s + '-' + optimizer
    s = s + '-lr.' + str(learningrate)
    s = s + '-loss.' + lossFunction
    s = s + extra
    return s


def myFirstNN(train, test):

    # define hyperparameters

    allfeatures = ['orderfrequency', 'dayfrequency', 'department_id', 'aisle_id', 'days_without_product_order','eval_days_since_prior_order',
                   'numproductorders', 'totaluserorders','day_number_of_last_product_order', 'eval_order_dow']

    features9 = ['orderfrequency', 'dayfrequency', 'days_without_product_order', 'department_id', 'aisle_id','eval_days_since_prior_order', 'numproductorders', 'totaluserorders','eval_days_since_prior_order']
    features4 = ['orderfrequency', 'dayfrequency', 'days_without_product_order', 'department_id']
    features1 = ['orderfrequency']

    bSaveTFSummary = False
    if (len(sys.argv) > 2 and sys.argv[2] == '1' and not (pcb.are_we_running_in_debug_mode())):
        bSaveTFSummary = True  # don't save summary in debug mode (incomplete data) or when explicitly requested not
    else:
        bSaveTFSummary = False

    tf.set_random_seed(42)

    possibleOptimizers = ['adagrad', 'adam']
    possibleFeatures = [features4, features9] #[features1, features4, features9]
    possibleDropoutRates = [0.75, 0.9, 1.0 ]
    possibleNetworkLayerShapes = [ [20], [20,20], [30,20], [30,20,10], [30], [10], [30,10] ]

    hyperParamExplorationDict = []
    for fdef in possibleFeatures:
        for nshape in possibleNetworkLayerShapes:
            for optimizerName in possibleOptimizers:
                for dr in possibleDropoutRates:
                    hyperParamExplorationDict.extend(
                    [
                        { 'features': fdef
                          ,'hiddenLayerSizes' : nshape
                          ,'dropoutRate' : dr
                          ,'optimizerName' :optimizerName  # gradientDescent, adagrad, adam
                          ,'lr' : 0.001
                          ,'lf' : 'softmaxxent'  # sigmoidxent, softmaxxent, weighted
                          ,'extra' : '-balancedinput'
                        }
                    ]
                    )

    # just one, the best
    # hyperParamExplorationDict = [{'features': ['orderfrequency', 'dayfrequency', 'days_without_product_order', 'department_id', 'aisle_id', 'eval_days_since_prior_order', 'numproductorders', 'totaluserorders', 'eval_days_since_prior_order'], 'hiddenLayerSizes': [50, 20], 'dropoutRate': 0.9, 'optimizerName': 'adam', 'lr': 0.001, 'lf': 'softmaxxent', 'extra': '-balancedinput'}]
    # hyperParamExplorationDict = [{'features': ['orderfrequency', 'dayfrequency', 'days_without_product_order', 'department_id'], 'hiddenLayerSizes': [20, 20], 'dropoutRate': 0.75, 'optimizerName': 'adam', 'lr': 0.001, 'lf': 'softmaxxent','extra': '-balancedinput'}, {'features': ['orderfrequency', 'dayfrequency', 'days_without_product_order', 'department_id', 'aisle_id', 'eval_days_since_prior_order', 'numproductorders', 'totaluserorders', 'eval_days_since_prior_order'], 'hiddenLayerSizes': [50, 20], 'dropoutRate': 0.9, 'optimizerName': 'adam', 'lr': 0.001, 'lf': 'softmaxxent', 'extra': '-balancedinput'}]
    hyperParamExplorationDict = [{'features': ['orderfrequency', 'dayfrequency', 'days_without_product_order', 'department_id', 'aisle_id','eval_days_since_prior_order', 'numproductorders', 'totaluserorders', 'eval_days_since_prior_order'],'hiddenLayerSizes': [50, 20, 5], 'dropoutRate': 0.9, 'optimizerName': 'adam', 'lr': 0.001, 'lf': 'softmaxxent','extra': '-balancedinput'}]
    # 0.32838
    bestScore = 0
    bestDefinition = {}
    bestDF = None
    allScores = {}

    lastfeatures = None
    x_train = None
    y_train = None

    for oneDefinition in hyperParamExplorationDict : #placeholder for hyperparam exploration
        features = oneDefinition['features']
        hiddenLayerSizes =oneDefinition['hiddenLayerSizes']
        dropoutRate =oneDefinition['dropoutRate']
        optimizerName =oneDefinition['optimizerName']
        lr=oneDefinition['lr']
        lf =oneDefinition['lf']
        extra = oneDefinition['extra']

        nbfeatures = len(features)

        if lastfeatures != features:
            x_train = train[features]
            y_train = np.column_stack(
                (list(float(not (i)) for i in train['reordered']),
                 list(float(i) for i in train['reordered'])))
            lastfeatures = features


        tf.reset_default_graph()

        # define input and output
        with tf.name_scope('input'):
            inputPlaceholder = tf.placeholder('float', [None, nbfeatures], name='myWonderfullInput')
            truthYPlaceholder = tf.placeholder('float', [None, 2], name="mylabels")

        with tf.name_scope('dropout_rate'):
            neuronDropoutRate = tf.placeholder('float')
            tf.summary.scalar('dropout_keep_probability', neuronDropoutRate)


        hyperParamStr = makeHyperParamString(hiddenLayerSizes, dropoutRate, nbfeatures, optimizerName, lr, lf, extra)

        previousLayer = inputPlaceholder
        previousLayerSize = nbfeatures
        count = 0
        for layerSize in hiddenLayerSizes:
            count = count+1
            with tf.name_scope('hiddenLayer' + str(count)):
                with tf.name_scope('weights'):
                    w = tf.Variable(tf.random_normal([previousLayerSize, layerSize]), name="w" + str(count))
                with tf.name_scope('biases'):
                    b = tf.Variable(tf.random_normal([layerSize]), name="b" + str(count))

                preact = tf.add(tf.matmul(previousLayer, w), b, name="preactivation" + str(count))
                act = tf.nn.relu(preact, name="relu" + str(count))
                with tf.name_scope('dropout'):
                    layer = tf.nn.dropout(act, neuronDropoutRate, name='dropout'+str(count))
                tf.summary.histogram("weights",w)
                tf.summary.histogram("biases", b)
                #tf.summary.histogram("preactivation", preact)
                tf.summary.histogram("activation", act)
            previousLayer = layer
            previousLayerSize = layerSize

        with tf.name_scope('output'):
            with tf.name_scope('weights'):
                w = tf.Variable(tf.random_normal([previousLayerSize, 2]), name="w" + str(count))
            preact = tf.matmul(previousLayer, w, name="activation" + str(count))
            # act = tf.nn.softmax(preact)
            previousLayer = preact
            tf.summary.histogram("weights", w)
            tf.summary.histogram("biases", b)
            tf.summary.histogram("activation", preact)
            #tf.summary.histogram("activation", act)

        output = previousLayer


        with tf.name_scope('measures'):
            if ( lf == 'weighted'):
                lossFunction = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(logits=output, targets=truthYPlaceholder, pos_weight=10), name="xent")
            elif ( lf == 'softmaxxent' ):
                lossFunction = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=truthYPlaceholder), name="xent")
            elif ( lf == 'sigmoidxent' ):
                lossFunction = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=output, labels=truthYPlaceholder), name="xent")

            tf.summary.scalar('lossFunction', lossFunction)

            #f1ScorePerUser = tf.placeholder('float', [1], name="f1ScorePerUser")
            #tf.summary.scalar('f1ScorePerUser', f1ScorePerUser)

            #set the prediction and truth values
            prediction = tf.argmax(output,1, name="prediction")
            tf.summary.histogram('prediction', prediction)

            trainingtruth = tf.argmax(truthYPlaceholder,1, name="truth")
            tf.summary.histogram('trainingtruth', trainingtruth)

            correct = tf.equal(prediction, trainingtruth, name="correct")

            with tf.name_scope('accuracy'):
                accuracy = tf.reduce_mean(tf.cast(correct, 'float'), name="accuracyMeasure")

            tf.summary.scalar("accuracy", accuracy)

        # create the optimizer (called training step)
        #
        with tf.name_scope('train'):
            if ( optimizerName == 'adagrad' ):
                train_step = tf.train.AdagradOptimizer(learning_rate=lr).minimize(lossFunction)
            elif ( optimizerName == 'adam'):
                train_step = tf.train.AdamOptimizer(learning_rate=lr).minimize(lossFunction)
            elif (optimizerName == 'gradientDescent'):
                train_step = tf.train.GradientDescentOptimizer(learning_rate=lr).minimize(lossFunction)


        # sess = tf.Session()
        # sess.run(tf.global_variables_initializer())
        # sess.run([train_step, cost], feed_dict={inputPlaceholder: x_train , truthYPlaceholder: y_train})
        # #train_step.run(feed_dict={inputPlaceholder: x_train, truthYPlaceholder: y_train})
        # xxx = sess.run(accuracy.eval({inputPlaceholder: x_train , truthYPlaceholder: y_train}))


        with tf.Session() as s:
            tf.set_random_seed(42)
            tf.global_variables_initializer().run()

            batchSize = 100
            curStep = 1

            if bSaveTFSummary == True:
                merged_summary = tf.summary.merge_all()
                file_writer = tf.summary.FileWriter('data/tflogs/' + hyperParamStr)
                file_writer.add_graph(s.graph)

            #graph = pd.DataFrame(columns=['index', 'accuracy', 'error']);
            for curStep in range(1,int(len(y_train)/batchSize)+1):
                batch_x = x_train[(curStep-1)*batchSize:(curStep)*batchSize]
                batch_y = y_train[(curStep-1)*batchSize:(curStep)*batchSize]

                # train
                feed_dict = {inputPlaceholder: batch_x , truthYPlaceholder: batch_y, neuronDropoutRate : dropoutRate}

                if ( bSaveTFSummary == True and curStep % 5 == 0 ):
                    summaryData = s.run(merged_summary,feed_dict)
                    file_writer.add_summary(summaryData,curStep)

                train_step.run(feed_dict=feed_dict)

                feed_dict = {inputPlaceholder: batch_x , truthYPlaceholder: batch_y, neuronDropoutRate : 1.0}
                acc, err = s.run([accuracy, lossFunction], feed_dict=feed_dict)
                if (curStep % 100 == 0):
                    print('Accuracy on self: %s error:%s ' % (str(acc), str(err)))

                # outputValues = prediction.eval(feed_dict={inputPlaceholder: batch_x, neuronDropoutRate: 1.0})

                #graph.loc[len(graph)] = [curStep, acc, err]


            # print('Generating graphic...')
            # fig, ax = plt.subplots()
            # ax2 = ax.twinx()
            #
            # sns.pointplot(x='index', y='accuracy', data=graph, color='blue', ax=ax, label = 'Accuracy', scale=0.2)
            # sns.pointplot(x='index', y='error', data=graph, color='green',ax=ax2, label = 'Error', scale=0.2)
            # #ax.legend()
            # plt.legend(loc='upper right')
            # labels = ax.get_xticklabels()
            # ax.set_xticklabels(labels, rotation=45)
            #
            # plt.xticks(rotation=45)
            # plt.show()

            x_test = test[features]
            y_test = test[['reordered']]
            #print('Accuracy on test:', accuracy.eval({inputPlaceholder: x_test , truthYPlaceholder: y_test}))

            #print(tf.reduce_mean(tf.cast(tf.equal(tf.argmax(output,1),tf.argmax(y_test)),'float')).eval(feed_dict={inputPlaceholder:x_test}))
            o = prediction.eval(feed_dict={inputPlaceholder:x_test, neuronDropoutRate : 1.0})

        df = pd.DataFrame(columns=('user_id', 'product_id', 'predy'))
        df['user_id'] = test['user_id']
        df['product_id'] = test['product_id']
        df['predy'] = o

        pcb.debugWithTimer("scoring prediction" + hyperParamStr)
        _, f1score = pcb.scorePrediction(df)
        pcb.debugWithTimer("done scoring")
        if  f1score > bestScore:
            bestDefinition = oneDefinition
            bestScore = f1score
            bestDF = df
        allScores[str(oneDefinition)] = f1score

    print( '********\n********\n********')
    print('best score:' + str(round(bestScore,5)) + ' with ' + str(bestDefinition) )
    print( '********\n********\n********')

    sortedScores = sorted(allScores.items(), key=operator.itemgetter(1))
    for (definition, score) in sortedScores:
        print("{:.5f}".format(score) + ':' +definition)

    return bestDF


def myThirdNN(train, test):

    # define hyperparameters

    allfeatures = ['orderfrequency', 'dayfrequency', 'department_id', 'aisle_id', 'days_without_product_order','eval_days_since_prior_order',
                   'numproductorders', 'totaluserorders','day_number_of_last_product_order', 'eval_order_dow', 'orderfreqoverratio', 'orderfreqlast5', 'orderfreqlast10',
       'orderfreqlast15', 'orderfreqlast20', 'orderfreqlast25',
       'orderfreqlast30', 'orderfreqlast35', 'orderfreqlast40',
       'orderfreqlast45', 'orderfreqlast50', 'orderfreqlast55',
       'orderfreqlast60', 'orderfreqlast65', 'orderfreqlast70',
       'orderfreqlast75', 'orderfreqlast80', 'orderfreqlast85',
       'orderfreqlast90', 'orderfreqlast95'
]

    features28 = ['orderfrequency', 'dayfrequency', 'days_without_product_order', 'department_id', 'aisle_id','eval_days_since_prior_order', 'numproductorders', 'totaluserorders', 'orderfreqoverratio', 'orderfreqlast5', 'orderfreqlast10',
       'orderfreqlast15', 'orderfreqlast20', 'orderfreqlast25',
       'orderfreqlast30', 'orderfreqlast35', 'orderfreqlast40',
       'orderfreqlast45', 'orderfreqlast50', 'orderfreqlast55',
       'orderfreqlast60' ]
    features9 = ['orderfrequency', 'dayfrequency', 'days_without_product_order', 'department_id', 'aisle_id','eval_days_since_prior_order', 'numproductorders', 'totaluserorders', 'orderfreqoverratio']
    features4 = ['orderfrequency', 'dayfrequency', 'days_without_product_order', 'department_id']
    features1 = ['orderfrequency']

    bSaveTFSummary = False
    if (len(sys.argv) > 2 and sys.argv[2] == '1' and not (pcb.are_we_running_in_debug_mode())):
        bSaveTFSummary = True  # don't save summary in debug mode (incomplete data) or when explicitly requested not
    else:
        bSaveTFSummary = False

    tf.set_random_seed(42)

    possibleOptimizers = ['adagrad', 'adam']
    possibleFeatures = [features28] # [features4, features9] #[features1, features4, features9]
    possibleDropoutRates =  [0.9] #[0.75, 0.9, 1.0 ]
    possibleNetworkLayerShapes = [ [20], [20,20], [50,20] ] #, [20,20,20], [100,50, 20] ]
    possibleNetworkLayerShapes = [ [20], [20,20], [30,20], [30,20,10], [30], [10], [15] ]
    possibleLearningRates = [0.1, 0.01]

    hyperParamExplorationDict = []
    for fdef in possibleFeatures:
        for nshape in possibleNetworkLayerShapes:
            for optimizerName in possibleOptimizers:
                for dr in possibleDropoutRates:
                    for lr in possibleLearningRates:
                        for vlr in [True]:
                            hyperParamExplorationDict.extend(
                            [
                                { 'features': fdef
                                  ,'hiddenLayerSizes' : nshape
                                  ,'dropoutRate' : dr
                                  ,'optimizerName' :optimizerName  # gradientDescent, adagrad, adam
                                  ,'lr' : lr
                                  ,'lf' : 'sigmoidxent'  # sigmoidxent, softmaxxent, weighted
                                  ,'vlr': vlr
                                  ,'extra' : '-balancedinput-vlr' + str(int(vlr))
                                }
                            ]
                            )

    # just one, the best
    # hyperParamExplorationDict = [{'features': ['orderfrequency', 'dayfrequency', 'days_without_product_order', 'department_id', 'aisle_id', 'eval_days_since_prior_order', 'numproductorders', 'totaluserorders', 'eval_days_since_prior_order'], 'hiddenLayerSizes': [50, 20], 'dropoutRate': 0.9, 'optimizerName': 'adam', 'lr': 0.001, 'lf': 'softmaxxent', 'extra': '-balancedinput'}]
    # hyperParamExplorationDict = [{'features': ['orderfrequency', 'dayfrequency', 'days_without_product_order', 'department_id'], 'hiddenLayerSizes': [20, 20], 'dropoutRate': 0.75, 'optimizerName': 'adam', 'lr': 0.001, 'lf': 'softmaxxent','extra': '-balancedinput'}, {'features': ['orderfrequency', 'dayfrequency', 'days_without_product_order', 'department_id', 'aisle_id', 'eval_days_since_prior_order', 'numproductorders', 'totaluserorders', 'eval_days_since_prior_order'], 'hiddenLayerSizes': [50, 20], 'dropoutRate': 0.9, 'optimizerName': 'adam', 'lr': 0.001, 'lf': 'softmaxxent', 'extra': '-balancedinput'}]
    # hyperParamExplorationDict = [{'features': ['orderfrequency', 'dayfrequency', 'days_without_product_order', 'department_id', 'aisle_id','eval_days_since_prior_order', 'numproductorders', 'totaluserorders'],'hiddenLayerSizes': [50, 20], 'dropoutRate': 0.9, 'optimizerName': 'adam', 'lr': 0.001, 'lf': 'sigmoidxent','extra': '-balancedinput'}]
    # 0.32838
    # 10k 0.33857: {'features': ['orderfrequency', 'dayfrequency', 'days_without_product_order', 'department_id', 'aisle_id',
    #                        'eval_days_since_prior_order', 'numproductorders', 'totaluserorders', 'orderfreqoverratio',
    #                        'orderfreqlast5', 'orderfreqlast10', 'orderfreqlast15', 'orderfreqlast20', 'orderfreqlast25',
    #                        'orderfreqlast30', 'orderfreqlast35', 'orderfreqlast40', 'orderfreqlast45',
    #                        'orderfreqlast50', 'orderfreqlast55', 'orderfreqlast60', 'orderfreqlast65',
    #                        'orderfreqlast70', 'orderfreqlast75', 'orderfreqlast80', 'orderfreqlast85',
    #                        'orderfreqlast90', 'orderfreqlast95'], 'hiddenLayerSizes': [10], 'dropoutRate': 0.9,
    #           'optimizerName': 'adam', 'lr': 0.1, 'lf': 'sigmoidxent', 'extra': '-balancedinput'}

    # hyperParamExplorationDict = [{'method': 'fourthNN-balancedInput', 'features': ['orderfrequency', 'dayfrequency', 'days_without_product_order', 'eval_days_since_prior_order', 'numproductorders', 'totaluserorders', 'orderfreqoverratio', 'orderfreqlast5', 'orderfreqlast10', 'orderfreqlast15', 'orderfreqlast20', 'orderfreqlast25', 'orderfreqlast30', 'orderfreqlast35', 'orderfreqlast40', 'orderfreqlast45', 'orderfreqlast50', 'orderfreqlast55', 'orderfreqlast60'], 'hiddenLayerSizes': [30, 10], 'dropoutRate': 0.8, 'optimizerName': 'adam', 'lr': 0.01, 'lf': 'sigmoidxent', 'vlr': False, 'extra': '', 'adsize': 30}]

    bestScore = 0
    bestDefinition = {}
    bestDF = None
    allScores = {}

    lastfeatures = None
    x_train = None
    y_train = None


    defCounter = 0
    for oneDefinition in hyperParamExplorationDict : #placeholder for hyperparam exploration
        defCounter = defCounter + 1
        print("evaluation definition: (" + str(defCounter) + "/" + str(len(hyperParamExplorationDict)) + ")")
        features = oneDefinition['features']
        hiddenLayerSizes =oneDefinition['hiddenLayerSizes']
        dropoutRate =oneDefinition['dropoutRate']
        optimizerName =oneDefinition['optimizerName']
        initiallr=oneDefinition['lr']
        lf =oneDefinition['lf']
        extra = oneDefinition['extra']
        variableLearningRate = oneDefinition['vlr']

        nbfeatures = len(features)

        if lastfeatures != features:
            x_train = train[features]
            y_train = train['reordered']
            y_train = list([i] for i in y_train)
            lastfeatures = features


        tf.reset_default_graph()

        # define input and output
        with tf.name_scope('input'):
            inputPlaceholder = tf.placeholder('float', [None, nbfeatures], name='myWonderfullInput')
            truthYPlaceholder = tf.placeholder('float', [None, 1], name="mylabels")
            tflr = tf.placeholder(tf.float32)

        with tf.name_scope('dropout_rate'):
            neuronDropoutRate = tf.placeholder('float')
            tf.summary.scalar('dropout_keep_probability', neuronDropoutRate)


        hyperParamStr = makeHyperParamString(hiddenLayerSizes, dropoutRate, nbfeatures, optimizerName, initiallr, lf, extra)

        previousLayer = inputPlaceholder
        previousLayerSize = nbfeatures
        count = 0
        for layerSize in hiddenLayerSizes:
            count = count+1
            with tf.name_scope('hiddenLayer' + str(count)):
                with tf.name_scope('weights'):
                    w = tf.Variable(tf.random_normal([previousLayerSize, layerSize]), name="w" + str(count))
                with tf.name_scope('biases'):
                    b = tf.Variable(tf.random_normal([layerSize]), name="b" + str(count))

                preact = tf.add(tf.matmul(previousLayer, w), b, name="preactivation" + str(count))
                act = tf.nn.relu(preact, name="relu" + str(count))
                with tf.name_scope('dropout'):
                    layer = tf.nn.dropout(act, neuronDropoutRate, name='dropout'+str(count))
                tf.summary.histogram("weights",w)
                tf.summary.histogram("biases", b)
                #tf.summary.histogram("preactivation", preact)
                tf.summary.histogram("activation", act)
            previousLayer = layer
            previousLayerSize = layerSize

        with tf.name_scope('output'):
            with tf.name_scope('weights'):
                w = tf.Variable(tf.random_normal([previousLayerSize, 1]), name="w" + str(count))
            preact = tf.matmul(previousLayer, w, name="activation" + str(count))

#            act = tf.sigmoid(preact, name="prediction")
            previousLayer = preact
            tf.summary.histogram("weights", w)
            tf.summary.histogram("biases", b)
            tf.summary.histogram("activation", preact)
            #tf.summary.histogram("activation", act)

        output = previousLayer

        with tf.name_scope('measures'):
            if ( lf == 'weighted'):
                lossFunction = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(logits=output, targets=truthYPlaceholder, pos_weight=10), name="xent")
            elif ( lf == 'softmaxxent' ):
                lossFunction = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=truthYPlaceholder), name="xent")
            elif ( lf == 'sigmoidxent' ):
                lossFunction = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=output, labels=truthYPlaceholder), name="xent")

            tf.summary.scalar('lossFunction', lossFunction)

            #f1ScorePerUser = tf.placeholder('float', [1], name="f1ScorePerUser")
            #tf.summary.scalar('f1ScorePerUser', f1ScorePerUser)

            #set the prediction and truth values
            prediction = tf.sigmoid(output, name="prediction")
            tf.summary.histogram('prediction', prediction)

            #trainingtruth = tf.argmax(truthYPlaceholder,1, name="truth")
            #tf.summary.histogram('trainingtruth', trainingtruth)

            error = tf.abs(tf.add(prediction, -truthYPlaceholder), name="correct")

            with tf.name_scope('accuracy'):
                accuracy = tf.reduce_mean(tf.cast(error, 'float'), name="accuracyMeasure")

            tf.summary.scalar("accuracy", accuracy)


        batch = tf.Variable(0, trainable=False)
        batchSize = 100

        # learning_rate = tf.train.exponential_decay(
        #     lr,  # Base learning rate.
        #     global_step=batch * batchSize,  # Current index into the dataset.
        #     decay_steps=1000,  # Decay step.
        #     decay_rate=0.95,  # Decay rate.
        #     staircase=True)

        # create the optimizer (called training step)
        #
        with tf.name_scope('train'):
            if ( optimizerName == 'adagrad' ):
                train_step = tf.train.AdagradOptimizer(learning_rate=tflr).minimize(lossFunction,
                                                             global_step=batch)
            elif ( optimizerName == 'adam'):
                train_step = tf.train.AdamOptimizer(learning_rate=tflr).minimize(lossFunction,
                                                             global_step=batch)
            elif (optimizerName == 'gradientDescent'):
                train_step = tf.train.GradientDescentOptimizer(learning_rate=tflr).minimize(lossFunction,
                                                             global_step=batch)

        with tf.Session() as s:
            tf.set_random_seed(42)
            tf.global_variables_initializer().run()

            curStep = 1

            if bSaveTFSummary == True:
                merged_summary = tf.summary.merge_all()
                file_writer = tf.summary.FileWriter('data/tflogs/' + hyperParamStr)
                file_writer.add_graph(s.graph)

            #graph = pd.DataFrame(columns=['index', 'accuracy', 'error']);
            for curStep in range(1,int(len(y_train)/batchSize)+1):
                batch_x = x_train[(curStep-1)*batchSize:(curStep)*batchSize]
                batch_y = y_train[(curStep-1)*batchSize:(curStep)*batchSize]

                learning_rate = 1
                if variableLearningRate == True:
                    max_learning_rate = initiallr #0.001
                    min_learning_rate = initiallr/100
                    decay_speed = 1000.0  # 0.003-0.0001-2000=>0.9826 done in 5000 iterations
                    learning_rate = min_learning_rate + (max_learning_rate - min_learning_rate) * math.exp(-(curStep*batchSize) / decay_speed)
                else:
                    learning_rate = initiallr


                # train
                feed_dict = {inputPlaceholder: batch_x , truthYPlaceholder: batch_y, neuronDropoutRate : dropoutRate, tflr: learning_rate}

                if ( bSaveTFSummary == True and curStep % 5 == 0 ):
                    summaryData = s.run(merged_summary,feed_dict)
                    file_writer.add_summary(summaryData,curStep)

                train_step.run(feed_dict=feed_dict)

                feed_dict = {inputPlaceholder: batch_x , truthYPlaceholder: batch_y, neuronDropoutRate : 1.0}
                acc, loss = s.run([accuracy, lossFunction], feed_dict=feed_dict)
                if (curStep % 100 == 0):
                    print('Accuracy on self: %s loss:%s ' % (str(acc), str(loss)))

            x_test = test[features]
            y_test = test[['reordered']]
            #print('Accuracy on test:', accuracy.eval({inputPlaceholder: x_test , truthYPlaceholder: y_test}))

            #print(tf.reduce_mean(tf.cast(tf.equal(tf.argmax(output,1),tf.argmax(y_test)),'float')).eval(feed_dict={inputPlaceholder:x_test}))
            o = prediction.eval(feed_dict={inputPlaceholder:x_test, neuronDropoutRate : 1.0})

        # cast the predictions as integers
        xx = list(round(i[0]) for i in o)

        df = pd.DataFrame(columns=('user_id', 'product_id', 'predy'))
        df['user_id'] = test['user_id']
        df['product_id'] = test['product_id']
        df['predy'] = xx

        pcb.debugWithTimer("scoring prediction" + hyperParamStr)
        _, f1score = pcb.scorePrediction(df)
        pcb.debugWithTimer("done scoring")
        if  f1score > bestScore:
            bestDefinition = oneDefinition
            bestScore = f1score
            bestDF = df
        allScores[str(oneDefinition)] = f1score

        # save score to file:
        f = open('NNscores.txt', 'a')
        f.write("{:.5f}".format(f1score) + ':' + hyperParamStr + ':' + str(oneDefinition) + "\n")
        f.close()

    f = open('NNscores.txt', 'a')
    f.write( '\n********\n********\n********')
    f.write( '\nbest score:' + "{:.5f}".format(bestScore) + ' with ' + str(bestDefinition) )
    f.write( '\n********\n********\n********')
    print( '********\n********\n********')
    print('best score:' + "{:.5f}".format(bestScore) + ' with ' + str(bestDefinition) )
    print( '********\n********\n********')

    sortedScores = sorted(allScores.items(), key=operator.itemgetter(1))
    for (definition, score) in sortedScores:
        print("{:.5f}".format(score) + ':' +definition)
        f.write("\n{:.5f}".format(score) + ':' +definition)
    f.close()

    return bestDF



# a useful function that takes an input and what size we want the output
# to be, and multiples the input by a weight matrix plus bias (also creating
# these variables)
def linear(input_, output_size, name, init_bias=0.0):
    shape = input_.get_shape().as_list()
    with tf.variable_scope(name):
        W = tf.get_variable("weight_matrix", [shape[-1], output_size], tf.float32, tf.random_normal_initializer(stddev=1.0 / math.sqrt(shape[-1])))
    if init_bias is None:
        return tf.matmul(input_, W)
    with tf.variable_scope(name):
        b = tf.get_variable("bias", [output_size], initializer=tf.constant_initializer(init_bias))
    return tf.matmul(input_, W) + b

def mySecondNN(train, test):

    # define hyperparameters

    allfeatures = ['orderfrequency', 'dayfrequency', 'department_id', 'aisle_id', 'days_without_product_order','eval_days_since_prior_order',
                   'numproductorders', 'totaluserorders','day_number_of_last_product_order', 'eval_order_dow']

    features9 = ['orderfrequency', 'dayfrequency', 'days_without_product_order', 'department_id', 'aisle_id','eval_days_since_prior_order', 'numproductorders', 'totaluserorders','eval_days_since_prior_order']
    features4 = ['orderfrequency', 'dayfrequency', 'days_without_product_order', 'department_id']
    features1 = ['orderfrequency']

    bSaveTFSummary = False
    if (len(sys.argv) > 2 and sys.argv[2] == '1' and not (pcb.are_we_running_in_debug_mode())):
        bSaveTFSummary = True  # don't save summary in debug mode (incomplete data) or when explicitly requested not
    else:
        bSaveTFSummary = False

    tf.set_random_seed(42)

    x_train = train['orderfrequency']
    y_train = np.column_stack(
        (list(float(not (i)) for i in train['reordered']),
         list(float(i) for i in train['reordered'])))

    tf.reset_default_graph()

    # define input and output
    with tf.name_scope('input'):
        inputTSPlaceholder = tf.placeholder(tf.float32, [None, 100, 2]) # history is 100 steps back at most, and expressed as boolean
        labels = tf.placeholder(tf.float, [None, 2])

    lstm_cell_1 = tf.contrib.rnn.LSTMCell(30)
    lstm_cell_2 = tf.contrib.rnn.LSTMCell(30)
    multi_lstm_cells = tf.contrib.rnn.MultiRNNCell(cells=[lstm_cell_1, lstm_cell_2], state_is_tuple=True)

    # define the op that runs the LSTM, across time, on the data
    _, final_state = tf.nn.dynamic_rnn(multi_lstm_cells, inputTSPlaceholder, dtype=tf.float32)

    #output layer
    output = linear(final_state[-1][-1], 1, name="output")

    # define cross entropy loss function
    loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=output, labels=labels)
    loss = tf.reduce_mean(loss)

    # round our actual probabilities to compute error
    prob = tf.nn.sigmoid(output)
    prediction = tf.to_float(tf.greater_equal(prob, 0.5))
    pred_err = tf.to_float(tf.not_equal(prediction, labels))
    pred_err = tf.reduce_sum(pred_err)
    pred_acc = tf.equal(prediction, labels)

    # define our optimizer to minimize the loss
    optimizer = tf.train.AdamOptimizer().minimize(loss)

    with tf.InteractiveSession() as sess:

        batchSize = 100
        curStep = 1

        for curStep in range(1, int(len(y_train) / batchSize) + 1):
            batch_x = x_train[(curStep - 1) * batchSize:(curStep) * batchSize]
            batch_y = y_train[(curStep - 1) * batchSize:(curStep) * batchSize]

            data = {input:batch_x, labels:batch_y}
            _, loss_value_train, error_value_train = sess.run([optimizer, loss, pred_err], feed_dict=data)

            data_testing = {}
            loss_value_test, error_value_test = sess.run([loss, pred_err], feed_dict=data_testing)

    o = prediction.eval(feed_dict={input:x_test})

    df = pd.DataFrame(columns=('user_id', 'product_id', 'predy'))
    df['user_id'] = test['user_id']
    df['product_id'] = test['product_id']
    df['predy'] = o

    _, f1score = pcb.scorePrediction(df)

    return df



def combinePredictions(predictionArray):


    df = pd.DataFrame(columns=('user_id', 'product_id', 'predy'))
    df['user_id'] = test['user_id']
    df['product_id'] = test['product_id']
    df['predy'] = o

    return df
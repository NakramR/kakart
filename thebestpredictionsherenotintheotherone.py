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

def generateRandomPrediction():
    randpred = pd.DataFrame(columns=('user_id', 'product_id', 'predy'))
    pcb.debugWithTimer('reading distinct user products')
    userpriorproducts = pcb.userproductstats

    pcb.debugWithTimer('iterating over prior products')
    temp = []
    for index, x in userpriorproducts.iterrows():
        newline = { 'user_id': x['user_id'], 'product_id': x['product_id'], 'predy': (random.random() > 0.5) }
        temp.append(newline)

    randpred = randpred.append(temp)

    return randpred

def predictOverFrequencyThreshold(threshold):
    userpriorproducts = pcb.userproductstats

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

def tfvariable_summaries(var):
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)

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
    possibleNetworkLayerShapes = [ [20], [20,20], [50,20], [20,20,20], [100,50, 20] ]

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
    hyperParamExplorationDict = [{'features': ['orderfrequency', 'dayfrequency', 'days_without_product_order', 'department_id'], 'hiddenLayerSizes': [20, 20], 'dropoutRate': 0.75, 'optimizerName': 'adam', 'lr': 0.001, 'lf': 'softmaxxent','extra': '-balancedinput'}, {'features': ['orderfrequency', 'dayfrequency', 'days_without_product_order', 'department_id', 'aisle_id', 'eval_days_since_prior_order', 'numproductorders', 'totaluserorders', 'eval_days_since_prior_order'], 'hiddenLayerSizes': [50, 20], 'dropoutRate': 0.9, 'optimizerName': 'adam', 'lr': 0.001, 'lf': 'softmaxxent', 'extra': '-balancedinput'}]
    #
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

    possibleOptimizers = ['adagrad', 'adam']
    possibleFeatures = [features4, features9] #[features1, features4, features9]
    possibleDropoutRates = [0.75, 0.9, 1.0 ]
    possibleNetworkLayerShapes = [ [20], [20,20], [50,20], [20,20,20], [100,50, 20] ]

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
    hyperParamExplorationDict = [{'features': ['orderfrequency', 'dayfrequency', 'days_without_product_order', 'department_id'], 'hiddenLayerSizes': [20, 20], 'dropoutRate': 0.75, 'optimizerName': 'adam', 'lr': 0.001, 'lf': 'softmaxxent','extra': '-balancedinput'}, {'features': ['orderfrequency', 'dayfrequency', 'days_without_product_order', 'department_id', 'aisle_id', 'eval_days_since_prior_order', 'numproductorders', 'totaluserorders', 'eval_days_since_prior_order'], 'hiddenLayerSizes': [50, 20], 'dropoutRate': 0.9, 'optimizerName': 'adam', 'lr': 0.001, 'lf': 'softmaxxent', 'extra': '-balancedinput'}]
    #
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




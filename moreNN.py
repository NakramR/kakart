import math
import operator
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import tensorflow as tf

import petitchatbase as pcb


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


def fullConnectedLayer(layerSize, name, depth, previousLayer, previousLayerSize ):
    global neuronDropoutRate

    with tf.name_scope(name + str(depth)):
        with tf.name_scope('weights'):
            w = tf.Variable(tf.random_normal([previousLayerSize, layerSize]), name="w" + str(depth))
        with tf.name_scope('biases'):
            b = tf.Variable(tf.random_normal([layerSize]), name="b" + str(depth))

        preact = tf.add(tf.matmul(previousLayer, w), b, name="preactivation" + str(depth))
        # TODO: add batch normalization after the matmul
        act = tf.nn.relu(preact, name="relu" + str(depth))
        with tf.name_scope('dropout'):
            layer = tf.nn.dropout(act, neuronDropoutRate, name='dropout' + str(depth))
        tf.summary.histogram("weights", w)
        tf.summary.histogram("biases", b)
        tf.summary.histogram("activation", act)
    previousLayer = layer
    previousLayerSize = layerSize

    return layer

def batchnorm(Ylogits, Offset, Scale, is_test, iteration):
    exp_moving_avg = tf.train.ExponentialMovingAverage(0.998, iteration) # adding the iteration prevents from averaging across non-existing iterations
    bnepsilon = 1e-5
    mean, variance = tf.nn.moments(Ylogits, [0])
    update_moving_everages = exp_moving_avg.apply([mean, variance])
    m = tf.cond(is_test, lambda: exp_moving_avg.average(mean), lambda: mean)
    v = tf.cond(is_test, lambda: exp_moving_avg.average(variance), lambda: variance)
    Ybn = tf.nn.batch_normalization(Ylogits, m, v, Offset, Scale, bnepsilon)
    return Ybn, update_moving_everages

uniqueproductperusercache = []
fulluserprodscache = {}
booltruthcache = {}

def fasterScoreProduction(predictionperitem):
    global  uniqueproductperusercache, fulluserprodscache, booltruthcache
    usercount = 0
    sumf1 = 0.0
    sumf1x = 0.0

    if predictionperitem is None:
        return 0,0

    # remove all negative predictions
    myprediction = predictionperitem[predictionperitem['predy'] == True]
    myprediction = myprediction.groupby('user_id')['product_id'].apply(list)

    # get all prior user products
    if ( len(uniqueproductperusercache) == 0 ):
        uniqueproductperuser = pcb.userProductStats.groupby('user_id')['product_id'].unique() #past products only
        uniqueproductperusercache = uniqueproductperuser
    else:
        uniqueproductperuser = uniqueproductperusercache

    # count people who have no prediction
    usercount = len(predictionperitem['user_id'].unique()) - len(myprediction)

    #iterate on users (index is user_id)
    for index, x in pcb.truthperuser.iteritems():

        if index in myprediction:
            usercount = usercount + 1

            # get the full product list, including entirely new products that were not present in training data
            if fulluserprodscache.__contains__(index):
                fulluserprods = fulluserprodscache[index]
            else:
                fulluserprods = set().union(list(uniqueproductperuser[index]),list(myprediction[index]))
                fulluserprodscache[index] = fulluserprods

            bTruth = []
            if booltruthcache.__contains__(index):
                bTruth = booltruthcache[index]
            else:
                bTruth = list(i in pcb.truthperuser[index] for i in fulluserprods)
                booltruthcache[index] = bTruth

            # get a boolean match between truth & full product list
            # get a boolean match between prediction & full product list
            bPred = list(i in myprediction[index] for i in fulluserprods)

            sumf1x = sumf1x + sklearn.metrics.f1_score(bTruth, bPred)

    if usercount != 0:
        sumf1 = sumf1 / usercount
        sumf1x = sumf1x / usercount
        print(" Scoring sklearn.f1:", end='')
        print(sumf1x)
    else:
        print("No user, no predictions. Pbbbbbt")

    return sumf1, sumf1x


def myFourthNN(train, holdout, test, usePriorResultFile=True):
    global neuronDropoutRate
    # define hyperparameters

    allfeatures = ['orderfrequency', 'dayfrequency', 'department_id', 'aisle_id', 'days_without_product_order','eval_days_since_prior_order',
                   'numproductorders', 'totaluserorders','day_number_of_last_product_order', 'eval_order_dow', 'orderfreqoverratio', 'orderfreqlast5', 'orderfreqlast10',
       'orderfreqlast15', 'orderfreqlast20', 'orderfreqlast25',
       'orderfreqlast30', 'orderfreqlast35', 'orderfreqlast40',
       'orderfreqlast45', 'orderfreqlast50', 'orderfreqlast55',
       'orderfreqlast60', 'orderfreqlast65', 'orderfreqlast70',
       'orderfreqlast75', 'orderfreqlast80', 'orderfreqlast85',
       'orderfreqlast90', 'orderfreqlast95','reordersperuser', 'ordertoreorderfreq'
]
#(49683, 49678, 49680, 49677)
    features28 = ['orderfrequency', 'dayfrequency', 'days_without_product_order', 'eval_days_since_prior_order', 'numproductorders', 'totaluserorders', 'orderfreqoverratio', 'orderfreqlast5', 'orderfreqlast10',
       'orderfreqlast15', 'orderfreqlast20', 'orderfreqlast25',
       'orderfreqlast30', 'orderfreqlast35', 'orderfreqlast40',
       'orderfreqlast45', 'orderfreqlast50', 'orderfreqlast55',
       'orderfreqlast60', 'reordersperuser','ordertoreorderfreq']
    features9 = ['orderfrequency', 'dayfrequency', 'days_without_product_order', 'department_id', 'aisle_id','eval_days_since_prior_order', 'numproductorders', 'totaluserorders', 'orderfreqoverratio','reordersperuser', 'ordertoreorderfreq']
    features4 = ['orderfrequency', 'dayfrequency', 'days_without_product_order', 'department_id']
    features1 = ['orderfrequency']

    bSaveTFSummary = False
    if (len(sys.argv) > 2 and sys.argv[2] == '1' and not (pcb.are_we_running_in_debug_mode())):
        bSaveTFSummary = True  # don't save summary in debug mode (incomplete data) or when explicitly requested not
    else:
        bSaveTFSummary = False

    tf.set_random_seed(42)

    possibleOptimizers = ['adam']
    possibleFeatures = [features28, features9] # [features4, features9] #[features1, features4, features9]
    possibleDropoutRates =  [0.8] #[0.75, 0.9, 1.0 ]
    possibleNetworkLayerShapes = [ [5], [15], [30], [30,10], [20,20,20], [50,20], [100,50, 20] ]
#    possibleNetworkLayerShapes = [ [20], [20,20], [30,20], [30,20,10], [30], [10], [15] ]
    possibleLearningRates = [0.005, 0.001]

    hyperParamExplorationDict = []
    for fdef in possibleFeatures:
        for nshape in possibleNetworkLayerShapes:
            for optimizerName in possibleOptimizers:
                for dr in possibleDropoutRates:
                    for lr in possibleLearningRates:
                        for vlr in [False]:
                            for adsize in [15]: #[60, 30, 10]:
                                for threshold in [0.8, 0.7, 0.6, 0.5]:
                                    hyperParamExplorationDict.extend(
                                    [
                                        { 'method': 'fourthNN-balancedInput-thresh'
                                          ,'features': fdef
                                          ,'hiddenLayerSizes' : nshape
                                          ,'dropoutRate' : dr
                                          ,'optimizerName' :optimizerName  # gradientDescent, adagrad, adam
                                          ,'lr' : lr
                                          ,'lf' : 'sigmoidxent'  # sigmoidxent, softmaxxent, weighted
                                          ,'vlr': vlr
                                          ,'extra' : ''
                                          ,'adsize' : adsize
                                          ,'threshold': threshold
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

    # hyperParamExplorationDict = [{'features': ['orderfrequency', 'dayfrequency', 'days_without_product_order', 'department_id', 'aisle_id',
    #                        'eval_days_since_prior_order', 'numproductorders', 'totaluserorders', 'orderfreqoverratio',
    #                        'orderfreqlast5', 'orderfreqlast10', 'orderfreqlast15', 'orderfreqlast20', 'orderfreqlast25',
    #                        'orderfreqlast30', 'orderfreqlast35', 'orderfreqlast40', 'orderfreqlast45',
    #                        'orderfreqlast50', 'orderfreqlast55', 'orderfreqlast60', 'orderfreqlast65',
    #                        'orderfreqlast70', 'orderfreqlast75', 'orderfreqlast80', 'orderfreqlast85',
    #                        'orderfreqlast90', 'orderfreqlast95'], 'hiddenLayerSizes': [10], 'dropoutRate': 0.9,
    #           'optimizerName': 'adam', 'lr': 0.1, 'lf': 'sigmoidxent', 'extra': '-balancedinput', 'vlr' : False}]

    bestScore = -1
    bestDefinition = {}
    bestDF = None
    bestDFHoldout = None
    allScores = {}

    lastfeatures = None
    x_train = None
    y_train = None

    # hyperParamExplorationDict = [{'method': 'fourthNN-balancedInput', 'features': ['orderfrequency', 'dayfrequency', 'days_without_product_order', 'eval_days_since_prior_order', 'numproductorders', 'totaluserorders', 'orderfreqoverratio', 'orderfreqlast5', 'orderfreqlast10', 'orderfreqlast15', 'orderfreqlast20', 'orderfreqlast25', 'orderfreqlast30', 'orderfreqlast35', 'orderfreqlast40', 'orderfreqlast45', 'orderfreqlast50', 'orderfreqlast55', 'orderfreqlast60','reordersperuser', 'ordertoreorderfreq'], 'hiddenLayerSizes': [30, 10], 'dropoutRate': 0.8, 'optimizerName': 'adam', 'lr': 0.01, 'lf': 'sigmoidxent', 'vlr': False, 'extra': '', 'adsize': 30}]
    hyperParamExplorationDict = [{'method': 'fourthNN-balancedInput-thresh', 'features': ['orderfrequency', 'dayfrequency', 'days_without_product_order', 'eval_days_since_prior_order', 'numproductorders', 'totaluserorders', 'orderfreqoverratio', 'orderfreqlast5', 'orderfreqlast10', 'orderfreqlast15', 'orderfreqlast20', 'orderfreqlast25', 'orderfreqlast30', 'orderfreqlast35', 'orderfreqlast40', 'orderfreqlast45', 'orderfreqlast50', 'orderfreqlast55', 'orderfreqlast60', 'reordersperuser', 'ordertoreorderfreq'], 'hiddenLayerSizes': [30], 'dropoutRate': 0.8, 'optimizerName': 'adam', 'lr': 0.005, 'lf': 'sigmoidxent', 'vlr': False, 'extra': '', 'adsize': 15, 'threshold': 0.7}]
    usePriorResultFile = False  #{'method': 'fourthNN-balancedInput-thresh', 'features': ['orderfrequency', 'dayfrequency', 'days_without_product_order', 'eval_days_since_prior_order', 'numproductorders', 'totaluserorders', 'orderfreqoverratio', 'orderfreqlast5', 'orderfreqlast10', 'orderfreqlast15', 'orderfreqlast20', 'orderfreqlast25', 'orderfreqlast30', 'orderfreqlast35', 'orderfreqlast40', 'orderfreqlast45', 'orderfreqlast50', 'orderfreqlast55', 'orderfreqlast60', 'reordersperuser', 'ordertoreorderfreq'], 'hiddenLayerSizes': [50, 20], 'dropoutRate': 0.8, 'optimizerName': 'adam', 'lr': 0.005, 'lf': 'sigmoidxent', 'vlr': False, 'extra': '', 'adsize': 15, 'threshold': 0.6}
    showPlots = False
    tmpCheckEarlyStopping = True

    acclossrecord = np.zeros((1,3))
    defCounter = 0
    for oneDefinition in hyperParamExplorationDict : #placeholder for hyperparam exploration
        defCounter = defCounter + 1
        print("evaluation definition: (" + str(defCounter) + "/" + str(len(hyperParamExplorationDict)) + ")")

        if usePriorResultFile and pcb.isInPriorRun(str(oneDefinition)):
           continue

        features = oneDefinition['features']
        hiddenLayerSizes = oneDefinition['hiddenLayerSizes']
        dropoutRate = oneDefinition['dropoutRate']
        optimizerName = oneDefinition['optimizerName']
        initiallr= oneDefinition['lr']
        lf = oneDefinition['lf']
        extra = oneDefinition['extra']
        variableLearningRate = oneDefinition['vlr']
        adsize = oneDefinition['adsize']
        threshold = oneDefinition['threshold']

        nbfeatures = len(features)

        if lastfeatures != features:
            x_train = train[features]
            x_dep = train['department_id']
            x_aisle = train['aisle_id']
            y_train = train['reordered']
            y_train = list([i] for i in y_train)
            lastfeatures = features

        hyperParamStr = makeHyperParamString(hiddenLayerSizes, dropoutRate, nbfeatures, optimizerName, initiallr, lf, extra + '-'+ str(adsize) + '-vlr' + str(variableLearningRate))


        tf.reset_default_graph()

        # define input and output
        with tf.name_scope('input'):
            inputPlaceholder = tf.placeholder('float', [None, nbfeatures], name='x_input')
            truthYPlaceholder = tf.placeholder('float', [None, 1], name="y_labels")
            tflr = tf.placeholder(tf.float32, name='learning_rate')
            neuronDropoutRate = tf.placeholder('float', name="dropout_rate")
            istest = tf.placeholder(tf.bool, name="is_test")
            currentiter = tf.placeholder(tf.int32, name="current_iteration")

            # add an aisle and department layer
            aislePlaceholder = tf.placeholder(tf.int32, [None], name="aisle")
            departmentPlaceholder = tf.placeholder(tf.int32, [None], name="department")


        numAisles = len(pcb.products['aisle_id'].unique())
        numDepartments = len(pcb.products['department_id'].unique())
        a_one = tf.one_hot(aislePlaceholder, numAisles, name="aisle-onehot")
        d_one = tf.one_hot(departmentPlaceholder, numDepartments, name="department-onehot")

        ad = tf.concat([a_one, d_one], 1, name="aisle_and_department")
        adx = fullConnectedLayer(adsize,"aisledep",1, ad, numAisles+numDepartments)

        allinput = tf.concat([inputPlaceholder, adx], 1, name="combined_inputs")
        previousLayer = allinput
        previousLayerSize = nbfeatures+adsize

        count = 0
        for layerSize in hiddenLayerSizes:
            count = count+1
            previousLayer = fullConnectedLayer(layerSize,"hidden",count,previousLayer,previousLayerSize)
            previousLayerSize = layerSize


        with tf.name_scope('output'):
            with tf.name_scope('weights'):
                w = tf.Variable(tf.random_normal([previousLayerSize, 1]), name="w" + str(count))
            preact = tf.matmul(previousLayer, w, name="activation" + str(count))

#            act = tf.sigmoid(preact, name="prediction")
            previousLayer = preact
            tf.summary.histogram("weights", w)
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

            #set the prediction and truth values
            prediction = tf.sigmoid(output, name="prediction")
            tf.summary.histogram('prediction', prediction)

            error = tf.abs(tf.add(prediction, -truthYPlaceholder), name="correct")

            with tf.name_scope('accuracy'):
                accuracy = tf.reduce_mean(tf.cast(error, 'float'), name="accuracyMeasure")

            tf.summary.scalar("accuracy", accuracy)


        global_step = tf.Variable(0, name='global_step', trainable=False)

        # create the optimizer (called training step)
        #
        with tf.name_scope('train'):
            if ( optimizerName == 'adagrad' ):
                train_step = tf.train.AdagradOptimizer(learning_rate=tflr).minimize(lossFunction,
                                                             global_step=global_step )
            elif ( optimizerName == 'adam'):
                train_step = tf.train.AdamOptimizer(learning_rate=tflr).minimize(lossFunction,
                                                             global_step=global_step )
            elif (optimizerName == 'gradientDescent'):
                train_step = tf.train.GradientDescentOptimizer(learning_rate=tflr).minimize(lossFunction,
                                                             global_step=global_step )
        init_op = tf.global_variables_initializer()
        saver = tf.train.Saver(max_to_keep=0)
        ff = 0 # used to store temporary F1
        lastF1 = -1 # used to determine whether to save a new checkpoint or not
        if showPlots == True:
            fig = plt.figure()
            plt.title(hyperParamStr)
        with tf.Session() as s:
            tf.set_random_seed(42)
            init_op.run()

            curStep = 1
            batchSize = 100

            if bSaveTFSummary == True:
                merged_summary = tf.summary.merge_all()
                file_writer = tf.summary.FileWriter('data/tflogs4/' + hyperParamStr)
                file_writer.add_graph(s.graph)

            maxStep = int(len(y_train)/batchSize)+1
            for curStep in range(1,maxStep+1):
                # get the actual train data chunk
                batch_x  = x_train[(curStep-1)*batchSize:(curStep)*batchSize]
                batch_ax = x_aisle[(curStep-1)*batchSize:(curStep)*batchSize]
                batch_dx = x_dep[(curStep-1)*batchSize:(curStep)*batchSize]
                batch_y = y_train[(curStep-1)*batchSize:(curStep)*batchSize]

                if len(batch_x) == 0:
                    continue

                if variableLearningRate == True:
                    max_learning_rate = initiallr #0.001
                    min_learning_rate = initiallr/100
                    decay_speed = 1000.0  # 0.003-0.0001-2000=>0.9826 done in 5000 iterations
                    learning_rate = min_learning_rate + (max_learning_rate - min_learning_rate) * math.exp(-(curStep*batchSize) / decay_speed)
                else:
                    learning_rate = initiallr

                # train
                feed_dict = {inputPlaceholder: batch_x , truthYPlaceholder: batch_y, neuronDropoutRate : dropoutRate, tflr: learning_rate, istest:False, currentiter:curStep, aislePlaceholder:batch_ax, departmentPlaceholder:batch_dx}
                train_step.run(feed_dict=feed_dict)

                # save summary for tensorboard
                if ( bSaveTFSummary == True and curStep % 5 == 0 ):
                    summaryData = s.run(merged_summary,feed_dict)
                    file_writer.add_summary(summaryData,curStep)


                # check F1 score for early stopping
                if ( curStep % 50 == 0 and tmpCheckEarlyStopping == True):
                    # get the value for scoring
                    x_test = holdout[features]
                    x_tdep = holdout['department_id']
                    x_taisle = holdout['aisle_id']

                    feed_dict = {inputPlaceholder: x_test, neuronDropoutRate: 1.0,
                                 istest: True, aislePlaceholder: x_taisle, departmentPlaceholder: x_tdep}
                    predHoldout = prediction.eval(feed_dict=feed_dict)
                    xx = list(int(i[0] > threshold) for i in predHoldout)

                    df = pd.DataFrame(columns=('user_id', 'product_id', 'predy'))
                    df['user_id'] = holdout['user_id']
                    df['product_id'] = holdout['product_id']
                    df['predy'] = xx

                    print('+', end='')
                    _, ff = fasterScoreProduction(df)
                    # _, ff = pcb.scorePrediction(df) f1:0.369394614599

                # record accuracy
                feed_dict = {inputPlaceholder: batch_x , truthYPlaceholder: batch_y, neuronDropoutRate : 1.0, tflr: learning_rate, istest:False, currentiter:curStep, aislePlaceholder:batch_ax, departmentPlaceholder:batch_dx}
                acc, loss = s.run([accuracy, lossFunction], feed_dict=feed_dict)
                acclossrecord = np.append(acclossrecord, [[acc, loss, ff]], axis=0)

                # save checkpoint
                if ( curStep % 50 == 0 ):
                    extrarr = ""
                    if ( lastF1 < ff ):
                        lastF1 = ff
                        extrarr = "**"
                    fileName = "data/checkpoints/nn4" + pcb.maxuserid + "-" + extrarr + str(curStep) + "-" + str(maxStep) + "f." + str(ff) + ".ckpt"
                    saver.save(s, fileName, global_step=global_step)

                # print the accuracy to stdout
                if (curStep % 100 == 0):
                    print('Accuracy: %s loss:%s ' % (str(acc), str(loss)))

                #graph the accuary/loss/F1
                if curStep % 50 == 0 and showPlots == True:
                    plt.clf()
                    movingavg = np.convolve(acclossrecord[:, 0], np.ones((50,)) / 50, mode='valid')
                    ax = fig.add_subplot(221)
                    ax2 = fig.add_subplot(222)
                    ax3 = fig.add_subplot(212)
                    ax4 = ax3.twinx()
                    ax2.set_yscale("log")
                    ax.plot(acclossrecord[:, 0])
                    ax.plot(movingavg, color='yellow')
                    ax2.plot(acclossrecord[:, 1])
                    ax3.plot(acclossrecord[:, 0])
                    ax3.plot(movingavg, color='yellow')
                    ax4.plot(acclossrecord[:, 2], color='orange')
                    ax3.axhline(max(movingavg), linestyle='--', color='grey', label='max accuracy', linewidth=0.5,
                                xmax=0.9)
                    ax4.axhline(max(acclossrecord[:, 2]), linestyle='--', color='red', label='max accuracy',
                                linewidth=0.5,
                                xmin=0.1)
                    ax4.grid(False)
                    plt.ion()
                    plt.pause(0.05)

            # show the final graph
            if showPlots == True:
                fig = plt.figure()
                movingavg = np.convolve(acclossrecord[:, 0], np.ones((50,)) / 50, mode='valid')
                ax = fig.add_subplot(221)
                ax2 = fig.add_subplot(222)
                ax3 = fig.add_subplot(212)
                ax4 = ax3.twinx()
                ax2.set_yscale("log")
                ax.plot(acclossrecord[:, 0])
                ax.plot(movingavg, color='yellow')
                ax2.plot(acclossrecord[:, 1])
                ax3.plot(acclossrecord[:, 0])
                ax3.plot(movingavg, color='yellow')
                ax4.plot(acclossrecord[:, 2], color='orange')
                ax3.axhline(max(movingavg), linestyle='--', color='grey', label='max accuracy', linewidth=0.5, xmax=0.9)
                ax4.axhline(max(acclossrecord[:, 2]), linestyle='--', color='red', label='max accuracy', linewidth=0.5,
                            xmin=0.1)
                ax4.grid(False)
                plt.ion()
                plt.pause(0.05)
                plt.show()

            # get the value for scoring (holdout)
            x_test = holdout[features]
            x_tdep = holdout['department_id']
            x_taisle = holdout['aisle_id']

            feed_dict = {inputPlaceholder: x_test, neuronDropoutRate: 1.0,
                         istest: True, aislePlaceholder: x_taisle,departmentPlaceholder: x_tdep}
            predHoldout = prediction.eval(feed_dict=feed_dict)

            if ( len(test) > 0 ):
                x_test = test[features]
                x_tdep = test['department_id']
                x_taisle = test['aisle_id']
                feed_dict = {inputPlaceholder: x_test, neuronDropoutRate: 1.0,
                             istest: True, aislePlaceholder: x_taisle, departmentPlaceholder: x_tdep}
                predTest = prediction.eval(feed_dict=feed_dict)


        # cast the predictions as integers
        xx = list(round(i[0]) for i in predHoldout)
        xx = list(int(i[0] > threshold) for i in predHoldout)

        df = pd.DataFrame(columns=('user_id', 'product_id', 'predy'))
        df['user_id'] = holdout['user_id']
        df['product_id'] = holdout['product_id']
        df['predy'] = xx
        df['floaty'] = predHoldout

        pcb.debugWithTimer("scoring prediction" + hyperParamStr)
        _, f1score = pcb.scorePrediction(df)
        pcb.debugWithTimer("done scoring")
        if  f1score > bestScore:
            bestDefinition = oneDefinition
            bestScore = f1score
            bestDFHoldout = df

            xx = list(int(i[0] > threshold) for i in predTest)
            bestDFTest = pd.DataFrame(columns=('user_id', 'product_id', 'predy'))
            bestDFTest['user_id'] = test['user_id']
            bestDFTest['product_id'] = test['product_id']
            bestDFTest['predy'] = xx
            bestDFTest['floaty'] = predTest

        allScores[str(oneDefinition)] = f1score

        if usePriorResultFile:
            pcb.saveRun(str(oneDefinition),f1score)

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

    bestDFHoldout.to_csv('data/results/nn' + pcb.maxuserid + '.csv', index=False)
    return bestDFHoldout, bestDFTest

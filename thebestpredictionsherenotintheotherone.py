import petitchatbase as pcb
import pandas as pd
import random
import sklearn
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
import tensorflow as tf
import numpy as np

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

def myFirstNN(train, test):
    features = ['orderfrequency', 'dayfrequency', 'department_id', 'aisle_id', 'orderfrequency', 'days_without_product_order','eval_days_since_prior_order']
    #features = ['orderfrequency']

    nbfeatures = len(features)

    tf.set_random_seed(42)

    x_train = train[features]
    #y_train = train[['reordered']] # this needs to be 1 column wide, not a list

    y_train = np.column_stack(
                (list(float(i) for i in train['reordered']),
                 list(float(not (i)) for i in train['reordered'])))

    #y_train['newcolumn3'] = list(float(not(i)) for i in y_train['reordered'].values)

    # define input and output
    inputPlaceholder = tf.placeholder('float', [None, nbfeatures])
    truthYPlaceholder = tf.placeholder('float', [None,2])
    neuronDropoutRate = tf.placeholder('float')

    hiddenLayerSizes = [200, 100, 50, 10]
    layerSize0 = nbfeatures
    hiddenLayerDefinitions = []

    for layerSize in hiddenLayerSizes:
        hiddenLayerDefinitions.append({'weights':tf.Variable(tf.random_normal([layerSize0,layerSize])),'biases':tf.Variable(tf.random_normal([layerSize]))})
        layerSize0 = layerSize
    print('Hidden layer sizes : %s ' % hiddenLayerSizes)

    outputLayerDefinition = {'weights':tf.Variable(tf.random_normal([10,2])),
                             'biases': tf.Variable(tf.random_normal([2]))}

    previousLayer = inputPlaceholder
    for definition in hiddenLayerDefinitions:
        layer = tf.add(tf.matmul(previousLayer, definition['weights']), definition['biases'])
        layer = tf.nn.relu(layer)
        layer = tf.nn.dropout(layer, neuronDropoutRate)
        previousLayer = layer

    #the output is also the model we're going to run (any layer can apparently be)
    #output = tf.nn.softmax(tf.matmul(l1, outputLayerDefinition['weights']))
    output = tf.matmul(previousLayer, outputLayerDefinition['weights'])

    #give the cost function for optimizer
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=truthYPlaceholder))


    # create the optimizer (also the training step)
    optimizer = tf.train.AdamOptimizer().minimize(cost)
    #optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(cost)

    #set the prediction and truth values
    prediction = tf.argmax(output,1)
    trainingtruth = tf.argmax(truthYPlaceholder,1)
    #prediction = output
    #trainingtruth = tf.cast(y_train,'float')

    #define correctness
    correct = tf.equal(prediction, trainingtruth)

    #define accuracy
    accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

    # sess = tf.Session()
    # sess.run(tf.global_variables_initializer())
    # sess.run([optimizer, cost], feed_dict={inputPlaceholder: x_train , truthYPlaceholder: y_train})
    # #optimizer.run(feed_dict={inputPlaceholder: x_train, truthYPlaceholder: y_train})
    # xxx = sess.run(accuracy.eval({inputPlaceholder: x_train , truthYPlaceholder: y_train}))


    with tf.Session() as s:
        tf.set_random_seed(42)
        tf.global_variables_initializer().run()

        batchSize = 100
        curStep = 1
        for curStep in range(1,int(len(y_train)/batchSize)+1):
            batch_x = x_train[(curStep-1)*100:(curStep)*100]
            batch_y = y_train[(curStep-1)*100:(curStep)*100]

            # train
            trainData = {inputPlaceholder: batch_x , truthYPlaceholder: batch_y, neuronDropoutRate : 1.0}
            optimizer.run(feed_dict=trainData)

            trainData = {inputPlaceholder: batch_x , truthYPlaceholder: batch_y, neuronDropoutRate : 1.0}
            acc, err = s.run([accuracy, cost], feed_dict=trainData)

            #xxx = accuracy.eval({inputPlaceholder: batch_x, truthYPlaceholder: batch_y})
            print('Accuracy on self: %s error:%s ' % (str(acc), str(err)))

        x_test = test[features]
        y_test = test[['reordered']]
        #print('Accuracy on test:', accuracy.eval({inputPlaceholder: x_test , truthYPlaceholder: y_test}))

        #print(tf.reduce_mean(tf.cast(tf.equal(tf.argmax(output,1),tf.argmax(y_test)),'float')).eval(feed_dict={inputPlaceholder:x_test}))
        o = prediction.eval(feed_dict={inputPlaceholder:x_test, neuronDropoutRate : 1.0})

    # features = np.array(list(features))
    # # pos: [1,0] , argmax: 0
    # # neg: [0,1] , argmax: 1
    # result = (sess.run(tf.argmax(prediction.eval(feed_dict={x: [features]}), 1)))
    # if result[0] == 0:
    #     print('Positive:', input_data)
    # elif result[0] == 1:
    #     print('Negative:', input_data)

    df = pd.DataFrame(columns=('user_id', 'product_id', 'predy'))
    df['user_id'] = test['user_id']
    df['product_id'] = test['product_id']
    df['predy'] = o
    return df


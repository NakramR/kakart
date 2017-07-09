import petitchatbase as pcb
import pandas as pd
import random
import sklearn
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
import tensorflow as tf
import numpy as np
from collections import Counter

def generateRandomPrediction():
    randpred = pd.DataFrame(columns=('user_id', 'product_id', 'ordered'))
    pcb.debugWithTimer('reading distinct user products')
    userpriorproducts = pcb.userproductstats

    pcb.debugWithTimer('iterating over prior products')
    temp = []
    for index, x in userpriorproducts.iterrows():
        newline = { 'user_id': x['user_id'], 'product_id': x['product_id'], 'ordered': (random.random() > 0.5) }
        temp.append(newline)

    randpred = randpred.append(temp)

    return randpred

def predictOverFrequencyThreshold(threshold):
    userpriorproducts = pcb.userproductstats

    userpriorproducts['ordered'] = userpriorproducts['orderfrequency'] > threshold

    #userpriorproducts['ordered'] = userpriorproducts.query("orderfrequency > " + str(threshold) + " or (dayfrequency > days_without_product_order + eval_days_since_prior_order)")

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

    df = pd.DataFrame(columns=('user_id', 'product_id', 'ordered'))
    df['user_id'] = test['user_id']
    df['product_id'] = test['product_id']
    df['ordered'] = y_pred
    return df

def myFirstNN(train, test):
    features = ['orderfrequency', 'dayfrequency', 'department_id', 'aisle_id', 'orderfrequency', 'days_without_product_order','eval_days_since_prior_order']
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
    outputPlaceholder = tf.placeholder('float', [None,2])


    # just some json definitions of what the layers are
    hiddenLayer1Definition = {'weights':tf.Variable(tf.random_normal([nbfeatures,50]))
                             ,'biases':tf.Variable(tf.random_normal([50]))}

    outputLayerDefinition = {'weights':tf.Variable(tf.random_normal([50,2])),
                             'biases': tf.Variable(tf.random_normal([2]))}

    #create layer from input to next layer
    l1 = tf.add(tf.matmul(inputPlaceholder, hiddenLayer1Definition['weights']), hiddenLayer1Definition['biases'])
    l1 = tf.nn.relu(l1)

    #the output is also the model we're going to run (any layer can apparently be)
    #output = tf.nn.softmax(tf.matmul(l1, outputLayerDefinition['weights']))
    output = tf.matmul(l1, outputLayerDefinition['weights'])

    #give the cost function for optimizer
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=outputPlaceholder))

    # create the optimizer
    optimizer = tf.train.AdamOptimizer().minimize(cost)
    #optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

    #set the prediction and truth values
    prediction = tf.argmax(output,1)
    trainingtruth = tf.argmax(y_train,1)
    #prediction = output
    #trainingtruth = tf.cast(y_train,'float')

    #define correctness
    correct = tf.equal(prediction, trainingtruth)

    #define accuracy
    accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

    # sess = tf.Session()
    # sess.run(tf.global_variables_initializer())
    # sess.run([optimizer, cost], feed_dict={inputPlaceholder: x_train , outputPlaceholder: y_train})
    # #optimizer.run(feed_dict={inputPlaceholder: x_train, outputPlaceholder: y_train})
    # xxx = sess.run(accuracy.eval({inputPlaceholder: x_train , outputPlaceholder: y_train}))


    with tf.Session() as s:
        tf.set_random_seed(42)
        tf.global_variables_initializer().run()
        optimizer.run(feed_dict={inputPlaceholder: x_train , outputPlaceholder: y_train})

        xxx = accuracy.eval({inputPlaceholder: x_train, outputPlaceholder: y_train})

        print('Accuracy on self:', xxx)

        x_test = test[features]
        y_test = test[['reordered']]
        #print('Accuracy on test:', accuracy.eval({inputPlaceholder: x_test , outputPlaceholder: y_test}))

        #print(tf.reduce_mean(tf.cast(tf.equal(tf.argmax(output,1),tf.argmax(y_test)),'float')).eval(feed_dict={inputPlaceholder:x_test}))
        o = prediction.eval(feed_dict={inputPlaceholder:x_test})
        print('Prediction data: %s, frequency: %s ' % (o, Counter(o)))

    # features = np.array(list(features))
    # # pos: [1,0] , argmax: 0
    # # neg: [0,1] , argmax: 1
    # result = (sess.run(tf.argmax(prediction.eval(feed_dict={x: [features]}), 1)))
    # if result[0] == 0:
    #     print('Positive:', input_data)
    # elif result[0] == 1:
    #     print('Negative:', input_data)

    df = pd.DataFrame(columns=('user_id', 'product_id', 'ordered'))
    df['user_id'] = test['user_id']
    df['product_id'] = test['product_id']
    df['ordered'] = o
    return df

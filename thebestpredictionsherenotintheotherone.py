import dothestuff as pcb
import pandas as pd
import random


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

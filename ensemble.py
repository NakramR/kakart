import pandas as pd
import petitchatbase as pcb

# pcb.maxuserid = '1000000000'
pcb.maxuserid = '10000'

pcb.initData(pcb.maxuserid )
pcb.train, pcb.holdout, pcb.test = pcb.balancedTrainAndTestForValidation()
def combinePredictions(row, *args):
    if(row['predy'] + row['predy_'] >= 1):
        return 1
    return 0

def ensemble(models):
    firstDf = pd.read_csv('data/results/'+ models[0] + pcb.maxuserid  + '.csv')

    for i in range(1, len(models)):
        df = pd.read_csv('data/results/' + models[i] + pcb.maxuserid + '.csv')
        firstDf = firstDf.merge(df, on=('user_id', 'product_id'), suffixes=('', '_'), how='outer')
        firstDf['floaty'] = firstDf['floaty'] + firstDf['floaty_']
        firstDf = firstDf.drop('floaty_', 1)
        firstDf = firstDf.drop('predy_', 1)


    df = pd.DataFrame(columns=('user_id', 'product_id', 'predy'))
    df['user_id'] = firstDf['user_id']
    df['product_id'] = firstDf['product_id']
    threshold = 0.5 * len(models)  # sum of 2 thresholds
    df['predy'] = list(int(el >= threshold) for el in firstDf['floaty'] )
    pcb.scorePrediction(df)

#
# df1 = pd.read_csv('data/results/nn'+ pcb.maxuserid  + '.csv')
# df2 = pd.read_csv('data/results/xgboost'+ pcb.maxuserid  + '.csv')
#
# # pcb.scorePrediction(df1)
# # pcb.scorePrediction(df2)
#
# res = df1.merge(df2, on=('user_id', 'product_id'), suffixes=('', '_'), how = 'outer')
# res['intersection'] =res['floaty'] * res['floaty_']
# res['combined'] = res['floaty'] + res['floaty_']
# threshold = 1 #sum of 2 thresholds
# res['predy'] = list(int(el >= threshold) for el in res['combined'] )
#
#
# df = pd.DataFrame(columns=('user_id', 'product_id', 'predy'))
# df['user_id'] = res['user_id']
# df['product_id'] = res['product_id']
# df['predy'] = res['predy']
#
#
# pcb.scorePrediction(df)
# res['combined'] = res.apply(combinePredictions, axis=1)

ensemble(['nn', 'xgboost'])
print('done')
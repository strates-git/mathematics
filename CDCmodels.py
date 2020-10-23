# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 18:17:48 2020

@author: Shane Strate
"""
import pandas as pd
import glob, os
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error, mean_absolute_error, confusion_matrix
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

columns = ['model', 'target', 'target_week_end_date', 'location_name', 'point', 'lower_bound', 'upper_bound', 'cumulative']
accuracyList = []
confusion = []  #Not Used Yet
#File operations
path = r"CDC Data"
actuals_path = r"CDC Data\Actuals\2020-10-22-actuals.csv"
model_files = glob.glob(os.path.join(path, "*.csv"))
modelData = pd.DataFrame()
modelData = pd.concat((pd.read_csv(file) for file in model_files))
actualData = pd.read_csv(actuals_path)
#initial cleanup
modelData['target_week_end_date'] = pd.to_datetime(modelData.target_week_end_date)
modelData.rename(columns={'quantile_0.025':'lower_bound', 'quantile_0.975':'upper_bound'}, inplace=True)
actualData['End Week'] = pd.to_datetime(actualData['End Week']) 
#mapping to rename 'target' field values
forecastWindows = {'1 wk ahead cum death': '1wk',
                  '2 wk ahead cum death': '2wk',
                  '3 wk ahead cum death': '3wk',
                  '4 wk ahead cum death': '4wk'}
modelData = modelData.replace('US', 'National')
print('Initial Model Count: ', len(pd.unique(modelData['model'])))

#Determine if observed data is within predicted 95% CI
def calcHit(row):
    if (row.cumulative >= row.lower_bound) & (row.cumulative <= row.upper_bound):
        val = 1
    else:
        val = 0
    iRange = row.upper_bound - row.lower_bound
    return val, iRange

#Filter data to National and Cumulative death counts. Filter out extreme outlier models
nationalData = modelData[(modelData['location_name']=='National')
                         & (modelData['target'].str.contains('cum'))
                         & (modelData['model'] != 'Imperial')
                         & (modelData['model'] != 'JHU-APL')
                         & (modelData['model'] != 'JHU-IDD')]

mergedData = pd.merge(nationalData, actualData, left_on=['target_week_end_date', 'location_name'], right_on=['End Week', 'State'])
print('Interim Model Count:', len(pd.unique(mergedData['model'])))
mergedData.dropna(axis=0, subset=['lower_bound', 'upper_bound'], inplace=True)

mergedData = mergedData[columns]
mergedData['missedCount'] = mergedData.apply(lambda row: row.point-row.cumulative, axis=1)
mergedData[['hit', 'iRange']] = mergedData.apply(lambda row: pd.Series(list(calcHit(row))), axis=1)
#Group Data into (Model, Target) pairs and calculate aggregate stats (not all stats used)
newData = mergedData.groupby(['model', 'target']).agg({'hit': ['sum', 'mean'], 
                                                       'missedCount': 'mean',
                                                       'iRange' : 'mean',
                                                       'model' : 'count'})
newData.columns=['hitSum', 'hitMean', 'missedCountMean', 'iRangeMean', 'modelObsCount']
newData.reset_index()
#remerged to get back 'cumulative' and 'point' fields
mergedData = pd.merge(mergedData, newData, on=['model', 'target'])
#Filter out small models and outliers
mergedData = mergedData[(mergedData.iRange <= 1000000) & (mergedData.modelObsCount >= 5)]
#not used
mergedData['iRangeScaled'] = preprocessing.MinMaxScaler().fit_transform(np.array(mergedData['iRangeMean']).reshape(-1,1))

#Slice data by models and targets to get accuracy measures for each (Model, Target) pair. 
for model in pd.unique(mergedData['model']):
    for target in pd.unique(mergedData['target']):
        slicedData = mergedData[(mergedData['model']== model) & (mergedData['target']==target)]
        if not slicedData.empty:
            MSE = mean_squared_error(slicedData['cumulative'], slicedData['point'])
            RMSE = np.sqrt(MSE)
            MAE = mean_absolute_error(slicedData['cumulative'], slicedData['point'])
            #conf = confusion_matrix(slicedData['cumulative'], slicedData['point'])
            accuracyList.append([model, target, MSE, RMSE, MAE])
            #confusion.append([model, target, conf])

#create a dataframe of accuracy measures, and merge with the rest of the data.      
accuracyDF = pd.DataFrame(accuracyList, columns=['model', 'target', 'MSE', 'RMSE', 'MAE'])
mergedData = mergedData.merge(accuracyDF, on=['model', 'target'])
#rename 'target' field values for readability
mergedData.target = mergedData.target.map(forecastWindows)

#cleanup and sort merged data prior to plotting
plotData = mergedData.groupby(['model', 'target']).agg('max').reset_index()
plotData = plotData.sort_values(by=['model', 'target'], ascending=[True, False])
plotData = plotData.round(4)

#slice on single (model, target) pair to demonstrate model vs. observation
timeSeriesData = mergedData[(mergedData['model']=='Ensemble') & (mergedData['target']=='4wk')][['target_week_end_date', 'point', 'cumulative']]
timeSeriesData.set_index('target_week_end_date', inplace=True)

#plotting options below
#plot = sns.lineplot(data=timeSeriesData, legend=True)
#plot.set(xlim=('2020-06-06', '2020-10-17'))
#plt.title('Ensemble - 4 Weeks Out')
#plt.xticks(rotation=45, horizontalalignment='right')
#plt.savefig('cumulativevEnd_date.pdf', dpi=300)
#plot = sns.lineplot(x='target', y='MAE', data=plotData, hue='model', legend=False)
#plt.savefig('MAEvTarget3.pdf', dpi=300)
plot = sns.FacetGrid(plotData, col='target', hue='model')
plot.map(sns.scatterplot, 'modelObsCount', "hitMean")
plot.add_legend()
plot.savefig('hitMeanvObsCount3.pdf', dpi=300)
#plot = sns.FacetGrid(plotData, col='model', col_wrap=6, hue='model')
#plot.map(sns.lineplot, 'target', "MAE")
#plot.add_legend()
#plot.savefig('MAEvTarget2.pdf', dpi=300)

print('Final Model Count:', len(pd.unique(mergedData['model'])))


mergedData.to_csv(r'CDC_Data.csv')


    
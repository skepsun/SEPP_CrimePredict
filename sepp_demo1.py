
# coding: utf-8

# In[9]:

import sys, os.path
sys.path.insert(0, os.path.abspath("F:/Git/PredictCode_0"))


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import csv
from open_cp import data
import datetime
import open_cp
import math
from open_cp import logger

logger.log_to_stdout()

# In[18]:
def cleaner(df, index):
    # param: df, type: dataframe
    # param: index
    point_size = len(df)
    cleaned_df = df.drop(index,axis=1)
    timestamps = np.zeros((point_size), dtype='str')
    cleaned_df['Dates'] = cleaned_df['Dates'].apply(lambda x: np.datetime64(x))
    cleaned_df = cleaned_df.values
    timestamps = cleaned_df[:,2]
    coordX = cleaned_df[:,0]
    coordY = cleaned_df[:,1]
    timedpoints = data.TimedPoints.from_coords(timestamps, coordX, coordY)
    return timedpoints


# In[19]:

data_raw = pd.read_csv('F:/Crimes/example_crime_data.csv')
num_points = len(data_raw)
for i in range(0, num_points):
    data_raw.set_value(i, 'Dates',datetime.datetime(2017, data_raw['Month'][i], data_raw['Day'][i], data_raw['Hour'][i]))
index = ['Month', 'Day', 'Hour']
data_raw = data_raw.sort_values('Dates')
CleanedData = cleaner(data_raw, index)
X_min = CleanedData.coords[0].min()
X_max = CleanedData.coords[0].max()
Y_min = CleanedData.coords[1].min()
Y_max = CleanedData.coords[1].max()
range_X = X_max - X_min
range_Y = Y_max - Y_min
tau = range_Y / range_X
region = data.RectangularRegion(xmin=X_min, xmax=X_max, ymin=Y_min, ymax=Y_max)
region.aspect_ratio == tau


# In[20]:

import open_cp.sepp as sepp
trainer = sepp.SEPPTrainer(k_time=100, k_space=15)
trainer.data = CleanedData
predictor = trainer.train(cutoff_time=None, iterations=40)


# In[21]:

predictor.data = CleanedData
test_datetime = datetime.datetime(2017,6,14)
prediction = predictor.predict(test_datetime,cutoff_time=None)
prediction


# In[24]:

cell_num_row = 30
cell_width = range_X / cell_num_row
grided = open_cp.predictors.GridPredictionArray.from_continuous_prediction_region(prediction, region, cell_width, cell_width*tau)
fig, ax = plt.subplots(ncols=2, figsize=(16,8))
ax[0].set(xlim=[region.xmin, region.xmax], ylim=[region.ymin, region.ymax])
ax[0].pcolormesh(*grided.mesh_data(), grided.intensity_matrix, cmap="Blues")
ax[0].set_title("Prediction of risk at"+test_datetime.strftime('%Y-%m-%d'))

grided_bground = open_cp.predictors.grid_prediction_from_kernel(predictor.background_kernel.space_kernel,
        region, cell_width, cell_width*tau)
ax[1].set(xlim=[region.xmin, region.xmax], ylim=[region.ymin, region.ymax])
ax[1].pcolormesh(*grided_bground.mesh_data(), grided_bground.intensity_matrix, cmap="Blues")
ax[1].set_title("Background risk")
plt.show()
print("Prediction={}".format(prediction._kernel))


# In[ ]:




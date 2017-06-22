# -*- coding: utf-8 -*-
"""
Created on Mon Jun 19 13:03:45 2017
Predict SEPP
@author: scx
"""

import pandas as pd
import numpy as np
import scipy.spatial as _spatial
import datetime
import random
import sys
from time import sleep
from operator import mod
from matplotlib import pyplot as plt

def compute_kernel(x, mean, var, D, N=1):
    result = 1
    if len(mean.shape)==1:
        feature_dimension = mean.shape[0]
        token = False
        if mean.shape[0] == 3:
            if (x[0]-mean[0]>0) :
                token = True
        elif mean.shape[0] == 1:
            token = True
        elif mean.shape[0] == 2:
            token = True
        if token:
            for j in range(0, feature_dimension):
                result = result/(var[j]*np.sqrt(np.pi)*D)*np.exp(-(x[j]-mean[j])**2/(2*(var[j]**2)*(D**2)))
        else:
            result = 0
            
    elif len(mean.shape)==2:
        result = 0
        num_points = mean.shape[0]
        for i in range(0, num_points):
            result = result+compute_kernel(x, mean[i], var, D[i])
    result = result/N
    return result

def predict(test_point, a):
    pred = compute_kernel(np.array([test_point[0,0]]), a[0][:,0][:,None], a[2], a[5], 1)
    pred *= compute_kernel(test_point[0,1:3], a[0][:,1:3], a[3], a[6], 1)
    pred += compute_kernel(test_point[0,0:3], a[1][:,0:3], a[4], a[7], 1)
    return pred

def pred_background(test_point, a):
    pred = compute_kernel(np.array([test_point[0,0]]), a[0][:,0][:,None], a[2], a[5], 1)
    pred *= compute_kernel(test_point[0,1:3], a[0][:,1:3], a[3], a[6], 1)
#    pred += compute_kernel(test_point[0,0:3], a[1][:,0:3], a[4], a[7], a[8])
    return pred

SEPP = np.load('F:/Crimes/sepp.npy')
ref_date = datetime.datetime(2017,1,1)
date = datetime.datetime(2017,7,1,1)
relat_date = (date - ref_date)/datetime.timedelta(hours=1)
num_randpoints = 1
grids = pd.read_csv('F:/Crimes/grid_3500.csv')

num_grids = len(grids['ID'])
risk = np.zeros([num_grids,1])
background = np.zeros([num_grids,1])
print('start prediction!')
for i in range(0, num_grids):
    if (mod(i,1000)==0):
        print('Predicting the %d th grid……'%(i),flush=True)
#    sys.stdout.flush()
#    sleep(1)
    left = grids['LeftDownX'][i]
    down = grids['LeftDownY'][i]
    right = grids['RightUpX'][i]
    up = grids['RightUpY'][i]
#    for j in range(0, num_randpoints):
#        x = random.uniform(left, right)
#        y = random.uniform(down, up)
    x = grids['CenterX'][i]
    y = grids['CenterY'][i]
    point = np.array([[relat_date, x, y]])
    pred = predict(point, SEPP)
    pred_b= pred_background(point, SEPP)
    risk[i] += pred
    background[i] +=pred_b
    risk[i] /= num_randpoints
    background[i] /= num_randpoints
    grids.set_value(i, 'Risk', risk[i])
grids.to_csv('F:/Crimes/'+date.strftime('%Y-%m-%d %H')+'.csv')
x_min = grids['LeftDownX'].min()
y_min = grids['LeftDownY'].min()
x_max = grids['RightUpX'].max()
y_max = grids['RightUpY'].max()
delta_x = grids['RightUpX'][0] - grids['LeftDownX'][0]
delta_y = grids['RightUpY'][0]- grids['LeftDownY'][0]
num_x = int(round((x_max - x_min)/delta_x))
num_y = int(round((y_max - y_min)/delta_y))
x_ = np.arange(x_min,x_max+0.001, delta_x)
y_ = np.arange(y_min,y_max+0.001, delta_y)

risk = risk.reshape([ num_x,num_y]).transpose()
background = background.reshape([ num_x,num_y]).transpose()
fig,ax = plt.subplots(ncols = 2, figsize = ( 10, 7) )

ax[0].pcolormesh( x_, y_, risk, cmap = 'Blues')
ax[0].axis( [ x_min, x_max, y_min, y_max ] )
ax[0].set_xlabel( 'x' )
ax[0].set_ylabel( 'y' )
ax[0].set_title('Prediction of date:'+date.strftime('%Y-%m-%d %H'))

ax[1].pcolormesh( x_, y_, backgrounds, cmap = 'Blues')
ax[1].axis( [ x_min, x_max, y_min, y_max ] )
ax[1].set_xlabel( 'x' )
ax[1].set_ylabel( 'y' )
ax[1].set_title('Backgrounds')

#plt.savefig('F:/Crimes/'+date.strftime('%Y-%m-%d %H')+'.png')
plt.show()
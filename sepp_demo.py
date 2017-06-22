# -*- coding: utf-8 -*-
"""
Created on Fri Jun  9 15:44:42 2017

@author: scx
Some code folked from MatthewDaws' project PredictCode
"""
import matplotlib.pyplot as plt
import numpy as np

import pandas as pd
import csv
import datetime
import math
import scipy.spatial as _spatial
#from . import kernels

def cleaner(df, index):
    # param: df, type: dataframe
    # param: index
#    point_size = len(df)
    cleaned_df = df[index]
#    cleaned_df = cleaned_df[cleaned_df['X']>116.15]
#    cleaned_df = cleaned_df[cleaned_df['X']<116.7]
#    cleaned_df = cleaned_df[cleaned_df['Y']>39.6]
#    cleaned_df = cleaned_df[cleaned_df['Y']<40.5]
#    cleaned_df = cleaned_df.reset_index(drop=True)
    num_points = len(cleaned_df)
    for i in range(0, num_points):
        cleaned_df.set_value(i, 'Dates',(datetime.datetime(2017, df['Month'][i], df['Day'][i], df['Hour'][i])\
                                                          -datetime.datetime(2017,1,1,1,0))/datetime.timedelta(hours=1) )
#    timestamps = np.zeros((point_size), dtype='str')
#    cleaned_df['Dates'] = cleaned_df['Dates'].apply(lambda x: (datetime.datetime.strptime(x, '%Y/%m/%d %H:%M')\
#                                                          -datetime.datetime(2017,1,1,1,0))/datetime.timedelta(hours=1))
    cleaned_df = cleaned_df[['Dates','X','Y']]
    cleaned_df = cleaned_df.sort_values('Dates', axis = 0, ascending = True)
    timedpoints = cleaned_df.values
    return timedpoints

def read_data(file_path, index):
    raw = pd.read_csv(file_path)
#    raw = raw[0:300]
    cleaned_data = cleaner(raw, index)
    cleaned_data = cleaned_data[4200:]
#    cleaned_data.reset_index(drop=True)
    return cleaned_data

def sample_points(points, p):
    """Using the probability matrix, sample background and triggered points.

    :param points: The (time, x, y) data.
    :param p: The probability matrix.

    :return: A pair of `(backgrounds, triggered)` where `backgrounds` is the
      `(time, x, y)` data of the points classified as being background events,
      and `triggered` is the `(time, x, y)` *delta* of the triggered events.
      That is, `triggered` represents the difference in space and time between
      each triggered event and the event which triggered it, as sampled from
      the probability matrix.
    """

    number_data_points = points.shape[0]
#    print (p)
    choice = np.array([ np.random.choice(j+1, p=p[0:j+1,j])for j in range(number_data_points)])
#    print(choice)
    mask = ( choice == np.arange(number_data_points) )

    backgrounds = points[mask,:]
    triggered = points[~mask,:]
    return backgrounds, triggered

def p_matrix(points, background_kernel, trigger_kernel):
    """Computes the probability matrix.

    :param points: The (time, x, y) data
    :param background_kernel: The kernel giving the background event intensity.
    :param trigger_kernel: The kernel giving the triggered event intensity.

    :return: A matrix `p` such that `p[i][i]` is the probability event `i` is a
      background event, and `p[i][j]` is the probability event `j` is triggered
      by event `i`.
    """

    number_data_points = points.shape[0]
    p = np.zeros([number_data_points, number_data_points])
    for j in range(0, number_data_points-1):
        d = points[j:, :] - points[j,:]
        p[j, j:] = trigger_kernel(d)
    b = background_kernel(points)
    for i in range(number_data_points):
        p[i, i] = b[i]
    return _normalise_matrix(p)

def initial_p_matrix(points, initial_time_bandwidth = 2.4,
        initial_space_bandwidth = 0.001):
    """Returns an initial estimate of the probability matrix.  Uses a Gaussian
    kernel in space, and an exponential kernel in time, both non-normalised.
    Diagonal (i.e. background "probabilities") are set to 1.  Finally the
    matrix is normalised.

    :param points: The (time, x, y) data.
    :param initial_time_bandwidth: The "scale" of the exponential.
    :param initial_space_bandwidth: The standard deviation of the Gaussian.
    """
    def bkernel(pts):
        return np.zeros(pts.shape[0]) + 1
    def tkernel(pts):
        time = np.exp( - pts[:,0] / initial_time_bandwidth )
        norm = 2 * initial_space_bandwidth ** 2
        space = np.exp( - (pts[:,1]**2 + pts[:,2]**2) / norm )
        return time * space
    return p_matrix(points, bkernel, tkernel)

def p_matrix_fast(points, background_kernel, trigger_kernel, time_cutoff=240, space_cutoff=0.05):
    """Computes the probability matrix.  Offers faster execution speed than
    :func:`p_matrix` by, in the calculation of triggered event
    probabilities, ignoring events which are beyond a space or time cutoff.
    These parameters should be set so that the `trigger_kernel` evaluates to
    (very close to) zero outside the cutoff zone.

    :param points: The (time, x, y) data
    :param background_kernel: The kernel giving the background event intensity.
    :param trigger_kernel: The kernel giving the triggered event intensity.
    :param time_cutoff: The maximum time between two events which can be
      considered in the trigging calculation.
    :param space_cutoff: The maximum (two-dimensional Eucliean) distance
      between two events which can be considered in the trigging calculation.

    :return: A matrix `p` such that `p[i][i]` is the probability event `i` is a
      background event, and `p[i][j]` is the probability event `j` is triggered
      by event `i`.
    """
    number_data_points = points.shape[-1]
    p = np.zeros((number_data_points, number_data_points))
    space_cutoff_sq = space_cutoff**2
    for j in range(1, number_data_points):
        d = points[:, j][:,None] - points[:, :j]
        dmask = (d[0] <= time_cutoff) & ((d[1]**2 + d[2]**2) <= space_cutoff_sq)
        d = d[:, dmask]
        if d.shape[-1] == 0:
            continue
        p[0:j, j][dmask] = trigger_kernel(d)
    p += np.diag(background_kernel(points))
    return _normalise_matrix(p)

def _normalise_matrix(p):
    column_sums = np.sum(p, axis=0)
    return p / column_sums

def compute_kernel(x, mean, var, D, N=1):
    result = 1
    if len(mean.shape)==1:
        feature_dimension = mean.shape[0]
        token = False
        if mean.shape[0] == 3:
#            x[0]-mean[0]<240 and and np.sqrt((x[1]-mean[1])**2+(x[2]-mean[2])**2<0.1) 
            if x[0]-mean[0]>=0 :
                token = True
        elif mean.shape[0] == 1:
            token = True
        elif mean.shape[0] == 2:
            token = True
        if token:
            for j in range(0, feature_dimension):
                result = result/(var[j]*np.sqrt(2*np.pi)*D)*np.exp(-(x[j]-mean[j])**2/(2*(var[j]**2)*(D**2)))
        else:
            result = 0
            
    elif len(mean.shape)==2:
        result = 0
        num_points = mean.shape[0]
        for i in range(0, num_points):
            result = result+compute_kernel(x, mean[i], var, D[i])
    result = result/N
    return result


def compute_kth_distance(points, k):
    eps = 0.00001
    if k>points.shape[0]-1:
        k=points.shape[0]-1
#    print(triggered)
    kth_distance = np.empty(points.shape[0])
#    if points.shape[0]>1:
    stds = np.std(points, axis = 0)
    stds = stds[None,:]
    final_stds = stds[0]
#    print(stds)
    scaled_points = points/stds
#    print(scaled_trigger)
    tree = _spatial.KDTree(scaled_points)
    kth_distance = np.empty(scaled_points.shape[0])
    for i,p in enumerate(scaled_points):
        distances, indexes = tree.query(p, k=k+1)
        kth_distance[i] = distances[-1]
#    else:
#        kth_distance = np.ones(points.shape[1])*1
#        final_stds = np.ones(points.shape[1])*0.00001
    return [kth_distance+eps, final_stds]

def predict(test_point, backgrounds, triggered, stds_nu, stds_mu, stds_trig, k_dist_nu, k_dist_mu, k_dist_trig, num_backgrounds, num_triggered):
    num_points = num_backgrounds + num_triggered
    pred_b = compute_kernel(np.array([test_point[0,0]]), backgrounds[:,0][:,None], stds_nu, k_dist_nu, 1)
    pred_b = pred_b*compute_kernel(test_point[0,1:3], backgrounds[:,1:3], stds_mu, k_dist_mu, 1)
    pred_t = compute_kernel(test_point[0,0:3], triggered[:,0:3], stds_trig, k_dist_trig, 1)
    
    return [pred_b+pred_t, pred_b, pred_t]

iter_num = 10
k_high = 100
k_low = 15
#file_path = 'F:/Learning/NLP/predictive policing/case_new.csv'
file_path = 'F:/Crimes/xingshi_new.csv'
points = read_data(file_path, ['Month','Day','Hour','X','Y'])
#points = np.array([[1,10,20],[2,21,41],[3,45,22],[4,24,54],[5,86,26],[6,28,64],[7,43,29]])
num_points = points.shape[0]
p = initial_p_matrix(points, initial_time_bandwidth = 240, initial_space_bandwidth = 0.1)
g = np.zeros([num_points,num_points])
mu = np.ones([num_points,])
nu = np.ones([num_points,])

for i in range(0,iter_num):
    [backgrounds, triggered] = sample_points(points,p)
    print(backgrounds.shape[0])
    num_backgrounds = backgrounds.shape[0]
    num_triggered = num_points - num_backgrounds
    [k_dist_trig, stds_trig] = compute_kth_distance(triggered, k_low)
    [k_dist_nu, stds_nu] = compute_kth_distance(backgrounds[:,0][:,None], k_high)
    [k_dist_mu, stds_mu] = compute_kth_distance(backgrounds[:,1:3], k_low)
    p = np.zeros([num_points,num_points])
    for i in range(0, num_points):
        nu[i] = compute_kernel(np.array([points[i,0]]), backgrounds[:,0][:,None], stds_nu, k_dist_nu, 1)
        mu[i] = compute_kernel(points[i,1:3], backgrounds[:,1:3], stds_mu, k_dist_mu, 1)
        p[i,i] = nu[i]*mu[i]
        if i>0:
            for j in range(0, i):
                g[j, i] = compute_kernel(points[j][:,None], points[i][:,None], stds_trig, k_dist_trig, 1)
                p[j, i] = g[j, i]
    print(p)
    p = _normalise_matrix(p)
np.save('F:/Crimes/sepp.npy',[backgrounds, triggered, stds_nu, stds_mu, stds_trig, k_dist_nu, k_dist_mu, k_dist_trig, num_backgrounds, num_triggered])
test_point1 = np.array([[3930, 116.483187, 39.956]])
test_point2 = np.array([[3950, 116.483187, 39.956]])
pred1 = predict(test_point1, backgrounds, triggered, stds_nu, stds_mu, stds_trig, k_dist_nu, k_dist_mu, k_dist_trig, num_backgrounds, num_triggered)
pred2 = predict(test_point2, backgrounds, triggered, stds_nu, stds_mu, stds_trig, k_dist_nu, k_dist_mu, k_dist_trig, num_backgrounds, num_triggered)
print(pred1)
print(pred2)
#print(pred1/pred2)
f1 = plt.figure(1)
plt.scatter(points[:,1], points[:,2])
plt.show()
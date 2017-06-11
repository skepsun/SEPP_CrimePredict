# -*- coding: utf-8 -*-
"""
Created on Fri Jun  9 15:44:42 2017

@author: scx
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import csv
import datetime
import math
from . import kernels

def cleaner(df, index):
    # param: df, type: dataframe
    # param: index
    point_size = len(df)
    cleaned_df = df.drop(index,axis=1)
    timestamps = np.zeros((point_size), dtype='str')
    cleaned_df['Dates'] = cleaned_df['Dates'].apply(lambda x: np.datetime64(x))
    timedpoints = cleaned_df.values
    return timedpoints

def read_data(file_path, index, category):
    raw = pd.read_csv(file_path)
    cleaned_data = cleaner(raw, index)
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

    number_data_points = points.shape[-1]
    print (p)
    choice = np.array([ np.random.choice(j+1, p=p[0:j+1, j])
        for j in range(number_data_points) ])
    print(choice)
    mask = ( choice == np.arange(number_data_points) )

    backgrounds = points[:,mask]
    triggered = (points - points[:,choice])[:,~mask]
    return backgrounds, triggered

def initial_p_matrix(points, initial_time_bandwidth = 0.1,
        initial_space_bandwidth = 50.0):
    """Returns an initial estimate of the probability matrix.  Uses a Gaussian
    kernel in space, and an exponential kernel in time, both non-normalised.
    Diagonal (i.e. background "probabilities") are set to 1.  Finally the
    matrix is normalised.

    :param points: The (time, x, y) data.
    :param initial_time_bandwidth: The "scale" of the exponential.
    :param initial_space_bandwidth: The standard deviation of the Gaussian.
    """
    def bkernel(pts):
        return _np.zeros(pts.shape[-1]) + 1
    def tkernel(pts):
        time = _np.exp( - pts[0] / initial_time_bandwidth )
        norm = 2 * initial_space_bandwidth ** 2
        space = _np.exp( - (pts[1]**2 + pts[2]**2) / norm )
        return time * space
    return p_matrix(points, bkernel, tkernel)

def p_matrix_fast(points, background_kernel, trigger_kernel, time_cutoff=150, space_cutoff=1):
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
    p += _np.diag(background_kernel(points))
    return _normalise_matrix(p)

def _normalise_matrix(p):
    column_sums = np.sum(p, axis=0)
    return p / column_sums[None,:]

def compute_kernel(t, x, y, mean, var, D, N)
    result = 0
    for i in range(0,timedpoints.shape[0]):
        delta = 1
        for j in range(0,len(mean)):
            delta = delta/var[j]*exp(-(x-mean[i])**2/(2*var[j]**2*D[i]**2))
        delta = delta/(2*np.pi)**(1.5)/D[i]**3
        result = result+delta/N
    return result

def compute_kth_distance(triggered, k)
    stds = np.std(triggered, axis = 1)
    scaled_trigger = triggered/stds
    tree = scipy.KDTree(scaled_trigger)
    kth_distance = np.empty(scaled_trigger.shape[0])
    for i,p in enumerate(scaled_trigger):
        distances, indexes = tree.query(p, k=k+1)
        kth_distance = distances[-1]
    return [kth_distance, stds]

def main()
    iter = 20
    k_high = 100
    k_low = 15
    file_path = 'case_new'
    points = read_data(file_path)
    num_points = len(points)
    p_matrix = initial_p_matrix(points, initial_time_bandwidth = 0.1,
            initial_space_bandwidth = 50.0)
    for i in range(0,iter):
        [backgrounds, triggered] = sample_points(points,p_matrix)
        N = backgrounds.shape[0]+triggered.shape[0]
        [k_dist_trig, stds_trig] = compute_kth_distance(triggered, k_low)
        [k_dist_nu, stds_nu] = compute_kth_distance(backgrounds[:,0], k_high)
        [k_dist_mu, stds_mu] = compute_kth_distance(backgrounds[:,1:2], k_low)
        trig = compute_kernel(points(0), points(1), points(2), trigger, k_dist_trig, N)
        mu = compute_kernel(points(0), points(1), points(2), backgrounds[:,1:2], k_dist_mu, N)
        nu = compute_kernel(points(0), points(1), points(2), backgrounds[:,0], k_dist_nu, N)
        lamb = mu*nu+trig
        for i in range(0,num_points):
            p_matrix(i,i) = nu(i)*mu(i)/lamb(i)
            for j in range(i+1, num_points):
                p_matrix(i,j) = trig(j)

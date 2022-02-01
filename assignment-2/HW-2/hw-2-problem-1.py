#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 27 08:50:08 2022

@author: milos
"""

#### stats basics: calculate sample mean, standard error and a confidence interval

## sample
import numpy as np
import pandas as pd
import scipy

#### the function calculates basic sample stats: sample mean, standard error and a confidence interval
def confidence_interval(sample, confidence_level):
    # the function calculates the sample mean, sample standard error, and confidence interval
    degrees_freedom = sample.size - 1
    sample_mean = np.mean(sample)
    sample_standard_error = scipy.stats.sem(sample)
    # compute confidence interval for `sample`
    confidence_interval = scipy.stats.t.interval(confidence_level, degrees_freedom, sample_mean, sample_standard_error)
    print('Sample mean', sample_mean)
    print('Sample standard error:', sample_standard_error)
    print('Confidence interval:',confidence_interval)
    return sample_mean, sample_standard_error, confidence_interval

# loading the data from the text file
dataframe = pd.read_csv("mean_study_data.txt")
array = dataframe.values
sample = array[:,0]
# set confidence_level
confidence_level = 0.95
# calculate basic sample stats for all examples
samp_mean, samp_stde, conf_interval=confidence_interval(sample,confidence_level)

print(samp_mean, samp_stde, conf_interval)




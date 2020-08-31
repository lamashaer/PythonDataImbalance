import csv

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import imblearn.over_sampling

filename = "quadX_norm_out_dd6.txt"
X = np.loadtxt(filename)

filename2 = "y_out_dd6.txt"
y = np.loadtxt(filename2)



countone = np.count_nonzero(y)
countzero = len(y) - countone
print(countone)
print(countzero)
print(len(y))

seed = 100

sm = imblearn.over_sampling.SMOTE(sampling_strategy= {1:600, 0:1400}, k_neighbors= 5, random_state=seed)
X_res, y_res = sm.fit_resample(X, y)

file = open("quadX_norm_resamplefinaldd6.txt","w")
np.savetxt('quadX_norm_resamplefinaldd6.txt', X_res)

file = open("yout_resamplefinaldd6.txt","w")
np.savetxt('yout_resamplefinaldd6.txt', y_res)
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 19 11:26:31 2018

@author: shkim
"""
import numpy as np

x1= np.array([[7,7,6,7,7,7],[6,6,6,6,6,5],
              [1,1,1,1,1,1],[5,5,5,5,5,4],
              [2,2,2,2,2,1],[4,4,4,4,4,3]])
x2= np.array([[1,1,1,2,2,2],[3,3,3,4,4,4]])

distances = np.sum(np.abs(x1 - x2[0, :]), axis=1)
print(distances)
print(np.argmin(distances), np.argmax(distances))  # 2 0

distances = np.sum(np.abs(x1 - x2[1, :]), axis=1)
print(distances)
print(np.argmin(distances), np.argmax(distances))  # 5 0

x1= np.array([[7,7,6,7,7,7],[6,6,6,6,6,5],
              [1,1,1,1,1,1],[5,5,5,5,5,4],
              [2,2,2,2,2,1],[4,4,4,4,4,3]])
x2= np.array([[1,1,1,2,2,2],[3,3,3,4,4,4]])
for i in range(x2.shape[0]):
    distances = np.sum(np.abs(x1 - x2[i, :]), axis=1)
    print(distances)
    print(np.argmin(distances), np.argmax(distances))
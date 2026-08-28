# -*- coding: utf-8 -*-
"""
Created on Mon Sep  2 13:38:04 2024

@author: allro
"""

import numpy as np
from sklearn.decomposition import NMF

C = np.array([[5, 4, 5, 4], 
             [2, 3, 3, 2], 
             [3, 0, 3, 5], 
             [0, 2, 5, 1], 
             [4, 4, 2, 3], 
             [4, 4, 1, 1], 
             [3, 5, 0, 3], 
             [5, 1, 0, 3], 
             [3, 0, 3, 0]])
D = np.array([[1, 2, 1, 5, 1, 4, 3, 3, 1],
             [2, 2, 5, 1, 0, 0, 4, 1, 2],
             [5, 1, 3, 3, 2, 3, 2, 4, 1],
             [1, 2, 1, 3, 1, 0, 4, 0, 0]])

A = C @ D
print('Matrix A:', A)
rankA = np.linalg.matrix_rank(A)
print('Rank of A:', rankA)

for i in range(4, 9):
    model = NMF(n_components=i, init='random', random_state=0, max_iter=1000)
    W = model.fit_transform(A)
    H = model.components_
    error = np.linalg.norm(A - W @ H, 'fro')
    print(f"i = {i}")
    print(f"W_{i} =\n{W}")
    print(f"H_{i} =\n{H}")
    print(f"Error_{i} = {error:.5f}\n")
    
B = D @ C
print('Matrix B:', B)
rankB = np.linalg.matrix_rank(B)
print('Rank of B:', rankB)

for j in range(2, 4):
    model = NMF(n_components=j, init='random', random_state=0, max_iter=1000)
    V = model.fit_transform(B)
    Q = model.components_
    error = np.linalg.norm(B - V @ Q, 'fro')
    print(f"j = {j}")
    print(f"W_{j} =\n{W}")
    print(f"H_{j} =\n{H}")
    print(f"Error_{j} = {error:.5f}\n")
    



# coding=utf-8

import numpy as np
from csl import CSL
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
X=pd.read_csv("./data/sim1.csv")[0:1000].values
true_dag=pd.read_csv("./data/ground truth1.csv").values
scaler = MinMaxScaler()
scaler.fit(X)
X = scaler.transform(X)
X=np.diff(X,axis=0)
ga = CSL(input_dim=5)
ga.learn(X)
print(ga.causal_matrix)
print(true_dag)
TP=0
FP=0
FN=0
TN=0
for i in range(true_dag.shape[0]):
        for j in range(true_dag.shape[1]):
                if ga.causal_matrix[i][j]==1 and true_dag[i][j]==1:
                        TP=TP+1
                elif ga.causal_matrix[i][j]==1 and true_dag[i][j]==0:
                        FP = FP + 1
                elif ga.causal_matrix[i][j]==0 and true_dag[i][j]==1:
                        FN = FN + 1
                else:
                        TN = TN + 1

recall=TP/(TP+FN)
precision=TP/(TP+FP)
print("precision")
print(precision)
print("recall")
print(recall)
print("F-score")
print(2*recall*precision/(recall+precision))

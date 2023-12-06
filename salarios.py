# -*- coding: utf-8 -*-
"""
@author: fred_
"""

import pandas as pd

base = pd.read_csv("salarios.csv")
previsores = base.iloc[:,0:1].values
classe = base.iloc[:,1].values

base.corr()

import matplotlib.pyplot as plt
plt.scatter(previsores, classe)
plt.title('Salário x Experiência')
plt.xlabel('Anos de Experiência')
plt.ylabel('Salário')
plt.show()

from sklearn.model_selection import train_test_split 
previsores_treino, previsores_teste, classe_treino, classe_teste = train_test_split(previsores, classe, test_size=0.20, random_state=0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(previsores_treino, classe_treino)
    
previsao = regressor.predict([[5]])

import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_squared_error
from math import sqrt
kfold = StratifiedKFold(n_splits = 5, shuffle=True, random_state = 5)
resultadosRL = []
palpitesRL = []
realRL = []
for indice_treinamento, indice_teste in kfold.split(previsores, np.zeros(shape=[previsores.shape[0],1])):
    regressor = LinearRegression()
    regressor.fit(previsores[indice_treinamento], classe[indice_treinamento])
    previsoesRL = regressor.predict(previsores[indice_teste])
    rmse = sqrt(mean_squared_error(classe[indice_teste], previsoesRL))
    palpitesRL.append(previsoesRL)
    realRL.append(classe[indice_teste])    
    resultadosRL.append(rmse)

pRL = []
rRL = []
for palpite in palpitesRL:
    for pal in palpite:
        pRL.append(pal)

for realidade in realRL:
    for reali in realidade:
        rRL.append(reali)
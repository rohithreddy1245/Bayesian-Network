import numpy as np
import pandas as pd

from pgmpy.module import DiscreteBayesianNetwork
from pgmpy.interference import ValueElimination
from pgmpy.estimators import MaximumLikelihoodEstimators , BayesianEstimators

heartDisease = pd.Dataframe({
    'age' : [25,50,44,34,29],
    'fbs' : [0,1,0,1,0],
    'chol' : [170,240,230,180,155],
    'restecg' : [0,1,1,0,1],
    'target' : [0,1,0,1,0],
    'thalach' : [150,140,110,120,135]
})

heartDisease = heartDisease.replace('?','np.nan')

model = DiscreteBayesianNetwork([
    ('age','fbs'),
    ('fbs','target'),
    ('target','restecg'),
    ('target','thalach'),
    ('target','chol')
])

model.fit(heartDisease, estimator = MaximumLikelihoodEstimator)

HeartDisease_infer = VariableElimination(model)

q = HeartDisease_infer.query(variables=['target'],evidence={'age': 34})

print(q)

from Perceptron import Perceptron as Per
import numpy as np

data_X=np.array([[3,3],[4,3],[1,1]])
data_Y=np.array([1,1,-1])
per=Per(data_X,data_Y,1)
per.train()
to_predit=np.array([3,1])
per.predict(to_predit)
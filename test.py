from SplineInterpolation import SplineInterpolation as si
import numpy as np


#data=np.array([[1,1],[2,3],[4,4],[5,2]])
#data=np.array([[75,2.768],[76,2.833],[77,2.903],[78,2.979],[79,3.062],[80,3.153]])
data=np.array([[0.25,0.5],[0.3,0.5477],[0.39,0.6245],[0.45,0.6708],[0.53,0.728]])
cond=np.array([1,0.6868])
#cond=np.array([0,0])
Si=si(len(data),data,1,cond)
x=0.35
print(Si.predict(x))
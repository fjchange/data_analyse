import numpy as np

def sign(x):
    if x>=0:return 1
    else:return -1

class Perceptron:
    def __init__(self,data_X,data_Y,step):
        self.w=np.zeros(len(data_X[0]))
        self.b=0
        self.X=data_X
        self.Y=data_Y
        self.learning_step=step

    def train(self):
        tag=True
        while(tag):
            tag=False
            for i in range(0,len(self.X)):
                if(self.Y[i]*(np.inner(self.w[:],self.X[i][:])+self.b)<=0):
                    self.w=self.w+self.learning_step*self.Y[i]*self.X[i]
                    self.b=self.b+self.learning_step*self.Y[i]
                    tag=True
        print("the weight"+self.w.__str__())
        print("the bias "+self.b.__str__())
    def predict(self,x):
        print(sign(np.inner(self.w,x)+self.b))


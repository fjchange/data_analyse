import numpy as np
import math

class SplineInterpolation:
    def __init__(self,data_num,data,condition_type,cond):
        self.data_num=data_num
        self.data=data
        self.X=data.transpose()[0]
        self.Y=data.transpose()[1]
        self.condition_type=condition_type
        self.cond=cond
        self.alpha=np.zeros(data_num)
        self.beta=np.zeros(data_num)
        self.h=self.X[1:self.data_num]-self.X[:self.data_num-1]

        if self.condition_type==1:
            self.alpha[0]=0
            self.beta[0]=2*self.cond[0]
            self.alpha[self.data_num-1]=1
            self.beta[self.data_num-1]=2*self.cond[1]
        else:
            self.alpha[0] = 1
            self.beta[0] = 3/self.h[0]*(self.Y[1]-self.Y[0])
            self.alpha[self.data_num-1] = 0
            self.beta[self.data_num-1] = 3/self.h[self.data_num-2]*(self.Y[self.data_num-1]-self.Y[self.data_num-2])

    def binary_search(self):
        if(self.x<self.X[0] or self.x>self.X[self.data_num-1]):
            return
        j=len(self.data)
        i=0

        while (1):
            mid=math.floor((i+j)/2)
            if(i==mid):return i
            elif(j==mid):return mid-1
            if(self.x>=self.X[mid] and self.x<=self.X[mid+1]):
                return mid
            elif(self.x<self.X[mid]):
                j=mid
            elif(mid==self.data_num or mid==self.data_num-1):
                return self.data_num-1
            else :
                i=mid

    def predict(self,x):
        self.x=x
        self.alpha[1:self.data_num-1]=np.true_divide(self.h[:self.data_num-2],self.h[:self.data_num-2]+self.h[1:self.data_num-1])
        self.beta[1:self.data_num-1]=3*(np.true_divide((1-self.alpha)[1:self.data_num-1],self.h[:self.data_num-2])*(self.Y[1:self.data_num-1]-self.Y[:self.data_num-2])
                                        +np.true_divide(self.alpha[1:self.data_num-1],self.h[1:self.data_num-1])*(self.Y[2:self.data_num]-self.Y[1:self.data_num-1]))

        self.a=np.zeros(self.data_num)
        self.b=np.zeros(self.data_num)

        self.a[0]=-self.alpha[0]/2
        self.b[0]=self.beta[0]/2

        for i in range(1,self.data_num):
            self.a[i]=-self.alpha[i]/(2+(1-self.alpha[i])*self.a[i-1])
            self.b[i]=(self.beta[i]-(1-self.alpha[i])*self.b[i-1])/(2+(1-self.alpha[i])*self.a[i-1])


        self.m=np.zeros(self.data_num+1)

        for i in range(1,self.data_num+1):
            self.m[self.data_num-i]=self.a[self.data_num-i]*self.m[self.data_num+1-i]+self.b[self.data_num-i]

        tmp=self.binary_search()


        return (1+2*(x-self.X[tmp])/self.h[tmp])*((x-self.X[tmp+1])/self.h[tmp])**2*self.Y[tmp]\
                +(1-2*(x-self.X[tmp+1])/self.h[tmp])*((x-self.X[tmp])/self.h[tmp])**2*self.Y[tmp+1]\
                +(x-self.X[tmp])*((x-self.X[tmp+1])/self.h[tmp])**2*self.m[tmp]\
                +(x-self.X[tmp+1])*((x-self.X[tmp])/self.h[tmp])**2*self.m[tmp+1]

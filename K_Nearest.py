import numpy as np
import math
class K_Nearest:
    def __init__(self,k,distance_func,data,y):
        self.data=data
        self.Y=y
        self.k=k
        self.distance_func=distance_func

    def distance(self,p1,p2):
        if(self.distance_func==1):
            return sum(abs(p1-p2))
        elif(self.distance_func==2):
            return math.sqrt(sum((p1-p2)**2))
        else:
            return max(abs(p1-p2))

    def find_K(self,p):
        K_Near_Set=np.array(self.k)
        K_Distance=list(len(self.k))
        for i in range(self.k):
            K_Distance[i]=self.distance(K_Near_Set[i],p)
        for i in range(self.k,len(self.data[0])):
            tmp=self.distance(self.data[i],p)
            if(tmp<max(K_Distance)):
                index=K_Distance.index(max(K_Distance))
                K_Distance[index]=tmp
                K_Near_Set[index]=i



#coding:utf-8
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import operator
from enum import Enum
file_path="H:\\Users\\kiwi feng\\Desktop\\data_set\\"

likeness = {
    b'didntLike':1,
    b'smallDoses':2,
    b'largeDoses':3
}

def text_to_matrix(path):
    text_file=open(path,'rb')
    lines=text_file.readlines()
    num_line=len(lines)
    matrix=np.zeros((num_line,3))
    classLabels=[]
    index=0

    for line in lines:
        line.strip()

        listFromLine=line.split('\t'.encode('gbk'))
        matrix[index][:]=listFromLine[:3]

        classLabels.append(likeness[listFromLine[-1].strip()])
        index+=1
    return matrix,classLabels

def auto_norm(data_set):
    minVals=data_set.min(0)
    maxVals=data_set.max(0)
    ranges=maxVals-minVals
    norm_dataset=np.zeros(data_set.shape)
    m=data_set.shape[0]
    norm_dataset=data_set-np.tile(minVals,[m,1])
    norm_dataset=norm_dataset/np.tile(ranges,[m,1])
    return norm_dataset,ranges,minVals

def classfy0(inX,dataSet,labels,k):
    dataSetSize=dataSet.shapw[0]
    diffMat=np.tile(inX,(dataSetSize,1))-dataSet
    sqDiffMat=diffMat**2
    sqDistances=sqDiffMat.sum(axis=1)
    distances=sqDistances**0.5
    sortedDistIndicies=distances.argsort()
    classCount={}
    for i in range(k):
        voteIlabel=labels[sortedDistIndicies[i]]
        classCount[voteIlabel]=classCount.get(voteIlabel,0)+1
    sortedClassCount=sorted(classCount.iteritems(),key=operator.itemgetter(1),reverse=True)
    return sortedClassCount[0][0]

def datingClassTest():
    hoRatio=0.10
    datingDataMat,datingLabels=text_to_matrix(path+'datingTestSet.txt')
    normMat,ranges,minVals=auto_norm(datingDataMat)
    m=normMat.shape[0]
    numTestVecs=int(m*hoRatio)
    errorCount=0.0
    for i in range(numTestVecs):
        classfierResult=classfy0()

path=file_path+"datingTestSet.txt"
fig=plt.figure()
ax=fig.add_subplot(111)
Matrix,labels=text_to_matrix(path)
Matrix,ranges,minVals=auto_norm(Matrix)
print(Matrix)
label=['didntLike','smallDoses','largeDoses']
ax.scatter(Matrix[:,1],Matrix[:,2],c=labels,s=45,alpha=0.75)
plt.show()

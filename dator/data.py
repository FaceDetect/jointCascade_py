import os
import numpy as np
import copy
import math
from utils  import *     
from reader import *
from shape  import *
from bootstrap import *
  
class PosSet(object):
    def __init__(self, winSize):        
        self.winSize    = winSize
        self.dataNum  = 0        
        self.pntNum   = 0
        self.imgDatas = []
        self.gtShapes = []
        self.initShapes = None
        self.meanShape  = None
        self.residuals  = None
        self.Ws = None
        self.confs = None
        
    def add(self, img, gtShape):
        self.imgDatas.append(img)
        self.gtShapes.append(gtShape)

    def calMeanShape(self):
        meanShape = np.zeros((self.pntNum, 2), 
                             dtype=np.float32)
        for i in xrange(self.dataNum):
            meanShape = np.add(meanShape, self.gtShapes[i])
        self.meanShape = meanShape/self.dataNum

    def config(self):
        ### Convert the list into array
        self.gtShapes = np.asarray(self.gtShapes,
                                   dtype = np.float32)
        self.imgDatas = np.asarray(self.imgDatas,
                                   dtype = np.float32) 
        self.dataNum, self.pntNum = self.gtShapes.shape[0:2]

        self.calMeanShape()
        self.Ws = np.zeros(self.dataNum, dtype=np.float32)
        self.Ws[:] = 1.0/self.dataNum
        self.confs = np.zeros(self.dataNum, dtype=np.float32)
        self.initShapes = np.zeros(self.gtShapes.shape,
                                   dtype = np.float32)

        ### Initialize the initShapes
        for i in xrange(self.dataNum):
            self.initShapes[i]=Shape.augment(self.meanShape)
        return
    
    def calResiduals(self):  
        self.residuals = np.zeros(self.gtShapes.shape)
        for i in range(self.dataNum):
            ### Project to meanshape coordinary       
            err = self.gtShapes[i]-self.initShapes[i]
            err = np.divide(err, (self.winSize[0], 
                                  self.winSize[1]))
            self.residuals[i,:] = err

    def refineData(self, th):
        idx = self.confs > th
        self.dataNum = np.sum(idx)
        self.imgDatas = self.imgDatas[idx]
        self.gtShapes = self.gtShapes[idx]
        self.initShapes = self.initShapes[idx]
        self.residuals  = self.residuals[idx]
        self.confs = self.confs[idx]
        self.Ws = np.exp(-self.confs)

class NegSet(object):
    def __init__(self, meanShape, winSize):
        self.winSize    = winSize
        self.meanShape  = meanShape
        self.dataNum    = 0 
        self.imgDatas   = None
        self.initShapes = None
        self.Ws = None
        self.confs = None

    def refineData(self, th):
        idx = self.confs > th
        self.dataNum = np.sum(idx)
        self.imgDatas = self.imgDatas[idx]
        self.initShapes = self.initShapes[idx]
        self.confs = self.confs[idx]
        self.Ws = np.exp(self.confs)

class DataWrapper(object):
    def __init__(self): 
        ### Config Para        
        self.winSize = None
        self.negListPath = None
        self.posListPath = None   
        self.bootstrap   = None

    def config(self, paras):
        self.winSize = paras['winSize']
        self.posListPath = paras['posList']
        
        if paras['negList'] != self.negListPath:
            bsPara = paras["bootstrapPara"]
            self.negListPath = paras['negList']
            if not os.path.exists(self.negListPath):
                raise Exception("Neg JPG list not exist") 
            negImgList = open(self.negListPath,
                              'r').readlines()
            bsPara['winSize'] = self.winSize
            bsPara['negImgList'] = negImgList
            self.bootstrap = Bootstrap(bsPara)
            
    def printParas(self):
        print('\tWindow Size    = %s'%(str(self.winSize)))
        print('\tPositive Path  = %s'%(self.posListPath))
        print('\tNegative Path  = %s'%(self.negListPath))
        print('\tBootstrap Paras')
        self.bootstrap.printParas()

    def genTrainSet(self):
        ### Load positive train set
        posSet = PosSet(self.winSize) 
        if not os.path.exists(self.posListPath):
            raise Exception("Positive set not exist")      
        paths = open(self.posListPath).readlines()
        for imgP in paths:
            img, gtShape = Reader.read(imgP, self.winSize)
            posSet.add(img, gtShape)               
        posSet.config()
        
        ### Load negative train set
        negSet = NegSet(posSet.meanShape, self.winSize)
        return posSet, negSet, self.bootstrap

    

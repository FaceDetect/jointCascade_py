import sys
import os
import numpy as NP
from scipy.sparse import lil_matrix
from sklearn.svm import LinearSVR
import time
from utils   import *
from cart  import  *
from dator import  *
import random

class BoostCart(object):
    """
    Boost Classification and Regression Tree
    """
    def __init__(self, paras):     
        self.weakCRNum  = paras["weakCRNum"]
        self.npRatio    = paras["npRatio"]
        self.tpRates    = paras["tpRates"]
        self.fpRates    = paras["fpRates"]
        self.treeDepth  = paras["treeDepth"]
        self.feaNum     = paras["feaNum"]
        self.radius     = paras["radius"]
        self.cProb      = paras["cProb"]
        self.carts      = []
        self.preCarms   = None
        self.globalReg  = []

    def train(self, posSet, negSet, bootstrap):
        """
        return : 0 -- OK
                 1 -- the negative sample is not enough
        """
        bNegIsNotEnough = 0
        ### Train the weak CART        
        for i in xrange(self.weakCRNum):
            self.feaDim = self.getFeaDim()
            print("\t\t%drd weaker begin ..."%i)

            ### Find Neg samples
            posNum = posSet.dataNum
            negNum = posNum * self.npRatio
            needNum = negNum - negSet.dataNum
            print("\t\tCurrent Pos Num : %d, Neg Num : %d, Need : %d"%(posNum, negSet.dataNum, needNum))
            
            begTime = time.time()
            findNum, consumed = self.getNegImgData(negSet,
                                                   needNum,
                                                   bootstrap)
            t = getTimeByStamp(begTime, 
                               time.time(), 'hour')
            s ="\n\t\tConsumed:%d, Find:%d, FP:%.2f, Time:%fh"
            print(s%(consumed, findNum, 
                     findNum/(consumed+1e-6), t))
            if negSet.dataNum < negNum/2:
                bNegIsNotEnough = 1
                break
            elif negSet.dataNum < negNum:
                bNegIsNotEnough = 1                

            ### Normalize the weight
            posSet.Ws = posSet.Ws/NP.sum(posSet.Ws)
            negSet.Ws = negSet.Ws/NP.sum(negSet.Ws)
            
            ### Train the CART
            pntIdx = i%posSet.pntNum
            cart = CART(self.treeDepth, 
                        self.feaNum, self.radius)
            bIsCls = random.random()<self.cProb
            cart.train(posSet, negSet, pntIdx, bIsCls)    
            

            ### Set the th
            thIdx = int((1-self.tpRates[i])*posSet.dataNum)
            eps = NP.finfo(NP.float32).eps
            th = NP.sort(posSet.confs)[thIdx] - eps
            curFP = NP.sum(negSet.confs>th)/(negSet.dataNum*1.0)
            cart.th = th
            self.carts.append(cart)
            ### Remove the Samples which is < th 
            negSet.refineData(th)
            posSet.refineData(th)            
            
            print("\t\tCurrent FP : %f"%curFP)
            print("\t\t%drd weaker end ...\n"%i)
            if 1==bNegIsNotEnough:
                break
        self.globalRegress(posSet, negSet)        
        return bNegIsNotEnough

    def globalRegress(self, posSet, negSet):
        self.feaDim = self.getFeaDim()
        ### Extract the local binary features
        begTime = time.time()
        posFeas = self.genFeaOnTrainset(posSet)
        negFeas = self.genFeaOnTrainset(negSet)
        t = getTimeByStamp(begTime, time.time(), 'min')
        print("\t\tExtract LBFs      : %f mins"%t)

        ### Global regression
        begTime = time.time()
        y = posSet.residuals
        y = y.reshape(y.shape[0], y.shape[1]*y.shape[2])
        for i in xrange(posSet.pntNum*2):
            ### TODO Show the training result 
            reg=LinearSVR(epsilon=0.0, 
                          C = 1.0/posFeas.shape[0],
                          loss='squared_epsilon_insensitive',
                          fit_intercept = True)
            reg.fit(posFeas, y[:, i])
            self.globalReg.append(reg)
        t = getTimeByStamp(begTime, time.time(), 'min')
        print("\t\tGlobal Regression : %f mins"%t)

        ### Update the initshapes
        begTime = time.time()
        for i in xrange(posSet.pntNum):
            regX = self.globalReg[2*i]
            regY = self.globalReg[2*i+1]
            
            x = regX.predict(posFeas)
            y = regY.predict(posFeas)
            delta = NP.squeeze(NP.dstack((x,y)))
            delta = NP.multiply(delta,
                                posSet.winSize)
            posSet.initShapes[:,i,:] = posSet.initShapes[:,i,:] + delta
            x = regX.predict(negFeas)
            y = regY.predict(negFeas)
            delta = NP.squeeze(NP.dstack((x,y)))
            delta = NP.multiply(delta,
                                negSet.winSize)
            negSet.initShapes[:,i,:] = negSet.initShapes[:,i,:] + delta
        t = getTimeByStamp(begTime, time.time(), 'min')
        print("\t\tUpdate Shape      : %f mins"%t)

    def getNegImgData(self, negSet, needNum, bootstrap):
        findNum  = 0
        consumed = 0
        if needNum < 1:
            return findNum, consumed
        
        data = NP.zeros((needNum,
                         negSet.winSize[0],
                         negSet.winSize[1]),
                        dtype = NP.uint8)
        shape = NP.zeros((needNum, 
                          negSet.meanShape.shape[0],
                          negSet.meanShape.shape[1]),
                         dtype = NP.float32)
        Ws = NP.zeros(needNum, dtype=NP.float32)
        confs = NP.zeros(needNum, dtype=NP.float32)
                
        if not sys.stdout.isatty():
            self.dataWrapper.logOffset = sys.stdout.tell()
            
        while True:
            flag, img, rect= bootstrap.nextDataFromJPG()
            if 0 == flag:
                break
            
            consumed += 1
            ### Validate the image 
            initShape  = Shape.augment(negSet.meanShape)
            flag, conf = self.validate(img[0], rect, initShape)
            if 1 == flag:
                data[findNum] = img[0, 
                                    rect[1]:rect[1]+rect[3],
                                    rect[0]:rect[0]+rect[2]]
                shape[findNum] = initShape
                confs[findNum] = conf
                Ws[findNum] = NP.exp(conf)
                findNum += 1                
                if findNum >= needNum :
                    break        

        ### Merge with the previous
        if negSet.dataNum > 0:
            negSet.imgDatas=NP.concatenate((negSet.imgDatas,
                                            data[0:findNum]))
            negSet.initShapes=NP.concatenate((negSet.initShapes,
                                              shape[0:findNum]))
            negSet.Ws = NP.concatenate((negSet.Ws,
                                        Ws[0:findNum]))
            negSet.confs = NP.concatenate((negSet.confs,
                                           confs[0:findNum]))
        else:
            ### The first weaker of first stage
            negSet.imgDatas = data[0:findNum]
            negSet.initShapes = shape[0:findNum]
            negSet.Ws = Ws[0:findNum]
            confs[0:findNum] = 1.0/findNum
            negSet.confs = confs[0:findNum]
        negSet.dataNum = negSet.imgDatas.shape[0]
        return findNum, consumed
        
    def getFeaDim(self):
        feaDim = 0
        for t in self.carts:
            feaDim = feaDim + t.leafNum
        return feaDim

    def genFeaOnTrainset(self, trainSet):
        sampleNum = trainSet.dataNum
        feas = lil_matrix((sampleNum, self.feaDim), 
                          dtype=NP.int8)
        
        for i in xrange(sampleNum):
            imgData = trainSet.imgDatas[i] 
            shape   = trainSet.initShapes[i]
            
            offset = 0
            for j, t in enumerate(self.carts):
                leafIdx, dim = t.genBinaryFea(imgData, 
                                              shape)
                feas[i, offset+leafIdx] = 1
                offset = offset + dim                    
        return feas

    def validate(self, img, rect, initShape):
        flag = 1
        conf = 0
        if None != self.preCarms:
            flag, conf = self.preCarms.validate(img,
                                                rect,
                                                initShape)
        if flag != 1:
            return flag, conf
        fea = lil_matrix((1, self.feaDim), 
                         dtype=NP.int8)
        offset = 0
        for cart in self.carts:
            _conf, leafIdx, dim = cart.validate(img,
                                                rect,
                                                initShape)
            conf += _conf
            if conf < cart.th:
                flag = 0
                break
            fea[0, offset+leafIdx] = 1
            offset = offset + dim    

        if len(self.globalReg) > 0:
            pntNum = initShape.shape[0]
            ### Get the residules
            for i in xrange(pntNum):
                regX = self.globalReg[2*i]
                regY = self.globalReg[2*i+1]
            
                x = regX.predict(fea)
                y = regY.predict(fea)
                delta = NP.squeeze(NP.dstack((x,y)))
                delta = NP.multiply(delta, 
                                    (rect[2],rect[3]))
                initShape[i,:] = initShape[i,:] + delta
        return flag, conf


            
        

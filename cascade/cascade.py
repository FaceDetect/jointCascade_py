import sys
import os
import numpy as NP
import time
import pickle
from utils   import *
from dator   import *
from carm    import *
import copy

class JointCascador(object):
    """
    Joint Cascade for face detection and alignment
    """
    def __init__(self):     
        self.winSize  = None
        self.name     = None
        self.version  = None
        self.stageNum = None
        
        self.dataWrapper = None
        self.carmWrapper  = None
        self.carms  = []
        self.meanShape   = None
        self.curStageIdx = 0

    def printParas(self):
        print('------------------------------------------')
        print('----------   Configuration    ------------')
        print('Name           = %s'%(self.name))
        print('Version        = %s'%(self.version))
        print('Stage Num      = %s'%(self.stageNum))        
        print('\n-- CARM Config --')
        self.carmWrapper.printParas()
        print('\n-- Data Config --')
        self.dataWrapper.printParas()
        print('---------   End of Configuration   -------')
        print('------------------------------------------\n')
                   
    def config(self, paras):
         self.name     = paras['name']
         self.version  = paras['version']
         self.stageNum = paras['stageNum']

         ### Construct the data wrapper
         self.dataWrapper = DataWrapper()
         self.dataWrapper.config(paras['dataPara'])

         ### Construct the carm wrapper
         self.carmWrapper = CarmWrapper(paras['carmPara'])

    def train(self, save_path):
        ### Make model folder for train model
        if not os.path.exists('%s/model'%(save_path)):
            os.mkdir('%s/model'%(save_path))
        
        ### Read data first 
        begTime = time.time()
        posSet, negSet, bootstrap = self.dataWrapper.genTrainSet()        
        self.meanShape = posSet.meanShape
        self.winSize   = posSet.winSize
        t = getTimeByStamp(begTime, 
                           time.time(), 'min')
        print("\tLoading Positive Data : %f mins"%(t))       
        posSet.calResiduals()
        sumR = NP.mean(NP.abs(posSet.residuals))
        print("\tManhattan Distance : %f\n"%sumR)

        idx = self.curStageIdx
        while idx < self.stageNum:
            print("\t%drd stage begin ..."%idx)
            begTime = time.time()
            
            ### Train one stage
            carm = self.carmWrapper.getClassInstance(idx)
            if len(self.carms) > 0:
                carm.preCarms = self.carms[-1]
            flag = carm.train(posSet, negSet, bootstrap)
            self.carms.append(carm) 
            
            ### Calculate the residuals
            posSet.calResiduals()
            sumR = NP.mean(NP.abs(posSet.residuals))
            print("\tManhattan Distance : %f"%sumR)

            t = getTimeByStamp(begTime, 
                               time.time(), 'min')
            print("\t%drd stage end : %f mins\n"%(idx, t))
            self.saveModel(save_path)
            idx += 1
            self.curStageIdx = idx
            if 1 == flag:
                break
        return
            
    def detect(self, imgArr, scaleF, step):
        ho, wo = imgArr.shape        
        scale = 1
        
        res = []
        confs = []        
        carm = self.carms[-1]
        while True:
            ws = (int)(wo*scale + 0.5)
            hs = (int)(ho*scale + 0.5)
            if ws<self.winSize[0] or hs<self.winSize[1]:
                break
            
            ##order : 0-nearest 1-bilinear 3-cubic(default)
            scaledArr = zoom(imgArr, (scale,scale), order=0)
            
            for h in xrange(0, hs-self.winSize[1]+1, step):
                for w in xrange(0,ws-self.winSize[0]+1,step):
                    initShape = copy.deepcopy(self.meanShape)
                    detectRect = (w, h, 
                                  self.winSize[0], 
                                  self.winSize[1])
                    flag, conf = carm.validate(scaledArr,
                                               detectRect,
                                               initShape)
                    if 1 == flag:
                        rect = (detectRect[0]/scale,
                                detectRect[1]/scale,
                                detectRect[2]/scale,
                                detectRect[3]/scale)
                        res.append(rect)
                        confs.append(conf)
            scale /= scaleF
        return res, confs
    
    def loadModel(self, model):
        path_obj = open(model, 'r').readline().strip()      
        folder = os.path.split(model)[0]
        objFile = open("%s/%s"%(folder, path_obj), 'rb')
        self = pickle.load(objFile)
        objFile.close()
        return self
        
    def saveModel(self, save_path):
        name = self.name.lower()
        model_path = "%s/model/train.model"%(save_path)
        model = open(model_path, 'w')        
        model.write("%s.pyobj"%(name))
        obj_path = "%s/model/%s.pyobj"%(save_path, name)
        
        objFile = open(obj_path, 'wb')
        pickle.dump(self, objFile)
        objFile.close()        
        model.close()
        
    


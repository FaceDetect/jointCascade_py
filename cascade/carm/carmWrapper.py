from  boostCart import *
import numpy as NP

class CarmWrapper(object):
    """
    Classifier and Regressor Machine Wrapper
    """
    def __init__(self, paras):        
        self.name        = paras['name'].upper()
        self.para        = paras['para']

    def printParas(self):        
        print('\t%-15s= %s'%('name', self.name))
        for key in self.para:
            print('\t%-15s= %s'%(key, str(self.para[key])))
                  
    def getParaBoostCart(self, idx):
        carmPara = dict()        

        length = len(self.para['weakCRNums'])
        _idx = min(idx, length-1)
        weakCRNum = self.para['weakCRNums'][_idx]
        carmPara['weakCRNum'] = weakCRNum 
        carmPara['npRatio'] = self.para['npRatio']        
        
        tpRates = NP.zeros(weakCRNum, dtype=NP.float32)
        fpRates = NP.zeros(weakCRNum, dtype=NP.float32)

        length = len(self.para['tpRates'])
        _idx = min(idx, length-1)
        length = len(self.para['tpRates'][_idx])
        tpRates[0:length] = self.para['tpRates'][_idx]
        tpRates[length:] = self.para['tpRates'][_idx][-1]
        carmPara['tpRates'] = tpRates

        length = len(self.para['fpRates'])
        _idx = min(idx, length-1)
        length = len(self.para['fpRates'][_idx])
        fpRates[0:length] = self.para['fpRates'][_idx]
        fpRates[length:] = self.para['fpRates'][_idx][-1]
        carmPara['fpRates'] = fpRates

        length = len(self.para['treeDepths'])
        _idx = min(idx, length-1)
        carmPara['treeDepth'] = self.para['treeDepths'][_idx] 

        length = len(self.para['feaNums'])
        _idx = min(idx, length-1)
        carmPara['feaNum'] = self.para['feaNums'][_idx] 

        length = len(self.para['radiuses'])
        _idx = min(idx, length-1)
        carmPara['radius'] = self.para['radiuses'][_idx] 

        length = len(self.para['cProbs'])
        _idx = min(idx, length-1)
        carmPara['cProb'] = self.para['cProbs'][_idx]         
        return carmPara

    def getClassInstance(self, idx):                   
        if "BOOSTCART"==self.name :
            carmPara  = self.getParaBoostCart(idx)
            carmClass = BoostCart
        else:
            raise Exception("Unsupport: %s "%(self.name))
        
        return carmClass(carmPara)
            

                     


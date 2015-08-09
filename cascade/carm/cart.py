import math
import numpy as NP
import random
from dator import *

class CART(object):
    def __init__(self, depth, feaNum, radius):
        ### paras
        self.depth  = depth
        self.feaNum = feaNum
        self.radius = radius

        ### tree 
        self.leafNum = 0
        self.tree    = None
        self.feaType = None
        self.tree    = None
        self.th      = 0
                
    def train(self, posSet, negSet, pntIdx, bIsCls):
        posIdxs=NP.arange(0, posSet.dataNum, dtype=NP.int32)
        negIdxs=NP.arange(0, negSet.dataNum, dtype=NP.int32)
        
        self.tree = self.split(posSet, posIdxs,
                               negSet, negIdxs, pntIdx,
                               bIsCls)
    
    def validate(self, img, rect, initShape):
        pass
    
    def split(self, posSet, posIdxs,negSet, negIdxs, 
              pntIdx, bIsCls):
        tree = {}
        if self.depth<0 or len(posIdxs)<100 or \
                (bIsCls and len(posIdxs)<100):
            ## Set the leaf index
            tree["leafIdx"] = self.leafNum            
            self.leafNum = self.leafNum+1

            ## Set the leaf prob
            posW = NP.sum(posSet.Ws[posIdxs])
            negW = NP.sum(negSet.Ws[negIdxs])
            eps = NP.finfo(NP.float32).eps
            tree["prob"] = NP.log((posW+eps)/(negW+eps),
                                  dtype=NP.float32)/2
            
            ## Extract samples' confidence
            posSet.confs[posIdxs] += tree["prob"]
            negSet.confs[negIdxs] += tree["prob"]
            return tree

        ### Generate feature types     
        feaTypes=self.genFeaType(self.feaNum, posSet.pntNum)

        ### Extract the pixel difference feature
        posFeas = self.genFea(posSet, posIdxs, feaTypes)
        negFeas = self.genFea(negSet, negIdxs, feaTypes)
        
        ### Find the best feature and threshold
        if bIsCls:
            bestIdx,th=self.bestSplitCls(posFeas,
                                         posSet.Ws[posIdxs],
                                         negFeas,
                                         negSet.Ws[negIdxs])
        else:
            errs = posSet.residuals[posIdxs, pntIdx]
            bestIdx,th=self.bestSplitReg(posFeas, errs)
        
        ### split left and right leaf recurrently
        lIdx = posFeas[:, bestIdx]<=th
        rIdx = posFeas[:, bestIdx]>th
        lPosIdxs = posIdxs[lIdx]
        rPosIdxs = posIdxs[rIdx]

        lIdx = negFeas[:, bestIdx]<=th
        rIdx = negFeas[:, bestIdx]>th
        lNegIdxs = negIdxs[lIdx]
        rNegIdxs = negIdxs[rIdx]
        
        self.depth = self.depth - 1
        tree["feaType"] = feaTypes[bestIdx]
        tree["threshold"] = th
        tree["left"] = self.split(posSet, lPosIdxs,
                                  negSet, lNegIdxs, pntIdx,
                                  bIsCls)
        tree["right"] = self.split(posSet, rPosIdxs,
                                   negSet, rNegIdxs, pntIdx,
                                   bIsCls)
        return tree

    
    def bestSplitCls(self, posFeas, posWs, negFeas, negWs):
        ### Split based on gini criterion
        feaDim = posFeas.shape[1] 
        bestGini = 0.5
        bestFeaIdx  = -1
        bestTh   = -1
        
        for feaIdx in xrange(feaDim):
            ### AccumWeight
            Ws = NP.zeros((2, 511), dtype=NP.double)        
            Ws[0,:] = NP.bincount(posFeas[:, feaIdx],
                                  weights=posWs, 
                                  minlength=511)
            Ws[1,:] = NP.bincount(negFeas[:, feaIdx],
                                  weights=negWs, 
                                  minlength=511)
            
            WP_l = 0
            WN_l = 0
            WP_r = NP.sum(Ws[0])    
            WN_r = NP.sum(Ws[1])    
            W = WP_r + WN_r
            gini = 2*(WP_r/W)*(1-WP_r/W);
            th  = -1
                      
            for thIdx in xrange(511-1):
                WP_l += Ws[0][thIdx]
                WN_l += Ws[1][thIdx]
                WP_r -= Ws[0][thIdx]
                WN_r -= Ws[1][thIdx]

                W_l = WP_l + WN_l + sys.float_info.epsilon
                W_r = WP_r + WN_r + sys.float_info.epsilon
                g = (W_l/W)*2*(WP_l/W_l)*(1-WP_l/W_l) + \
                    (W_r/W)*2*(WP_r/W_r)*(1-WP_r/W_r)
                if g<gini:
                    gini = g
                    th = thIdx
            ###
            if gini < bestGini:
                bestGini = gini
                bestFeaIdx  = feaIdx
                bestTh   = th
        return bestFeaIdx, bestTh
    
    
    def bestSplitReg(self, feas, errs):
        sampNum, feaNum = feas.shape
        sortedFeas = NP.sort(feas, axis=0)        
        lossAndTh = NP.zeros((feaNum, 2))
        
        for idxFea in xrange(feaNum):      
            ### Randomly split on each feature              
            ### TODO choose the best split
            ind = int(sampNum*(0.5 + 0.9*(random.random()-0.5)));
            th = sortedFeas[ind, idxFea]
            lIdx = feas[:, idxFea]<=th
            rIdx = feas[:, idxFea]>th
                                
            lErrs = errs[lIdx]
            rErrs = errs[rIdx]
            lNum  = lErrs.shape[0]
            rNum  = rErrs.shape[0]   
            if lNum<2:
                lVar = 0;
            else:
                lVar = NP.sum(NP.mean(NP.power(lErrs, 2), 
                                      axis=0) - 
                              NP.power(NP.mean(lErrs, 
                                               axis=0),2))
            if rNum < 2:
                rVar = 0
            else:
                rVar = NP.sum(NP.mean(NP.power(rErrs, 2), 
                                      axis=0) - 
                              NP.power(NP.mean(rErrs, 
                                               axis=0),2))
            lossAndTh[idxFea] = (lNum*lVar + rNum*rVar, th) 
        bestFeaIdx = lossAndTh[:,0].argmin()
        return bestFeaIdx, lossAndTh[bestFeaIdx, 1]

    def genFeaType(self, feaNum, pntNum):
        feaType = NP.zeros((feaNum, 7), dtype=NP.float32)
        for i in xrange(feaNum):
            scale  = random.randint(0,2)
            pntIdx0 = random.randint(0,pntNum-1)
            pntIdx1 = random.randint(0,pntNum-1)
            offsetX0 = (random.random()-0.5)*self.radius
            offsetY0 = (random.random()-0.5)*self.radius
            offsetX1 = (random.random()-0.5)*self.radius
            offsetY1 = (random.random()-0.5)*self.radius
            feaType[i] = (scale, pntIdx0, offsetX0, offsetY0,
                          pntIdx1, offsetX1, offsetY1)
        return feaType

    def genFea(self, dataset, sampleIdxs, feaTypes):
        dataNum = len(sampleIdxs)
        feaNum  = feaTypes.shape[0]
        
        pdFea = NP.zeros((dataNum, feaNum), 
                         dtype=NP.int32)
        imgW, imgH = dataset.winSize
        coord_a = NP.zeros((feaNum, 2))
        coord_b = NP.zeros((feaNum, 2))
        
        for i in xrange(dataNum):   
            scale = NP.power(2, feaTypes[:, 0])
            initShape = dataset.initShapes[sampleIdxs[i]]
            imgData   = dataset.imgDatas[sampleIdxs[i]]
            coord_a[:,:] = initShape[feaTypes[:,1].astype(NP.int32)]
            coord_b[:,:] = initShape[feaTypes[:,4].astype(NP.int32)]

            ### Add the offset
            coord_a[:,0] = coord_a[:,0] + feaTypes[:,2]*imgW
            coord_a[:,1] = coord_a[:,1] + feaTypes[:,3]*imgH
            coord_b[:,0] = coord_b[:,0] + feaTypes[:,5]*imgW
            coord_b[:,1] = coord_b[:,1] + feaTypes[:,6]*imgH
            
            ### Divide the scale
            coord_a = NP.divide(coord_a, 
                                scale.reshape(feaNum, 1))
            coord_b = NP.divide(coord_b, 
                                scale.reshape(feaNum, 1))

            coord_a = NP.around(coord_a)
            coord_b = NP.around(coord_b)
            
            ### Check with the image size
            coord_a[coord_a<0]=0 
            coord_a[coord_a[:,0]>imgW-1, 0]=imgW-1 
            coord_a[coord_a[:,1]>imgH-1, 1]=imgH-1 
            coord_b[coord_b<0]=0 
            coord_b[coord_b[:,0]>imgW-1, 0]=imgW-1 
            coord_b[coord_b[:,1]>imgH-1, 1]=imgH-1 

            ### Construct the idx list for get the elements
            idx_a = NP.transpose(coord_a).tolist()
            idx_a[0], idx_a[1] = idx_a[1], idx_a[0]
            idx_b = NP.transpose(coord_b).tolist()
            idx_b[0], idx_b[1] = idx_b[1], idx_b[0]

            ### get the diff          
            pdFea[i,:] = NP.subtract(imgData[idx_a],
                                     imgData[idx_b],
                                     dtype = NP.int16)+255
        return pdFea

    def genBinaryFea(self, imgData, shape):
        tree = self.tree
        imgH, imgW  = imgData.shape
        point_a = NP.zeros(2, dtype=shape.dtype)
        point_b = NP.zeros(2, dtype=shape.dtype)
        while 'leafIdx' not in tree:
            feaType  = tree["feaType"] 
            th = tree["threshold"]
            point_a = shape[feaType[1]]
            point_b = shape[feaType[4]]            
            point_a[0] = feaType[2]*imgW
            point_a[1] = feaType[3]*imgH
            point_b[0] = feaType[5]*imgW
            point_b[1] = feaType[6]*imgH
            point_a = NP.around(point_a/pow(2,feaType[0]))
            point_b = NP.around(point_b/pow(2,feaType[0]))
            
            ### Check with the image size
            point_a[point_a<0]=0             
            point_b[point_b<0]=0 
            if point_a[0]>imgW-1:
                point_a[0]=imgW-1
            if point_a[1]>imgH-1:
                point_a[1]=imgH-1
            if point_b[0]>imgW-1:
                point_b[0]=imgW-1
            if point_b[1]>imgH-1:
                point_b[1]=imgH-1

            ### Construct the idx list for get the elements
            fea = NP.subtract(imgData[point_a[1], 
                                      point_a[0]] ,
                              imgData[point_b[1], 
                                      point_b[0]],
                              dtype=NP.int16)+255

            if fea <= th:
                tree = tree["left"]
            else:
                tree = tree["right"]
        
        leafIdx = tree["leafIdx"]
        return leafIdx, self.leafNum      

    def validate(self, imgData, rect, shape):
        tree = self.tree
        imgH, imgW  = rect[2], rect[3]
        point_a = NP.zeros(2, dtype=shape.dtype)
        point_b = NP.zeros(2, dtype=shape.dtype)
        while 'leafIdx' not in tree:
            feaType  = tree["feaType"] 
            th = tree["threshold"]
            point_a = shape[feaType[1]]
            point_b = shape[feaType[4]]            
            point_a[0] = feaType[2]*imgW
            point_a[1] = feaType[3]*imgH
            point_b[0] = feaType[5]*imgW
            point_b[1] = feaType[6]*imgH
            point_a = NP.around(point_a/pow(2,feaType[0]))
            point_b = NP.around(point_b/pow(2,feaType[0]))
            
            ### Check with the image size
            point_a[point_a<0]=0             
            point_b[point_b<0]=0 
            if point_a[0]>imgW-1:
                point_a[0]=imgW-1
            if point_a[1]>imgH-1:
                point_a[1]=imgH-1
            if point_b[0]>imgW-1:
                point_b[0]=imgW-1
            if point_b[1]>imgH-1:
                point_b[1]=imgH-1
            
            point_a = point_a + (rect[0], rect[1])
            point_b = point_b + (rect[0], rect[1])    

            ### Construct the idx list for get the elements
            fea = NP.subtract(imgData[point_a[1], 
                                      point_a[0]] ,
                              imgData[point_b[1], 
                                      point_b[0]],
                              dtype=NP.int16)+255

            if fea <= th:
                tree = tree["left"]
            else:
                tree = tree["right"]
        
        leafIdx = tree["leafIdx"]
        return tree["prob"], leafIdx, self.leafNum    

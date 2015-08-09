import os
import sys
import numpy as np
from   PIL  import Image
from scipy.ndimage.interpolation import zoom 

class Bootstrap(object):
    def __init__(self, paras):
        self.ch = 1
        self.winSize     = paras['winSize']
        self.stepFactor  = paras['stepFactor']
        self.scaleFactor = paras['scaleFactor']
        self.offsetStep  = paras['offsetStep']   
        self.negImgList  = paras['negImgList']

        ###  Bootstrape status paras
        self.curImgIdx = -1        
        self.offset = [0,0]
        self.point  = [0,0]
        self.curScale = 1

        ###  Bootstrape TEMP variables
        self.Image = None     
        self.scaledArr = np.empty((0,0,0))
        self.scanStep=(int(self.stepFactor*self.winSize[0]),
                       int(self.stepFactor*self.winSize[1]))

    def printParas(self):       
        print('\t\tStep  Factor = %f'%(self.stepFactor))
        print('\t\tScale Factor = %f'%(self.scaleFactor))
        print('\t\toffset Step  = %s'%(str(self.offsetStep)))

    def updateBootstrape(self):        
        imgNum = len(self.negImgList)
        self.curImgIdx += 1   
        
        ### Update the offset in the next round
        if 0 == self.curImgIdx%imgNum and self.curImgIdx>0:
            self.offset[0] += self.offsetStep[0]
            if self.offset[0]>=self.scanStep[0]:
               self.offset[0] = 0
               self.offset[1] += self.offsetStep[1]

    def nextImage(self):
        self.updateBootstrape()
        if self.offset[1] >=self.scanStep[1] or \
           None==self.negImgList:
            return False

        ### Print the bootstrape progress
        imgNum = len(self.negImgList)
        roundN = self.curImgIdx / imgNum        
        imgIdx = self.curImgIdx % imgNum
        p = (imgIdx+1)*100.0/imgNum
        if sys.stdout.isatty():   
            log = "\r\t\t-- %%%07.3f in round %d"%(p, roundN)
            sys.stdout.write(log)
        else:
            sys.stdout.seek(self.logOffset)     
            log = "\t\t-- %%%07.3f in round %d"%(p, roundN)
            sys.stdout.write(log)
        sys.stdout.flush()  

        ### Read the image
        self.scaledArr = np.empty((0,0,0))            
        path = self.negImgList[imgIdx].strip()       
        try:
            img = Image.open(path)
        except KeyboardInterrupt:
            raise KeyboardInterrupt
        except :
            print("\tIt's not an image: %s"%(path))
            self.updateBootstrape()
            return self.nextImage()
        
        ### Get smallest scale
        w, h= img.size
        scale = max(self.winSize[0]*1.0/w, 
                    self.winSize[1]*1.0/h)           
        ws = (int)(w*scale + 0.5)
        hs = (int)(h*scale + 0.5)
        self.point[:] = self.offset[:]

        while True:
            if self.point[0] <= ws-self.winSize[0] and \
               self.point[1] <= hs-self.winSize[1]:
                break
            scale *= self.scaleFactor
            ws = (int)(w*scale + 0.5)
            hs = (int)(h*scale + 0.5)  
        ### TODO:: Here scale maybe > 1.
        
        if 'L' != img.mode:
            img = img.convert("L")
        
        self.image = np.empty((self.ch, h, w),
                              dtype=np.uint8)
        ### Copy as [ch, h, w] array
        img = np.asarray(img)
        if self.ch > 1:
            for c in xrange(self.ch):
                self.image[c,:,:] = img[:,:,c]
        else:
            self.image[:,:,:] = img[:,:]

        s = (1, scale, scale)
        self.scaledArr = zoom(self.image, s)     
        self.curScale = scale*self.scaleFactor
        return True

    def updateRect(self):
        ## calculate the offset
        c, h, w = self.scaledArr.shape

        self.point[0] += self.scanStep[0]
        if self.point[0] <= w-self.winSize[0]:
            return 2
        else:
            self.point[0] = self.offset[0]
            self.point[1] += self.scanStep[1]
            if self.point[1] <= h-self.winSize[1]:
                return 2
            else:
                if self.curScale >= 1:
                    if True == self.nextImage():
                        return 1
                else:
                    s = (1, self.curScale, self.curScale)
                    self.scaledArr = zoom(self.image, s)
                    
                    self.curScale *= self.scaleFactor
                    self.point[:] = self.offset[:]
                    return 1
        return 0              

    def nextDataFromJPG(self):
        '''
        return flag, img, rect
        flag : 0 bootstrape end
               1 Image is changed(scale or another image)
               2 Image is the same with pre-frame 
        '''        
        flag = self.updateRect()
        rect = (self.point[0], self.point[1], 
                self.winSize[0], self.winSize[1])

        if  0 == flag:
            return 0, 0, 0
        else:
            return flag, self.scaledArr, rect

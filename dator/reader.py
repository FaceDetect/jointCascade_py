import os
import numpy as np
from   numpy import loadtxt
from PIL import Image
from scipy.ndimage.interpolation import zoom 
import copy
import re
import cv2

class Reader(object):    
    @classmethod
    def getAffineMatrix3P(cls, srcP1, srcP2, srcP3, 
                          dstP1, dstP2, dstP3):
        """
               p1 ---- p2
                 \    /
                   p3
        """
        src_p = np.float32([[srcP1[0], srcP1[1]], 
                            [srcP2[0], srcP2[1]], 
                            [srcP3[0], srcP3[1]]]) 
        
        dst_p = np.float32([[dstP1[0], dstP1[1]], 
                            [dstP2[0], dstP2[1]], 
                            [dstP3[0], dstP3[1]]]) 
        
        return cv2.getAffineTransform(src_p, dst_p)

    @classmethod
    def getBndBoxAndAffineT(cls, gtShape, winSize):
        ### Compute the face centre
        eyeL = (gtShape[39] + gtShape[36])/2
        eyeR = (gtShape[45] + gtShape[42])/2
        eyeC = (eyeR + eyeL)/2
        mouth= (gtShape[54] + gtShape[48])/2
        cent = (eyeC + mouth)/2

        ### Compute the face size
        pupDist = np.sqrt(np.sum(np.power(eyeR-eyeL, 2)))
        m2eDist   = np.sqrt(np.sum(np.power(eyeC-mouth, 2)))
        size = round(max(pupDist, m2eDist)*2.5)
        ori  = np.round(cent - size/2)

        ### Compute the affine matrix
        cent = ori + (size-1)/2
        
        dstP1 = (eyeL-ori)/size*winSize
        dstP2 = (eyeR-ori)/size*winSize
        dstP3 = (cent-ori)/size*winSize
        affineT = cls.getAffineMatrix3P(eyeL, eyeR, cent,
                                        dstP1, dstP2, dstP3)
        
        return (ori[0], ori[1], size, size), affineT
    
    @classmethod
    def read(cls, imgPath, winSize):
        imgP = imgPath.strip()
        folder, name = os.path.split(imgP)
        file_name,_ = os.path.splitext(name)
        annP = "%s/%s.pts"%(folder, file_name)
        
        ### Load the ground truth of shape
        lines = open(annP, 'r').readlines()
        gtShape = []
        for line in lines:
            line = line.strip()
            if not str.isdigit(line[0]):
                continue
            x, y = re.split(',| ', line)
            gtShape.append((x,y))
            
        gtShape = np.asarray(gtShape, dtype=np.float32)
        faceR, affineT = cls.getBndBoxAndAffineT(gtShape,
                                                 winSize)
        gtShape = np.subtract(gtShape, (faceR[0], faceR[1]))
        scale   = (faceR[2]*1.0/winSize[0], 
                   faceR[3]*1.0/winSize[1])
        gtShape = np.divide(gtShape, scale, dtype=np.float32)
        
        ### Load the image data
        img = Image.open(imgP)
        if 'L' != img.mode.upper():
            img = img.convert("L")

        ### Get the subImage and resize to winSize
        img = np.asarray(img, dtype=np.uint8)
        subImg = np.empty((winSize[0], winSize[1]),
                          dtype=np.uint8)
        cv2.warpAffine(img, affineT, 
                       (winSize[0], winSize[1]), subImg,
                       cv2.INTER_NEAREST)
        gtShape = copy.deepcopy(gtShape[17:])
        
        # ### Debug Show the Image
        # cv2.namedWindow("Landmark")
        # showImg = cv2.cvtColor(subImg, cv2.COLOR_GRAY2BGR)
        # for i in xrange(gtShape.shape[0]):
        #     cv2.circle(showImg, (gtShape[i, 0], 
        #                          gtShape[i, 1]),
        #                3, (0,0,255), -1)

        # cv2.imshow("Landmark", showImg)
        # key = cv2.waitKey()
        # if key in [ord("q"), 27]:
        #     raise Exception("Quit")
        
        return subImg, gtShape
        
        
        
   
     
    

    

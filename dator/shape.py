import numpy as np
import math
import random as RD

### API for shape operation
class Shape(object):    
    @classmethod
    def augment(cls, shape):
        shape = cls.scale(shape)
        shape = cls.rotate(shape)
        shape = cls.shift(shape)
        return shape

    @classmethod
    def scale(cls, shape):
        ### scale in [0.9, 1.1]
        scale = 1 + 0.2*(RD.random()-0.5)
        cent = np.mean(shape, axis=0)
        newShape = scale*(shape-cent) + cent
        return newShape

    @classmethod
    def rotate(cls, shape):
        ### Rotate in [-30,30]
        angle = (RD.random()-0.5)*math.pi/3
        
        ### TODO try the middle point instead of mean point
        # cent  = (np.max(shape, axis=0)-
        #          np.min(shape, axis=0))/2.0
        cent  = np.mean(shape, axis=0)
        rotateT = np.array(((math.cos(angle),
                             -math.sin(angle)),
                           (math.sin(angle), 
                            math.cos(angle))), 
                           dtype=np.float32)
        newShape = np.dot(shape-cent, rotateT) + cent
        return newShape

    @classmethod
    def shift(cls, shape):
        ### shift in [-0.1, 0.1] 
        shiftX = 0.2*(RD.random()-0.5)
        shiftY = 0.2*(RD.random()-0.5)
        shift = np.max(shape,axis=0) - np.min(shape,axis=0)
        shift = shift * (shiftX, shiftY)
        newShape = shape + shift        
        return newShape

        
        
        
   
     
    

    

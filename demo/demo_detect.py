#!/usr/bin/env python
import os
import sys
import getopt
import time 
import cv2
from PIL import Image, ImageDraw
### Add load path
base = os.path.dirname(__file__)
if '' == base:
    base = '.'
sys.path.append('%s/../'%base)

from cascade import *
from utils   import *

def usage():
    print("-----------------------------------------------")
    print('[[Usage]]::')
    print('\t%s [Paras] train.model test.jpg'%(sys.argv[0]))
    print("[[Paras]]::")
    print("\thelp|h : Print the help information ")
    print("-----------------------------------------------")
    return 

def detect_jpg(detector, jpg_path):
    if not os.path.exists(jpg_path):
        raise Exception("Image not exist:"%(jpg_path))
    
    src_img = Image.open(jpg_path)
    if 'L' != src_img.mode:
        img = src_img.convert("L")
    else:
        img = src_img
        
    maxSide = 200.0
    w, h = img.size
    scale = max(1, max(w/maxSide, h/maxSide))
    ws = int(w/scale + 0.5)
    hs = int(h/scale + 0.5)
    img = img.resize((ws,hs), Image.NEAREST)
    imgArr = np.asarray(img).astype(np.uint8)
    
    time_b = time.time()
    rects, confs = detector.detect(imgArr, 1.1, 2)
    t = getTimeByStamp(time_b, time.time(), 'sec')    

    ### TODO: Use NMS to merge the candidate rects and show the landmark, Now merge the rects with opencv,   
    res = cv2.groupRectangles(rects, 3, 0.2)    
    rects = res[0]

    ### Show Result
    draw = ImageDraw.Draw(img)
    for i in xrange(len(rects)):
        draw.rectangle((rects[i][0], 
                        rects[i][1],
                        rects[i][0]+rects[i][2], 
                        rects[i][1]+rects[i][3]), 
                       outline = "red")

    print("Detect time : %f s"%t)        
    img.show()
    #########################################
    
    if len(rects) > 0:
        num, _ = rects.shape
        for i in xrange(num):
            rects[i][0] = int(rects[i][0]*scale +0.5)
            rects[i][1] = int(rects[i][1]*scale +0.5)
            rects[i][2] = int(rects[i][2]*scale +0.5)
            rects[i][3] = int(rects[i][3]*scale +0.5)     
    return

    
def main(argv):
    try:
        options, args = getopt.getopt(argv, 
                                      "h", 
                                      ["help"])
    except getopt.GetoptError:  
        usage()
        return
    
    if len(argv) < 2:
        usage()
        return

    for opt , arg in options:
        if opt in ('-h', '--help'):
            usage()
            return
    
    detector = JointCascador()  
    detector = detector.loadModel(args[0])    
    detect_jpg(detector, args[1])    
            
if __name__ == '__main__' :
    main(sys.argv[1:])

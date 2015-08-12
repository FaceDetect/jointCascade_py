import numpy
config = {
    'name'       : "face" ,      
    'version'    : "1.0"  ,
    'stageNum'   :  2 ,
    
    ### Classifier and Regressor Machine Paras
    'carmPara'    :
        {
        'name'       :  'boostCart',
        'para'       :
            {                
            'tpRates'    : [[0.99]],
            'fpRates'    : [[0.30]],            
            'npRatio'    : 1.0 ,
            'cProbs'     : [0.9, 0.8, 0.7, 0.6, 0.5, 0.4],
            'radiuses'   : [0.4, 0.3, 0.2, 0.15, 0.12],
            'weakCRNums' : [20], 
            'treeDepths' : [4],
            'feaNums'    : [2000]
            }
        },
    
    'dataPara'   :
        {
	### The list file include image path list, .e.g.
	###    ./fonder/image1.jpg      
	###    ./fonder/image2.jpg
        ###    ...
        'winSize'     : (80, 80),
        'posList'     : "/home/samuel/project/sandbox/jointCascade_py/config/afw.txt",
        'negList'     : "/home/samuel/project/sandbox/jointCascade_py/config/neg.txt",
        'bootstrapPara':
            {
            'stepFactor'  : 0.5 ,
            'scaleFactor' : 1.414,
            'offsetStep'  : (8, 8)
            } 
        }
    }
 

    

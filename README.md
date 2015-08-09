Joint Cascade Face Detection and Alignment in Python
====
Implementing the joint cascade face detector on [AFW](http://ibug.doc.ic.ac.uk/resources/facial-point-annotations/) dataset. And this implementation is based on [landmark_py](https://github.com/FacialLandmark/landmark_py). All the things have benn tested on Ubuntu 14.04.


#### __Dependencies__    
---    
       
All of the following modules can be easily installed by `pip`    
> [PIL](http://www.pythonware.com/products/pil/)    
> [numpy](http://www.numpy.org/)    
> [scipy](http://www.scipy.org/)    
> [scikit-learn](http://scikit-learn.org/stable/)    
> [OpenCV](http://opencv.org/) 

Install script on Ubuntu 14.04   
>sudo aptitude install python-pip gfortran     
>sudo pip install pillow numpy scipy sklearn    
>sudo aptitude install python-opencv


#### __Demo on AFW__    
---    

Because AFW only contain 337 faces. If you want get a good result, please train on the big trainSet

1. Download the AFW dataset [here](http://ibug.doc.ic.ac.uk/resources/facial-point-annotations/)
2. Replace the location of afw by yourself in `afw_test.lst`, `afw_train.lst` and `neg.txt` in config folder(Mine is `/home/samuel/data`)
3. Change `afw_config.py:dataPara:posList, negList` by yourself       

* __Train on AFW__     
>python -W ignore ./demo_train.py ../config/afw_config.py    

* __Evaluate on AFW(Coming Soon...)__       
>./demo_evaluate.py  ../config/afw_model/train.model  ../config/afw_test.lst       


#### __References__    
---    
1. Face Alignment at 3000 FPS via Regressing Local Binary Features    
2. Joint Cascade Face Detection and Alignment    


#### __Contact__    
---    
If you have any questions, please email `shenfei1208@gmail.com` or creating an issue on GitHub.

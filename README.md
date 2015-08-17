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
>sudo aptitude install python-pip gfortran imagemagick     
>sudo pip install pillow numpy scipy sklearn    
>sudo aptitude install python-opencv


#### __Demo on AFW__    
---    

`I have trained a model based on AFW(only contain 337 faces) with 5 non-face images for demo`. You should train the model on a big dataset

1. Download the AFW dataset [here](http://ibug.doc.ic.ac.uk/resources/facial-point-annotations/)
2. Replace the location of afw by yourself in `afw.txt` and `neg.txt` in config folder(Mine is `/home/samuel/data`)
3. Change `afw_config.py:dataPara:posList, negList` by yourself       

* __Train on AFW__     
>python -W ignore ./demo_train.py ../config/afw_config.py    

* __Detection with the Pretrained AFW Model__   
>python -W ignore ./demo_detect.py  ../config/afw_model/train.model  ../config/pos.jpg       


####  __TODO__    
---    
1. Optimize the global regression into tree traversing     
2. Use the __non-maximum suppression__  to find the best face rects instead to merge the rects via cv2.grouprects

#### __References__    
---    
1. Face Alignment at 3000 FPS via Regressing Local Binary Features    
2. Joint Cascade Face Detection and Alignment    


#### __Contact__    
---    
If you have any questions, please email `shenfei1208@gmail.com` or creating an issue on GitHub.

#!/usr/bin/env python
import os
import sys
import getopt
import time 
import imp
import random

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
    print('\t{0}  [Paras]  config.py'.format(sys.argv[0]))
    print("[[Paras]]::")
    print("\thelp|h     : Print the help information ")
    print("-----------------------------------------------")
    return 

def main(argv):
    try:
        options, args = getopt.getopt(argv, 
                                      "h", 
                                      ["help"])
    except getopt.GetoptError:  
        usage()
        return
    
    if len(sys.argv) < 2:
        usage()
        return

    for opt , arg in options:
        if opt in ('-h', '--help'):
            usage()
            return

    #Get the paras for training
    try:
        config_str = open(args[0],"r").read()
        config = imp.new_module('config')
        exec(config_str, config.__dict__)
    except IndexError:
        print("ERROR:: Please input the config file")
        return
    except :
        raise
        
    save_path = os.path.split(args[0])[0]
    if '' == save_path:
        save_path = '.'

    cascade = JointCascador()  
    cascade.config(config.config)

    ### Strat to train    
    cascade.printParas()

    print("\n------------------------------------------")
    print("-----------  Training Progress  ----------")
    print("Start training ... ")
    begTime = time.time()
    cascade.train(save_path)    
    trainTime = getTimeByStamp(begTime, time.time(), 'hour')
    print("Finish Training : %f hours"%(trainTime))
    print("------------------------------------------")
       
if __name__ == '__main__' :
    main(sys.argv[1:])

# -*- coding: utf-8 -*-
"""
Created on Thu Aug  6 13:22:45 2015

@author: annedwyer
"""

############################################################################
#
# This code creates training and test sets for learning unbalanced 
# two class problems.
#
##########################################################################
import numpy as np
import pandas as pd
from pandas import read_table, DataFrame
import random
import sys

file = sys.argv[0] #csv file name
type1 = sys.argv[1] #name of minority class
type2 = sys.argv[2] #name of majority class
classLabel = sys.argv[3] # name of column with class values

# This function gets a random index for any size array.
def getIndex(arrayLen, size):
    index = np.random.choice(arrayLen, size=size, replace = False)
    return index

#This function concatenates minority and majority array and shuffles the dataset.   
def conCatShuf(array1, array2):
    array = np.concatenate((array1,array2), axis = 0)
    index = getIndex(array.shape[0], array.shape[0])
    new = array[index,:]
    return new
 
#This function splits the dataset according to class and changes the text label
# to numeric    
def splitClass(df, type,type1,type2, label):
    df = df[df[label]==type]
    t = df.replace({label:{type1:0,type2:1}})
    return t

#This function splits the minority class according to chosen split percentage.    
def createMinorityClass(df,split):
    random.seed(10)
    new = np.asarray(df)
    num = int(split*new.shape[0])# num of examples in training set
    index = getIndex(new.shape[0], new.shape[0])
    trmin = new[index[:num],:]
    unmin = new[index[num:],:]
    return unmin, trmin
  
#This function creates the train and test datasets.
#The train dataset has equal numbers of minority and majority classes.
# The test dataset mimics the minority/majority ratio of the original
# dataset.  
def createDatasets(unmin,trmin,mClass):
    random.seed(20)
    maj = np.asarray(mClass)
    perMin = float(unmin.shape[0] + trmin.shape[0])/float(unmin.shape[0] + trmin.shape[0] + maj.shape[0])
    size = int(unmin.shape[0]/perMin) - unmin.shape[0] + trmin.shape[0]
    index = getIndex(maj.shape[0], size)
    trmaj = maj[index[:trmin.shape[0]],:]
    testmaj = maj[index[trmin.shape[0]:],:]
    train = conCatShuf(trmin,trmaj)
    test = conCatShuf(unmin,testmaj)
    return train, test
    


df = pd.read_csv(file)
#type1 is the name of the minority class if text, type2 is the name of the majority class.
#split is the training set split expressed as a decimal
unmin, trmin = createMinorityClass(splitClass(df, type1, type1, type2,classLabel),.80) 

train, test = createDatasets(unmin,trmin,splitClass(df,type2, type1, type2,classLabel))

#This part of the code splits the training and test sets into x and y arrays.
# This is how datasets are input into the machine learning algorithms.
train_x = train[:,:train.shape[1]-1]
train_y = train[:,train.shape[1]-1]
test_x = test[:,:test.shape[1]-1]
test_y = test[:,test.shape[1]-1]


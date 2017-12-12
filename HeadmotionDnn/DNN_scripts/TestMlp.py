from keras.models import Sequential
import scipy.io
import numpy
import sys
from keras.layers.core import Dense, Activation
from keras.optimizers import SGD, RMSprop, Adagrad, Adadelta, Adam, Nadam
import h5py
#from  normalize_data import normalize,denormalize
from sklearn.cross_validation import train_test_split
from keras.utils.generic_utils import Progbar
from keras.callbacks import EarlyStopping
import os
from keras.callbacks import TensorBoard
import ntpath
import logging
import time

print(len(sys.argv))
print(sys.argv[1])
print(sys.argv[2])
mdl_fldr = sys.argv[1];
cnfg = sys.argv[2];
datafile = sys.argv[3];
xname = sys.argv[4];
yname = sys.argv[5];
pred_name = sys.argv[6];

#train_flag = int(sys.argv[4]);
newDict = {}
#cnfg='nnet_confg.txt';
with open(cnfg) as f:
  for line in f:
    line=line.rstrip('\n')
    #print line
    splitLine = line.split(':')
    if((splitLine[0]=='indim') | (splitLine[0]=='outdim')|(splitLine[0]=='nbepochs')|(splitLine[0]=='batchsize')| (splitLine[0]=='normalizex')|(splitLine[0]=='normalizey')):
      newDict[(splitLine[0])] = int(splitLine[1].rstrip()) 
    elif(splitLine[0]=='loss'):
      newDict[(splitLine[0])] = (splitLine[1].strip()) 	
    else:
      line = splitLine[1];
      line = line.split(' ')
      newDict[(splitLine[0])] = line
    
#    if(splitLine[0]!='nbepochs'):
	#mdlfile = mdlfile+splitLine[0]+'_'+ str(newDict[(splitLine[0])])


#print newDict
#nins = newDict['indim']
#nouts = newDict['outdim']
activations = newDict['activation']

hiddenlayers = newDict['hiddenlayers']
hiddenlayers=[x for x in hiddenlayers if(x!='')]


data = scipy.io.loadmat(datafile);
Xtrain = data[xname]
Ytrain = data[yname]
#Ytrain = numpy.log(Ytrain)
print Xtrain.shape
if(newDict['normalizex']==1):
 mux=numpy.loadtxt(mdl_fldr+'mux.txt');
 sigmax=numpy.loadtxt(mdl_fldr+'sigmax.txt');
 
#Xtrain = (Xtrain - mux) / sigmax;

	
print 'number of infinite values:' + str(sum(sum(~numpy.isfinite(Xtrain))))
xtrain_size=Xtrain.shape
print 'input number of greater than +-1:' + str(sum(sum(abs(Xtrain)>1))*100.00/(xtrain_size[0]*xtrain_size[1]))
print 'input data dimention is '+str(Xtrain.shape)        
model = Sequential()
#print '..building model'
#print('creating the layer: Input {} -> Output {} with activation {}'.format(Xtrain.shape[1], int(hiddenlayers[1]), activations[0]))
model.add(Dense(output_dim=int(hiddenlayers[1]), input_dim=Xtrain.shape[1],activation=activations[0]))    
for k in xrange(2,len(hiddenlayers)-1):
  #print('creating the layer: Input {} -> Output {} with activation {}'.format(int(hiddenlayers[k-1]), int(hiddenlayers[k]), activations[k]))
  model.add(Dense(output_dim=int(hiddenlayers[k]),activation=activations[k]))    

#print('creating the layer: Input {} -> Output {} with activation {}'.format(int(hiddenlayers[len(hiddenlayers)-2]),Ytrain.shape[1], activations[-1]))  
model.add(Dense(output_dim=int(Ytrain.shape[1]),activation=activations[-1]))   

  
#print '..compiling model'  
model.compile(loss=newDict['loss'], optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08))
model.load_weights(mdl_fldr+'/best_model');
pred = model.predict(Xtrain);
print 'model loaded from '+mdl_fldr
print 'data loaded from '+datafile
#if(pred.shape[0] != Ytrain.shape[0]):
#  pred=pred.transpose();
#print(numpy.sqrt(numpy.mean((pred-Ytrain)**2)))
scipy.io.savemat(mdl_fldr+'/'+pred_name,{'pred':pred})
time.sleep(5)


#   

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


cnfg = sys.argv[1];
datafile = sys.argv[2];
xname = sys.argv[3];
yname = sys.argv[4];
mdlfile = sys.argv[5];
#train_flag = int(sys.argv[4]);
newDict = {}

with open(cnfg) as f:
  for line in f:
    line=line.rstrip('\n')
    print line
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

#name=os.path.splitext(datafile)[0]
#C= name.split('/')
#print C[-1],C[-2]
#mdlfile = C[-1]+'_'+C[-2]+'_'+xname+'_'+yname;    
directory='../models/'+mdlfile;    
logging.basicConfig(level=logging.DEBUG)

print directory
if not os.path.exists(directory):
    print "creating directory "+ mdlfile
    os.makedirs(directory)
    train_flag=1;
    
else:
    print "model already exists"
    train_flag=1;	
    

os.system('cp '+cnfg+' '+directory+'/nnet_config.txt')

  
print newDict
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
 print "normalizing input data"
 mux = numpy.mean(Xtrain,axis=0);	      
 sigmax= numpy.std(Xtrain,axis=0);
 print mux.shape,sigmax.shape
 sigmax[sigmax<1e-5]=1;
 numpy.savetxt(directory+'/mux.txt',mux);
 numpy.savetxt(directory+'/sigmax.txt',sigmax);
 
#Xtrain = (Xtrain - mux) / sigmax;

print 'output data dimention is '+str(Ytrain.shape)  

if(train_flag==1):
    order=numpy.arange(Xtrain.shape[0]);
    numpy.random.shuffle(order)
    Xtrain=Xtrain[order,:];
    Ytrain=Ytrain[order,:];
    numpy.savetxt(directory+'/order.txt',order);
else:
    order=(numpy.loadtxt(directory+'/order.txt',dtype=numpy.int16));
    Xtrain=Xtrain[order,:];
    Ytrain=Ytrain[order,:];

	
print 'number of infinite values:' + str(sum(sum(~numpy.isfinite(Xtrain))))
xtrain_size=Xtrain.shape
ytrain_size=Ytrain.shape
print 'input number of greater than +-1:' + str(sum(sum(abs(Xtrain)>1))*100.00/(xtrain_size[0]*xtrain_size[1]))
print 'input data dimention is '+str(Xtrain.shape)        
model = Sequential()
print '..building model'
print('creating the layer: Input {} -> Output {} with activation {}'.format(Xtrain.shape[1], int(hiddenlayers[1]), activations[0]))
model.add(Dense(output_dim=int(hiddenlayers[1]), input_dim=Xtrain.shape[1],activation=activations[0]))    
for k in xrange(2,len(hiddenlayers)-1):
  print('creating the layer: Input {} -> Output {} with activation {}'.format(int(hiddenlayers[k-1]), int(hiddenlayers[k]), activations[k]))
  model.add(Dense(output_dim=int(hiddenlayers[k]),activation=activations[k]))    

print('creating the layer: Input {} -> Output {} with activation {}'.format(int(hiddenlayers[len(hiddenlayers)-2]),Ytrain.shape[1], activations[-1]))  
model.add(Dense(output_dim=int(Ytrain.shape[1]),activation=activations[-1]))   

  
print '..compiling model'  
model.compile(loss=newDict['loss'], optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08))

if(train_flag==1):
  print '..fitting model'
  batch_size=int(newDict['batchsize'])
  nb_batch=Xtrain.shape[0]/batch_size  
  progbar = Progbar(nb_batch)
  history1=model.fit(Xtrain,Ytrain, nb_epoch=5, batch_size=int(newDict['batchsize']),verbose=1, validation_split=0.2)
  early_stopping = EarlyStopping(monitor='val_loss', patience=3)
  history2=model.fit(Xtrain,Ytrain, nb_epoch=int(newDict['nbepochs']), batch_size=int(newDict['batchsize']),verbose=2, validation_split=0.2, callbacks=[early_stopping])
  model.save_weights(directory+'/best_model',overwrite=True)
 
else:
  print '..continue the training'
  model.load_weights(directory+'/best_model');
  early_stopping = EarlyStopping(monitor='val_loss', patience=3)
  history2=model.fit(Xtrain,Ytrain, nb_epoch=int(newDict['nbepochs']), batch_size=int(newDict['batchsize']),verbose=1, validation_split=0.2, callbacks=[early_stopping])
  model.save_weights(directory+'/best_model',overwrite=True)

#   

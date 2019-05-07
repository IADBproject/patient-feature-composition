#################################################################################
#                                       										#
#   Towards a Green Intelligence Applied to Health Care and Well-Being  		#
###-----------------------------------------------------------------------------#
# diagnosenet_splitpoint.py														#
# Split the binary patient phenotype representation								#
# and write in data_train, data_valid, data_test								#
#################################################################################

import os, sys
import numpy as np
import time
#import memory_profiler
import pickle
import logging

import sklearn
from sklearn.cross_validation import train_test_split

## dIAgnoseNET library
from diagnosenet_vocabularycomposition import VocabularyComposition
logger = logging.getLogger('_dIAgnoseNET_UnsupervisedEmbedding')


class SplitPoint(object):
	"""
	Split the binary patient phenotype representation
	and write in data_train, data_valid, data_test
	"""

	def __init__(self, *args, **kwards):
		self.features_name = args[0]
		self.sandbox = args[1]
		self.year = args[2]
		self.sp_factor = args[3]
		self.embd_dir = args[4]
		self.medicaltarget = args[5]


		## Split point variables
		SplitPoint.local_dir = "/1_Mining-Stage/binary_representation/"

		### First stage directories to locates and load the Binary Phenotype Representation files
		SplitPoint.dirPatient = str(self.sandbox+self.local_dir+"BPPR-"+self.features_name+"-"+self.year+".txt")
		if self.medicaltarget == 'y1':
			SplitPoint.dirDisease= str(self.sandbox+self.local_dir+"labels_Y1-"+self.features_name+"-"+self.year+".txt")
		elif self.medicaltarget == 'y2':
			SplitPoint.dirDisease= str(self.sandbox+self.local_dir+"labels_Y2-"+self.features_name+"-"+self.year+".txt")
		elif self.medicaltarget == 'y3':
			SplitPoint.dirDisease= str(self.sandbox+self.local_dir+"labels_Y3-"+self.features_name+"-"+self.year+".txt")
		else:
			logger.warning('!!! Medical target dont selected !!!')

		### Second stage directories to write the data splited
		#SplitPoint.dir_X_train = str(self.embd_dir+"/data_train/X_train-")
		#SplitPoint.dir_y_train = str(self.embd_dir+"/data_train/y_train-")
		#SplitPoint.dir_X_valid = str(self.embd_dir+"/data_valid/X_valid-")
		#SplitPoint.dir_y_valid = str(self.embd_dir+"/data_valid/y_valid-")
		#SplitPoint.dir_X_test = str(self.embd_dir+"/data_test/X_test-")
		#SplitPoint.dir_y_test = str(self.embd_dir+"/data_test/y_test-")


	def _read_file(self, file_name):
		patientCorpus = []
		f = open(file_name, 'r')
		for line in f:
			patientCorpus.append(line)
		f.close()
		return patientCorpus


	def _split_Dataset(self):

		## Read the Binary Patient Representation
		_X_ = self._read_file( self.dirPatient )

		## Read the One-hot target representation
		_Y_ = self._read_file( self.dirDisease )

		X_train, X_temp, y_train, y_temp = train_test_split(_X_,_Y_, train_size=0.85)
		X_valid, X_test, y_valid, y_test = train_test_split(X_temp,y_temp, train_size=0.33)

		logger.info('---------------------------------------------------------')
		logger.info('++ Split Dataset ++')
		logger.info('-- Train shapes:      {} | {} --'.format(len(X_train), len(y_train)))
		logger.info('-- Temp shapes:       {} | {} --'.format(len(X_temp), len(y_temp)))
		logger.info('-- Validation shapes: {} | {} --'.format(len(X_valid), len(y_valid)))
		logger.info('-- Test shapes:       {} | {} --'.format(len(X_test), len(y_test)))
		return (X_train, y_train, X_valid, y_valid, X_test, y_test)

	def _write_file(self, data, dir_):
		#print("Data: %s " % data)
		#print("Len Data %s" % len(data))
		#print("Data %s" % dir_)
		filename = 1
		for i in range(len(data)):
			if i % self.sp_factor == 0:
				open(str(dir_)+str(self.features_name)+str("-")+str(self.year)+str("-")+
										str(filename)+str('.txt'), 'w+').writelines(data[i:i+self.sp_factor])
				filename += 1



### End SplitPoint()
#####################

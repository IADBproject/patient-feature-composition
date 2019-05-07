#################################################################################
#                                                                               #
#   Towards a Green Intelligence Applied to Health Care and Well-Being          #
###-----------------------------------------------------------------------------#
# diagnosenet_labelcomposition.py												#
# Label Composition to extract patient's attributes for driving					#
# their phenotype representation from Intensive Care Unit (ICU) PMSI-PACA       #
#################################################################################


from __future__ import print_function
import os, sys
import numpy as np
import pandas as pd
import time
import logging

## JSON Data Structure Format to Serialization
import json
from objectjson import ObjectJSON

### Method to remove all unicode characters
from ast import literal_eval

## Build Vocbulary
import nltk
from nltk import FreqDist
from collections import OrderedDict

## dIAgnoseNET library
logger = logging.getLogger('_dIAgnoseNET_DataMining')

class LabelComposition(object):
	"""
	This class get the medical target from the CDA schema in a JSON format
	to build a one-hot vector representation.
	"""

	def __init__(self, *args, **kwargs):
	## dIAgnoseNet path variables
		#self.dataset_dir = args[0]
		self.features_name = args[0]
		self.sandbox = args[1]
		self.year = args[2]

		## label composition
		LabelComposition.localdir = "/1_Mining-Stage/binary_representation/"
		LabelComposition.pm_Dictionary = dict()
		LabelComposition.pm_label = []
		LabelComposition.pm_lenght = []
		LabelComposition.binary_label = []
		LabelComposition.procedures_label_Voc = []
		LabelComposition.cp_binary_label = []
		LabelComposition.cp_lenght = []
		LabelComposition.destination_label_Voc = []
		LabelComposition.d_lenght = []
		LabelComposition.d_binary_label = []


	def _set_primaryMorbidityDictionary(self, cda_object, x8_name=None):
		"""
		Set primary morbidity label
		"""
		### Features by electronic health record entity
		# if not x8_name: x8_name = ['care_purpose','morbidity','etiology','major_clinical_category']
		if not x8_name: x8_name = ['major_clinical_category']

		### Buil a primary morbidity label
		pm = []
		for record_object in cda_object:
			major_category = [(getattr(record_object.x8_primary_morbidity,i)) for i in x8_name]
			# /home/jagarcia/Documents/05_dIAgnoseNET/diagnosenet/src/diagnosenet_labelcomposition.py
			##print("primary morbidity label: %s" % [(getattr(record_object.x8_primary_morbidity,i)) for i in x8_name])
			# pm.append(str(care_purpose +"-"+ morbidity +"-"+ etiology +"-"+ major_category))
			pm.append(str(major_category))

		## Extract key terms for clean the Medical Vocabulary
		lab_terms = list(dict.fromkeys(pm))
		lab_terms.sort()
		self.pm_lenght = len(dict.fromkeys(pm))

		## Build a primary label dictionary values
		value = 1
		for label in lab_terms:
			self.pm_Dictionary[label] =  value
			value +=1
		##print("pm %s" % self.pm_Dictionary)


	def _get_PrimaryMorbidityLabel(self, cda_object, x8_name=None):
		"""
		Get primary morbidity label
		"""
		### Features by electronic health record entity
		# if not x8_name: x8_name = ['care_purpose','morbidity','etiology','major_clinical_category']
		if not x8_name: x8_name = ['major_clinical_category']

		## Set vocabulary
		self._set_primaryMorbidityDictionary(cda_object)

		for record_object in cda_object:
			major_category = [(getattr(record_object.x8_primary_morbidity,i)) for i in x8_name]
			# care_purpose, morbidity, etiology, major_category = [(getattr(record_object.x8_primary_morbidity,i)) for i in x8_name]
			##print("primary morbidity label: %s" % [(getattr(record_object.x7_primary_morbidity,i)) for i in x7_name])

			## Add a Primary morbidity label by patient record
			self.pm_label.append(self.pm_Dictionary[str(major_category)])
			# self.pm_label.append(self.pm_Dictionary[str(care_purpose +"-"+ morbidity +"-"+ etiology +"-"+ major_category)])
			##print("+++ Record Patient Label: %s +++" % type(self.pm_Dictionary[
			##											str(care_purpose+"-"+morbidity+"-"+ etiology+"-"+major_category)]) )

		return self.pm_label


	def _build_BinaryPrimaryMorbidity(self):
		"""
		Build a One-Hot Vector for Primary Morbidity Multilabels
		"""

		logger.info('-- Build a One-Hot Vector for Primary Morbidity Multilabel --')

		### Get the primary morbidity representation values 'self.pm_Dictionary.values()' to build a 'one-hot vector' as target
		### len of vector is equal to number of prmary morbidty representation values
		vocabulary_label = np.asarray(self.pm_Dictionary.values(), dtype='|S5')
		vocabulary_label.sort()
		#logger.info('-- Number of primary morbidity multilabels in One-hot Vector: [{}]'.format(len(vocabulary_label)))

		for record_label in self.pm_label:

			## Get the indice term from the vocabulary representation
			one_hot_vector = [0] * len(self.pm_Dictionary.values())
			##print("+++ record label %s" % record_label)
			indice =vocabulary_label.tolist().index(str(record_label))
			##print("+++ indice %s " % indice)
			one_hot_vector.pop(indice)
			one_hot_vector.insert(indice, 1)
			##print("+ 1H-v: %s +" % one_hot_vector)
			self.binary_label.append(one_hot_vector)

			## Add a vector

	def _write_PrimaryMorbidityLabel(self):
		"""
		Write the primary morbidity values as y1 label
		"""
		file_y1_labels = str(self.sandbox+self.localdir+str("labels_Y1-"+self.features_name+"-"+self.year+".txt"))
		np.savetxt(file_y1_labels, self.binary_label, delimiter=',', fmt='%s')
		logger.info('-- Path of Y1_labels file: {} --'.format(file_y1_labels))


		file_voc_y1 = str(self.sandbox+self.localdir+str("labels_Y1_Vocabulary"+self.features_name+"-"+self.year+".txt"))
		vocabulary_label = np.asarray(self.pm_Dictionary.values(), dtype='|S5')
		vocabulary_label.sort()
		np.savetxt(file_voc_y1, vocabulary_label,delimiter=',', fmt='%s')

		logger.info('-- Path of Y1_Vocabulary file: {} --'.format(file_voc_y1))

		return self.pm_lenght


	############################################################################
	############################################################################
	def _set_clinicalProceduresVoc(self, cda_object, x9_name=None):
		"""
		Set clinical procedures to concatenate all the procedures associated to one record
		"""
		### Features by electronic health record entity
		if not x9_name: x9_name = ['x9_clinical_procedures']

		### Buil a clinical_proceduresy labels
		for record_object in cda_object:
			x9_elements = literal_eval(getattr(record_object,"x9_clinical_procedures"))

			## Row to concatenate all the clinical procedures by record
			if x9_elements[0] != '0':
				## Create a new string with clean values or elements
				chars_to_remove=['[',']','"',"'"]   ## on list
				clean_elements = str(x9_elements).translate(None, ''.join(chars_to_remove))
				##print("clean_elements: %s" % clean_elements)

				## Build label by each patient record
				procedures_label = clean_elements.replace(',','-')

				### If the label not exists this is append on the vocabulary
				try:
					self.procedures_label_Voc.index(procedures_label)
				except ValueError:
					self.procedures_label_Voc.append(procedures_label)
					##print("procedures_label: %s" % self.procedures_label)
			else:
				pass

		self.procedures_label_Voc.append('0')
		##print("procedures_label_Voc: %s" % procedures_label_Voc)


	def _build_clinicalProcedures(self, cda_object):
		"""
		Build a One-Hot Vector for Clinical Procedures Multilabels
		"""

		logger.info('-- Build a One-Hot Vector for Clinical Procedures Multilabel --')

		### Get the primary morbidity representation values 'self.pm_Dictionary.values()' to build a 'one-hot vector' as target
		### len of vector is equal to number of prmary morbidty representation values
		vocabulary_label = np.asarray(self.procedures_label_Voc, dtype='|S100')
		vocabulary_label.sort()
		#logger.info('-- Number of clinical procedures multilabels in One-hot Vector: [{}]'.format(len(vocabulary_label)))
		self.cp_lenght = len(vocabulary_label)


		for record_object in cda_object:
			x9_elements = literal_eval(getattr(record_object,"x9_clinical_procedures"))

			## Row to concatenate all the clinical procedures by record
			if x9_elements[0] != '0':
				## Create a new string with clean values or elements
				chars_to_remove=['[',']','"',"'"]   ## on list
				clean_elements = str(x9_elements).translate(None, ''.join(chars_to_remove))
				##print("clean_elements: %s" % clean_elements)

				## Build label by each patient record
				procedures_label = clean_elements.replace(',','-')
				#print("Procedures_label: %s" % procedures_label)
			else:
				pass

			## Get the indice term from the vocabulary representation
			one_hot_vector = [0] * len(vocabulary_label)

			##print("+++ record label %s" % record_label)
			indice =vocabulary_label.tolist().index(str(procedures_label))
			##print("+++ indice %s " % indice)
			one_hot_vector.pop(indice)
			one_hot_vector.insert(indice, 1)
			##print("+ 1H-v: %s" % one_hot_vector)
			self.cp_binary_label.append(one_hot_vector)


	def _write_clinicalProceduresLabel(self):
		"""
		Write the clinical procedures values as y2 label
		"""
		file_y2_labels = str(self.sandbox+self.localdir+str("labels_Y2-"+self.features_name+"-"+self.year+".txt"))
		np.savetxt(file_y2_labels, self.cp_binary_label, delimiter=',', fmt='%s')
		logger.info('-- Path of Y2_labels file: {} --'.format(file_y2_labels))


		file_voc_y2 = str(self.sandbox+self.localdir+str("labels_Y2_Vocabulary"+self.features_name+"-"+self.year+".txt"))
		vocabulary_label = np.asarray(self.procedures_label_Voc, dtype='|S100')
		vocabulary_label.sort()
		np.savetxt(file_voc_y2, vocabulary_label, delimiter=',', fmt='%s')
		logger.info('-- Path of Y2_vocabulary file: {} --'.format(file_voc_y2))

		return self.cp_lenght


	############################################################################
	############################################################################
	def _set_destinationVoc(self, cda_object, x10_name=None):
		"""
		Set clinical procedures to concatenate all the procedures associated to one record
		"""
		### Features by electronic health record entity
		if not x10_name: x10_name = ['output_mode']

		### Buil a clinical_proceduresy labels
		for record_object in cda_object:
			x10_elements = [(getattr(record_object.x10_destination,i)) for i in x10_name]

			## Row to concatenate all the clinical procedures by record
			if x10_elements[0] != '0':
				## Create a new string with clean values or elements
				chars_to_remove=['[',']','"',"'"]   ## on list
				clean_elements = str(x10_elements).translate(None, ''.join(chars_to_remove))
				##print("clean_elements: %s" % clean_elements)

				## Build label by each patient record
				destination_label = clean_elements.replace(',','-')

				### If the label not exists this is append on the vocabulary
				try:
					self.destination_label_Voc.index(destination_label)
				except ValueError:
					self.destination_label_Voc.append(destination_label)
					#print("destination_label: %s" % destination_label)
			else:
				pass

		self.destination_label_Voc.append('0')
		#print("destination_label_Voc: %s" % self.destination_label_Voc)


	def _build_Destination(self, cda_object, x10_name=None):
		"""
		Build a One-Hot Vector for Destination Multilabels
		"""

		logger.info('-- Build a One-Hot Vector for Destination Multilabel --')

		### Features by electronic health record entity
		if not x10_name: x10_name = ['output_mode']

		### Get the primary morbidity representation values 'self.pm_Dictionary.values()' to build a 'one-hot vector' as target
		### len of vector is equal to number of prmary morbidty representation values
		vocabulary_label = np.asarray(self.destination_label_Voc, dtype='|S7')
		vocabulary_label.sort()
		#logger.info('-- Number of clinical procedures multilabels in One-hot Vector: [{}]'.format(len(vocabulary_label)))
		self.d_lenght = len(vocabulary_label)


		for record_label in cda_object:
			x10_elements = [(getattr(record_label.x10_destination, i)) for i in x10_name]

			## Row to concatenate all the clinical procedures by record
			if x10_elements[0] != '0':
				## Create a new string with clean values or elements
				chars_to_remove=['[',']','"',"'"]   ## on list
				clean_elements = str(x10_elements).translate(None, ''.join(chars_to_remove))
				##print("clean_elements: %s" % clean_elements)

				## Build label by each patient record
				destination_label = clean_elements.replace(',','-')
				#print("Destination_label: %s" % destination_label)
			else:
				pass

			## Get the indice term from the vocabulary representation
			one_hot_vector = [0] * len(vocabulary_label)

			#print("+++ record label %s" % one_hot_vector)
			indice = vocabulary_label.tolist().index(str(destination_label))
			#print("+++ indice %s " % indice)
			one_hot_vector.pop(indice)
			one_hot_vector.insert(indice, 1)
			##print("+ 1H-v: %s" % one_hot_vector)
			self.d_binary_label.append(one_hot_vector)


	def _write_DestinationLabel(self):
		"""
		Write the destination values as y2 label
		"""
		file_y3_labels = str(self.sandbox+self.localdir+str("labels_Y3-"+self.features_name+"-"+self.year+".txt"))
		np.savetxt(file_y3_labels, self.d_binary_label, delimiter=',', fmt='%s')
		logger.info('-- Path of Y3_labels file: {} --'.format(file_y3_labels))


		file_voc_y3 = str(self.sandbox+self.localdir+str("labels_Y3_Vocabulary"+self.features_name+"-"+self.year+".txt"))
		vocabulary_label = np.asarray(self.destination_label_Voc, dtype='|S5')
		vocabulary_label.sort()
		np.savetxt(file_voc_y3, vocabulary_label, delimiter=',', fmt='%s')
		logger.info('-- Path of Y3_vocabulary file: {} --'.format(file_voc_y3))

		return self.d_lenght




### End LabelComposition()
##########################

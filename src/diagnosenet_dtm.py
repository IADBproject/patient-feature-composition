#################################################################################
#   Towards a Green Intelligence Applied to Health Care and Well-Being          #
###-----------------------------------------------------------------------------#
# diagnosenet_dtm.py															#
# Used this class for building a Medical Vocabulary                             #
# from feature composition groups PMSI-PACA (features_serializations)           #
# and get the binary patient phenotype representation                           #
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
import random

## Tokenize terms to create the vocbulary
## and a dense Document-term matrix
import nltk
from nltk import FreqDist
#from collections import OrderedDict
#nltk.download()

## dIAgnoseNET library
from diagnosenet_vocabularycomposition import VocabularyComposition
logger = logging.getLogger('_dIAgnoseNET_DataMining')


class DocumentTermMatrix(object):
	"""
	This class build a binary petient phenotype representation from their features selected.
	The rows correspond to patient phenotype (or profile)
	and the columns correspond to terms (or features).
	"""

	def __init__(self, *args, **kwargs):
	## dIAgnoseNet path variables
		self.dataset_dir = args[0]
		self.features_name = args[1]
		self.sandbox = args[2]
		self.year = args[3]
		#self.vocabulary_type = args[4]

		## DTM variables
		DocumentTermMatrix.localdir = "/1_Mining-Stage/binary_representation/"
		DocumentTermMatrix.binary_PPR = []
		DocumentTermMatrix.len_BPPR = 0

	def _get_cleanValues(self, record_object, entity_name):
		"""
		Clean values and remove zeros values
		"""
		if entity_name[0] == 'x7_associated_diagnosis':
			## convert the patient's features list in string values
			row_values = ','.join( str(value) for value in record_object )
			##print("+++ raw_terms: %s " % raw_values)

			## Create a new string with clean values or elements
			chars_to_remove=["'',", "[", "]"]   ## on list
			clean_values = row_values.translate(None, ''.join(chars_to_remove))
			## clean_values = row_values.translate(None, ''.join(chars_to_remove))
			##print("clean_values %s " % type(clean_values.split()) )

				## Store non-zero elements in a list
			values_non_zeros = filter(lambda a: a != '0', clean_values.split() )
			return values_non_zeros

		elif entity_name[0] == 'x9_clinical_procedures':
			## convert the patient's features list in string values
			row_values = ','.join( str(value) for value in record_object )
			##print("+++ raw_terms: %s " % raw_values)

			## Create a new string with clean values or elements
			chars_to_remove=['[',']','"']   ## on list
			clean_values = row_values.translate(None, ''.join(chars_to_remove))
			##print("clean_values %s " % type(clean_values.split()) )

			## Store non-zero elements in a list
			values_non_zeros = filter(lambda a: a != '0', clean_values.split(",") )

			return values_non_zeros


	def _get_binaryRecord(self, record_object, vocabulary, entity_name):
		"""
		This function creates a document-term matrix for each patient's feature.
		There are two kinf the features; the first one is when the feature is represent by one value
		such as {age: 56 || etiology: E10} and the second is whe the patient feature is represent
		by a list of features the same language such {related_diagnoses: {DAS1: E212, DAS2: E780, ...}}.
		"""

		if 'list' in str(type(record_object)):

			## Strip returns the sting after stripping whitespace to 'empty' objects --- empty list
			## the numpy object is transformed to list 'record_object.ravel().tolist()' as example:
			## x3_related_diagnoses: [u'[["F331", "F172","0","0","0","0","0","0","0","0","0",...,"0"]]']

			### Clean values and remove zeros values
			values_non_zeros = self._get_cleanValues(record_object, entity_name)
			## Store non-zero elements in a list
			############################################
			## Others alternatives to store non-zero elements in a list
			## values_non_zeros = filter(lambda a: a != "0", clean_values)
			## values_non_zeros = ''.join(clean_values.replace('0','').split()
			## values_non_zeros = [i.split("0", 1)[0] for i in clean_values]
			#print("values_non_zeros %s" % values_non_zeros)

			## Create a empty vector for the binary document representation (or patient record representation)
			## for the feature vocabulary lenght
			binary_record = [0] * len(vocabulary)

			## turn-on the value-term (or patient value-feature) correspond into the feature vocabulary
			for i in range(len(values_non_zeros)):
				## Get indice term from the vocabulary representation
				## Condicional for working with custom and dynamic vocabulary composition
				if 'list' in str(type(vocabulary)):
					## custom vocabulary
					indice = vocabulary.index(values_non_zeros[i])
				else:
					## dynamic vocabulary
					indice = vocabulary.tolist().index(values_non_zeros[i])

				## Removed the zero value feature from the binary patient record representation
				## and turn-on the one value feature correspond
				binary_record.pop(indice)
				binary_record.insert(indice, 1)

			##print("feature-list: %s" % binary_record )
			return binary_record

		else:
			## Create a empty vector for the binary document representation (or patient record representation)
			## for the feature vocabulary lenght
			binary_record = [0] * len(vocabulary)

			## Get the indice term from the vocabulary representation from patient feature vocabulary
			## as example of recor object: 'self.vocabulary_age.tolist().index'
			## Condicional for working with custom and dynamic vocabulary composition
			if 'list' in str(type(vocabulary)):
				## custom vocabulary
				index = vocabulary.index(record_object)
			else:
				## dynamic vocabulary
				index = vocabulary.tolist().index(record_object)

			## Removed the zero value feature from the binary patient record representation
			## and turn-on the one value feature correspond
			binary_record.pop(index)
			binary_record.insert(index, 1)

			##print("x3-value: %s" % binary_record )
			return binary_record


	def _concatenate_BinaryFeatures(self, features):
		"""
		Function to concatenate the features selected for build a binary patient representation.
		"""
		features_list = []
		for f in features:
			if (not f) is False:
            	#print("f: {}".format(f))
				features_list.append(",".join([str(f)]))
			else:
            	##print("Feture not used as input")
				pass
		return ','.join(features_list)


	### https://github.com/apanimesh061/Term_Doc_Matrix_ES/blob/master/Term-Document%20Matrix%20from%20Elasticsearch.ipynb
	### https://github.com/hrs/python-tf-idf/blob/master/tfidf.py
	def _build_binaryPhenotype(self,cda_object,x1_name,x2_name,x3_name,x4_name,x5_name,
								x6_name,x7_name,x8_name,x9_name,x10_name,voc_x1,voc_x2,
								voc_x3,voc_x4,voc_x5,voc_x6,voc_x7,voc_x8,voc_x9,voc_x10):
		"""
		Create a binary corpus using Term-document Matrix
		get features values from record_object to find their binary representation
		get terms for each feature vocabulary
		get the feature binary representation by each feature vocabulary
		"""

		# #######################################################################
		# ## Counter to Vocabulary Composition
		# counter_vc = time.time()
		#
		# if self.vocabulary_type != 'custom':
		# 	logger.info('---------------------------------------------------------')
		# 	logger.info('+++ Dynamic Vocabulary +++')
		#
		# 	## Buil a dynamic vocabulary
		# 	vocabularycomposition = VocabularyComposition(self.dataset_dir, self.features_name, self.sandbox, self.year)
		# 	voc_x1,voc_x2,voc_x3,voc_x4,voc_x5,voc_x6,voc_x7,voc_x8,voc_x9,voc_x10 = vocabularycomposition._dynamic_Vocabulary(cda_object,
		# 								x1_name,x2_name,x3_name,x4_name,x5_name,x6_name,x7_name,x8_name,x9_name,x10_name)
		# 	## Write dynamic vocabulary
		# 	vocabularycomposition._write_Vocabulary(x1_name,x2_name,x3_name,x4_name,x5_name,x6_name,x7_name,x8_name,x9_name,x10_name)
		#
		# else:
		# 	##Buil a dynamic vocabulary
		# 	logger.info('---------------------------------------------------------')
		# 	logger.info('+++ Custom Vocabulary +++')
		#
		# 	### Read Custom Vocabulary
		# 	vocabularycomposition = VocabularyComposition(self.dataset_dir, self.features_name, self.sandbox, self.year)
		# 	voc_x1,voc_x2,voc_x3,voc_x4,voc_x5,voc_x6,voc_x7,voc_x8,voc_x9,voc_x10 = vocabularycomposition._custom_Vocabulary(x1_name,
		# 								x2_name,x3_name,x4_name, x5_name,x6_name, x7_name, x8_name, x9_name, x10_name)
		#
		# ## End Counter to label Composition
		# time_vocabulary = time.time() - counter_vc
		# logger.debug('* Label Composition Time: {} *'.format(time_vocabulary))


		logger.info('---------------------------------------------------------')
		logger.info('++ Build a Binary Petient Phenotype Representation (BPPR) ++')


		## Build the binary patient phenotype representation for each features selected
		## first receive for each feature by 'getattr(record_object."entity", "feature")'
		## Get the terms for each feature vocabulary call instance 'self._dynamic_vocabulary'
		## Get the feature binary representation using 'self._get_BinaryRecord'
		## and give the feature=value and feture=vocabulary
		## feature=value='(getattr(record_object."entity", "key"))'
		## feature=vocabulary='voc_x1_demographics["feature"])'
		for record_object in cda_object:
			#print("\n+ ID: %s +" % record_object.x0_header.ID_RSA)

			## x1_demographics
			if (not x1_name) is False and x1_name[0] != 'None':
				##x1.append([(self._get_binaryRecord(getattr(record_object.x1_demographics, i), voc_x1[i])) for i in x1_name])
				#print("+ x1: %s +" %  [(self._get_binaryRecord(getattr(record_object.x1_demographics, i), voc_x1[i], x1_name)) for i in x1_name])
				x1_ = [(self._get_binaryRecord(getattr(record_object.x1_demographics, i), voc_x1[i], x1_name)) for i in x1_name]
			else:
				x1_ = None
				##print("+++ None: x1_demographics +++")

			## x2_admission_details
			if (not x2_name) is False and x2_name[0] != 'None':
				#print("+ x2: %s +" % [(self._get_binaryRecord(getattr(record_object.x2_admission_details,i),voc_x2[i],x2_name)) for i in x2_name])
				x2_ = [(self._get_binaryRecord(getattr(record_object.x2_admission_details, i), voc_x2[i], x2_name)) for i in x2_name]
			else:
				x2_ = None
				##print("+++ None: x2_admission_details +++")

			## x3_hospitalization_details
			if (not x3_name) is False and x3_name[0] != 'None':
				##x3.append([(self._get_binaryRecord(getattr(record_object.x3_hospitalization_details, i), voc_x1[i])) for i in x3_name])
				#print("+ x3: %s +" %  [(self._get_binaryRecord(getattr(record_object.x3_hospitalization_details,i), voc_x3[i], x3_name)) for i in x3_name])
				x3_ = [(self._get_binaryRecord(getattr(record_object.x3_hospitalization_details, i), voc_x3[i], x3_name)) for i in x3_name]
			else:
				x3_ = None
				##print("+++ Que es lo que sucede: x3_hospitalization_details +++")

			## x4_physical_dependence
			if (not x4_name) is False and x4_name[0] != 'None':
				##x4.append( [( self._get_binaryRecord(getattr(record_object.x4_physical_dependence, i ), voc_x4[ i ]) ) for i in x4_name] )
				#print("+ x4: %s +" % [(self._get_binaryRecord(getattr(record_object.x4_physical_dependence, i), voc_x4[i], x4_name)) for i in x4_name] )
				x4_ = [(self._get_binaryRecord(getattr(record_object.x4_physical_dependence, i), voc_x4[i], x4_name)) for i in x4_name]
			else:
				x4_ = None
				##print("+++ None: x4_physical_dependence +++")

			## x5_cognitive_dependence
			if (not x5_name) is False and x5_name[0] != 'None':
				##x5.append( [( self._get_binaryRecord(getattr(record_object.x5_cognitive_dependence, i ), voc_x5[ i ]) ) for i in x5_name] )
				#print("+ x5: %s +" % [(self._get_binaryRecord(getattr(record_object.x5_cognitive_dependence,i),voc_x5[i],x5_name)) for i in x5_name] )
				x5_ = [(self._get_binaryRecord(getattr(record_object.x5_cognitive_dependence, i), voc_x5[i], x5_name)) for i in x5_name]
			else:
				x5_ = None
				##print("+++ None: x5_cognitive_dependence +++")

			## x6_rehabilitation_time
			if (not x6_name) is False and x6_name[0] != 'None':
				##x6.append( [( self._get_binaryRecord(getattr(record_object.x6_rehabilitation_time, i ), voc_x6[ i ]) ) for i in x6_name] )
				#print("+ x6: %s +" % [(self._get_binaryRecord(getattr(record_object.x6_rehabilitation_time,i),voc_x6[i],x6_name)) for i in x6_name] )
				x6_ = [(self._get_binaryRecord(getattr(record_object.x6_rehabilitation_time, i), voc_x6[i], x6_name)) for i in x6_name]
			else:
				x6_ = None
				##print("+++ None: x6_rehabilitation_time +++")

			## x7_related_diagnoses
			###################################################
			## When was encode a range of columns 'record.iloc[:,37:57]' into JSON format
			## such as the case of x7_associated_diagnosis their string representation is malformed
			## For solve this we used 'ast.literal_eval' to interpreting the parse tree elements
			## and replacing them with their literal equivalents.
			if (not x7_name) is False and x7_name[0] != 'None':
				x7_elements = literal_eval(getattr(record_object, "x7_associated_diagnosis"))
				#print("+ x7: %s +" % self._get_binaryRecord(x7_elements,voc_x7[ "x7_associated_diagnosis"],x7_name) )
				x7_ = self._get_binaryRecord(x7_elements, voc_x7["x7_associated_diagnosis"], x7_name)
			else:
				x7_ = None
				##print("+++ None: x7_related_diagnoses +++")

			## x8_primary_morbidity
			if (not x8_name) is False and x8_name[0] != 'None':
				##x8.append( [( self._get_binaryRecord(getattr(record_object.x8_primary_morbidity, i ), voc_x8[ i ]) ) for i in x8_name] )
				#print("+ x8: %s +" % [(self._get_binaryRecord(getattr(record_object.x8_primary_morbidity,i),voc_x8[i],x8_name)) for i in x8_name] )
				x8_ = [(self._get_binaryRecord(getattr(record_object.x8_primary_morbidity,i),voc_x8[i],x8_name)) for i in x8_name]
			else:
				x8_ = None
				##print("+++ None: x8_primary_morbidity +++")

			## x9_clinical_procedures
			if (not x9_name) is False and x9_name[0] != 'None':
				x9_elements = literal_eval(getattr(record_object,"x9_clinical_procedures"))
				#print("+ x9: %s +" % [(self._get_binaryRecord(x9_elements,voc_x9["x9_clinical_procedures"],x9_name))  for i in x9_name] )
				x9_ = [(self._get_binaryRecord(x9_elements,voc_x9["x9_clinical_procedures"],x9_name))  for i in x9_name]
			else:
				x9_ = None
				##print("+++ None: x9_clinical_procedures +++")

			## x10_destination
			if (not x10_name) is False and x10_name[0] != 'None':
				##x10.append([(self._get_binaryRecord(getattr(record_object.x10_destination, i), voc_x10[i])) for i in x10_name])
				#print("+ x10: %s +" % [(self._get_binaryRecord(getattr(record_object.x10_destination, i), voc_x10[i], x10_name)) for i in x10_name])
				x10_ = [(self._get_binaryRecord(getattr(record_object.x10_destination, i), voc_x10[i], x10_name)) for i in x10_name]
			else:
				x10_ = None
				##print("+++ None: x10_destination +++")

			#######################################
			## Concatenate all the binary patient features
			features = [x1_, x2_, x3_, x4_, x5_, x6_, x7_, x8_, x9_, x10_]
			record_BR = self._concatenate_BinaryFeatures(features)
			##print("+++ row_br %s +++" % record_BR)
			self.len_BPPR = len(record_BR)

			## Create one vector for the patient phenoty binary representation
			## Create a new string with clean values or elements
			chars_to_remove=["[", "]"]
			self.binary_PPR.append(record_BR.translate(None, ''.join(chars_to_remove)))
			##print("+ %s +" % record_BR.translate(None, ''.join(chars_to_remove)) )

		##print( "+ all: %s +" % patient_binary_representation )
		return self.binary_PPR


	def _write_binaryPhenotype(self):
		"""
		Write the binary patient phenotype representation in a txt and JSON format used for log-term storage
		is writed into the first stage directory
		../healthData/sandbox-pre-trained/1_Mining-Stage/binary_representation/
		"""

		##file_bppr = str(self.sandbox)+str('/1_Mining-Stage/binary_representation/')+str('BPPR-')+self.year+('.txt')
		file_bppr = str(self.sandbox+self.localdir)+str("BPPR-")+self.features_name+"-"+self.year+(".txt")
		np.savetxt(file_bppr, self.binary_PPR, delimiter=',', fmt='%s')
		##np.savetxt(str(self.sandbox)+str('/1_Mining-Stage/binary_representation/')+str('binaryPPR.txt'), self.binary_PPR, delimiter=',', fmt='%s')
		logger.info('-- Path of BPPR file: {}'.format(file_bppr))

		return self.len_BPPR



### End VocabularyComposition()
###############################

#################################################################################
#																				#
#	Towards a Green Intelligence Applied to Health Care and Well-Being			#
###-----------------------------------------------------------------------------#
# diagnosenet_vocabularycomposition.py                                          #
# Used this class for building a Medical Vocabulary								#
# from feature composition groups PMSI-PACA (features_serializations)			#
# and get the binary patient phenotype representation							#
#################################################################################


from __future__ import print_function
import os, sys
import numpy as np
import pandas as pd
import time
import logging
import glob

## JSON Data Structure Format to Serialization
import json
from objectjson import ObjectJSON

### Temporal libraries
from ast import literal_eval

## Build Vocbulary
import nltk
from nltk import FreqDist
#nltk.download()

## File Helper
from utils.file_helper import FileHelper

## dIAgnoseNET library
logger = logging.getLogger('_dIAgnoseNET_DataMining')

class VocabularyComposition:
	"""
	Used this class for building a Medical Vocabulary
	from feature composition groups PMSI-PACA (features_serializations)
	and get the binary patient phenotype representation.
	"""

	def __init__(self, *args, **kwargs):
		## dIAgnoseNet path variables
		self.dataset_dir = args[0]
		self.features_name = args[1]
		self.sandbox = args[2]
		self.year = args[3]

		## Vocabulary variables
		VocabularyComposition.localdir = str(self.dataset_dir + "/vocabularies_repository/")
		VocabularyComposition.voc_x1_demographics = {}
		VocabularyComposition.voc_x2_admission_details = {}
		VocabularyComposition.voc_x3_hospitalization_details = {}
		VocabularyComposition.voc_x4_physical_dependence = {}
		VocabularyComposition.voc_x5_cognitive_dependence = {}
		VocabularyComposition.voc_x6_rehabilitation_time = {}
		VocabularyComposition.voc_x7_associated_diagnosis = {}
		VocabularyComposition.voc_x8_primary_morbidity = {}
		VocabularyComposition.voc_x9_clinical_procedures = {}
		VocabularyComposition.voc_x10_destination = {}


	def _set_customVocabulary(self, dir_, index, x_name):
		"""
		Load the features vocabularies from entity file,
		indexing each feature vocabulary to get all terms
		and build a dictionary by entity
		to contain all the features vocabularies.
		"""
		## Load the vocabularies from entity file
		file_ = str(dir_[index])
		data = pd.read_csv(file_, sep=',', dtype=str)
		feature_names = list(x_name)
		#feature_names = list(data)
		##print("-- Features names: %s" % feature_names)

		### Indexing each feature vocabulary to get all terms
		entity_dictionary = {}
		for i in feature_names:
			feature_vocabulary = data.iloc[:][i].dropna(how='all')
			##print("-- Name: %s" % i)
			##print("-- Data: %s" % vocabulary_by_feature.tolist())
			entity_dictionary[i] = feature_vocabulary.tolist()
		return entity_dictionary


	def _custom_Vocabulary(self,x1_name,x2_name,x3_name,x4_name,x5_name,
								x6_name,x7_name,x8_name,x9_name,x10_name):
		"""
		Read each feature vocabulary for log-term storage is writed into the first stage directory
		../sandbox-xx/healthData/vocabularies_repository/
		"""

		logger.info('-- Read vocabularies by each feature selected --')
		logger.info('-- Path of repository vocabularies: {}'.format(str(self.localdir)))

		file_vocabulary = glob.glob(str(self.localdir+str('vocabulary-*.csv')))
		file_vocabulary.sort()

		#####################################
		##for i in range(len(file_vocabulary)):
			##print("+++ Vocabularies respository: {}".format(i))
			##entity_dictionary = self._set_Vocabulary(file_vocabulary, i)
			##print("entity dictionary: %s" % entity_dictionary)
		#####################################

		## x1_demographics
		if (not x1_name) is False and x1_name[0] != 'None':
			self.voc_x1_demographics = self._set_customVocabulary(file_vocabulary, 1, x1_name)
			##print("voc_x1_demographics %s" % self.voc_x1_demographics)
		else:
			self.voc_x1_demographics = 0
			##print("+++ None: x1_demographics +++")

		## x2_admission_details
		if (not x2_name) is False and x2_name[0] != 'None':
			self.voc_x2_admission_details = self._set_customVocabulary(file_vocabulary, 2, x2_name)
			##print("voc_x2_admission_details %s" % self.voc_x2_admission_details)
		else:
			self.voc_x2_admission_details = 0
			##print("+++ None: x2_admission_detail  +++")

		## x3_hospitalization_details
		if (not x3_name) is False and x3_name[0] != 'None':
			self.voc_x3_hospitalization_details = self._set_customVocabulary(file_vocabulary, 3, x3_name)
			##print("voc_x3_hospitalization_details %s" % self.voc_x3_hospitalization_details)
		else:
			self.voc_x3_hospitalization_details = 0
			##print("+++ None: x3_hospitalization_details  +++")

		## x4_physical_dependence
		if (not x4_name) is False and x4_name[0] != 'None':
			self.voc_x4_physical_dependence = self._set_customVocabulary(file_vocabulary, 4, x4_name)
			##print("voc_x4_physical_dependence %s" % self.voc_x4_physical_dependence)
		else:
			self.voc_x4_physical_dependence = 0
			##print("+++ None: x4_physical_dependence +++")

		## x5_cognitive_dependence
		if (not x5_name) is False and x5_name[0] != 'None':
			self.voc_x5_cognitive_dependence = self._set_customVocabulary(file_vocabulary, 5, x5_name)
			#print("voc_x5_cognitive_dependence %s" % self.voc_x5_cognitive_dependence)
		else:
			self.voc_x5_cognitive_dependence = 0
			##print("+++ x5_cognitive_dependence  +++")

		## x6_rehabilitation_time
		if (not x6_name) is False and x6_name[0] != 'None':
			self.voc_x6_rehabilitation_time = self._set_customVocabulary(file_vocabulary, 6, x6_name)
			#print("voc_x6_rehabilitation_time %s" % self.voc_x6_rehabilitation_time)
		else:
			self.voc_x6_rehabilitation_time = 0
			##print("+++ x6_rehabilitation_time  +++")

		## x7_associated_diagnosis
		if (not x7_name) is False and x7_name[0] != 'None':
			self.voc_x7_associated_diagnosis = self._set_customVocabulary(file_vocabulary, 7, x7_name)
			##print("voc_x7_associated_diagnosis%s" % self.voc_x7_associated_diagnosis)
		else:
			self.voc_x7_associated_diagnosis = 0
			##print("+++ None: x7_associated_diagnosis +++")

		## x8_primary_morbidity
		if (not x8_name) is False and x8_name[0] != 'None':
			self.voc_x8_primary_morbidity = self._set_customVocabulary(file_vocabulary, 8, x8_name)
			##print("voc_x8_primary_morbidity: %s" % self.voc_x8_primary_morbidity)
		else:
			self.voc_x8_primary_morbidity = 0
			##print("+++ None: x8_primary_morbidity +++")

		## x9_clinical_procedures
		if (not x9_name) is False and x9_name[0] != 'None':
			self.voc_x9_clinical_procedures= self._set_customVocabulary(file_vocabulary, 9, x9_name)
			##print("voc_x9_clinical_procedures: %s" % self.voc_x9_clinical_procedures)
		else:
			self.voc_x9_clinical_procedures = 0
			##print("+++ None: x9_clinical_procedures +++")

		## x10_destination
		if (not x10_name) is False and x10_name[0] != 'None':
			self.voc_x10_destination = self._set_customVocabulary(file_vocabulary, 0, x10_name)
			##print("voc_x10_destination: %s" % self.voc_x10_destination)
		else:
			self.voc_x10_destination = 0
			##print("+++ x10_destination +++")

		return self.voc_x1_demographics, self.voc_x2_admission_details, self.voc_x3_hospitalization_details, self.voc_x4_physical_dependence, self.voc_x5_cognitive_dependence, self.voc_x6_rehabilitation_time,  self.voc_x7_associated_diagnosis, self.voc_x8_primary_morbidity, self.voc_x9_clinical_procedures , self.voc_x10_destination



	def indexing_documents(self, sentence):
		tokens = nltk.word_tokenize(sentence)
		return tokens

	def _set_dynamicVocabulary(self, terms):
		"""
		Tokenize and extract the vocabulary
		"""
 
		## Convert the age terms in String
		raw_terms = str(' '.join(terms))

		############################################################
		############################################################
		## Create a new string with clean values or elements
#		chars_to_remove=["``", "[", "]"]
#		clean_values = raw_terms.translate(None, ''.join(chars_to_remove))
#		print("clean_values %s " % clean_values.split())

		## Tokenize the terms to create a vocabulary for each feature group
		raw_tokens_terms = self.indexing_documents(raw_terms)
		## Extract key terms for clean the Medical Vocabulary
		voc_terms = list(dict.fromkeys(raw_tokens_terms))
		voc_terms.sort()
		vocabulary_terms = np.asarray(voc_terms, dtype='|S50')

		return vocabulary_terms


	def _dynamic_Vocabulary(self,cda_object,x1_name,x2_name,x3_name,x4_name,x5_name,x6_name,x7_name,x8_name,x9_name,x10_name):
		"""
		Indexing the Clinical Document Architecture Object list
		to create a data personilize vocabulary
		"""

		### Features by electronic health record entity
		x1 = []	## x1_demographics
		x2 = []	## x2_admission_details
		x3 = []	## x3_hospitalization_details
		x4 = []	## x4_physical_dependence
		x5 = []	## x5_cognitive_dependence
		x6 = []	## x6_rehabilitation_time
		x7 = []	## x7_associated_diagnosis
		x8 = []	## x8_primary_morbidity
		x9 = []	## x9_clinical_procedures
		x10 = [] ## x10_destination

		## Get list values by '__getattr_(json, key)' from ObjectJSON
		# Convert json string to json object
		#cda_object = json.loads(cda_object)
		# Clinical doc
		# #clinical_obj = cda_object['clinicalDocument']
		# from utils.file_helper import FileHelper
		# fname = '/Users/lion/Documents/py-workspare/tutorship/patient-feature-composition/cda.txt'
		# FileHelper.save_text(fname, json.dumps(cda_object))
		## append list values by EHR entity
		for record_object in cda_object:
			## x1_demographics
			if (not x1_name) is False and x1_name[0] != 'None':
				x1.append([(getattr(record_object.x1_demographics, i )) for i in x1_name])
				##print( "\n++ Call x1_demographics %s ++" % [ ( getattr(record_object.x1_demographics, i ) ) for i in x1_name])
			else:
				#logger.warning('!!! x1: Demographics features are not being used as input !!!')
				x1_name = None
				pass

			## x2_admission_details
			if (not x2_name) is False and x2_name[0] != 'None':
				x2.append([(getattr(record_object.x2_admission_details, i)) for i in x2_name])
				##print("\n++ Call admission details %s ++" % [ ( getattr(record_object.x2_admission_details, i) ) for i in x2_name])
				print("\n++ Call x2 admission details %s ++" % x2)
			else:
				#logger.warning('!!! x2: Admission Details features are not being used as input !!!')
				x2_name = None
				pass

			## x3_hospitalization_details
			if (not x3_name) is False and x3_name[0] != 'None':
				x3.append([(getattr(record_object.x3_hospitalization_details, i)) for i in x3_name])
				##print("\n++ Call x3_hospitalization_details %s ++" % [(getattr(record_object.x3_hospitalization_details, i)) for i in x3_name])
				print("\n++ Call x3_hospitalization_details %s ++" % x3)
			else:
				#logger.warning('!!! x3: Hospitalization details features are not being used as input !!!')
				x3_name = None
				pass

			## x4_physical_dependence
			if (not x4_name) is False and x4_name[0] != 'None':
				x4.append([(getattr(record_object.x4_physical_dependence, i)) for i in x4_name])
				##print("++ Call x4_physical_dependence %s ++" % [ ( getattr(record_object.x4_physical_dependence, i) ) for i in x4_name])
			else:
				#logger.warning('!!! x4: Physical dependence features are not being used as input !!!')
				x4_name = None
				pass

			## x5_cognitive_dependence
			if (not x5_name) is False and x5_name[0] != 'None':
				x5.append([(getattr(record_object.x5_cognitive_dependence, i)) for i in x5_name])
				##print("++ Call x5_cognitive_dependence %s ++" % [( getattr(record_object.x5_cognitive_dependence, i) ) for i in x5_name])
			else:
				#logger.warning('!!! x5: Cognitive dependence features are not being used as input !!!')
				x5_name = None
				pass

			## x6_rehabilitation_time
			if (not x6_name) is False and x6_name[0] != 'None':
				x6.append([(getattr(record_object.x6_rehabilitation_time, i)) for i in x6_name])
				##print("++ Call x6_rehabilitation_time %s ++" % [(getattr(record_object.x6_rehabilitation_time, i)) for i in x6_name])
			else:
				#logger.warning('!!! x6: Rehabilitation time features are not being used as input !!!')
				x6_name = None
				pass

			## x7_associated_diagnosis
			if (not x7_name) is False and x7_name[0] != 'None':
				x7.append(record_object.x7_associated_diagnosis)
				#print("++ Call Related Diagnostics %s ++" %  record_object.x7_associated_diagnosis)
			else:
				#logger.warning('!!! x7: Associated diagnosis features are not being used as input !!!')
				x7_name = None
				pass

			## x8_primary_morbidity
			if (not x8_name) is False and x8_name[0] != 'None':
				x8.append([(getattr(record_object.x8_primary_morbidity, i)) for i in x8_name])
				##print( "++ Call x8_primary_morbidity %s ++" % [ ( getattr(record_object.x8_primary_morbidity, i) ) for i in x8_name] )
			else:
				#logger.warning('!!! x8: Etiological disease feature is not being used as input !!!')
				x8_name = None
				pass

			## x9_clinical_procedures
			if (not x9_name) is False and x9_name[0] != 'None':
				x9.append(record_object.x9_clinical_procedures)
				#print( "++ Call Clinical Procedures %s ++" % record_object.x9_clinical_procedures )
			else:
				#logger.warning('!!! x9: Clinical procedures features are not being used as input !!!')
				x9_name = None
				pass

			## x10_destination
			if (not x10_name) is False and x10_name[0] != 'None':
				x10.append([(getattr(record_object.x10_destination, i)) for i in x10_name])
				##print("++ Call x10_destinatione %s ++" % [(getattr(record_object.x10_destination, i)) for i in x10_name])
			else:
				#logger.warning('!!! x10: Destination features are not being used as input !!!')
				x10_name = None
				pass


		### Get all the terms in a data frame
		x1_demographics = pd.DataFrame(x1, columns=x1_name)
		x2_admission_details = pd.DataFrame(x2, columns=x2_name)
		x3_hospitalization_details = pd.DataFrame(x3, columns=x3_name)
		x4_physical_dependence = pd.DataFrame(x4, columns=x4_name)
		x5_cognitive_dependence = pd.DataFrame(x5, columns=x5_name)
		x6_rehabilitation_time = pd.DataFrame(x6, columns=x6_name)
		x7_associated_diagnosis = pd.DataFrame(x7, columns=x7_name)
		x8_primary_morbidity = pd.DataFrame(x8, columns=x8_name)
		x9_clinical_procedures = pd.DataFrame(x9, columns=x9_name)
		x10_destination = pd.DataFrame(x10, columns=x10_name)

		## Set vocabulary by each feature index 'x1_name['age','sexe','activity']'
		## and extract 'voc_terms' key terms for each feature vocabulary
		## and Add the vocabularies to add to a list entity 'vocabulary_x1_demographics'

		## x1_demographics
		if (not x1_name) is False:
			##voc_x1_demographics = {}
			for i in range(len(x1_name)):
				self.voc_x1_demographics[x1_name[i]] = (self._set_dynamicVocabulary(x1_demographics[x1_name[i]].tolist()))
			##print("voc_x1_demographics %s" % voc_x1_demographics)
		else:
			self.voc_x1_demographics = 0
			##print("+++ None: x1_demographics +++")

		## x2_admission_details
		if (not x2_name) is False:
			##voc_x2_admission_details = {}
			for i in range(len(x2_name)):
				self.voc_x2_admission_details[x2_name[i]] = (self._set_dynamicVocabulary(x2_admission_details[x2_name[i]].tolist()))
			#print("voc_x2_admission_details %s" % voc_x2_admission_details)
		else:
			self.voc_x2_admission_details = 0
			##print("+++ None: x2_admission_detail  +++")

		## x3_hospitalization_details
		if (not x3_name) is False:
			##voc_x3_hospitalization_details = {}
			for i in range(len(x3_name)):
				self.voc_x3_hospitalization_details[x3_name[i]] = (self._set_dynamicVocabulary(x3_hospitalization_details[x3_name[i]].tolist()))
			#print("voc_x3_hospitalization_details %s" % voc_x3_hospitalization_details)
		else:
			self.voc_x3_hospitalization_details = 0
			##print("+++ None: x3_hospitalization_details  +++")

		## x4_physical_dependence
		if (not x4_name) is False:
			##voc_x4_physical_dependence = {}
			for i in range(len(x4_name)):
				self.voc_x4_physical_dependence[x4_name[i]] = (self._set_dynamicVocabulary(x4_physical_dependence[x4_name[i]].tolist()))
			##print("voc_x4_physical_dependence %s" % voc_x4_physical_dependence)
		else:
			self.voc_x4_physical_dependence = 0
			##print("+++ None: x4_physical_dependence +++")

		## x5_cognitive_dependence
		if (not x5_name) is False:
			##voc_x5_cognitive_dependence = {}
			for i in range(len(x5_name)):
				self.voc_x5_cognitive_dependence[x5_name[i]] = (self._set_dynamicVocabulary(x5_cognitive_dependence[x5_name[i]].tolist()))
			##print("voc_x5_cognitive_dependence %s" % voc_x5_cognitive_dependence)
		else:
			self.voc_x5_cognitive_dependence = 0
			##print("+++ x5_cognitive_dependence  +++")

		## x6_rehabilitation_time
		if (not x6_name) is False:
			##voc_x6_rehabilitation_time = {}
			for i in range(len(x6_name)):
				self.voc_x6_rehabilitation_time[x6_name[i]] = (self._set_dynamicVocabulary(x6_rehabilitation_time[x6_name[i]].tolist()))
			##print("vocx6_rehabilitation_time %s" % voc_x6_rehabilitation_time)
		else:
			self.voc_x6_rehabilitation_time = 0
			##print("+++ x6_rehabilitation_time  +++")

		## x7_associated_diagnosis
		if (not x7_name) is False:
			##voc_x7_associated_diagnosis = self._set_vocabulary( x7_associated_diagnosis['x7_associated_diagnosis'] )
			##voc_x7_associated_diagnosis = {}
			#		x7_RD = []
			for i in range(len(x7_name)):
    				
				self.voc_x7_associated_diagnosis[x7_name[i]] = (self._set_dynamicVocabulary(x7_associated_diagnosis[x7_name[i]].tolist()))
				#self.voc_x7_associated_diagnosis[x7_name[i]] = (self._set_dynamicVocabulary(x7_associated_diagnosis['x7_associated_diagnosis']))
		else:
			self.voc_x7_associated_diagnosis = 0
			##print("+++ None: x7_associated_diagnosis +++")

		#########################################################################################
		#########################################################################################
#		x7_RD = ( self._set_vocabulary(x7_associated_diagnosis['x7_associated_diagnosis']))
		##print("voc_x7_associated_diagnosis %s" % voc_x7_associated_diagnosis) ## Real print
# 		print("voc_x7_associated_diagnosis %s" % type(voc_x7_associated_diagnosis) )

#		print("voc x7_RD %s" % x7_RD)
#		print("voc x7_RD %s" % type(x7_RD) )

#		for i in x7_RD:
#			print("for: %s" %i)
		#############################################################################################
		##############################################################################################

		## x8_primary_morbidity
		if (not x8_name) is False:
			##voc_x8_primary_morbidity = {}
			for i in range(len(x8_name)):
				self.voc_x8_primary_morbidity[x8_name[i]] = (self._set_dynamicVocabulary(x8_primary_morbidity[x8_name[i]].tolist()))
				##print("voc_x8_primary_morbidity %s" % voc_x8_primary_morbidity)
		else:
			self.voc_x8_primary_morbidity = 0
			##print("+++ None: x8_primary_morbidity +++")

		## x9_clinical_procedures
		if (not x9_name) is False:
			##voc_y2_clinical_procedures = {}
			for i in range(len(x9_name)):
				self.voc_x9_clinical_procedures[x9_name[i]] = (self._set_dynamicVocabulary(x9_clinical_procedures['x9_clinical_procedures']))
		else:
			self.voc_x9_clinical_procedures = 0
			##print("+++ None: x9_clinical_procedures +++")

		## x10_destination
		if (not x10_name) is False:
			##voc_x6_rehabilitation_time = {}
			for i in range(len(x10_name)):
				self.voc_x10_destination[x10_name[i]] = (self._set_dynamicVocabulary(x10_destination[x10_name[i]].tolist()))
				##print("voc_x10_destination %s" % voc_x10_destination)
		else:
			self.voc_x10_destination = 0
			##print("+++ x10_destination +++")

		#print( self.voc_x1_demographics, type(self.voc_x2_admission_details), type(self.voc_y1_primary_morbidity), type(self.voc_x3_related_diagnoses), type(self.voc_x4_physical_dependence), type(self.voc_x5_cognitive_dependence), type(self.voc_y2_clinical_procedures) )
		print( self.voc_x1_demographics, type(self.voc_x2_admission_details))
		return self.voc_x1_demographics, self.voc_x2_admission_details, self.voc_x3_hospitalization_details, self.voc_x4_physical_dependence, self.voc_x5_cognitive_dependence, self.voc_x6_rehabilitation_time,  self.voc_x7_associated_diagnosis, self.voc_x8_primary_morbidity, self.voc_x9_clinical_procedures , self.voc_x10_destination


	def _write_Vocabulary(self,x1_name,x2_name,x3_name,x4_name,x5_name,x6_name,x7_name,x8_name,x9_name,x10_name):
		"""
		Write each feature vocabulary for log-term storage is writed into the first stage directory
		../sandbox-xx/healthData/vocabularies_repository/
		"""
		dynamic_dir = str(self.sandbox+"/1_Mining-Stage/vocabularies_repository/")
		logger.info('-- Path of Vocabularies respository: {}'.format(str(dynamic_dir)))

		## x1_demographics
		if (not x1_name) is False and x1_name[0] != 'None':
			voc_x1_name = [','.join(str(value) for value in self.voc_x1_demographics[i] ) for i in x1_name]
			file_voc = './' + str(dynamic_dir)+str("vocabulary-x1_demographics-")+self.features_name+"-"+self.year+(".txt")
			FileHelper.save_arraytotext(file_voc, voc_x1_name)
			#np.savetxt(file_voc, voc_x1_name, delimiter=',', fmt='%s')
		else:
			voc_x1_name = 0

		## x2_admission_details
		if (not x2_name) is False and x2_name[0] != 'None':
			voc_x2_name = [','.join(str(value) for value in self.voc_x2_admission_details[i] ) for i in x2_name]
			file_voc = str(dynamic_dir)+str("vocabulary-x2_admission_details-")+self.features_name+"-"+self.year+(".txt")
			#np.savetxt(file_voc, voc_x2_name, delimiter=',', fmt='%s')
			FileHelper.save_arraytotext(file_voc, voc_x2_name)
		else:
			voc_x2_name = 0

		## x3_hospitalization_details
		if (not x3_name) is False and x3_name[0] != 'None':
			voc_x3_name = [','.join(str(value) for value in self.voc_x3_hospitalization_details[i] ) for i in x3_name]
			file_voc = str(dynamic_dir)+str("vocabulary-x3_hospitalization_details-")+self.features_name+"-"+self.year+(".txt")
			#np.savetxt(file_voc, voc_x3_name, delimiter=',', fmt='%s')
			FileHelper.save_arraytotext(file_voc, voc_x3_name)
		else:
			voc_x3_name = 0

		## x4_physical_dependence
		if (not x4_name) is False and x4_name[0] != 'None':
			voc_x4_name = [','.join(str(value) for value in self.voc_x4_physical_dependence[i] ) for i in x4_name]
			file_voc = str(dynamic_dir)+str("vocabulary-x4_physical_dependence-")+self.features_name+"-"+self.year+(".txt")
			#np.savetxt(file_voc, voc_x4_name, delimiter=',', fmt='%s')
			FileHelper.save_arraytotext(file_voc, voc_x4_name)
		else:
			voc_x4_name = 0

		## x5_cognitive_dependence
		if (not x5_name) is False and x5_name[0] != 'None':
			voc_x5_name = [','.join(str(value) for value in self.voc_x5_cognitive_dependence[i] ) for i in x5_name]
			file_voc = str(dynamic_dir)+str("vocabulary-x5_cognitive_dependence-")+self.features_name+"-"+self.year+(".txt")
			#np.savetxt(file_voc, voc_x5_name, delimiter=',', fmt='%s')
			FileHelper.save_arraytotext(file_voc, voc_x5_name)
		else:
			voc_x5_name = 0

		## x6_rehabilitation_time
		if (not x6_name) is False and x6_name[0] != 'None':
			voc_x6_name = [','.join(str(value) for value in self.voc_x6_rehabilitation_time[i] ) for i in x6_name]
			file_voc = str(dynamic_dir)+str("vocabulary-x6_rehabilitation_time-")+self.features_name+"-"+self.year+(".txt")
			#np.savetxt(file_voc, voc_x6_name, delimiter=',', fmt='%s')
			FileHelper.save_arraytotext(file_voc, voc_x6_name)
		else:
			voc_x5_name = 0

		## x7_associated_diagnosis
		if (not x7_name) is False and x7_name[0] != 'None':
			voc_x7_name = [','.join(str(value) for value in self.voc_x7_associated_diagnosis[i] ) for i in x7_name]
			file_voc = str(dynamic_dir)+str("vocabulary-x7_associated_diagnosis-")+self.features_name+"-"+self.year+(".txt")
			#np.savetxt(file_voc, voc_x7_name, delimiter=',', fmt='%s')
			FileHelper.save_arraytotext(file_voc, voc_x7_name)
		else:
			voc_x7_name = 0

		## x8_primary_morbidity
		if (not x8_name) is False and x8_name[0] != 'None':
			voc_x8_name = [','.join(str(value) for value in self.voc_x8_primary_morbidity[i] ) for i in x8_name]
			file_voc = str(dynamic_dir)+str("vocabulary-x8_primary_morbidity-")+self.features_name+"-"+self.year+(".txt")
			#np.savetxt(file_voc, voc_x8_name, delimiter=',', fmt='%s')
			FileHelper.save_arraytotext(file_voc, voc_x8_name)
		else:
			voc_x8_name = 0

		## x9_clinical_procedures
		if (not x9_name) is False and x9_name[0] != 'None':
			voc_x9_name = [','.join(str(value) for value in self.voc_x9_clinical_procedures[i] ) for i in x9_name]
			file_voc = str(dynamic_dir)+str("vocabulary-x9_clinical_procedures-")+self.features_name+"-"+self.year+(".txt")
			#np.savetxt(file_voc, voc_x9_name, delimiter=',', fmt='%s')
			FileHelper.save_arraytotext(file_voc, voc_x9_name)
		else:
			voc_x9_name = 0

		## x10_destination
		if (not x10_name) is False and x10_name[0] != 'None':
			voc_x10_name = [','.join(str(value) for value in self.voc_x10_destination[i] ) for i in x10_name]
			file_voc = str(dynamic_dir)+str("vocabulary-x10_destination-")+self.features_name+"-"+self.year+(".txt")
			#np.savetxt(file_voc, voc_x10_name, delimiter=',', fmt='%s')
			FileHelper.save_arraytotext(file_voc, voc_x10_name)
		else:
			voc_x10_name = 0


### End VocabularyComposition()
###############################

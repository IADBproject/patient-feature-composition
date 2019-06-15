#################################################################################
#																				#
#	Towards a Green Intelligence Applied to Health Care and Well-Being			#
###-----------------------------------------------------------------------------#
# diagnosenet_featurescomposition.py											#
# Features Composition to extract patient's attributes for driving				#
# their phenotype representation from Intensive Care Unit (ICU) PMSI-PACA		#
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

## Build Vocbulary
import nltk
from nltk import FreqDist
from collections import OrderedDict
#nltk.download()

## dIAgnoseNET library
from diagnosenet.featuresengineerig import FeaturesEngineeringRules
from diagnosenet.cdajson import cdaJSON
logger = logging.getLogger('_dIAgnoseNET_DataMining')

class FeaturesComposition:
	"""
	Features Composition to extract patient's attributes for driving
	their phenotype representation from Intensive Care Unit (ICU) PMSI-PACA

	* This version implements the structured Clinical Documents Architecture (CDA)
	in a JSON format to mining the Hospital Data Warehouse (PMSI) in PACA.
	"""

	def __init__(self, *args, **kwargs):
		## dIAgnoseNet path variables
		self.dataset_dir = args[0]
		self.features_name = args[1]
		self.sandbox = args[2]
		self.year = args[3]

		## CDA variables
		FeaturesComposition.cinical_document = ""
		FeaturesComposition.Dir_rawdata = str(self.dataset_dir + "/PMSI-PACA/")


	def _set_featuresSerializer(self, icu_rsa):
		"""
        CDA is a document markup standard that specifies the structure and semantics
        of a clinical document (such as a discharge summary or progress note) for the purpose of exchange.
        A CDA document is a defined and complete information object that can include text, images, sounds,
        and other multimedia content. It can be transferred within a message and can exist independently,
        outside the transferring message." [Dolin et al, 2006]
		"""

		## Get 'icu_rsa' the EHR_Features-Patient Matrix for each year
		#logger.info('-- Get ICU Data Shape: {} --'.format(str(icu_rsa.shape)))

		## Call CDA shema acording the RSA version
		#cda = cdaJSON(self.Dir_rawdata, self.features_name, self.sandbox, self.year)
		cda = cdaJSON(self.Dir_rawdata, self.sandbox, self.year)

		## Call the 'cda.cdaSchemaM24(icu_rsa, i)' class for
		## building the clinical document schema record by record loaded in 'icu_rsa'
		features_serialized = {
								#"clinicalDocument": [(cda.cdaSchemaM24(icu_rsa, i)) for i in range( 100 ) ]
								"clinicalDocument": [(cda.cdaMimic(icu_rsa, i)) for i in range( icu_rsa.shape[0] ) ]
								}

		## Clinical document structured in JSON format
		# Set indent=None to auto-generate a new line feed (\n) in json file using Python.
		self.cinical_document = json.dumps( features_serialized, sort_keys=True, indent=None, separators=(',', ': ') )

		## Used return 'cda_serialized' for working	with raw CDA in JSON format
		#cda_serialized = self.cinical_document
		return self.cinical_document


	def _get_featuresSerialized(self, icu_rsa):
		"""
		Get the Clinical Document Architecture Object JSON list
		for processing all terms of feature group for each record
		"""

		## Set 'icu_rsa' the EHR_Features-Patient Matrix for each year
		self._set_featuresSerializer(icu_rsa)

		## Reading 'self.clinical_document' in memory all records serialized in a CDA JSON format
		json_cda = json.loads(self.cinical_document )
		#print("+++ json_cda %s  " % json_cda)

		## Encapsulation of 'cda_object'
		## for bundling the 'self.clinical_document in a ObjectJSON Class
		cda_object = []

		## Encapsulates record by record
		for record in json_cda["clinicalDocument"]:
			cda_object.append( ObjectJSON(record) )

			## Processing all terms of feature group for each record
			#record_object = ObjectJSON(record)
			#print("Example of CDA Object | age: %s" % record_object.x1_demographics.age )
			#print("Example of CDA Object | etiology: %s" % record_object.y1_primary_morbidity.etiology )

		return cda_object


	def _write_featuresSerialized(self, icu_rsa):
		"""
		Write the clinical document in a JSON format used for long-term storage,
		is writed into the first stage directory:
		../healthData/sandbox-(name_EHR_db)/1_Mining-Stage/features_serialization/
		"""

		logger.info('-- Build the Clinical Document Architecture in a JSON Format --')

		# Set the EHR_Features-Patient Matrix for each year
		self._set_featuresSerializer(icu_rsa)
		file_name = str(self.dataset_dir+'/CDA_Serialization/'+'clinical_document_mimic-'+self.year+'.json')
		##with open(self.dataset_dir+'/CDA_Serialization/'+'clinical_document-'+self.year+'.json', 'w+') as f:
		##with open(self.sandbox+'/1_Mining-Stage/features_serialization/'+'clinical_document-'+self.year+'.json', 'w+') as f:
		with open(file_name, 'w+') as f:
			f.write(self.cinical_document)
		logger.info('-- Path of CDA Records: {}--'.format(file_name))

		return self.cinical_document


	def _read_featuresSerialized(self):
		"""
		Read the Clinical Document Architecture in a JSON format from disk and
		return an CDA Object JSON for processing all terms of feature group for each record
		"""

		logger.info('-- Read the Clinical Document Architecture in a JSON Format --')

		## load the clinical_document-year-.json
		file_name = str(self.dataset_dir+'/CDA_Serialization/'+'clinical_document_mimic-'+self.year+'.json')
		#file_name = str(self.sandbox+'/1_Mining-Stage/features_serialization/'+'clinical_document-'+self.year+'.json')
		logger.info('-- Path of CDA Records: {}--'.format(file_name))

		with open(file_name) as clinical_d:
			## 'json_cda' load record by record serialized in a CDA JSON format
			json_cda = json.load(clinical_d)

			## Encapsulation of 'cda_object'
			## for bundling the 'self.clinical_document in a ObjectJSON Class
			cda_object = []

			## Encapsulates record by record
			for record in json_cda["clinicalDocument"]:
				cda_object.append(ObjectJSON(record))

			## Processing all terms of feature group for each record
			#record_object = ObjectJSON(record)
			#print("Example of CDA Object | age: %s" % record_object.x1_demographics.age )
			#print("Example of CDA Object | etiology: %s" % record_object.y1_primary_morbidity.etiology )

		return cda_object



### End FeaturesComposition()
#############################

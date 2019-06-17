#########################################################################################
#																						#
#		Towards a Green Intelligence Applied to Health Care and Well-Being				#
#---------------------------------------------------------------------------------------#
# diagnosenet_cda.py		                                                   			#
# This class specified the rules for grouping values of features                		#
# to create an additional (engineered) features                                 		#
#########################################################################################

from __future__ import print_function
import os, sys
import numpy as np
import pandas as pd
import time, math

import json

### Method to remove all unicode characters
from ast import literal_eval

## dIAgnoseNET API
from diagnosenet.featuresengineerig import FeaturesEngineeringRules


class cdaJSON:
	"""
	CDA is a document that consists of a header and body, It is build in a JSON format:
	** Header: Includes patient information, author, creation date,
			   document type, provider, etc.
	** Body: Includes clinical details, diagnosis, medications, follow-up, etc.
			 presented as free text in one or multiple sections, and
			 may optionally also include coded entries.
	[https://corepointhealth.com/resource-center/hl7-resources/hl7-cda]
	"""

	def __init__(self, *args, **kwargs):
		## dIAgnoseNet path variables
		self.Dir_Rawdata = args[0]
		self.sandbox = args[1]
		self.year = args[2]

	def _get_clinicalprocedures(self, id_record, num_procedures):
		"""
		Get the clinical procedures associated to each patient record.
		"""

		## load the clinical procedures asociates
		icu_procedures=pd.read_csv(self.Dir_Rawdata+self.year+"/"+str("ssr-acte-"+self.year+".txt"),
									header=0,sep=";",encoding='latin-1',dtype="str")
		## Second target predict the clinical procedures
		x9 = []
		if int(num_procedures) > 0:

			######################
			row = icu_procedures.groupby('idnum').agg({"CodActe": lambda x: x.tolist()}).reindex([id_record])
			##row2 = icu_procedures.groupby('idnum').agg( lambda x: x.tolist() ).reindex([id_record])
			x9.append(row['CodActe'].to_json(orient="records"))
			##print("------------------------------------")
			##print("++++ row2: %s || procedures %s ++++" % (id_record, row2['CodActe'].to_json(orient="records")) )
			##print("++++ row2: %s ++++\n\n" % type(row2) )
			##procedures = row2['CodActe'].values
			##procedures = proc.to_string(index = False)

			##print("+++++ procedures: %s" % type(procedures) )
			##print("+++++ patient: %s || procedures %s ++++" % (id_record, procedures) )
			##print("------------------------------------")
			##y2.append( row2.iloc[0]['CodActe'])
			#print("+++ row2: +++ \n%s" % row2.iloc[0]['CodActe'] )

		elif int(num_procedures) == 0:
			x9.append('0')
		##print("++++ cda-y2: %s ++++" % y2)

		return x9


	def cdaMimic(self, icu_rsa, id_record):
    	#####################################################
		## Get the EHR_Features-Patient Matrix for each year
		## Indexing by number of row patient record
		record = icu_rsa.reindex([id_record])

		# x11 = record.iloc[:,2].astype('str').values	 	### AGE
		# x12 = record.iloc[:,8].astype('str').values		### DOB
		# x13 = record.iloc[:,32].astype('str').values	### Marital status
		# x14 = record.iloc[:,33].astype('str').values	### Ethnicity

		# ### x2_admission_details
		# x21 = record.iloc[:,26].astype('str').values	### admission_type
		# x22 = record.iloc[:,27].astype('str').values	### admission_location
		# x23 = record.iloc[:,28].astype('str').values	### discharge_location
		# x24 = record.iloc[:,29].astype('str').values	### insurance
		# x25 = record.iloc[:,7].astype('str').values		### expire_flag

		# ### x3_hospitalization_details
		# x35 = record.iloc[:,61].astype('str').values	### first_week
		# x36 = record.iloc[:,62].astype('str').values	### first_week
		# x37 = record.iloc[:,67].astype('str').values	### first_week
		# #x38 = record.iloc[:,84].astype('str').values	### first_week

		# ### x7_associated_diagnosis
		# x71 =  json.dumps(record.iloc[:,84].values.tolist())

		# ### 
		# x101 = record.iloc[:,67].astype('str').values	### destination

		x11 = record.iloc[:,17].astype('str').values	 	### AGE
		x12 = record.iloc[:,6].astype('str').values		### DOB
		x13 = record.iloc[:,50].astype('str').values	### Marital status
		x14 = record.iloc[:,44].astype('str').values	### Ethnicity

		### x2_admission_details
		x21 = record.iloc[:,19].astype('str').values	### admission_type
		x22 = record.iloc[:,18].astype('str').values	### admission_location
		x23 = record.iloc[:,34].astype('str').values	### discharge_location
		x24 = record.iloc[:,48].astype('str').values	### insurance
		x25 = record.iloc[:,16].astype('str').values		### expire_flag

		### x3_hospitalization_details
		x35 = record.iloc[:,55].astype('str').values	### icu_first_careunit
		x36 = record.iloc[:,66].astype('str').values	### icu_last_careunit
		#x37 = record.iloc[:,66].astype('str').values	### first_week
		#x38 = record.iloc[:,84].astype('str').values	### first_week

		### x7_associated_diagnosis
		x71 =  json.dumps(record.iloc[:,84].values.tolist())

		### 
		x101 = record.iloc[:,68].astype('str').values	### destination

		######################################################################
		### Serialize each record to create a clinical document in JSON format
		### SSR_Methodologique_2016
		cda_record = {
					'x1_demographics': {
						'gender': x11[0],
						'age': x12[0],
						'marital_status': x13[0],
						'ethnicity': x14[0],
					},
					'x2_admission_details': {
						'admission_type': x21[0],
						'admission_location': x22[0],
						'discharge_location': x23[0],
						'insurance': x24[0],
						'expire_flag': x25[0]
					},
					'x3_hospitalization_details': {
						'icu_first_careunit': x35[0],
						'icu_last_careunit': x36[0]
						#'icu_los': x37[0]
					},
					# 'x7_associated_diagnosis': {
					# 	'procedure': x71[0]
					# },
					"x10_destination":{
						"length_of_stay": x101[0]
					}
				}

		return cda_record

	def cdaSchemaM24(self, icu_rsa, id_record):
		#####################################################
		## Get the EHR_Features-Patient Matrix for each year
		## Indexing by number of row patient record
		record = icu_rsa.reindex([id_record])

		### x0_header
		x01 = record.iloc[:,0].astype('str').values		### ID_RSA
		x02 = record.iloc[:,1].astype('str').values		### ID_Hospital
		x03 = record.iloc[:,2].astype('str').values		### RSA_Version
		x04 = record.iloc[:,14].astype('str').values	### ID_Patient

		### x1_demographics
		x11 = record.iloc[:,15].astype('str').values	### activity
		x12 = record.iloc[:,16].astype('str').values	### age
		x13 = record.iloc[:,17].astype('str').values	### sexe
		x14 = record.iloc[:,18].astype('str').values	### postal_code

		### Feature engineering rules for groupoing values of features
		engineering_rules = FeaturesEngineeringRules()
		age_group = engineering_rules.setAgeGroup(int(x12[0]))	### age_group

		### x2_admission_details
		x21 = record.iloc[:,19].astype('str').values	### input_mode
		x22 = record.iloc[:,20].astype('str').values	### input_source
		x23 = record.iloc[:,23].astype('str').values	### previous_state
		x24 = record.iloc[:,29].astype('str').values	### first_week

		### x3_hospitalization_details
		x31 = record.iloc[:,25].astype('str').values	### numdays_week
		x32 = record.iloc[:,27].astype('str').values	### numdays_weekend
		x33 = record.iloc[:,28].astype('str').values	### sequence_number
		x34 = record.iloc[:,33].astype('str').values	### surgery_time

		numdays_hospitalized = record.iloc[:,25].astype('int').values + record.iloc[:,27].astype('int').values
		x35 = numdays_hospitalized.astype('str')

		### x4_physical_dependence
		x41 = record.iloc[:,57].astype('str').values	### dressing
		x42 = record.iloc[:,58].astype('str').values	### displacement
		x43 = record.iloc[:,59].astype('str').values	### feeding
		x44 = record.iloc[:,60].astype('str').values	### continence
		x45 = record.iloc[:,63].astype('str').values	### wheelchair

		### x5_cognitive_dependence
		x51 = record.iloc[:,61].astype('str').values	### comportement
		x52 = record.iloc[:,62].astype('str').values	### communication

		### x6_rehabilitation_time
		x61 = record.iloc[:,64].astype('str').values	### mechanical_rehab
		x62 = record.iloc[:,65].astype('str').values	### motorsensory_rehab
		x63 = record.iloc[:,66].astype('str').values	### neuropsychological_rehab
		x64 = record.iloc[:,67].astype('str').values	### cardiorespiratory_rehab
		x65 = record.iloc[:,68].astype('str').values	### nutritional_rehab
		x66 = record.iloc[:,69].astype('str').values	### urogenitalsphincter_rehab
		x67 = record.iloc[:,70].astype('str').values	### kidneys_rehab
		x68 = record.iloc[:,71].astype('str').values	### electrical_equipment
		x69 = record.iloc[:,72].astype('str').values	### collective-rehab
		x610 = record.iloc[:,73].astype('str').values	### bilans
		x611 = record.iloc[:,74].astype('str').values	### physiotherapy
		x612 = record.iloc[:,75].astype('str').values	### balneotherapy

		### x7_associated_diagnosis
		#x7 = record.iloc[:,37:57].to_json(orient="records")
		x7 =  json.dumps(record.iloc[:,37:57].values.tolist() )
		##print("+++ x7: %s +++" % type(x3) )

		### x8_nosological_group
		x81 = record.iloc[:,34].astype('str').values	### care_purpose
		x82 = record.iloc[:,35].astype('str').values	### morbidity_manifestation
		x83 = record.iloc[:,36].astype('str').values	### etiological_condition
		x84 = record.iloc[:,4].astype('str').values		### patient_period
		x85 = record.iloc[:,5].astype('str').values		### major_clinical_category

		### Target number One
		dG = record.iloc[:,81].astype('str').values		### Change the nomenclature

		### x9_clinical_procedures
		x91 = record.iloc[:,76].astype('int').values	### number_procedures
		x92 = self._get_clinicalprocedures(x01[0], x91)	### clinical_procedures
		##print("+++ x9: %s +++ " % y2)

		### x10_destination
		x101 = record.iloc[:,30].astype('str').values	### last_week
		x102 = record.iloc[:,21].astype('str').values	### output_mode (home, transfer or death)
		x103 = record.iloc[:,22].astype('str').values	### destination


		######################################################################
		### Serialize each record to create a clinical document in JSON format
		### SSR_Methodologique_2016
		cda_record = {
					"x0_header": {
						"ID_RSA":  x01[0],
						"ID" : id_record,
						"hospital": x02[0],
						"patient": x04[0],
						"patient_Rol": "Inpatient",
						"rsa_V": x03[0]
					},
					"x1_demographics": {
						"age": x12[0],
						"sexe": x13[0],
						"age_group": age_group,
						"activity": x11[0],
						"postal_code": x14[0]
					},
					"x2_admission_details": {
						"input_mode": x21[0],
						"input_source": x22[0],
						"previous_state": x23[0],
						"first_week": x24[0]
					},
					"x3_hospitalization_details":{
						"numdays_hospitalized": x35[0],
						"sequence_number": x33[0],
						"surgery_time": x34[0],
					},
					"x4_physical_dependence":{
						"dressing": x41[0],
						"displacement": x42[0],
						"feeding": x43[0],
						"continence": x44[0],
						"wheelchair": x45[0]
					},
					"x5_cognitive_dependence": {
						"comportement": x51[0],
						"communication": x52[0]
					},
					"x6_rehabilitation_time":{
						"mechanical_rehab": x61[0],
						"motorsensory_rehab": x62[0],
						"neuropsychological_rehab": x63[0],
						"cardiorespiratory_rehab": x64[0],
						"nutritional_rehab": x65[0],
						"urogenitalsphincter_rehab": x66[0],
						"kidneys_rehab": x67[0],
						"electrical_equipment": x68[0],
						"collective-rehab": x69[0],
						"bilans":  x610[0],
						"physiotherapy": x611[0],
						"balneotherapy": x612[0],
					},
					"x7_associated_diagnosis": x7,
					"x8_primary_morbidity":{
						"care_purpose": x81[0],
						"morbidity": x82[0],
						"etiology": x83[0],
						"major_clinical_category": x84[0] + x85[0]	## Concatenation
					},
					"x9_clinical_procedures": str(x92),
					"x10_destination":{
						"last_week": x101[0],
						"output_mode": x102[0],
						"destination": x103[0]
					}
				}

		return cda_record


### End cdaJSON()
#################

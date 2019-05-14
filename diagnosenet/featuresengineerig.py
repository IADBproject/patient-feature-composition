#################################################################################
#																				#
#	Towards a Green Intelligence Applied to Health Care and Well-Being			#
###-----------------------------------------------------------------------------#
# FeatureEngineeringRules.py                                                    #
# This class specified the rules for grouping values of patient features		#
# to create an additional (engineered) features in a phenotype representation	#
#################################################################################

from __future__ import print_function
import os, sys
import numpy as np
import pandas as pd
import time


class FeaturesEngineeringRules:
	"""
	This class specified the rules for grouping values of patient features
	to create an additional (engineered) features in a phenotype representation
	"""

	def __init__(self, *args, **kwargs):		
		self.data = ""

	def setAgeGroup(self, age):
		age_group = age

		if age < 7:	age_group = "0-6"
		elif age > 6 and age <= 12:	age_group = "7-12"
		elif age > 12 and age <= 17: age_group = "13-17"
		elif age > 17 and age <= 29: age_group = "18-29"
		elif age > 29 and age <= 59: age_group = "30-59"
		elif age > 59 and age <= 74: age_group = "60-74"
		else: age_group = "+74"

		return str(age_group)
		
	
### End FeaturesEngineeringRules()
##################################

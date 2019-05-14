#################################################################################
#                                                                               #
#   ObjectJSON															        #
###-----------------------------------------------------------------------------#
# objectjson.py				                                                    #
# Is a tool to create Python objects from large, complicated JSON objects.		#
# Apache License 2.0, January 2004												#
# url='https://github.com/abmohan/objectjson'									#
#################################################################################

import json

class ObjectJSON:

	def __init__(self, json_data ):

		self.json_data = ""

		if isinstance(json_data, str):
			json_data = json.loads(json_data)
			self.json_data = json_data
		
		elif isinstance(json_data, dict):
			self.json_data = json_data

	def __getattr__(self, key):
		##########################################################

		#print("+++ Key: %s +++" % key)

		##########################################################
		if key in self.json_data:
			if isinstance(self.json_data[key], (list, dict)):
				return ObjectJSON(self.json_data[key])
			else:
				return self.json_data[key]
		else:
			raise Exception('There is no json_data[\'{key}\'].'.format(key=key))


	def __repr__(self):
		out = self.__dict__
		return '%r' % (out['json_data'])



### End ObjectJSON()
####################


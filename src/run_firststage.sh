#!/bin/bash

#################################################################################
#                                                                               #
#   Towards a Green Intelligence Applied to Health Care and Well-Being          #
###-----------------------------------------------------------------------------#
# run_firststage.sh                                                             #
# dIAgnoseNet data-management													                          #
# Execution steps to execute a dIAgnoseNet data-mining library at first stage	  #
# to create a hierarchical sandbox and linked with enerGyPU monitor.			      #
#################################################################################

#################################################################################
###		 dIAgnoseNet Data-management Parameters Configuration
###-----------------------------------------------------------------------------#
###		Global variables to data indexing
###-----------------------------------------------------------------------------#
ANN_NAME="deep_patient"
FEATURES_NAME="CUSTOM_x1_x2_x3_x4_x5_x7_x83_Y1"
DATASET_Dir="../healthData/"
RAWDATA_NAME="PMSI-PACA"
SANDBOX="../healthData/sandbox-"$FEATURES_NAME
YEAR="2008"

###-----------------------------------------------------------------------------#
###		First Stage: Select features to build a Binary Patient phenotype
###-----------------------------------------------------------------------------#
##features_X[1]='age','sexe','age_group','activity','postal_code'
features_X[1]='sexe','age_group','activity'
##features_X[1]='None'
features_X[2]='input_mode','input_source','previous_state','first_week'
##features_X[2]='None'
features_X[3]='numdays_hospitalized','sequence_number','surgery_time'
##features_X[3]='None'
features_X[4]='dressing','feeding','displacement','continence'
##features_X[4]='None'
features_X[5]='communication','comportement'
##features_X[5]='None'
X60='mechanical_rehab','motorsensory_rehab','neuropsychological_rehab',
X61='cardiorespiratory_rehab','nutritional_rehab','urogenitalsphincter_rehab',
X62='kidneys_rehab','electrical_equipment','collective-rehab',
X63='bilans','physiotherapy','balneotherapy'
##features_X[6]=$X60$X61$X62$X63
features_X[6]='None'
features_X[7]='x7_associated_diagnosis'
##features_X[7]='None'
##features_X[8]='care_purpose','morbidity','etiology','major_clinical_category'
features_X[8]='etiology'
##features_X[8]='None'
##features_X[9]='x9_clinical_procedures'
features_X[9]='None'
##features_X[10]='last_week','output_mode','destination'
features_X[10]='None'

## Select the labels
MEDICAL_TARGET='y1'
#MEDICALTARGET='y2'
#MEDICAL_TARGET='y3'

## Select function to build or read the feature vocabularies
#VOCABULARY_TYPE='custom'
VOCABULARY_TYPE='None'

###-----------------------------------------------------------------------------#
###     enerGyPU Monitor: Global variables to locate power consumption measures
###-----------------------------------------------------------------------------#
TESTBED_Dir="../enerGyPU/testbed/"
HOST=$(hostname)
DATE=`date +%Y%m%d%H%M`
TESTBED_ARGV=$HOST-$ANN_NAME-$MEDICAL_TARGET-$FEATURES_NAME-$DATE

###     Parameters Configuration
#################################################################################


#################################################################################
###     System Parameters Configuration                                         #
###-----------------------------------------------------------------------------#
###     Set python 2.7 version to execute dIAgnoseNet-0.1
###-----------------------------------------------------------------------------#
alias python=/usr/bin/python2.7

###     End System Configuration parameters
#################################################################################


#################################################################################
###         enerGyPU Monitor Parameters Configuration
###-----------------------------------------------------------------------------#
###     Global variables to setting the Testbed
###-----------------------------------------------------------------------------#
mkdir $TESTBED_Dir/$TESTBED_ARGV

###-----------------------------------------------------------------------------#
###     Identification of the GPU in the machine
###-----------------------------------------------------------------------------#
##i=0; for id in $(nvidia-smi | grep 0000 | awk '{print $8}'); do GPU[$i]=$id; i=$i+1; done;
##if [ ${#GPU[@]} == 1 ]; then
    ## Executes the enerGyPU_record.sh
##    ../enerGyPU/dataCapture/enerGyPU_record-desktop.sh $TESTBED_Dir $TESTBED_ARGV &
##elif [ ${#GPU[@]} == 8 ]; then
    ## Executes the enerGyPU_record.sh
##    ../enerGyPU/dataCapture/enerGyPU_record-cluster.sh $TESTBED_Dir $TESTBED_ARGV &
##else
##    echo "++ no NVIDIA GPU detected ++"
##fi
###     End enerGyPU Monitor Parameters Configuration
#################################################################################


#################################################################################
###     First Stage Parameters Configuration
###-----------------------------------------------------------------------------#
###		Global variables to setting the First Sandbox
###-----------------------------------------------------------------------------#
if [ ! -d "$SANDBOX" ]; then
mkdir $SANDBOX
mkdir $SANDBOX"/1_Mining-Stage/"
mkdir $SANDBOX"/1_Mining-Stage/binary_representation/"
mkdir $SANDBOX"/1_Mining-Stage/vocabularies_repository/"

###------------------------------------------------------------------------------
###		dIAgnoseNET executions
###------------------------------------------------------------------------------
python diagnosenet_datamining.py $DATASET_Dir $FEATURES_NAME $RAWDATA_NAME $SANDBOX $YEAR $TESTBED_ARGV ${features_X[@]} $MEDICAL_TARGET $VOCABULARY_TYPE
else
echo "------------------------------------------------------------------------"
echo "+++ Using the BPPR from exists directory: "$SANDBOX" +++"
echo "------------------------------------------------------------------------"
fi

##kill %1
###     End dIAgnoseNET executions
#################################################################################

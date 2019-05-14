#################################################################################
#                                                                               #
#	Towards a Green Intelligence Applied to Health Care and Well-Being          #
###-----------------------------------------------------------------------------#
# diagnosenet_loggingconfig.py                                                  #
# Class for logging config and access to all methods in dIAgnoseNET Library.    #
#################################################################################

import logging
import logging.handlers


class Config:
    """
    Logging config allows the access to all methods in DiagnoseNET framework.
    """

    def __init__(self):
        pass

    def _setup_logger(self, logger_name, log_file, level):
        logger = logging.getLogger(logger_name)

        ## Set format to write into the file
        file_formatter = logging.Formatter('[%(asctime)s] {%(pathname)s:%(lineno)d} %(name)s %(levelname)s %(message)s', '%Y-%m-%d %H:%M:%S')
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(file_formatter)

        ## Set format to shows in console
        console_formatter = logging.Formatter('[%(asctime)s] %(name)s - %(levelname)s %(message)s', '%H:%M:%S')
        console = logging.StreamHandler()
        console.setFormatter(console_formatter)

        ### Add configures to general logging
        logger.setLevel(level)
        logger.addHandler(file_handler)
        logger.addHandler(console)


### End LoggingConfig()
#######################

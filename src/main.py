import logging
import argparse

# Defines the default dataset file path
DATASET_DEFAULT_PATH = '../data/raw/cleveland.data'

# Defines the global constant for the default log level
LOG_DEFAULT_LEVEL = 'CRITICAL'

# Defines the global variable for the argument parser
parser = None

# Defines the global variable for the command-line arguments
args = None

# Creates and adds the necessary arguments to the parser
def preconfigure_parser():

	# Interprets parser as a global variable inside the scope
	global parser

	# Create the Argument Parser
	parser = argparse.ArgumentParser(description='Simple Binary Classification Neural Network for the Cleveland Heart Disease Dataset')

	# Add the positional argument for the path
	parser.add_argument(
		'path',
		type=str,
		nargs='?',
		default=DATASET_DEFAULT_PATH,
		help='The path to the dataset file (must be comma-separated)'
	)

	# Add the positional argument for the log level
	parser.add_argument(
		'log',
		nargs='?',
		default=LOG_DEFAULT_LEVEL,
		choices=[
			'DEBUG',
			'INFO',
			'WARNING',
			'ERROR',
			'CRITICAL'
		],
		help=f'Set the logging level (default: {LOG_DEFAULT_LEVEL})'
	)

# Configures the logger debugging level
def preconfigure_logger():

	# Interprets parser as a global variable inside the scope
	global parser

	# Interprets args as a global variable inside the scope
	global args

	# Maps the argument string to the log level data type
	log_level = getattr(logging, args.log)

	# Sets the log level in the default logger
	logging.basicConfig(level=log_level)

# Main file execution guard
if __name__ == '__main__':

	# Creates and adds the necessary arguments to the parser
	preconfigure_parser()

	# Parse the argument
	args = parser.parse_args()

	# Configures the logger debugging level
	preconfigure_logger()



import os
import logging
import argparse
import pandas
import tensorflow

# Defines the current python script absolute path
CURRENT_PATH = os.path.dirname(os.path.abspath(__file__))

# Dataset file name
DATASET_FILENAME = None

# Defines the default dataset file path
DATASET_DEFAULT_PATH = os.path.join(CURRENT_PATH, '..', 'data', 'raw', 'cleveland.data')

# Defines the default data output path
OUTPUT_DEFAULT_PATH = os.path.dirname(DATASET_DEFAULT_PATH)

# Defines the global constant for the default log level
LOG_DEFAULT_LEVEL = 'CRITICAL'

# Defines the list of unwanted collums
UNWANTED_COLLUM_INDEX_LIST = []

# Creates and adds the necessary arguments to the parser
def preconfigure_parser():

	# Create the Argument Parser
	parser = argparse.ArgumentParser(description='Simple Binary Classification Neural Network for the Cleveland Heart Disease Dataset')

	# Add the positional argument for the dataset input path
	parser.add_argument(
		'input',
		type=str,
		nargs='?',
		default=DATASET_DEFAULT_PATH,
		help='The input file path to the dataset (the dataset must be a comma-separated CSV file)'
	)

	# Add the positional argument for the output directory path
	parser.add_argument(
		'output',
		type=str,
		nargs='?',
		default=OUTPUT_DEFAULT_PATH,
		help='The output directory path (equal to the input path by default)'
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

	# Add the positional argument for the dataset read mode
	parser.add_argument(
		'--processing-mode',
		nargs='?',
		default='volatile',
		choices=[
			'inplace'
			'persistent',
			'volatile'
		],
		help='Set the dataset processing mode (default: volatile)'
	)

	# Returns the sucesfully created and configured parser
	return parser

# Configures the logger debugging level
def preconfigure_logger(args):

	# Maps the argument string to the log level data type
	log_level = getattr(logging, args.log)

	# Sets the log level in the default logger
	logging.basicConfig(level=log_level)

# Clears the dataset for processing
def cleanup_dataset(dataset, args):

	# Removes all unwanted columns from the dataset in inplace mode
	dataset.drop(dataset.columns[UNWANTED_COLLUM_INDEX_LIST], axis = 1, inplace = True)

	# Substitutes all the dataset non-numeric values with NaN
	dataset = dataset.apply(pandas.to_numeric, errors='coerce')

	# Removes all NaN values from the dataset in inplace mode
	dataset.dropna(inplace = True)

	# Returns the succesfully threated clean dataset
	return dataset

# Preformats the dataset for processing
def preformat_dataset(dataset, args):

	# Converts the target values to a binary form by setting output either to 0 or 1
	dataset.iloc[:, -1] = dataset.iloc[:, -1].apply(lambda x: 1 if x != 0 else 0)

	# Returns the succesfully preformatted dataset
	return dataset

# Main file execution guard
if __name__ == '__main__':

	# Creates and adds the necessary arguments to the parser
	parser = preconfigure_parser()

	# Parse the argument
	args = parser.parse_args()

	# Sets the value of DATASET_FILENAME as the openned dataset file name
	DATASET_FILENAME = os.path.basename(args.input)

	# Configures the logger debugging level
	preconfigure_logger(args)

	# Loads the dataset file in-memory without headers
	full_dataset = pandas.read_csv(args.input, header = None)

	# Clears the dataset (removes unwanted columns together with non-numeric, NaN and empty-valued lines)
	full_dataset = cleanup_dataset(full_dataset, args)

	# Preformats the dataset (converts the target values to a binary form: either 0 or 1)
	full_dataset = preformat_dataset(full_dataset, args)

	# If in inplace processing mode, overwrites the un-threated dataset with a pre-processed one, otherwise saves it in the output directory path
	if (args.processing_mode == 'inplace') :
		full_dataset.to_csv(args.input)
	elif (args.processing_mode == 'persistent'):
		full_dataset.to_csv(os.path.join(args.output, 'processed_' + DATASET_FILENAME))

	model = tensorflow.keras.Sequential(name='Testing...')

	# Separar Dataset

	# Treinar

	# Validar

	# Testar

	# Obter métricas

	# Prever

	# Salvar Resultados e Pesos

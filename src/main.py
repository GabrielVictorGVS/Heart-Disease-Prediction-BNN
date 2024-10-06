import os
import logging
import argparse
import pandas
import tensorflow
from matplotlib import pyplot
from sklearn.metrics import confusion_matrix
import seaborn

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

# Defines the prediction model learning rate (IDEAL 0.00007)
MODEL_LEARNING_RATE = 0.00002

# Defines the prediction model training epochs amount
MODEL_TRAINING_EPOCHS_AMOUNT = 3500

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

	# Add the positional argument for the dataset processing mode
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

# Plots the model training loss and metrics, and saves them only in 'persistent' or 'inplace' processing mode.
def plot_metrics(history, metrics, args):

	# If no metrics keys are provided, plot all metrics from the model history dictionary
	if metrics is None:
		metrics = [key for key in history.history.keys() if not key.startswith('val_')]

	# Iterates trought and plots all available model metrics
	for metric in metrics:

		# Defines the metric color palette
		metric_color_palette = seaborn.color_palette("rocket", n_colors=2)

		# Defines the graph figure
		pyplot.figure(figsize=(10, 5))

		# Defines the capitalized and properly spaced metric name
		metric_name = metric.capitalize().replace('_', ' ')

		# Plots the training metric
		pyplot.plot(history.history[metric], label = f'Training {metric_name}', color = metric_color_palette[0])

		# Plots the equivalent validation metric if it exists
		if f'val_{metric}' in history.history:
			pyplot.plot(history.history[f'val_{metric}'], label=f'Validation {metric_name}', color = metric_color_palette[1], linestyle = ':')

		# Defines the graph title as the metric name
		pyplot.title(f'Model {metric_name}')

		# Defines the Y-axis label as the metric name
		pyplot.ylabel(f'Model {metric_name}')

		# Defines the X-axis label as 'Epochs'
		pyplot.xlabel('Epochs')

		# Saves the metric graph if either in 'inplace' or 'persistent' mode
		if args.processing_mode in ['inplace', 'persistent']:
			# Creates a 'metrics' subfolder if non-existent
			if not os.path.exists(os.path.join(args.output, 'metrics')):
				os.makedirs(os.path.join(args.output, 'metrics'))
			# Removes the old metric graph, if it exists
			if os.path.exists(os.path.join(args.output, 'metrics', metric + '.png')):
				os.remove(os.path.join(args.output, 'metrics', metric + '.png'))
			# Saves the metric graph
			pyplot.savefig(os.path.join(args.output, 'metrics', metric + '.png'), dpi=300)

		# Enables the graph grid
		pyplot.grid(visible = True)

		# Enables the graph legend
		pyplot.legend()

		# Shows the graph figure on the screen
		pyplot.show()

		# Closes the plot to release memory resources
		pyplot.close()

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
		full_dataset.to_csv(args.input, header=False, index=False)
	elif (args.processing_mode == 'persistent') :
		full_dataset.to_csv(os.path.join(args.output, 'processed_' + DATASET_FILENAME), header=False, index=False)

	# Extracts a training dataset from 70% of the full dataset
	training_dataset = full_dataset.sample(frac=0.7)

	# Extracts the features from the training dataset
	training_feature_dataset = training_dataset.iloc[:, :-1]

	# Extracts the targets from the training dataset
	training_target_dataset = training_dataset.iloc[:, -1]

	# Extracts a testing dataset from 15% of the remaining full dataset ((1 - 0.7) * 0.5 = 0.15)
	testing_dataset = full_dataset.drop(training_dataset.index).sample(frac=0.5)

	# Extracts the features from the testing dataset
	testing_feature_dataset = testing_dataset.iloc[:, :-1]

	# Extracts the targets from the testing dataset
	testing_target_dataset = testing_dataset.iloc[:, -1]

	# Extracts a validation dataset from 15% of the remaining full dataset (1 - 0.7 - 0.15 = 0.15)
	validation_dataset = full_dataset.drop(training_dataset.index).drop(testing_dataset.index)

	# Extracts the features from the validation dataset
	validation_feature_dataset = validation_dataset.iloc[:, :-1]

	# Extracts the targets from the validation dataset
	validation_target_dataset = validation_dataset.iloc[:, -1]

	# Creates the main sequential keras model
	model = tensorflow.keras.Sequential(name='hearth-disease-binary-neural-network')

	# Adds the input layer to the model with the input dimension as same the as inputs amount, and a 'relu' activation, so to ignore negative values
	model.add(tensorflow.keras.layers.Dense(39, input_dim = len(training_feature_dataset.columns), activation = 'relu', name = 'input_dense'))

	# Adds a hidden dense layer to the model 13-0.41, 26-0.35, 43-0.35,
	model.add(tensorflow.keras.layers.Dense(29, activation = 'relu', name = 'hidden_dense_1'))

	# Adds the output layer to the model with a 'sigmoid' activation, so to limit the output between 0 and 1
	model.add(tensorflow.keras.layers.Dense(1, activation = 'sigmoid', name = 'output_dense'))

	# Defines the optimizer of the model
	optimizer = tensorflow.keras.optimizers.Adam(learning_rate = MODEL_LEARNING_RATE)

	# Compiles the model with a 'binary_crossentropy' loss and 'binary_accurary', 'precision' and 'recall' metrics
	model.compile(loss = 'binary_crossentropy', optimizer=optimizer, metrics = ['binary_accuracy', 'precision', 'recall'])

	# Prints the model summary to the terminal
	print(model.summary())

	# Trains the model with the training dataset
	history = model.fit(training_feature_dataset, training_target_dataset, epochs = MODEL_TRAINING_EPOCHS_AMOUNT, validation_data = (validation_feature_dataset, validation_target_dataset), verbose = 1)

	# Saves the entire model and weights after training if in 'inplace' or 'persistent' processing mode
	if (args.processing_mode in ['inplace', 'persistent']) :
		model.save(os.path.join(args.output, 'model', 'model.h5'))
		model.save_weights(os.path.join(args.output, 'model', 'weights.weights.h5'))

	# Plots all the model metrics, saves them if in 'inplace' or 'persistent' processing mode
	plot_metrics(history, None, args)

	# Tests the model prediction capability with unknown test values and converts the results to a binary format based on a 0.5 threshold
	predicted_targets = (model.predict(testing_feature_dataset) > 0.5).astype(int)

	# Generates a confusion matrix between the predicted and real targets from the testing dataset
	testing_confusion_matrix = confusion_matrix(testing_target_dataset, predicted_targets)

	# Defines the graph figure
	pyplot.figure(figsize=(10, 10))

	# Generates a heatmap based on the confusion matrix
	seaborn.heatmap(testing_confusion_matrix, annot=True, fmt='g', cmap='magma', cbar=True, xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])

	# Defines the graph title
	pyplot.title('Confusion Matrix for the Heart Disease Prediction Model')

	# Defines the Y-axis label
	pyplot.ylabel('Actual Diagnosis')

	# Defines the X-axis label
	pyplot.xlabel('Predicted Diagnosis')

	# Saves the metric graph if either in 'inplace' or 'persistent' mode
	if args.processing_mode in ['inplace', 'persistent']:
		# Removes the old metric graph, if it exists
		if os.path.exists(os.path.join(args.output, 'metrics', 'confusion_matrix' + '.png')):
			os.remove(os.path.join(args.output, 'metrics', 'confusion_matrix' + '.png'))
		# Saves the metric graph
		pyplot.savefig(os.path.join(args.output, 'metrics', 'confusion_matrix' + '.png'))

	# Shows the graph figure on the screen
	pyplot.show()

	# Closes the plot to release memory resources
	pyplot.close()

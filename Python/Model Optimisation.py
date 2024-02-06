# libraries
import pandas as pd
import matplotlib.pyplot as plt
from keras import Sequential # base model
from keras.layers import Dense, Normalization, InputLayer # layers
from keras.optimizers import SGD, RMSprop, Adam, AdamW, Adadelta, Adagrad, Adamax, Adafactor, Nadam, Ftrl # optimisers
from keras_tuner import HyperModel, Hyperband, BayesianOptimization # for hyperparameter tuning
from keras.callbacks import EarlyStopping, TerminateOnNaN # regularisation
from sklearn.metrics import mean_squared_error # evaluation metrics
from keras.backend import clear_session
from tensorflow_docs.modeling import EpochDots
from keras.models import load_model
from keras.utils import plot_model

# temp data
temp_X_train = pd.read_csv("data/temperature_train.csv").values[:,:-1]
temp_y_train = pd.read_csv("data/temperature_train.csv").values[:,-1]
temp_X_val = pd.read_csv("data/temperature_val.csv").values[:,:-1]
temp_y_val = pd.read_csv("data/temperature_val.csv").values[:,-1]
temp_X_test = pd.read_csv("data/temperature_test.csv").values[:,:-1]
temp_y_test = pd.read_csv("data/temperature_test.csv").values[:,-1]
temp_X_combined = pd.read_csv("data/temperature_combined.csv").values[:,:-1]
temp_y_combined = pd.read_csv("data/temperature_combined.csv").values[:,-1]

# pressure data
pressure_X_train = pd.read_csv("data/pressure_train.csv").values[:,:-1]
pressure_y_train = pd.read_csv("data/pressure_train.csv").values[:,-1]
pressure_X_val = pd.read_csv("data/pressure_val.csv").values[:,:-1]
pressure_y_val = pd.read_csv("data/pressure_val.csv").values[:,-1]
pressure_X_test = pd.read_csv("data/pressure_test.csv").values[:,:-1]
pressure_y_test = pd.read_csv("data/pressure_test.csv").values[:,-1]
pressure_X_combined = pd.read_csv("data/pressure_combined.csv").values[:,:-1]
pressure_y_combined = pd.read_csv("data/pressure_combined.csv").values[:,-1]

# hyperparameter search space
units_Int = (1, 100) # number of units in hidden layer
hidden_Int = (1, 3) # number of hidden layers
activation_Choice = ['relu', 'leaky_relu', 'elu', 'gelu', 'silu'] # activation function
optimizer_Choice = ['sgd', 'rmsprop', 'adam'] # optimiser
learning_rate_Log = (0.001, 0.1) # learning rate
batch_size_Int = (16, 64) # batch size

# model template
def BuildModelTemplate(X_train):
	class ModelTemplate(HyperModel):

		# utility function to build optimiser with custom learning rate
		def build_optimizer(self, hp):
			optimizer_string = hp.Choice('optimizer', values=optimizer_Choice)
			learning_rate = hp.Float('learning_rate', *learning_rate_Log, sampling='log')
			with hp.conditional_scope('optimizer', ['adam']):
				if optimizer_string == 'adam': optimzer = Adam(learning_rate)
			with hp.conditional_scope('optimizer', ['adamw']):
				if optimizer_string == 'adamw': optimzer = AdamW(learning_rate)
			with hp.conditional_scope('optimizer', ['rmsprop']):
				if optimizer_string == 'rmsprop': optimzer = RMSprop(learning_rate)
			with hp.conditional_scope('optimizer', ['sgd']):
				if optimizer_string == 'sgd': optimzer = SGD(learning_rate)
			with hp.conditional_scope('optimizer', ['adadelta']):
				if optimizer_string == 'adadelta': optimzer = Adadelta(learning_rate)
			with hp.conditional_scope('optimizer', ['adagrad']):
				if optimizer_string == 'adagrad': optimzer = Adagrad(learning_rate)
			with hp.conditional_scope('optimizer', ['adamax']):
				if optimizer_string == 'adamax': optimzer = Adamax(learning_rate)
			with hp.conditional_scope('optimizer', ['adafactor']):
				if optimizer_string == 'adafactor': optimzer = Adafactor(learning_rate)
			with hp.conditional_scope('optimizer', ['nadam']):
				if optimizer_string == 'nadam': optimzer = Nadam(learning_rate)
			with hp.conditional_scope('optimizer', ['ftrl']):
				if optimizer_string == 'ftrl': optimzer = Ftrl(learning_rate)
			return optimzer

		# model structure
		def build(self, hp):
			model = Sequential() # base
			model.add(InputLayer((6,))) # input

			# z-score normalisation
			NormLayer = Normalization() 
			NormLayer.adapt(X_train)
			model.add(NormLayer)

			# hidden layers
			for i in range(hp.Int('hidden', *hidden_Int)):
				with hp.conditional_scope('hidden', list(range(i+1, hidden_Int[1]+1))):
					model.add(Dense(
						units = hp.Int(f'units{i}', *units_Int), # units
						activation = hp.Choice(f'activation{i}', activation_Choice), # activation
					))
			
			model.add(Dense(1)) # output
			model.compile(
				optimizer = self.build_optimizer(hp), 
				loss = 'mse' # mean squared error loss function
			)
			return model

		def fit(self, hp, model, *args, **kwargs):
			batch_size = hp.Int('batch_size', *batch_size_Int)
			return model.fit(batch_size = batch_size, *args, **kwargs)
		
	return ModelTemplate()

# search

max_epochs = 1000 # for hyperband
factor = 3 # for hyperband
hyperband_iterations = 1 # for hyperband
max_trials = 2000 # for bayesian optimisation

directory = 'trials'

def HyperbandSearch(X_train, y_train, X_val, y_val, project_name):
	tuner = Hyperband(
		BuildModelTemplate(X_train),
		objective = 'val_loss',
		max_epochs = max_epochs,
		factor = factor,
		hyperband_iterations = hyperband_iterations,
		directory = directory,
		project_name = project_name,
		max_consecutive_failed_trials = 10
	)
	tuner.search(
		x = X_train,
		y = y_train,
		validation_data = (X_val, y_val),
		verbose = 2,
		callbacks = [EarlyStopping(monitor='val_loss', patience=10), TerminateOnNaN()]
	)
	return tuner

def BayesianOptimisationSearch(X_train, y_train, X_val, y_val, project_name):
	tuner = BayesianOptimization(
		BuildModelTemplate(X_train),
		objective = 'val_loss',
		max_trials = max_trials,
		directory = directory,
		project_name = project_name,
		max_consecutive_failed_trials = 10
	)
	tuner.search(
		x = X_train,
		y = y_train,
		validation_data = (X_val, y_val),
		epochs = 10,
		verbose = 2,
		callbacks = [EarlyStopping(monitor='val_loss', patience=10), TerminateOnNaN()],
	)
	return tuner

# search pressure models - Hyperband
pressure_tuner_hyperband = HyperbandSearch(pressure_X_train, pressure_y_train, pressure_X_val, pressure_y_val, 'pressure_hyperband')
temp_tuner_hyperband = HyperbandSearch(temp_X_train, temp_y_train, temp_X_val, temp_y_val, 'temp_hyperband')
pressure_tuner_bayesian = BayesianOptimisationSearch(pressure_X_train, pressure_y_train, pressure_X_val, pressure_y_val, 'pressure_bayesian')
temp_tuner_bayesian = BayesianOptimisationSearch(temp_X_train, temp_y_train, temp_X_val, temp_y_val, 'temp_bayesian')

# best N hyperparameters
N = 50
pressure_best_hps_hyperband = pressure_tuner_hyperband.get_best_hyperparameters(num_trials=N)
pressure_best_hps_bayesian = pressure_tuner_hyperband.get_best_hyperparameters(num_trials=N)
temp_best_hps_hyperband = temp_tuner_hyperband.get_best_hyperparameters(num_trials=N)
temp_best_hps_bayesian = temp_tuner_bayesian.get_best_hyperparameters(num_trials=N)

# model training
def train_models(X_train, y_train, X_val, y_val, tuner, hps):
    models = []
    for i, hp in enumerate(hps):
        try:
            clear_session()

            # load hyperparameters
            model = tuner.hypermodel.build(hp)
            batch_size = hp.get_config()['values']['batch_size']
            print(f'\nTraining Model {i}:')
            print(f'Batch Size: {batch_size}')

            # fit model
            model.fit(
                x = X_train,
                y = y_train,
                validation_data = (X_val, y_val),
                batch_size = batch_size,
                epochs = 1000,
                verbose = 0,
                callbacks = [
                    EpochDots(report_every=100), 
                    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
                ]
            )
            models.append(model)
        except: pass
    return models

pressure_models_hyperband = train_models(pressure_X_train, pressure_y_train, pressure_X_val, pressure_y_val, pressure_tuner_hyperband, pressure_best_hps_hyperband)
pressure_models_bayesian = train_models(pressure_X_train, pressure_y_train, pressure_X_val, pressure_y_val, pressure_tuner_bayesian, pressure_best_hps_bayesian)
temp_models_hyperband = train_models(temp_X_train, temp_y_train, temp_X_val, temp_y_val, temp_tuner_hyperband, pressure_best_hps_hyperband)
temp_models_bayesian = train_models(temp_X_train, temp_y_train, temp_X_val, temp_y_val, temp_tuner_bayesian, temp_best_hps_bayesian)

# rmse scores
# stored in pressure_results_hyperband, pressure_results_bayesian, temp_results_hyperband, temp_results_bayesian

# utility function for obtaining RMSE for train, validation, test and combined sets for a single model
def evaluate_model(X_train, y_train, X_val, y_val, X_test, y_test, X_combined, y_combined, model):
    rmse = lambda y, y_pred: mean_squared_error(y, y_pred, squared=False)
    return [
        rmse(y_train, model.predict(X_train)),
        rmse(y_val, model.predict(X_val)),
        rmse(y_test, model.predict(X_test)),
        rmse(y_combined, model.predict(X_combined))
    ]

# utility function for obtaining RMSE for train, validation, test and combined sets for a list of models
def evaluate_models(X_train, y_train, X_val, y_val, X_test, y_test, X_combined, y_combined, models):
    results = []
    for model in models:
        try: results.append(evaluate_model(X_train, y_train, X_val, y_val, X_test, y_test, X_combined, y_combined, model))
        except: pass
    return pd.DataFrame(results, columns=['rmse_train', 'rmse_val', 'rmse_test', 'rmse_combined'])

pressure_results_hyperband = evaluate_models(
    pressure_X_train, pressure_y_train,
    pressure_X_val, pressure_y_val,
    pressure_X_test, pressure_y_test,
    pressure_X_combined, pressure_y_combined,
    pressure_models_hyperband
)
pressure_results_bayesian = evaluate_models(
    pressure_X_train, pressure_y_train,
    pressure_X_val, pressure_y_val,
    pressure_X_test, pressure_y_test,
    pressure_X_combined, pressure_y_combined,
    pressure_models_bayesian
)
temp_results_hyperband = evaluate_models(
    temp_X_train, temp_y_train,
    temp_X_val, temp_y_val,
    temp_X_test, temp_y_test,
    temp_X_combined, temp_y_combined,
    temp_models_hyperband
)
temp_results_bayesian = evaluate_models(
    temp_X_train, temp_y_train,
    temp_X_val, temp_y_val,
    temp_X_test, temp_y_test,
    temp_X_combined, temp_y_combined,
    temp_models_bayesian
)

# save best model
best_pressure_model_hyperband = pressure_models_hyperband[pressure_results_hyperband['rmse_combined'].idxmin()]
best_pressure_model_bayesian = pressure_models_bayesian[pressure_results_bayesian['rmse_combined'].idxmin()]
best_temp_model_hyperband = temp_models_hyperband[temp_results_hyperband['rmse_combined'].idxmin()]
best_temp_model_bayesian = temp_models_bayesian[temp_results_bayesian['rmse_combined'].idxmin()]

# save models
best_pressure_model_hyperband.save(f'pressure_model_hyperband.keras')
best_pressure_model_bayesian.save(f'pressure_model_bayesian.keras')
best_temp_model_hyperband.save(f'temp_model_hyperband.keras')
best_temp_model_bayesian.save(f'temp_model_bayesian.keras')

# actual vs prediction plot
def actual_vs_pred_plot(X_test, y_test, X_combined, y_combined, model, x_label, y_label, title):
	plt.rcParams.update({'font.size': 12}) # font size 16

	# make predictions
	y_pred_test = model.predict(X_test)
	y_pred_combined = model.predict(X_combined)
	
	# plot
	_, ax = plt.subplots(figsize=(5,5))
	ideal_line_lim = max(max(y_combined), max(y_test))
	ax.plot([0, ideal_line_lim], [0, ideal_line_lim], 'k', linewidth=1, label='Ideal', zorder=-1) # ideal line (i.e. x=y)
	ax.scatter(y_combined, y_pred_combined, s=7, color=(0.00, 0.00, 0.55), label='Combined') # scatter plot for combined set
	ax.scatter(y_test, y_pred_test, s=7, color=(0.91, 0.41, 0.17), label='Test') # scatter plot for test set
	ax.set_title(title)
	ax.set_xlabel(x_label)
	ax.set_ylabel(y_label)
	plt.legend()
	plt.show()

actual_vs_pred_plot(pressure_X_test, pressure_y_test, pressure_X_combined, pressure_y_combined, best_pressure_model_hyperband,
					'True ΔP / Pa', 'Predicted ΔP / Pa', 'Pressure Model (Hyperband)')
actual_vs_pred_plot(pressure_X_test, pressure_y_test, pressure_X_combined, pressure_y_combined, best_pressure_model_bayesian,
					'True ΔP / Pa', 'Predicted ΔP / Pa', 'Pressure Model (Bayesian)')
actual_vs_pred_plot(temp_X_test, temp_y_test, temp_X_combined, temp_y_combined, best_temp_model_hyperband,
					'True $T$ / $^\circ\mathrm{C}}$', 'Predicted $T$ / $^\circ\mathrm{C}}$', 'Temperature Model (Hyperband)')
actual_vs_pred_plot(temp_X_test, temp_y_test, temp_X_combined, temp_y_combined, best_temp_model_bayesian,
					'True $T$ / $^\circ\mathrm{C}}$', 'Predicted $T$ / $^\circ\mathrm{C}}$', 'Temperature Model (Bayesian)')
from keras.models import load_model
import optuna
import numpy as np
from IPython.display import clear_output

# load models
temp_model = load_model('models/temp_model_hyperband.keras')
pressure_model = load_model('models/pressure_model_hyperband.keras')

def objective(trial):
    # range of values to try for each input parameter
    hc = trial.suggest_float('hc / mm', 0.6, 2.4)
    wc = trial.suggest_float('wc / mm', 0.2, 1.8)
    length = trial.suggest_float('length / mm', 12, 108)
    Tamb = trial.suggest_float('Tamb / degC', 25, 31.6)
    Q = trial.suggest_int('Q / Wm-2', 1272, 5040)
    Uin = trial.suggest_float('Uin / ms-1', 0, 20)

    # temperature and pressure prediction
    temp = temp_model.predict([hc, wc, length, Tamb, Q, Uin], verbose=0)[0, 0]
    pressure = pressure_model.predict([hc, wc, length, Tamb, Q, Uin], verbose=0)[0, 0]

    # constraints
    if 55 <= temp <= 60 and pressure >= 0:
        return pressure
    return np.Inf

sampler = optuna.samplers.TPESampler() # Tree-structured Parzen Estimator
study = optuna.create_study(direction = "minimize", sampler=sampler) # minimise objective function using sampler
study.optimize(objective, n_trials=1000) # 1000 trials

# pressure prediction based on optimised parameters
pressure = pressure_model.predict(list(study.best_params.values()), verbose=0)[0, 0]
print(f'Lowest Pressure: {pressure} Pa')

# temperature prediction based on optimised parameters
temp = temp_model.predict(list(study.best_params.values()), verbose=0)[0, 0]
print(f'Temperature: {temp} degC')

# optimised parameters
study.best_params
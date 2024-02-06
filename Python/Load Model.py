from keras.models import load_model
temp_model = load_model('models/temp_model_hyperband.keras')
pressure_model = load_model('models/pressure_model_hyperband.keras')
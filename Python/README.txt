===== PYTHON IMPLEMENTATION =====
Located in this folder are the Python files used during the project. In total, three aspects of the project were implemented in Python: (a) Model Optimisation, (b) PEMFC Input Parameters Optimisation, and (c) Manual Implementation of an Artificial Neural Network.

The .ipynb files are notebook files containing headers and descriptions written in markdown, while the .py files are the complete, uniterupted code. An additional file, 'Load Model.py', is provided as a framework for loading the optimised models. A 'Models' folder which contains the optimised models is also included.

Python Version: 11.6

NOTE: As of writing this, Python 12.0 is NOT compatible, due to TensorFlow's incompability with that version of Python.

Libraries:
- numpy 1.24.4: 		For numerical computation
- pandas 1.5.3:			For table manipulation using DataFrames
- matplotlib 3.6.0:		Graphing tool 
- keras 2.15.0:			For deep learning, specifically Sequential model construction
- tensorflow 2.15.0:	Backend deep learning engine (not directly used, but utilised by keras)
- keras_tuner 1.4.6: 	For optimisation of keras models
- scikit-learn 1.2.2: 	Generic machine learning library 
- optuna 3.4.0: 		Generic objective optimisation library

Upon installation of the above libaries, either using pip or conda, their dependencies will automatically be installed as well.

If using pip, the following command can be run to install the latest version of each libary:

>>> pip install numpy pandas matplotlib keras tensorflow keras_tuner scikit-learn optuna

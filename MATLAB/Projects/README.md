# Projects
This folder contains all the MATLAB files used to implement deep NN for temperature and pressure targets

## Brief description of code
1. **RLApp**: File used to format data to be used for Regression Learner App
2. **resultsTable .csv**: csv files containing results for test set and combined sets
3. **sess .mat**: Saved sessions for regression learner app

Remaining files are helper functions and data files

## Instructions
1. Open pressure_DL_v3 or temp_DL_v3 as a project in MATLAB
2. Open the experiment manager app
3. Open the pressure_DL_v3 or temp_DL_v3 project within the experiment manager app. Use the app to view previous results or run new trials
## To view the results/test a model
1. Export a model and its training history from the experiment manager app
2. Open the '''Output.mlx''' live script
3. Adjust the name of the model and training history in the MATLAB workspace to match the name used in the Output.mlx file
4. Change the minibatch size 

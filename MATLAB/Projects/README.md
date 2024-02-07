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
## To view the results of a model
1. Load the temp_DL.mat or pressure_DL.mat file into the MATLAB workspace (or export a new model from experiment manager app, make sure to rename the model and traininfo files)
2. Open the Output.mlx live script and change the MiniBatchSize variable to match the minibatch size used for the model
3. Run as a live script to view results

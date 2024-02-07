# Projects
This folder contains all the MATLAB files used to implement deep NN for temperature and pressure targets

## Brief description of code
1. **pressure_DL_v3** and **temp_DL_v3**: folders containing MATLAB projects for pressure and temperature deep learning experiments
2. **Pressure_Temp_v1.mat** and **DL_Temp_v1.mat**: Files used to configures experiment
3. **Output.mlx**: Show the training history and true vs pred plots of a model

Remaining files are helper functions, data files and files to obtain model metrics during training

## Instructions
1. Open pressure_DL_v3 or temp_DL_v3 as a project in MATLAB
2. Open the experiment manager app
3. Open the pressure_DL_v3 or temp_DL_v3 project within the experiment manager app. Use the app to view previous results or run new trials
## To view the results of a model
1. Load the temp_DL.mat or pressure_DL.mat file into the MATLAB workspace (or export a new model from experiment manager app, make sure to rename the model and traininfo files)
2. Open the Output.mlx live script and change the MiniBatchSize variable to match the minibatch size used for the model
3. Run as a live script to view results

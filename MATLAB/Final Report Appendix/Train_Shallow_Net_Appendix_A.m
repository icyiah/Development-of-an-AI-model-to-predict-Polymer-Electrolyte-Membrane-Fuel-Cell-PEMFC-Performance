%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%
%       SHALLOW NN TRAINING
%
%
% Training of pressure and temperature shallow NN models with Bayesopt.
% 
% NOTE: This MATLAB script was created by combining multiple MATLAB live
% scripts. The live script implementation can be found in the Github
% Repository.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Clean up workspace and old figures and number format for display %%
clear
clear all
close all
clc
format shorteng

%% Load experimental data %%
combined_set = readtable("pressure_combined.csv");      % pressure
x_pressure = combined_set{:,1:6}';
y_pressure = combined_set{:,7}';

combined_set = readtable("temperature_combined.csv");   % temperature
x_temp = combined_set{:,1:6}';
y_temp = combined_set{:,7}';

idxTrain = 1:3550;
idxTest = 3551:4176;

x_train_pressure = x_pressure(:, idxTrain);
y_train_pressure = y_pressure(:, idxTrain);
x_test_pressure = x_pressure(:, idxTest);
y_test_pressure = y_pressure(:, idxTest);
x_train_temp = x_temp(:, idxTrain);
y_train_temp = y_temp(:, idxTrain);
x_test_temp = x_temp(:, idxTest);
y_test_temp = y_temp(:, idxTest);
clear combined_set

% set colours
blue = [0.00 0.00 0.55];
red = [0.65 0.16 0.16];
green = [0.00 0.39 0.00];
pink = [1.00,0.00,1.00];
orange = [0.91,0.41,0.17];

%% Define the optimization parameters and ranges %%
optimVars = [
    optimizableVariable('layerSize', [1 100], 'Type', 'integer');
    optimizableVariable('activationFcn', {'poslin' 'logsig' 'tansig' 'elliotsig'}, 'Type', 'categorical')
];

% define Objective function
fun_pressure = @(T)objFcn(T, x_pressure, y_pressure, idxTrain, idxTest);
fun_temp = @(T)objFcn(T, x_temp, y_temp, idxTrain, idxTest);

% run bayesopt
results_pressure = bayesopt(fun_pressure,optimVars,...
   'MaxObjectiveEvaluations',2000,...
   'IsObjectiveDeterministic',false,...
   'PlotFcn', [],...
   'UseParallel',true);

results_temp = bayesopt(fun_temp,optimVars,...
   'MaxObjectiveEvaluations',2000,...
   'IsObjectiveDeterministic',false,...
   'PlotFcn', [],...
   'UseParallel',true);

%% Obtain best hyperparams %%
bestHyperparameters_pressure = bestPoint(results_pressure, 'Criterion','min-observed')
size_pressure = bestHyperparameters_pressure.layerSize;
aFcn_pressure = bestHyperparameters_pressure.activationFcn;

bestHyperparameters_temp = bestPoint(results_temp, 'Criterion','min-observed')
size_temp = bestHyperparameters_temp.layerSize;
aFcn_temp = bestHyperparameters_temp.activationFcn;

%% Train Best Model (Pressure) %%
rng('default');

% Create a neural network
net = fitnet(size_pressure, 'trainbr');
net.layers{1}.transferFcn = string(aFcn_pressure);

% Settings for neural network
net.input.processFcns = {'mapstd'};
net.performFcn = 'mse';
net.divideMode = 'sample';
net.trainParam.showWindow = 0;

% Divide data using train and test index
net.divideFcn = 'divideind';
net.divideParam.trainInd  = idxTrain;
net.divideParam.testInd = idxTest;

net.trainParam.showWindow = false;

% Train the network
net = configure(net, x_pressure, y_pressure);
[pressure_shallow_net, pressure_shallow_tr] = train(net,x_pressure,y_pressure);

%% Train Best Model (Temperature) %%
rng('default');

% Create a neural network
net = fitnet(size_temp, 'trainbr');
net.layers{1}.transferFcn = string(aFcn_temp);

% Settings for neural network
net.input.processFcns = {'mapstd'};
net.performFcn = 'mse';
net.divideMode = 'sample';
net.trainParam.showWindow = 0;

% Divide data using train and test index
net.divideFcn = 'divideind';
net.divideParam.trainInd  = idxTrain;
net.divideParam.testInd = idxTest;

net.trainParam.showWindow = false;

% Train the network
net = configure(net, x_temp, y_temp);
[temp_shallow_net, temp_shallow_tr] = train(net,x_temp,y_temp);

%% Evaluate Best Model (Pressure) %%
% Training History
epochs = pressure_shallow_tr.epoch;
training_RMSE = sqrt(pressure_shallow_tr.perf);
testing_RMSE = sqrt(pressure_shallow_tr.tperf);

figure(1)
hold on
plot(epochs,training_RMSE, 'Color', blue)
plot(epochs,testing_RMSE, 'Color', red);
hold off

v=get(1,'currentaxes');
xlabel("Epoch")
ylabel("RMSE/Pa")
% legend("Train", "Test")
set(v,'FontSize',14,'FontName','Times New Roman', 'YScale','log')
box on

% Results (Test Set)
y_test_pred_pressure = pressure_shallow_net(x_test_pressure);
test_RMSE_pressure = rmse(y_test_pressure,y_test_pred_pressure)
% Results (Combined Set)
y_comb_pred_pressure = pressure_shallow_net(x_pressure);
comb_RMSE_pressure = rmse(y_pressure, y_comb_pred_pressure)

%% True vs Predicted Plot (Pressure) %%
% obtain predictions for train set
y_train_pred = pressure_shallow_net(x_train_pressure);
% set axis
axs = [0 3000 0 3000];
line = [0 axs(2)];

figure(2)
hold on
% plot ideal line
plot(line, line, "Color", "k");
% plot points
plot(y_train_pressure,y_train_pred,'o','MarkerEdgeColor',blue,'MarkerFaceColor',blue,'MarkerSize',4);
plot(y_test_pressure,y_test_pred_pressure,'o','MarkerEdgeColor',orange,'MarkerFaceColor',orange,'MarkerSize',4);
hold off

v=get(2,'currentaxes');
xlabel('True\rm \Delta{\itP}/Pa')
ylabel('Predicted \Delta{\itP}/Pa')
% legend('Ideal', '', 'Location', 'nw');
set(v,'FontSize',14,'FontName','Times New Roman', 'XTick', get(v, 'YTick'), 'XTickLabelRotation', 0);
axis(axs)
box on
axis square

% Create textbox
annotation('textbox',...
    [0.087 0.888 0.049 0.0738],'String','a)',...
    'FontSize',14,...
    'FontName','Times New Roman',...
    'FitBoxToText','off',...
    'EdgeColor','none');

%% Evaluate Best Model (Temperature) %%
% Training History
epochs = temp_shallow_tr.epoch;
training_RMSE = sqrt(temp_shallow_tr.perf);
testing_RMSE = sqrt(temp_shallow_tr.tperf);

figure(3)
hold on
plot(epochs,training_RMSE, 'Color', blue)
plot(epochs,testing_RMSE, 'Color', red);
hold off

v=get(3,'currentaxes');
xlabel("Epoch")
ylabel("RMSE/{{\circ}}C")
% legend("Train", "Test")
set(v,'FontSize',14,'FontName','Times New Roman', 'YScale','log')
box on

% Results (Test Set)
y_test_pred_temp = temp_shallow_net(x_test_temp);
test_RMSE_temp = rmse(y_test_temp,y_test_pred_temp)
% Results (Combined Set)
y_comb_pred_temp = temp_shallow_net(x_temp);
comb_RMSE_temp = rmse(y_temp, y_comb_pred_temp)

%% True vs Predicted Plot (Temperature) %%
% obtain predictions for train set
y_train_pred = temp_shallow_net(x_train_temp);
% set axis
axs = [0 180 0 180];
line = [0 axs(2)];

figure(4)
hold on
% plot ideal line
plot(line, line, "Color", "k");
% plot points
plot(y_train_temp,y_train_pred,'o','MarkerEdgeColor',blue,'MarkerFaceColor',blue,'MarkerSize',4);
plot(y_test_temp,y_test_pred_temp,'o','MarkerEdgeColor',orange,'MarkerFaceColor',orange,'MarkerSize',4);
hold off

v=get(4,'currentaxes');
xlabel("True\rm {\itT}/{{\circ}}C") 
ylabel("Predicted {\itT}/{{\circ}}C")
% legend('Ideal', '', 'Location', 'nw');
set(v,'FontSize',14,'FontName','Times New Roman', 'XTick', get(v, 'YTick'), 'XTickLabelRotation', 0);
axis(axs)
box on
axis square

% Create textbox
annotation('textbox',...
    [0.105 0.888 0.049 0.0738],'String','b)',...
    'FontSize',14,...
    'FontName','Times New Roman',...
    'FitBoxToText','off',...
    'EdgeColor','none');

%% Other Plots (uncomment code to display)
%%% Pressure %%%
% plot(results_pressure)  % Bayesian optimisation Results
% view(pressure_shallow_net);
% figure, plotperform(pressure_shallow_tr);
% figure, plottrainstate(pressure_shallow_tr);
% 
% e = gsubtract(y_test_pressure,y_test_pred_pressure);    % test set error historgram
% figure, ploterrhist(e)                          
% figure, plotregression(y_test_pressure,y_test_pred_pressure)
% 
% e = gsubtract(y_pressure, y_comb_pred_pressure);        % combined set error historgram
% figure, ploterrhist(e)
% figure, plotregression(y_pressure,y_comb_pred_pressure)

%%% Temperature %%%
% plot(results_temp)      % Bayesian optimisation Results
% view(temp_shallow_net);
% figure, plotperform(temp_shallow_tr);
% figure, plottrainstate(temp_shallow_tr);
% 
% e = gsubtract(y_test_temp,y_test_pred_temp);    % test set error historgram
% figure, ploterrhist(e)                          
% figure, plotregression(y_test_temp,y_test_pred_temp)
% 
% e = gsubtract(y_temp, y_comb_pred_temp);        % combined set error historgram
% figure, ploterrhist(e)
% figure, plotregression(y_temp,y_comb_pred_temp)

%% Objective Function
function perf = objFcn(T, x, y, idxTrain, idxTest)
    rng('default');
    % Create a neural network
    net = fitnet(T.layerSize, 'trainbr');
    net.layers{1}.transferFcn = string(T.activationFcn);
    
    % Settings for neural network
    net.input.processFcns = {'mapstd'};
    net.performFcn = 'mse';
    net.divideMode = 'sample';
    net.trainParam.showWindow = 0;
    
    % Divide data using train and test index
    net.divideFcn = 'divideind';
    net.divideParam.trainInd  = idxTrain;
    net.divideParam.testInd = idxTest;

    net.trainParam.showWindow = false;

    % Train the network
    net = configure(net, x, y);
    [~,tr] = train(net,x, y);

    % optimise MSE
    perf = tr.best_tperf;
end
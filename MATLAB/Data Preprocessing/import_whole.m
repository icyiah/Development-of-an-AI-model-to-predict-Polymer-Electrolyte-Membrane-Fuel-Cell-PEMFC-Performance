function data = import_whole(filename, dataLines)
%IMPORTFILE Import data from a text file
%  DATA = IMPORTFILE(FILENAME) reads data from text file FILENAME for
%  the default selection.  Returns the data as a table.
%
%  DATA = IMPORTFILE(FILE, DATALINES) reads data for the specified row
%  interval(s) of text file FILENAME. Specify DATALINES as a positive
%  scalar integer or a N-by-2 array of positive scalar integers for
%  dis-contiguous row intervals.
%
%  Example:
%  data = importfile("C:\Users\isaia\OneDrive - Ngee Ann Polytechnic\Documents\MATLAB\Set 1.txt", [2, Inf]);
%
%  See also READTABLE.
%
% Auto-generated by MATLAB on 02-Nov-2023 13:03:46

%% Input handling

% If dataLines is not specified, define defaults
if nargin < 2
    dataLines = [2, Inf];
end

%% Set up the Import Options and import the data
opts = delimitedTextImportOptions("NumVariables", 19);

% Specify range and delimiter
opts.DataLines = dataLines;
opts.Delimiter = "\t";

% Specify column names and types
opts.VariableNames = ["hc", "wc", "length", "Tamb", "Q", "Uin", "TemperaturedegCBoundaryProbe1", "TemperaturedegCBackAirTemperatureProbe2", "TemperaturedegCBackSolidTemperatureProbe3", "TemperaturedegCFrontAllTemperatureProbe4", "TemperaturedegCFrontAirTemperatureProbe5", "TemperaturedegCFrontSolidTemperatureProbe6", "PressurePaPressureProbe1", "PressurePaPressureProbe2", "PressurePaPressureProbe3", "VelocityMagnitudemsSpeedProbe1", "VelocityMagnitudemsSpeedProbe2", "TemperaturedegCStackTemperatureProbe1", "TemperaturedegCStackTemperatureProbe"];
opts.VariableTypes = ["double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double"];

% Specify file level properties
opts.ExtraColumnsRule = "ignore";
opts.EmptyLineRule = "read";

% Import the data
data = readtable(filename, opts);

end
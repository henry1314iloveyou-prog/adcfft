function build_dsm_decim_model(bits, fs, fc)
%BUILD_DSM_DECIM_MODEL  Build & run a Simulink decimation-filter model.
%
%   build_dsm_decim_model(bits, fs, fc)   e.g. (bits, 12e6, 10e3)
%
%   Model:  From Workspace -> CIC FIR -> Downsample(125) -> LP FIR -> Downsample(2) -> To Workspace
%   Output variable 'decimOut' (timeseries) is left in the base workspace.
%   Needs Simulink, DSP System Toolbox (Downsample) and Signal Processing Toolbox.
%   Block library paths were written without a MATLAB install to test them;
%   if one fails to resolve, open the library browser and adjust the path.

x = double(bits(:));
if all(x == 0 | x == 1), x = 2*x - 1; end
Ts = 1/fs;
assignin('base', 'bitsIn', timeseries(x, (0:numel(x)-1)'*Ts));

[~, fsOut, h] = dsm_decimate(x(1:1000), fs, fc);   % just to get coefficients
assignin('base', 'cicCoef', h.cic.');
assignin('base', 'lpCoef',  h.lp.');

mdl = 'dsm_decim';
if bdIsLoaded(mdl), close_system(mdl, 0); end
new_system(mdl); open_system(mdl);

add_block('simulink/Sources/From Workspace', [mdl '/Bits'], ...
    'VariableName', 'bitsIn', 'SampleTime', num2str(Ts), ...
    'OutputAfterFinalValue', 'Setting to zero', 'Position', [30 40 110 70]);
add_block('simulink/Discrete/Discrete FIR Filter', [mdl '/CIC (order 4, R=125)'], ...
    'Coefficients', 'cicCoef', 'SampleTime', num2str(Ts), 'Position', [150 40 260 70]);
add_block('dspsigops/Downsample', [mdl '/Down 125'], ...
    'N', '125', 'Position', [300 40 370 70]);
add_block('simulink/Discrete/Discrete FIR Filter', [mdl '/LPF fc=10k'], ...
    'Coefficients', 'lpCoef', 'SampleTime', num2str(125*Ts), 'Position', [410 40 520 70]);
add_block('dspsigops/Downsample', [mdl '/Down 2'], ...
    'N', '2', 'Position', [560 40 630 70]);
add_block('simulink/Sinks/To Workspace', [mdl '/Out'], ...
    'VariableName', 'decimOut', 'SaveFormat', 'Timeseries', 'Position', [670 40 750 70]);

names = {'Bits','CIC (order 4, R=125)','Down 125','LPF fc=10k','Down 2','Out'};
for k = 1:numel(names)-1
    add_line(mdl, [names{k} '/1'], [names{k+1} '/1']);
end
for b = {'Down 125','Down 2'}
    try, set_param([mdl '/' b{1}], 'RateOptions', 'Allow multirate'); catch, end
end

set_param(mdl, 'StopTime', num2str((numel(x)-1)*Ts), 'SolverType', 'Fixed-step', ...
    'Solver', 'FixedStepDiscrete', 'FixedStep', num2str(Ts));
save_system(mdl);
sim(mdl);
fprintf('Simulink output: %d samples at fs = %g Hz\n', numel(decimOut.Data), fsOut);
end

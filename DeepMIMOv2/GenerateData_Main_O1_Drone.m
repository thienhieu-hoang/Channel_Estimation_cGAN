% Generate channel between BSs and drones

% ----------------- Add the path of DeepMIMO function --------------------%
addpath('DeepMIMO_functions')

% -------------------- DeepMIMO Dataset Generation -----------------------%
% Load Dataset Parameters
dataset_params = read_params('parameters.m');
% run parameters.m

%% Settings 
bs_ant = dataset_params.num_ant_BS(1) * dataset_params.num_ant_BS(2) *dataset_params.num_ant_BS(3); % M = 1x4x1 BS Antennas
subs = dataset_params.OFDM_limit; % subcarriers
pilot_l = 16; % 8; % Pilots length is 8
snr  = 0; % SNR = 0 dB

filename = ['Outdoor1_60_',num2str(bs_ant),'ant_',num2str(subs),'subcs_',num2str(pilot_l),'pilot'];

%% Generate channel dataset H                            
% dataset_params.saveDataset = 1;
 
% -------------------------- Dataset Generation -----------------%
[DeepMIMO_dataset, dataset_params] = DeepMIMO_generator(dataset_params); % Get H (i.e.,DeepMIMO_dataset )
    % Output: 
    %   DeepMIMO_dataset == 1x1 cell == 1 x (no active BSs) cell
    %       DeepMIMO_dataset{1} == 1x1 struct 
    %           DeepMIMO_dataset{1}.user == 1x noUE cell (total number of UEs = 2211)
    %               each cell: DeepMIMO_dataset{1}.user{1} == 1x1 struct, with attributes: 
    %                   DeepMIMO_dataset{1}.user{1}.channel == (M_UE antens) x (M_BS antens) x subcs
    %                   DeepMIMO_dataset{1}.user{1}.loc     == 1x3 double
    %   dataset_params : add 2 attributes: 
    %       dataset_params.num_BS   ()
    %       dataset_params.num_user ()




%% Generate Pilots 
pilot = uniformPilotsGen(pilot_l);
pilot = pilot{1,1};
pilot_user = repmat(pilot,[1 subs])'; % subc x pilot


%% Genrate Quantized Siganl Y with Noise
channels = zeros(length(DeepMIMO_dataset{1}.user),bs_ant,subs);       % noUE x M_BS x subc 
Y        = zeros(length(DeepMIMO_dataset{1}.user),bs_ant,pilot_l);    % noUE x M_BS x pilot
Y_noise  = zeros(length(DeepMIMO_dataset{1}.user),bs_ant,pilot_l);    % noUE x M_BS x pilot
Y_sign   = zeros(length(DeepMIMO_dataset{1}.user),bs_ant,pilot_l,2);  % noUE x M_BS x pilot x 2

for i = 1:length(DeepMIMO_dataset{1}.user)
    channels(i,:,:) = normalize(DeepMIMO_dataset{1}.user{i}.channel,'scale');   % M_BS x subc
           Y(i,:,:) = squeeze(DeepMIMO_dataset{1}.user{i}.channel) *pilot_user; % M_BS x subc * subc x pilot == M_BS x pilot
     Y_noise(i,:,:) = awgn(Y(i,:,:),snr,'measured');                            % M_BS x pilot
end


%% Convert complex data to two-channel data
Y_sign(:,:,:,1) = sign(real(Y_noise)); % real part of Y
Y_sign(:,:,:,2) = sign(imag(Y_noise)); % imag papt of Y


channels_r(:,:,:,1) = real(channels); % real part of H
channels_r(:,:,:,2) = imag(channels); % imag part of H

% Shuffle data 
shuff      = randi([1,length(DeepMIMO_dataset{1}.user)],length(DeepMIMO_dataset{1}.user),1);
Y_sign     = Y_sign(shuff,:,:,:);
channels_r = channels_r(shuff,:,:,:);


%% Split data for training
numOfSamples = length(DeepMIMO_dataset{1}.user);
trRatio = 0.7;
numTrSamples = floor( trRatio*numOfSamples);
numValSamples = numOfSamples - numTrSamples;

input_da = Y_sign(1:numTrSamples,:,:,:);
output_da = channels_r(1:numTrSamples,:,:,:);

input_da_test = Y_sign(numTrSamples+1:end,:,:,:);
output_da_test = channels_r(numTrSamples+1:end,:,:,:);


%% Visualization of Y and H
figure
imshow(squeeze(input_da(1,:,:,1)))
title('Visualization of Y')
figure
imshow(squeeze(output_da(1,:,:,1)))
title('Visualization of H')

%% Save data
save(['Gan_Data/64ant_32sub_16pilot_548_560/Gan_',num2str(snr),'_dB',filename],'input_da','output_da','input_da_test','output_da_test', "dataset_params",'-v7.3');




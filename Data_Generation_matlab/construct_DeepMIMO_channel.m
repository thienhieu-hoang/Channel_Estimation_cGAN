% --------- DeepMIMO: A Generic Dataset for mmWave and massive MIMO ------%
% Author: Ahmed Alkhateeb
% Date: Sept. 5, 2018 
% Goal: Encouraging research on ML/DL for mmWave MIMO applications and
% providing a benchmarking tool for the developed algorithms
% ---------------------------------------------------------------------- %

% Output:
%   channel == 64 x 32 complex
function [channel]=construct_DeepMIMO_channel(params,num_ant_x,num_ant_y,num_ant_z,BW,...
    ofdm_num_subcarriers,output_subcarrier_downsampling_factor,output_subcarrier_limit,antenna_spacing_wavelength_ratio)

kd       = 2*pi*antenna_spacing_wavelength_ratio;
ang_conv = pi/180; % convert from degree to radian
Ts       = 1/BW;

Mx_Ind  = 0:1:num_ant_x-1; % 1 x num_ant_x (1x 1)
My_Ind  = 0:1:num_ant_y-1; % 1 x num_ant_y (1x64) 
Mz_Ind  = 0:1:num_ant_z-1; % 1 x num_ant_z (1x 1)
Mxx_Ind = repmat(Mx_Ind,1,num_ant_y*num_ant_z)'; 
    % 64 x 1 = [1 x num_ant_x*num_ant_y*num_ant_z]'
    %        = [num_ant_x*num_ant_y*num_ant_z x 1] 
Myy_Ind = repmat(reshape(repmat(My_Ind,num_ant_x,1),1,num_ant_x*num_ant_y),1,num_ant_z)';
    % 64 x 1 = repmat(reshape([num_ant_x x num_ant_y], 1, num_ant_x*num_ant_y), 1, num_ant_z)'
    %        = repmat([1 x num_ant_x*num_ant_y], 1, num_ant_z)'
    %        = [1 x num_ant_x*num_ant_y*num_ant_z]'
    %        = [num_ant_x*num_ant_y*num_ant_z x 1]
Mzz_Ind = reshape(repmat(Mz_Ind,num_ant_x*num_ant_y,1),1,num_ant_x*num_ant_y*num_ant_z)';
    % 64 x 1 = reshape([num_ant_x*num_ant_y x num_ant_z],1,num_ant_x*num_ant_y*num_ant_z)'
    %        = [1 x num_ant_x*num_ant_y*num_ant_z]'
    %        = [num_ant_x*num_ant_y*num_ant_z x 1]
M       = num_ant_x*num_ant_y*num_ant_z;  % 64 (total antennas?)
 
k = 0:output_subcarrier_downsampling_factor:output_subcarrier_limit-1; % 1 x 32 (subcarrier)
num_sampled_subcarriers = length(k); 
channel = zeros(M,num_sampled_subcarriers); % 64 x 32

for l = 1:1:params.num_paths
    gamma_x = 1j*kd*sin(params.DoD_theta(l)*ang_conv)*cos(params.DoD_phi(l)*ang_conv);
    gamma_y = 1j*kd*sin(params.DoD_theta(l)*ang_conv)*sin(params.DoD_phi(l)*ang_conv);
    gamma_z = 1j*kd*cos(params.DoD_theta(l)*ang_conv);
    gamma_comb = Mxx_Ind*gamma_x+Myy_Ind*gamma_y + Mzz_Ind*gamma_z; % 64x1 complex == num_ant_x*num_ant_y*num_ant_z x 1
    array_response   = exp(gamma_comb);
    delay_normalized = params.ToA(l)/Ts;
    channel = channel+array_response*sqrt(params.power(l)/ofdm_num_subcarriers)*exp(1j*params.phase(l)*ang_conv)*exp(-1j*2*pi*(k/ofdm_num_subcarriers)*delay_normalized);     
end 

end
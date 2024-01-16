% --------- DeepMIMO: A Generic Dataset for mmWave and massive MIMO ------%
% Author: Ahmed Alkhateeb
% Date: Sept. 5, 2018 
% Goal: Encouraging research on ML/DL for mmWave MIMO applications and
% providing a benchmarking tool for the developed algorithms
% ---------------------------------------------------------------------- %

% Output: 
%   DeepMIMO_dataset == 1x1 cell
%       DeepMIMO_dataset{1,1} == 1x1 struct 
%           DeepMIMO_dataset{1,1}.user == 1x2211 cell (total number of UEs = 2211)
%               each cell: DeepMIMO_dataset{1,1}.user{1,1} == 1x1 struct, with attributes: 
%                   DeepMIMO_dataset{1,1}.user{1,1}.channel == 64x32 complex double
%                   DeepMIMO_dataset{1,1}.user{1,1}.loc     == 1x3 double
%   params : add 2 attributes: 
%       params.num_BS   (=64)
%       params.num_user (=2211)
function [DeepMIMO_dataset, params]=DeepMIMO_generator(params)


% -------------------------- DeepMIMO Dataset Generation -----------------%
fprintf(' DeepMIMO Dataset Generation started \n')

% Read scenario parameters
file_scenario_params = strcat('./RayTracing Scenarios/',params.scenario,'/',params.scenario,'.params.mat');
    % file_scenario_params = './RayTracing Scenarios/I1_2p4/I1_2p4.params.mat'
load(file_scenario_params)
    % load .mat file, including 
    %   num_BS = 64 (in data file, it is 64 antennas)
    %   user_grids = [  1 401 201;
    %                 402 502 701]

params.num_BS = num_BS;

num_rows = max(min(user_grids(:,2), params.active_user_last) - max(user_grids(:,1),params.active_user_first)+1,0);
    % num_row = [11; 0]
params.num_user=sum(num_rows.*user_grids(:,3));                     % total number of users
    % total number of users = 2211 = 11*201 + 0
    % it is just: giving the first and the last row of the active user, calculate the total number of UEs
 
current_grid = min(find(max(params.active_user_first,user_grids(:,2))==user_grids(:,2)));
user_first=sum((max(min(params.active_user_first,user_grids(:,2))-user_grids(:,1)+1,0)).*user_grids(:,3))-user_grids(current_grid,3)+1;
user_last=user_first+params.num_user-1;
% calculate the index of the first and the last UE of the user set
 
BW = params.bandwidth*1e9;                                     % Bandwidth in Hz
 
% Reading ray tracing data
fprintf(' Reading the channel parameters of the ray-tracing scenario %s', params.scenario)
count_done=0;
reverseStr=0;
percentDone = 100 * count_done / length(params.active_BS);
msg = sprintf('- Percent done: %3.1f', percentDone); %Don't forget this semicolon
fprintf([reverseStr, msg]);
reverseStr = repmat(sprintf('\b'), 1, length(msg));
    
for t=1:1:params.num_BS
    if sum(t == params.active_BS) ==1  % if t == active_BS (=32)
        filename_DoD=strcat('./RayTracing Scenarios/',params.scenario,'/',params.scenario,'.',int2str(t),'.DoD.mat');
        filename_CIR=strcat('./RayTracing Scenarios/',params.scenario,'/',params.scenario,'.',int2str(t),'.CIR.mat');
        filename_Loc=strcat('./RayTracing Scenarios/',params.scenario,'/',params.scenario,'.Loc.mat');
        [TX{t}.channel_params]=read_raytracing(filename_DoD,filename_CIR,filename_Loc,params.num_paths,user_first,user_last); 
 
        count_done=count_done+1;
        percentDone = 100 * count_done / length(params.active_BS);
        msg = sprintf('- Percent done: %3.1f', percentDone); %Don't forget this semicolon
        fprintf([reverseStr, msg]);
        reverseStr = repmat(sprintf('\b'), 1, length(msg));
    end
end
    % TX = 1 x params.active_BS cell (== 1x 32 cell)
    %   TX{t}  = []
    %   TX{32} = 1x1 struct
    %       TX{32}.channel_params == 1 x 2211 (number of UEs) struct 
    %           with fields:
    %               DoD_phi
    %               DoD_theta
    %               phase
    %               ToA
    %               power
    %               num_paths
    %               loc
    %           TX{32}.channel_params(1) == 1x1 struct


% Constructing the channel matrices 
TX_count=0;
for t=1:1:params.num_BS % 1:64
    if sum(t == params.active_BS) ==1   % if t == active_BS (=32)
        fprintf('\n Constructing the DeepMIMO Dataset for BS %d', t)
        reverseStr=0;
        percentDone = 0;
        msg = sprintf('- Percent done: %3.1f', percentDone); %Don't forget this semicolon
        fprintf([reverseStr, msg]);
        reverseStr = repmat(sprintf('\b'), 1, length(msg));
        TX_count=TX_count+1;
        for user=1:1:params.num_user         
          [DeepMIMO_dataset{TX_count}.user{user}.channel]=construct_DeepMIMO_channel(TX{t}.channel_params(user),params.num_ant_x,params.num_ant_y,params.num_ant_z, ...
              BW,params.num_OFDM,params.OFDM_sampling_factor,params.OFDM_limit,params.ant_spacing);
          DeepMIMO_dataset{TX_count}.user{user}.loc=TX{t}.channel_params(user).loc;
          
          percentDone = 100* round(user / params.num_user,2);
          msg = sprintf('- Percent done: %3.1f', round(percentDone,2)); %Don't forget this semicolon
          fprintf([reverseStr, msg]);
          reverseStr = repmat(sprintf('\b'), 1, length(msg));
        end       
    end   
end

if params.saveDataset==1
    sfile_DeepMIMO=strcat('./DeepMIMO Dataset/DeepMIMO_dataset.mat');
    save(sfile_DeepMIMO,'DeepMIMO_dataset','-v7.3');
end

fprintf('\n DeepMIMO Dataset Generation completed \n')
% not done
addpath("DeepMIMO_functions\")

filename_DoD=strcat('./RayTracing Scenarios/I1_2p4/I1_2p4','.',int2str(55),'.DoD.mat');
filename_CIR=strcat('./RayTracing Scenarios/I1_2p4/I1_2p4','.',int2str(55),'.CIR.mat');
filename_Loc=strcat('./RayTracing Scenarios/I1_2p4/I1_2p4','.Loc.mat');

[a]=read_raytracing(filename_DoD,filename_CIR,filename_Loc,1,1,1000); 

figure;
% 3-D
for i = 1:length(a)
    plot3(a(i).loc(1), a(i).loc(2), a(i).loc(3), '.');
    hold on
end

figure
% 2-D
for i = 1:length(a)
    plot(a(i).loc(1), a(i).loc(2), '.');
    hold on
end
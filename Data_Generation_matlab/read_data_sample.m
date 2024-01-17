filename_DoD=strcat('./RayTracing Scenarios/I1_2p4/I1_2p4','.',int2str(55),'.DoD.mat');
filename_CIR=strcat('./RayTracing Scenarios/I1_2p4/I1_2p4','.',int2str(55),'.CIR.mat');
filename_Loc=strcat('./RayTracing Scenarios/I1_2p4/I1_2p4','.Loc.mat');
[a]=read_raytracing(filename_DoD,filename_CIR,filename_Loc,1,1,1); 
 
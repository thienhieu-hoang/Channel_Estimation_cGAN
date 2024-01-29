% run DeepMIMO_Dataset_Generator.m
figure;
% 3-D
for i = 1:length(DeepMIMO_dataset{1}.user)
    plot3(DeepMIMO_dataset{1}.user{i}.loc(1), DeepMIMO_dataset{1}.user{i}.loc(2), DeepMIMO_dataset{1}.user{i}.loc(3), '.');
    hold on
end
for i = 1:length(DeepMIMO_dataset{1}.basestation)
    plot3(DeepMIMO_dataset{1}.basestation{i}.loc(1), DeepMIMO_dataset{1}.basestation{i}.loc(2), DeepMIMO_dataset{1}.basestation{i}.loc(3), '*');
    hold on
end

figure
% 2-D
for i = 1:length(DeepMIMO_dataset{1}.user)
    plot(DeepMIMO_dataset{1}.user{i}.loc(1), DeepMIMO_dataset{1}.user{i}.loc(2), '.');
    hold on
end
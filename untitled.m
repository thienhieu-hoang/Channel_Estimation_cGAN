% mean_val = mean(DeepMIMO_dataset{1}.user{1}.channel{1}(:));
% std_dev = std(DeepMIMO_dataset{1}.user{1}.channel{1}(:));
% normalized_matrix = (DeepMIMO_dataset{1}.user{1}.channel{1} - mean_val) / std_dev;
% 
% figure(); imagesc(abs(DeepMIMO_dataset{1}.user{1}.channel{1}))
% figure(); imagesc(real(DeepMIMO_dataset{1}.user{1}.channel{1}))
% figure(); imagesc(imag(DeepMIMO_dataset{1}.user{1}.channel{1}))
% 
% cmax = max(abs(DeepMIMO_dataset{1}.user{1}.channel{1}(:)));

Y = squeeze(input_da(1,1,:,:,1))' + 1i* squeeze(input_da(1,2,:,:,1))';
H = squeeze(output_da(1,1,:,:,1))' + 1i* squeeze(output_da(1,2,:,:,1))';

% H1 = squeeze(channel_data(1,1,:,:,1))' + j*squeeze(channel_data(1,2,:,:,1))';
% figure(); 
% imagesc(abs(H1));
% title('Visualization of abs(H1)')

figure
imshow(abs(pilot_tx))
title('Visualization of X')

figure(); 
imagesc(abs(pilot_tx));
title('Visualization of abs(X)')

figure
imshow(abs(Y))
title('Visualization of Y')

figure(); 
imagesc(abs(Y));
title('Visualization of abs(Y)')

figure(); 
imagesc(real(Y));
title('Visualization of real(Y)')

figure(); 
imagesc(imag(Y));
title('Visualization of imag(Y)')

figure
imshow(abs(H))
title('Visualization of H')

figure(); 
imagesc(abs(H));
title('Visualization of abs(H)')

figure(); 
imagesc(real(H));
title('Visualization of real(H)')

figure(); 
imagesc(imag(H));
title('Visualization of imag(H)')
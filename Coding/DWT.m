% Step 1: Load and resize the host image to 512x512
hostImage = imread('test1.jpeg'); % Replace with your image path
hostImage = imresize(hostImage, [512, 512]);

% Step 2: Prepare the watermark (binary image or a sequence of bits)
% Assuming watermark is initially a color image, convert it to grayscale
watermark = imread('wtrmrk.jpg'); % Replace with your watermark image path
watermark = imbinarize(rgb2gray(watermark), 0.5); % Convert color watermark to grayscale and binarize
    
% Step 3: Apply Discrete Wavelet Transform (DWT) on the host image
alpha = 0.015; % Adjust this value to control the watermark strength

[cA, cH, cV, cD] = dwt2(hostImage, 'haar'); % Perform DWT on the host image

% Step 4: Resize the watermark to match the LL subband size
watermark = imresize(watermark, [size(cA, 1), size(cA, 2)]);

% Step 5: Embed the watermark into the approximation coefficient (cA)
watermarked_cA = cA;
watermarked_cA(watermark == 1) = cA(watermark == 1) + alpha * cA(watermark == 1);
watermarked_cA(watermark == 0) = cA(watermark == 0) - alpha * cA(watermark == 0);

% Step 6: Apply the inverse DWT to obtain the watermarked image
watermarkedImage = idwt2(watermarked_cA, cH, cV, cD, 'haar');

% Display the watermarked image
% figure;
subplot(1, 2, 1);
imshow(hostImage);
title('Original Host Image');

subplot(1, 2, 2);
imshow(uint8(watermarkedImage));
title('Watermarked Image');

% Convert the watermarkedImage to uint8 (same data type as hostImage)
watermarkedImage = uint8(watermarkedImage);

% ... (rest of the code remains the same)
% Calculate MSE (Mean Squared Error)
mse = sum(sum((double(hostImage) - double(watermarkedImage)).^2)) / (512*512);

% Calculate PSNR (Peak Signal-to-Noise Ratio) in dB
psnr = 10 * log10(255^2 / mse);

% Calculate SSIM (Structural Similarity Index)
ssim_val = ssim(hostImage, watermarkedImage);

% Calculate UACI (Underwater Image Quality Measure)
uaci = sum(sum(abs(double(hostImage) - double(watermarkedImage)))) / (512*512*255);

% Calculate NPCR (Number of Pixel Change Rate) and CC (Correlation Coefficient)
[rows, cols] = size(hostImage);
npcr = sum(sum(hostImage ~= watermarkedImage)) / (rows * cols) * 100;

cc = corrcoef(double(hostImage(:)), double(watermarkedImage(:)));
cc = cc(1, 2); % Extract the correlation coefficient value from the matrix

% Calculate NCC (Normalized Cross-Correlation)
ncc = sum(sum(hostImage .* watermarkedImage)) / sqrt(sum(sum(hostImage .^ 2)) * sum(sum(watermarkedImage .^ 2)));

% Calculate entropy of the watermarked image
entropy_val = entropy(watermarkedImage);
entropy_val1 = entropy(hostImage);

% Display metrics
disp(['MSE: ', num2str(mse)]);
disp(['PSNR: ', num2str(psnr), ' dB']);
disp(['SSIM: ', num2str(ssim_val)]);
disp(['UACI: ', num2str(uaci)]);
disp(['NPCR: ', num2str(npcr), '%']);
disp(['CC: ', num2str(cc)]);
disp(['NCC: ', num2str(ncc)]);
disp(['Entropy Ori: ', num2str(entropy_val1)]);
disp(['Entropy: ', num2str(entropy_val)]);

% Display histograms
figure;
subplot(1, 2, 1);
imhist(hostImage);
title('Original Host Image Histogram');

subplot(1, 2, 2);
imhist(watermarkedImage);
title('Watermarked Image Histogram');

% subplot(1, 3, 3);
% imhist(hostImage - watermarkedImage);
% title('Difference Histogram');

% Save the histograms
% saveas(gcf, 'path_to_save_histograms.jpg');

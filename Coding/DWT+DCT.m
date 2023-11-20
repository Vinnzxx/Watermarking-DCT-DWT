% Step 1: Load and resize the host image to 512x512
hostImage = imread('test1.jpeg'); % Replace with your image path
hostImage = imresize(hostImage, [512, 512]);

% Step 2: Prepare the watermark (binary image or a sequence of bits)
% Assuming watermark is initially a color image, convert it to grayscale
watermark = imread('wtrmrk.jpg'); % Replace with your watermark image path
watermark = rgb2gray(watermark); % Convert color watermark to grayscale
watermark = imbinarize(watermark, 0.5); % Binarize the grayscale watermark

% Step 3: Apply DWT on the host image
alpha_dwt = 0.015; % Adjust this value to control the DWT watermark strength

[cA, cH, cV, cD] = dwt2(hostImage, 'haar'); % Perform DWT on the host image

% Step 4: Resize the watermark to match the LL subband size
watermark_dwt = imresize(watermark, [size(cA, 1), size(cA, 2)]);

% Step 5: Embed the watermark into the approximation coefficient (cA) using DWT
watermarked_cA = cA;
watermarked_cA(watermark_dwt == 1) = cA(watermark_dwt == 1) + alpha_dwt * cA(watermark_dwt == 1);
watermarked_cA(watermark_dwt == 0) = cA(watermark_dwt == 0) - alpha_dwt * cA(watermark_dwt == 0);

% Step 6: Apply inverse DWT to obtain the watermarked image after DWT
watermarkedImage_dwt = idwt2(watermarked_cA, cH, cV, cD, 'haar');

% Step 7: Prepare the watermarkedImage_dwt to be used for DCT watermarking
watermarkedImage_dct = watermarkedImage_dwt;

% Step 8: Divide the host image into 8x8 blocks for DCT watermarking
blockSize_dct = 8;
[height, width] = size(watermarkedImage_dct);
numBlocksH_dct = height / blockSize_dct;
numBlocksW_dct = width / blockSize_dct;

% Step 9: Apply DCT on each block of watermarkedImage_dct and embed the watermark
alpha_dct = 0.01; % Adjust this value to control the DCT watermark strength

% Resize the watermark to match the number of DCT blocks in watermarkedImage_dct
watermark_dct = imresize(watermark, [numBlocksH_dct, numBlocksW_dct]);

for i = 1:numBlocksH_dct
    for j = 1:numBlocksW_dct
        block_dct = watermarkedImage_dct((i-1)*blockSize_dct+1:i*blockSize_dct, (j-1)*blockSize_dct+1:j*blockSize_dct);
        dctBlock = dct2(block_dct);
        
        % Step 10: Embed the watermark into the DCT coefficients
        if watermark_dct(i, j) == 1
            % If watermark bit is 1, add watermark to DC coefficient
            dctBlock(1, 1) = dctBlock(1, 1) + alpha_dct * dctBlock(1, 1);
        else
            % If watermark bit is 0, subtract watermark from DC coefficient
            dctBlock(1, 1) = dctBlock(1, 1) - alpha_dct * dctBlock(1, 1);
        end
        
        % Step 11: Apply the inverse DCT to obtain the watermarked block
        watermarkedBlock_dct = idct2(dctBlock);
        
        watermarkedImage_dct((i-1)*blockSize_dct+1:i*blockSize_dct, (j-1)*blockSize_dct+1:j*blockSize_dct) = watermarkedBlock_dct;
    end
end

% Display the watermarked image after DCT
figure;
subplot(1, 2, 1);
imshow(hostImage);
title('Original Host Image');
% 
% % subplot(1, 3, 2);
% % imshow(uint8(watermarkedImage_dwt));
% % title('Watermarked Image after DWT');
% 
subplot(1, 2, 2);
imshow(uint8(watermarkedImage_dct));
title('Watermarked Image after DCT');

% Convert the watermarkedImage_dct to uint8 (same data type as hostImage)
watermarkedImage_dct = uint8(watermarkedImage_dct);

% Calculate MSE (Mean Squared Error) after both DWT and DCT watermarking
mse = sum(sum((double(hostImage) - double(watermarkedImage_dct)).^2)) / (512*512);

% Calculate PSNR (Peak Signal-to-Noise Ratio) in dB after both DWT and DCT watermarking
psnr = 10 * log10(255^2 / mse);

% Calculate SSIM (Structural Similarity Index) after both DWT and DCT watermarking
ssim_val = ssim(hostImage, watermarkedImage_dct);

% Calculate UACI (Underwater Image Quality Measure) after both DWT and DCT watermarking
uaci = sum(sum(abs(double(hostImage) - double(watermarkedImage_dct)))) / (512*512*255);

% Calculate NPCR (Number of Pixel Change Rate) and CC (Correlation Coefficient) after both DWT and DCT watermarking
npcr = sum(sum(hostImage ~= watermarkedImage_dct)) / (512*512) * 100;
cc = corrcoef(double(hostImage(:)), double(watermarkedImage_dct(:)));
cc = cc(1, 2); % Extract the correlation coefficient value from the matrix

% Calculate NCC (Normalized Cross-Correlation) after both DWT and DCT watermarking
ncc = sum(sum(hostImage .* watermarkedImage_dct)) / sqrt(sum(sum(hostImage .^ 2)) * sum(sum(watermarkedImage_dct .^ 2)));

% Calculate entropy of the watermarked image after DWT and DCT watermarking
entropy_val = entropy(watermarkedImage_dct);
entropy_val1 = entropy(hostImage);

% Display metrics after both DWT and DCT watermarking
disp(['MSE: ', num2str(mse)]);
disp(['PSNR: ', num2str(psnr), ' dB']);
disp(['SSIM: ', num2str(ssim_val)]);
disp(['UACI: ', num2str(uaci)]);
disp(['NPCR: ', num2str(npcr), '%']);
disp(['CC: ', num2str(cc)]);
disp(['NCC: ', num2str(ncc)]);
disp(['Entropy Ori: ', num2str(entropy_val1)]);
disp(['Entropy: ', num2str(entropy_val)]);

% Display histograms after both DWT and DCT watermarking
figure;
subplot(1, 2, 1);
imhist(hostImage);
title('Original Host Image Histogram');

subplot(1, 2, 2);
imhist(watermarkedImage_dct);
title('Watermarked Image Histogram after DWT and DCT');

% subplot(1, 2, 2);
% imhist(hostImage - watermarkedImage_dct);
% title('Difference Histogram after DWT and DCT');
% 
% % Save the histograms after both DWT and DCT watermarking
% saveas(gcf, 'path_to_save_histograms.jpg');

% Step 1: Load and resize the host image to 512x512
hostImage = imread('test1.jpeg'); % Replace with your image path
hostImage = imresize(hostImage, [512, 512]);

% Step 2: Prepare the watermark (binary image or a sequence of bits)
% Assuming watermark is initially a color image, convert it to grayscale
watermark = imread('wtrmrk.jpg'); % Replace with your watermark image path
watermark = rgb2gray(watermark); % Convert color watermark to grayscale
watermark = imbinarize(watermark, 0.5); % Binarize the grayscale watermark
% figure, imshow(watermark)

% Step 3: Divide the host image into 8x8 blocks
blockSize = 8;
[height, width] = size(hostImage);
numBlocksH = height / blockSize;
numBlocksW = width / blockSize;
watermarkedImage = zeros(size(hostImage));

% Step 4: Apply Discrete Cosine Transform (DCT) on each block
alpha = 0.015; % Adjust this value to control the watermark strength

% Resize the watermark to match the number of DCT blocks in the host image
watermark = imresize(watermark, [numBlocksH, numBlocksW]);

for i = 1:numBlocksH
    for j = 1:numBlocksW
        block = hostImage((i-1)*blockSize+1:i*blockSize, (j-1)*blockSize+1:j*blockSize);
        dctBlock = dct2(block);
        
        % Step 5: Embed the watermark into the DCT coefficients
        if watermark(i, j) == 1
            % If watermark bit is 1, add watermark to DC coefficient
            dctBlock(1, 1) = dctBlock(1, 1) + alpha * dctBlock(1, 1);
        else
            % If watermark bit is 0, subtract watermark from DC coefficient
            dctBlock(1, 1) = dctBlock(1, 1) - alpha * dctBlock(1, 1);
        end
        
        % Step 6: Apply the inverse DCT to obtain the watermarked block
        watermarkedBlock = idct2(dctBlock);
        
        watermarkedImage((i-1)*blockSize+1:i*blockSize, (j-1)*blockSize+1:j*blockSize) = watermarkedBlock;
    end
end

% Display the watermarked image
figure;
subplot(1, 2, 1);
imshow(hostImage);
title('test1.jpeg');

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
figure,imhist(hostImage);
title('Original Host Image Histogram');

figure,imhist(watermarkedImage);
title('Watermarked Image Histogram');

% subplot(1, 3, 3);
% imhist(hostImage - watermarkedImage);
% title('Difference Histogram');

% Save the histograms
% saveas(gcf, 'path_to_save_histograms.jpg');

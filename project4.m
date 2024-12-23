% 读取图像
img = imread("C:\Users\30983\Pictures\Saved Pictures\8.jpg");
grayImage = rgb2gray(img);

% 添加高斯噪声
sigma = 0.05; % 噪声标准差
J_gaussian = imnoise(grayImage, 'gaussian', 0, sigma);

% 添加椒盐噪声
density = 0.02; % 噪声密度
J_salt_pepper = imnoise(grayImage, 'salt & pepper', density);

% 显示原始图像和加噪图像
figure;
subplot(5,3,1); imshow(grayImage); title('原始图像');
subplot(5,3,2); imshow(J_gaussian); title('高斯噪声图像');
subplot(5,3,3); imshow(J_salt_pepper); title('椒盐噪声图像');

% 均值滤波
h = fspecial('average', [5 5]); % 5x5均值滤波器
K_gaussian_avg = imfilter(J_gaussian, h);

% 显示滤波结果
subplot(5,3,4); imshow(J_gaussian); title('高斯噪声图像');
subplot(5,3,5); imshow(K_gaussian_avg); title('均值滤波结果');

% 中值滤波
K_salt_pepper_med = medfilt2(J_salt_pepper, [5 5]); % 5x5中值滤波器

% 显示滤波结果
subplot(5,3,7); imshow(J_salt_pepper); title('椒盐噪声图像');
subplot(5,3,8); imshow(K_salt_pepper_med); title('中值滤波结果');

% 高斯低通滤波
D0 = 30; % 截止频率
h_gaussian = fspecial('gaussian', [size(grayImage,1) size(grayImage,2)], D0);
H_gaussian = fftshift(h_gaussian);

% 将图像转换到频域
F = fft2(J_gaussian);
Fshift = fftshift(F);

% 应用高斯低通滤波器
Gshift = Fshift .* H_gaussian;

% 将图像转换回空间域
G = ifftshift(Gshift);
K_gaussian_gaussian = ifft2(G);

% 显示滤波结果
subplot(5,3,10); imshow(J_gaussian); title('高斯噪声图像');
subplot(5,3,11); imshow(abs(K_gaussian_gaussian)); title('高斯低通滤波结果');

% 中值滤波（频域）
K_salt_pepper_med_freq = medfilt2(J_salt_pepper, [5 5]);

% 显示滤波结果
subplot(5,3,13); imshow(J_salt_pepper); title('椒盐噪声图像');
subplot(5,3,14); imshow(K_salt_pepper_med_freq); title('中值滤波结果');

% 读取图像
img = imread("C:\Users\30983\Pictures\Saved Pictures\8.jpg");

% 转换为灰度图像
grayImg = rgb2gray(img);

% 线性变换
a = 2; % 斜率
b = 50; % 截距
linearTransformedImg = a * grayImg + b;
linearTransformedImg = uint8(min(max(linearTransformedImg, 0), 255)); % 限制在 0 - 255 范围内

% 对数变换
grayImgNormalized = mat2gray(grayImg);
logTransformedImg = log(1 + grayImgNormalized);
logTransformedImg = uint8((logTransformedImg - min(logTransformedImg(:))) / (max(logTransformedImg(:)) - min(logTransformedImg(:))) * 255); % 归一化到 0 - 255

% 指数变换
c = 2; % 底数
expTransformedImg = c.^ grayImg;
expTransformedImg = uint8((expTransformedImg - min(expTransformedImg(:))) / (max(expTransformedImg(:)) - min(expTransformedImg(:))) * 255); % 归一化到 0 - 255

% 显示原始灰度图像
figure;subplot(221)
imshow(grayImg);
title('原始灰度图像');

% 显示线性变换后的图像
subplot(222)
imshow(linearTransformedImg);
title('线性变换后的图像');

% 显示对数变换后的图像
subplot(223)
imshow(logTransformedImg);
title('对数变换后的图像');

% 显示指数变换后的图像
subplot(224)
imshow(expTransformedImg);
title('指数变换后的图像');
% 读取图像
img = imread("C:\Users\30983\Pictures\Saved Pictures\8.jpg");

% 转换为灰度图像
grayImg = rgb2gray(img);


% 直方图均衡化
NewgrayImg=histeq(grayImg,256);


% 规定化的目标直方图（示例）
targetHistogram = [0 0 0 0 0 0 0 0 0 10 20 30 40 50 60 70 80 90 100];

% 直方图匹配（规定化）
matchedImg = histeq(grayImg, targetHistogram);

% 显示原始图像的灰度直方图
subplot(321);imshow(grayImg);title('灰度图像');
subplot(322);imhist(grayImg);title('Gray Histogram');	

% 显示均衡化后的结果图像
subplot(323);imshow(NewgrayImg);title('histeq均衡化结果');
% 显示均衡化后的直方图
subplot(324);imhist(NewgrayImg);title('histeq均衡化直方图');	

% 显示匹配（规定化）后的图像的灰度直方图
subplot(325)
[counts, bins] = imhist(matchedImg);
bar(bins, counts);
title('匹配（规定化）后图像灰度直方图');
xlabel('灰度级');
ylabel('像素数量');
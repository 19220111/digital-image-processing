I = imread("C:\Users\30983\Pictures\Saved Pictures\8.jpg");  % 读取图片，放在 I 中
[H, W, G] = size(I);  % 读取图片的三维坐标信息
H1 = H/4; H2 = H*2/3; W1 = W/10; W2 = W*85/100;
S = (W2 - W1) * (H2 - H1);
I = imcrop(I, [W1, H1, W2, H2]);  % 适当对图片四周进行裁剪

subplot(2, 3, 1);
imshow(I);
title("源图像");  % 显示源图像

I1 = rgb2gray(I);  % 灰度处理

I2 = im2bw(I1, 0.6);  % 二值化处理

subplot(2, 3, 2);
imshow(I1);
title("灰度图像");  % 显示灰度图像中的内容

I2 = ~I2;  % 取反，黑白交换


se = strel('disk', 5);% 创建指定形状的结构元素

I2 = imclose(I2, se);
I2 = imopen(I2, se);% 消除小块区域

subplot(2, 3, 3);
imshow(I2);
title('开运算闭运算图像');  % 显示开运算闭运算图像中的内容


L = bwlabel(I2);  % 获取连通区域标签和面积

subplot(2, 3, 4);
imshow(L);
title('边缘图像上的区域');

STATS = regionprops(L, 'all');  % 获得图像各个区域的属性

strNum = max(L(:));


for i = 1:1:strNum
    rectangle('Position', STATS(i).BoundingBox,'edgecolor', 'r');  % 对所有区域框选
end

subplot(2, 3, 5);
imshow(I);
title('源图像上的区域');  % 在源图像上显示


Ar = cat(1, STATS.ConvexArea);  % 提取各个区域内的像素个数信息

ind = find(Ar > S/4);% 找到其中符合特征条件的区域

rectangle('Position', STATS(ind).BoundingBox,"edgecolor",'r');  % 对所选区域框选

CK = cat(1, STATS.BoundingBox);  % 提取各个区域的坐标信息

I4 = imcrop(I, [CK(ind, 1), CK(ind, 2), CK(ind, 3), CK(ind, 4)]);  % 裁剪区域

subplot(2, 3, 6);
imshow(I4);
title("剪切图");
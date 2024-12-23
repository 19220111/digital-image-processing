function featureExtraction()
    % 读取原始图像
    originalImg = imread("C:\Users\30983\Pictures\Saved Pictures\8.jpg");  % 读取图片，放在 originalImg 中
    [H, W, G] = size(originalImg);  % 读取图片的三维坐标信息
    H1 = H/4; H2 = H*2/3; W1 = W/10; W2 = W*85/100;
    S = (W2 - W1) * (H2 - H1);
    originalImg = imcrop(originalImg, [W1, H1, W2, H2]);  % 适当对图片四周进行裁剪

    % 灰度处理
    grayImg = rgb2gray(originalImg);

    % 二值化处理
    binaryImg = im2bw(grayImg, 0.6);
    binaryImg = ~binaryImg;  % 取反，黑白交换

    % 形态学处理
    se = strel('disk', 5);
    binaryImg = imclose(binaryImg, se);
    binaryImg = imopen(binaryImg, se);

    % 连通区域标记
    L = bwlabel(binaryImg);

    % 获取区域属性
    STATS = regionprops(L, 'all');

    % 找到符合特征条件的区域
    Ar = cat(1, STATS.ConvexArea);
    ind = find(Ar > S/4);

    % 提取目标图像
    targetImg = imcrop(originalImg, [STATS(ind).BoundingBox]);

    % LBP 特征提取
    originalLBP = extractLBPFeatures(originalImg);
    targetLBP = extractLBPFeatures(targetImg);

    % HOG 特征提取
    originalHOG = extractHOGFeatures(single(originalImg));  % 转换为单精度浮点数
    targetHOG = extractHOGFeatures(single(targetImg));      % 转换为单精度浮点数

    % 显示或进一步处理提取的特征
    disp('Original Image LBP Features:');
    disp(originalLBP);
    disp('Target Image LBP Features:');
    disp(targetLBP);

    disp('Original Image HOG Features:');
    disp(originalHOG);
    disp('Target Image HOG Features:');
    disp(targetHOG);
end

function lbpFeatures = extractLBPFeatures(image)
    [rows, cols] = size(image);
    lbpFeatures = zeros(rows, cols);

    for i = 2 : rows - 1
        for j = 2 : cols - 1
            centerPixel = image(i, j);
            code = 0;

            if image(i - 1, j - 1) >= centerPixel
                code = code + 128;
            end
            if image(i - 1, j) >= centerPixel
                code = code + 64;
            end
            if image(i - 1, j + 1) >= centerPixel
                code = code + 32;
            end
            if image(i, j + 1) >= centerPixel
                code = code + 16;
            end
            if image(i + 1, j + 1) >= centerPixel
                code = code + 8;
            end
            if image(i + 1, j) >= centerPixel
                code = code + 4;
            end
            if image(i + 1, j - 1) >= centerPixel
                code = code + 2;
            end
            if image(i, j - 1) >= centerPixel
                code = code + 1;
            end

            lbpFeatures(i, j) = code;
        end
    end
end

function hogFeatures = extractHOGFeatures(image)
    if ~exist('vl_setup', 'file')
        vl_setup;
    end
    hogFeatures = vl_hog(image, 8);
end
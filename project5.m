function edgeExtraction()
    % 读取图像
    img = imread("C:\Users\30983\Pictures\Saved Pictures\8.jpg");
    % 转换为灰度图像
    grayImg = rgb2gray(img);

    % Robert 算子
    robertH = [0 1; -1 0];
    robertV = [1 0; 0 -1];
    robertEdgeH = imfilter(grayImg, robertH);
    robertEdgeV = imfilter(grayImg, robertV);
    robertEdge = uint8(sqrt(double(robertEdgeH).^2 + double(robertEdgeV).^2));

    % Prewitt 算子
    prewittH = [-1 -1 -1; 0 0 0; 1 1 1];
    prewittV = [-1 0 1; -1 0 1; -1 0 1];
    prewittEdgeH = imfilter(grayImg, prewittH);
    prewittEdgeV = imfilter(grayImg, prewittV);
    prewittEdge = uint8(sqrt(double(prewittEdgeH).^2 + double(prewittEdgeV).^2));

    % Sobel 算子
    sobelH = [-1 -2 -1; 0 0 0; 1 2 1];
    sobelV = [-1 0 1; -2 0 2; -1 0 1];
    sobelEdgeH = imfilter(grayImg, sobelH);
    sobelEdgeV = imfilter(grayImg, sobelV);
    sobelEdge = uint8(sqrt(double(sobelEdgeH).^2 + double(sobelEdgeV).^2));

    % 拉普拉斯算子
    laplacianKernel = [0 -1 0; -1 4 -1; 0 -1 0];
    laplacianEdge = imfilter(grayImg, laplacianKernel);

    % 显示结果
    figure;
    subplot(2, 2, 1);
    imshow(robertEdge);
    title('Robert 算子边缘提取');

    subplot(2, 2, 2);
    imshow(prewittEdge);
    title('Prewitt 算子边缘提取');

    subplot(2, 2, 3);
    imshow(sobelEdge);
    title('Sobel 算子边缘提取');

    subplot(2, 2, 4);
    imshow(laplacianEdge);
    title('拉普拉斯算子边缘提取');
end

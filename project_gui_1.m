function image_processing_gui()
    % 创建主窗口
    fig = figure('Position',[100 100 800 600],'Name','Image Processing GUI');
    
    % 打开图像按钮
    uicontrol('Style','pushbutton','String','Open Image',...
        'Position',[50 50 100 30],...
        'Callback',@open_image_callback);
    
    % 灰度直方图显示按钮
    uicontrol('Style','pushbutton','String','Show Grayscale Histogram',...
        'Position',[50 100 150 30],...
        'Callback',@show_histogram_callback);
    
    % 直方图均衡化按钮
    uicontrol('Style','pushbutton','String','Histogram Equalization',...
        'Position',[50 150 150 30],...
        'Callback',@histogram_equalization_callback);
    
    % 直方图匹配按钮
    uicontrol('Style','pushbutton','String','Histogram Matching',...
        'Position',[50 200 150 30],...
        'Callback',@histogram_matching_callback);
    
    % 对比度增强 - 线性变换按钮
    uicontrol('Style','pushbutton','String','Contrast Enhancement - Linear',...
        'Position',[50 250 150 30],...
        'Callback',@linear_contrast_enhancement_callback);
    
    % 对比度增强 - 对数变换按钮
    uicontrol('Style','pushbutton','String','Contrast Enhancement - Logarithmic',...
        'Position',[50 300 150 30],...
        'Callback',@logarithmic_contrast_enhancement_callback);
    
    % 对比度增强 - 指数变换按钮
    uicontrol('Style','pushbutton','String','Contrast Enhancement - Exponential',...
        'Position',[50 350 150 30],...
        'Callback',@exponential_contrast_enhancement_callback);
    
    % 图像缩放按钮
    uicontrol('Style','pushbutton','String','Image Scaling',...
        'Position',[50 400 150 30],...
        'Callback',@image_scaling_callback);
    
    % 图像旋转按钮
    uicontrol('Style','pushbutton','String','Image Rotation',...
        'Position',[50 450 150 30],...
        'Callback',@image_rotation_callback);
    
    % 图像加噪及滤波按钮
    uicontrol('Style','pushbutton','String','Image Noise and Filtering',...
        'Position',[50 500 150 30],...
        'Callback',@image_noise_and_filtering_callback);
    
    % 边缘提取按钮
    uicontrol('Style','pushbutton','String','Edge Extraction',...
        'Position',[250 50 150 30],...
        'Callback',@edge_extraction_callback);
    
    % 目标提取按钮
    uicontrol('Style','pushbutton','String','Object Extraction',...
        'Position',[250 100 150 30],...
        'Callback',@object_extraction_callback);
    
    % 特征提取按钮
    uicontrol('Style','pushbutton','String','Feature Extraction',...
        'Position',[250 150 150 30],...
        'Callback',@feature_extraction_callback);
end

function open_image_callback(hObject, eventdata)
    [filename, pathname] = uigetfile({'*.jpg;*.png;*.bmp'}, 'Select an image');
    if isequal(filename,0) || isequal(pathname,0)
        return;
    end
    global image;
    image = imread(fullfile(pathname, filename));
    axes(handles.axes1);
    imshow(image);
end

function show_histogram_callback(hObject, eventdata)
    global image;
    if isempty(image)
        warndlg('Please open an image first.');
        return;
    end
    gray_image = rgb2gray(image);
    histogram(gray_image);
    title('Grayscale Histogram');
end

function histogram_equalization_callback(hObject, eventdata)
    global image;
    if isempty(image)
        warndlg('Please open an image first.');
        return;
    end
    gray_image = rgb2gray(image);
    equalized_image = histeq(gray_image);
    axes(handles.axes1);
    imshow(equalized_image);
    title('Histogram Equalized Image');
end

function histogram_matching_callback(hObject, eventdata)
    global image;
    if isempty(image)
        warndlg('Please open an image first.');
        return;
    end
    gray_image = rgb2gray(image);
    target_histogram = ones(256, 1) / 256;  % 示例目标直方图，您可以根据需要修改
    matched_image = histeq(gray_image, target_histogram);
    axes(handles.axes1);
    imshow(matched_image);
    title('Histogram Matched Image');
end

function linear_contrast_enhancement_callback(hObject, eventdata)
    global image;
    if isempty(image)
        warndlg('Please open an image first.');
        return;
    end
    gray_image = rgb2gray(image);
    a = 2; b = 50;  % 线性变换参数，您可以根据需要修改
    enhanced_image = a * gray_image + b;
    enhanced_image = uint8(enhanced_image);
    axes(handles.axes1);
    imshow(enhanced_image);
    title('Linearly Enhanced Image');
end

function logarithmic_contrast_enhancement_callback(hObject, eventdata)
    global image;
    if isempty(image)
        warndlg('Please open an image first.');
        return;
    end
    gray_image = rgb2gray(image);
    c = 1;  % 对数变换参数，您可以根据需要修改
    enhanced_image = c * log(1 + double(gray_image));
    enhanced_image = uint8(enhanced_image);
    axes(handles.axes1);
    imshow(enhanced_image);
    title('Logarithmically Enhanced Image');
end

function exponential_contrast_enhancement_callback(hObject, eventdata)
    global image;
    if isempty(image)
        warndlg('Please open an image first.');
        return;
    end
    gray_image = rgb2gray(image);
    c = 1;  % 指数变换参数，您可以根据需要修改
    enhanced_image = c * exp(double(gray_image));
    enhanced_image = uint8(enhanced_image);
    axes(handles.axes1);
    imshow(enhanced_image);
    title('Exponentially Enhanced Image');
end

function image_scaling_callback(hObject, eventdata)
    global image;
    if isempty(image)
        warndlg('Please open an image first.');
        return;
    end
    scale_factor = inputdlg('Enter the scaling factor:', 'Image Scaling', 1, {'0.5'});
    scale_factor = str2double(scale_factor{1});
    scaled_image = imresize(image, scale_factor);
    axes(handles.axes1);
    imshow(scaled_image);
    title('Scaled Image');
end

function image_rotation_callback(hObject, eventdata)
    global image;
    if isempty(image)
        warndlg('Please open an image first.');
        return;
    end
    angle = inputdlg('Enter the rotation angle (in degrees):', 'Image Rotation', 1, {'45'});
    angle = str2double(angle{1});
    rotated_image = imrotate(image, angle);
    axes(handles.axes1);
    imshow(rotated_image);
    title('Rotated Image');
end

function image_noise_and_filtering_callback(hObject, eventdata)
    global image;
    if isempty(image)
        warndlg('Please open an image first.');
        return;
    end
    noise_type = questdlg('Select noise type:', 'Noise Type', 'Gaussian', 'Salt & Pepper', 'Gaussian');
    if strcmp(noise_type, 'Gaussian')
        mean = 0;
        var = 0.01;
        noisy_image = imnoise(image, 'gaussian', mean, var);
    elseif strcmp(noise_type, 'Salt & Pepper')
        density = 0.05;
        noisy_image = imnoise(image,'salt & pepper', density);
    end
    % 空域滤波 - 均值滤波
    mean_filtered_image = filter2(fspecial('average', [3 3]), noisy_image) / 255;
    % 空域滤波 - 中值滤波
    median_filtered_image = medfilt2(noisy_image, [3 3]);
    % 频域滤波 - 低通滤波
    freq_domain_filtered_image = lowPassFilter(noisy_image);
    
    figure;
    subplot(2, 3, 1);
    imshow(noisy_image);
    title('Noisy Image');
    subplot(2, 3, 2);
    imshow(mean_filtered_image);
    title('Mean Filtered Image');
    subplot(2, 3, 3);
    imshow(median_filtered_image);
    title('Median Filtered Image');
    subplot(2, 3, 4);
    imshow(freq_domain_filtered_image);
    title('Frequency Domain Filtered Image');
end

function edge_extraction_callback(hObject, eventdata)
    global image;
    if isempty(image)
        warndlg('Please open an image first.');
        return;
    end
    gray_image = rgb2gray(image);
    % Robert 算子
    robertH = [0 1; -1 0];
    robertV = [1 0; 0 -1];
    robertEdgeH = imfilter(gray_image, robertH);
    robertEdgeV = imfilter(gray_image, robertV);
    robertEdge = sqrt(robertEdgeH.^2 + robertEdgeV.^2);
    % Prewitt 算子
    prewittH = [-1 -1 -1; 0 0 0; 1 1 1];
    prewittV = [-1 0 1; -1 0 1; -1 0 1];
    prewittEdgeH = imfilter(gray_image, prewittH);
    prewittEdgeV = imfilter(gray_image, prewittV);
    prewittEdge = sqrt(prewittEdgeH.^2 + prewittEdgeV.^2);
    % Sobel 算子
    sobelH = [-1 -2 -1; 0 0 0; 1 2 1];
    sobelV = [-1 0 1; -2 0 2; -1 0 1];
    sobelEdgeH = imfilter(gray_image, sobelH);
    sobelEdgeV = imfilter(gray_image, sobelV);
    sobelEdge = sqrt(sobelEdgeH.^2 + sobelEdgeV.^2);
    % 拉普拉斯算子
    laplacianKernel = [0 -1 0; -1 4 -1; 0 -1 0];
    laplacianEdge = imfilter(gray_image, laplacianKernel);
    figure;
    subplot(2, 2, 1);
    imshow(robertEdge);
    title('Robert Operator Edge');
    subplot(2, 2, 2);
    imshow(prewittEdge);
    title('Prewitt Operator Edge');
    subplot(2, 2, 3);
    imshow(sobelEdge);
    title('Sobel Operator Edge');
    subplot(2, 2, 4);
    imshow(laplacianEdge);
    title('Laplacian Operator Edge');
end

function object_extraction_callback(hObject, eventdata)
    global image;
    if isempty(image)
        warndlg('Please open an image first.');
        return;
    end
    % 假设以下是您的目标提取代码
    [H, W, G] = size(image);
    H1 = H/4; H2 = H*2/3; W1 = W/10; W2 = W*85/100;
    S = (W2 - W1) * (H2 - H1);
    image = imcrop(image, [W1, H1, W2, H2]);
    gray_image = rgb2gray(image);
    binary_image = im2bw(gray_image, 0.6);
    binary_image = ~binary_image;
    se = strel('disk', 5);
    binary_image = imclose(binary_image, se);
    binary_image = imopen(binary_image, se);
    L = bwlabel(binary_image);
    STATS = regionprops(L, 'all');
    strNum = max(L(:));
    for i = 1:1:strNum
        rectangle('Position', STATS(i).BoundingBox,'edgecolor', 'r');
    end
    axes(handles.axes1);
    imshow(image);
    title('Object Extracted Image');
end

function feature_extraction_callback(hObject, eventdata)
    global image;
    if isempty(image)
        warndlg('Please open an image first.');
        return;
    end
    % 假设以下是您的特征提取代码
    % 例如 LBP 特征提取
    lbp_features = extractLBPFeatures(image);
    % 例如 HOG 特征提取
    hog_features = extractHOGFeatures(image);
    % 可以在此显示或处理提取的特征
end

function filteredImage = lowPassFilter(image)
    % 将图像转换到频域
    F = fft2(double(image));
    Fshift = fftshift(F);
    % 创建低通滤波器掩码
    [M, N] = size(image);
    D0 = 30; % 截止频率
    u = 0:(M-1);
    v = 0:(N-1);
    idx = find(u > M/2); u(idx) = u(idx) - M;
    idy = find(v > N/2); v(idy) = v(idy) - N;
    [V, U] = meshgrid(v, u);
    D = sqrt(U.^2 + V.^2);
    H = double(D <= D0);
    % 应用低通滤波器
    G = H.* Fshift;
    G(abs(U) > D0 | abs(V) > D0) = 0;
    % 逆傅里叶变换
    g = ifftshift(G);
    filteredImage = real(ifft2(g));
    filteredImage = uint8(filteredImage);
end

function lbpFeatures = extractLBPFeatures(image)
    % 自定义 LBP 特征提取函数实现
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
    % 自定义 HOG 特征提取函数实现或使用合适的库
    % 这里为示例，您可能需要根据实际需求修改
    hogFeatures = zeros(1, 100);  % 假设的 HOG 特征向量
end


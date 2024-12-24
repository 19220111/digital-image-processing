function varargout = GUI_1(varargin)

gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @GUI_1_OpeningFcn, ...
                   'gui_OutputFcn',  @GUI_1_OutputFcn, ...
                   'gui_LayoutFcn',  [] , ...
                   'gui_Callback',   []);
if nargin && ischar(varargin{1})
    gui_State.gui_Callback = str2func(varargin{1});
end

if nargout
    [varargout{1:nargout}] = gui_mainfcn(gui_State, varargin{:});
else
    gui_mainfcn(gui_State, varargin{:});
end
end


% --- Executes just before GUI_1 is made visible.
function GUI_1_OpeningFcn(hObject, eventdata, handles, varargin)
    handles.output = hObject;
    handles.axes1 = axes('Parent', hObject);
    guidata(hObject, handles);
end
% UIWAIT makes GUI_1 wait for user response (see UIRESUME)
% uiwait(handles.figure1);


% --- Outputs from this function are returned to the command line.
function varargout = GUI_1_OutputFcn(hObject, eventdata, handles) 
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
varargout{1} = handles.output;
end

% --- Executes on button press in pushbutton1.
function pushbutton1_Callback(hObject, eventdata, handles)
    [filename, pathname] = uigetfile({'*.jpg;*.png;*.bmp'}, 'Select an image');
    if isequal(filename,0) || isequal(pathname,0)
        return;
    end
    global image;
    image = imread(fullfile(pathname, filename));
end
                                                                                                                        
% hObject    handle to pushbutton1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% --- Executes on button press in pushbutton2.
 function pushbutton2_Callback(hObject, eventdata, handles)
    global image;
    if isempty(image)
        warndlg('Please select an image first.');
        return;
    end
    axes(handles.axes1);
    imshow(image);
    title('Original Image');
    handles.img = img;
    guidata(hObject, handles);
end
% hObject    handle to pushbutton2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% --- Executes on button press in pushbutton3.
function pushbutton3_Callback(hObject, eventdata, handles)
    global image;
    if isempty(image)
        warndlg('Please select an image first.');
        return;
    end
    gray_image = rgb2gray(image);
    axes(handles.axes2);
    imhist(gray_image);
    title('Original Image Grayscale Histogram');
end
% hObject    handle to pushbutton3 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% --- Executes on button press in pushbutton4.
function pushbutton4_Callback(hObject, eventdata, handles)
    global image;
    if isempty(image)
        warndlg('Please select an image first.');
        return;
    end
    gray_image = rgb2gray(image);
    equalized_image = histeq(gray_image);
    axes(handles.axes2);
    imshow(equalized_image);
    title('Histogram Equalized Image');
end
% hObject    handle to pushbutton4 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% --- Executes on button press in pushbutton5.
function pushbutton5_Callback(hObject, eventdata, handles)
    global image;
    if isempty(image)
        warndlg('Please select an image first.');
        return;
    end
    gray_image = rgb2gray(image);
    target_histogram = ones(256, 1) / 256;  % 示例目标直方图，您可以根据需要修改
    matched_image = histeq(gray_image, target_histogram);
    axes(handles.axes1);
    imshow(matched_image);
    title('Histogram Matched Image');
end
% hObject    handle to pushbutton5 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% --- Executes on button press in pushbutton6.
function pushbutton6_Callback(hObject, eventdata, handles)
    global image;
    if isempty(image)
        warndlg('Please select an image first.');
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
% hObject    handle to pushbutton6 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% --- Executes on button press in pushbutton7.
function pushbutton7_Callback(hObject, eventdata, handles)
    global image;
    if isempty(image)
        warndlg('Please select an image first.');
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
% hObject    handle to pushbutton7 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% --- Executes on button press in pushbutton8.
function pushbutton8_Callback(hObject, eventdata, handles)
    global image;
    if isempty(image)
        warndlg('Please select an image first.');
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
% hObject    handle to pushbutton8 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% --- Executes on button press in pushbutton9.
function pushbutton9_Callback(hObject, eventdata, handles)
    global image;
    if isempty(image)
        warndlg('Please select an image first.');
        return;
    end
    % 获取用户输入的缩放因子（这里假设编辑文本框的Tag为edit_ScaleFactor）
    scaleFactor = str2double(get(handles.edit_ScaleFactor, 'String'));
    if scaleFactor <= 0
        warng('缩放因子应大于0');
        return;
    end
    scale_factor = inputdlg('Enter the scaling factor:', 'Image Scaling', 1, {'0.5'});
    scale_factor = str2double(scale_factor{1});
    scaled_image = imresize(image, scale_factor);
    axes(handles.axes2);
    imshow(scaled_image);
    title('Scaled Image');
    % 缩放变换
    scaledImg = imresize(handles.imageData, scaleFactor);
    % 在同一个figure中显示原始图像和缩放后的图像
    figure;
    subplot(1, 2, 1);
    imshow(handles.imageData);
    title('原始图像');
    subplot(1, 2, 2);
    imshow(scaledImg);
    title('缩放后的图像');
end
% hObject    handle to pushbutton9 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% --- Executes on button press in pushbutton10.
function pushbutton10_Callback(hObject, eventdata, handles)
    global image;
    if isempty(image)
        warndlg('Please select an image first.');
        return;
    end
    angle = inputdlg('Enter the rotation angle (in degrees):', 'Image Rotation', 1, {'45'});
    angle = str2double(angle{1});
    rotated_image = imrotate(image, angle);
    axes(handles.axes1);
    imshow(rotated_image);
    title('Rotated Image');
end
% hObject    handle to pushbutton10 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% --- Executes on button press in pushbutton11.
function pushbutton11_Callback(hObject, eventdata, handles)
    global image;
    if isempty(image)
        warndlg('Please select an image first.');
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
    axes(handles.axes1);
    imshow(noisy_image);
    title('Noisy Image');
end
% hObject    handle to pushbutton11 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% --- Executes on button press in pushbutton12.
function pushbutton12_Callback(hObject, eventdata, handles)
    global image;
    if isempty(image)
        warndlg('Please select an image first.');
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
% hObject    handle to pushbutton12 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% --- Executes on button press in pushbutton13.
function pushbutton13_Callback(hObject, eventdata, handles)
    global image;
    if isempty(image)
        warndlg('Please select an image first.');
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
% hObject    handle to pushbutton13 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% --- Executes on button press in pushbutton14.
function pushbutton14_Callback(hObject, eventdata, handles)
    global image;
    if isempty(image)
        warndlg('Please select an image first.');
        return;
    end
    % 提取目标图像
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
% hObject    handle to pushbutton14 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% --- Executes on button press in pushbutton15.
function pushbutton15_Callback(hObject, eventdata, handles)
    global image;
    if isempty(image)
        warndlg('Please select an image first.');
        return;
    end
    % 提取目标图像
    [H, W, G] = size(image);
    H1 = H/4; 
    H2 = H*2/3; 
    W1 = W/10; 
    W2 = W*85/100;
    S = (W2 - W1) * (H2 - H1);
    image = imcrop(image, [W1, H1, W2, H2]);
    gray_image = rgb2gray(image);

    % 二值化处理
    binary_image = im2bw(gray_image, 0.6);
    binary_image = ~binary_image;

    % 形态学处理
    se = strel('disk', 5);
    binary_image = imclose(binary_image, se);
    binary_image = imopen(binary_image, se);

    % 连通区域标记
    L = bwlabel(binary_image);

    % 获取区域属性
    STATS = regionprops(L, 'all');

    % 找到符合特征条件的区域
    Ar = cat(1, STATS.ConvexArea);
    ind = find(Ar > S/4);

    % 提取目标图像
    target_image = imcrop(image, [STATS(ind).BoundingBox]);

    % LBP 特征提取
    original_lbp = extractLBPFeatures(gray_image);
    target_lbp = extractLBPFeatures(rgb2gray(target_image));

    % HOG 特征提取
    original_hog = extractHOGFeatures(single(gray_image));
    target_hog = extractHOGFeatures(single(rgb2gray(target_image)));

    % 显示或进一步处理提取的特征
    disp('Original Image LBP Features:');
    disp(original_lbp);
    disp('Target Image LBP Features:');
    disp(target_lbp);

    disp('Original Image HOG Features:');
    disp(original_hog);
    disp('Target Image HOG Features:');
    disp(target_hog);
end
% hObject    handle to pushbutton15 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

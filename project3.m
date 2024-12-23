function imageTransformations()
    % 读取图像
    img = imread("C:\Users\30983\Pictures\Saved Pictures\8.jpg");
    % 缩放变换
    scaleFactor = 2;  % 缩放因子
    scaledImg = imresize(img, scaleFactor);

    % 旋转变换
    rotationAngle = 45;  % 旋转角度（度）
    rotatedImg = imrotate(img, rotationAngle);

    % 显示原始图像
    figure;
    imshow(img);
    title('原始图像');

    % 显示缩放后的图像
    figure;
    imshow(scaledImg);
    title('缩放后的图像');

    % 显示旋转后的图像
    figure;
    imshow(rotatedImg);
    title('旋转后的图像');
end
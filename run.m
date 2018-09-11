clear all
close all
datapath = 'C:\Users\Merve\Documents\MATLAB\FaceRecognition\Train'; 
TestImage = 'C:\Users\Merve\Documents\MATLAB\FaceRecognition\Test\2.jpg';

A = imread(TestImage);
%CascadeObjectDetector ile yüz kordinatlarý alýnýr
FaceDetector = vision.CascadeObjectDetector();
%Bulunan yüzlerin koordinat deðerlerini BBOX þeklinde bir matrise atýlýr.
BBOX = step(FaceDetector, A); 
imagecount = size(BBOX,1);
%bir matris oluþturur 1ximagecount
image = zeros(1,imagecount);

rec = []; 
% tüm resimleri dolaþýyor
for sayi=1:imagecount
    I2 = imcrop(A,BBOX(sayi,:));% tüm resimleri ayný boyutta kesiyor
    I2 = imresize (I2,[200 180]); % tüm resimleri ayný boyuta getirilir.
    %pca metodu çalýþtýrýlarak tanýnan dosyalar alýnýr
    recog_img = pca(datapath,TestImage);
end
%matching metodu çalýþtýrýlýr

result = strcat('the recognized image is : ',recog_img);
disp(result);

selected_img = strcat(datapath,'\',recog_img);
select_img = imread(selected_img);
imshow(select_img);
title('Bulunan Resim')
test_img = imread(TestImage);
figure,imshow(test_img);  
title('Test Resmi')
%for i=1:length(image)
 %   matching = match(image(i), rec(i));
  %  word(i) = {matching};
%end
%B = insertObjectAnnotation(A,'rectangle', BBOX, word,'TextBoxOpacity',0.8,'FontSize',30);
%imshow(B);
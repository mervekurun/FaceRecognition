clear all
close all
datapath = 'C:\Users\Merve\Documents\MATLAB\FaceRecognition\Train'; 
TestImage = 'C:\Users\Merve\Documents\MATLAB\FaceRecognition\Test\2.jpg';

A = imread(TestImage);
%CascadeObjectDetector ile y�z kordinatlar� al�n�r
FaceDetector = vision.CascadeObjectDetector();
%Bulunan y�zlerin koordinat de�erlerini BBOX �eklinde bir matrise at�l�r.
BBOX = step(FaceDetector, A); 
imagecount = size(BBOX,1);
%bir matris olu�turur 1ximagecount
image = zeros(1,imagecount);

rec = []; 
% t�m resimleri dola��yor
for sayi=1:imagecount
    I2 = imcrop(A,BBOX(sayi,:));% t�m resimleri ayn� boyutta kesiyor
    I2 = imresize (I2,[200 180]); % t�m resimleri ayn� boyuta getirilir.
    %pca metodu �al��t�r�larak tan�nan dosyalar al�n�r
    recog_img = pca(datapath,TestImage);
end
%matching metodu �al��t�r�l�r

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
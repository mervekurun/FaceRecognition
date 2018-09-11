function [recognized_img]=pca(datapath,testimg)


D = dir(datapath);
%resim sayýsýnýn bulunmasý
imgcount = 0;
for i=1 : size(D,1)
    if not(strcmp(D(i).name,'.')|strcmp(D(i).name,'..')|strcmp(D(i).name,'Thumbs.db'))
        imgcount = imgcount + 1; 
    end
end
%resimlerin matrise yerleþtirilmesi
X=[];
for i = 1 : imgcount
    str = strcat(datapath,'\',int2str(i),'.jpg');
    img = imread(str);
    img = rgb2gray(img);
    [r c] = size(img);
    temp = reshape(img',r*c,1);%2D image 1D vektöre dönüþtürülür
                              
       X=[X , temp]; %temp X matrisisne eklenir (sütün eklenmesi) 
end

% kovaryans matrisinin bulunmasý
m = mean(X,2);% matrisin satýr(2'nin anlamý 1 olsa sütun ort. alacaktý) ortalamasýný bulur
imgcount = size(X,2);

%Ortalamalar çýkartýlarak A matrisi hesaplanýyor
A = [];
for i=1 : imgcount
    temp = double(X(:,i)) - m;
    A = [A temp];
end
L= A' * A; 
%V özvektör natrisi D: Özdeðer matrisi
[V,D]=eig(L); %eig özdeðer komutu
% w izdüþüm matrisi bulunmasý
L_eig_vec = [];
for i = 1 : size(V,2) 
    if( D(i,i) > 1 )
        L_eig_vec = [L_eig_vec V(:,i)];
    end
end

%eigenfaces bulunmasý
%Seçilen öz vektörler, çýkarma iþleminden elde edilen matris ile çarpýlarak azaltýlmýþ öz yüzuzayý (eigenfaces)  elde edilir.
eigenfaces = A * L_eig_vec;

%w matrisi kullanýlarak öznitelik vektörleri düþük boyutlu bir alt uzaya
%dönüþtürülür, eigenfacesler oluþturulur
projectimg = [ ];  % matrisin izdüþümü
for i = 1 : size(eigenfaces,2)
    temp = eigenfaces' * A(:,i);
    projectimg = [projectimg temp];
end


%Test resminin PCA'ya göre düzenlenmesi
test_image = imread(testimg);
test_image = test_image(:,:,1);
[r c] = size(test_image);
temp = reshape(test_image',r*c,1); %test resminden (MxN)x1 vektor oluþturuluyor
temp = double(temp)-m; % ana vektör deðerlerinden satýr ortamamalarý çýkartýlýr
projtestimg = eigenfaces'*temp; %test resminin yüzdeki izdüþümü

%oklit uzaklýklarýnýn hesaplanýr ve test resmi ile databasedeki tüm
%resimler ile karþýlaþtýrýlýr.

euclide_dist = [ ];
for i=1 : size(eigenfaces,2)
    temp = (norm(projtestimg-projectimg(:,i)))^2;
    euclide_dist = [euclide_dist temp];
end
[euclide_dist_min recognized_index] = min(euclide_dist);
recognized_img = strcat(int2str(recognized_index),'.jpg');
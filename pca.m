function [recognized_img]=pca(datapath,testimg)


D = dir(datapath);
%resim say�s�n�n bulunmas�
imgcount = 0;
for i=1 : size(D,1)
    if not(strcmp(D(i).name,'.')|strcmp(D(i).name,'..')|strcmp(D(i).name,'Thumbs.db'))
        imgcount = imgcount + 1; 
    end
end
%resimlerin matrise yerle�tirilmesi
X=[];
for i = 1 : imgcount
    str = strcat(datapath,'\',int2str(i),'.jpg');
    img = imread(str);
    img = rgb2gray(img);
    [r c] = size(img);
    temp = reshape(img',r*c,1);%2D image 1D vekt�re d�n��t�r�l�r
                              
       X=[X , temp]; %temp X matrisisne eklenir (s�t�n eklenmesi) 
end

% kovaryans matrisinin bulunmas�
m = mean(X,2);% matrisin sat�r(2'nin anlam� 1 olsa s�tun ort. alacakt�) ortalamas�n� bulur
imgcount = size(X,2);

%Ortalamalar ��kart�larak A matrisi hesaplan�yor
A = [];
for i=1 : imgcount
    temp = double(X(:,i)) - m;
    A = [A temp];
end
L= A' * A; 
%V �zvekt�r natrisi D: �zde�er matrisi
[V,D]=eig(L); %eig �zde�er komutu
% w izd���m matrisi bulunmas�
L_eig_vec = [];
for i = 1 : size(V,2) 
    if( D(i,i) > 1 )
        L_eig_vec = [L_eig_vec V(:,i)];
    end
end

%eigenfaces bulunmas�
%Se�ilen �z vekt�rler, ��karma i�leminden elde edilen matris ile �arp�larak azalt�lm�� �z�y�zuzay� (eigenfaces)  elde edilir.
eigenfaces = A * L_eig_vec;

%w matrisi kullan�larak �znitelik vekt�rleri d���k boyutlu bir alt uzaya
%d�n��t�r�l�r, eigenfacesler olu�turulur
projectimg = [ ];  % matrisin izd���m�
for i = 1 : size(eigenfaces,2)
    temp = eigenfaces' * A(:,i);
    projectimg = [projectimg temp];
end


%Test resminin PCA'ya g�re d�zenlenmesi
test_image = imread(testimg);
test_image = test_image(:,:,1);
[r c] = size(test_image);
temp = reshape(test_image',r*c,1); %test resminden (MxN)x1 vektor olu�turuluyor
temp = double(temp)-m; % ana vekt�r de�erlerinden sat�r ortamamalar� ��kart�l�r
projtestimg = eigenfaces'*temp; %test resminin y�zdeki izd���m�

%oklit uzakl�klar�n�n hesaplan�r ve test resmi ile databasedeki t�m
%resimler ile kar��la�t�r�l�r.

euclide_dist = [ ];
for i=1 : size(eigenfaces,2)
    temp = (norm(projtestimg-projectimg(:,i)))^2;
    euclide_dist = [euclide_dist temp];
end
[euclide_dist_min recognized_index] = min(euclide_dist);
recognized_img = strcat(int2str(recognized_index),'.jpg');
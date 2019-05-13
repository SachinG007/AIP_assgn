img = im2double(imread('cameraman.tif'));
imgo = img;
img = poissrnd(img*16);
img = img/max(img(:));

net_pretrained = denoisingNetwork('DnCNN');
load net1

rec_img_pret = denoiseImage(img,net_pretrained);

img = reshape(img,[256 256 1]);
rec_img_my = zeros(256,256,1);
for i=1:8
    for j=1:8
        start_x = (i-1)*32 + 1;
        start_y = (j-1)*32 + 1;
        sub_img = img(start_x:start_x+31,start_y:start_y+31,:);
        rec_sub_img = predict(net1,sub_img);
        rec_img_my(start_x:start_x+31,start_y:start_y+31,:) = rec_sub_img;
    end
end

imList = [];
imList = [imList imgo];
imList = [imList img];
imList = [imList rec_img_pret ];
imList = [imList rec_img_my];
montage(imList);

rmse_pretrained = norm(imgo - rec_img_pret)/norm(imgo)
rmse_mynet = norm(imgo-rec_img_my)/norm(imgo)
ssim_pretrained = ssim(imgo,rec_img_pret)
ssim_mynet = ssim(imgo,rec_img_my)


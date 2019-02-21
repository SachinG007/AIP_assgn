rng(1) %seed set
%Read Input
input = double(imread('barbara256.png'));

input = padarray(input,[8,8]);
oinput = input;
[h,w] = size(input);
%Add noiseclear
noise = normrnd(0,2,[h,w]);
% noise = noise - min(min(noise));
% noise = noise/max(max(noise));
noise = noise * 20;
input = input + noise;
input = input - min(min(input));
input = input/max(max(input));
input = input * 255;
% J = imnoise(input,'gaussian',0,0.02)
subplot(1,2,1);
imshow(mat2gray(input));
%%
%Generate phi matrix
phi = eye(32,64);
U = kron(dctmtx(8)',dctmtx(8)');
A = phi * U;

%%
final_img = zeros(h,w);

for i=1:h-8
    i
    for j = 1:w-8
        %Generate yi 
        start_x = i;
        start_y = j;
        sub_img = input(start_x:start_x+7,start_y:start_y+7);
        v_sub_img = reshape(sub_img',[8*8,1]);
        y = phi * v_sub_img; %this acts as the input to the reconstruction algo
        
        %Reconstruction
        result = ista(y,A);
        result = U * result;
        result = reshape(result,[8,8])';
        
        final_img(start_x:start_x+7,start_y:start_y+7) = final_img(start_x:start_x+7,start_y:start_y+7) + result(:,:);
    
    end
end
%%
final_img = final_img/64.0;
subplot(1,2,1);
imshow(mat2gray(input(:,:)));
title('Noisy Input Image std_dev=2')
subplot(1,2,2);
imshow(mat2gray(final_img(:,:)));
title('Output of ISTA')    
%%
rmse = norm(oinput - final_img)/norm(oinput)

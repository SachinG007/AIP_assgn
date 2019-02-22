rng(1) %seed set
%Read Input
input = double(imread('barbara256.png'));
input = padarray(input,[8,8]);
[h,w] = size(input)
%Add noise
% noise = normrnd(0,2,[h,w]);
% input = input + noise;
%%
%Generate phi matrix
phi = normrnd(0,1,[32,64]);
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
subplot(2,1,1);
imshow(mat2gray(input(:,:)));
title('Original Image')
subplot(2,1,2);
imshow(mat2gray(final_img(:,:)));
title('Reconstructed CS Image')    
%%
rmse = norm(input - final_img)/norm(input)

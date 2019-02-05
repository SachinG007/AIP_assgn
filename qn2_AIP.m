%Read input 
ep = .128;
inp_video = VideoReader('cars.mp4');
%Read only first t frames
t = 7;
inp_video_frame = read(inp_video,[1,t]);
[h,w,chan,t] = size(inp_video_frame)
frame = zeros(h,w,t);
%Extract frames and convert to gray
for j = 1:t
    frame(:,:,j) = im2double(rgb2gray(inp_video_frame(:,:,:,j)));
%     subplot(1,4,j);
%     imshow(frame(:,:,j));
end


frame = padarray(frame,[8,8]);
[h,w,t] = size(frame)
%Generate code pattern
rng(1) %seed set
code_pattern = randi([0, 1], [h,w,t]);

%Generate the coded snapshot
code_snap = zeros(h,w);

for j = 1:t
    a = frame(:,:,j);
    code_snap = code_snap + code_pattern(:,:,j).*a;
end


 %title('Snap without noise');
%Add noise to the coded snap
rng(1)
noise = normrnd(0,0.008,[h,w]); %calculate as same percent of 
%2 when image is on scale of 0-255
code_snap = code_snap + noise;
subplot(1,1,1)
imshow(im2double(code_snap));
%title('Noise Added Coded Snap');


%%
%Perform Reconstruction
%8*8 Submatrix 
% code_snap = padarray(code_snap,[8 8]);
% [h,w] = size(code_snap);
iDCTm = dctmtx(8*8*t)';
final_img = zeros(h,w,t);
sub_pattern = zeros(8,8,t);
v_sub_pattern = zeros(8*8,t);

for i=100:290%10:h-10%(h-8)/2 + 1
    i
    for j = 100:350%10:w-10%(w-8)/2 + 1
        j;
        A = [];
        start_x = i;%1 + (i-1)*2;
        start_y = j;%1 + (j-1)*2;
        sub_snap = code_snap(start_x:start_x+7,start_y:start_y+7);
        v_sub_snap = reshape(sub_snap',[8*8,1]);
        for k = 1:t
            sub_pattern(:,:,k) = code_pattern(start_x:start_x+7,start_y:start_y+7,k);
            v_sub_pattern(:,k) = reshape(sub_pattern(:,:,k)',[8*8,1]);
            B = diag(v_sub_pattern(:,k));
            A = [A B];
        end
        A = A*iDCTm;
        size(A);
        
        result = omp(A,v_sub_snap,ep);
        result = iDCTm * result;
        result = reshape(result,[8,8,t]);
        for p=1:t
            final_img(start_x:start_x+7,start_y:start_y+7,p) = final_img(start_x:start_x+7,start_y:start_y+7,p) + result(:,:,p)';
        end
        %size(fina)
    end
end
%%
final_img = final_img/64.0;

X = frame(110:280,110:340,:);
Y = final_img(110:280,110:340,:);
%%
subplot(3,2,2);
imshow(Y(:,:,2));
subplot(3,2,4);
imshow(Y(:,:,4));
subplot(3,2,6);
imshow(Y(:,:,6));

subplot(3,2,1);
imshow(X(:,:,2));
subplot(3,2,3);
imshow(X(:,:,4));
subplot(3,2,5);
imshow(X(:,:,6));


% X = frame(100:300,100:350,:);
% Y = final_img(100:300,100:350,:);
err = immse(X,Y)
fprintf('Mean Squared Reconstruction Error is : %d.\n', err)
peaksnr = psnr(Y,X)
fprintf('PSNR : %d.\n', peaksnr)

                
        
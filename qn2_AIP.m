%Read input 
ep = 5;
inp_video = VideoReader('cars.mp4');
%Read only first t frames
t = 3;
inp_video_frame = read(inp_video,[1,t]);
[h,w,chan,t] = size(inp_video_frame)
frame = zeros(h,w,t);
%Extract frames and convert to gray
for j = 1:t
    frame(:,:,j) = im2double(rgb2gray(inp_video_frame(:,:,:,j)));
    subplot(1,4,j);
    imshow(frame(:,:,j));
end

frame = padarray(frame,[8,8]);
[h,w,t] = size(frame)
%Generate code pattern
rng(1) %seed set
code_pattern = randi([0, 1], [h,w,3]);

%Generate the coded snapshot
code_snap = zeros(h,w);

for j = 1:t
    a = frame(:,:,j);
    code_snap = code_snap + code_pattern(:,:,j).*a;
end


 %title('Snap without noise');
%Add noise to the coded snap
rng(1)
noise = normrnd(0,2,[h,w])/255;
code_snap = code_snap + noise;
subplot(2,1,2)
imshow(code_snap);
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

for i=10:h-10%(h-8)/2 + 1
    for j = 10:w-10%(w-8)/2 + 1
        i;
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
%         result = omp2(eps,A,v_sub_snap);
        result = iDCTm * result;
        result = reshape(result,[8,8,t]);
        for p=1:3
            final_img(start_x:start_x+7,start_y:start_y+7,p) = final_img(start_x:start_x+7,start_y:start_y+7,p) + result(:,:,p)';
        end
        %size(fina)
    end
end
final_img = final_img/64.0;
subplot(2,1,1);
imshow(final_img(:,:,1));
                
        
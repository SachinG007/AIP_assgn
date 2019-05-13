digitDatasetPath = fullfile(matlabroot,'toolbox','nnet', ...
    'nndemos','nndatasets','DigitDataset');
imds = imageDatastore(digitDatasetPath, ...
    'IncludeSubfolders',true, ...
    'LabelSource','foldernames');
%should be power of 2 generally
imds.ReadSize = 512;
rng(0)
imds = shuffle(imds);
[imdsTrain,imdsVal,imdsTest] = splitEachLabel(imds,0.9,0.1);


dsTrain = combine(imdsTrain,imdsTrain);
dsVal = combine(imdsVal,imdsVal);
dsTest = combine(imdsTest,imdsTest);

dsTraino = transform(dsTrain,@commonPreprocessing);
dsValo = transform(dsVal,@commonPreprocessing);
dsTesto = transform(dsTest,@commonPreprocessing);

dsTrain,orgTrain = transform(dsTraino,@addNoise);
dsVal,orgVal = transform(dsValo,@addNoise);
dsTest,orgTest = transform(dsTesto,@addNoise);

dsTrain = transform(dsTrain,@augmentImages);


load net1
ypred = predict(net1,dsTest);
%%
inputImageExamples = preview(dsTest);

for i=1:8
    ref = inputImageExamples{i,2};
    imList = [];
    imList = [imList ref];
    imList = [imList inputImageExamples{i}];
    imList = [imList ypred(:,:,:,i)];
    list{i} = imList;
    
    %rmse cal
    e1 = norm(ref-inputImageExamples{i})/norm(ref);
    e2 = norm(ref-ypred(:,:,:,i))/norm(ref);
    ssim1 = ssim(ref,inputImageExamples{i});
    ssim2 = ssim(ref,ypred(:,:,:,i));
    noisy_rmse_list{i} = e1;
    rec_rmse_list{i} = e2;
    noisy_ssim_list{i} = ssim1;
    rec_ssim_list{i} = ssim2;
end

%%
montage(list,'Size',[8,1]);
noisy_rmse_list
rec_rmse_list
noisy_ssim_list
rec_ssim_list

%%
net1.Layers
%%
%collage of filters
layers = [2 4 7 9 ];
channels = 1:8;

for layer = layers
    I = deepDreamImage(net,layer,channels, ...
        'Verbose',false, ...
        'PyramidLevels',1);
    
    figure
    I = imtile(I,'ThumbnailSize',[128 128]);
    imshow(I)
    name = net.Layers(layer).Name;
    title(['Layer ',name,' Features'])
end

%%
layers = [12 14 16 19 21 ];
channels = 1:4;

for layer = layers
    I = deepDreamImage(net,layer,channels, ...
        'Verbose',false, ...
        'PyramidLevels',1);
    
    figure
    I = imtile(I,'ThumbnailSize',[128 128]);
    imshow(I)
    name = net.Layers(layer).Name;
    title(['Layer ',name,' Features'])
end

%%
layers = [23 25 27 29  ];
channels = 1:8;

for layer = layers
    I = deepDreamImage(net,layer,channels, ...
        'Verbose',false, ...
        'PyramidLevels',1);
    
    figure
    I = imtile(I,'ThumbnailSize',[128 128]);
    imshow(I)
    name = net.Layers(layer).Name;
    title(['Layer ',name,' Features'])
end


%%
ref = inputImageExamples{1,2};
layers = [2 4 7 9 12 14 16 19 21 23 25 27 29 ];
for layer = layers
    
    act1 = activations(net,ref,net.Layers(layer).Name);
    sz = size(act1);
    act1 = reshape(act1,[sz(1) sz(2) 1 sz(3)]);
    I = imtile(mat2gray(act1));
    figure
    imshow(I)

end
'hi'

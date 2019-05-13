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
dsVal = dsValNoisy;
dsTest = dsTestNoisy;
%%

net_pretrained = denoisingNetwork('DnCNN');
load net1

inputImageExamples = preview(dsTest);
for i=1:8
    img = inputImageExamples{i};
    newImg = cat(2,img,img);
    newImg = cat(1,newImg,newImg);
    B = denoiseImage(newImg,net_pretrained);
    
    rec_img_pret = B(1:32,1:32);
    rec_img_my = predict(net1,img);
    
    ref = inputImageExamples{i,2};
    imList = [];
    imList = [imList ref];
    imList = [imList inputImageExamples{i}];
    imList = [imList rec_img_pret ];
    imList = [imList rec_img_my];
    list{i} = imList;
    
    %rmse cal
    e1 = norm(ref-rec_img_pret)/norm(ref);
    e2 = norm(ref-rec_img_my)/norm(ref);
    ssim1 = ssim(ref,rec_img_pret);
    ssim2 = ssim(ref,rec_img_my);
    pretrained_rmse_list{i} = e1;
    my_rmse_list{i} = e2;
    pretrained_ssim_list{i} = ssim1;
    my_ssim_list{i} = ssim2;
end
%%
montage(list,'Size',[8,1]);
pretrained_rmse_list
my_rmse_list
pretrained_ssim_list
my_ssim_list

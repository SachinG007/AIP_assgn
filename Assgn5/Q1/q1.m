digitDatasetPath = fullfile(matlabroot,'toolbox','nnet', ...
    'nndemos','nndatasets','DigitDataset');
imds = imageDatastore(digitDatasetPath, ...
    'IncludeSubfolders',true, ...
    'LabelSource','foldernames');
%%
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
aa = 1
%%
exampleData = preview(dsTrain);
inputs = exampleData(:,1);
responses = exampleData(:,2);
minibatch = cat(2,inputs,responses);

montage(minibatch','Size',[8 2])
title('Inputs (Left) and Responses (Right)')

imageLayer = imageInputLayer([32,32,1]);

%using the network inspired by VGG net for getting the encoding sapce
%it has 5 sets of CNNs
%first 2 sets have 2CNN layers each
%last 3 sets have 3 CNN layers each
%since VGG net gives good encodings,I am using it here for encoder in hope
%of getting better latent space representation
encodingLayers = [ ...
    convolution2dLayer(3,32,'Padding','same','Name','conv11'), ...
    reluLayer, ...
    convolution2dLayer(3,16,'Padding','same','Name','conv12'), ...
    reluLayer, ...    
    maxPooling2dLayer(2,'Padding','same','Stride',2), ...
    convolution2dLayer(3,8,'Padding','same','Name','conv21'), ...
    reluLayer, ...
    convolution2dLayer(3,8,'Padding','same','Name','conv22'), ...
    reluLayer, ...
    maxPooling2dLayer(2,'Padding','same','Stride',2), ... 
    convolution2dLayer(3,4,'Padding','same','Name','conv31'), ...
    reluLayer, ...
    convolution2dLayer(3,4,'Padding','same','Name','conv32'), ...
    reluLayer, ...
    convolution2dLayer(3,4,'Padding','same','Name','conv33'), ...
    reluLayer, ...
    maxPooling2dLayer(2,'Padding','same','Stride',2)];

%adding conv layers also along with deconv to maintain kind of same
%structure, cant add exact same amount of conv layers as in encoder because
%trarining time increases
decodingLayers = [ ...
    createUpsampleTransponseConvLayer(2,4), ...
    reluLayer, ...
    convolution2dLayer(3,4,'Padding','same','Name','dconv3'), ...
    reluLayer, ...
    createUpsampleTransponseConvLayer(2,8), ...
    reluLayer, ...
    convolution2dLayer(3,8,'Padding','same','Name','dconv2'), ...
    reluLayer, ...
    createUpsampleTransponseConvLayer(2,16), ...
    reluLayer, ...
    convolution2dLayer(3,16,'Padding','same','Name','dconv2'), ...
    reluLayer, ...
    convolution2dLayer(3,1,'Padding','same'), ...
    clippedReluLayer(1.0), ...
    regressionLayer];       

layers = [imageLayer,encodingLayers,decodingLayers];
options = trainingOptions('adam', ...
    'MaxEpochs',150, ...
    'MiniBatchSize',imds.ReadSize, ...
    'ValidationData',dsVal, ...
    'Shuffle','never', ...
    'Plots','training-progress', ...
    'Verbose',false);

net = trainNetwork(dsTrain,layers,options);

%%
net1 = net;
 save net1
inputImageExamples = preview(dsTest);
montage({inputImageExamples{1},ypred(:,:,:,1)});

%%
function [dataOut orgDt] = addNoise(data)

orgDt = data;
dataOut = data;
for idx = 1:size(data,1)
   dataOut{idx} = poissrnd(dataOut{idx}*16);
   dataOut{idx} = dataOut{idx}/max(dataOut{idx}(:));
   
end
end

function dataOut = augmentImages(data)

dataOut = cell(size(data));
for idx = 1:size(data,1)
    rot90Val = randi(4,1,1)-1;
    dataOut(idx,:) = {rot90(data{idx,1},rot90Val),rot90(data{idx,2},rot90Val)};
end
end

function dataOut = commonPreprocessing(data)

dataOut = cell(size(data));
for col = 1:size(data,2)
    for idx = 1:size(data,1)
        temp = single(data{idx,col});
        temp = imresize(temp,[32,32]);
        temp = rescale(temp);
        dataOut{idx,col} = temp;
    end
end
end

function out = createUpsampleTransponseConvLayer(factor,numFilters)

filterSize = 2*factor - mod(factor,2); 
cropping = (factor-mod(factor,2))/2;
numChannels = 1;

out = transposedConv2dLayer(filterSize,numFilters, ... 
    'NumChannels',numChannels,'Stride',factor,'Cropping',cropping);
end
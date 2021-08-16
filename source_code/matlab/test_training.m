clear all
close all
%gpuDevice(1)
load('../new_dataset/waveforms/dataset_2G_n_multi_mobile_app_multi_channel_100k.mat')

OriginalXTrain=X;
catnames = {'mgmt','ctr','qos'};
Y=cell2mat(Y);
OriginalYTrain = discretize(Y,[0 1 2 3],'categorical',catnames);
OriginalYTrain=OriginalYTrain';

summaryClasses = tabulate(Y);
sizeMinClass = min(summaryClasses(:,2));
classes_labels = summaryClasses(:,1);
num_classes = numel(classes_labels);

indexes_balanced_classes = zeros(1,sizeMinClass*num_classes);
ind_0 = 1;
ind_1 = sizeMinClass;
for i = 1:numel(classes_labels)
    temp = find(Y==classes_labels(i));
    idx = randperm(size(temp,2),sizeMinClass);
    temp = temp(idx);
    indexes_balanced_classes(ind_0:ind_1)=temp;
    ind_0=ind_1+1;
    ind_1=ind_0+sizeMinClass-1;
end


%idx = randperm(size(OriginalXTrain,2),sizeMinClass);
%XTrain = OriginalXTrain(idx);
%YTrain = OriginalYTrain(idx);
XTrain = OriginalXTrain(indexes_balanced_classes);
YTrain = OriginalYTrain(indexes_balanced_classes);

tabulate(YTrain)

idx = randperm(size(XTrain,2),int32(0.3*size(XTrain,2)));
XValidation = XTrain(idx);
XTrain(idx) = [];
YValidation = YTrain(idx);
YTrain(idx) = [];

idx = randperm(size(XValidation,2),int32(0.5*size(XValidation,2)));
XTest = XValidation(idx);
XValidation(idx) = [];
YTest = YValidation(idx);
YValidation(idx) = [];

%XTrain = X;
%YTrain = Y;
figure
plot(XTrain{1}')
xlabel("Time Step")
title("Training Observation 1")
numFeatures = size(XTrain{1},1);
legend("Feature " + string(1:numFeatures),'Location','northeastoutside')

numObservations = numel(XTrain);
sequenceLengths=zeros(1,numObservations);
for i=1:numObservations
    sequence = XTrain{i};
    sequenceLengths(i) = size(sequence,2);
end

[sequenceLengths,idx] = sort(sequenceLengths);
XTrain = XTrain(idx);
YTrain = YTrain(idx);

figure
bar(sequenceLengths)
xlabel("Sequence")
ylabel("Length")
title("Sorted Data")
miniBatchSize = 10;

inputSize = numFeatures;
numHiddenUnits = 100;
numHiddenUnits1 = 50;
numClasses = num_classes;

layers = [ ...
    sequenceInputLayer(inputSize)
    lstmLayer(numHiddenUnits,'OutputMode','sequence')
    dropoutLayer(0.5)
    bilstmLayer(numHiddenUnits1,'OutputMode','last')
    dropoutLayer(0.3)
    fullyConnectedLayer(30)
    fullyConnectedLayer(numClasses)
    softmaxLayer
    classificationLayer];
% 
% inputSize = numFeatures;
% numHiddenUnits1 = 30;
% numHiddenUnits2 = 20;
% numClasses = num_classes;
% layers = [ ...
%     sequenceInputLayer(inputSize)
%     lstmLayer(numHiddenUnits1,'OutputMode','sequence')
%     dropoutLayer(0.2)
%     lstmLayer(numHiddenUnits2,'OutputMode','last')
%     dropoutLayer(0.2)
%     fullyConnectedLayer(numClasses)
%     softmaxLayer
%     classificationLayer];

maxEpochs = 50;

val_freq = floor(numel(YTrain)/miniBatchSize);
options = trainingOptions('adam', ...
    'ExecutionEnvironment','gpu', ...
    'GradientThreshold',1, ...
    'MaxEpochs',maxEpochs, ...
    'MiniBatchSize',miniBatchSize, ...
    'SequenceLength','shortest', ...
    'Verbose',1, ...
    'ValidationData', {XValidation,YValidation}, ...
    'ValidationFrequency', val_freq, ...
    'InitialLearnRate', 0.01, ...
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropFactor', 0.1, ...
    'LearnRateDropPeriod', 5, ...    
    'Plots','training-progress');

net = trainNetwork(XTrain,YTrain,layers,options);

YPred = classify(net,XTest, ...
    'MiniBatchSize',miniBatchSize, ...
    'SequenceLength','shortest');

acc = sum(YPred == YTest)./numel(YTest);
pc=plotconfusion(YTest,YPred);

[m,order] = confusionmat(YTest,YPred)
figure
cm = confusionchart(m,order);
clc;
clear;
close all;
%% GP regression model for the Dense and Sparse data
disp(' We will implement Deep Neural network for the Tauth model' );
disp( 'Author: Clement Etienam' )
disp( 'Supervisor: Professor Kody Law' )
disp( 'DNN modelling' )
tic
load('JM_tauth_data')
input=[r0,a0,kappa,delta,ip,b0,nebar,zeff,ploss];
inputtrain=log(input(1:2100,:)); %select the first 2100 for training
[m,n]=size(inputtrain);
inputtest=log(input(2101:end,:)); %use the remaining data for testing
output=[tauth];
outputtrain=log(output(1:2100,:)); %select the first 2100 for training
outputtest=log(output(2101:end,:)); %use the remaining data for testing


layers = [
    imageInputLayer([n m 1])

    convolution2dLayer(3,8,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    averagePooling2dLayer(2,'Stride',2)

    convolution2dLayer(3,16,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    averagePooling2dLayer(2,'Stride',2)
  
    convolution2dLayer(3,32,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    dropoutLayer(0.2)
    fullyConnectedLayer(2100)
    regressionLayer];


miniBatchSize  = 128;
validationFrequency = floor(numel(outputtrain)/miniBatchSize);
options = trainingOptions('adam', ...
    'MiniBatchSize',miniBatchSize, ...
    'MaxEpochs',30, ...
    'InitialLearnRate',1e-3, ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropFactor',0.1, ...
    'LearnRateDropPeriod',20, ...
    'Shuffle','every-epoch', ...
    'ValidationData',{inputtrain',outputtrain'}, ...
    'ValidationFrequency',validationFrequency, ...
    'Plots','training-progress', ...
    'Verbose',true);

net = trainNetwork(inputtrain',outputtrain',layers,options);
m = predict(net,inputtest);

ypredtest=m';
figure()
subplot(2,2,1)
plot(outputtest,exp(ypredtest),'o');hold;
xlabel('Real output','FontName','Helvetica', 'Fontsize', 13);
ylabel('GP estimate','FontName','Helvetica', 'Fontsize', 13)
title('JMtauth-RM-Dense','FontName','Helvetica', 'Fontsize', 13);
set(gca, 'FontName','Helvetica', 'Fontsize', 13)
set(gcf,'color','white')

%% Compute L2 and R^2 for the predicted and actual data (dense/spatial)
%For the JMtauth-RM data
Lerror=(norm(outputtest-exp(ypredtest))/norm(outputtest))^0.5;
L_2spatial=1-(Lerror^2);
%Coefficient of determination
for i=1:numel(outputtest)
    outputreq(i)=outputtest(i)-mean(outputtest);
end

outputreq=outputreq';
CoDspatial=1-(norm(outputtest-exp(ypredtest))/norm(outputreq));
CoDspatial=1 - (1-CoDspatial)^2 ;



toc
% poolobj = gcp('nocreate');
% delete(poolobj);
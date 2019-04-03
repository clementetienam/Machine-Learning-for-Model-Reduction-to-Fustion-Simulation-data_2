% demo
clc;
clear all;
close all;
dd=importdata('chi_itg.dat',' ',1);
test=dd.data;
out=test;
outgp=test;
test=out;
%X=zscore(test(1:600000,1:10));
X=log(test(1:600000,1:10));
y=(test(1:600000,11));
outputtest=y(290000+1:end,:);
y2=zeros(600000,1);
for i=1:600000
    if y(i)==0
        y2(i)=0;
    end
    
    if y(i)>0
        y2(i)=1;
    end
        
end
y=y2;
inputtrainclass=X(1:290000,:);
outputtrainclass=y(1:290000,:);
% [model, llh] = logitBin(inputtrainclass,outputtrainclass);
% plot(llh);
% y = logitBinPred(model,inputtrainclass);

rng default
Mdl = fitclinear(inputtrainclass,outputtrainclass,...
    'OptimizeHyperparameters','auto','HyperparameterOptimizationOptions',...
    struct('AcquisitionFunctionName','expected-improvement-plus'));
labelDA = predict(Mdl,inputtrainclass);
EVALdaal = Evaluate(outputtrainclass,labelDA);
disp(['accuracy of the classifier is  = ' num2str(EVALdaal(:,1))]);













%% Logistic logistic regression for binary classification
% clear; close all;
% k = 2;
% n = 1000;
% [X,t] = kmeansRnd(2,k,n);
% [model, llh] = logitBin(X,t-1);
% plot(llh);
% y = logitBinPred(model,X)+1;
% binPlot(model,X,y)
%% Logistic logistic regression for multiclass classification
% clear
% k = 3;
% n = 1000;
% [X,t] = kmeansRnd(2,k,n);
% [model, llh] = logitMn(X,t);
% y = logitMnPred(model,X);
% plotClass(X,y)
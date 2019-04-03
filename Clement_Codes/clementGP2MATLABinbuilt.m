clc;
clear;
close all;
disp( 'Author: Clement Etienam' )
disp( 'Supervisor: Professor Kody Law' )
disp( 'GP Regression Model using MATLAB functions' )
load('JM_tauth_data')
input=[r0,a0,kappa,delta,ip,b0,nebar,zeff,ploss];
inputtrain=log(input(1:2100,:)); %select the first 2100 for training
inputtest=log(input(2101:end,:)); %use the remaining data for testing
output=[tauth];
outputtrain=log(output(1:2100,:)); %select the first 2100 for training
outputtest=log(output(2101:end,:)); %use the remaining data for testing
P=size(inputtest,1);
sigmaL0 = sqrt(0.2); % Length scale for predictors
%sigmaL0=sigmaL0(1:8,:);
sigmaF0 = (0.01); % Signal standard deviation
sigmaN0 = (0.01); % Initial noise standard deviation, nugget

opts = statset('fitrgp'); % relative gradient norm
opts.TolFun = 1e-2;
gprMdl = fitrgp(inputtrain,outputtrain,'KernelFunction','squaredexponential','Verbose',1, ...
    'Optimizer','quasinewton','OptimizeHyperparameters','all','OptimizerOptions',opts, ...
    'KernelParameters',[sigmaL0;sigmaF0],'Sigma',sigmaN0,'BasisFunction','linear','HyperparameterOptimizationOptions',...
     struct('AcquisitionFunctionName','expected-improvement-plus'));


%% Output the learned lenght scales
sigmaL = gprMdl.KernelInformation.KernelParameters(1:end-1); % Learned length scales


%ypred4 = resubPredict(gprMdl4);
ypredtest = predict(gprMdl,inputtest); % predict the new output with the input data not seen by the GP

%% Plot the predicted output data with the true output data
figure()
plot(outputtest,ypredtest,'o');hold;
xlabel('Real output','FontName','Helvetica', 'Fontsize', 13);
ylabel('GP estimate','FontName','Helvetica', 'Fontsize', 13)
title('JMtauth-RM','FontName','Helvetica', 'Fontsize', 13);
set(gca, 'FontName','Helvetica', 'Fontsize', 13)
set(gcf,'color','white')



%% Compute L2 and R^2 for the predicted and actual data
%For the JMtauth-RM data
Lerror=(norm(outputtest-ypredtest)/norm(outputtest))^0.5;
L_2=1-(Lerror^2);
%Coefficient of determination
for i=1:numel(outputtest)
    outputreq(i)=outputtest(i)-mean(outputtest);
end

outputreq=outputreq';
CoD=1-(norm(outputtest-ypredtest)/norm(outputreq));
CoD=1 - (1-CoD)^2 ;
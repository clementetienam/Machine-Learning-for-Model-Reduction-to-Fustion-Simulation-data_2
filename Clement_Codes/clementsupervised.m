%----------------------------------------------------------------------------------
% Supervised learning
% Compare various supervised learning algorithm
% Author: Clement Etienam ,PhD Petroleum Engineering 2015-2018
% Supervisor:Professor Kody Law
%-----------------------------------------------------------------------------------
%% 
clc;
clear;
close all;

disp(' We will implement the dense and sparse GP for the Tauth model' );
disp( 'Author: Clement Etienam' )
disp( 'Supervisor: Professor Kody Law' )
disp( 'serveral supervised machine learning' )
N22=input( ' Given prior information to the data Ascii How many rows do you want as training  '); %

oldfolder=cd;
cd(oldfolder) ;

%disp( 'create the folders')
load('jm_data.mat')
output=[ptotped, betanped, wped];
input=[r a kappa delta bt ip neped betan zeffped];
Xin=log(input);
[rowinput,columninput]=size(Xin);
yin=log(output);
columnoutput=size(yin,2);

Xtest=Xin(1:N22,:);
ytest=yin(1:N22,:);

Xpred=Xin(N22+1:end,:);
ypred=exp(yin(N22+1:end,:));

%% Use artificial neural network
parpool
   trainFcn = 'trainlm';
  hiddenLayerSize = 100; %number of hidden layer neurons
    net = fitnet(hiddenLayerSize,trainFcn); %create a fitting network
     net.divideParam.trainRatio = 70/100;  %use 70% of data for training 
    net.divideParam.valRatio = 15/100;  %15% for validation
    net.divideParam.testRatio = 15/100; %15% for testing
    [net,tr] = train(net,Xtest',ytest'); % train the network
    yann = (net(Xpred'))';
figure()
for i=1:columnoutput
subplot(2,2,i)
plot(ypred(:,i),exp(yann(:,i)),'o');
xlabel('Real output','FontName','Helvetica', 'Fontsize', 13);
ylabel('ANN estimate','FontName','Helvetica', 'Fontsize', 13)
title (sprintf('output %d (for Artificial Neural Network) ',i))
%title('JMdata','FontName','Helvetica', 'Fontsize', 13);
set(gca, 'FontName','Helvetica', 'Fontsize', 13)
set(gcf,'color','white')
end 
for i=1:3
    EWOP1=immse(ypred(:,i),exp(yann(:,i)));
    EWOP1=EWOP1./std(ypred(:,i));
    Errora(:,i)=EWOP1;
end
Eann=sum(Errora);

 %% Train using SVM regression
 for i=1:3
 rng default
Mdl = fitrsvm(Xtest,ytest(:,i),'OptimizeHyperparameters','auto',...
    'HyperparameterOptimizationOptions',struct('AcquisitionFunctionName',...
    'expected-improvement-plus'));
ysvm = predict(Mdl,Xpred);
ysvmm(:,i)=ysvm;
 end
figure()
for i=1:columnoutput
subplot(2,2,i)
plot(ypred(:,i),exp(ysvmm(:,i)),'o');
xlabel('Real output','FontName','Helvetica', 'Fontsize', 13);
ylabel('SVM estimate','FontName','Helvetica', 'Fontsize', 13)
title (sprintf('output %d (for support vector) ',i))
%title('JMdata','FontName','Helvetica', 'Fontsize', 13);
set(gca, 'FontName','Helvetica', 'Fontsize', 13)
set(gcf,'color','white')
end 
 for i=1:3
    EWOP1=immse(ypred(:,i),exp(ysvmm(:,i)));
    EWOP1=EWOP1./std(ypred(:,i));
    Errorsvm(:,i)=EWOP1;
end
Esvm=sum(Errorsvm);
%% Use radial basis function
spread=.4;
netrbb = newrbe(Xtest',ytest',spread);
yrb = netrbb(Xpred')';

figure()
for i=1:columnoutput
subplot(2,2,i)
plot(ypred(:,i),exp(yrb(:,i)),'o');
xlabel('Real output','FontName','Helvetica', 'Fontsize', 13);
ylabel('Radial basis estimate','FontName','Helvetica', 'Fontsize', 13)
title (sprintf('output %d for radial basis function) ',i))
%title('JMdata','FontName','Helvetica', 'Fontsize', 13);
set(gca, 'FontName','Helvetica', 'Fontsize', 13)
set(gcf,'color','white')
end 
 for i=1:3
    EWOP1=immse(ypred(:,i),exp(yrb(:,i)));
    EWOP1=EWOP1./std(ypred(:,i));
    Errorrb(:,i)=EWOP1;
end
Erb=sum(Errorrb);
%% Use cnb
% for i=1:3
%   Mcnb = fitcnb(Xtest,ytest(:,i));
%   ycnb = (predict(Mcnb,Xpred));
%   ycnbb(:,i)=ycnb;
% end
% figure()
% for i=1:3
% subplot(2,2,i)
% plot(ypred(:,i),ycnbb(:,i),'o');
% xlabel('Real output','FontName','Helvetica', 'Fontsize', 13);
% ylabel('cnb estimate','FontName','Helvetica', 'Fontsize', 13)
% title (sprintf('output %d vs input %d (for radial basis function) ',i,i))
% %title('JMdata','FontName','Helvetica', 'Fontsize', 13);
% set(gca, 'FontName','Helvetica', 'Fontsize', 13)
% set(gcf,'color','white')
% end 
%  
%%  %% Use KNN search tree
for i=1:3
  Mdlknn = fitcknn(Xtest,ytest(:,i),'NumNeighbors',3,...
     'NSMethod','exhaustive','Distance','minkowski',...
    'Standardize',1);
yknn = predict(Mdlknn,Xpred);
yknna(:,i)=yknn;
end
figure()
for i=1:3
subplot(2,2,i)
plot(ypred(:,i),exp(yknna(:,i)),'o');
xlabel('Real output','FontName','Helvetica', 'Fontsize', 13);
ylabel('knn estimate','FontName','Helvetica', 'Fontsize', 13)
title (sprintf('output %d(for KNN search tree) ',i))
%title('JMdata','FontName','Helvetica', 'Fontsize', 13);
set(gca, 'FontName','Helvetica', 'Fontsize', 13)
set(gcf,'color','white')
end 
 for i=1:columnoutput
    EWOP1=immse(ypred(:,i),exp(yknna(:,i)));
    EWOP1=EWOP1./std(ypred(:,i));
    Errorknn(:,i)=EWOP1;
end
Eknn=sum(Errorknn);
%% Use shallow neural network
%parpool
net = feedforwardnet(10);
net = train(net,Xtest',ytest','useParallel','yes','showResources','yes');
yshallow = net(Xpred')';
figure()
for i=1:3
subplot(2,2,i)
plot(ypred(:,i),exp(yshallow(:,i)),'o');
xlabel('Real output','FontName','Helvetica', 'Fontsize', 13);
ylabel('SNN estimate','FontName','Helvetica', 'Fontsize', 13)
title (sprintf('output %d(for Shallow Neural Network) ',i))
%title('JMdata','FontName','Helvetica', 'Fontsize', 13);
set(gca, 'FontName','Helvetica', 'Fontsize', 13)
set(gcf,'color','white')
end 
 for i=1:columnoutput
    EWOP1=immse(ypred(:,i),exp(yshallow(:,i)));
    EWOP1=EWOP1./std(ypred(:,i));
    Errorshallow(:,i)=EWOP1;
end
Eshl=sum(Errorshallow);
cd(oldfolder);
%% GP
meanfunc = [];                 % empty: don't use a mean function
covfunc = @covSEiso; hyp2.cov = log([9.5;12.5]); hyp2.lik = log(0.99);
disp('likfunc = @likGauss; sn = 0.1; hyp.lik = log(sn);')
likfunc = @likGauss; sn = 0.1; hyp2.lik = log(sn);
hyp2 = minimize2(hyp2, @gp, -100, @infGaussLik, [], covfunc, likfunc, Xtest, ytest);
[mGP, s2] = gp(hyp2, @infGaussLik, [], covfunc, likfunc, Xtest, ytest, Xpred);
figure()
for i=1:3
subplot(2,2,i)
plot(ypred(:,i),exp(mGP(:,i)),'o');
xlabel('Real output','FontName','Helvetica', 'Fontsize', 13);
ylabel('GP estimate','FontName','Helvetica', 'Fontsize', 13)
title (sprintf('output %d (for Gaussian Process) ',i))
%title('JMdata','FontName','Helvetica', 'Fontsize', 13);
set(gca, 'FontName','Helvetica', 'Fontsize', 13)
set(gcf,'color','white')
end 
 for i=1:columnoutput
    EWOP1=immse(ypred(:,i),exp(mGP(:,i)));
    EWOP1=EWOP1./std(ypred(:,i));
    ErrorGP(:,i)=EWOP1;
end
Egp=sum(ErrorGP);

%% Sparse GP
meanfunc=[];% empty: don't use a mean function
%hyp2.mean = [0.5; 1;5;0.5;0.5;0.5;0.5;0.5;0.5;0.5,;0.5];
%  fff=zeros(10,1);
%  fff(1:end)=normrnd(0,1,10,1);
% fff=fff';
%  hyp.mean=fff;
% hyp.mean=fff;
n = 30; sn = 0.99;

 lik = {@likGauss};    hyp2.lik = log(sn); inf = @infGaussLik;
 cov = {@covSEiso}; 
 %cov={@covMaternard ,3}; 
 hyp2.cov = log([9.5;12.5]); % Matern class d=5
p=size(Xtest,2);
for j=1:p
xu = normrnd(0,1,100,1); 
xsparse(:,j)=xu;
end
hyp2.xu=xsparse;
cov = {'apxSparse',cov,xsparse};           % inducing points
%parpool
infv  = @(varargin) inf(varargin{:},struct('s',1.0));
infr=@infFITC;
infe = @infFITC_EP;
hyp2 = minimize(hyp2,@gp,-100,infv,meanfunc,cov,lik,Xtest,ytest);%minimise the hyperparamters

[mGPsp, s2]  = gp(hyp2,infv,meanfunc,cov,lik,Xtest, ytest, Xpred);%Inference with GP

figure()
for i=1:3
subplot(2,2,i)
plot(ypred(:,i),exp(mGPsp(:,i)),'o');
xlabel('Real output','FontName','Helvetica', 'Fontsize', 13);
ylabel('GP(sparse) estimate','FontName','Helvetica', 'Fontsize', 13)
title (sprintf('output %d (for sparse Gaussian Process) ',i))
%title('JMdata','FontName','Helvetica', 'Fontsize', 13);
set(gca, 'FontName','Helvetica', 'Fontsize', 13)
set(gcf,'color','white')
end 
 for i=1:columnoutput
    EWOP1=immse(ypred(:,i),exp(mGPsp(:,i)));
    EWOP1=EWOP1./std(ypred(:,i));
    ErrorGPsp(:,i)=EWOP1;
end
Egpsp=sum(ErrorGPsp);
%% Train Deep neural netowk
% layers = [ ...
%     imageInputLayer([28 28 1])
%     convolution2dLayer(12,25)
%     reluLayer
%     fullyConnectedLayer(1)
%     regressionLayer];
% options = trainingOptions('sgdm', ...
%     'InitialLearnRate',0.001, ...
%     'Verbose',false, ...
%     'Plots','training-progress');
% net = trainNetwork(Xtest,ytest,layers,options);
% ydeep = predict(net,Xpred);

%% no hidden layers
net = feedforwardnet([]);
% % one hidden layer with linear transfer functions
% net = feedforwardnet([10]);
% net.layers{1}.transferFcn = 'purelin';

% set early stopping parameters
net.divideParam.trainRatio = 1.0; % training set [%]
net.divideParam.valRatio   = 0.0; % validation set [%]
net.divideParam.testRatio  = 0.0; % test set [%]
% train a neural network
net.trainParam.epochs = 200;
net = train(net,Xtest',ytest');
%---------------------------------

% view net
view (net)
% simulate a network over complete input range
ylinear = net(Xpred')';% Linear regression
figure()
for i=1:3
subplot(2,2,i)
plot(ypred(:,i),exp(ylinear(:,i)),'o');
xlabel('Real output','FontName','Helvetica', 'Fontsize', 13);
ylabel('Linear regression','FontName','Helvetica', 'Fontsize', 13)
title (sprintf('output %d (for Linear regression) ',i))
%title('JMdata','FontName','Helvetica', 'Fontsize', 13);
set(gca, 'FontName','Helvetica', 'Fontsize', 13)
set(gcf,'color','white')
end 
 for i=1:columnoutput
    EWOP1=immse(ypred(:,i),exp(ylinear(:,i)));
    EWOP1=EWOP1./std(ypred(:,i));
    ErrorLP(:,i)=EWOP1;
end
Elp=sum(ErrorLP);
%% Radial basis function
% choose a spread constant
spread = .2;
% choose max number of neurons
K = 40;
% performance goal (SSE)
goal = 0;
% number of neurons to add between displays
Ki = 5;
% create a neural network
net = newrb(Xtest',ytest',goal,spread,K,Ki);
%---------------------------------

% view net
view (net)
% simulate a network over complete input range
yrbf2 = net(Xpred')';
figure()
for i=1:3
subplot(2,2,i)
plot(ypred(:,i),exp(yrbf2(:,i)),'o');
xlabel('Real output','FontName','Helvetica', 'Fontsize', 13);
ylabel('Radial basis estimate','FontName','Helvetica', 'Fontsize', 13)
title (sprintf('output %d (for radial basis function) ',i))
%title('JMdata','FontName','Helvetica', 'Fontsize', 13);
set(gca, 'FontName','Helvetica', 'Fontsize', 13)
set(gcf,'color','white')
end 
 for i=1:columnoutput
    EWOP1=immse(ypred(:,i),exp(yrbf2(:,i)));
    EWOP1=EWOP1./std(ypred(:,i));
    Errorrbf2(:,i)=EWOP1;
end
Erbf2=sum(Errorrbf2);
%% GRNN
% choose a spread constant
spread = .12;
% create a neural network
net = newgrnn(Xtest',ytest',spread);
%---------------------------------

% view net
view (net)
% simulate a network over complete input range
ygrnn = net(Xpred')';
figure()
for i=1:3
subplot(2,2,i)
plot(ypred(:,i),exp(ygrnn(:,i)),'o');
xlabel('Real output','FontName','Helvetica', 'Fontsize', 13);
ylabel('GRNN estimate','FontName','Helvetica', 'Fontsize', 13)
title (sprintf('output %d (for Generalised regression network) ',i))
%title('JMdata','FontName','Helvetica', 'Fontsize', 13);
set(gca, 'FontName','Helvetica', 'Fontsize', 13)
set(gcf,'color','white')
end 
 for i=1:columnoutput
    EWOP1=immse(ypred(:,i),exp(ygrnn(:,i)));
    EWOP1=EWOP1./std(ypred(:,i));
    Errorgrnn(:,i)=EWOP1;
end
Egrn=sum(Errorgrnn);
%% Multi-layered perceptron
% create a neural network
net = feedforwardnet([12 6]);
% set early stopping parameters
net.divideParam.trainRatio = 1.0; % training set [%]
net.divideParam.valRatio   = 0.0; % validation set [%]
net.divideParam.testRatio  = 0.0; % test set [%]
% train a neural network
net.trainParam.epochs = 200;
net = train(net,Xtest',ytest');
%---------------------------------

% view net
view (net)
% simulate a network over complete input range
ymlp = net(Xpred')';
figure()
for i=1:3
subplot(2,2,i)
plot(ypred(:,i),exp(ymlp(:,i)),'o');
xlabel('Real output','FontName','Helvetica', 'Fontsize', 13);
ylabel('MLP estimate','FontName','Helvetica', 'Fontsize', 13)
title (sprintf('output %d (for multi-layered perceptron) ',i))
%title('JMdata','FontName','Helvetica', 'Fontsize', 13);
set(gca, 'FontName','Helvetica', 'Fontsize', 13)
set(gcf,'color','white')
end 
 for i=1:columnoutput
    EWOP1=immse(ypred(:,i),exp(ymlp(:,i)));
    EWOP1=EWOP1./std(ypred(:,i));
    ErrormLP(:,i)=EWOP1;
end
Emlp=sum(ErrormLP);
%% Polynomial regression
for j=1:3
reg=MultiPolyRegress(Xtest,ytest(:,j),3,'figure');

for i=1:size(Xpred,1)
NewDataPoint=Xpred(i,:);
NewScores=repmat(NewDataPoint,[length(reg.PowerMatrix) 1]).^reg.PowerMatrix;
EvalScores=ones(length(reg.PowerMatrix),1);
for ii=1:size(reg.PowerMatrix,2)
EvalScores=EvalScores.*NewScores(:,ii);
end
yhatNew=reg.Coefficients'*EvalScores ;
yhatNew=exp(yhatNew);
ypredpoly(i,:)=yhatNew;
end
yall(:,j)=ypredpoly;
end
figure()
for i=1:3
subplot(2,2,i)
plot(ypred(:,i),(yall(:,i)),'o');
xlabel('Real output','FontName','Helvetica', 'Fontsize', 13);
ylabel('Polynomial Regression estimate','FontName','Helvetica', 'Fontsize', 13)
title (sprintf('output %d (for Polynomial Regression) ',i))
%title('JMdata','FontName','Helvetica', 'Fontsize', 13);
set(gca, 'FontName','Helvetica', 'Fontsize', 13)
set(gcf,'color','white')
end 
 for i=1:columnoutput
    EWOP1=immse(ypred(:,i),(yall(:,i)));
    EWOP1=EWOP1./std(ypred(:,i));
    ErrormLPpoly(:,i)=EWOP1;
end
Emlppoly=sum(ErrormLPpoly);

%% standard linear regression
Theta=inv(Xtest'*Xtest)*Xtest'*ytest;
Ylinearpred=exp(Xpred*Theta);
figure()
for i=1:3
subplot(2,2,i)
plot(ypred(:,i),(Ylinearpred(:,i)),'o');
xlabel('Real output','FontName','Helvetica', 'Fontsize', 13);
ylabel('off-line Linear Regression estimate','FontName','Helvetica', 'Fontsize', 13)
title (sprintf('output %d (for off-line Linear Regression) ',i))
%title('JMdata','FontName','Helvetica', 'Fontsize', 13);
set(gca, 'FontName','Helvetica', 'Fontsize', 13)
set(gcf,'color','white')
end 
 for i=1:columnoutput
    EWOP1=immse(ypred(:,i),(Ylinearpred(:,i)));
    EWOP1=EWOP1./std(ypred(:,i));
    ErrormLPpoly(:,i)=EWOP1;
end
EmloffL=sum(ErrormLPpoly);
%% Compute L2 and R^2 for the predicted and actual data
disp(['Error using Arteficial Neural Network = ' num2str(Eann)]);
disp(['Error using Support vector regression = ' num2str(Esvm)]);
disp(['Error using radial basis function = ' num2str(Erb)]);
disp(['Error using KNN search treee = ' num2str(Eknn)]);
disp(['Error using shallow Neural Network = ' num2str(Eshl)]);
disp(['Error using gaussian process = ' num2str(Egp)]);
disp(['Error using  sparse gaussian process = ' num2str(Egpsp)]);
disp(['Error using linear regression = ' num2str(Elp)]);
disp(['Error using radial basis function 2 = ' num2str(Erbf2)]);
disp(['Error using Genaralized regression network = ' num2str(Egrn)]);
disp(['Error using multi layered perceptron = ' num2str(Emlp)]);
disp(['Error using polynomial regression = ' num2str(Emlppoly)]);
disp(['Error using off-line linear regression = ' num2str(EmloffL)]);
poolobj = gcp('nocreate');
delete(poolobj);
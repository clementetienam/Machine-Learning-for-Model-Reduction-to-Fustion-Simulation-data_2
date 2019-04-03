clc;
clear;
close all;
%% GP regression model for the Dense and Sparse data
disp(' We will implement the dense and sparse GP for the Tauth model' );
disp( 'Author: Dr Clement Etienam' )
disp( 'Supervisor: Professor Kody Law' )
disp( 'GP Regression Modelling' )
tic
load('JM_tauth_data')
input=[r0,a0,kappa,delta,ip,b0,nebar,zeff,ploss];
inputtrain=log(input(1:2100,:)); %select the first 2100 for training
inputtest=log(input(2101:end,:)); %use the remaining data for testing
output=[tauth];
outputtrain=log(output(1:2100,:)); %select the first 2100 for training
outputtest=(output(2101:end,:)); %use the remaining data for test
%% For online regression
inputtrain2=log(input(1:200,:)); %select the first 2100 for training
inputtnewpoints=log(input(201:2100,:)); %select the first 2100 for training

outputtrain2=log(output(1:200,:)); %select the first 2100 for training
outputtnewpoints=log(output(201:2100,:)); %select the first 2100 for training
%parpool
%meanfunc = @meanLinear ;                % empty: don't use a mean function
meanfunc=[];
%hyp2.mean = [0.5; 1;5;0.5;0.5;0.5;0.5;0.5;0.5;0.5,;0.5];
fff=zeros(9,1);
fff(1:end)=0.5;
%fff=fff';
% hyp2.mean=fff;
% hyp.mean=fff;

covfunc = @covSEiso; hyp2.cov = log([9.5;12.5]); hyp2.lik = log(0.99);
disp('likfunc = @likGauss; sn = 0.1; hyp.lik = log(sn);')
likfunc = @likGauss; sn = 0.99; hyp2.lik = log(sn);
hyp2 = minimize(hyp2, @gp, -100, @infGaussLik, meanfunc, covfunc, likfunc, inputtrain, outputtrain);
[m s2] = gp(hyp2, @infGaussLik, meanfunc, covfunc, likfunc, inputtrain, outputtrain, inputtest);
ypredtest=exp(m);



%% Compute L2 and R^2 for the predicted and actual data (dense/spatial)
%For the JMtauth-RM data
Lerror=(norm(outputtest-ypredtest)/norm(outputtest))^0.5;
L_2spatial=1-(Lerror^2);
%Coefficient of determination
for i=1:numel(outputtest)
    outputreq(i)=outputtest(i)-mean(outputtest);
end

outputreq=outputreq';
CoDspatial=1-(norm(outputtest-ypredtest)/norm(outputreq));
CoDspatial=1 - (1-CoDspatial)^2 ;

%% Sparse GP approximation
%a = 0.3; b = 1.2;             % underlying function
n = 30; sn = 0.99;
meanfunc=[];
 lik = {@likGauss};    hyp.lik = log(sn); inf = @infGaussLik;
 cov = {@covSEiso}; 
 hyp.cov = log([9.5;12.5]); 
 hyp.lik = log(sn);
%disp('likfunc = @likGauss; sn = 0.1; hyp.lik = log(sn);')
%likfunc = @likGauss; sn = 0.1; hyp.lik = log(sn);
%pkg load statistics
for j=1:9
xu = normrnd(0,1,100,1); 
xsparse(:,j)=xu;
end
hyp.xu=xsparse;
cov = {'apxSparse',cov,xsparse};           % inducing points
infv  = @(varargin) inf(varargin{:},struct('s',1.0));           % VFE, opt.s = 0
hyp = minimize(hyp,@gp,-100,infv,meanfunc,cov,lik,inputtrain,outputtrain);
[ymuv,ys2v] = gp(hyp,infv,meanfunc,cov,lik, inputtrain, outputtrain, inputtest);
ymuv=exp(ymuv);


Lerror=(norm(outputtest-ymuv)/norm(outputtest))^0.5;
L_2sparse=1-(Lerror^2);
%Coefficient of determination
for i=1:numel(outputtest)
    outputreq(i)=outputtest(i)-mean(outputtest);
end

outputreq=outputreq';
CoDsparse=1-(norm(outputtest-ymuv)/norm(outputreq));
CoDsparse=1 - (1-CoDsparse)^2 ;

%% Weight space view- Linear regression

LearnedB=(inputtrain'*inputtrain)\(inputtrain'*outputtrain);
Predicty=inputtest*LearnedB;

Predicty=exp(Predicty);


Lerror=(norm(outputtest-Predicty)/norm(outputtest))^0.5;
L_2weight=1-(Lerror^2);
%Coefficient of determination
for i=1:numel(outputtest)
    outputreq(i)=outputtest(i)-mean(outputtest);
end

outputreq=outputreq';
CoDweight=1-(norm(outputtest-Predicty)/norm(outputreq));
CoDweight=1 - (1-CoDweight)^2 ;
%% Online linear regression
inputtrain2=log(input(1:200,:)); %select the first 2100 for training
inputnewpoints=log(input(201:2100,:)); %select the first 2100 for training

outputtrain2=log(output(1:200,:)); %select the first 2100 for training
outputnewpoints=log(output(201:2100,:)); %select the first 2100 for training

Con1=pinv((inputtrain2'*inputtrain2)); % initial covariance
Conini=Con1;
Hess1=(inputtrain2'*inputtrain2);
thetaon1=(Con1*inputtrain2')*outputtrain2;
%thetaini=thetatot;

for i=1:size(inputnewpoints,1);
    a=inputnewpoints(i,:);
    b=outputnewpoints(i,:);
    Kalman=Con1*(a')*pinv((1+a*(Con1)*a'));
    put=Kalman;
    Kalmanall(:,i)=put;
    Connew=(eye(size(inputnewpoints,2))-(Kalman*a))*Con1;
    put2=reshape(Connew,[],1);
    Connal(:,i)=put2;
    thetanew=((eye(size(inputnewpoints,2))-(Kalman*a))*thetaon1)+(Connew*a'*b);
    put3=thetanew;
    thetal(:,i)=put3;
    thetaon1=thetanew;
    Con1=Connew;
end

zz=exp(inputtest*thetaon1);
Lerror=(norm(outputtest-zz)/norm(outputtest))^0.5;
L_2zz=1-(Lerror^2);
%Coefficient of determination
for i=1:numel(outputtest)
    outputreq(i)=outputtest(i)-mean(outputtest);
end

outputreq=outputreq';
CoDpoly=1-(norm(outputtest-zz)/norm(outputreq));
CoDzz=1 - (1-CoDpoly)^2 ;

%% Polynomial Regression
reg=MultiPolyRegress(inputtrain,outputtrain,3,'figure');

for i=1:size(inputtest,1)
NewDataPoint=inputtest(i,:);
NewScores=repmat(NewDataPoint,[length(reg.PowerMatrix) 1]).^reg.PowerMatrix;
EvalScores=ones(length(reg.PowerMatrix),1);
for ii=1:size(reg.PowerMatrix,2)
EvalScores=EvalScores.*NewScores(:,ii);
end
yhatNew=reg.Coefficients'*EvalScores ;
yhatNew=exp(yhatNew);
ypredpoly(i,:)=yhatNew;
end
figure()
subplot(3,2,1)
plot(outputtest,ypredtest,'o');hold;
xlabel('Real output','FontName','Helvetica', 'Fontsize', 13);
ylabel('GP estimate','FontName','Helvetica', 'Fontsize', 13)
title('JMtauth-RM-Dense','FontName','Helvetica', 'Fontsize', 13);
set(gca, 'FontName','Helvetica', 'Fontsize', 13)
set(gcf,'color','white')

subplot(3,2,3)
plot(outputtest,Predicty,'o');hold;
xlabel('Real output','FontName','Helvetica', 'Fontsize', 13);
ylabel('Linear regression estimate','FontName','Helvetica', 'Fontsize', 13)
title('JMtauth-RM-weight','FontName','Helvetica', 'Fontsize', 13);
set(gca, 'FontName','Helvetica', 'Fontsize', 13)
set(gcf,'color','white')

subplot(3,2,2)
plot(outputtest,ymuv,'o');hold;
xlabel('Real output','FontName','Helvetica', 'Fontsize', 13);
ylabel('GP-sparse estimate','FontName','Helvetica', 'Fontsize', 13)
title('JMtauth-RM-sparse','FontName','Helvetica', 'Fontsize', 13);
set(gca, 'FontName','Helvetica', 'Fontsize', 13)
set(gcf,'color','white')

subplot(3,2,5)
plot(outputtest,zz,'o');
xlabel('Real output','FontName','Helvetica', 'Fontsize', 13);
ylabel('on-line LR estimate','FontName','Helvetica', 'Fontsize', 13)
title('JMtauth-RM-on-Line regression','FontName','Helvetica', 'Fontsize', 13);
set(gca, 'FontName','Helvetica', 'Fontsize', 13)
set(gcf,'color','white')

subplot(3,2,4)
plot(outputtest,ypredpoly,'o');
xlabel('Real output','FontName','Helvetica', 'Fontsize', 13);
ylabel('Polynomial estimate','FontName','Helvetica', 'Fontsize', 13)
title('JMtauth-RM-Polynomial regression','FontName','Helvetica', 'Fontsize', 13);
set(gca, 'FontName','Helvetica', 'Fontsize', 13)
set(gcf,'color','white')

Lerror=(norm(outputtest-ypredpoly)/norm(outputtest))^0.5;
L_2poly=1-(Lerror^2);
%Coefficient of determination
for i=1:numel(outputtest)
    outputreq(i)=outputtest(i)-mean(outputtest);
end

outputreq=outputreq';
CoDpoly=1-(norm(outputtest-ypredpoly)/norm(outputreq));
CoDpoly=1 - (1-CoDpoly)^2 ;






errorpredict = sqrt(mean((Predicty - outputtest).^2)); % root mean squared error
errorsparse = sqrt(mean((ymuv - outputtest).^2)); % root mean squared error
errorspatial = sqrt(mean((ypredtest - outputtest).^2)); % root mean squared error
errorpoly = sqrt(mean((ypredpoly - outputtest).^2)); % root mean squared error
errorzz = sqrt(mean((zz - outputtest).^2)); % root mean squared error


disp(['CoD of weight space approach = ' num2str(CoDweight)]);
disp(['CoD of sparse  approach = ' num2str(CoDsparse)]);
disp(['CoD of spatial approach = ' num2str(CoDspatial)]);
disp(['CoD of polynomial regression approach = ' num2str(CoDpoly)]);
disp(['CoD of on-line regression approach = ' num2str(CoDzz)]);
disp(['Error of spatial GP Regression = ' num2str(errorspatial)]);
disp(['Error of sparse GP regression = ' num2str(errorsparse)]);
disp(['Error of Linear regression = ' num2str(errorpredict)]);
disp(['Error of polynomial regression approach = ' num2str(errorpoly)]);
disp(['Error of on-line regression approach = ' num2str(errorzz)]);
figure()
subplot(2,2,1)
hist(outputtest-ypredtest);
title('JMtauth-RM-GP regression','FontName','Helvetica', 'Fontsize', 13);
set(gca, 'FontName','Helvetica', 'Fontsize', 13)
set(gcf,'color','white')

subplot(2,2,2)
hist(outputtest-ymuv);
title('JMtauth-RM-Sparse-GP regression','FontName','Helvetica', 'Fontsize', 13);
set(gca, 'FontName','Helvetica', 'Fontsize', 13)
set(gcf,'color','white')

subplot(2,2,3)
hist(outputtest-Predicty );
title('JMtauth-RM-Linear regression','FontName','Helvetica', 'Fontsize', 13);
set(gca, 'FontName','Helvetica', 'Fontsize', 13)
set(gcf,'color','white')

subplot(2,2,4)
hist(outputtest-ypredpoly);
title('JMtauth-RM-Polynomial regression','FontName','Helvetica', 'Fontsize', 13);
set(gca, 'FontName','Helvetica', 'Fontsize', 13)
set(gcf,'color','white')

% rstool(inputtrain,outputtrain,'quadratic')
% vv=beta2;
% load sgsim.out;
% load sgsimfinal.out;
% sgsim=reshape(sgsim,72000,100);
% sgsimuse=sgsim(:,1);
% sgsimuse=reshape(sgsimuse,120,60,10);
% sgsimpr=sgsimuse(:,:,3:7);
% sgsimpr=reshape(sgsimpr,36000,1);
% 
% 
% sgsim1=reshape(sgsimfinal,72000,100);
% sgsimuse1=sgsim1(:,1);
% sgsimuse1=reshape(sgsimuse1,120,60,10);
% sgsimpr1=sgsimuse1(:,:,3:7);
% sgsimpr1=reshape(sgsimpr1,36000,1);
% n = 30; sn = 0.5;
% 
%  lik = {@likGauss};    hyp.lik = log(sn); inf = @infGaussLik;
%  cov = {@covSEiso}; 
%  hyp.cov = [0; 0]; 
%  hyp.lik = log(0.5);
% %disp('likfunc = @likGauss; sn = 0.1; hyp.lik = log(sn);')
% %likfunc = @likGauss; sn = 0.1; hyp.lik = log(sn);
% 
% xu = normrnd(0,1,100,1); 
% xsparse=xu;
% hyp.mean=[];
% cov = {'apxSparse',cov,xsparse};           % inducing points
% infv  = @(varargin) inf(varargin{:},struct('s',1.0));   
% hyp = minimize(hyp,@gp,-100,inf,[],cov,lik,sgsimpr,sgsimpr);
%         % VFE, opt.s = 0
% %value=zeros(36000,1);
% 
% [ymuv,ys2v] = gp(hyp,infv,[],cov,lik, sgsimpr, sgsimpr, sgsimpr1);

toc
%poolobj = gcp('nocreate');
%delete(poolobj);
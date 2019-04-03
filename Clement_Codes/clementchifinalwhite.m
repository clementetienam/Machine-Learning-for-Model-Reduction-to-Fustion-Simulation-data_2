clc;
clear;
close all;
%% GP classification/Regression model for the Chi data
disp(' We will implement the sparse GP for the Chi model' );
disp( 'Author: Dr Clement Etienam' )
disp( 'Supervisor: Professor Kody Law' )
disp( 'Supervised classfication Modelling' )
set(0,'defaultaxesfontsize',20); format long

dd=importdata('chi_itg.dat',' ',1);
test=dd.data;
out=test;
outgp=test;
test=out;
%X=zscore(test(1:600000,1:10));
X=(test(1:600000,1:10));
y=(test(1:600000,11));
outputtest=y(290000+1:end,:);
y2=zeros(600000,1);
for i=1:600000
    if y(i)==0
        y2(i)=-1;
    end
    
    if y(i)>0
        y2(i)=1;
    end
        
end
y=y2;
inputtrainclass=X(1:290000,:);
inputtrainclass=whitening(inputtrainclass);

outputtrainclass=y(1:290000,:);
inputtest=X(290000+1:end,:);
inputtest=whitening(inputtest);

%outputtest=y(290000+1:end,:);
p=10;


GPmatrix=[inputtrainclass (test(1:290000,11))];
outgp=GPmatrix;
outgp(any(outgp==0,2),:) = [];

outputtrainGP=log(outgp(:,11));
inputtrainGP=(outgp(:,1:10));
inputtrainGP=whitening(inputtrainGP);
%% Train the classifier (Discriminant analysis model)
[Mdl2]= fitcdiscr(inputtrainclass,outputtrainclass,...
    'OptimizeHyperparameters','auto',...
    'HyperparameterOptimizationOptions',struct('MaxObjectiveEvaluations',30,...
    'AcquisitionFunctionName','expected-improvement-plus'));

Mdl = fitclinear(inputtrainclass,outputtrainclass,...
    'OptimizeHyperparameters','auto','HyperparameterOptimizationOptions',...
    struct('AcquisitionFunctionName','expected-improvement-plus'));
%% Train the GP Sparse approximation
meanfunc=[];% empty: don't use a mean function
n = 30; sn = 0.99;
 lik = {@likGauss};    hyp.lik = log(sn); inf = @infGaussLik;
 cov = {@covSEiso}; 
 hyp.cov = log([9.5;12.5]); % Matern class d=5
p=size(inputtest,2);
for j=1:p
xu = normrnd(0,1,100,1); 
xsparse(:,j)=xu;
end
hyp.xu=xsparse;
cov = {'apxSparse',cov,xsparse};           % inducing points
%parpool
infv  = @(varargin) inf(varargin{:},struct('s',1.0));
infr=@infFITC;
infe = @infFITC_EP;
hyp = minimize(hyp,@gp,-100,infv,meanfunc,cov,lik,inputtrainGP,outputtrainGP);%minimise the hyperparamters
ff=size(inputtest,1);
ff=500;
%% Prediction now
parfor ii=1:ff %size(inputtest,1)
labelDA = predict(Mdl,inputtest(ii,:));
labelDA2 = predict(Mdl2,inputtest(ii,:));
if labelDA==-1
    regressoutput=0;
end
if labelDA==1
[regressoutput,ys2v] = gp(hyp,infv,meanfunc,cov,lik, inputtrainGP, outputtrainGP, inputtest(ii,:));%Inference with GP
regressoutput=exp(regressoutput);
end
clement(ii,:)=regressoutput;

if labelDA2==-1
    regressoutput2=0;
end
if labelDA2==1
[regressoutput2,ys2v2] = gp(hyp,infv,meanfunc,cov,lik, inputtrainGP, outputtrainGP, inputtest(ii,:));%Inference with GP
regressoutput2=exp(regressoutput2);
end
clement3(ii,:)=regressoutput2;

fprintf('Finished testpoint %d out of %d .\n', ii,ff)
end
disp('finished Discriminant analysis method')
poolobj = gcp('nocreate');
delete(poolobj);

rossmary=outputtest(1:500,:);
%Coefficient of determination
for i=1:numel(rossmary)
    outputreq(i)=rossmary(i)-mean(rossmary);
end

outputreq=outputreq';
CoDpoly=1-(norm(rossmary-clement)/norm(outputreq));
CoDpoly2=1-(norm(rossmary-clement3)/norm(outputreq));
CoDnaieve=1 - (1-CoDpoly)^2 ;
CoDlinear=1 - (1-CoDpoly2)^2 ;
%%
disp('Predict at once')
labelDA2 = predict(Mdl,inputtest);
index1=find(labelDA2==-1); %output that gave a zero
index2=find(labelDA2==1); % output that didnt give a zero

clement2=zeros(size(inputtest,1),1);
clement2(index1,:)=0; %values that the classifier predicts to give a 0
[regressoutput2,ys2v] = gp(hyp,infv,meanfunc,cov,lik, inputtrainGP, outputtrainGP, inputtest(index2,:));%Inference with GP
regressoutput2=exp(regressoutput2);
clement2(index2,:)=regressoutput2;


%%
disp('Predict at once')
labelDA2 = predict(Mdl2,inputtest);
index1=find(labelDA2==-1); %output that gave a zero
index2=find(labelDA2==1); % output that didnt give a zero

clement4=zeros(size(inputtest,1),1);
clement4(index1,:)=0; %values that the classifier predicts to give a 0
[regressoutput2,ys2v] = gp(hyp,infv,meanfunc,cov,lik, inputtrainGP, outputtrainGP, inputtest(index2,:));%Inference with GP
regressoutput2=exp(regressoutput2);
clement4(index2,:)=regressoutput2;

%%
rossmary=outputtest;
%Coefficient of determination
for i=1:numel(rossmary)
    outputreq(i)=rossmary(i)-mean(rossmary);
end

outputreq=outputreq';
CoDpoly=1-(norm(rossmary-clement2)/norm(outputreq));
CoDpoly2=1-(norm(rossmary-clement4)/norm(outputreq));
CoDnaieve=1 - (1-CoDpoly)^2 ;
CoDlinear=1 - (1-CoDpoly2)^2 ;




 figure()
 subplot(2,2,1)
plot(outputtest,clement2,'o');
xlabel('Real output','FontName','Helvetica', 'Fontsize', 13);
ylabel('GP estimate','FontName','Helvetica', 'Fontsize', 13)
title('chi data-Sparse GP ')
%title('TGLF data','FontName','Helvetica', 'Fontsize', 13);
set(gca, 'FontName','Helvetica', 'Fontsize', 13)
set(gcf,'color','white')

subplot(2,2,2)
plot(outputtest(1:500),'red')
hold on
plot(clement(1:500),'blue')
title('CHI data ')
%legend('real data','Predicted data','FontName','Helvetica', 'Fontsize', 9)
set(gcf,'color','white')
legend({'real data','Predicted data'},'FontSize',9)

subplot(2,2,3)
hist(outputtest(1:500)-clement)
title('CHI data ')
%legend('real data','Predicted data','FontName','Helvetica', 'Fontsize', 9)
set(gcf,'color','white')



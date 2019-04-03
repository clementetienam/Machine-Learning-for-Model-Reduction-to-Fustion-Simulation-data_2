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
% x1=log(test(1:500,1));
% x2=log(test(1:500,2));
% y=log(test(1:500,11));
% [x1g,x2g]=meshgrid(x1,x2);
% [y1,y2]=meshgrid(y,y);
% 
% figure()
% surf(x1g,x2g,y1)
% shading flat
% %axis([1 500 1 500 ])
% grid off
% ylabel('X2', 'FontName','Helvetica', 'Fontsize', 13);
% xlabel('X1', 'FontName','Helvetica', 'Fontsize', 13);
% colormap('jet')
% %caxis([0.1 0.4])
% h = colorbar;
% ylabel(h, 'Y-values','FontName','Helvetica', 'Fontsize', 13);
% %set(h, 'ylim', [0.1 0.4])
% set(gca, 'FontName','Helvetica', 'Fontsize', 13)
% set(gcf,'color','white')
% set(gca,'xticklabel',[])
% set(gca,'yticklabel',[])

%outgp=test;
test=out;
testuse=test(1:400000,:);
X=log(test(1:400000,1:10));
y=test(1:400000,11);
%outputtest=test(400000+1:end,:);
y2=zeros(400000,1);
for i=1:400000
    if y(i)==0
        y2(i)=-1;
    end
    
    if y(i)>0
        y2(i)=1;
    end
        
end
y=y2;
inputtrainclass=X;
outputtrainclass=y;
inputtest=log(test(400001:end,1:10));
outputtest=test(400000+1:end,11);
p=10;


testuse(any(testuse==0,2),:) = []; %remove zeros
outgp=testuse;
outputtrainGP=log(outgp(:,11));
inputtrainGP=log(outgp(:,1:10));

%% Train the classifier (Discriminant analysis model (Naieve Bayes)
[Mdl]= fitcdiscr(inputtrainclass,outputtrainclass,...
    'OptimizeHyperparameters','auto',...
    'HyperparameterOptimizationOptions',struct('Holdout',0.1,'MaxObjectiveEvaluations',200,...
    'AcquisitionFunctionName','expected-improvement-plus'));

disp('check the accuracy of the classifier')
labelDA2 = predict(Mdl,inputtrainclass);
EVALdaal = Evaluate(outputtrainclass,labelDA2);
disp(['accuracy of the classifier is  = ' num2str(EVALdaal(:,1))]);
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
%ff=size(inputtest,1);
%ff=500;
%% Prediction now
disp( 'predict the classifier now to get the clustering group')
labelDA = predict(Mdl,inputtest);
index1=find(labelDA==-1); %output that gave a zero
index2=find(labelDA==1); % output that didnt give a zero

nn=size(inputtest,1);
disp( 'allocate the size of the output data')
clement=zeros(nn,1); % one single output for GP
clementkody=zeros(nn,1); %for polynomial regression
clement(index1,:)=0; %values that the classifier predicts to give a 0
clementkody(index1,:)=0;
disp(' Predict the nominal value the classfier predicts not to give a 0')
Gptest=inputtest(index2,:);
[regressoutput,ys2v] = gp(hyp,infv,meanfunc,cov,lik, inputtrainGP, outputtrainGP,Gptest );%Inference with GP
regressoutput=exp(regressoutput);
clement(index2,:)=regressoutput;

%Coefficient of determination
for i=1:numel(outputtest)
    outputreq(i)=outputtest(i)-mean(outputtest);
end

outputreq=outputreq';
CoDpoly=1-(norm(outputtest-clement)/norm(outputreq));
CoD=1 - (1-CoDpoly)^2 ;
disp(['accuracy of the machine with GP and Naieve Bayes is  = ' num2str(CoD)]);
%% Polynomial regression
 %% Polynomial regression
 reg=MultiPolyRegress(inputtrainGP,outputtrainGP,3,'figure');

 Kody=inputtest(index2,:);
for i=1:size(inputtest(index2,:),1)
NewDataPoint=Kody(i,:);
NewScores=repmat(NewDataPoint,[length(reg.PowerMatrix) 1]).^reg.PowerMatrix;
EvalScores=ones(length(reg.PowerMatrix),1);
for ii=1:size(reg.PowerMatrix,2)
EvalScores=EvalScores.*NewScores(:,ii);
end
yhatNew=reg.Coefficients'*EvalScores ;
yhatNew=exp(yhatNew);
ypredpoly(i,:)=yhatNew;
end
clementkody(index2,:)=ypredpoly;
EVALdaal3 =immse(outputtest,clementkody);
%Coefficient of determination
for i=1:numel(outputtest)
    outputreq(i)=outputtest(i)-mean(outputtest);
end

outputreq=outputreq';
CoDpoly=1-(norm(outputtest-clementkody)/norm(outputreq));
CoDonpoly=1 - (1-CoDpoly)^2 ;
disp(['accuracy of the machine with polynomial regression and Naieve Bayes is  = ' num2str(CoDonpoly)]);

%disp(['accuracy of the machine with polynomial regression and Naieve Bayes is  = ' num2str(EVALdaal3(:,1))]);
%% On-Line and adaptive regression
Con1=pinv((inputtrainGP'*inputtrainGP)); % initial covariance
Conini=Con1;
Hess1=(inputtrainGP'*inputtrainGP);
thetaon1=(Con1*inputtrainGP')*outputtrainGP;
thetaini=thetatot;

for i=1:size(inputnewpoints,1);
    a=inputnewpoints(i,:);
    b=outputnewpoints(i,:);
    Kalman=Con1*(a')*pinv((1+a*(Con1)*a'));
    put=Kalman;
    Kalmanall(:,i)=put;
    Connew=(eye(10)-(Kalman*a))*Con1;
    put2=reshape(Connew,100,1);
    Connal(:,i)=put2;
    thetanew=((eye(10)-(Kalman*a))*thetaon1)+(Connew*a'*b);
    put3=thetanew;
    thetal(:,i)=put3;
    thetaon1=thetanew;
    Con1=Connew;
end

zz=exp(inputtest*thetaon1);

%%
 figure()
 subplot(3,2,1)
plot(outputtest,clement,'o');
xlabel('Real output','FontName','Helvetica', 'Fontsize', 13);
ylabel('GP estimate','FontName','Helvetica', 'Fontsize', 13)
title('chi data-Sparse GP ')
%title('TGLF data','FontName','Helvetica', 'Fontsize', 13);
set(gca, 'FontName','Helvetica', 'Fontsize', 13)
set(gcf,'color','white')

subplot(3,2,2)
plot(outputtest,'red')
hold on
plot(clement,'blue')
title('CHI data ')
%legend('real data','Predicted data','FontName','Helvetica', 'Fontsize', 9)
set(gcf,'color','white')
legend({'real data','Predicted data'},'FontSize',9)

subplot(3,2,3)
hist(outputtest-clement)
title('CHI data ')
%legend('real data','Predicted data','FontName','Helvetica', 'Fontsize', 9)
set(gcf,'color','white')

 subplot(3,2,4)
plot(outputtest,clementkody,'o');
xlabel('Real output','FontName','Helvetica', 'Fontsize', 13);
ylabel('Polynomial regression estimate','FontName','Helvetica', 'Fontsize', 13)
title('chi data ')
%title('TGLF data','FontName','Helvetica', 'Fontsize', 13);
set(gca, 'FontName','Helvetica', 'Fontsize', 13)
set(gcf,'color','white')

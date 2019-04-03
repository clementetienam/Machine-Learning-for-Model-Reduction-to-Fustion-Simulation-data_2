clc;
clear;
close all;
%% GP regression model for the Dense and Sparse data
disp(' We will implement the sparse GP for the Chi model' );
disp( 'Author: Dr Clement Etienam' )
disp( 'Supervisor: Professor Kody Law' )
disp( 'GP Regression Modelling' )
set(0,'defaultaxesfontsize',20); format long

dd=importdata('chi_itg.dat',' ',1);
test=dd.data;
out=test;
out(any(test==0,2),:) = [];

test=out;
X=log(test(1:200000,1:10));
% rowmin = min(X,[],2);
% rowmax = max(X,[],2);
% 
% X = rescale(X,'InputMin',rowmin,'InputMax',rowmax);

y=log(test(1:200000,11));
inputtrain=X(1:190000,:);
outputtrain=y(1:190000,:);
inputtest=X(190000+1:end,:);
outputtest=y(190000+1:end,:);
p=10;
%% Sparse approximation
%meanfunc = @meanLinear ;
%meanfunc = {@meanPoly ,1}; %degree polynomial 2
%hypp = [1;1;2;3];
meanfunc=[];% empty: don't use a mean function
%hyp2.mean = [0.5; 1;5;0.5;0.5;0.5;0.5;0.5;0.5;0.5,;0.5];
%  fff=zeros(10,1);
%  fff(1:end)=normrnd(0,1,10,1);
% fff=fff';
%  hyp.mean=fff;
% hyp.mean=fff;
n = 30; sn = 0.99;

 lik = {@likGauss};    hyp.lik = log(sn); inf = @infGaussLik;
 cov = {@covSEiso}; 
 %cov={@covMaternard ,3}; 
 hyp.cov = log([9.5;12.5]); % Matern class d=5
 %hyp.cov = [1.5; 2.5]; 
 %hyp.lik = log(0.5);
%disp('likfunc = @likGauss; sn = 0.1; hyp.lik = log(sn);')
%likfunc = @likGauss; sn = 0.1; hyp.lik = log(sn);
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
hyp = minimize(hyp,@gp,-100,infv,meanfunc,cov,lik,inputtrain,outputtrain);%minimise the hyperparamters
%[hyp,nlZ] = vfe_xu_opt(hyp,meanfunc,cov,inputtrain,outputtrain,xsparse,2000); %optimise the inducing points
%cov = {'apxSparse',cov,hyp.xu}; 
[ymuv,ys2v] = gp(hyp,infv,meanfunc,cov,lik, inputtrain, outputtrain, inputtest);%Inference with GP
%[ymuv2,ys2v2] = gp(hyp,infe,meanfunc,cov,lik, inputtrain, outputtrain, inputtest);%Infeence with FITC
ymuvGP=exp(ymuv);

outputtest=exp(outputtest);


Lerror=(norm(outputtest-ymuvGP)/norm(outputtest))^0.5;
L_2GP=1-(Lerror^2);
%Coefficient of determination
for i=1:numel(outputtest)
    outputreq(i)=outputtest(i)-mean(outputtest);
end

outputreq=outputreq';
CoDpoly=1-(norm(outputtest-ymuvGP)/norm(outputreq));
CoDGP=1 - (1-CoDpoly)^2 ;


errorsparse = sqrt(mean((ymuvGP - outputtest).^2)); % root mean squared error


%%Plot figures

%% Online-learning

inputnewpoints=X(190001:end,:);
outputnewpoints=y(190001:end,:);
p=size(inputtrain,2);
alpha=0.001;
Iterations=1000;
Theta=zeros(p,1);
Theta_Iterated = gradientadaptive(inputtrain,outputtrain,Theta,Iterations,alpha);
Theta_initial=Theta_Iterated;

offlinetheta=pinv(inputtrain'*inputtrain)*inputtrain'*outputtrain;

 for i=1:(size(inputnewpoints,1))
    aa=inputnewpoints(i,:);
    bb=outputnewpoints(i,:);
    Theta_Iterated = gradientadaptive2(aa,bb,Theta_Iterated,Iterations,alpha);
 end
 
 predicted=exp(inputtest*Theta_Iterated);
 
 initialpredicted=exp(inputtest*offlinetheta);

 ymuvOL=predicted;
 
 %% Polynomial regression
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
 %%
 figure()
subplot(2,2,1)
plot(outputtest,ymuvGP,'o');
xlabel('Real output','FontName','Helvetica', 'Fontsize', 13);
ylabel('GP estimate','FontName','Helvetica', 'Fontsize', 13)
title('chi data-Sparse GP ')
%title('TGLF data','FontName','Helvetica', 'Fontsize', 13);
set(gca, 'FontName','Helvetica', 'Fontsize', 13)
set(gcf,'color','white')

subplot(2,2,2)
plot(outputtest,predicted,'o');
xlabel('Real output','FontName','Helvetica', 'Fontsize', 13);
ylabel('Linear Regression (off-line)','FontName','Helvetica', 'Fontsize', 13)
title('chi data-offline Regression ')
%title('TGLF data','FontName','Helvetica', 'Fontsize', 13);
set(gca, 'FontName','Helvetica', 'Fontsize', 13)
set(gcf,'color','white')

subplot(2,2,3)
plot(outputtest,initialpredicted,'o');
xlabel('Real output','FontName','Helvetica', 'Fontsize', 13);
ylabel('Linear Regression(on-line)','FontName','Helvetica', 'Fontsize', 13)
title('chi data=on-line ')
%title('TGLF data','FontName','Helvetica', 'Fontsize', 13);
set(gca, 'FontName','Helvetica', 'Fontsize', 13)
set(gcf,'color','white')

subplot(2,2,4)
plot(outputtest,ypredpoly,'o');
xlabel('Real output','FontName','Helvetica', 'Fontsize', 13);
ylabel('Polynomial regression','FontName','Helvetica', 'Fontsize', 13)
title('chi data-Polynomial Regression ')
%title('TGLF data','FontName','Helvetica', 'Fontsize', 13);
set(gca, 'FontName','Helvetica', 'Fontsize', 13)
set(gcf,'color','white')

figure
subplot(2,2,1)
plot(outputtest,'red')
hold on
plot(ymuvGP,'blue')
title('CHI data-GP ')
%legend('real data','Predicted data','FontName','Helvetica', 'Fontsize', 9)
set(gcf,'color','white')
legend({'real data','Predicted data'},'FontSize',9)

subplot(2,2,2)
plot(outputtest,'red')
hold on
plot(predicted,'blue')
title('CHI data-(off-line learning) ')
%legend('real data','Predicted data','FontName','Helvetica', 'Fontsize', 9)
set(gcf,'color','white')
legend({'real data','Predicted data'},'FontSize',9)

subplot(2,2,3)
plot(outputtest,'red')
hold on
plot(initialpredicted,'blue')
title('CHI data-(on-line learning) ')
%legend('real data','Predicted data','FontName','Helvetica', 'Fontsize', 9)
set(gcf,'color','white')
legend({'real data','Predicted data'},'FontSize',9)
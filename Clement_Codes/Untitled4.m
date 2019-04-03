%% GP regression model for the Dense and Sparse data
clc;
clear;
close all;
disp(' We will implement the sparse GP for the TGLF model' );
disp( 'Author: Clement Etienam' )
disp( 'Supervisor: Professor Kody Law' )
disp( 'GP Regression Modelling' )
set(0,'defaultaxesfontsize',20); format long


noEM= readtable('TGLFdata.csv');
noEM = noEM{:,:};
test=noEM;
inputin=test(:,1:22);
output=test(:,23:28);
A=inputin';
colmin = min(A);
colmax = max(A);
Bcol =((rescale(A,'InputMin',colmin,'InputMax',colmax))');
inputtrain=Bcol(1:3e5,:);
%inputtrain(:,6)=[];
inputtest=Bcol(3e5+1:end,:);
%inputtest(:,6)=[];
outputtrain=asinh(output(1:3e5,:));
outputtest=output(3e5+1:end,:);

p=22;
%% Sparse approximation
%meanfunc = @meanLinear ;
%meanfunc = {@meanPoly ,2}; %degree polynomial 2
%hypp = [1;1;2;3];
meanfunc=[];% empty: don't use a mean function
%hyp2.mean = [0.5; 1;5;0.5;0.5;0.5;0.5;0.5;0.5;0.5,;0.5];
 fff=zeros(44,1);
 fff(1:end)=normrnd(0,1,44,1);
%fff=fff';
%  hyp.mean=fff;
% hyp.mean=fff;
n = 30; sn = 0.8;

 lik = {@likGauss};    hyp.lik = log(sn); inf = @infGaussLik;
 cov = {@covSEiso}; 
 %cov={@covMaternard ,3}; 
 hyp.cov = log([1.5;2.5]); % Matern class d=5
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
hyp = minimize(hyp,@gp,-2000,infv,meanfunc,cov,lik,inputtrain,outputtrain);%minimise the hyperparamters
%[hyp,nlZ] = vfe_xu_opt(hyp,meanfunc,cov,inputtrain,outputtrain,xsparse,2000); %optimise the inducing points
[ymuv,ys2v] = gp(hyp,infv,meanfunc,cov,lik, inputtrain, outputtrain, inputtest);%Inference with GP
[ymuv2,ys2v2] = gp(hyp,infr,meanfunc,cov,lik, inputtrain, outputtrain, inputtest);%Infeence with FITC
ymuv=sinh(ymuv);
ymuv2=sinh(ymuv2);

 for j=1:6
     outputin=outputtest(:,j);
     
 Lerror=(norm(outputin-ymuv(:,j))/norm(outputin))^0.5;
L_2sparse=1-(Lerror^2);

L_2all(:,j)=L_2sparse;
%Coefficient of determination
for i=1:numel(outputin)
    outputreq(i)=outputin(i)-mean(outputin);
end

outputreq=outputreq';
CoDsparse=1-(norm(outputin-ymuv(:,j))/norm(outputreq));
CoDsparse=1 - (1-CoDsparse)^2 ;
CoDall(:,j)=CoDsparse;
 end
  
 CoDallv=sum(CoDall)/6;
 L_2allv=sum(L_2all)/6;


figure()
for i=1:6
    subplot(2,3,i)
plot(outputtest(:,i),ymuv(:,i),'o');
xlabel('Real output','FontName','Helvetica', 'Fontsize', 13);
ylabel('GP estimate','FontName','Helvetica', 'Fontsize', 13)
title(['TGLF data ',sprintf('%d',i)])
%title('TGLF data','FontName','Helvetica', 'Fontsize', 13);
set(gca, 'FontName','Helvetica', 'Fontsize', 13)
set(gcf,'color','white')
end

figure
for i=1:6
    subplot(2,3,i)
plot(outputtest(:,i),'red')
hold on
plot(ymuv(:,i),'blue')
title(['TGLF data ',sprintf('%d',i)])
%legend('real data','Predicted data','FontName','Helvetica', 'Fontsize', 9)
set(gcf,'color','white')
legend({'real data','Predicted data'},'FontSize',9)

end




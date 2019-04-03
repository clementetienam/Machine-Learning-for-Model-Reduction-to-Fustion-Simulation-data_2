clc;
clear;
close all;
%% GP regression model for the Dense and Sparse data
disp(' We will implement the sparse GP for the TGLF model' );
disp( 'Author: Clement Etienam' )
disp( 'Supervisor: Professor Kody Law' )
disp( 'GP Regression Modelling' )
set(0,'defaultaxesfontsize',20); format long
load tglf_plus
N=size(test,1);
[v,i]=find(max(abs(test')) > 1e5);
Ni=size(i,2);
H=sparse(Ni,N);H(:,i)=speye(Ni);
ix=find(sum(H,1)==0);
test=test(ix,:);

input=test(:,1:22);
% rescale the input;

output=test(:,23:28);
inputtest=test(:,1:22);
input=input(1:3e5,:);
output=output(1:3e5,:);
inputtest=inputtest(3e5+1:end,:);
oo=5;
ix=find(output(:,oo)>0);
%input=input(ix,:);output=output(ix,:);

%   ix=find((output(:,1)<=10));%.*(output(:,1)>=exp(-2)));
    output1=log(output(ix,oo));
    input1=asinh(input(ix,:));
     inputtest(inputtest==0)=1;
%      =asinh(inputtest(ix,:));
inputin=input1;
 %output1=log(output(:,1));
 outputin=output1;
%input=asinh(input);
X=input;[M,p]=size(X);N=ceil(M/2);%N=M-1000;
%% Sparse approximation
%meanfunc = @meanLinear ;
meanfunc=[];% empty: don't use a mean function
%hyp2.mean = [0.5; 1;5;0.5;0.5;0.5;0.5;0.5;0.5;0.5,;0.5];
% fff=zeros(p,1);
% fff(1:end)=0.5;
% %fff=fff';
% hyp.mean=fff;
% hyp.mean=fff;
n = 30; sn = 0.5;

 lik = {@likGauss};    hyp.lik = log(sn); inf = @infGaussLik;
 cov = {@covSEiso}; 
 hyp.cov = [0; 0]; 
 hyp.lik = log(0.5);
%disp('likfunc = @likGauss; sn = 0.1; hyp.lik = log(sn);')
%likfunc = @likGauss; sn = 0.1; hyp.lik = log(sn);
for j=1:p
xu = normrnd(0,1,100,1); 
xsparse(:,j)=xu;
end
cov = {'apxSparse',cov,xsparse};           % inducing points
%parpool
infv  = @(varargin) inf(varargin{:},struct('s',1.0)); 
infe = @infFITC_EP;
hyp = minimize(hyp,@gp,-100,infv,meanfunc,cov,lik,inputin,outputin);
[ymuv,ys2v] = gp(hyp,infv,meanfunc,cov,lik, inputin, outputin, inputin);


 Lerror=(norm(outputin-ymuv)/norm(outputin))^0.5;
L_2sparse=1-(Lerror^2);
%Coefficient of determination
for i=1:numel(outputin)
    outputreq(i)=outputin(i)-mean(outputin);
end

outputreq=outputreq';
CoDsparse=1-(norm(outputin-ymuv)/norm(outputreq));
CoDsparse=1 - (1-CoDsparse)^2 ;

figure()
plot(outputin,ymuv,'o');hold;
xlabel('Real output','FontName','Helvetica', 'Fontsize', 13);
ylabel('GP estimate','FontName','Helvetica', 'Fontsize', 13)
title('TGLF data','FontName','Helvetica', 'Fontsize', 13);
set(gca, 'FontName','Helvetica', 'Fontsize', 13)
set(gcf,'color','white')
%poolobj = gcp('nocreate');
%delete(poolobj);
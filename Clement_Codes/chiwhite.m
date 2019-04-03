clc;
clear all;
close all;
set(0,'defaultaxesfontsize',20); format long

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
        y2(i)=-1;
    end
    
    if y(i)>0
        y2(i)=1;
    end
        
end
y=y2;
inputtrainclass=X(1:290000,:);
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
[regressoutput,ys2v] = gp(hyp,infv,meanfunc,cov,lik, inputtrainGP, outputtrainGP, inputtest);%Inference with GP
regressoutput=exp(regressoutput);
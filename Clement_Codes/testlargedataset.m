clear;set(0,'defaultaxesfontsize',20); format long
load sgsim.out;
load sgsimporo.out;

sgsim=log(sgsim);
inputin=sgsim(1:2e4,:);
inputtest=sgsim(2e4+1:end,:);
 %output1=log(output(:,1));
 outputin=sgsimporo(1:2e4,:);
 outputest=sgsimporo(2e4+1:end,:);
%input=asinh(input);
X=inputin;[M,p]=size(X);N=ceil(M/2);%N=M-1000;
%% Sparse approximation
%a = 0.3; b = 1.2;             % underlying function
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
parpool
infv  = @(varargin) inf(varargin{:},struct('s',1.0));   
hyp = minimize(hyp,@gp,-100,infv,[],cov,lik,inputin,outputin);


        % VFE, opt.s = 0

[ymuv,ys2v] = gp(hyp,infv,[],cov,lik, inputin, outputin, inputin);
poolobj = gcp('nocreate');
delete(poolobj);
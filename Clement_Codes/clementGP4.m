clc;
clear;
close all;
%% GP regression model for the Dense and Sparse data
disp(' We will implement the dense and sparse GP for the Tauth model' );
disp( 'Author: Clement Etienam' )
disp( 'Supervisor: Professor Kody Law' )
disp( 'GP Regression Model using MATLAB functions' )
tic
load('JM_tauth_data')
input=[r0,a0,kappa,delta,ip,b0,nebar,zeff,ploss];
inputtrain=log(input(1:2100,:)); %select the first 2100 for training
inputtest=log(input(2101:end,:)); %use the remaining data for testing
output=[tauth];
outputtrain=log(output(1:2100,:)); %select the first 2100 for training
outputtest=log(output(2101:end,:)); %use the remaining data for testing
[inputtrain,A,B] = scaletrain(inputtrain);
inputtest = scaletest(inputtest,A,B);
[outputtrain,A,B] = scaletrain(outputtrain);
outputtest = scaletest(outputtest,A,B);
[n,D] = size(inputtrain);
nstar = size(inputtest,1);
% optimize the hyperparameters
m = 200;  % choose size of the subset, m <= n; you can try larger one
% now select random training set of size m
perm = randperm(length(outputtrain));
INDEX = perm(1:m);
Xm = inputtrain(INDEX,:);
ym = outputtrain(INDEX);
% logtheta = [log(ell_1), log(ell_2), ... log(ell_D),log(sigma_f),log(sigma_n)
covfunc = {'covSum', {'covSEard','covNoise'}};
% train hyperparameters
logtheta0 = zeros(D+2,1);              % starting values of log hyperparameters
logtheta0(D+2) = -1.15;                 % starting value for log(noise std dev)

logtheta = minimize(logtheta0, 'gpr', -100, covfunc, Xm, ym);
[n,d] = size(inputtrain);
tol = 1e-4; % stop tolerance
activesize = 100; % size of active set
cachesize = 1e8; % size of cache
kparam = [1./exp(logtheta(1:d)).^2/2; exp(logtheta(d+1)).^2]; % transform gpml-matlab hyperparameters format to that of GBCD
nvar = exp(logtheta(d+2)).^2; % transform gpml-matlab hyperparameters format to that of GBCD

% make a prediction using all training samples
[alpha,grad,time] = gbcd(inputtrain',outputtrain,kparam,nvar,tol,activesize,cachesize); % take about 2 minutes
result = gprpred(inputtest',inputtrain',kparam,alpha);
error = sqrt(mean((result - outputtest).^2)); % root mean squared error
disp(['Mean Squared Error = ' num2str(error)]);
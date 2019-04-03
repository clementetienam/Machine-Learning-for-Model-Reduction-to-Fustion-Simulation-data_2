%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Authors: Liefeng Bo, Cristian Sminchisescu                         %                                         
% Date: 01/22/2010                                                   %
%                                                                    % 
% Copyright (c) 2010  L. Bo, C. Sminchisescu - All rights reserved   %
%                                                                    %
% This software is free for non-commercial usage only. It must       %
% not be distributed without prior permission of the author.         %
% The author is not responsible for implications from the            %
% use of this software. You can run it at your own risk.             %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% demonstrate GBCD with learning hyperparameters (GPML Matlab Code is needed, 
% available at http://www.gaussianprocess.org/gpml/code/matlab/doc/)

clear
% initialization
load sarcos_inv;
load sarcos_inv_test;
X = sarcos_inv(:,1:21);
y = sarcos_inv(:,22);
Xstar = sarcos_inv_test(:,1:21);
ystar = sarcos_inv_test(:,22);
clear sarcos_inv sarcos_inv_test;

[X,A,B] = scaletrain(X);
Xstar = scaletest(Xstar,A,B);
[y,A,B] = scaletrain(y);
ystar = scaletest(ystar,A,B);
[n,D] = size(X);
nstar = size(Xstar,1);

% optimize the hyperparameters
m = 1000;  % choose size of the subset, m <= n; you can try larger one
% now select random training set of size m
perm = randperm(length(y));
INDEX = perm(1:m);
Xm = X(INDEX,:);
ym = y(INDEX);
% logtheta = [log(ell_1), log(ell_2), ... log(ell_D),log(sigma_f),log(sigma_n)
covfunc = {'covSum', {'covSEard','covNoise'}};
% train hyperparameters
logtheta0 = zeros(D+2,1);              % starting values of log hyperparameters
logtheta0(D+2) = -1.15;                 % starting value for log(noise std dev)
tic;
logtheta = minimize(logtheta0, 'gpr', -100, covfunc, Xm, ym);
time = toc;
save sarcos10000 X y Xstar ystar logtheta time

% GBCD
[n,d] = size(X);
tol = 1e-4; % stop tolerance
activesize = 500; % size of active set
cachesize = 1e8; % size of cache
kparam = [1./exp(logtheta(1:d)).^2/2; exp(logtheta(d+1)).^2]; % transform gpml-matlab hyperparameters format to that of GBCD
nvar = exp(logtheta(d+2)).^2; % transform gpml-matlab hyperparameters format to that of GBCD

% make a prediction using all training samples
[alpha,grad,time] = gbcd(X',y,kparam,nvar,tol,activesize,cachesize); % take about 2 minutes
result = gprpred(Xstar',X',kparam,alpha);
error = sqrt(mean((result - ystar).^2)); % root mean squared error
disp(['Mean Squared Error = ' num2str(error)]);

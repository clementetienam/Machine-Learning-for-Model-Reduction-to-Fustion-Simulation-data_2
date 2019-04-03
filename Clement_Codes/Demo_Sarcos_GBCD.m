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

clear
% make a prediction using 10000 training samples, it takes about 2 minutes
load sarcos10000

% % make a prediction using full training samples
% load sarcos_inv;
% load sarcos_inv_test;
% [X,A,B] = scaletrain(sarcos_inv(:,1:21));
% Xstar = scaletest(sarcos_inv_test(:,1:21),A,B);
% [y,A,B] = scaletrain(sarcos_inv(:,22));
% ystar = scaletest(sarcos_inv_test(:,22),A,B);
% clear sarcos_inv sarcos_inv_test;

[n,d] = size(X);
tol = 1e-4; % stop tolerance
activesize = 500; % size of active set
cachesize = 1e8; % size of cache
kparam = [1./exp(logtheta(1:d)).^2/2; exp(logtheta(d+1)).^2]; % transform gpml-matlab hyperparameters format to that of GBCD
nvar = exp(logtheta(d+2)).^2; % transform gpml-matlab hyperparameters format to that of GBCD

% make a prediction using 10000 training samples
[alpha,grad,time] = gbcd(X',y,kparam,nvar,tol,activesize,cachesize);
result = gprpred(Xstar',X',kparam,alpha);
error = sqrt(mean((result - ystar).^2)); % root mean squared error
disp(['Mean Squared Error = ' num2str(error)]);


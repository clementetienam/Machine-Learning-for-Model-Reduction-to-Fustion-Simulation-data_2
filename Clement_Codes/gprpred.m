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


function result = gprpred(Xstar, X, kparam, alpha)


%% Input
    % Xstar: test inputs, dxm matrix, each column is an sample, d is dimension and m number of test samples;
    % X: training inputs, dxn matrix, each column is an sample, d is dimension and n number of traing samples;
    % kparam: (d+1)x1 vecotr, parameters of Gaussian kernel
    % k(x^p,x^q) = kparam(1)*exp(-(x^p - x^q)'*P*(x^p - x^q))
    % where P is diagonal matrix with ARD parameters kparam(2),...,kparam(d+1);
    % alpha: solution of linear system (K + variance*I)*alpha = y (GBCD returns)
%% Output
    % result: predicted outputs for test samples
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


function [alpha, grad, time] = gbcd(X, y, kparam, nvar, tol, activesize, cachesize)

%% Input
    % X: training input, dxn matrix, each column is an sample, d is dimension and n number of traing samples;
    % y: training outputs, nx1 vector
    % kparam: (d+1)x1 vecotr, parameters of Gaussian kernel
    %   k(x^p,x^q) = kparam(1)*exp(-(x^p - x^q)'*P*(x^p - x^q))
    %   where P is diagonal matrix with ARD parameters kparam(2),...,kparam(d+1);
    % nvar: noise variance.
    % tol: stop tolerance, for example 1e-4. If the infinite norm of gradient is smaller than tol, stop
    % activesize: size of active set, for example, 500 
    % cachesize: size of cache, for example, 1e8 = 0.8 GB memory
%% Output
    % alpha: solution of linear system (K + variance*I)*alpha = y (GBCD returns)
    % grad: grad = (K + variance*I)*alpha - y
    % time: run time, seconds
clc;
clear;
close all;
M = csvread('TGLFdata.csv',1,0);
ix = ismissing(M);
completeData = M(~any(ix,2),:);
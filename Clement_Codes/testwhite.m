clc;
clear;
close all;
%% GP classification/Regression model for the Chi data
disp(' We will implement the sparse GP together with classification for the Chi model' );
disp( 'Author: Dr Clement Etienam' )
disp( 'Supervisor: Professor Kody Law' )
disp( 'Supervised classfication Modelling' )
set(0,'defaultaxesfontsize',20); format long


dd=importdata('chi_itg.dat',' ',1);
test=dd.data;
out=test;
outgp=test;
test=out;
%X=zscore(test(1:600000,1:10));
X=log(test(1:600000,1:10));
dat = bsxfun(@minus,X,mean(X));
cov=dat*dat'./9;
matrixwhite= whitening2(X);
y=(test(1:600000,11));
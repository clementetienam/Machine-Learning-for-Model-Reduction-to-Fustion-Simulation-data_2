clc;
clear;
close all;
load inputtianke.out;
load outputtianke.out;
load inputtianketest.out;
load outputtianketest.out;
X=reshape(inputtianke,500,6);
xtest=reshape(inputtianketest,500,6);
y=reshape(outputtianke,500,1);
ytest=reshape(outputtianketest,500,1);
disp('degree polynomial 3')

sigma=1;
Ne=input('Enter the size of the ensemble ');
disp('Create initial ensemble')
for i=1:Ne
Theta=sigma*randn(6,3);
Thetainitia;
end


options.MaxFunEvals=1000000;
options.MaxIter=1000000;
disp('optimise theta')
[Theta1,fval,exitflag,output]=fminsearch(@(Theta)optpoly(y,Theta,X),Theta,options);

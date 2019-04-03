clc;
clear all;
close all;
disp(' We will implement the dense and sparse GP for the Tauth model' );
disp( 'Author: Dr Clement Etienam' )
disp( 'Supervisor: Professor Kody Law' )
disp( 'GP Regression Modelling' )
load atom_proj.txt
disp('Compute the pairwise distance matrix')
Aa=atom_proj;

% for i=1:size(Aa,1)
%     A=Aa(i,:);
%     A(A==-1)=NaN;
% n=numel(A);
% B=-diff(A(fullfact([n n]))')';
% Bb(i,:)=B';
% end

for i=1:size(Aa,1)
     A=Aa(i,:);
    A(A==-1)=NaN;
out = (bsxfun(@minus,A,A'));
out=reshape(out,36,1);
out=out';
Bb(i,:)=out;
end
disp('Set initial guess of theta')
Theta=[-0.6;-0.5]; % initial guess

options.MaxFunEvals=10000;
options.MaxIter=10000;
disp('optimise theta')
[Theta1,fval,exitflag,output]=fminsearch(@(Theta)clementSI(Bb,Theta),Theta,options);%,sigg),pp);
%Thetaoptimised=log(alpha1);


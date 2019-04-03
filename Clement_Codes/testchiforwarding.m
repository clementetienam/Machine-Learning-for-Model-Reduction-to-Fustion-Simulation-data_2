clc;
clear;
close all;
disp( 'Implement forwarding of Chi data')
dd=importdata('chi_itg.dat',' ',1);
test=dd.data;
out=test;
outgp=test;
test=out;

ne=test(:,1);
te=test(:,2);
ti=test(:,3);
zeff=test(:,4);
q=test(:,5);
shat=test(:,6);
rlni=test(:,7);
rlti=test(:,8);
a=test(:,9);
r0=test(:,10);

parfor i=1:size(test,1)
chi(i,:)=forwarding(ne(i,:), te(i,:), ti(i,:), zeff(i,:), q(i,:), shat(i,:), rlni(i,:), rlti(i,:), a(i,:), r0(i,:), 1, 1 );
end
 poolobj = gcp('nocreate');
delete(poolobj);
%% For 1D maginal
N=10;
X=test(:,1:N);
L=length(X);
numpts=100;
mean=sum(X)/L;
X1=linspace(min(X(:,1)),max(X(:,1)),numpts)';
Xslice1input=[X1,repmat(mean(2:end),numpts,1)];
test=Xslice1input;
ne=test(:,1);
te=test(:,2);
ti=test(:,3);
zeff=test(:,4);
q=test(:,5);
shat=test(:,6);
rlni=test(:,7);
rlti=test(:,8);
a=test(:,9);
r0=test(:,10);
ds=size(test,1);
parfor i=1:ds
chi1(i,:)=forwarding(ne(i,:), te(i,:), ti(i,:), zeff(i,:), q(i,:), shat(i,:), rlni(i,:), rlti(i,:), a(i,:), r0(i,:), 1, 1 );
end

poolobj = gcp('nocreate');
delete(poolobj);
%% 2D maginal

X1=linspace(min(X(:,1)),max(X(:,1)),numpts)';
X2=linspace(min(X(:,2)),max(X(:,2)),numpts)';
[XX1,XX2]=meshgrid(X1,X2);
Xslice2input=[XX1(:),XX2(:),repmat(mean(3:end),numpts^2,1)];
test=Xslice2input;
ne=test(:,1);
te=test(:,2);
ti=test(:,3);
zeff=test(:,4);
q=test(:,5);
shat=test(:,6);
rlni=test(:,7);
rlti=test(:,8);
a=test(:,9);
r0=test(:,10);
parfor i=1:size(test,1)
chi2(i,:)=forwarding(ne(i,:), te(i,:), ti(i,:), zeff(i,:), q(i,:), shat(i,:), rlni(i,:), rlti(i,:), a(i,:), r0(i,:), 1, 1 );
end
poolobj = gcp('nocreate');
delete(poolobj);


% %X=zscore(test(1:600000,1:10));
% X=log(test(1:600000,1:10));
% y=(test(1:600000,11));
% y3=y;
% outputtest=y(290000+1:end,:);
% y2=zeros(600000,1);
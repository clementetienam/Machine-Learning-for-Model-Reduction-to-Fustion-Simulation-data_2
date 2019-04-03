clc;
clear;
close all;
clear;
set(0,'defaultaxesfontsize',20);
format long
load('jm_data.mat')
output=[ptotped, betanped, wped];
input=[r a kappa delta bt ip neped betan zeffped];
X=input;
[M,p]=size(X);
N=M/2; %training rowes
pin=p;
y=output;
yuse=y(1:N,:);
for i=1:M
    for j=1:M
        C0(i,j) = exp(-1/2*norm(X(i,:)-X(j,:))^2);
    end 
end
ytest=y(N+1:end,:);
%C0test = pdist2(X,X,'euclidean'); %same as C0
m=zeros(M,1);
pp=ones(3,1);
% pp(1)=sigma;
% pp(2)=l;
% pp(3)=sigg;
for k=1:3
 sigma=0.2;
    l=0.1;
    sigg=0.1;
    pp(1,1)=(sigma);
    pp(2,1)=(l);
    pp(3,1)=(sigg);
C=(sigma^2)*C0.^(1/exp(l^2)); %covariance function with 

%% Predict the mean of new inputs
fmean=(((C(1:N,1:N)+sigg^2*eye(N)))\C(1:N,N+1:end))*y(1:N,k);
fall(:,k)=fmean;
% options = optimset('PlotFcns',@optimplotfval,'Display','iter','TolFun',1e-6,'MaxFunEvals',1e10);
% [pp1,fvalss,iter]=minimize2(pp,'mle_gpclementB',-100,N,pin,C0,y(:,k));
%meanfunc = [];                 % empty: don't use a mean function
meanfunc = @meanLinear ;                % empty: don't use a mean function
%hyp2.mean = [0.5; 1;5;0.5;0.5;0.5;0.5;0.5;0.5;0.5,;0.5];
fff=zeros(9,1);
fff(1:end)=0.5;
%fff=fff';
hyp2.mean=fff;
%hyp.mean=fff;
covfunc = @covSEiso; hyp2.cov = [0; 0]; hyp2.lik = log(0.1);
disp('likfunc = @likGauss; sn = 0.1; hyp.lik = log(sn);')
likfunc = @likGauss; sn = 0.1; hyp.lik = log(sn);
hyp2 = minimize(hyp2, @gp, -100, @infGaussLik, meanfunc, covfunc, likfunc, X(1:N,:), yuse(:,k));
valout=hyp2.cov;
pp1(1)=valout(2,:);
pp1(2)=valout(1,:);
pp1(3)=hyp2.lik;
%like=mle_gpclementB(p,N,pin,C0,y)
pp1=((pp1));
ppk(:,k)=pp1;
sigma=(pp1(1));
l=exp(pp1(2));
sigg=(pp1(3));
C=sigma^2.*C0.^(1/l^2); %covariance function  
%% compute L
% L=chol(C(1:N,1:N)+sigg^2*eye(N));
% alpha=L'\(L\y(1:N,k));
%% Predict the mean of new inputs
fmean=(((C(1:N,1:N)+sigg^2.*eye(N)))\C(1:N,N+1:end))*y(1:N,k);
fallcorrected(:,k)=fmean;
%% Predict the variance
% varia=L\C(1:N,N+1:end);
% actualvaria=C(N+1:end,N+1:end)-varia'*varia;
% varallcorrected(:,:,k)=actualvaria;
end
figure()
for i=1:3
subplot(2,2,i)
plot(ytest(:,i),fallcorrected(:,i),'o');hold;
xlabel('Real output','FontName','Helvetica', 'Fontsize', 13);
ylabel('GP estimate','FontName','Helvetica', 'Fontsize', 13)
title (sprintf('output %d vs input %d',i,i))
%title('JMdata','FontName','Helvetica', 'Fontsize', 13);
set(gca, 'FontName','Helvetica', 'Fontsize', 13)
set(gcf,'color','white')
end
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
xuse=X(1:N,:);
xtest=X(N+1:end,:);

for i=1:M
    for j=1:M
        C0(i,j) = exp(-1/2*norm(X(i,:)-X(j,:))^2);
    end 
end
ytest=y(N+1:end,:);
%C0test = pdist2(X,X,'euclidean'); %same as C0
m=zeros(M,1);
pp=ones(3,1);

for k=1:3
    sigma=0.2;
    l=(0.1);
    sigg=0.1;
    pp(1,1)=(sigma);
    pp(2,1)=(l);
    pp(3,1)=(sigg);
    C=sigma^2*C0.^(1/l^2);
%C=(sigma^2)*C0.^(1/exp(l^2)); %covariance function with 

%% Predict the mean of new inputs
% upper left corner of K
Kaa = C(1:N,1:N);
% lower right corner of K
Kbb = C(N+1:end,N+1:end);
% upper right corner of K
Kab = C(N+1:end,1:N);
% mean of posterior
 % get beta
 meann=xtest*((xuse'*xuse)\(xuse'*yuse(:,k)));
fmean2 = meann+(Kab*((Kaa+sigg^2*eye(N))\(y(1:N,k)-meann)));
pp(1)=sigma;pp(2)=l;pp(3)=sigg;
fall(:,k)=fmean2;
[pp1,fval,exitflag]=fminsearch(@(pp)clementmin(pp,N,C0,yuse(:,k),xuse),pp);%,sigg),pp);
sigma=pp1(1);
l=pp1(2);
sigg=pp1(3);


ppk(:,k)=pp1;

%C=sigma^2.*C0.^(1/l^2); %covariance function  
%% compute L
% L=chol(C(1:N,1:N)+sigg^2*eye(N));
% alpha=L'\(L\y(1:N,k));
%% Predict the mean of new inputs
C=sigma^2*C0.^(1/l^2);
%C=(sigma^2)*C0.^(1/exp(l^2)); %covariance function with 

%% Predict the mean of new inputs
% upper left corner of K
Kaa = C(1:N,1:N);
% lower right corner of K
Kbb = C(N+1:end,N+1:end);
% upper right corner of K
Kab = C(N+1:end,1:N);
 meann=xtest*((xuse'*xuse)\(xuse'*yuse(:,k)));
fmean = meann+(Kab*((Kaa+sigg^2*eye(N))\(yuse(1:N,k)-meann)));

fallcorrected(:,k)=fmean;

end
figure()
for i=1:3
subplot(2,2,i)
plot(ytest(:,i),fallcorrected(:,i),'o');
xlabel('Real output','FontName','Helvetica', 'Fontsize', 13);
ylabel('GP estimate','FontName','Helvetica', 'Fontsize', 13)
title (sprintf('output %d vs input %d',i,i))
%title('JMdata','FontName','Helvetica', 'Fontsize', 13);
set(gca, 'FontName','Helvetica', 'Fontsize', 13)
set(gcf,'color','white')
end

figure()
for i=1:3
subplot(2,2,i)
plot(ytest(:,i),fall(:,i),'o');
xlabel('Real output','FontName','Helvetica', 'Fontsize', 13);
ylabel('GP estimate','FontName','Helvetica', 'Fontsize', 13)
title (sprintf('output %d vs input %d (unlearned)',i,i))
%title('JMdata','FontName','Helvetica', 'Fontsize', 13);
set(gca, 'FontName','Helvetica', 'Fontsize', 13)
set(gcf,'color','white')
end
%% Calculate CoD and L2
for i=1:3
Lerror=(norm(ytest(:,i)-fallcorrected(:,i))/norm(ytest(:,i)))^0.5;
L_2sparse=1-(Lerror^2);
Lall(:,i)=L_2sparse;
%Coefficient of determination
 outputtest=ytest(:,i);
for ii=1:numel(ytest(:,1))
    outputreq(ii)=outputtest(ii)-mean(outputtest);
end

outputreq=outputreq';
CoDsparse=1-(norm(ytest(:,i)-fallcorrected(:,i))/norm(outputreq));
CoDsparse=1 - (1-CoDsparse)^2 ;
CoDall(:,i)=CoDsparse;
end

for i=1:3
Lerror=(norm(ytest(:,i)-fall(:,i))/norm(ytest(:,i)))^0.5;
L_2sparse=1-(Lerror^2);
Lallu(:,i)=L_2sparse;
%Coefficient of determination
 outputtest=ytest(:,i);
for ii=1:numel(ytest(:,1))
    outputreq(ii)=outputtest(ii)-mean(outputtest);
end

outputreq=outputreq';
CoDsparse=1-(norm(ytest(:,i)-fall(:,i))/norm(outputreq));
CoDsparse=1 - (1-CoDsparse)^2 ;
CoDallu(:,i)=CoDsparse;
end
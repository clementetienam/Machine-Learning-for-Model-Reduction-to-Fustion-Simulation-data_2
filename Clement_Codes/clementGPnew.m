clc;
clear;
close all;
clear;
set(0,'defaultaxesfontsize',20);
format long
load('jm_data.mat')
output=[ptotped, betanped, wped];
input=[r a kappa delta bt ip neped betan zeffped];
X=(input);
[M,p]=size(X);
pin=p;
y=(output);
Xuse=X(1:M/2,:);
Xtest=X(M/2+1:end,:);
yuse=y(1:M/2,:);
ytest=y(M/2+1:end,:);
meanfunc = []                 % empty: don't use a mean function
covfunc = @covSEiso; hyp2.cov = [0; 0]; hyp2.lik = log(0.1);
disp('likfunc = @likGauss; sn = 0.1; hyp.lik = log(sn);')
likfunc = @likGauss; sn = 0.1; hyp.lik = log(sn);
[muntrained, s2untrained] = gp(hyp2, @infGaussLik, [], covfunc, likfunc, Xuse, yuse, Xtest);
hyp2 = minimize(hyp2, @gp, -100, @infGaussLik, [], covfunc, likfunc, Xuse, yuse);
[m, s2] = gp(hyp2, @infGaussLik, [], covfunc, likfunc, Xuse, yuse, Xtest);

%% For learned GP
figure()
for i=1:3
subplot(2,2,i)
plot(ytest(:,i),m(:,i),'o');hold;
xlabel('Real output','FontName','Helvetica', 'Fontsize', 13);
ylabel('GP estimate','FontName','Helvetica', 'Fontsize', 13)
title (sprintf('output %d vs input %d (for learned GP) ',i,i))
%title('JMdata','FontName','Helvetica', 'Fontsize', 13);
set(gca, 'FontName','Helvetica', 'Fontsize', 13)
set(gcf,'color','white')
end
%%Calculate CoD and L2
for i=1:3
Lerror=(norm(ytest(:,i)-m(:,i))/norm(ytest(:,i)))^0.5;
L_2sparse=1-(Lerror^2);
Lall(:,i)=L_2sparse;
%Coefficient of determination
 outputtest=ytest(:,i);
for ii=1:numel(ytest(:,1))
    outputreq(ii)=outputtest(ii)-mean(outputtest);
end

outputreq=outputreq';
CoDsparse=1-(norm(ytest(:,i)-m(:,i))/norm(outputreq));
CoDsparse=1 - (1-CoDsparse)^2 ;
CoDall(:,i)=CoDsparse;
end

%% For initial GP
figure()
for i=1:3
subplot(2,2,i)
plot(ytest(:,i),muntrained(:,i),'o');hold;
xlabel('Real output','FontName','Helvetica', 'Fontsize', 13);
ylabel('GP estimate','FontName','Helvetica', 'Fontsize', 13)
title (sprintf('output %d vs input %d (for learned GP) ',i,i))
%title('JMdata','FontName','Helvetica', 'Fontsize', 13);
set(gca, 'FontName','Helvetica', 'Fontsize', 13)
set(gcf,'color','white')
end
%%Calculate CoD and L2
for i=1:3
Lerror=(norm(ytest(:,i)-muntrained(:,i))/norm(ytest(:,i)))^0.5;
L_2sparse=1-(Lerror^2);
Lallu(:,i)=L_2sparse;
%Coefficient of determination
 outputtest=ytest(:,i);
for ii=1:numel(ytest(:,1))
    outputreq(ii)=outputtest(ii)-mean(outputtest);
end

outputreq=outputreq';
CoDsparse=1-(norm(ytest(:,i)-muntrained(:,i))/norm(outputreq));
CoDsparse=1 - (1-CoDsparse)^2 ;
CoDallu(:,i)=CoDsparse;
end

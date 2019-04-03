clc;
clear;
close all;
%% Online and adaptve learning
disp(' We will implement the sparse GP for the Chi model' );
disp( 'Author: Clement Etienam' )
disp( 'Supervisor: Professor Kody Law' )
disp( 'Online GLM' )
set(0,'defaultaxesfontsize',20); format long

dd=importdata('chi_itg.dat',' ',1);
test=dd.data;
out=test;
out(any(test==0,2),:) = [];

test=out;
X=log(test(1:200000,1:10));
% rowmin = min(X,[],2);
% rowmax = max(X,[],2);
% 
% X = rescale(X,'InputMin',rowmin,'InputMax',rowmax);

y=log(test(1:200000,11));
inputtrain=X(1:9000,:);
outputtrain=y(1:9000,:);
inputnewpoints=X(9001:19000,:);
outputnewpoints=y(9001:19000,:);
inputtest=X(19000+1:end,:);
outputtest=y(19000+1:end,:);
p=size(inputtrain,2);


inputot=[inputtrain;inputnewpoints];
outputot=[outputtrain;outputnewpoints];

thetatot=(inputot'*inputot)\inputot'*outputot;

%% Kody's Method
Con1=pinv((inputtrain'*inputtrain)); % initial covariance
Conini=Con1;
Hess1=(inputtrain'*inputtrain);
thetaon1=(Con1*inputtrain')*outputtrain;
thetaini=thetatot;

for i=1:size(inputnewpoints,1);
    a=inputnewpoints(i,:);
    b=outputnewpoints(i,:);
    Kalman=Con1*(a')*pinv((1+a*(Con1)*a'));
    put=Kalman;
    Kalmanall(:,i)=put;
    Connew=(eye(10)-(Kalman*a))*Con1;
    put2=reshape(Connew,100,1);
    Connal(:,i)=put2;
    thetanew=((eye(10)-(Kalman*a))*thetaon1)+(Connew*a'*b);
    put3=thetanew;
    thetal(:,i)=put3;
    thetaon1=thetanew;
    Con1=Connew;
end

zz=inputtest*thetaon1;


%% My Method
alpha=0.001;
Iterations=1000;
Theta=zeros(p,1);
Theta_Iterated = gradientadaptive(inputtrain,outputtrain,Theta,Iterations,alpha);
Theta_initial=Theta_Iterated;

 for i=1:(size(inputnewpoints,1))
    aa=inputnewpoints(i,:);
    bb=outputnewpoints(i,:);
    Theta_Iterated = gradientadaptive2(aa,bb,Theta_Iterated,Iterations,alpha);
 end
 
 predicted=exp(inputtest*Theta_Iterated);
 outputtest=exp(outputtest);
 initialpredicted=exp(inputtest*thetaini);
 kody=exp(zz);

 ymuv=predicted;
 
 Lerror=(norm(outputtest-ymuv)/norm(outputtest))^0.5;
L_2online=1-(Lerror^2);
%Coefficient of determination
for i=1:numel(outputtest)
    outputreq(i)=outputtest(i)-mean(outputtest);
end

outputreq=outputreq';
CoDpoly=1-(norm(outputtest-ymuv)/norm(outputreq));
CoDonline=1 - (1-CoDpoly)^2 ;


erroronline = sqrt(mean((ymuv - outputtest).^2)); % root mean squared error

%% Off-line

Lerror=(norm(outputtest-initialpredicted)/norm(outputtest))^0.5;
L_2offline=1-(Lerror^2);
%Coefficient of determination
for i=1:numel(outputtest)
    outputreq(i)=outputtest(i)-mean(outputtest);
end

outputreq=outputreq';
CoDpoly=1-(norm(outputtest-initialpredicted)/norm(outputreq));
CoDffline=1 - (1-CoDpoly)^2 ;


figure()
subplot(2,2,1)
plot(outputtest,ymuv,'o');
xlabel('Real output','FontName','Helvetica', 'Fontsize', 13);
ylabel('GLM estimate (on-line)-me','FontName','Helvetica', 'Fontsize', 13)
title('chi data-online my method ')
%title('TGLF data','FontName','Helvetica', 'Fontsize', 13);
set(gca, 'FontName','Helvetica', 'Fontsize', 13)
set(gcf,'color','white')

subplot(2,2,3)
plot(outputtest,kody,'o');
xlabel('Real output','FontName','Helvetica', 'Fontsize', 13);
ylabel('GLM estimate (on-line-Kody)','FontName','Helvetica', 'Fontsize', 13)
title('chi data on line kody ')
%title('TGLF data','FontName','Helvetica', 'Fontsize', 13);
set(gca, 'FontName','Helvetica', 'Fontsize', 13)
set(gcf,'color','white')

subplot(2,2,2)
plot(outputtest,initialpredicted,'o');
xlabel('Real output','FontName','Helvetica', 'Fontsize', 13);
ylabel('Linear-Regression (off-line)','FontName','Helvetica', 'Fontsize', 13)
title('chi data-linear regression ')
%title('TGLF data','FontName','Helvetica', 'Fontsize', 13);
set(gca, 'FontName','Helvetica', 'Fontsize', 13)
set(gcf,'color','white')

figure
subplot(2,2,1)
plot(outputtest,'red')
hold on
plot(ymuv,'blue')
title('CHI data(On-line)-my method ')
%legend('real data','Predicted data','FontName','Helvetica', 'Fontsize', 9)
set(gcf,'color','white')
legend({'real data','Predicted data'},'FontSize',9)

subplot(2,2,2)
plot(outputtest,'red')
hold on
plot(initialpredicted,'blue')
title('CHI data (off-line)')
%legend('real data','Predicted data','FontName','Helvetica', 'Fontsize', 9)
set(gcf,'color','white')
legend({'real data','Predicted data'},'FontSize',9)

subplot(2,2,3)
plot(outputtest,'red')
hold on
plot(kody,'blue')
title('CHI data (on-line-Kody)')
%legend('real data','Predicted data','FontName','Helvetica', 'Fontsize', 9)
set(gcf,'color','white')
legend({'real data','Predicted data'},'FontSize',9)
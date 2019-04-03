clc;
clear;
close all;
load('JM_tauth_data')
input=[r0,a0,kappa,delta,ip,b0,nebar,zeff,ploss];
inputtrain=log(input(1:2100,:)); %select the first 2100 for training
inputtest=log(input(2101:end,:)); %use the remaining data for testing
output=[tauth];
outputtrain=log(output(1:2100,:)); %select the first 2100 for training
outputtest=(output(2101:end,:)); %use the remaining data for testing
reg=MultiPolyRegress(inputtrain,outputtrain,5,'figure');

for i=1:size(inputtest,1)
NewDataPoint=inputtest(i,:);
NewScores=repmat(NewDataPoint,[length(reg.PowerMatrix) 1]).^reg.PowerMatrix;
EvalScores=ones(length(reg.PowerMatrix),1);
for ii=1:size(reg.PowerMatrix,2)
EvalScores=EvalScores.*NewScores(:,ii);
end
yhatNew=reg.Coefficients'*EvalScores ;
yhatNew=exp(yhatNew);
ypred(i,:)=yhatNew;
end

figure()
subplot(2,2,1)
plot(outputtest,ypred,'o');
xlabel('Real output','FontName','Helvetica', 'Fontsize', 13);
ylabel('GP estimate','FontName','Helvetica', 'Fontsize', 13)
title('JMtauth','FontName','Helvetica', 'Fontsize', 13);
set(gca, 'FontName','Helvetica', 'Fontsize', 13)
set(gcf,'color','white')


subplot(2,2,2)
hist(outputtest-ypred);
xlabel('Real output','FontName','Helvetica', 'Fontsize', 13);
ylabel('GP estimate','FontName','Helvetica', 'Fontsize', 13)
title('JMtauth','FontName','Helvetica', 'Fontsize', 13);
set(gca, 'FontName','Helvetica', 'Fontsize', 13)
set(gcf,'color','white')
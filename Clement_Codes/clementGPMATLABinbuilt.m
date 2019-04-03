clc;
clear;
close all;
disp( 'Author: Clement Etienam' )
disp( 'Supervisor: Professor Kody Law' )
disp( 'GP Regression Model using MATLAB functions' )

%% The first test case

load('jm_data.mat') % Load the Data set
output=[ptotped, betanped, wped];
input=[r a kappa delta bt ip neped betan zeffped]; %3 output and 9 input
[M,N]=size(output);
gprMdl1 = fitrgp(input,output(:,1),'KernelFunction','squaredexponential',...
    'OptimizeHyperparameters','auto','HyperparameterOptimizationOptions',...
    struct('AcquisitionFunctionName','expected-improvement-plus'));
gprMdl2 = fitrgp(input,output(:,2),'KernelFunction','squaredexponential',...
    'OptimizeHyperparameters','auto','HyperparameterOptimizationOptions',...
    struct('AcquisitionFunctionName','expected-improvement-plus'));
gprMdl3 = fitrgp(input,output(:,3),'KernelFunction','squaredexponential',...
    'OptimizeHyperparameters','auto','HyperparameterOptimizationOptions',...
    struct('AcquisitionFunctionName','expected-improvement-plus'));
ypred1 = resubPredict(gprMdl1);
ypred2 = resubPredict(gprMdl2);
ypred3 = resubPredict(gprMdl3);



figure()
plot(output(:,1),ypred1,'o');hold;
xlabel('Real output','FontName','Helvetica', 'Fontsize', 13);
ylabel('GP estimate','FontName','Helvetica', 'Fontsize', 13)
title('1st output RM','FontName','Helvetica', 'Fontsize', 13);
set(gca, 'FontName','Helvetica', 'Fontsize', 13)
set(gcf,'color','white')


figure()
plot(output(:,2),ypred2,'o');hold;
xlabel('Real output','FontName','Helvetica', 'Fontsize', 13);
ylabel('GP estimate','FontName','Helvetica', 'Fontsize', 13)
title('2nd output RM','FontName','Helvetica', 'Fontsize', 13);
set(gca, 'FontName','Helvetica', 'Fontsize', 13)
set(gcf,'color','white')

figure()
plot(output(:,3),ypred3,'o');hold;
xlabel('Real output','FontName','Helvetica', 'Fontsize', 13);
ylabel('GP estimate','FontName','Helvetica', 'Fontsize', 13)
title('3rd output RM','FontName','Helvetica', 'Fontsize', 13);
set(gca, 'FontName','Helvetica', 'Fontsize', 13)
set(gcf,'color','white')

%% The second test case
load('JM_tauth_data')
input4=[r0,a0,kappa,delta,ip,b0,nebar,zeff,ploss];
input4=log(input4);
output4=[tauth];
output4=log(output4);

gprMdl4 = fitrgp(input4,output4,'KernelFunction','squaredexponential',...
    'OptimizeHyperparameters','auto','HyperparameterOptimizationOptions',...
    struct('AcquisitionFunctionName','expected-improvement-plus'));

ypred4 = resubPredict(gprMdl4);


figure()
plot(output4,ypred4,'o');hold;
xlabel('Real output','FontName','Helvetica', 'Fontsize', 13);
ylabel('GP estimate','FontName','Helvetica', 'Fontsize', 13)
title('JMtauth-RM','FontName','Helvetica', 'Fontsize', 13);
set(gca, 'FontName','Helvetica', 'Fontsize', 13)
set(gcf,'color','white')

xx=norm(output4);

%% Compute L2 snf R^2
%For the JMtauth-RM data
Lerror=(norm(output4-ypred4)/norm(output4))^0.5;

%Coefficient of determination
for i=1:numel(output4)
    outputreq(i)=output4(i)-mean(output4);
end

outputreq=outputreq';
CoD=1-(norm(output4-ypred4)/norm(outputreq));

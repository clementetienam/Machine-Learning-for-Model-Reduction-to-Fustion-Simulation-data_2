clc;
clear;
close all;
disp(' We will implement the dense and sparse GP for the Tauth model' );
disp( 'Author: Dr Clement Etienam' )
disp( 'Supervisor: Professor Kody Law' )
disp( 'Polynomial regression' )

N=input( 'what size of ensemble do you want ');
p=input( 'what degree polynomial do you want ');
iter=input('Number of iteration count ');
alpha=iter;

 linecolor1 = colordg(4);


% load inputtianke.out;
% load outputtianke.out;
% load inputtianketest.out;
% load outputtianketest.out;
% Xtrain=reshape(inputtianke,500,6);
% Xtest=reshape(inputtianketest,500,6);
% ytrain=reshape(outputtianke,500,1);
% ytest=reshape(outputtianketest,500,1);

tic
load('JM_tauth_data')
input=[r0,a0,kappa,delta,ip,b0,nebar,zeff,ploss];
Xtrain=log(input(1:1000,:)); %select the first 2100 for training
Xtest=log(input(1001:2000,:)); %use the remaining data for testing
output=[tauth];
ytrain=log(output(1:1000,:)); %select the first 2100 for training
ytest=(output(1001:2000,:)); %use the remaining data for test
disp('degree polynomial 3')
sizetrain2=size(Xtrain,2);
sizetrain1=size(Xtrain,1);
time=size(ytrain,1);
Time=(1:time)';
sigma=1;
for ii=1:iter
%% Weights for the input to hidden layer 1
if ii==1
    disp('Weights and biases are initialised initially')
for i =1:N
Theta=sigma*randn(sizetrain2,p);
Thetain=reshape(Theta,sizetrain2*p,1);
constantin=sigma*randn(sizetrain1,1);
Thetainitial(:,i)=Thetain;
constantinitial(:,i)=constantin;
end
%%
Thetanow=[Thetainitial;constantinitial];
else
    disp('Weights are gotten from update')
    Thetanow=Thetaupdated;
end

%% Forwarding  for all the ensemble

parfor i =1:N
    a1=Thetanow(1:9,i);
    a2=Thetanow(10:18,i);
    a3=Thetanow(19:27,i);
    a4=Thetanow(28:sizetrain1+27,i);
% From input layer to hidden layer 1;
p1=((Xtrain*a1)+(Xtrain.^2*a2)+(Xtrain.^3*a3))+a4;
simulated(:,i)=p1;
end
%%
disp('Trainig Error for current iteration')
for i=1:N
    Trainingerroall(:,i)=immse(simulated(:,i),reshape(ytrain,sizetrain1,1));
end
TOTALERROR=ones(N,1);
TOTALERROR=(Trainingerroall./std(reshape(ytrain,sizetrain1,1)));
% TOTALERROR=TOTALERROR./66;
jj=min(TOTALERROR);
index = TOTALERROR; 
bestnorm = find(index == min(index));
fprintf('The best Norm Realization is number %i with value %4.4f \n',bestnorm,jj);

clementbest(ii,:)=jj;
bestnormseries(ii,:)=bestnorm;
reali=[1:N]';

%%
simulatedall(:,:,ii)=simulated;
Thetaupdated=mainpoly2(N,reshape(ytrain,sizetrain1,1),simulated,alpha,Thetanow);
disp('Now optimise with ES-MDA')
Thetanow=Thetaupdated;
fprintf('Finished Iteration %d .\n', ii);
end
%% Plot the values
simulatedfirst=simulatedall(:,:,1);
simulatedlast=simulatedall(:,:,iter);
figure()
 subplot(3,2,1)
 plot(Time,exp(simulatedfirst(:,1:N)),'Color',linecolor1,'LineWidth',2)
xlabel('Days','FontName','Helvetica', 'Fontsize', 13);
ylabel('Oil price in $','FontName','Helvetica', 'Fontsize', 13);
% ylim([1 80])
title('Match for First Run','FontName','Helvetica', 'Fontsize', 13)
a = get(gca,'Children');
hold on
plot(Time,exp(ytrain'),'r','LineWidth',2)
b = get(gca,'Children');
 set(gca, 'FontName','Helvetica', 'Fontsize', 13)
set(gcf,'color','white')
% line([2500 2500], [0 1000],'Color','black','LineStyle','--')
h = [b;a];
legend(h,'True signal','Realisations','location','northeast');
hold off

 subplot(3,2,2)
 plot(Time,exp(simulatedlast(:,1:N)),'Color',linecolor1,'LineWidth',2)
xlabel('Days','FontName','Helvetica', 'Fontsize', 13);
ylabel('Oil price in $','FontName','Helvetica', 'Fontsize', 13);
% ylim([1 80])
title('Match for last run','FontName','Helvetica', 'Fontsize', 13)
a = get(gca,'Children');
hold on
plot(Time,exp(ytrain'),'r','LineWidth',2)
b = get(gca,'Children');
 set(gca, 'FontName','Helvetica', 'Fontsize', 13)
set(gcf,'color','white')
% line([2500 2500], [0 1000],'Color','black','LineStyle','--')
h = [b;a];
legend(h,'True signal','Realisations','location','northeast');
hold off

 subplot(3,2,3)
 plot(Time,exp(simulatedlast(:,bestnorm)),'k','LineWidth',2)
xlabel('Days','FontName','Helvetica', 'Fontsize', 13);
ylabel('Oil price in $','FontName','Helvetica', 'Fontsize', 13);
% ylim([1 80])
title('Mean with True function','FontName','Helvetica', 'Fontsize', 13)
hold on
plot(Time,exp(ytrain'),'r','LineWidth',2)
 set(gca, 'FontName','Helvetica', 'Fontsize', 13)
set(gcf,'color','white')
legend('Mean function','True signal','location','northeast');
hold off

 subplot(3,2,4)
 plot(1:size(clementbest,1),clementbest,'k','LineWidth',2)
xlabel('Iteration','FontName','Helvetica', 'Fontsize', 13);
ylabel('Cost function value','FontName','Helvetica', 'Fontsize', 13);
% ylim([2 840])
title('Cost Function Value with Iteration','FontName','Helvetica', 'Fontsize', 11)
subplot(3,2,5)
plot(exp(ytrain'),exp(simulatedlast(:,bestnorm)),'o');hold;
xlabel('Real output','FontName','Helvetica', 'Fontsize', 13);
ylabel('Machine estimate','FontName','Helvetica', 'Fontsize', 13)
title('Tauth data','FontName','Helvetica', 'Fontsize', 13);
set(gca, 'FontName','Helvetica', 'Fontsize', 13)
set(gcf,'color','white')
saveas(gcf,sprintf('RMS_iter.fig'))
close(figure)

%% Gte the optimised weights using the best one in the ensemble
Thetaoptimnised=Thetanow(:,bestnorm);

disp('Now test the forwarding with this new weights')
    a1=Thetaoptimnised(1:9,1);
    a2=Thetaoptimnised(10:18,1);
    a3=Thetaoptimnised(19:27,1);
    a4=Thetaoptimnised(28:sizetrain1+27,1);

p1=((Xtrain*a1)+(Xtrain.^2*a2)+(Xtrain.^3*a3))+a4;


TrainError=immse(p1,ytrain);
fprintf('The Training error for best norm is %4.4f \n',TrainError);
%%
disp('Now predict with test inputs')
% From input layer to hidden layer 1;
p1=((Xtest*a1)+(Xtest.^2*a2)+(Xtest.^3*a3))+a4;

TestError=immse(exp(p1),ytest);
fprintf('The Testing error is %4.4f \n',TestError);

figure()
subplot(2,2,1)
 plot(Time,exp(p1),'k','LineWidth',2)
xlabel('Days','FontName','Helvetica', 'Fontsize', 13);
ylabel('Oil price in $','FontName','Helvetica', 'Fontsize', 13);
% ylim([1 70])
title('Predicted Oil value in $','FontName','Helvetica', 'Fontsize', 13)
hold on
plot(Time,(ytest'),'r','LineWidth',2)
 set(gca, 'FontName','Helvetica', 'Fontsize', 13)
set(gcf,'color','white')
legend('Mean function','True test signal','location','northeast');
hold off

subplot(2,2,2)
 plot((ytest),exp(p1),'k','LineWidth',2)
xlabel('True','FontName','Helvetica', 'Fontsize', 13);
ylabel('machine','FontName','Helvetica', 'Fontsize', 13);
 set(gca, 'FontName','Helvetica', 'Fontsize', 13)
set(gcf,'color','white')
% ylim([1 70])
title('Predicted Oil value in $','FontName','Helvetica', 'Fontsize', 13)

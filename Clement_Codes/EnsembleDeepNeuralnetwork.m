clc;
clear;
close all;
disp(' We will implement the dense and sparse GP for the Tauth model' );
disp( 'Author: Dr Clement Etienam' )
disp( 'Supervisor: Professor Kody Law' )
disp( 'Ensemble Deep neural network' )
disp('Develop a regression model using ENN')
disp('Using 2 hidden layers with 5 neurons each')
N=input( 'what size of ensemble do you want ');
iter=input('Number of iteration count ');
alpha=iter;
disp('Activation function is ReLU')
disp( 'Inverse problem is solved using batch es-mda')
disp(' The network consists of 2 hidden layer with 5 neuron each')
disp('use sigmoid for the activation of inner layers but linear function for output layer in regression problem')

 linecolor1 = colordg(4);
Xtrain=[1 9 5 7 3 11];
ytrain=((0.4.*Xtrain)-(0.25.*(Xtrain.^2))+(0.78.*(Xtrain.^3))-(0.034.*(Xtrain.^4)))+2.4;
%ytrain=[3 5 9 13 17,19];
time=size(ytrain,2);
Time=(1:time)';
Xtest=[4 2 8 6 10 4.8];
ytest=((0.4.*Xtest)-(0.25.*(Xtest.^2))+(0.78.*(Xtest.^3))-(0.034.*(Xtest.^4)))+2.4;
for ii=1:iter
%% Weights for the input to hidden layer 1
if ii==1
    disp('Weights and biases are initialised initially')
for i =1:N
a1 = normrnd(0,1,5,6); % weights for the input layer to hidden layer 1
a1=reshape(a1,30,1);
a1all(:,i)=a1;

b1= normrnd(0,1,5,1); % biases for the input layer to hidden layer 1
b1=reshape(b1,5,1);
b1all(:,i)=b1;
end

%% Weights for the hidden layer 1 to hidden layer 2
for i=1:N
a2 = normrnd(0,1,5,5); % weights for the hidden layer 1 to hidden layer 2
a2=reshape(a2,25,1);
a2all(:,i)=a2;

b2= normrnd(0,1,5,1); % biases 
b2=reshape(b2,5,1);
b2all(:,i)=b2;
end

%% Weights for the hidden layer 2 to output layer
for i=1:N
a3 = normrnd(0,1,6,5); % weights for the hidden layer 2 to hidden layer 1
a3=reshape(a3,30,1);
a3all(:,i)=a3;

b3= normrnd(0,1,6,1); % biases for the input layer to hidden layer 1
b3=reshape(b3,6,1);
b3all(:,i)=b3;
end
aensemble=[a1all;a2all;a3all];
bensemble=[b1all;b2all;b3all];
else
    disp('Weights are gotten from update')
    aensemble=aupdated;
    bensemble=bupdated;
end

%% Forwarding  for all the ensemble

parfor i =1:N
    a1=aensemble(1:30,i);
    a2=aensemble(31:55,i);
    a3=aensemble(56:85,i);
    
    b1=aensemble(1:5,i);
    b2=aensemble(6:10,i);
    b3=aensemble(11:16,i);
    
    a1=reshape(a1,5,6);
    a2=reshape(a2,5,5);
    a3=reshape(a3,6,5);
% From input layer to hidden layer 1;
p1=(a1*Xtrain')+b1;
p1(p1<0)=0; %Relu function

% p1=1+exp(-p1);
% p1=(p1).^(-1);

% From hidden layer 1 to hidden layer 2;
p2=(a2*p1)+b2;
p2(p2<0)=0;
% p2=1+exp(-p2);
% p2=(p2).^(-1);

% From hidden layer 2 to output layer;
p3=(a3*p2)+b3;

% p3=1+exp(-p3);
% p3=(p3).^(-1);
simulated(:,i)=p3;
end
%%
disp('Trainig Error for current iteration')
for i=1:N
    Trainingerroall(:,i)=immse(simulated(:,i),reshape(ytrain,6,1));
end
TOTALERROR=ones(N,1);
TOTALERROR=(Trainingerroall./std(reshape(ytrain,6,1)));
TOTALERROR=TOTALERROR./66;
jj=min(TOTALERROR);
index = TOTALERROR; 
bestnorm = find(index == min(index));
fprintf('The best Norm Realization is number %i with value %4.4f \n',bestnorm,jj);

clementbest(ii,:)=jj;
bestnormseries(ii,:)=bestnorm;
reali=[1:N]';

% if ii==1|iter
%  figure()
%  subplot(2,2,1)
%  bar(reali,index,'cyan');
%  xlabel('Realizations', 'FontName','Helvetica', 'Fontsize', 13);
%  ylabel('RMSE value', 'FontName','Helvetica', 'Fontsize', 13);
%  title('Cost function for Realizations','FontName','Helvetica', 'Fontsize', 13)
%  set(gcf,'color', 'white');
%  hold on
%  scatter(reali,index,'black','filled');
%  xlabel('Realizations', 'FontName','Helvetica', 'Fontsize', 13)
%  ylabel('RMSE value', 'FontName','Helvetica', 'Fontsize', 13)
%  hold off
%  xlim([1,N]);
%  
%  subplot(2,2,2)
%  %subplot(2,2,4)
%  plot(Time,simulated(:,1:N),'Color',linecolor1,'LineWidth',2)
% xlabel('Number of elements (days)','FontName','Helvetica', 'Fontsize', 13);
% ylabel('y value','FontName','Helvetica', 'Fontsize', 13);
% title('ENN match','FontName','Helvetica', 'Fontsize', 13)
% a = get(gca,'Children');
% hold on
% plot(Time,ytrain','r','LineWidth',2)
% b = get(gca,'Children');
%  set(gca, 'FontName','Helvetica', 'Fontsize', 13)
% set(gcf,'color','white')
% % line([2500 2500], [0 1000],'Color','black','LineStyle','--')
% h = [b;a];
% legend(h,'True signal','Realisations','location','northeast');
% hold off
%  saveas(gcf,sprintf('RMS_iter%d.fig',ii))
% close(figure)
% end
%%
simulatedall(:,:,ii)=simulated;
[aupdated,bupdated]=main_ESMDA(N,reshape(ytrain,6,1),simulated,alpha,aensemble,bensemble);
disp('Now optimise with ES-MDA')
aensemble=aupdated;
bensemble=bupdated;
fprintf('Finished Iteration %d .\n', ii);
end
%% Plot the values
simulatedfirst=simulatedall(:,:,1);
simulatedlast=simulatedall(:,:,iter);
figure()
 subplot(2,2,1)
 plot(Time,simulatedfirst(:,1:N),'Color',linecolor1,'LineWidth',2)
xlabel('Number of elements','FontName','Helvetica', 'Fontsize', 13);
ylabel('y value','FontName','Helvetica', 'Fontsize', 13);
ylim([2 840])
title('ENN match for first run','FontName','Helvetica', 'Fontsize', 13)
a = get(gca,'Children');
hold on
plot(Time,ytrain','r','LineWidth',2)
b = get(gca,'Children');
 set(gca, 'FontName','Helvetica', 'Fontsize', 13)
set(gcf,'color','white')
% line([2500 2500], [0 1000],'Color','black','LineStyle','--')
h = [b;a];
legend(h,'True signal','Realisations','location','northeast');
hold off

 subplot(2,2,2)
 plot(Time,simulatedlast(:,1:N),'Color',linecolor1,'LineWidth',2)
xlabel('Number of elements','FontName','Helvetica', 'Fontsize', 13);
ylabel('y value','FontName','Helvetica', 'Fontsize', 13);
ylim([2 840])
title('ENN match for last run','FontName','Helvetica', 'Fontsize', 13)
a = get(gca,'Children');
hold on
plot(Time,ytrain','r','LineWidth',2)
b = get(gca,'Children');
 set(gca, 'FontName','Helvetica', 'Fontsize', 13)
set(gcf,'color','white')
% line([2500 2500], [0 1000],'Color','black','LineStyle','--')
h = [b;a];
legend(h,'True signal','Realisations','location','northeast');
hold off

 subplot(2,2,3)
 plot(Time,simulatedlast(:,bestnorm),'k','LineWidth',2)
xlabel('Number of elements','FontName','Helvetica', 'Fontsize', 13);
ylabel('y value','FontName','Helvetica', 'Fontsize', 13);
ylim([2 840])
title('Mean with True function','FontName','Helvetica', 'Fontsize', 13)
hold on
plot(Time,ytrain','r','LineWidth',2)
 set(gca, 'FontName','Helvetica', 'Fontsize', 13)
set(gcf,'color','white')
legend('Mean function','True signal','location','northeast');
hold off
 saveas(gcf,sprintf('RMS_iter%d.fig',ii))
close(figure)
%% Get the optimised weights (using the mean)
% aoptimnised=mean(aensemble,2);
% disp('Now test the forwarding with this new weights')
% a1=aoptimnised(1:30,:);
%     a2=aoptimnised(31:55,:);
%     a3=aoptimnised(56:85,:);
%     
%     a1=reshape(a1,5,6);
%     a2=reshape(a2,5,5);
%     a3=reshape(a3,6,5);
% % From input layer to hidden layer 1;
% p1=a1*Xtrain';
% p1(p1<0)=0;
% % p1=1+exp(-p1);
% % p1=(p1).^(-1);
% 
% % From hidden layer 1 to hidden layer 2;
% p2=a2*p1;
% p2(p2<0)=0;
% % p2=1+exp(-p2);
% % p2=(p2).^(-1);
% 
% % From hidden layer 2 to output layer;
% p3=a3*p2;
% % p3=1+exp(-p3);
% % p3=(p3).^(-1);
% 
% TrainError=immse(p3,ytrain');
% fprintf('The Training error is %4.4f \n',TrainError);
%% Gte the optimised weights using the best one in the ensemble
aoptimnised=aensemble(:,bestnorm);
boptimnised=bensemble(:,bestnorm);
disp('Now test the forwarding with this new weights')
    a1=aoptimnised(1:30,:);
    a2=aoptimnised(31:55,:);
    a3=aoptimnised(56:85,:);
    
    b1=aoptimnised(1:5,:);
    b2=aoptimnised(6:10,:);
    b3=aoptimnised(11:16,:);
    
    a1=reshape(a1,5,6);
    a2=reshape(a2,5,5);
    a3=reshape(a3,6,5);
% From input layer to hidden layer 1;
p1=(a1*Xtrain')+b1;
p1(p1<0)=0;
% p1=1+exp(-p1);
% p1=(p1).^(-1);

% From hidden layer 1 to hidden layer 2;
p2=(a2*p1)+b2;
p2(p2<0)=0;
% p2=1+exp(-p2);
% p2=(p2).^(-1);

% From hidden layer 2 to output layer;
p3=(a3*p2)+b3;
% p3=1+exp(-p3);
% p3=(p3).^(-1);

TrainError=immse(p3,ytrain');
fprintf('The Training error for best norm is %4.4f \n',TrainError);
%%
disp('Now predict with test inputs')
% From input layer to hidden layer 1;
p1=(a1*Xtest')+b1;
p1(p1<0)=0; %RelU
% p1=1+exp(-p1);
% p1=(p1).^(-1);

% From hidden layer 1 to hidden layer 2;
p2=(a2*p1)+b2;
p2(p2<0)=0; % ReLU
% p2=1+exp(-p2);
% p2=(p2).^(-1);

% From hidden layer 2 to output layer;
p3test=(a3*p2)+b3;
% p3=1+exp(-p3);
% p3test=(p3).^(-1);

TestError=immse(p3test,ytest');
fprintf('The Testing error is %4.4f \n',TestError);

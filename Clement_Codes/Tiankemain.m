clc;
clear;
close all;
disp(' We will implement the dense and sparse GP for the Tauth model' );
disp( 'Author: Dr Clement Etienam' )
disp( 'Supervisor: Professor Kody Law' )
disp( 'Ensemble Deep neural network' )
disp('Develop a regression model using ENN')
disp('Using 2 hidden layers with 3 neurons each')
N=input( 'what size of ensemble do you want ');
iter=input('Number of iteration count ');
alpha=iter;
disp('Activation function is ReLU')
disp( 'Inverse problem is solved using batch es-mda')
disp(' The network consists of 2 hidden layer with 5 neuron each')
disp('use sigmoid for the activation of inner layers but linear function for output layer in regression problem')

 linecolor1 = colordg(4);
Xtrain=[2 3 6 4 1];
ytrain=(2.*Xtrain)+1;
%ytrain=[3 5 9 13 17,19];
time=size(ytrain,2);
Time=(1:time)';
Xtest=[1 5 1.6 4.8 3.9];
ytest=(2.*Xtest)+1;
for ii=1:iter
%% Weights for the input to hidden layer 1
if ii==1
    disp('Weights and biases are initialised initially')
for i =1:N
a1 = normrnd(0,1,3,5); % weights for the input layer to hidden layer 1
a1=reshape(a1,15,1);
a1all(:,i)=a1;

b1= normrnd(0,1,3,1); % biases for the input layer to hidden layer 1
b1=reshape(b1,3,1);
b1all(:,i)=b1;
end

%% Weights for the hidden layer 1 to hidden layer 2
for i=1:N
a2 = normrnd(0,1,3,3); % weights for the hidden layer 1 to hidden layer 2
a2=reshape(a2,9,1);
a2all(:,i)=a2;

b2= normrnd(0,1,3,1); % biases 
b2=reshape(b2,3,1);
b2all(:,i)=b2;
end

%% Weights for the hidden layer 2 to output layer
for i=1:N
a3 = normrnd(0,1,5,3); % weights for the hidden layer 2 to hidden layer 1
a3=reshape(a3,15,1);
a3all(:,i)=a3;

b3= normrnd(0,1,5,1); % biases for the input layer to hidden layer 1
b3=reshape(b3,5,1);
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
    a1=aensemble(1:15,i);
    a2=aensemble(16:24,i);
    a3=aensemble(25:39,i);
    
    b1=aensemble(1:3,i);
    b2=aensemble(4:6,i);
    b3=aensemble(7:11,i);
    
    a1=reshape(a1,3,5);
    a2=reshape(a2,3,3);
    a3=reshape(a3,5,3);
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
    Trainingerroall(:,i)=immse(simulated(:,i),reshape(ytrain,5,1));
end
TOTALERROR=ones(N,1);
TOTALERROR=(Trainingerroall./std(reshape(ytrain,5,1)));
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
[aupdated,bupdated]=Tianke2(N,reshape(ytrain,5,1),simulated,alpha,aensemble,bensemble);
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
ylim([2 150])
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
ylim([2 150])
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
ylim([2 150])
title('Mean with True function','FontName','Helvetica', 'Fontsize', 13)
hold on
plot(Time,ytrain','r','LineWidth',2)
 set(gca, 'FontName','Helvetica', 'Fontsize', 13)
set(gcf,'color','white')
legend('Mean function','True signal','location','northeast');
hold off

 subplot(2,2,4)
 plot(1:size(clementbest,1),clementbest,'k','LineWidth',2)
xlabel('Iteration','FontName','Helvetica', 'Fontsize', 13);
ylabel('Cost function value','FontName','Helvetica', 'Fontsize', 13);
% ylim([2 840])
title('Cost Function Value with Iteration','FontName','Helvetica', 'Fontsize', 11)
hold on
saveas(gcf,sprintf('RMS_iter.fig'))
close(figure)

%% Gte the optimised weights using the best one in the ensemble
aoptimnised=aensemble(:,bestnorm);
boptimnised=bensemble(:,bestnorm);
disp('Now test the forwarding with this new weights')
    a1=aoptimnised(1:15,:);
    a2=aoptimnised(16:24,:);
    a3=aoptimnised(25:39,:);
    
    b1=aoptimnised(1:3,:);
    b2=aoptimnised(4:6,:);
    b3=aoptimnised(7:11,:);
    
    a1=reshape(a1,3,5);
    a2=reshape(a2,3,3);
    a3=reshape(a3,5,3);
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

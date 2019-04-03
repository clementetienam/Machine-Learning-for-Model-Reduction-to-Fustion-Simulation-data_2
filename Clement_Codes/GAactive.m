clc;
clear;
close all;
%% GP classification/Regression model for the Chi data
disp(' We will implement the active learning sparse GP for the sin(x) function' );
disp( 'Author: Dr Clement Etienam' )
disp( 'Supervisor: Professor Kody Law' )
disp( 'Supervised active learning Modelling' )
set(0,'defaultaxesfontsize',20); format long

tic
load('JM_tauth_data')
input=[r0,a0,kappa,delta,ip,b0,nebar,zeff,ploss];
inputtrain=log(input(1:2100,:)); %select the first 2100 for training
ininitial=inputtrain(1:500,:);
inpool=inputtrain(501:end,:);
inputtest=log(input(2101:end,:)); %use the remaining data for testing
output=[tauth];
outputtrain=log(output(1:2100,:)); %select the first 2100 for training
outpool=outputtrain(501:end,:);
outinitial=outputtrain(1:500,:);
outputtest=(output(2101:end,:)); %use the remaining data for test
%% For online regression

%%

%% Train the initial GP Sparse approximation, The initial Gp is important too
meanfunc=[];% empty: don't use a mean function
covfunc = @covSEiso; hyp.cov = log([9.5;12.5]); hyp.lik = log(0.99);
disp('likfunc = @likGauss; sn = 0.1; hyp.lik = log(sn);')
likfunc = @likGauss; sn = 0.99; hyp.lik = log(sn);

disp('First train the GP with these initial points')
hyp = minimize(hyp,@gp,-100,@infGaussLik,meanfunc,covfunc,likfunc,ininitial,outinitial);%minimise the hyperparamters
hypinital=hyp;
disp('Predict with initial Gp')
y_predini=gp(hyp,@infGaussLik,meanfunc,covfunc,likfunc, ininitial, outinitial, inputtest);%Inference with GP
y_predini=exp(y_predini);
figure()
scatter(1:size(inputtest,1),outputtest,'k')
hold on
plot(1:size(inputtest,1),y_predini,'r')
hold off
%legend('real data','Predicted data','FontName','Helvetica', 'Fontsize', 9)
set(gcf,'color','white')
%% 
%[~,ys2v] = gp(hyp,infv,meanfunc,cov,lik, inputtrainGP, outputtrainGP, inputtest(index2,:));%Inference with GP
n_queries =1000; %(it should be half the number of sample points)
for idx=1:n_queries
 disp('predict from the pool or oracle')
hyp.cov=hyp.cov;
hyp.lik=hyp.lik;
[~,ys2v] = gp(hyp,@infGaussLik,meanfunc,covfunc,likfunc, ininitial, outinitial, inpool);%Inference with GP
[P_max,id_max]=max(ys2v);
disp('select those new points')
inpoolnew=inpool(id_max,:);
outpoolnew=outpool(id_max,:);
disp('append those new points and retrain the Gp')
newinitial=[ininitial; inpoolnew];
newoutinitial=[outinitial;outpoolnew];

ininitial=newinitial;
outinitial=newoutinitial;
disp('Retrain the Gp with these augmented points')
hyp = minimize(hyp,@gp,-100,@infGaussLik,meanfunc,covfunc,likfunc,ininitial,outinitial);%minimise the hyperparamters
fprintf('Finished query point %d out of %d,%d left to go .\n', idx,n_queries,(n_queries-idx));
end


%% Prediction

[regressoutput2,ys2v] = gp(hyp,@infGaussLik,meanfunc,covfunc,likfunc, ininitial, outinitial, inputest);%Inference with GP

clement1=exp(regressoutput2);

figure()
scatter(1:size(inputtest,1),outputtest,'k')
hold on
plot(1:size(inputtest,1),clement1)
hold off

set(gcf,'color','white')
clc;
clear;
close all;
%% GP classification model for the Dense and Sparse data
disp(' We will implement the sparse GP for the Chi model' );
disp( 'Author: Dr Clement Etienam' )
disp( 'Supervisor: Professor Kody Law' )
disp( 'Supervised classfication Modelling' )
set(0,'defaultaxesfontsize',20); format long

dd=importdata('chi_itg.dat',' ',1);
test=dd.data;
out=test;


test=out;
X=log(test(1:600000,1:10));
y=(test(1:600000,11));
y2=zeros(600000,1);
for i=1:600000
    if y(i)==0
        y2(i)=-1;
    end
    
    if y(i)>0
        y2(i)=1;
    end
        
end
y=y2;
inputtrain=X(1:290000,:);
outputtrain=y(1:290000,:);
inputtest=X(290000+1:end,:);
outputtest=y(290000+1:end,:);
p=10;
%% Sparse approximation

meanfunc=@meanConst;% empty: don't use a mean function


 lik = @likErf;    
 inf = @infLaplace;   
 cov = @covSEard; 
 hyp.mean = 0;
 ell = 1.0; 
 sf = 1.0;
 hyp.cov = log([ell ell ell ell ell ell ell ell ell ell sf]);
p=size(inputtest,2);
for j=1:p
xu = normrnd(0,1,100,1); 
xsparse(:,j)=xu;
end
hyp.xu=xsparse;
cov = {'apxSparse',{cov},xsparse};           % inducing points
infv  = @(varargin) inf(varargin{:},struct('s',1.0));
infr=@infLaplace;   

hyp = minimize(hyp,@gp,-100,infv,meanfunc,cov,lik,inputtrain,outputtrain);%minimise the hyperparamters
parfor ii=1:size(inputtest,1);
[ymuv,ys2v] = gp(hyp,infv,meanfunc,cov,lik, inputtrain, outputtrain, inputtest);%Inference with GP
ymuvall(ii,:)=ymuv;
end
%% Use Bayes model
% Bayes_Model = fitcnb(inputtrain, outputtrain, 'Distribution','kernel');
% parfor ii=1:size(inputtest,1);
% [Bayes_Predicted] = predict(Bayes_Model,inputtest);
% Byaesall(ii,:)=Bayes_Predicted;
% end
% disp('finished Naieve bayes method')
% %% Use decision tree
% b2 = TreeBagger(250,inputtrain,outputtrain,'oobvarimp','on');
% parfor ii=1:size(inputtest,1);
% [Predicted, Class_Score] = predict(b2,inputtest(ii,:));
% predtree(ii,:)=Predicted;
% end
% disp('finished decision tree method')
%% Discriminant analysis model
[Mdl]= fitcdiscr(inputtrain,outputtrain,...
    'OptimizeHyperparameters','auto',...
    'HyperparameterOptimizationOptions',struct('Holdout',0.1,'MaxObjectiveEvaluations',100,...
    'AcquisitionFunctionName','expected-improvement-plus','Repartition',1));
parfor ii=1:size(inputtest,1);
labelDA = predict(Mdl,inputtest(ii,:));
Daall(ii,:)=labelDA;
end
EVALdaal = Evaluate(outputtest,Daall);
disp('finished Discriminant analysis method')
poolobj = gcp('nocreate');
delete(poolobj);
%% Ensemble tree
[Mdltree] = fitctree(inputtrain,outputtrain,...
    'OptimizeHyperparameters','auto',...
    'HyperparameterOptimizationOptions',struct('Holdout',0.1,...
    'AcquisitionFunctionName','expected-improvement-plus'));
parfor ii=1:size(inputtest,1);
labelDA = predict(Mdltree,inputtest(ii,:));
Daall2(ii,:)=labelDA;
end
EVALdaal2 = Evaluate(outputtest,Daall2);
disp('finished Ensemble tree method')
poolobj = gcp('nocreate');
delete(poolobj);
%%
% %% SVM
% [Mdl2]= fitcecoc(inputtrain,outputtrain,...
%     'OptimizeHyperparameters','auto',...
%     'HyperparameterOptimizationOptions',struct('Holdout',0.3,...
%     'AcquisitionFunctionName','expected-improvement-plus'));
% parfor ii=1:size(inputtest,1);
% labelSVM2 = fitcecoc(Mdl2,inputtest(ii,:));
% SVMall(ii,:)=labelSVM2;
% end
% disp('finished support vector method')
% poolobj = gcp('nocreate');
% delete(poolobj);
%% Ensemble for learners-Slow
% [Mdl3]= fitcensemble(inputtrain,outputtrain,...
%     'OptimizeHyperparameters','auto',...
%     'HyperparameterOptimizationOptions',struct('Holdout',0.3,...
%     'AcquisitionFunctionName','expected-improvement-plus'));
% parfor ii=1:size(inputtest,1);
% labelEL = fitcensemble(Mdl3,inputtest(ii,:));
% labelall(ii,:)=labelEL;
% end
% disp('finished ensemble for learners method')
% poolobj = gcp('nocreate');
% delete(poolobj);
%% KNN classfier
% [Mdl4]= fitcknn(inputtrain,outputtrain,...
%     'OptimizeHyperparameters','auto',...
%     'HyperparameterOptimizationOptions',struct('Holdout',0.3,...
%     'AcquisitionFunctionName','expected-improvement-plus'));
% parfor ii=1:size(inputtest,1);
% labelKNN = fitcensemble(Mdl4,inputtest(ii,:));
% knnall(ii,:)=labelKNN;
% end
% disp('finished KNN method')
% poolobj = gcp('nocreate');
% delete(poolobj);
%% Ensemble moderate
% [Mdl5]= fitcensemble(inputtrain,outputtrain,...
%     'OptimizeHyperparameters','auto',...
%     'HyperparameterOptimizationOptions',struct('Holdout',0.3,...
%     'AcquisitionFunctionName','expected-improvement-plus'));
% parfor ii=1:size(inputtest,1);
% labelSVM = fitcsvm(Mdl5,inputtest(ii,:));
% svmall(ii,:)=labelSVM;
% end
% EVALdaal3 = Evaluate(outputtest,svmall);
% disp('finished SVM moerate method')
% poolobj = gcp('nocreate');
% delete(poolobj);
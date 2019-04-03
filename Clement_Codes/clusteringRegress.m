clc;
clear;
close all;
%% GP regression model for the Dense and Sparse data
disp(' We will implement the clustering and Regression on Chi model' );
disp( 'Author: Dr Clement Etienam' )
disp( 'Supervisor: Professor Kody Law' )
disp( 'Adaptive GP Regression Modelling' )
set(0,'defaultaxesfontsize',20); format long

%% The idea is we want to first cluster the data then do GP independently fir those data
%plotting an histogram can help determine the number of bins(cluster to
%use)
%% Import and analyse the data
dd=importdata('chi_itg.dat',' ',1);
test=dd.data;
testin=log(test(:,1:10));
testout=test(:,11);
test=zeros([],[]);
test(:,1:10)=testin;
test(:,11)=testout;
X=test(1:500000,:);

disp('visualise the output data')
hist(X(:,11),10); % Histogram with 10 bins
cc=max(X(:,11));
%% Cluster the data based on output first
numClust=ceil(cc);
numClust=100;
%numClust=input(' Enter the  required cluster you want  '); %choose 8-10 not too much
% Using too many a cluster would overfit the data

options = statset('UseParallel',1);
[idx,C] = kmeans(X(:,11),numClust,'MaxIter',10000,...
    'Display','final','Replicates',10,'Options',options);

for i=1:numClust
indices=find(idx==i);
test1=test((indices),:);
testall{:,i}=test1;
end
%% Get representative of each cluster
disp( 'Now get the representative for each cluster')
for i=1:numClust
    dd=testall{:,i};
    [idx,C] = kmeans(dd(:,1:10),1,'MaxIter',10000,...
    'Display','final','Replicates',10,'Options',options);

ccall(i,:)=C; 
end

%% Start GP learning and prediction
testingdata=test(500001:end,:);

inputtest=testingdata(:,1:10);
ff=size(inputtest,1);
outputtest=testingdata(:,11);
disp( 'train GP and predict for each cluster set')
% initializae the GP hyper-parameters
%  meanfunc=[];
%  n = 30; sn = 0.99;
%  lik = {@likGauss};    
%  hyp.lik = log(sn); 
%  inf = @infGaussLik;
%  cov = {@covSEiso}; 
%  hyp.cov = log([9.5;12.5]); % Matern class d=5
%  p=size(inputtest,2);
% for j=1:p
% xu = normrnd(0,1,100,1); 
% xsparse(:,j)=xu;
% end
% hyp.xu=xsparse;
% cov = {'apxSparse',cov,xsparse};           % inducing points
% infv  = @(varargin) inf(varargin{:},struct('s',1.0));


for i=1:ff
  meanfunc=[];
 n = 30; sn = 0.99;
 lik = {@likGauss};    
 hyp.lik = log(sn); 
 inf = @infGaussLik;
 cov = {@covSEiso}; 
 hyp.cov = log([9.5;12.5]); % Matern class d=5
 p=size(inputtest,2);
for j=1:p
xu = normrnd(0,1,100,1); 
xsparse(:,j)=xu;
end
hyp.xu=xsparse;
cov = {'apxSparse',cov,xsparse};           % inducing points
infv  = @(varargin) inf(varargin{:},struct('s',1.0));
 newdata=inputtest(i,:);
% Get the cluster the new test input belongs to first
for j=1:numClust
    ccmeans=ccall(j,:);
    value(j,:)=norm(newdata'-ccmeans');
end
jj=min(value);
indexfinal = value; 
bestnorm = find(indexfinal == min(indexfinal)); % Thus is the closest cluster the new point belongs to
newtraining=testall{:,bestnorm};% Retrieve the point from the data
inputtrain=newtraining(:,1:10);
outputtrain=newtraining(:,11);
hyp = minimize(hyp,@gp,-100,infv,meanfunc,cov,lik,inputtrain,outputtrain);%minimise the hyperparamters
[ymuv,ys2v] = gp(hyp,infv,meanfunc,cov,lik, inputtrain, outputtrain, newdata);%Inference with GP
theanswer(i,:)=ymuv;
fprintf('Finished testpoint %d out of %d .\n', i,ff);
end


%% Get the best k
% [idx,C,SUMD,K]=best_kmeans(X(1:1000,11));
% for i=1:K
% indices=find(idx==i);
% test1=test((indices),:);
% testall{:,i}=test1;
% end

%% Get the best K using Heirachcal clustering
% eucD = pdist(X(1:1000,:),'euclidean');
% %%cc=cophenet(clustTreeEuc,eucD);
% clustTreeEuc = linkage(eucD,'average');
% [h,nodes] = dendrogram(clustTreeEuc,0);
% h_gca = gca;
% h_gca.TickDir = 'out';
% h_gca.TickLength = [.002 0];
%h_gca.XTickLabel = [];




clc;
clear;
close all;
%% GP regression model for the Dense and Sparse data
disp(' We will implement the sparse GP for the TGLF model' );
disp( 'Author: Clement Etienam' )
disp( 'Supervisor: Professor Kody Law' )
disp( 'GP Regression Modelling' )
set(0,'defaultaxesfontsize',20); format long

dd=load ('tglf_plus.mat');
test=dd.test;

noEM2= readtable('TGLFdata.csv');
test2 = noEM2{:,1:28};
test2output=test2;
test2input=test2;
output=[test2output(:,3),test2output(:,8),test2output(:,17),test2output(:,18),test2output(:,23),test2output(:,25)];
test2input(:,[3,8,17,18,23,25])=[];
seconddata=zeros(398892,28);
seconddata(:,1:22)=test2input;
seconddata(:,23:28)=output;

overalldata=zeros(895618,28);
overalldata(1:496726,:)=test;
overalldata(496727:895618,:)=seconddata;

 overalldata(any(isnan(overalldata), 2), :) = [];

test=overalldata;
[v,i]=find(sum(abs(test')) < 1e3);
% Ni=size(i,2);
% H=sparse(Ni,N);H(:,i)=speye(Ni);
% ix=find(sum(H,1)==0);
test=test(i,:);
N=size(test,1);

    mean=sum(test)/N;
    var=(test'-mean'*ones(1,N))*(test'-mean'*ones(1,N))'/N;

    test1=test;
        
    for i=1:28
        
    ix=find((test1(:,i)<ones(N,1)*(mean(i)+2*sqrt(var(i,i)))).*(test1(:,i)>ones(N,1)*(mean(i)-2*sqrt(var(i,i)))));
    test1=test1(ix,:);N=size(test1,1);
    
    end



%load X.out;
X=test1(:,1:22);

y=(test1(:,23:28));
inputtrain=X(1:end,:);
outputtrain=y(1:end,:);
inputtest=X(9000+1:end,:);
outputtest=y(9000+1:end,:);
p=22;
%% Sparse approximation
%meanfunc = @meanLinear ;
%meanfunc = {@meanPoly ,2}; %degree polynomial 2
%hypp = [1;1;2;3];
meanfunc=[];% empty: don't use a mean function
%hyp2.mean = [0.5; 1;5;0.5;0.5;0.5;0.5;0.5;0.5;0.5,;0.5];
 fff=zeros(44,1);
 fff(1:end)=normrnd(0,1,44,1);
%fff=fff';
%  hyp.mean=fff;
% hyp.mean=fff;
n = 30; sn = 0.99;

 lik = {@likGauss};    hyp.lik = log(sn); inf = @infGaussLik;
 cov = {@covSEiso}; 
 %cov={@covMaternard ,3}; 
 hyp.cov = log([9.5;12.5]); % Matern class d=5
 %hyp.cov = [1.5; 2.5]; 
 %hyp.lik = log(0.5);
%disp('likfunc = @likGauss; sn = 0.1; hyp.lik = log(sn);')
%likfunc = @likGauss; sn = 0.1; hyp.lik = log(sn);
p=size(inputtest,2);
for j=1:p
xu = normrnd(0,1,100,1); 
xsparse(:,j)=xu;
end
hyp.xu=xsparse;
cov = {'apxSparse',cov,xsparse};           % inducing points
%parpool
infv  = @(varargin) inf(varargin{:},struct('s',1.0));
infr=@infFITC;
infe = @infFITC_EP;
hyp = minimize(hyp,@gp,-1350,infv,meanfunc,cov,lik,inputtrain,outputtrain);%minimise the hyperparamters
%[hyp,nlZ] = vfe_xu_opt(hyp,meanfunc,cov,inputtrain,outputtrain,xsparse,2000); %optimise the inducing points
%cov = {'apxSparse',cov,hyp.xu}; 
[ymuv,ys2v] = gp(hyp,infv,meanfunc,cov,lik, inputtrain, outputtrain, inuttrain);%Inference with GP
%[ymuv2,ys2v2] = gp(hyp,infe,meanfunc,cov,lik, inputtrain, outputtrain, inputtest);%Infeence with FITC
ymuv=(ymuv);
%ymuv2=sinh(ymuv2);

%  for j=1:6
%      outputin=outputtest(:,j);
%      
%  Lerror=(norm(outputin-ymuv(:,j))/norm(outputin))^0.5;
% L_2sparse=1-(Lerror^2);
% 
% L_2all(:,j)=L_2sparse;
% %Coefficient of determination
% for i=1:numel(outputin)
%     outputreq(i)=outputin(i)-mean(outputin);
% end
% 
% outputreq=outputreq';
% CoDsparse=1-(norm(outputin-ymuv(:,j))/norm(outputreq));
% CoDsparse=1 - (1-CoDsparse)^2 ;
% CoDall(:,j)=CoDsparse;
%  end
%   
%  CoDallv=sum(CoDall)/6;
%  L_2allv=sum(L_2all)/6;


figure()
for i=1:6
    subplot(2,3,i)
plot(outputtest(:,i),ymuv(:,i),'o');
xlabel('Real output','FontName','Helvetica', 'Fontsize', 13);
ylabel('GP estimate','FontName','Helvetica', 'Fontsize', 13)
title(['TGLF data ',sprintf('%d',i)])
%title('TGLF data','FontName','Helvetica', 'Fontsize', 13);
set(gca, 'FontName','Helvetica', 'Fontsize', 13)
set(gcf,'color','white')
end

figure
for i=1:6
    subplot(2,3,i)
plot(outputtest(:,i),'red')
hold on
plot(ymuv(:,i),'blue')
title(['TGLF data ',sprintf('%d',i)])
%legend('real data','Predicted data','FontName','Helvetica', 'Fontsize', 9)
set(gcf,'color','white')
legend({'real data','Predicted data'},'FontSize',9)

end

%poolobj = gcp('nocreate');
%delete(poolobj);
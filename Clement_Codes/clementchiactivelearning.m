clc;
clear;
close all;
%% GP classification/Regression model for the Chi data
disp(' We will implement the active learning sparse GP for the Chi model' );
disp( 'Author: Dr Clement Etienam' )
disp( 'Supervisor: Professor Kody Law' )
disp( 'Supervised active learning Modelling' )
set(0,'defaultaxesfontsize',20); format long

dd=importdata('chi_itg.dat',' ',1);
test=dd.data;
out=test;
outgp=test;
test=out;
%X=zscore(test(1:600000,1:10));
X=log(test(1:600000,1:10));
y=(test(1:600000,11));
outputtest=y(290000+1:end,:);
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
inputtrainclass=X(1:290000,:);

outputtrainclass=y(1:290000,:);
inputtest=X(290000+1:end,:);
%outputtest=y(290000+1:end,:);
p=10;
Matrixdata=[inputtrainclass outputtrainclass];

GPmatrix=[inputtrainclass (test(1:290000,11))];
outgp=GPmatrix;
outgp(any(outgp==0,2),:) = [];

outputtrainGP=log(outgp(:,11));
inputtrainGP=(outgp(:,1:10));


%%
outinitial=outputtrainGP(1:1000,:);
ininitial=inputtrainGP(1:1000,:);

inpool=inputtrainGP(1001:end,:);
outpool=outputtrainGP(1001:end,:);
%% Train the initial GP Sparse approximation
meanfunc=[];% empty: don't use a mean function
n = 30; sn = 0.99;
 lik = {@likGauss};    hyp.lik = log(sn); inf = @infGaussLik;
 cov = {@covSEiso}; 
 hyp.cov = log([9.5;12.5]); % Matern class d=5
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

disp('First train the GP with these initial points')
hyp = minimize(hyp,@gp,-100,infv,meanfunc,cov,lik,ininitial,outinitial);%minimise the hyperparamters
hypinital=hyp;
%% 
%[~,ys2v] = gp(hyp,infv,meanfunc,cov,lik, inputtrainGP, outputtrainGP, inputtest(index2,:));%Inference with GP
n_queries = 50000;
for idx=1:n_queries
 disp('predict from the pool or oracle')
[~,ys2v] = gp(hyp,infv,meanfunc,cov,lik, ininitial, outinitial, inpool);%Inference with GP
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
hyp = minimize(hyp,@gp,-100,infv,meanfunc,cov,lik,ininitial,outinitial);%minimise the hyperparamters
fprintf('Finished query point %d out of %d,%d left to go .\n', idx,n_queries,(n_queries-idx));
end


%% Prediction

[regressoutput2,ys2v] = gp(hyp,infv,meanfunc,cov,lik, ininitial, outinitial, inputtrainGP);%Inference with GP
regressoutput2=exp(regressoutput2);
clement1=regressoutput2;

%%
% rossmary=outputtest;
% %Coefficient of determination
% for i=1:numel(rossmary)
%     outputreq(i)=rossmary(i)-mean(rossmary);
% end
% 
% outputreq=outputreq';
% CoDpoly=1-(norm(rossmary-clement1)/norm(outputreq));
% CoDpoly2=1-(norm(rossmary-clement2)/norm(outputreq));
% CoDpoly3=1-(norm(rossmary-clement3)/norm(outputreq));
% CoDlinear=1 - (1-CoDpoly)^2 ;
% CoDDscr=1 - (1-CoDpoly2)^2 ;
% CoDtree=1 - (1-CoDpoly3)^2 ;
% 
% 
% 
%  figure()
%  subplot(3,2,1)
% plot(outputtest(1:500),clement1(1:500),'o');
% xlabel('Real output','FontName','Helvetica', 'Fontsize', 13);
% ylabel('Machine estimate','FontName','Helvetica', 'Fontsize', 13)
% title('chi data-Sparse GP-Logisitc ')
% %title('TGLF data','FontName','Helvetica', 'Fontsize', 13);
% set(gca, 'FontName','Helvetica', 'Fontsize', 13)
% set(gcf,'color','white')
% 
% subplot(3,2,2)
% plot(outputtest(1:500),'red')
% hold on
% plot(clement1(1:500),'blue')
% title('Logistic method ')
% %legend('real data','Predicted data','FontName','Helvetica', 'Fontsize', 9)
% set(gcf,'color','white')
% legend({'real data','Predicted data'},'FontSize',9)
% 
%  subplot(3,2,3)
% plot(outputtest(1:500),clement2(1:500),'o');
% xlabel('Real output','FontName','Helvetica', 'Fontsize', 13);
% ylabel('Machine estimate','FontName','Helvetica', 'Fontsize', 13)
% title('chi data-Sparse GP-Naieve-Bayes ')
% %title('TGLF data','FontName','Helvetica', 'Fontsize', 13);
% set(gca, 'FontName','Helvetica', 'Fontsize', 13)
% set(gcf,'color','white')
% 
% subplot(3,2,4)
% plot(outputtest(1:500),'red')
% hold on
% plot(clement2(1:500),'blue')
% title('Naieve-Bayes method ')
% %legend('real data','Predicted data','FontName','Helvetica', 'Fontsize', 9)
% set(gcf,'color','white')
% legend({'real data','Predicted data'},'FontSize',9)
% 
%  subplot(3,2,5)
% plot(outputtest(1:500),clement3(1:500),'o');
% xlabel('Real output','FontName','Helvetica', 'Fontsize', 13);
% ylabel('Machine estimate','FontName','Helvetica', 'Fontsize', 13)
% title('chi data-Sparse GP-TreeBagging ')
% %title('TGLF data','FontName','Helvetica', 'Fontsize', 13);
% set(gca, 'FontName','Helvetica', 'Fontsize', 13)
% set(gcf,'color','white')
% 
% subplot(3,2,6)
% plot(outputtest(1:500),'red')
% hold on
% plot(clement3(1:500),'blue')
% title('TreeBagging method ')
% %legend('real data','Predicted data','FontName','Helvetica', 'Fontsize', 9)
% set(gcf,'color','white')
% legend({'real data','Predicted data'},'FontSize',9)
% 
% 

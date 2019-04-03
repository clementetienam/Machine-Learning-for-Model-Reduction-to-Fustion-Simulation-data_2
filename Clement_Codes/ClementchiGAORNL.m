clc;
clear;
close all;
%% GP classification/Regression model for the Chi data
disp(' We will implement the sparse GP together with classification for the Chi model' );
disp( 'Author: Dr Clement Etienam' )
disp( 'Supervisor: Professor Kody Law' )
disp( 'Supervised classfication Modelling' )
set(0,'defaultaxesfontsize',20); format long


dd=importdata('chi_itg.dat',' ',1);
test=dd.data;
out=test;
outgp=test;
test=out;
%X=zscore(test(1:600000,1:10));
X=log(test(1:600000,1:10));
y=(test(1:600000,11));
y3=y;
outputtest=y(290000+1:end,:);
y2=zeros(600000,1);
for i=1:600000
    if y(i)==0
        y2(i)=abs(0+normrnd(0,0.02));
        y2(i)=-(y2(i));
    end
    
    if y(i)>0
        y2(i)=log(y3(i));
    end
        
end
y=y2;
inputtrainclass=X(1:290000,:);

outputtrainclass=y(1:290000,:);
inputtest=X(290000+1:end,:);
% %outputtest=y(290000+1:end,:);
% p=10;
% Matrixdata=[inputtrainclass outputtrainclass];
% 
% GPmatrix=[inputtrainclass (test(1:290000,11))];
% outgp=GPmatrix;
% outgp(any(outgp==0,2),:) = [];
% 
% outputtrainGP=log(outgp(:,11));
% inputtrainGP=(outgp(:,1:10));

%% Train the GP Sparse approximation
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
hyp = minimize(hyp,@gp,-100,infv,meanfunc,cov,lik,inputtrainclass,outputtrainclass);%minimise the hyperparamters
ff=size(inputtest,1);
ff=500;

%%
% disp('Predict at once')
% labelDA = predict(Mdl,inputtest);
% index1=find(labelDA==-1); %output that gave a zero
% index2=find(labelDA==1); % output that didnt give a zero

% clement1=zeros(size(inputtest,1),1);
% clement1(index1,:)=0; %values that the classifier predicts to give a 0
[regressoutput2,ys2v] = gp(hyp,infv,meanfunc,cov,lik, inputtrainclass, outputtrainclass, inputtest);%Inference with GP
regressoutput2(regressoutput2<=0)=0;
index1=find(regressoutput2==0); %output that gave a zero
index2=find(regressoutput2>0); % output that didnt give a zero

clement1=zeros(size(inputtest,1),1);
regressoutput=exp(regressoutput2(index2,:));
clement1(index2,:)=regressoutput;


%%
% disp('Predict at once')
% labelDA2 = predict(Mdl2,inputtest);
% index1=find(labelDA2==-1); %output that gave a zero
% index2=find(labelDA2==1); % output that didnt give a zero
% 
% clement2=zeros(size(inputtest,1),1);
% clement2(index1,:)=0; %values that the classifier predicts to give a 0
% [regressoutput2,ys2v] = gp(hyp,infv,meanfunc,cov,lik, inputtrainGP, outputtrainGP, inputtest(index2,:));%Inference with GP
% regressoutput2=exp(regressoutput2);
% clement2(index2,:)=regressoutput2;
% %%
% disp('Predict at once')
% labelDA3 = predict(Mdl3,inputtest);
% unie=zeros(size(labelDA2));
% 
% for i=1:size(labelDA2,1)
% unie(i,:)=str2double(labelDA3{i,1});
% end
% labelDA3 = unie;
% index1=find(labelDA3==-1); %output that gave a zero
% index2=find(labelDA3==1); % output that didnt give a zero
% 
% clement3=zeros(size(inputtest,1),1);
% clement3(index1,:)=0; %values that the classifier predicts to give a 0
% [regressoutput2,ys2v] = gp(hyp,infv,meanfunc,cov,lik, inputtrainGP, outputtrainGP, inputtest(index2,:));%Inference with GP
% regressoutput2=exp(regressoutput2);
% clement3(index2,:)=regressoutput2;
% %%
rossmary=outputtest;
%Coefficient of determination
for i=1:numel(rossmary)
    outputreq(i)=rossmary(i)-mean(rossmary);
end

outputreq=outputreq';
CoDpoly=1-(norm(rossmary-clement1)/norm(outputreq));
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
% % subplot(2,2,2)
% % plot(outputtest(1:500),'red')
% % hold on
% % plot(clement(1:500),'blue')
% % title('CHI data ')
% % %legend('real data','Predicted data','FontName','Helvetica', 'Fontsize', 9)
% % set(gcf,'color','white')
% % legend({'real data','Predicted data'},'FontSize',9)
% % 
% % subplot(2,2,3)
% % hist(outputtest(1:500)-clement)
% % title('CHI data ')
% % %legend('real data','Predicted data','FontName','Helvetica', 'Fontsize', 9)
% % set(gcf,'color','white')
% 
% 

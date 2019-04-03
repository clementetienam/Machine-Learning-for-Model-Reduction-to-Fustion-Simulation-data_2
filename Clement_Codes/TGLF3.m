clc;
clear;
close all;
%% GP regression model for the Dense and Sparse data
disp(' We will implement the sparse GP for the TGLF model' );
disp( 'Author: Clement Etienam' )
disp( 'Supervisor: Professor Kody Law' )
disp( 'GP Regression Modelling' )
set(0,'defaultaxesfontsize',20); format long
disp(' For GP process it is good to rescale the data and use Gaussian distribution of data')
load Xresclae.out; %rescaled data from python
%load X.out;
X=Xresclae;
load y.out;
inputtest=X(3e5+1:end,:);
outputtest=y(3e5+1:end,:);
p=22;
load features1.out;
load target1.out;

load features2.out;
load target2.out;

load features3.out;
load target3.out;

load features4.out;
load target4.out;

load features5.out;
load target5.out;

load features6.out;
load target6.out;
%% Sparse approximation
meanfunc = @meanLinear ;
%meanfunc=[];% empty: don't use a mean function
%hyp2.mean = [0.5; 1;5;0.5;0.5;0.5;0.5;0.5;0.5;0.5,;0.5];
 fff=zeros(p,1);
 fff(1:end)=0.5;
%fff=fff';
 hyp.mean=fff;
hyp.mean=fff;
n = 30; sn = 0.9;

 lik = {@likGauss};    hyp.lik = log(sn); inf = @infGaussLik;
 cov = {@covSEiso}; 
 hyp.cov = [0; 0]; 
 %hyp.lik = log(0.5);
%disp('likfunc = @likGauss; sn = 0.1; hyp.lik = log(sn);')
%likfunc = @likGauss; sn = 0.1; hyp.lik = log(sn);
p=size(inputtest,2);
for j=1:p
xu = normrnd(0,1,100,1); 
xsparse(:,j)=xu;
end
cov = {'apxSparse',cov,xsparse};           % inducing points
%parpool
infv  = @(varargin) inf(varargin{:},struct('s',1.0)); 
infe = @infFITC_EP;
hyp = minimize(hyp,@gp,-100,infv,meanfunc,cov,lik,features1,target1);
[ymuv1,ys2v] = gp(hyp,infv,meanfunc,cov,lik, features1, target1, inputtest);


 outputin=outputtest(:,1);
     
 Lerror=(norm(outputin-ymuv1)/norm(outputin))^0.5;
L_2sparse1=1-(Lerror^2);

%Coefficient of determination
for i=1:numel(outputin)
    outputreq(i)=outputin(i)-mean(outputin);
end

outputreq=outputreq';
CoDsparse=1-(norm(outputin-ymuv1)/norm(outputreq));
CoDsparse1=1 - (1-CoDsparse)^2 ;

figure()
plot(outputtest(:,1),ymuv1,'o');
xlabel('Real output','FontName','Helvetica', 'Fontsize', 13);
ylabel('GP estimate','FontName','Helvetica', 'Fontsize', 13)
title('TGLF data 1', 'FontName','Helvetica', 'Fontsize', 13')
%title('TGLF data','FontName','Helvetica', 'Fontsize', 13);
set(gca, 'FontName','Helvetica', 'Fontsize', 13)
set(gcf,'color','white')


hyp = minimize(hyp,@gp,-100,infv,meanfunc,cov,lik,features2,target2);
[ymuv2,ys2v] = gp(hyp,infv,meanfunc,cov,lik, features2, target2, inputtest);


 outputin=outputtest(:,2);
     
 Lerror=(norm(outputin-ymuv2)/norm(outputin))^0.5;
L_2sparse2=1-(Lerror^2);

%Coefficient of determination
for i=1:numel(outputin)
    outputreq(i)=outputin(i)-mean(outputin);
end

outputreq=outputreq';
CoDsparse=1-(norm(outputin-ymuv1)/norm(outputreq));
CoDsparse2=1 - (1-CoDsparse)^2 ;

figure()
plot(outputtest(:,2),ymuv2,'o');
xlabel('Real output','FontName','Helvetica', 'Fontsize', 13);
ylabel('GP estimate','FontName','Helvetica', 'Fontsize', 13)
title('TGLF data 2', 'FontName','Helvetica', 'Fontsize', 13')
%title('TGLF data','FontName','Helvetica', 'Fontsize', 13);
set(gca, 'FontName','Helvetica', 'Fontsize', 13)
set(gcf,'color','white')


hyp = minimize(hyp,@gp,-100,infv,meanfunc,cov,lik,features3,target3);
[ymuv3,ys2v] = gp(hyp,infv,meanfunc,cov,lik, features3, target3, inputtest);


 outputin=outputtest(:,3);
     
 Lerror=(norm(outputin-ymuv3)/norm(outputin))^0.5;
L_2sparse3=1-(Lerror^2);

%Coefficient of determination
for i=1:numel(outputin)
    outputreq(i)=outputin(i)-mean(outputin);
end

outputreq=outputreq';
CoDsparse=1-(norm(outputin-ymuv3)/norm(outputreq));
CoDsparse3=1 - (1-CoDsparse)^2 ;

figure()
plot(outputtest(:,3),ymuv3,'o');
xlabel('Real output','FontName','Helvetica', 'Fontsize', 13);
ylabel('GP estimate','FontName','Helvetica', 'Fontsize', 13)
title('TGLF data 3', 'FontName','Helvetica', 'Fontsize', 13')
%title('TGLF data','FontName','Helvetica', 'Fontsize', 13);
set(gca, 'FontName','Helvetica', 'Fontsize', 13)
set(gcf,'color','white')



hyp = minimize(hyp,@gp,-100,infv,meanfunc,cov,lik,features4,target4);
[ymuv4,ys2v] = gp(hyp,infv,meanfunc,cov,lik, features4, target4, inputtest);


 outputin=outputtest(:,4);
     
 Lerror=(norm(outputin-ymuv4)/norm(outputin))^0.5;
L_2sparse4=1-(Lerror^2);

%Coefficient of determination
for i=1:numel(outputin)
    outputreq(i)=outputin(i)-mean(outputin);
end

outputreq=outputreq';
CoDsparse=1-(norm(outputin-ymuv4)/norm(outputreq));
CoDsparse4=1 - (1-CoDsparse)^2 ;

figure()
plot(outputtest(:,4),ymuv4,'o');
xlabel('Real output','FontName','Helvetica', 'Fontsize', 13);
ylabel('GP estimate','FontName','Helvetica', 'Fontsize', 13)
title('TGLF data 4', 'FontName','Helvetica', 'Fontsize', 13')
%title('TGLF data','FontName','Helvetica', 'Fontsize', 13);
set(gca, 'FontName','Helvetica', 'Fontsize', 13)
set(gcf,'color','white')


hyp = minimize(hyp,@gp,-100,infv,meanfunc,cov,lik,features5,target5);
[ymuv5,ys2v] = gp(hyp,infv,meanfunc,cov,lik, features5, target5, inputtest);


 outputin=outputtest(:,5);
     
 Lerror=(norm(outputin-ymuv1)/norm(outputin))^0.5;
L_2sparse5=1-(Lerror^2);

%Coefficient of determination
for i=1:numel(outputin)
    outputreq(i)=outputin(i)-mean(outputin);
end

outputreq=outputreq';
CoDsparse=1-(norm(outputin-ymuv1)/norm(outputreq));
CoDsparse5=1 - (1-CoDsparse)^2 ;

figure()
plot(outputtest(:,5),ymuv1,'o');
xlabel('Real output','FontName','Helvetica', 'Fontsize', 13);
ylabel('GP estimate','FontName','Helvetica', 'Fontsize', 13)
title('TGLF data 5', 'FontName','Helvetica', 'Fontsize', 13')
%title('TGLF data','FontName','Helvetica', 'Fontsize', 13);
set(gca, 'FontName','Helvetica', 'Fontsize', 13)
set(gcf,'color','white')

hyp = minimize(hyp,@gp,-100,infv,meanfunc,cov,lik,features6,target6);
[ymuv6,ys2v] = gp(hyp,infv,meanfunc,cov,lik, features6, target6, inputtest);


 outputin=outputtest(:,6);
     
 Lerror=(norm(outputin-ymuv6)/norm(outputin))^0.5;
L_2sparse6=1-(Lerror^2);

%Coefficient of determination
for i=1:numel(outputin)
    outputreq(i)=outputin(i)-mean(outputin);
end

outputreq=outputreq';
CoDsparse=1-(norm(outputin-ymuv1)/norm(outputreq));
CoDsparse6=1 - (1-CoDsparse)^2 ;

figure()
plot(outputtest(:,6),ymuv6,'o');
xlabel('Real output','FontName','Helvetica', 'Fontsize', 13);
ylabel('GP estimate','FontName','Helvetica', 'Fontsize', 13)
title('TGLF data 6', 'FontName','Helvetica', 'Fontsize', 13')
%title('TGLF data','FontName','Helvetica', 'Fontsize', 13);
set(gca, 'FontName','Helvetica', 'Fontsize', 13)
set(gcf,'color','white')


%poolobj = gcp('nocreate');
%delete(poolobj);
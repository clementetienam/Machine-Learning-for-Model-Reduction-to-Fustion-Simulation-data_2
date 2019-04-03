clc;
clear;
close all;
%% GP regression model for the Dense and Sparse data
disp(' We will implement the sparse GP for the TGLF model' );
disp( 'Author: Clement Etienam' )
disp( 'Supervisor: Professor Kody Law' )
disp( 'GP Regression Modelling' )
set(0,'defaultaxesfontsize',20); format long
load tglf_plus

input=test(:,1:22);
input2=system('python rescaledd.py input');
%input=python('rescaledd.py','input');
% rescale the input;
input = rescale(input);
output=test(:,23:28);

inputin=input(1:3e5,:);

outputin=output(1:3e5,:);
outputtest=output(3e5+1:end,:);
inputtest=input(3e5+1:end,:);

%% Sparse approximation
%meanfunc = @meanLinear ;
meanfunc=[];% empty: don't use a mean function
%hyp2.mean = [0.5; 1;5;0.5;0.5;0.5;0.5;0.5;0.5;0.5,;0.5];
% fff=zeros(p,1);
% fff(1:end)=0.5;
% %fff=fff';
% hyp.mean=fff;
% hyp.mean=fff;
n = 30; sn = 0.5;

 lik = {@likGauss};    hyp.lik = log(sn); inf = @infGaussLik;
 cov = {@covSEiso}; 
 hyp.cov = [0; 0]; 
 hyp.lik = log(0.5);
%disp('likfunc = @likGauss; sn = 0.1; hyp.lik = log(sn);')
%likfunc = @likGauss; sn = 0.1; hyp.lik = log(sn);
for j=1:22
xu = normrnd(0,1,100,1); 
xsparse(:,j)=xu;
end
cov = {'apxSparse',cov,xsparse};           % inducing points
%parpool
infv  = @(varargin) inf(varargin{:},struct('s',1.0)); 
infe = @infFITC_EP;
hyp = minimize(hyp,@gp,-100,infv,meanfunc,cov,lik,inputin,outputin);
[ymuv,ys2v] = gp(hyp,infv,meanfunc,cov,lik, inputin, outputin, inputtest);


 Lerror=(norm(outputin-ymuv)/norm(outputin))^0.5;
L_2sparse=1-(Lerror^2);
%Coefficient of determination
for i=1:numel(outputin)
    outputreq(i)=outputin(i)-mean(outputin);
end

outputreq=outputreq';
CoDsparse=1-(norm(outputin-ymuv)/norm(outputreq));
CoDsparse=1 - (1-CoDsparse)^2 ;

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
clc;
clear;
close all;
M = csvread('TGLFdata.csv',1,0);
ix = ismissing(M);
completeData = M(~any(ix,2),:);
input=completeData(:,1:22);
output=completeData(:,23:28);
inputtrain=input(1:3e5,:);
inputtrain = (rescale(inputtrain',0,1))';
inputtest=input(3e5+1:end,:);

outputtrain=output(1:3e5,:);
outputtrain = (rescale(outputtrain',0,1))';
outputtest=output(3e5+1:end,:);
for j=1:22
x=inputtrain(:,j);
X=dct(x);
[XX,ind] = sort(abs(X),'descend');


Bsig = cumsum(XX);
valuesig=Bsig(end);
valuesig=valuesig*0.99;
indices = find(cumsum(XX) >= valuesig );
indu=indices(1,:);
induall(:,j)=indu;

induuse=max(induall);
indibig(:,j)=ind;

valuee=X(ind(1:induuse,:));

xxout(:,j)=valuee;
end


for j=1:22
    xrecon=zeros(300000,22);
    xreconn=xrecon(:,j);
    xworkon=xxout(:,j);
    ind=indibig(:,j);
 xworkon(ind(induuse+1:end)) = 0;   

  xx = idct(xworkon);
xxall(:,j)=xx;
end
clc;
clear;
close all;
dd=load ('tglf_plus.mat');
test=dd.test;
overalldata=test;
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
    
    file4 = fopen('TGLGmum.out','w+'); 
 for k=1:numel(test1)                                                                       
 fprintf(file4,' %4.4f \n',test1(k) );             
 end
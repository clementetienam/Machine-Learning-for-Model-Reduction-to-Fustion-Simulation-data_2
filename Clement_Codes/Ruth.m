clc;
clear;
close all;
noEM2= readtable('TGLF.csv');
test2 = noEM2{:,1:28};
test2output=test2;
test2input=test2;
output=[test2output(:,3),test2output(:,8),test2output(:,17),test2output(:,18),test2output(:,23),test2output(:,25)];
test2input(:,[3,8,17,18,23,25])=[];
seconddata=zeros(398892,28);
seconddata(:,1:22)=test2input;
seconddata(:,23:28)=output;
overalldata=seconddata;
overalldata(any(isnan(overalldata), 2), :) = [];
%overalldata(any((overalldata==0), 2), :) = [];
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
file2 = fopen('orso.out','w+'); 
 for k=1:numel(test)                                                                       
 fprintf(file2,' %4.4f \n',test(k) );             
 end
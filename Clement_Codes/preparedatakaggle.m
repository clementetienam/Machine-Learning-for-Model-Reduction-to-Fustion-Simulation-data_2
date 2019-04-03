clc;
clear;
close all;
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
% % F = fillmissing(overalldata,'previous', 'EndValues', 'nearest');
% % F = typecast(F, 'double');

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





% R=overalldata;
% idx = bsxfun(@gt, R, mean(R) + std(R)) | bsxfun(@lt, R, mean(R) - std(R));
% idx = any(idx, 2);
% R(idx, :) = [];
% 
% figure()
% for i=1:28
% subplot(6,6,i)
% plot(R(:,i))
% end
% % 
fid=fopen('Finaldata.out','w');
 b=fprintf(fid,'%.8f\n',test1);
 fclose(fid);

% dlmwrite('Features.txt',input,'delimiter',' ','precision','%.6f');
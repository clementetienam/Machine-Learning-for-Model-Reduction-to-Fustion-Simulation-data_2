clc;
clear;
close all;
 True= importdata('oil_price.xlsx',' ',1);
 True=True.data;
 
 for i=1:1000
     x(i,:)=log(True(i:i+5,:));
     y(i,:)=True(i+6,:);
 end
 inputtrain=x(1:500,:);
 outputtrain=y(1:500,:);
 inputtest=x(501:1000,:);
 outputtest=y(501:1000,:);
 %
  file1 = fopen('inputtianke.out','w+'); 
 for k=1:numel(inputtrain)                                                                       
 fprintf(file1,' %4.4f \n',inputtrain(k) );             
 end
 
   file2 = fopen('outputtianke.out','w+'); 
 for k=1:numel(outputtrain)                                                                       
 fprintf(file2,' %4.4f \n',outputtrain(k) );             
 end
 
   file3 = fopen('inputtianketest.out','w+'); 
 for k=1:numel(inputtest)                                                                       
 fprintf(file3,' %4.4f \n',inputtest(k) );             
 end
 
   file4 = fopen('outputtianketest.out','w+'); 
 for k=1:numel(outputtest)                                                                       
 fprintf(file4,' %4.4f \n',outputtest(k) );             
 end
 %% Sparse GP approximation


% Lerror=(norm(outputtest-ymuv)/norm(outputtest))^0.5;
% L_2sparse=1-(Lerror^2);
% %Coefficient of determination
% for j=1:2
% for i=1:numel(outputtest)
%     outputreq(i)=outputtest(i)-mean(outputtest);
% end
% end
% 
% outputreq=outputreq';
% CoDsparse=1-(norm(outputtest-ymuv)/norm(outputreq));
% CoDsparse=1 - (1-CoDsparse)^2 ;
%%
%% Polynomial Regression

clc;
clear;
close all;

[y,x]=this;

[y2,x2]=this;

figure()
plot(x,y,'x')

disp('  output to ASCII files  ');
file = fopen('inputtestactive.out','w+'); 
 for k=1:numel(x2)                                                                       
 fprintf(file,' %4.4f \n',x2(k) );             
 end


 file = fopen('outputtestactive.out','w+'); 
 for k=1:numel(y2)                                                                       
 fprintf(file,' %4.4f \n',y2(k) );             
 end

 
 
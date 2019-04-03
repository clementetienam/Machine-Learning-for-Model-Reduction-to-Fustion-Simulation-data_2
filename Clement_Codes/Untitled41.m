clc;
clear;
close all;
disp('Lets get the 2D piecewsie function')
x1 = (linspace(-5,5,10000))';
y1=piecewise(x1);
%y2 = piecewise_eval(x1,[-5 0 2 3],{2,'sin(x)','x.^2',6,-1});
%y1=y2;
figure()
subplot(2,2,1)
plot(x1,y1)
grid off
title('1D piecewise functions','FontName','Helvetica', 'Fontsize', 13);
ylabel('f(x1)', 'FontName','Helvetica', 'Fontsize', 13);
xlabel('x1', 'FontName','Helvetica', 'Fontsize', 13);
set(gca, 'FontName','Helvetica', 'Fontsize', 13)
set(gcf,'color','white')
%  For       x < -5, y = 2
%  For -5 <= x < 0,  y = sin(x)
%  For  0 <= x < 2,  y = x.^2
%  For  2 <= x < 3,  y = 6
%  For  3 <= x,      y = inf
%
 y2 = piecewise_eval(x1,[-5 0 2 3],{2,'sin(x)','x.^2',6,-1});
 subplot(2,2,2)
plot(x1,y2)
grid off
title('1D piecewise functions','FontName','Helvetica', 'Fontsize', 13);
ylabel('f(x2)', 'FontName','Helvetica', 'Fontsize', 13);
xlabel('x2', 'FontName','Helvetica', 'Fontsize', 13);
set(gca, 'FontName','Helvetica', 'Fontsize', 13)
set(gcf,'color','white')

Yout=y1;
Xout=x1;




Xtest = (linspace(-4,4,10000))';
Ytest=piecewise(Xtest);



% Ytest=y1(8001:end,:);
% Xtest=x1(8001:end,:);
file2 = fopen('inpiecewise.out','w+'); 
 for k=1:numel(Xout)                                                                       
 fprintf(file2,' %4.4f \n',Xout(k) );             
 end
 file3 = fopen('outpiecewise.out','w+'); 
 for k=1:numel(Yout)                                                                       
 fprintf(file3,' %4.4f \n',Yout(k) );             
 end
 file22 = fopen('intestpiecewise.out','w+'); 
 for k=1:numel(Xtest)                                                                       
 fprintf(file22,' %4.4f \n',Xtest(k) );             
 end
 file32 = fopen('outtestpiecewise.out','w+'); 
 for k=1:numel(Ytest)                                                                       
 fprintf(file32,' %4.4f \n',Ytest(k) );             
 end


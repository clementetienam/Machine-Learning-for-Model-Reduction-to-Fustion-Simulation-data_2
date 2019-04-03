clc;
clear;
close all;
disp('Lets get the 2D piecewsie function')
x1 = (linspace(-30,30,100))';
y1=piecewise(x1);
x2=(linspace(-20,20,100))';
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
 y2 = piecewise_eval(x2,[-5 0 2 3],{2,'sin(x)','x.^2',6,-1});
 subplot(2,2,2)
plot(x2,y2)
grid off
title('1D piecewise functions','FontName','Helvetica', 'Fontsize', 13);
ylabel('f(x2)', 'FontName','Helvetica', 'Fontsize', 13);
xlabel('x2', 'FontName','Helvetica', 'Fontsize', 13);
set(gca, 'FontName','Helvetica', 'Fontsize', 13)
set(gcf,'color','white')
disp('make the 2D now')
[Xarin,Zarin]=meshgrid(x1,x2);
X=[reshape(Xarin,10000,1) reshape(Zarin,10000,1)];
Y=reshape(y1*y2',10000,1);
subplot(2,2,3)
plot(Y)
grid off
title('2D slice of (y) piecewise functions','FontName','Helvetica', 'Fontsize', 13);
ylabel('f(x1,x2)', 'FontName','Helvetica', 'Fontsize', 13);
xlabel('number of points', 'FontName','Helvetica', 'Fontsize', 13);
set(gca, 'FontName','Helvetica', 'Fontsize', 13)
set(gcf,'color','white')
Yout=Y(1:8000,:);
Xout=X(1:8000,:);
Ytest=Y(8001:end,:);
Xtest=X(8001:end,:);
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

[Xx,Yy] = meshgrid(1:100,1:100);
subplot(2,2,4);
surf(Xx',Yy',reshape(Y,100,100))
shading flat
%axis([1 100 1 100 ])
grid off
title('2D Discountionus function','FontName','Helvetica', 'Fontsize', 13);
ylabel('X2', 'FontName','Helvetica', 'Fontsize', 13);
xlabel('X1', 'FontName','Helvetica', 'Fontsize', 13);
colormap('jet')
h = colorbar;
ylabel(h, 'y-Function values','FontName','Helvetica', 'Fontsize', 13);
set(gca, 'FontName','Helvetica', 'Fontsize', 13)
set(gcf,'color','white')
set(gca,'xticklabel',[])
set(gca,'yticklabel',[])
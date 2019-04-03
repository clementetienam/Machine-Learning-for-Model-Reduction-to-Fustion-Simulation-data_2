clc;
clear;
close all;
clear
clear
N=200;f=zeros(N,1);
x=linspace(-5,10,N)';
clear
N=100;f=zeros(N,1);
x=linspace(-5,10,N)';
for i=1:N
% if x(i)<-5
%     f(i)=sin(x(i));
if x(i)<5
    f(i)=exp(-x(i)^2/10);
elseif x(i)<8
    f(i)=1;
elseif x(i)<9
    f(i)=-1;
else
    f(i)=0;
end
end
   
y1=f;figure
plot(x,y1);
 
y2=f*f';
[X,Z]=meshgrid(x,x);
X2=[X(:),Z(:)];
Y2=y2(:);
 
figure
surfc(x,x,y2)  
y1=f;figure
plot(x,y1);
 
y2=f*f';
[X,Z]=meshgrid(x,x);
X2=[X(:),Z(:)];
Y2=y2(:);
 
figure
surfc(x,x,y2)
file2 = fopen('inpiecewise.out','w+'); 
 for k=1:numel(X2)                                                                       
 fprintf(file2,' %4.4f \n',X2(k) );             
 end
 file3 = fopen('outpiecewise.out','w+'); 
 for k=1:numel(Y2)                                                                       
 fprintf(file3,' %4.4f \n',Y2(k) );             
 end
 
figure
surfc(x,x,y2)
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

figure()
[X,Y] = meshgrid(1:100,1:100);


figure()

surf(X',Y',y2)

shading flat
axis([1 100 1 100 ])
grid off
title('Function plot','FontName','Helvetica', 'Fontsize', 13);
ylabel('X1', 'FontName','Helvetica', 'Fontsize', 13);
xlabel('X2', 'FontName','Helvetica', 'Fontsize', 13);
colormap('jet')
%caxis([1 10])
h = colorbar;
ylabel(h, 'y-Function values','FontName','Helvetica', 'Fontsize', 13);
%set(h, 'ylim', [1 10])
set(gca, 'FontName','Helvetica', 'Fontsize', 13)
set(gcf,'color','white')
set(gca,'xticklabel',[])
set(gca,'yticklabel',[])

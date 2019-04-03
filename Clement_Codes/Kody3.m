clc;
clear all;
close all;
N=100;f=zeros(N,1);
x=linspace(-4,10,N)';
for i=1:N
% if x(i)<-5
%     f(i)=sin(x(i));
if x(i)<4
    f(i)=exp(-x(i)^2/20);
elseif x(i)<6
    f(i)=1;
elseif x(i)<8
    f(i)=-1;
else
    f(i)=0;
end
end
   
facx=max(x)-min(x);
x=(x-min(x))/facx;

y1=f;
facy=max(y1)-min(y1);
y1=20*(y1-min(y1))/facy;
figure
plot(x,y1);

y2=f*f';
facy2=max(max(y2))-min(min(y2));
y2=20*(y2-min(min(y2)))/facy2;
[X,Z]=meshgrid(x,x);
X2=[X(:),Z(:)];
Y2=y2(:);

[Xx,Yy] = meshgrid(1:100,1:100);


figure()
%subplot(2,2,1)
surf(Xx',Yy',reshape(Y2,100,100))

shading flat
axis([1 100 1 100 ])
grid off
title('True Layer 1','FontName','Helvetica', 'Fontsize', 13);
ylabel('Y', 'FontName','Helvetica', 'Fontsize', 13);
xlabel('X', 'FontName','Helvetica', 'Fontsize', 13);
colormap('jet')
%caxis([1 24])
h = colorbar;
ylabel(h, 'y-function values','FontName','Helvetica', 'Fontsize', 13);
%set(h, 'ylim', [1 24])
set(gca, 'FontName','Helvetica', 'Fontsize', 13)
set(gcf,'color','white')
set(gca,'xticklabel',[])
set(gca,'yticklabel',[])
% figure
% surfc(x,x,y2)
% 
% scatter3(X(:),Z(:),Y2,'o')

XX=[X2,Y2];
IDX=kmeans(XX,24);
figure()
surf(Xx',Yy',reshape(IDX,100,100))

shading flat
axis([1 100 1 100 ])
grid off
title('True','FontName','Helvetica', 'Fontsize', 13);
ylabel('Y', 'FontName','Helvetica', 'Fontsize', 13);
xlabel('X', 'FontName','Helvetica', 'Fontsize', 13);
colormap('jet')
%caxis([1 24])
h = colorbar;
ylabel(h, 'Cluster labels','FontName','Helvetica', 'Fontsize', 13);
%set(h, 'ylim', [1 24])
set(gca, 'FontName','Helvetica', 'Fontsize', 13)
set(gcf,'color','white')
set(gca,'xticklabel',[])
set(gca,'yticklabel',[])


file2 = fopen('inpiecewise2.out','w+'); 
 for k=1:numel(X2)                                                                       
 fprintf(file2,' %4.4f \n',X2(k) );             
 end
 file3 = fopen('outpiecewise2.out','w+'); 
 for k=1:numel(Y2)                                                                       
 fprintf(file3,' %4.4f \n',Y2(k) );             
 end














% imagesc(x,x,reshape(IDX,N,N))
% figure(3)
% imagesc(x,x,y2)
% 
% IDX=kmeans([x,y1],4);
% figure(1)
% plot(x,y1,'o');hold
% plot(x,IDX);hold
% %ixx=find(IDX==2);
% %plot(x(ixx),y1(ixx),'o')
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
 
matrix=[X2 Y2];


[cluster_idx, cluster_center] = kmeans(matrix,24,'distance','sqEuclidean', ...
                                      'Replicates',3);
pixel_labels = reshape(cluster_idx,100,100);

   
    [X,Y] = meshgrid(1:100,1:100);


figure()
%subplot(2,2,1)
surf(X',Y',pixel_labels)

shading flat
axis([1 100 1 100 ])
grid off
title('True Layer 1','FontName','Helvetica', 'Fontsize', 13);
ylabel('Y', 'FontName','Helvetica', 'Fontsize', 13);
xlabel('X', 'FontName','Helvetica', 'Fontsize', 13);
colormap('jet')
caxis([1 24])
h = colorbar;
ylabel(h, 'Cluster values','FontName','Helvetica', 'Fontsize', 13);
set(h, 'ylim', [1 24])
set(gca, 'FontName','Helvetica', 'Fontsize', 13)
set(gcf,'color','white')
set(gca,'xticklabel',[])
set(gca,'yticklabel',[])

%subplot(2,2,2)
N=100;
x=linspace(-5,10,N)';
figure()
surfc(x,x,pixel_labels)
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


E = evalclusters(matrix,'kmeans','silhouette','klist',[1:24]);

figure;
plot(E)

figure;
gscatter(matrix(:,1),matrix(:,2),E.OptimalY,'rbgk','xod')
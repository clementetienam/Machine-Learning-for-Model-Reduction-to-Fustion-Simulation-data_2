clc;
clear;
close all;
M = csvread('TGLFdata.csv',1,0);
ix = ismissing(M);
completeData = M(~any(ix,2),:);
input=completeData(:,1:22);
output=completeData(:,23:28);
inputtrain=input(1:3e5,:);
inputtest=input(3e5+1:end,:);

outputtrain=output(1:3e5,:);
outputtest=output(3e5+1:end,:);
p=size(input,2);

n=800;
y=200;
% re
sgsim=(inputtrain);
 for j=1:22
         value=sgsim(:,j);
         value=reshape(value,1000,300);
         usdf=mirt_dctn(value);
         usdfall(:,j)=reshape(usdf,300000,1);
 end
      
             [X,Y] = meshgrid(1:300000,1:22);


disp('  extract the significant DCT coefficients  ');
 for j=1:22
    val11=reshape(usdfall(:,j),1000,300); % touch here
   
for jjj=1
    val1=val11(1:n,1:y,jjj);
    val1=reshape(val1,n*y,1);
   val2(:,jjj)=val1;
end
  sdfbig2=val2;
  clement2(:,j)=sdfbig2;

 end 
 
disp( 'reconstruct ')
value1=clement2(1:n*y,1:22);


valuepermjoy=value1;
for ii=1:22
    lf=reshape(valuepermjoy(:,ii),n,y);
    
         valueperm=lf;
         big=zeros(1000,300);

        big(1:n,1:y)=valueperm;
        kkperm=big(:,:);
        rec = mirt_idctn(kkperm);
       % rec=((rec));
         usdf=reshape(rec,300000,1);
         young=usdf;
   
      sdfbig=reshape(young,300000,1);
  clementperm(:,ii)=sdfbig;
end
        
        figure()
subplot(2,2,1)
surf(X',Y',inputtrain)

shading flat
axis([1 300000 1 22 ])
grid off
title('True Layer 1','FontName','Helvetica', 'Fontsize', 13);
ylabel('Y', 'FontName','Helvetica', 'Fontsize', 13);
xlabel('X', 'FontName','Helvetica', 'Fontsize', 13);
colormap('jet')

caxis([-1e4 1e4])
h = colorbar;
ylabel(h, 'Log K(mD)','FontName','Helvetica', 'Fontsize', 13);
set(h, 'ylim', [-1e4 1e4])
set(gca, 'FontName','Helvetica', 'Fontsize', 13)
set(gcf,'color','white')
set(gca,'xticklabel',[])
set(gca,'yticklabel',[])
        
   subplot(2,2,2)
surf(X',Y',clementperm)

shading flat
axis([1 300000 1 22 ])
grid off
title('True Layer 1','FontName','Helvetica', 'Fontsize', 13);
ylabel('Y', 'FontName','Helvetica', 'Fontsize', 13);
xlabel('X', 'FontName','Helvetica', 'Fontsize', 13);
colormap('jet')

caxis([-1e4 1e4])
h = colorbar;
ylabel(h, 'Log K(mD)','FontName','Helvetica', 'Fontsize', 13);
set(h, 'ylim', [-1e4 1e4])
set(gca, 'FontName','Helvetica', 'Fontsize', 13)
set(gcf,'color','white')
set(gca,'xticklabel',[])
set(gca,'yticklabel',[])
 

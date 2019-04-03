function lizz2=marginals2(labelDA,inputtest,inputtrainGP,outputtrainGP)
%% 
disp(' Lets view the 2D marginals' );
disp( 'Author: Dr Clement Etienam' )
disp( 'Supervisor: Professor Kody Law' )

set(0,'defaultaxesfontsize',20); format long
%%

test=inputtest;
ne=test(:,1);
te=test(:,2);
ti=test(:,3);
zeff=test(:,4);
q=test(:,5);
shat=test(:,6);
rlni=test(:,7);
rlti=test(:,8);
a=test(:,9);
r0=test(:,10);
ds=size(test,1);

disp('actual forwarding from JM data')
for i=1:ds
chi1(i,:)=forwarding(ne(i,:), te(i,:), ti(i,:), zeff(i,:), q(i,:), shat(i,:), rlni(i,:), rlti(i,:), a(i,:), r0(i,:), 1, 1 );
end

%% Train the GP Sparse approximation
meanfunc=[];% empty: don't use a mean function
n = 30; sn = 0.99;
 lik = {@likGauss};    hyp.lik = log(sn); inf = @infGaussLik;
 cov = {@covSEiso}; 
 hyp.cov = log([9.5;12.5]); % Matern class d=5
 
p=size(inputtest,2);

pkg load statistics
for j=1:p
xu = normrnd(0,1,100,1); 
xsparse(:,j)=xu;
end
hyp.xu=xsparse;
cov = {'apxSparse',cov,xsparse};           % inducing points
%parpool
infv  = @(varargin) inf(varargin{:},struct('s',1.0));

hyp = minimize(hyp,@gp,-100,infv,meanfunc,cov,lik,inputtrainGP,outputtrainGP);%minimise the hyperparamters
ff=size(inputtest,1);
ff=500;

inputtest=log(inputtest);
disp('Predict at once')

index1=find(labelDA==-1); %output that gave a zero
index2=find(labelDA==1); % output that didnt give a zero

clement=zeros(size(inputtest,1),1);
clement(index1,:)=0; %values that the classifier predicts to give a 0
[regressoutput2,ys2v] = gp(hyp,infv,meanfunc,cov,lik, inputtrainGP, outputtrainGP, inputtest(index2,:));%Inference with GP
regressoutput2=exp(regressoutput2);
clement(index2,:)=regressoutput2;

lizz2(:,1)=clement;
lizz2(:,2)=chi1;

[X,Y] = meshgrid(1:100,1:100);

figure()
subplot(2,2,1)
surf(X',Y',reshape(clement,100,100))
shading flat
axis([1 100 1 100 ])
grid off
title('2D Marginals from Machine','FontName','Helvetica', 'Fontsize', 13);
ylabel('X1', 'FontName','Helvetica', 'Fontsize', 13);
xlabel('X2', 'FontName','Helvetica', 'Fontsize', 13);
colormap('jet')
%caxis([1 5])
h = colorbar;
ylabel(h, 'Log K(mD)','FontName','Helvetica', 'Fontsize', 13);
%set(h, 'ylim', [1 5])
set(gca, 'FontName','Helvetica', 'Fontsize', 13)
set(gcf,'color','white')
% set(gca,'xticklabel',[])
% set(gca,'yticklabel',[])

subplot(2,2,2)
surf(X',Y',reshape(chi1,100,100))
shading flat
axis([1 100 1 100 ])
grid off
title('2D Marginals from JM','FontName','Helvetica', 'Fontsize', 13);
ylabel('X1', 'FontName','Helvetica', 'Fontsize', 13);
xlabel('X2', 'FontName','Helvetica', 'Fontsize', 13);
colormap('jet')
%caxis([1 5])
h = colorbar;
ylabel(h, 'Log K(mD)','FontName','Helvetica', 'Fontsize', 13);
%set(h, 'ylim', [1 5])
set(gca, 'FontName','Helvetica', 'Fontsize', 13)
set(gcf,'color','white')
% set(gca,'xticklabel',[])
% set(gca,'yticklabel',[])


end


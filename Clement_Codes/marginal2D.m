function Xslice2input=marginal2D()
dd=importdata('chi_itg.dat',' ',1);
test=dd.data;
out=test;

test=out;

N=10;
X=test(:,1:N);
L=length(X);
numpts=100;
mean=sum(X)/L;
X1=linspace(min(X(:,1)),max(X(:,1)),numpts)';
X2=linspace(min(X(:,2)),max(X(:,2)),numpts)';
[XX1,XX2]=meshgrid(X1,X2);
Xslice2input=[XX1(:),XX2(:),repmat(mean(3:end),numpts^2,1)];

end
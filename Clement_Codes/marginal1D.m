function Xslice1input=marginal1D()
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
Xslice1input=[X1,repmat(mean(2:end),numpts,1)];
end
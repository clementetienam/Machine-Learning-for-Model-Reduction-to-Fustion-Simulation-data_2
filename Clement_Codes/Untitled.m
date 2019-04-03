clear;set(0,'defaultaxesfontsize',20); format long
load('jm_data.mat')
output=[ptotped, betanped, wped];
input=[r a kappa delta bt ip neped betan zeffped];
X=input;
[M,p]=size(X);
y=output;
sigma=0;
N=M/2;
sd=1;rng(sd);% choose random number seed
XX(:,1:p)=X;
for i=1:M
    for j=1:M
        C0(i,j) = exp(-1/2*norm(X(i,:)-X(j,:))^2);
    end 
end

C02=((X-mean(X,2))*(X-mean(X,2))')./M-1;

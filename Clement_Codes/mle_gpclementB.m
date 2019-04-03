function [likeout,df,ddf]=mle_gpclementB(pc,N,pin,C0,y)
p=exp(pc);
sigma=p(1);
l=p(2);
sigg=p(3);
C0=(sigma^2*C0(1:N,1:N).^(1/l^2))+sigg^2*eye(N);
%d = det(A);
A=0.5*(y(1:N,:)')*(((C0))\y(1:N,:));
B=0.5*(log(det(C0)));
C=((pin/2)*log(2*3.142));
likeout = -(A+B+C);
end
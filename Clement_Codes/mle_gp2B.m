function like=mle_gp2B(p,N,m,C0,y,X);%,sigg)

%sigg=0;
sigma=p(1);
l=p(2);
sigg=p(3);
n=size(X,2);
%beta=p(4:end);
C=sigma^2*C0.^(1/l^2);
CI=(C(1:N,1:N)+sigg^2*eye(N))\eye(N);
Sig=(X'*CI*X+1e-6*eye(n))\eye(n);

like = sum(log(eig(C(1:N,1:N)+sigg^2*eye(N)))) - sum(log(eig(Sig))) + ...
    y'*CI*y - y'*CI*X*Sig*X'*CI*y
p
function like=mle_gp2(p,N,m,C0,y,X);%,sigg)

%sigg=0;
sigma=p(1);l=p(2);sigg=p(3);
beta=p(4:end);
C=sigma^2*C0.^(1/l^2);

like = sum(log(eig(C(1:N,1:N)+sigg^2*eye(N)))) + ...
    (y-X*beta)'*((C(1:N,1:N)+sigg^2*eye(N))\(y-X*beta));
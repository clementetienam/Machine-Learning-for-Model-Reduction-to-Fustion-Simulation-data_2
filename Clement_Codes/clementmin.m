function like=clementmin(p,N,C0,y,X);%,sigg)

%sigg=0;
sigma=p(1);l=p(2);sigg=p(3);
beta=(X'*X)\(X'*y);
C=sigma^2*C0.^(1/l^2);

like = sum(log(eig(C(1:N,1:N)+sigg^2*eye(N)))) + ...
    (y-X*beta)'*((C(1:N,1:N)+sigg^2*eye(N))\(y-X*beta));
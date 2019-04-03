%function loglikelihood=clementSI(Bb,Theta)
function logpost=optpoly(y,Theta,X)
tic

Theta
logpost=sum((X*Theta(:,1))+(X.^2*Theta(:,2))+(X.^3*Theta(:,3))-y);

toc
end
% I2 = trapz(F,2);
% victor2=sum(I2);
% loglikelihood2=marginalv+victor2;
% like = sum(log(eig(C(1:N,1:N)+sigg^2*eye(N)))) + ...
%     (y-X*beta)'*((C(1:N,1:N)+sigg^2*eye(N))\(y-X*beta));
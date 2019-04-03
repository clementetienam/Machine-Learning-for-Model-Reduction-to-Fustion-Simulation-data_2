function Bout=mumyminimise()
set(0,'defaultaxesfontsize',20);
format long
X = importdata('inputtrain.DAT');
y= importdata('outputtrain.DAT');
[M,p]=size(X);
datause=size(y,2);
N=M;
sd=5;rng(sd);% choose random number seed
 XX(:,1:p)=X;
 ixe=p;
for ii=1:p
    ix=ixe+1;ixe=ix+p-ii;
    XX(:,ix:ixe)=X(:,ii:end).*repmat(X(:,ii),1,p+1-ii);
end
XX=[ones(M,1),XX];

XX=[ones(M,1),X];
N2=size(XX,2);
for ii=1:M
    for j=1:M
        C0(ii,j) = exp(-1/2*norm(X(ii,:)-X(j,:))^2);
    end 
end
m=zeros(M,1);
betat=(XX(1:N,:)'*XX(1:N,:))\XX(1:N,:)'*y(1:N,:);
for k=1:datause
    pp=zeros(3+N2,1);
    sigma=1;l=1;sigg=1;
    pp(1)=sigma;
    pp(2)=l;
    pp(3)=sigg;
    C=sigma^2*C0.^(1/l^2);    
pp(4:end)=(XX(1:N,:)'*XX(1:N,:))\XX(1:N,:)'*y(1:N,k);

[pp1,fval,exitflag]=fminsearch(@(pp)mle_gp2(pp,N,m,C0,y(1:N,k),XX(1:N,:)),pp);
ppoutt(:,k)=pp1;

end
Bout = reshape(ppoutt,[],1);
end


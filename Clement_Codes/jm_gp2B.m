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
ixe=p;
for i=1:p
    ix=ixe+1;
    ixe=ix+p-i;
    XX(:,ix:ixe)=X(:,i:end).*repmat(X(:,i),1,p+1-i);
end
XX=[ones(M,1),XX];
N2=size(XX,2);

% XX=[ones(M,1),X];
% N2=size(XX,2);

% GP y \sim N(m(x),C(x,x'))
% m(x) = 0
% C(x,x') = sigma^2*exp(-1/2/l^2*(x-x')^2)
% want y|xobs 
% in particular ywant|xobs, where ywant = f(xwant);

for i=1:M
    for j=1:M
        C0(i,j) = exp(-1/2*norm(X(i,:)-X(j,:))^2);
    end 
end

%C0test = pdist2(X,X,'euclidean'); %same as C0
m=zeros(M,1);

sigma=10;l=2;sigg=0;%pp=zeros(2+N2,1);
betat=(XX(1:N,:)'*XX(1:N,:))\XX(1:N,:)'*y(1:N,:);
pp(1)=sigma;pp(2)=l;pp(3)=sigg;
% pp(4:3+N2)=(XX(1:N,:)'*XX(1:N,:))\XX(1:N,:)'*y(1:N,:);
%pp=randn(3,1);
C=sigma^2*C0.^(1/l^2);
Cwant = C(N+1:end,N+1:end) - C(N+1:end,1:N)*((C(1:N,1:N)+sigg^2*eye(N))\C(1:N,N+1:end));

for k=1:3
%k=1;
    ym = sum(y(N+1:end,k))/N;
    sigma=ym;
    l=2;%pp=zeros(3+N2,1);
    sigma=1;
    l=1;
    sigg=1;
    pp(1)=sigma;
    pp(2)=l;
    pp(3)=sigg;
    C=sigma^2*C0.^(1/l^2);
    %sigg=1e-4; %ym*0.1;
    
    ywantb(:,k) = XX(N+1:end,:)*betat(:,k) + C(N+1:end,1:N)*((C(1:N,1:N)+sigg^2*eye(N))\(y(1:N,k)-XX(1:N,:)*betat(:,k)));
    outsampsb(k) = norm(ywantb(:,k)-y(N+1:end,k))/norm(y(N+1:end,k));
    r2b(k) = 1 - sum((ywantb(:,k)-y(N+1:end,k)).^2)/sum((y(N+1:end,k)-ym).^2);
    
figure(11)
plot(y(N+1:end,k),ywantb(:,k),'o');hold;
%plot(y(N+1:end,k),ywant(:,k)+sqrt(diag(Cwant)),'rx');
%plot(y(N+1:end,k),ywant(:,k)-sqrt(diag(Cwant)),'rx');
plot(y(N+1:end,k),y(N+1:end,k),'--');
hold

%pp=randn(N2+3,1); 
%pp(4:end)=(XX(1:N,:)'*XX(1:N,:))\XX(1:N,:)'*y(1:N,k);
pp=randn(3,1);

tic
[pp1,fval,exitflag]=fminsearch(@(pp)mle_gp2B(pp,N,m,C0,y(1:N,k),XX(1:N,:)),pp);%,sigg),pp);
%,sigg),pp);
optim_time_mle = toc


sigma=pp1(1);
l=pp1(2);
sigg=pp1(3);

%pp=pp1;
C=sigma^2*C0.^(1/l^2);
CI=(C(1:N,1:N)+sigg^2*eye(N))\eye(N);
Sigbeta=(XX(1:N,:)'*CI*XX(1:N,:))\eye(N2);
mbeta = Sigbeta*XX(1:N,:)'*CI*y(1:N,k);
mbetas(:,k) = mbeta;
ppp(:,k)=pp1;

Cwant=C(N+1:end,N+1:end)+sigg^2*eye(N) - C(N+1:end,1:N)*CI*C(N+1:end,1:N)' + ...
    (XX(N+1:end,:)-C(N+1:end,1:N)*CI*XX(1:N,:))*Sigbeta*(XX(N+1:end,:)-C(N+1:end,1:N)*CI*XX(1:N,:))';

Cwants(:,:,k) = Cwant; 
    ywant(:,k) = (XX(N+1:end,:)-C(N+1:end,1:N)*CI*XX(1:N,:))*mbeta + ...
        C(N+1:end,1:N)*CI*y(1:N,k);
    outsamps(k) = norm(ywant(:,k)-y(N+1:end,k))/norm(y(N+1:end,k))    
    outsampsm(k) = sum((ywant(:,k)-y(N+1:end,k))./y(N+1:end,k))/N
    
    r2(k) = 1 - sum((ywant(:,k)-y(N+1:end,k)).^2)/sum((y(N+1:end,k)-ym).^2)
    
figure(k)
plot(y(N+1:end,k),ywant(:,k),'o');hold;
plot(y(N+1:end,k),ywant(:,k)+sqrt(diag(Cwant)),'x');
plot(y(N+1:end,k),ywant(:,k)-sqrt(diag(Cwant)),'x');
plot(sort(y(N+1:end,k)),sort(y(N+1:end,k)),'--');
xlabel('Real output');ylabel('GP estimate')
legend('mean','mean+std','mean-std','truth')
hold

fra(k)=sum((ywant(:,k)-sqrt(diag(Cwant))<=y(N+1:end,k)).* ...
    (y(N+1:end,k)<=ywant(:,k)+sqrt(diag(Cwant))))/N

k
end


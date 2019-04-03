clear;set(0,'defaultaxesfontsize',20); format long
load('jm_data.mat')
output=[ptotped, betanped, wped];
input=[r a kappa delta bt ip neped betan zeffped];
X=input;[M,p]=size(X);
y=output;sigma=0;
N=M/2;
sd=5;rng(sd);% choose random number seed
 XX(:,1:p)=X;ixe=p;
for i=1:p
    ix=ixe+1;ixe=ix+p-i;
    XX(:,ix:ixe)=X(:,i:end).*repmat(X(:,i),1,p+1-i);
end
XX=[ones(M,1),XX];
% N2=size(XX,2);

XX=[ones(M,1),X];
N2=size(XX,2);

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
m=zeros(M,1);
ytest=y(N+1:end,:);
sigma=10;l=2;sigg=0;pp=zeros(2+N2,1);
betat=(XX(1:N,:)'*XX(1:N,:))\XX(1:N,:)'*y(1:N,:);
pp(1)=sigma;pp(2)=l;%pp(3)=sigg;
% pp(4:3+N2)=(XX(1:N,:)'*XX(1:N,:))\XX(1:N,:)'*y(1:N,:);
%pp=randn(3,1);
C=sigma^2*C0.^(1/l^2);
Cwant = C(N+1:end,N+1:end) - C(N+1:end,1:N)*((C(1:N,1:N)+sigg^2*eye(N))\C(1:N,N+1:end));

for k=1:3

    ym = sum(y(N+1:end,k))/N;
    sigma=ym;l=2;pp=zeros(3+N2,1);
    sigma=1;l=1;sigg=1;
    pp(1)=sigma;pp(2)=l;pp(3)=sigg;
    C=sigma^2*C0.^(1/l^2);
    %sigg=1e-4; %ym*0.1;
    
    ywantb(:,k) = XX(N+1:end,:)*betat(:,k) + C(N+1:end,1:N)*((C(1:N,1:N)+sigg^2*eye(N))\(y(1:N,k)-XX(1:N,:)*betat(:,k)));
    outsampsb(k) = norm(ywantb(:,k)-y(N+1:end,k))/norm(y(N+1:end,k))
    r2b(k) = 1 - sum((ywantb(:,k)-y(N+1:end,k)).^2)/sum((y(N+1:end,k)-ym).^2)
    
figure(11)
plot(y(N+1:end,k),ywantb(:,k),'o');hold;
%plot(y(N+1:end,k),ywant(:,k)+sqrt(diag(Cwant)),'rx');
%plot(y(N+1:end,k),ywant(:,k)-sqrt(diag(Cwant)),'rx');
plot(y(N+1:end,k),y(N+1:end,k),'--');
hold

%pp=randn(N2+3,1); 
pp(4:end)=(XX(1:N,:)'*XX(1:N,:))\XX(1:N,:)'*y(1:N,k);

tic
[pp1,fval,exitflag]=fminsearch(@(pp)mle_gp2(pp,N,m,C0,y(1:N,k),XX(1:N,:)),pp);%,sigg),pp);
%,sigg),pp);
optim_time_mle = toc

% tol=1;fval=mle_gp2(pp,N,m,C0,y(1:N,k),XX(1:N,:));%,sigg);
% step=1;pp1=pp;
% 
% while tol>1e-3
% 
%     pp=pp1;fvalo=fval;
%     gg = dlike(pp,N,m,C0,y(1:N,k),XX(1:N,:));%,sigg);
%     gg=gg/norm(gg);
%     step=100;
%     pp1 = pp - step*gg;
%     fval = mle_gp2(pp1,N,m,C0,y(1:N,k),XX(1:N,:));%,sigg);
%     while fval>fvalo
%         step=step/2;
%         pp1 = pp - step*gg;
%         fval = mle_gp2(pp1,N,m,C0,y(1:N,k),XX(1:N,:));%,sigg);
%     end   
% %         
% %     while tols>1e-3
% %     gg1 = dlike(pp1,N,m,C0,y(1:N,k),XX(1:N,:),sigg);
% %     tols = norm(gg1);
% %     step = step + gg1'*gg;
% %     end
%     tol = norm(fvalo-fval)/norm(fvalo);
%    
% end
% fval

sigma=pp1(1)
l=pp1(2)
sigg=pp1(3)
beta=pp1(4:end);

%pp=pp1;
C=sigma^2*C0.^(1/l^2);
Cwant = C(N+1:end,N+1:end) - C(N+1:end,1:N)*((C(1:N,1:N)+sigg^2*eye(N))\C(1:N,N+1:end));

    ywant(:,k) = XX(N+1:end,:)*beta + C(N+1:end,1:N)*((C(1:N,1:N)+sigg^2*eye(N))\(y(1:N,k)-XX(1:N,:)*beta));
    outsamps(k) = norm(ywant(:,k)-y(N+1:end,k))/norm(y(N+1:end,k))    
    outsampsm(k) = sum((ywant(:,k)-y(N+1:end,k))./y(N+1:end,k))/N
    
    r2(k) = 1 - sum((ywant(:,k)-y(N+1:end,k)).^2)/sum((y(N+1:end,k)-ym).^2)
    
figure(k)
plot(y(N+1:end,k),ywant(:,k),'o');hold;
plot(y(N+1:end,k),ywant(:,k)+sqrt(diag(Cwant)),'rx');
plot(y(N+1:end,k),ywant(:,k)-sqrt(diag(Cwant)),'rx');
plot(y(N+1:end,k),y(N+1:end,k),'--');
hold

end

figure()
for i=1:3
subplot(2,2,i)
plot(ytest(:,i),ywant(:,i),'o');
xlabel('Real output','FontName','Helvetica', 'Fontsize', 13);
ylabel('GP estimate','FontName','Helvetica', 'Fontsize', 13)
title (sprintf('output %d vs input %d',i,i))
%title('JMdata','FontName','Helvetica', 'Fontsize', 13);
set(gca, 'FontName','Helvetica', 'Fontsize', 13)
set(gcf,'color','white')
end

%% Calculate CoD and L2
for i=1:3
Lerror=(norm(ytest(:,i)-ywant(:,i))/norm(ytest(:,i)))^0.5;
L_2sparse=1-(Lerror^2);
Lall(:,i)=L_2sparse;
%Coefficient of determination
 outputtest=ytest(:,i);
for ii=1:numel(ytest(:,1))
    outputreq(ii)=outputtest(ii)-mean(outputtest);
end

outputreq=outputreq';
CoDsparse=1-(norm(ytest(:,i)-ywant(:,i))/norm(outputreq));
CoDsparse=1 - (1-CoDsparse)^2 ;
CoDall(:,i)=CoDsparse;
end


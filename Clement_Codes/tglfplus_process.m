clear;set(0,'defaultaxesfontsize',20); format long
load tglf_plus
N=size(test,1)
[v,i]=find(sum(abs(test')) < 1e3);
% Ni=size(i,2);
% H=sparse(Ni,N);H(:,i)=speye(Ni);
% ix=find(sum(H,1)==0);
test=test(i,:);
N=size(test,1);

    mean=sum(test)/N
    var=(test'-mean'*ones(1,N))*(test'-mean'*ones(1,N))'/N

    test1=test;
        
    for i=1:28
        
    ix=find((test1(:,i)<ones(N,1)*(mean(i)+2*sqrt(var(i,i)))).*(test1(:,i)>ones(N,1)*(mean(i)-2*sqrt(var(i,i)))));
    test1=test1(ix,:);N=size(test1,1);
    
    end
    
% for i=1:28
% 
%     var=sum(test(:,i)
%     
% end
    
    input=test(:,2:22);output=test(:,23:28);
%input=input(1:3e5,:);output=output(1:3e5,:);


oo=1;ix=1:length(output);
%ix=find(output(:,oo)>0);
%input=input(ix,:);output=output(ix,:);

%   ix=find((output(:,1)<=10));%.*(output(:,1)>=exp(-2)));
    output1=asinh(output(ix,oo));
    input1=asinh(input(ix,:));
input=input1;
 %output1=log(output(:,1));
 output=output1;
%input=asinh(input);
X=input;[M,p]=size(X);N=ceil(M/2);%N=M-1000;

for i=1:10
xm=1/M*(ones(1,M)*X);
Xm=ones(M,1)*xm;
SS=(X-Xm)'*(X-Xm)/M;S=sqrtm(SS);
% [pv,pe]=eig(SS);
% md=X'.*(SS\X');
% for i=1:p
%     mds=(S\(X-ones(length(X),1)*xm)');
%     ixm=find(abs(mds(i,:))<=1.5);
%     X=X(ixm,:);output=output(ixm,:);
% end
% [M,p]=size(X);

mahdis=sum((X-Xm)'.*(SS\(X-Xm)'));
%mahdis=sum(X'.*(SS\X'));
%mm=sum(mahdis)/M;
%ms=sqrt(sum((mahdis-mm).^2)/M);
%ixm=find(abs(mahdis-mm)<=1.5);
ixm=find(mahdis<1.2*p);
X=X(ixm,:);[M,p]=size(X);N=ceil(M/2);%N=M-1000;
output=output(ixm,:);
y=output;M
end

ym=1/M*(ones(1,M)*y);
Ym=ones(M,1)*ym;
SSy=(y-Ym)'*(y-Ym)/M;Sy=sqrtm(SSy);
mahdisy=(y-Ym).^2/SSy;
ixm=find(mahdisy<1.5);
X=X(ixm,:);[M,p]=size(X);N=ceil(M/2);%N=M-1000;
output=output(ixm,:);
y=output;M


xm=1/M*(ones(1,M)*X);
Xm=ones(M,1)*xm;
SS=(X-Xm)'*(X-Xm)/M;S=sqrtm(SS);
X1=(S\(X-Xm)')';

% [pv,pe]=eig(SS);
% p=18;
% sum(diag(pe(1:p,1:p)))/sum(diag(pe))
% X1=(sqrtm(pe(1:p,1:p))\((X-Xm)*pv(:,1:p))')';
% %norm(X1*pv(:,1:p)'-(X-Xm),'fro')/norm(X,'fro')

% X1=X*pv(:,1:p);
% norm(X1*pv(:,1:p)'-X,'fro')/norm(X,'fro')


% N=round(M/2);
% h=max(1,sort(unique(round(rand(N,1)*M))));
% X1=X1(h,:);y=y(h,:);M=size(h,1);

X=X1;

tic
XX(:,1:p)=X;ixe=p;
for i=1:p
    ix=ixe+1;ixe=ix+p-i;
    XX(:,ix:ixe)=X(:,i:end).*repmat(X(:,i),1,p+1-i);
end
toc
tic
N2=size(XX,2);
XXX(:,1:N2)=XX;ixe=N2;num=0;
for i=1:p
    ix=ixe+1;ixe=ix+(N2-p-num)-1;
    XXX(:,ix:ixe)=XX(:,p+num+1:end).*repmat(X(:,i),1,(N2-p-num));
    num=num+(p+1-i);
end
toc
%% XXX is cubic but still not sufficient.  It might be necessary to go to much higher degree polynomials.

tic
 V = gpcbasis_create('M', 'm', p, 'p', 4);
 toc
 tic
 XT=gpcbasis_evaluate(V, X');X10=XT';
toc

X=[ones(M,1),X];
XX=[ones(M,1),XX];
XXX=[ones(M,1),XXX];

rng(1)

% h=1:N;
% ix=N+1:M;

N=round(M/2);
h=max(1,sort(unique(round(rand(N,1)*M))));
N=length(h);
H=sparse(N,M);H(:,h)=speye(N);
ix=find(sum(H,1)==0);

beta=(X(h,:)'*X(h,:))\(X(h,:)'*y(h,:));
norm(X*beta-y,'fro')/norm(y,'fro')
out1=norm(X(ix,:)*beta-y(ix,:),'fro')/norm(y(ix,:),'fro')

beta2=(XX(h,:)'*XX(h,:))\(XX(h,:)'*y(h,:));
norm(XX*beta2-y,'fro')/norm(y,'fro')
out2=norm(XX(ix,:)*beta2-y(ix,:),'fro')/norm(y(ix,:),'fro')

beta3=(XXX(h,:)'*XXX(h,:)+1*eye(size(XXX(h,:),2)))\(XXX(h,:)'*y(h,:));
norm(XXX*beta3-y,'fro')/norm(y,'fro')
out3=norm(XXX(ix,:)*beta3-y(ix,:),'fro')/norm(y(ix,:),'fro')
    ym = sum(y(ix,1))/(M-N);
    r2_3(1) = 1 - sum((XXX(ix,:)*beta3(:,1)-y(ix,1)).^2)/sum((y(ix,1)-ym).^2)

tic
b10=(X10(h,:)'*X10(h,:)+1*eye(size(X10(h,:),2)))\(X10(h,:)'*y(h,:));
t1=toc

    ym = sum(y(ix,1))/(M-N);
    r2_10(1) = 1 - sum((X10(ix,:)*b10(:,1)-y(ix,1)).^2)/sum((y(ix,1)-ym).^2)

    
norm(X10*b10-y,'fro')/norm(y,'fro')
out10=norm(X10(ix,:)*b10-y(ix,:),'fro')/norm(y(ix,:),'fro')
%var10=inv(X10'*X10+1*eye(size(X10(h,:),2)));

N10=size(b10,1)

itth=[1:N10]';b10n=b10;
ittho=itth;thresh=1;
j=1

while size(itth,1)>10 && thresh>0

    itth=find(abs(b10)/max(abs(b10))>.01);
    thresh=size(ittho,1)-size(itth,1)
    
    ittho=ittho(itth);
    
    b10=(X10(h,ittho)'*X10(h,ittho)+10*eye(size(X10(h,ittho),2)))\(X10(h,ittho)'*y(h,:));

    norm(X10(:,ittho)*b10-y,'fro')/norm(y,'fro');
    j=j+1;
        
end

size(ittho)

bee=zeros(N10,1);
bee(ittho)=b10;


figure
plot(bee);

    ym = sum(y(ix,1))/(M-N);
    r2_10(1) = 1 - sum((X10(ix,ittho)*b10(:,1)-y(ix,1)).^2)/sum((y(ix,1)-ym).^2)

    
norm(X10(:,ittho)*b10-y,'fro')/norm(y,'fro')
out10=norm(X10(ix,ittho)*b10-y(ix,:),'fro')/norm(y(ix,:),'fro')



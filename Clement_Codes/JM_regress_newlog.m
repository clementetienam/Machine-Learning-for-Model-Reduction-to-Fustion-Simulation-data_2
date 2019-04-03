clear;set(0,'defaultaxesfontsize',20); format long
load('JM_tauth_data')
input=[r0,a0,kappa,delta,ip,b0,nebar,zeff,ploss];
output=[tauth];
%folder = fileparts(which(mfilename)); 
addpath(genpath('C:\Work\GSLIB\sgsim\postdoc law\repreliminary\sglib-master\sglib-master'));
% load ../jm_data
% input=[a,betan,bt,delta,ip,kappa,neped,r,zeffped];
% output=[ptotped,betanped,wped];

% load('JM_eped_data')
% %input=[r,a,kappa,delta,ip,bt,betan,neped,nesep,zeffped];
% input=[r,a,kappa,delta,ip,bt,neped,zeffped,betan];
% % input=[a,betan,bt,delta,ip,kappa,neped,r,zeffped];
% output=[ptotped,ptottop,betanped,wped];

%input=input(1:598,:);output=output(1:598,:);

input=log(input);output=log(output);

X=input;[M,p]=size(X);N=2100;%ceil(M/2);%N=M-1000;

XX(:,1:p)=X;ixe=p
for i=1:p
    ix=ixe+1;ixe=ix+p-i;
    XX(:,ix:ixe)=X(:,i:end).*repmat(X(:,i),1,p+1-i);
end
N2=size(XX,2);
XXX(:,1:N2)=XX;ixe=N2;num=0;
for i=1:p
    ix=ixe+1;ixe=ix+(N2-p-num)-1;
    XXX(:,ix:ixe)=XX(:,p+num+1:end).*repmat(X(:,i),1,(N2-p-num));
    num=num+(p+1-i);
end


V = gpcbasis_create('M', 'm', p, 'p', 4);
XT=gpcbasis_evaluate(V, X');X10=XT';


X=[ones(M,1),X];
XX=[ones(M,1),XX];
XXX=[ones(M,1),XXX];

y=output;K=size(y,2);sigma=1;
linreg=(X'*X+sigma^2*eye(size(X,2)))\X'*y;
tot1=norm(X*linreg-y)/norm(y);

linreg=(X(1:N,:)'*X(1:N,:))\X(1:N,:)'*y(1:N,:);
insamp = norm(X(1:N,:)*linreg-y(1:N,:))/norm(y(1:N,:));
outsamp = norm(exp(X(N+1:end,:)*linreg)-exp(y(N+1:end,:)))/norm(exp(y(N+1:end,:)));
outsamps(1) = outsamp
%norm(X(N+1:end,:)*linreg(:,1)-y(N+1:end,1))/norm(y(N+1:end,1));
var=diag(X(N+1:end,:)*inv(X(1:N,:)'*X(1:N,:))*X(N+1:end,:)');
ym = sum(exp(y(N+1:end,1)))/(M-N);
r1b(1) = 1 - sum((exp(X(N+1:end,:)*linreg(:,1))-exp(y(N+1:end,1))).^2)/sum((exp(y(N+1:end,1))-ym).^2);


lin0=randn(p+1,1);
options=optimset('MaxFunEvals',1e4,'MaxIter',1e4);
lin0=linreg;tic
[linr,fval,exitflag]=fminsearch(@(lin)flo(lin,X(1:N,:),y(1:N,:)),lin0,options);
toc

outnlin=norm(exp(X(N+1:end,:)*linr)-exp(y(N+1:end,:)))/norm(exp(y(N+1:end,:)))
linr

% 
% figure(11)
% plot(y(N+1:end,1), X(N+1:end,:)*linreg(:,1), 'o');hold;
%  plot(y(N+1:end,1), X(N+1:end,:)*linreg(:,1) + sum(y(:,1))/N*sqrt(var), '+')
%  plot(y(N+1:end,1),X(N+1:end,:)*linreg(:,1) - sum(y(:,1))/N*sqrt(var), '+')
% xlabel('real output');ylabel('linear model')
% plot(sort(y(N+1:end,1)),sort(y(N+1:end,1)),'--');hold


linreg2=(XX'*XX+sigma^2*eye(size(XX,2)))\XX'*y;
tot2=norm(XX*linreg2-y)/norm(y);

linreg2=(XX(1:N,:)'*XX(1:N,:)+sigma^2*eye(N2+1))\XX(1:N,:)'*y(1:N,:);
insamp2 = norm(XX(1:N,:)*linreg2-y(1:N,:))/norm(y(1:N,:));
outsamp2 = norm(XX(N+1:end,:)*linreg2-y(N+1:end,:))/norm(y(N+1:end,:));
outsamps2(1) = norm(exp(XX(N+1:end,:)*linreg2(:,1))-exp(y(N+1:end,1)))/norm(exp(y(N+1:end,1)));

ym = sum(exp(y(N+1:end,1)))/(M-N);
    r2b(1) = 1 - sum((XX(N+1:end,:)*linreg2(:,1)-y(N+1:end,1)).^2)/sum((exp(y(N+1:end,1))-ym).^2)


        
var=diag(XX(N+1:end,:)*inv(XX(1:N,:)'*XX(1:N,:)+sigma^2*eye(N2+1))*XX(N+1:end,:)');


% figure(11)
% plot(y(N+1:end,1), XX(N+1:end,:)*linreg2(:,1), 'o');hold;
% plot(y(N+1:end,1), XX(N+1:end,:)*linreg2(:,1) + sum(y(:,1))/N*sqrt(var), '+')
% plot(y(N+1:end,1), XX(N+1:end,:)*linreg2(:,1) - sum(y(:,1))/N*sqrt(var), '+')
% xlabel('real output');ylabel('regression');
% %title('ptotped')
% plot(sort(y(N+1:end,1)),sort(y(N+1:end,1)),'--');hold



sigma=.2;
P=1/sigma^2*speye(size(XXX,2));%diag([0.01*ones(p,1);5*ones(N2-p,1);1000*ones(size(XXX,2)-N2,1)]);
%P=0;
linreg3=(XXX'*XXX+P)\XXX'*y;
tot3=norm(XXX*linreg3-y)/norm(y);

linreg3=(XXX(1:N,:)'*XXX(1:N,:)+P)\XXX(1:N,:)'*y(1:N,:);
insamp3 = norm(XXX(1:N,:)*linreg3-y(1:N,:))/norm(y(1:N,:))
outsamp3 = norm(XXX(N+1:end,:)*linreg3-y(N+1:end,:))/norm(y(N+1:end,:))
outsamps3(1) = norm(exp(XXX(N+1:end,:)*linreg3(:,1))-exp(y(N+1:end,1)))/norm(exp(y(N+1:end,1)));
% outsamps3(2) = norm(XXX(N+1:end,:)*linreg3(:,2)-y(N+1:end,2))/norm(y(N+1:end,2));
% outsamps3(3) = norm(XXX(N+1:end,:)*linreg3(:,3)-y(N+1:end,3))/norm(y(N+1:end,3));

ym = sum(exp(y(N+1:end,1)))/(M-N);
    r3b(1) = 1 - sum((exp(XXX(N+1:end,:)*linreg3(:,1))-exp(y(N+1:end,1))).^2)/sum((exp(y(N+1:end,1))-ym).^2)

    yme = sum(exp(y(N+1:end,1)))/(M-N);    
    r3e(1) = 1 - sum((exp(XXX(N+1:end,:)*linreg3(:,1))-exp(y(N+1:end,1))).^2)/sum((exp(y(N+1:end,1))-yme).^2);
    yma = sum(y(1:end,1))/M;
    r3a = 1 - sum((XXX*linreg3(:,1)-y(:,1)).^2)/sum((y(:,1)-yma).^2)
    ymi = sum(y(1:N,1))/N;
    r3i = 1 - sum((XXX(1:N,:)*linreg3(:,1)-y(1:N,1)).^2)/sum((y(1:N,1)-ymi).^2)
    %     ym = sum(y(N+1:end,2))/(M-N);
%     r3b(2) = 1 - sum((XXX(N+1:end,:)*linreg3(:,2)-y(N+1:end,2)).^2)/sum((y(N+1:end,2)-ym).^2);
%     ym = sum(y(N+1:end,3))/(M-N);
%     r3b(3) = 1 - sum((XXX(N+1:end,:)*linreg3(:,3)-y(N+1:end,3)).^2)/sum((y(N+1:end,3)-ym).^2);

    
var=diag(XXX(N+1:end,:)*inv(XXX(1:N,:)'*XXX(1:N,:)+sigma^2*eye(size(XXX,2)))*XXX(N+1:end,:)');


h=1:N;ix=N+1:M;
N10=size(X10,2);
sigma=.1
tic
b10=(X10(h,:)'*X10(h,:)+1/sigma^2*eye(size(X10(h,:),2)))\X10(h,:)'*y(1:N,:);
t1=toc

norm(X10*b10-y,'fro')/norm(y,'fro')
%out10=norm(X10(ix,:)*b10-y(ix,:),'fro')/norm(y(ix,:),'fro')
out10=norm(exp(X10(ix,:)*b10)-exp(y(ix,:)),'fro')/norm(exp(y(ix,:)),'fro')
var10=inv(X10'*X10+1*eye(size(X10(h,:),2)));

% ym = sum(exp(y(N+1:end,1)))/(M-N);
%     r10b(1) = 1 - sum((X10(N+1:end,:)*b10(:,1)-y(N+1:end,1)).^2)/sum((y(N+1:end,1)-ym).^2)
     
    ym = sum(exp(y(N+1:end,1)))/(M-N);
    r10b(1) = 1 - sum((exp(X10(N+1:end,:)*b10(:,1))-exp(y(N+1:end,1))).^2)/sum((exp(y(N+1:end,1))-ym).^2)

    V = gpcbasis_create('M', 'm', p, 'p', 5);
XT=gpcbasis_evaluate(V, X(:,2:end)');X5=XT';

  
b5=(X5(h,:)'*X5(h,:)+1/sigma^2*eye(size(X5(h,:),2)))\X5(h,:)'*y(1:N,:);
t1=toc

norm(X5*b5-y,'fro')/norm(y,'fro')
%out5=norm(X5(ix,:)*b5-y(ix,:),'fro')/norm(y(ix,:),'fro')
out5=norm(exp(X5(ix,:)*b5)-exp(y(ix,:)),'fro')/norm(exp(y(ix,:)),'fro')
var5=inv(X5'*X5+1*eye(size(X5(h,:),2)));

% ym = sum(exp(y(N+1:end,1)))/(M-N);
%     r5b(1) = 1 - sum((X5(N+1:end,:)*b5(:,1)-y(N+1:end,1)).^2)/sum((y(N+1:end,1)-ym).^2)
     
    ym = sum(exp(y(N+1:end,1)))/(M-N);
    r5b(1) = 1 - sum((exp(X5(N+1:end,:)*b5(:,1))-exp(y(N+1:end,1))).^2)/sum((exp(y(N+1:end,1))-ym).^2)

    
        V = gpcbasis_create('M', 'm', p, 'p', 6);
XT=gpcbasis_evaluate(V, X(:,2:end)');X6=XT';

  
b6=(X6(h,:)'*X6(h,:)+1/sigma^2*eye(size(X6(h,:),2)))\X6(h,:)'*y(1:N,:);
t1=toc

norm(X6*b6-y,'fro')/norm(y,'fro')
%out6=norm(X6(ix,:)*b6-y(ix,:),'fro')/norm(y(ix,:),'fro')
out6=norm(exp(X6(ix,:)*b6)-exp(y(ix,:)),'fro')/norm(exp(y(ix,:)),'fro')
var6=inv(X6'*X6+1*eye(size(X6(h,:),2)));

% ym = sum(exp(y(N+1:end,1)))/(M-N);
%     r6b(1) = 1 - sum((X6(N+1:end,:)*b6(:,1)-y(N+1:end,1)).^2)/sum((y(N+1:end,1)-ym).^2)
     
    ym = sum(exp(y(N+1:end,1)))/(M-N);
    r6b(1) = 1 - sum((exp(X6(N+1:end,:)*b6(:,1))-exp(y(N+1:end,1))).^2)/sum((exp(y(N+1:end,1))-ym).^2)


% for i=1:4
% figure(11+i)
% plot(y(N+1:end,i), XXX(N+1:end,:)*linreg3(:,i), 'o');hold;
% %plot(y(N+1:end,1), XXX(N+1:end,:)*linreg3(:,1) + sum(y(:,1))/M*sqrt(var), '+')
% %plot(y(N+1:end,1), XXX(N+1:end,:)*linreg3(:,1) - sum(y(:,1))/M*sqrt(var), '+')
% % plot(y(N+1:end,1), XXX(N+1:end,:)*linreg3(:,1) + sqrt(var/N)*2, '+')
% % plot(y(N+1:end,1), XXX(N+1:end,:)*linreg3(:,1) - sqrt(var/N)*2, '+')
% plot(y(1:N,i), XXX(1:N,:)*linreg3(:,i), 'o');
% xlabel('real output');ylabel('regression');
% %title('ptotped')
% plot(sort(y(N+1:end,i)),sort(y(N+1:end,i)),'--');hold
%     
% yma = sum(y(1:end,i))/M;
% r3a = 1 - sum((XXX*linreg3(:,i)-y(:,i)).^2)/sum((y(:,i)-yma).^2)
%     
% end

fig2loglin=exp(X(N+1:end,:)*linreg(:,1));
fig2logsix=exp(X6(N+1:end,:)*b6(:,1));
fig2true=exp(y(N+1:end,1));


% q1=5;q2=6
% plot(X(:,q1),X(:,q2),'o');hold
% plot(X(ixpoint,q1),X(ixpoint,q2),'o');hold

ixpoint=round(rand*M)
X(ixpoint,2:end);
xb=X(ixpoint,2:end)'*ones(1,1000);

 p0=log([5,2,1.7,0.45,10,6,7,1.45,250]');
 xb=p0*ones(1,1000);


figure(2)
for k=1:p

%figure(k);
xx=linspace(min(X(:,k+1)),max(X(:,k+1)),1000);
%xb=X(ixpoint,2:end)'*ones(1,1000);
xb=p0*ones(1,1000);
xb(k,:)=xx;
xxt=gpcbasis_evaluate(V, xb);
% plot(xx,b10'*xxt,'Linewidth',2);hold;
% plot(xx,linreg(1)+linreg(2:end)'*xb,'Linewidth',2);hold
% figure
subplot(3,3,k), plot(exp(xx),exp(b6'*xxt),'Linewidth',2);hold;
plot(exp(xx),exp(linreg(1)+linreg(2:end)'*xb),'Linewidth',2);hold

variable(k,:)=exp(xx);
slicelog6(k,:)=exp(b6'*xxt);
slicelog1(k,:)=exp(linreg(1)+linreg(2:end)'*xb);

title(strcat('variable #',num2str(k)));

end




legend('sextic','linear')

R2=[r1b,r2b,r3b,r10b,r5b,r6b]
L2rel=[outsamps,outsamps2,outsamps3,out10,out5,out6]

figure
plot(R2,'-o','Linewidth',2);hold
plot(1-L2rel,'-x','Linewidth',2);hold
legend('R^2','1-L^2_{rel}')


figure
plot(exp(y(N+1:end,1)), exp(X(N+1:end,:)*linreg(:,1)), 'o');hold;
plot(exp(y(N+1:end,1)),exp(X6(N+1:end,:)*b6(:,1)),'o')
plot(sort(exp(y(N+1:end,1))),sort(exp(y(N+1:end,1))),'--');hold
legend('log linear','log sextic')



%     vidObj = VideoWriter('random_slices.avi');
%     open(vidObj);
% 
% for i=1:500
%     
%     i
%     ixpoint=round(rand*M)
%     %X(ixpoint,2:end);
% 
% xb=X(ixpoint,2:end)'*ones(1,1000);
% 
% figure(2)
% for k=1:p
% 
% %figure(k);
% xx=linspace(min(X(:,k+1)),max(X(:,k+1)),1000);
% xb=X(ixpoint,2:end)'*ones(1,1000);
% xb(k,:)=xx;
% xxt=gpcbasis_evaluate(V, xb);
% % plot(xx,b10'*xxt,'Linewidth',2);hold;
% % plot(xx,linreg(1)+linreg(2:end)'*xb,'Linewidth',2);hold
% % figure
% subplot(3,3,k), plot(exp(xx),exp(b6'*xxt),'Linewidth',2);hold;
% plot(exp(xx),exp(linreg(1)+linreg(2:end)'*xb),'Linewidth',2);hold
% 
% title(strcat('variable #',num2str(k)));
% 
% end
% 
% legend('sextic','linear')
% frame=getframe(gcf);       
% writeVideo(vidObj,frame);
%   
% end
%     
%     % Close the file.
%     close(vidObj);
%     
% 
%     for k=1:p
%         subplot(3,3,k), hist(exp(X(:,k+1)))
% title(strcat('variable #',num2str(k)));
% 
%     end
    
    
    
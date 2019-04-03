%----------------------------------------------------------------------------------
% GP modelling of high data using parallel architecture
% Running Ensembles
% Author: Clement Etienam ,PhD Petroleum Engineering 2015-2018
% Supervisor:Professor Kody Law
%-----------------------------------------------------------------------------------
%% 
clc;
clear;
close all;

disp(' We will implement the dense and sparse GP for the Tauth model' );
disp( 'Author: Clement Etienam' )
disp( 'Supervisor: Professor Kody Law' )
disp( 'GP Regression Model' )
N22=input( ' Given prior information to the data Ascii How many rows do you want as testing  '); %
N=input( ' enter the division of the input data X that you want (it must be divisible with value above)  '); % 

oldfolder=cd;
cd(oldfolder) ;

disp( 'create the folders')
load('jm_data.mat')
output=[ptotped, betanped, wped];
input=[r a kappa delta bt ip neped betan zeffped];
Xin=input;
[rowinput,columninput]=size(Xin);
yin=output;
columnoutput=size(yin,2);

% disp(['Number of row of dense input data = ' num2str(rowinput)]);
%  %pause
% 
% % N22=input( ' Given information above How many rows do you want as testing  '); %
% disp(['Number of row of testing input data = ' num2str(N22)]);

Xtest=Xin(1:N22,:);
ytest=yin(1:N22,:);

Xpred=Xin(N22+1:end,:);
ypred=yin(N22+1:end,:);



for j=1:N
f = 'MASTER';
folder = strcat(f, sprintf('%.d',j));
mkdir(folder);
end
cd(oldfolder) ;

%% Coppying simulation files
disp( 'copy files')
for j=1:N
f = 'MASTER';
folder = strcat(f, sprintf('%.d',j));
copyfile('mumyminimise.m',folder)
copyfile('mle_gp2.m',folder)
end
cd(oldfolder) ;
tic;

C = mat2cell(Xtest, repmat(N22/N, N, 1), columninput);
D = mat2cell(ytest, repmat(N22/N, N, 1), columnoutput);
cd(oldfolder) ;
for i=1:N %list of folders 
    
    f = 'MASTER';
   folder = strcat(f, sprintf('%.d',i));
   
    cd(folder) % changing directory 
    
    input=C{i};
    output=D{i};
   
    
    save('inputtrain.DAT','input','-ascii');
    save('outputtrain.DAT','output','-ascii');
    
    cd(oldfolder) % returning to original cd
    
end
cd(oldfolder);

parfor ii=1:N %list of folders 
    
    f = 'MASTER';
    folder = strcat(f, sprintf('%.d',ii));
    cd(folder)

Bout(:,ii)=mumyminimise();

cd(oldfolder) % setting original directory
fprintf('Finished folder %d .\n', ii);
end
cd(oldfolder) ;
ppout=reshape(Bout,[],columnoutput,N);

%% Now with the various sub hyper paramteres predict the new values given test input data
set(0,'defaultaxesfontsize',20); format long

ppin=mean(ppout,3);
%pp=mean(pp,2);

ain=0;
bin=0;
for i=1:N
    ain=C{i}+ain;
    bin=D{i}+bin;
    
end

Xs=ain./4;

ys=bin./4;

X=[Xs;Xpred];
y=[ys;ypred];

[M,p]=size(X);

N=size(Xs,1);
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
for i=1:M
    for j=1:M
        C0(i,j) = exp(-1/2*norm(X(i,:)-X(j,:))^2);
    end 
end

%C0test = pdist2(X,X,'euclidean'); %same as C0
m=zeros(M,1);

betat=(XX(1:N,:)'*XX(1:N,:))\XX(1:N,:)'*y(1:N,:);
for k=1:columnoutput
%k=1;
  pp=ppin(:,k);
    ym = sum(y(N+1:end,k))/N;
    sigmaa=ym;
sigmaa=pp(1);
l=pp(2);
sigg=pp(3);

%pp=pp1;
C=sigmaa^2*C0.^(1/l^2);
CI=(C(1:N,1:N)+sigg^2*eye(size(Xpred,1)))\eye(size(Xpred,1));
Sigbeta=(XX(1:N,:)'*CI*XX(1:N,:))\eye(N2);
mbeta = Sigbeta*XX(1:N,:)'*CI*y(1:N,k);
mbetas(:,k) = mbeta;
%ppp(:,k)=pp1;

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
toc;
function clement=predictclassregress(labelDA,inputtest,inputtrainGP,outputtrainGP)
%% GP classification/Regression model for the Chi data
disp(' We will implement the sparse GP together with classification for the Chi model' );
disp( 'Author: Dr Clement Etienam' )
disp( 'Supervisor: Professor Kody Law' )
disp( 'Supervised classfication Modelling' )
set(0,'defaultaxesfontsize',20); format long


%% Train the GP Sparse approximation
meanfunc=[];% empty: don't use a mean function
n = 30; sn = 0.99;
 lik = {@likGauss};    hyp.lik = log(sn); inf = @infGaussLik;
 cov = {@covSEiso}; 
 hyp.cov = log([9.5;12.5]); % Matern class d=5
p=size(inputtest,2);

pkg load statistics
for j=1:p
xu = normrnd(0,1,100,1); 
xsparse(:,j)=xu;
end
hyp.xu=xsparse;
cov = {'apxSparse',cov,xsparse};           % inducing points
%parpool
infv  = @(varargin) inf(varargin{:},struct('s',1.0));
infr=@infFITC;
infe = @infFITC_EP;
hyp = minimize(hyp,@gp,-100,infv,meanfunc,cov,lik,inputtrainGP,outputtrainGP);%minimise the hyperparamters
ff=size(inputtest,1);
ff=500;
disp('Predict at once')

index1=find(labelDA==-1); %output that gave a zero
index2=find(labelDA==1); % output that didnt give a zero

clement=zeros(size(inputtest,1),1);
clement(index1,:)=0; %values that the classifier predicts to give a 0
[regressoutput2,ys2v] = gp(hyp,infv,meanfunc,cov,lik, inputtrainGP, outputtrainGP, inputtest(index2,:));%Inference with GP
regressoutput2=exp(regressoutput2);
clement(index2,:)=regressoutput2;

end


function ymuv= surrogate(truefield,trueoutput,realisation)
inputtrain=truefield;
outputtrain=trueoutput;
inputtest=realisation;
n = 30; sn = 0.5;

 lik = {@likGauss};    hyp.lik = log(sn); inf = @infGaussLik;
 cov = {@covSEiso}; 
 hyp.cov = [0; 0]; 
 hyp.lik = log(0.5);
%disp('likfunc = @likGauss; sn = 0.1; hyp.lik = log(sn);')
%likfunc = @likGauss; sn = 0.1; hyp.lik = log(sn);
for j=1:size(inputtrain,2)
xu = normrnd(0,1,100,1); 
xsparse(:,j)=xu;
end
cov = {'apxSparse',cov,xsparse};           % inducing points
infv  = @(varargin) inf(varargin{:},struct('s',1.0));           % VFE, opt.s = 0
hyp = minimize(hyp,@gp,-100,infv,[],cov,lik,inputtrain,outputtrain);
[ymuv,ys2v] = gp(hyp,infv,[],cov,lik, inputtrain, outputtrain, inputtest);

end
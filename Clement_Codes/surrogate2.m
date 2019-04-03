function ymuv= surrogate2(truefield,trueoutput,realisation)
inputtrain=truefield;
outputtrain=trueoutput;
inputtest=realisation;
covfunc = @covSEiso; hyp2.cov = [0; 0]; hyp2.lik = log(0.1);
disp('likfunc = @likGauss; sn = 0.1; hyp.lik = log(sn);')
likfunc = @likGauss; sn = 0.1; hyp.lik = log(sn);
hyp2 = minimize(hyp2, @gp, -100, @infGaussLik, [], covfunc, likfunc, inputtrain, outputtrain);
[m s2] = gp(hyp2, @infGaussLik, [], covfunc, likfunc, inputtrain, outputtrain, inputtest);
ymuv=m;

end
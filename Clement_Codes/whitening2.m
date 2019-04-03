function matrixwhite= whitening2(data)
dat = bsxfun(@minus,data,mean(data));
covmat=cov(dat);
[U,S,V] = svd(covmat,'econ');
matrixwhite=U*(S^0.5)*U';
end
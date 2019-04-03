function a=whitening(x)

% De-mean
X = bsxfun(@minus,x,mean(x));
% Do the PCA
[coeff,score,latent,~,explained] = pca(X);
% Calculate eigenvalues and eigenvectors of the covariance matrix
covarianceMatrix = cov(X);
[V,D] = eig(covarianceMatrix);
a = X*coeff;
end
#include "mex.h"
#include "math.h"

// dot multiplication
double dot(int p, int q, int dim,double *psamples1, double *psamples2, double *kparam)
{
	double sum = 0;
	int i, count1, count2;
	count1 = p*dim;
	count2 = q*dim;

	for (i=0; i< dim; i++)
	{
		sum += *(psamples1 + count1 + i) * (*(psamples2 + count2 + i)) * kparam[i];
	}
	return sum;
};

// Gaussian Kernel evaluation
double kernel(double *kparam, int p,int q, int dim, double *psamples1, double *psamples2, double *square1, double *square2)
{
	double sum;
	sum = dot(p, q, dim, psamples1, psamples2, kparam);
	return kparam[dim]*exp(-(square1[p] + square2[q] - 2*sum));
};

// Interface 
void mexFunction (int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
	double *ptestsamples,*psamples,*plabels,*kparam,*alpha,*result,*square,*testsquare;
	double samplesnum,testsamplesnum;
	int i,j,dim;

	ptestsamples= mxGetPr(prhs[0]);
	psamples	= mxGetPr(prhs[1]);
	kparam		= mxGetPr(prhs[2]);
	alpha		= mxGetPr(prhs[3]);

	testsamplesnum	= mxGetN(prhs[0]);
	samplesnum  = mxGetN(prhs[1]);
	dim			= mxGetM(prhs[1]);
	
	plhs[0]		= mxCreateDoubleMatrix(testsamplesnum,1,mxREAL);
	result		= mxGetPr(plhs[0]);

	square		= mxCalloc(samplesnum,sizeof(double));
	testsquare  = mxCalloc(testsamplesnum,sizeof(double));

	
	for (i=0;i<samplesnum;i++)
		square[i] = dot(i,i,dim,psamples,psamples,kparam);
	for (i=0;i<testsamplesnum;i++)
	{
		testsquare[i] = dot(i,i,dim,ptestsamples,ptestsamples,kparam);
		result[i] = 0;
	}

	for(i=0;i<testsamplesnum;i++)
	{
		for (j=0;j<samplesnum;j++)
		{
			result[i] = result[i] + alpha[j]*kernel(kparam,i,j,dim,ptestsamples,psamples,testsquare,square);
		}
	}
	return;
}

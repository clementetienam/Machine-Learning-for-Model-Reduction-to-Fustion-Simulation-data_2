#include <mex.h>
#include <math.h>
#include <time.h>

// compute the absolute maximum
double absmax(double *gradvec, int n, int *gradindex)
{
	double maxvalue;
	int i;
	maxvalue  = fabs(gradvec[0]);
	*gradindex = 0;
	for (i=1;i<n;i++)
	{
		if (fabs(gradvec[i]) > maxvalue)
		{
			maxvalue  = fabs(gradvec[i]);
			*gradindex = i;
		}
	}
	return maxvalue;
};

// dot multiplication
double dot(int p, int q, int dim,double *psamples, double *kparam)
{
	double sum = 0;
	int i, count1, count2;
	count1 = p*dim;
	count2 = q*dim;

	for (i=0; i< dim; i++)
	{
		sum += *(psamples + count1 + i) * (*(psamples + count2 + i)) * kparam[i];
	}
	return sum;
};

// Gaussian Kernel evaluation
double kernel(double *kparam, int p,int q, int dim, double *psamples, double *square)
{
	double sum;
	sum = dot(p, q, dim, psamples, kparam);
	return kparam[dim]*exp(-(square[p] + square[q] - 2*sum));
};

// Update invK and talpha
void inverseupdate(double *invK,double *talpha, double *Ks, double *plabelss, int varnum, int m)
{
	double *beta, ksbeta, labelsbeta, gamma;
    int i,j,count=0;

	beta = mxCalloc(varnum,sizeof(double));

	// update invK
	for (i=0;i<varnum;i++)
	{
		beta[i] = 0;
		for(j=0;j<varnum;j++)
		{
			beta[i] = beta[i]+invK[i*m+j]*Ks[j];
		}
	}
	ksbeta = 0;
	for (i=0;i<varnum;i++)
	{
		ksbeta = ksbeta + beta[i]*Ks[i];
	}
	gamma = 1/(Ks[varnum] - ksbeta);

    for(i=0;i<varnum;i++)
    {
        for(j=0;j<varnum;j++)
        {
            *(invK+i*m+j) = *(invK+i*m+j) + *(beta+i) * *(beta+j)*gamma; 
         }
      }
	for(i=0;i<varnum;i++)
	{
		*(invK+i*m + varnum) = -gamma* *(beta+i);
	}
	for(i=0;i<varnum;i++)
	{
		*(invK + varnum*m+i) = -gamma* *(beta+i);
	}
	*(invK + varnum*m + varnum) = gamma;

	// update talpha
	labelsbeta = 0;
	for (i=0;i<varnum;i++)
	{
		labelsbeta = labelsbeta + beta[i]*plabelss[i];
	}
	labelsbeta = gamma*(labelsbeta - plabelss[varnum]);
	for (i=0;i<varnum;i++)
	{
		talpha[i] = talpha[i] + labelsbeta*beta[i];
	}
	talpha[varnum] = -labelsbeta;
	
	mxFree(beta);
};

void subproblem(double *psamples, double *plabels, double *kparam, double lamda, int n,
				int d, int m, double *square, double *talpha, double *gradvec, int *svi,
				double *cache, int *indexcache, int *oldindex, int it, int l)
{
	double *Kmatrix,*invK,*Ks,*plabelss,*beta,gradmax;
	int *gradindex,*Q,*pindex,i,j,p,q,flag,add,tgradindex,lenQ,ppsize;

	ppsize		= 60;
	lenQ		= n;
	Kmatrix		= mxCalloc(n*m,sizeof(double));
	Ks			= mxCalloc(m,sizeof(double));
	plabelss	= mxCalloc(m,sizeof(double));
	invK		= mxCalloc(m*m,sizeof(double));
	beta		= mxCalloc(m,sizeof(double));
    gradindex   = mxCalloc(1, sizeof(int));
	Q			= mxCalloc(n,sizeof(double));
	pindex		= mxCalloc(n,sizeof(int));


	for (i=0;i<n;i++)
	{
		Q[i] = i;
		pindex[i] = i;
	}
	// select the variable
	gradmax = absmax(gradvec, n, gradindex);
	tgradindex = pindex[*gradindex];
	*gradindex = Q[tgradindex];

	for (i=0;i<m;i++)
	{
		// compute new kernel elements
		flag = indexcache[*gradindex];
		if (flag)
		{
			for(j=0;j<n;j++)
				Kmatrix[i*n+j] = cache[(flag-1)*n + j];
		}
		else
		{
			add = fmod(it*m+i,l);
			for(j=0;j<n;j++)
			{
				Kmatrix[i*n+j] = kernel(kparam,j,*gradindex,d,psamples,square);
				cache[add*n+j] = Kmatrix[i*n+j];
			}
			Kmatrix[i*n + *gradindex] = Kmatrix[i*n + *gradindex] + lamda;
			cache[add*n + *gradindex] = Kmatrix[i*n + *gradindex];

			flag = oldindex[add];
			if(flag)
				indexcache[flag-1] = 0;
			oldindex[add] = *gradindex+1;
			indexcache[*gradindex] = add+1;
		}

		// compute invK and talpha
		if (i == 0)
		{			
			svi[i]    = *gradindex;
			plabelss[i] = plabels[*gradindex];
			invK[0] = 1/Kmatrix[i*n + (*gradindex)];
			talpha[0] = invK[0]*plabelss[i];
		}
		else
		{
			svi[i] = *gradindex;
			plabelss[i] = plabels[*gradindex];
			for (p=0; p<=i;p++)
			{
				Ks[p]= Kmatrix[i*n+svi[p]];
			}
			inverseupdate(invK,talpha,Ks,plabelss,i,m);
		}

		// update gradvec
		lenQ = lenQ-1;
		for (p=tgradindex; p<lenQ; p++)
			Q[p] = Q[p+1];
		for (p=0;p<ppsize;p++)
		{
			pindex[p] = ceil(rand()*1.0/(1+RAND_MAX)*lenQ);
		}

		for (p=0;p<ppsize;p++)
			gradvec[p] = -plabels[Q[pindex[p]]];
		for (q=0;q<=i;q++)
			for (p=0;p<ppsize;p++)
				gradvec[p] = gradvec[p] +talpha[q]*Kmatrix[q*n+Q[pindex[p]]];
		
		// select the variable
		gradmax = absmax(gradvec, ppsize, gradindex);
		tgradindex = pindex[*gradindex];
		*gradindex = Q[tgradindex];
	}
	
	// compute gradvec
	for (p=0;p<n;p++)
		gradvec[p] = -plabels[p];
	for (q=0;q<m;q++)
		for (p=0;p<n;p++)
			gradvec[p] = gradvec[p] +talpha[q]*Kmatrix[q*n+p];

	// free the pointers
	mxFree(Kmatrix);
	mxFree(Ks);
	mxFree(plabelss);
	mxFree(invK);
	mxFree(beta);
	mxFree(gradindex);
	mxFree(Q);
	mxFree(pindex);
};


void mexFunction (int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
	double *psamples,*plabels,*kparam,*tlabels,*square,*cache,*talpha,*alpha,*gradvec,*bias,*time,
		*knum,*primal,*dual;
	double lamda,csize,tol,gradmax,start,finish,it;
	int *svi,*tempindex,*indexcache,*oldindex,n,m,l,d,i;
	
	psamples	= mxGetPr(prhs[0]);
	plabels		= mxGetPr(prhs[1]);
	kparam		= mxGetPr(prhs[2]);
	lamda		= mxGetScalar(prhs[3]);
	tol			= mxGetScalar(prhs[4]);
	m			= mxGetScalar(prhs[5]);
	csize		= mxGetScalar(prhs[6]);
	d			= mxGetM(prhs[0]);
	n			= mxGetN(prhs[0]);

	plhs[0]		= mxCreateDoubleMatrix(n,1,mxREAL);
    plhs[1]		= mxCreateDoubleMatrix(n,1,mxREAL);
	plhs[2]		= mxCreateDoubleMatrix(1,1,mxREAL);
	plhs[3]		= mxCreateDoubleMatrix(1,1,mxREAL);
	plhs[4]		= mxCreateDoubleMatrix(1,1,mxREAL);

	alpha		= mxGetPr(plhs[0]);
	gradvec		= mxGetPr(plhs[1]);    
	time		= mxGetPr(plhs[2]);
	knum		= mxGetPr(plhs[3]);
	dual		= mxGetPr(plhs[4]);

	l = floor(csize/n/8.0);
	if (l > n)
		l = n;
    if (l < 2)
        l = 2;
	cache		= mxCalloc(n*l,sizeof(double));
	indexcache	= mxCalloc(n,sizeof(int));
	oldindex	= mxCalloc(l,sizeof(int));
    square		= mxCalloc(n,sizeof(double));
	tlabels		= mxCalloc(n,sizeof(double));
	talpha		= mxCalloc(m,sizeof(double));
	svi			= mxCalloc(m,sizeof(int));

	start = clock();
	// initialization
    for (i=0; i<n; i++)
	{
		square[i]	= dot(i, i, d, psamples, kparam);
		tlabels[i]	= plabels[i];
		gradvec[i]	= -plabels[i];
		alpha[i]	= 0.0;
		indexcache[i] = 0;
	}
	for (i=0; i<l; i++)
		oldindex[i] = 0;

	gradmax = tol+1.0;
	it = 0;
	while (gradmax > tol)
	{
		subproblem(psamples,tlabels,kparam,lamda,n,d,m,square,talpha,
			gradvec,svi,cache,indexcache,oldindex,it,l);

		for(i=0;i<m;i++)
			alpha[svi[i]] = alpha[svi[i]] + talpha[i];
		for(i=0;i<n;i++)
			tlabels[i]=-gradvec[i];
		
		gradmax = absmax(gradvec,n,tempindex);
		it = it+1;
	}
	*knum = m*n*it;
	*dual = 0.0;
	for(i=0;i<n;i++)
		*dual = *dual + 0.5*alpha[i]*(gradvec[i]-plabels[i]);
	finish = clock();
	*time = (finish - start)/CLK_TCK;
	mxFree(square);
	mxFree(tlabels);
	mxFree(svi);
	printf("Number of kernel evaluation is %.0f.\n", knum[0]);
	printf("Objectve function is %f\n",dual[0]);
	printf("Elapsed time is %f seconds.\n",time[0]);
	return;
}
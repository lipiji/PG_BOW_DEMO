/******************************************************************************
 *
 *
 *
 * Compute matrix of histogram intersection kernel values between two sets of vectors.
 * If one argument is a matrix of size m x o (where m is the number of vectors and
 * o is the dimensionality of each vector) and the second one is of size n x o,
 * the output is a matrix of size m x n.
 *
 * From MATLAB, compile this mex function with the following command:
 * mex hist_isect_c.c -lm
 *
 * Adapted from the svm_v0.55 toolbox: http://theoval.sys.uea.ac.uk/~gcc/svm/toolbox
 *
 *
 *
 ******************************************************************************/



#include <math.h>

#include "mex.h"



#define min(a, b) (((a)<(b))?(a):(b))



void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])

{
    
    double *x1, *x2, *y;
    
    
    
    int row, i, j, k, m, n, o, N;
    
    
    
    /* check number of input and output arguments */
    
    
    
    if (nrhs != 2)
        
    {
        
        mexErrMsgTxt("Wrong number of input arguments.");
        
    }
    
    else if (nlhs > 1)
        
    {
        
        mexErrMsgTxt("Too many output arguments.");
        
    }
    
    
    
    /* get input arguments */
    
    
    
    if (!mxIsDouble(prhs[0]) || mxIsComplex(prhs[0]))
        
    {
        
        mexErrMsgTxt("x1 must be a double matrix.");
        
    }
    
    
    
    m  = mxGetM(prhs[0]);
    
    x1 = mxGetPr(prhs[0]);
    
    
    
    if (!mxIsDouble(prhs[1]) || mxIsComplex(prhs[1]))
        
    {
        
        mexErrMsgTxt("x2 must be a double matrix.");
        
    }
    
    
    
    n  = mxGetM(prhs[1]);
    
    o  = mxGetN(prhs[1]);
    
    x2 = mxGetPr(prhs[1]);
    
    
    
    /* allocate and initialise output matrix */
    
    
    
    plhs[0] = mxCreateDoubleMatrix(m, n, mxREAL);
    
    
    
    y = mxGetPr(plhs[0]);
    
    
    
    /* compute kernel matrix */
    
    
    
    if (n > m)
        
    {
        
        for (i = 0; i < m; i++)
            
        {
            
            for (k = 0; k < o; k++)
                
            {
                
                if (x1[i+k*m]==0) continue;
                
                
                
                for (j = 0; j < n; j++)
                    
                {
                    
                    y[i+j*m] += min(x1[i+k*m], x2[j+k*n]);
                    
                }
                
            }
            
        }
        
    }
    
    else
        
    {
        
        for (j = 0; j < n; j++)
            
        {
            
            for (k = 0; k < o; k++)
                
            {
                
                if (x2[j+k*n]==0) continue;
                
                
                
                for (i = 0; i < m; i++)
                    
                {
                    
                    y[i+j*m] += min(x1[i+k*m], x2[j+k*n]);
                    
                }
                
            }
            
        }
        
    }
    
    
    
}






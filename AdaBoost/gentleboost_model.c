
/*

  Gentle AdaBoost Classifier with two different weak-learners : Decision Stump and Perceptron.
  Multi-class problem is performed with the one-against-all strategy.

  Usage
  ------

  model = gentleboost_model(X , y , [T] , [options]);

  
  Inputs
  -------

  X                                     Features matrix (d x N) 
  y                                     Labels (1 x N). If y represent binary labels vector then y_i={-1,1}, i=1,...,N
  T                                     Number of weak learners (default T = 100)
  options
              weaklearner               Choice of the weak learner used in the training phase
			                            weaklearner = 0 <=> minimizing the weighted error : sum(w * |z - h(x;(th,a,b))|^2) / sum(w), where h(x;(th,a,b)) = (a*(x>th) + b) in R
			                            weaklearner = 1 <=> minimizing the weighted error : sum(w * |z - h(x;(a,b))|^2), where h(x;(a,b)) = sigmoid(x ; a,b) in R

			  epsi                      Epsilon constant in the sigmoid function used in the perceptron (default epsi = 1)
              lambda                    Regularization parameter for the perceptron's weights update (default lambda = 1e-3)
			  max_ite                   Maximum number of iterations of the perceptron algorithm (default max_ite = 100)

  Outputs
  -------
  
  model                                 Structure of model ouput
  
	          featureIdx                Features index of the T best weaklearners (T x m) where m is the number of class. 
			                            For binary classification m is force to 1.
			  th                        Optimal Threshold parameters (1 x T)
			  a                         Affine parameter(1 x T)
			  b                         Bias parameter (1 x T)




  To compile
  ----------


  mex  -output gentleboost_model.dll gentleboost_model.c

  mex  -f mexopts_intel10.bat -output gentleboost_model.dll gentleboost_model.c



  Example 1
  ---------


  load iris %wine
  %load heart
  %load wbc
  %y(y==0)                 = -1;
  T                       = 12;

  options.weaklearner     = 0;
  options.epsi            = 0.5;
  options.lambda          = 1e-3;
  options.max_ite         = 3000;
  
  model                   = gentleboost_model(X , y , T , options);
  [yest , fx]             = gentleboost_predict(X , model , options);
  Perf                    = sum(y == yest)/length(y)

  plot(fx')


 Author : Sébastien PARIS : sebastien.paris@lsis.org
 -------  Date : 01/27/2009

 Ref    : Friedman, J. H., Hastie, T. and Tibshirani, R. "Additive Logistic Regression: a Statistical View of Boosting." (Aug. 1998) 
 ------- 



*/


#include <time.h>
#include <math.h>
#include <mex.h>


#define huge 1e300


#define znew   (z = 36969*(z&65535) + (z>>16) )
#define wnew   (w = 18000*(w&65535) + (w>>16) )
#define MWC    ((znew<<16) + wnew )
#define SHR3   ( jsr ^= (jsr<<17), jsr ^= (jsr>>13), jsr ^= (jsr<<5) )



#define randint SHR3
#define rand() (0.5 + (signed)randint*2.328306e-10)
#define sign(a) ((a) >= (0) ? (1.0) : (-1.0))


#ifdef __x86_64__
    typedef int UL;
#else
    typedef unsigned long UL;
#endif

static UL jsrseed = 31340134 , jsr;




struct opts
{

  int    weaklearner;

  double epsi;

  double lambda;

  int    max_ite;

};


struct weak_learner
{

	double *featureIdx;

	double *th;

	double *a;

	double *b;

};



/* Function prototypes */

void randini(void);

void qs( double * , int , int  ); 

void qsindex( double * , int * , int , int  );

void transpose(double *, double * , int , int);


void  gentelboost_decision_stump(double * , double * , int , struct opts , double * , int , 
								 double *, double *, double * , double *,
		                         int  , int );



void  gentelboost_perceptron(double * , double * , int , struct opts , double * , int ,
							 double *, double *, double * ,
		                     int  , int );


/*-------------------------------------------------------------------------------------------------------------- */



void mexFunction( int nlhs, mxArray *plhs[] , int nrhs, const mxArray *prhs[] )

{
    
    double *X , *y;
    
    int d , N , T=100;
    
    mxArray *mxtemp;
    
    struct weak_learner model;
    
    struct opts options = {0 , 1 , 1e-3 , 100};
    
    const char *fieldnames_model[4] = {"featureIdx" , "th" , "a" , "b"};
    
    double *tmp , *ysorted , *labels;
    
    int i , tempint , m=0 , currentlabel;
    
    
    if ((nrhs < 2) )

	{

		mexErrMsgTxt("At least two arguments are requiered, i.e. model = gentleboost_model(X , y , [T] , [options]);");

	}


    
    /* Input 1  */
    
    if( (mxGetNumberOfDimensions(prhs[0]) !=2) || !mxIsDouble(prhs[0]) )
    {
        
        mexErrMsgTxt("X must be (d x N) in double format");
        
    }
    
    
    X           = mxGetPr(prhs[0]);
    
    d           = mxGetM(prhs[0]);
    
    N           = mxGetN(prhs[0]);
    
    
    
    /* Input 2  */
    
    if ((nrhs > 1) && !mxIsEmpty(prhs[1]) )
        
    {
        
        y        = mxGetPr(prhs[1]);
        
    }
    
    ysorted      = (double *)mxMalloc(N*sizeof(double));
    
    for ( i = 0 ; i < N ; i++ )
    {
        
        ysorted[i] = y[i];
        
    }
    
    
    qs( ysorted , 0 , N - 1 );
    
    
    labels       = (double *)mxMalloc(sizeof(double));
    
    labels[m]    = ysorted[0];
    
    currentlabel = labels[0];
    
    for (i = 0 ; i < N ; i++)
    {
        if (currentlabel != ysorted[i])
        {
            labels       = (double *)mxRealloc(labels , (m+2)*sizeof(double));
            
            labels[++m]  = ysorted[i];
            
            currentlabel = ysorted[i];
            
        }
    }
    
    m++;
    
    if( m == 2) /* Binary case */
    {
        
        m         = 1;

        labels[0] = 1.0; /*Force positive label in the first position*/
           
    }
    
    /* Input 3  */
    
    
    if ((nrhs > 2) && !mxIsEmpty(prhs[2]) )
        
    {
        
        T          = (int) mxGetScalar(prhs[2]);
        
    }
    
    
    
    /* Input 4  */
    
    
    if ((nrhs > 3) && !mxIsEmpty(prhs[3]) )
        
    {
        
        mxtemp                            = mxGetField(prhs[3] , 0 , "weaklearner");
        
        if(mxtemp != NULL)
        {
            
            tmp                           = mxGetPr(mxtemp);
            
            tempint                       = (int) tmp[0];
            
            if((tempint < 0) || (tempint > 1))
            {
                
                mexErrMsgTxt("weaklearner = {0,1}, force to 0");
                
                options.weaklearner        = 0;
                
            }
            else
            {
                
                options.weaklearner        = tempint;
                
            }
            
        }


        mxtemp                            = mxGetField(prhs[3] , 0 , "epsi");
        
        if(mxtemp != NULL)
        {
            
            tmp                           = mxGetPr(mxtemp);
            
            options.epsi                  = tmp[0];
            
            
        }
        
        
        mxtemp                            = mxGetField(prhs[3] , 0 , "lambda");
        
        if(mxtemp != NULL)
        {
            
            tmp                           = mxGetPr(mxtemp);
            
            options.lambda                = tmp[0];
            
            
        }
        
        
        mxtemp                            = mxGetField(prhs[3] , 0 , "max_ite");
        
        if(mxtemp != NULL)
        {
            
            tmp                           = mxGetPr(mxtemp);
            
            tempint                       = (int) tmp[0];
            
            if(tempint < 1)
            {
                
                mexErrMsgTxt("max_ite > 0, force to default value");
                
                options.max_ite           = 10;
                
            }
            
            else
                
            {
                options.max_ite           =  tempint;
                
            }
            
        }
    }
    
        /*------------------------ Main Call ----------------------------*/
    
    if(options.weaklearner == 0)
    {
        
        
        plhs[0]              =  mxCreateStructMatrix(1 , 1 , 4 , fieldnames_model);
        
        
        for(i = 0 ; i < 4 ; i++)
        {
            
            
            mxSetFieldByNumber(plhs[0] ,0 , i , mxCreateNumericMatrix(T , m , mxDOUBLE_CLASS,mxREAL));
            
        }
        
        model.featureIdx = mxGetPr( mxGetField( plhs[0], 0, fieldnames_model[0] ) );
        model.th         = mxGetPr( mxGetField( plhs[0], 0, fieldnames_model[1] ) );
        model.a          = mxGetPr( mxGetField( plhs[0], 0, fieldnames_model[2] ) );
        model.b          = mxGetPr( mxGetField( plhs[0], 0, fieldnames_model[3] ) );
        
        
        
        gentelboost_decision_stump(X , y , T , options , labels , m ,
                                   model.featureIdx , model.th , model.a , model.b,
                                   d , N);
        
    }
    
    if(options.weaklearner == 1)
    {
        
        
        plhs[0]              =  mxCreateStructMatrix(1 , 1 , 4 , fieldnames_model);

       
        for(i = 0 ; i < 4 ; i++)
        {
            
            
            mxSetFieldByNumber(plhs[0] ,0 , i , mxCreateNumericMatrix(T , m , mxDOUBLE_CLASS,mxREAL));
            
        }
        
        model.featureIdx = mxGetPr( mxGetField( plhs[0], 0, fieldnames_model[0] ) );
        model.th         = mxGetPr( mxGetField( plhs[0], 0, fieldnames_model[1] ) );
        model.a          = mxGetPr( mxGetField( plhs[0], 0, fieldnames_model[2] ) );
        model.b          = mxGetPr( mxGetField( plhs[0], 0, fieldnames_model[3] ) );
        
        
        randini();
        
        gentelboost_perceptron(X , y , T , options , labels , m,
                               model.featureIdx , model.a , model.b,
                               d , N);
    }
}


/*----------------------------------------------------------------------------------------------------------------------------------------- */

void  gentelboost_decision_stump(double *X , double *y , int T , struct opts options , double *labels , int m,
                                 double *featuresIdx, double *th , double *a, double *b,
                                 int d , int N)
{
    
    
    double cteN =1.0/(double)N;
    
    
    int i , j , t , c  , cT;
    
    int indN , Nd = N*d , ind, N1 = N - 1 , featuresIdx_opt , currentlabel , templabel;
    
    
    double *w, *Xt, *Xtsorted , *xtemp , *ytemp , *wtemp  ;
    
    int *Ytsorted , *index, *idX;
    
    double atemp , btemp  , sumSw , Eyw , fm  , sumwyy , error , errormin, th_opt , a_opt , b_opt , label ;
    
	double  temp , Sw , Syw;
    
    
    Xt           = (double *)mxMalloc(Nd*sizeof(double));
    
    Xtsorted     = (double *)mxMalloc(Nd*sizeof(double));



    ytemp        = (double *)mxMalloc(N*sizeof(double));

    wtemp        = (double *)mxMalloc(N*sizeof(double));


    
    Ytsorted     = (int *)mxMalloc(Nd*sizeof(int));
    
    idX          = (int *)mxMalloc(Nd*sizeof(int));
    
    
    w            = (double *)mxMalloc(N*sizeof(double));
    
    index        = (int *)mxMalloc(N*sizeof(int));
    
    xtemp        = (double *)mxMalloc(N*sizeof(double));
    
         
    
    /* Transpose data to speed up computation */
    
    transpose(X , Xt , d , N);
    
    
    /* Sorting data to speed up computation */
    
    indN   = 0;
    
    for(j = 0 ; j < d ; j++)
    {
        
        for(i = 0 ; i < N ; i++)
        {
            
            index[i] = i;
            
            xtemp[i] = Xt[i + indN];
        }
        
        
        qsindex(xtemp , index , 0 , N1);
        
        
        for(i = 0 ; i < N ; i++)
        {
            ind                = index[i];
            
            Xtsorted[i + indN] = xtemp[i];
            
            Ytsorted[i + indN] = (int)y[ind];
            
            idX[i + indN]      = ind;
            
            
        }
        
        indN   += N;
        
    }
    
    cT   = 0;
    
    
    for (c = 0 ; c < m ; c++)
    {
        
		label        = labels[c];
        
        currentlabel = (int)label;


        for(i = 0 ; i < N ; i++)
        {
            
            w[i]  = cteN;

        }

           
        for(t = 0 + cT ; t < T + cT; t++)
            
        {

            
            errormin         = huge;
            
            indN             = 0;
            
            for(j = 0 ; j < d  ; j++)
                
            {
               		
				Eyw              = 0.0;
				
				sumwyy           = 0.0;

                for(i = 0 ; i < N ; i++)
                    
                {
                    ind       = i + indN;
                    
                    xtemp[i]  = Xtsorted[ind];
                    
                    templabel = Ytsorted[ind];
                    
                    if(templabel == currentlabel)
                    {
                        ytemp[i]  = 1.0;
                        
                    }
                    else
                    {
                        
                        ytemp[i]  = -1.0;
                        
                    }
                    
                    wtemp[i]   = w[idX[ind]];

					temp       = ytemp[i]*wtemp[i];

					Eyw       += temp;

					sumwyy    += ytemp[i]*temp;

                    
                }
                
                Sw          = 0.0;
				
				Syw         = 0.0;
                
                for(i = 0 ; i < N ; i++)
                    
                {
                    
                    ind         = i + indN;
									
					Sw         += wtemp[i];
					
					Syw        += ytemp[i]*wtemp[i];
									
					btemp       = Syw/Sw;
					
					
					if(Sw != 1.0)
					{
						
						atemp  = (Eyw - Syw)/(1.0 - Sw) - btemp;
						
					}
					else
					{
						atemp  = (Eyw - Syw) - btemp;
						
					}
					
					
					error   = sumwyy - 2.0*atemp*(Eyw - Syw) - 2.0*btemp*Eyw + (atemp*atemp + 2.0*atemp*btemp)*(1.0 - Sw) + btemp*btemp;
					
					if(error < errormin)					
					{
						
						errormin        = error;
						
						featuresIdx_opt = j;
						
						
						if(i < N1)
						{
							
							th_opt     = (xtemp[i] + xtemp[i + 1])/2;
							
						}
						else
						{
							
							th_opt     = xtemp[i];
							
						}
						
						a_opt          = atemp;
						
						b_opt          = btemp;					
					}
					
                    
                }
                
                indN   += N;
            }
            
			/* Best parameters for weak-learner t */
            
            featuresIdx[t]   = (double) (featuresIdx_opt + 1);
            
            th[t]            = th_opt;
            
            a[t]             = a_opt;
            
            b[t]             = b_opt;
            
            
			/* Weigth's update */
            
            ind              = featuresIdx_opt*N;
            
            sumSw            = 0.0;
            
            for (i = 0 ; i < N ; i++)
            {
                
                fm       = a_opt*(Xt[i + ind] > th_opt) + b_opt;

				if(y[i] == label)
				{
					
					
					w[i]    *= exp(-fm);
					
				}
				else
				{
					
					w[i]    *= exp(fm);
					
					
				}
                
                sumSw   += w[i];
            }
            
            
            sumSw            = 1.0/sumSw;
            
            for (i = 0 ; i < N ; i++)
            {
                
                w[i]         *= sumSw;
            }
            
        }
        
        cT    += T;
     
    }
    
    
    mxFree(w);
    
    mxFree(Xt);
    
    mxFree(Xtsorted);
    
    mxFree(Ytsorted);
    
    mxFree(index);
    
    mxFree(xtemp);
    
    mxFree(idX);

    mxFree(ytemp);

	mxFree(wtemp);
    
    
}


/*----------------------------------------------------------------------------------------------------------------------------------------- */

void  gentelboost_perceptron(double *X , double *y , int T , struct opts options , double *labels , int m ,
                             double *featuresIdx, double *a , double *b,
                             int d , int N)
{
    
    
    double epsi = options.epsi , lambda = options.lambda , error , errormin , cteN =1.0/(double)N;
    
    int t , j , i , k , c , cT ;
    
    int featuresIdx_opt;
    
    int max_ite = options.max_ite, Nd = N*d , indN , index;
    
    double *Xt , *w;
    
    double *ytemp;
    
    double atemp , btemp , xi  , temp , fx , tempyifx , sum , fm , currentlabel;
    
    double a_opt, b_opt;
    
    
    Xt           = (double *)mxMalloc(Nd*sizeof(double));
    
    w            = (double *)mxMalloc(N*sizeof(double));
    
    ytemp        = (double *)mxMalloc(N*sizeof(double));
    
    
    
    transpose(X , Xt , d , N);
    
    
    
    
    
    cT  = 0;
    
    for (c = 0 ; c < m ; c++)
    {
        
        currentlabel = labels[c];
        
        for(i = 0 ; i < N ; i++)
        {
            
            w[i]    = cteN;
                        
            if(y[i] == currentlabel)
            {
               
				ytemp[i]  = 1.0;
                
            }
            else
            {
                
                ytemp[i]  = -1.0;
                
            }
        }
        
        
        for(t = 0 + cT ; t < T + cT ; t++)
            
        {
            
            errormin = huge;
            
            indN     = 0;
            
            
            for(j = 0 ; j < d  ; j++)
                
            {
                /* Random initialisation of weights */
                
                index  = floor(N*rand());
                
                atemp  = Xt[index + indN];
                
                index  = floor(N*rand());
                
                btemp  = Xt[index + indN];
                
                
                /* Weight's optimization  */
                
                for(k = 0 ; k < max_ite ; k++)
                {             
					
					for(i = 0 ; i < N ; i++)
                        
                    {
                        
                        xi         = Xt[i + indN];
                        
                        fx         = (2.0/( 1.0 + exp(-2.0*epsi*(atemp*xi + btemp)) )) - 1.0; /* sigmoid in [-1 , 1] */


                        temp       = lambda*(ytemp[i] - fx)*epsi*(1.0 - fx*fx);	/* d(sig(x;epsi))/dx = epsi*(1 - fx²) */
                        
                        
                        atemp     += (temp*xi);
                        
                        btemp     += temp;
                        
                    }
                    
                }
                
                /* Weigthed error */
                
                
                error         = 0.0;
                
                for(i = 0 ; i < N ; i++)
                    
                {
                    
                    fx        = (2.0/(1.0 + exp(-2.0*epsi*(atemp*Xt[i + indN] + btemp)))) - 1.0;
                    
                    tempyifx  = (ytemp[i] - fx);
                    
                    error    += w[i]*tempyifx*tempyifx;
                    
                }
                
                if(error < errormin)
                    
                {
                    
                    errormin        = error;
                    
                    featuresIdx_opt = j;
                    
                    
                    a_opt           = atemp;
                    
                    b_opt           = btemp;
                    
                }
                
                indN    += N;
                
            }
            
            
            featuresIdx[t]   = (double) (featuresIdx_opt + 1);
            
            a[t]             = a_opt;
            
            b[t]             = b_opt;
            
            
            
            index            = featuresIdx_opt*N;
            
            sum              = 0.0;
            
            for (i = 0 ; i < N ; i++)
            {
                
			    fm           = (2.0/(1.0 + exp(-2.0*epsi*(a_opt*Xt[i + index] + b_opt)))) - 1.0;

                w[i]        *= exp(-ytemp[i]*fm);
                
                sum         += w[i];
            }
            
            
            sum              = 1.0/sum;
            
            for (i = 0 ; i < N ; i++)
            {
                
                w[i]         *= sum;
            }
            
        }
        
        cT    += T;      

    }
    
    mxFree(Xt);
    
    mxFree(w);
    
    mxFree(ytemp);
    
}



/*----------------------------------------------------------------------------------------------------------------------------------------- */


void qs(double  *a , int lo, int hi)
{
/*  lo is the lower index, hi is the upper index
    of the region of array a that is to be sorted
*/
    int i=lo, j=hi;
    double x=a[(lo+hi)/2] , h;

    /*  partition  */
    do
    {    
        while (a[i]<x) i++; 
        while (a[j]>x) j--;
        if (i<=j)
        {
            h        = a[i]; 
			a[i]     = a[j]; 
			a[j]     = h;
            i++; 
			j--;
        }
    }
	while (i<=j);

    /*  recursion  */

    if (lo<j) qs(a , lo , j);
    if (i<hi) qs(a , i , hi);
}

/*----------------------------------------------------------------------------------------------------------------------------------------- */

void qsindex (double  *a, int *index , int lo, int hi)
{
/*  lo is the lower index, hi is the upper index
    of the region of array a that is to be sorted
*/
    int i=lo, j=hi , ind;
    double x=a[(lo+hi)/2] , h;

    /* partition */
    do
    {    
        while (a[i]<x) i++; 
        while (a[j]>x) j--;
        if (i<=j)
        {
            h        = a[i]; 
			a[i]     = a[j]; 
			a[j]     = h;
			ind      = index[i];
			index[i] = index[j];
			index[j] = ind;
            i++; 
			j--;
        }
    }
	while (i<=j);

    /*  recursion */
    if (lo<j) qsindex(a , index , lo , j);
    if (i<hi) qsindex(a , index , i , hi);
}


/*----------------------------------------------------------------------------------------------------------------------------------------- */

void transpose(double *A, double *B , int m , int n)

{
    
    int i , j , jm , in;
    
    jm     = 0;
    
    for (j = 0 ; j<n ; j++)
        
    {
        in  = 0;
        
        for(i = 0 ; i<m ; i++)
        {
            
            B[j + in] = A[i + jm];
            
            in       += n;
            
        }
        
        jm  += m;
    }
}


/*----------------------------------------------------------------------------------------------------------------------------------------- */


void randini(void)

{
    
     /* SHR3 Seed initialization */
    
    jsrseed  = (UL) time( NULL );
    
    jsr     ^= jsrseed;
    
    
}

/*----------------------------------------------------------------------------------------------------------------------------------------- */

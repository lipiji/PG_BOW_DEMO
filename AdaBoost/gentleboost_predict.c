
/*

  Predict data label with a Strong Classifier trained with Gentle AdoBoost classifier

  Usage
  ------

  [yest , fx] = gentleboost_predict(X , model , [options]);

  
  Inputs
  -------

  X                                    Features matrix (d x N)
  model                                Trained model structure
  options
          weaklearner                  Choice of the weak learner used in the training phase
	                                   weaklearner = 0 <=> minimizing the weighted error : sum(w * |z - h(x;(th,a,b))|^2) / sum(w), where h(x;(th,a,b)) = (a*(x>th) + b) in R
			                           weaklearner = 1 <=> minimizing the weighted error : sum(w * |z - h(x;(a,b))|^2), where h(x;(th,a)) = sigmoid(x ; a,b) in R
		  epsi                         Epsilon constant in the sigmoid function used in the perceptron (default epsi = 1)
  

  Outputs
  -------
  
  yest                                 Estimated labels (1 x N)
  fx                                   Additive models (1 x N)




  To compile
  ----------


  mex  -output gentleboost_predict.dll gentleboost_predict.c

  mex  -f mexopts_intel10.bat -output gentleboost_predict.dll gentleboost_predict.c



  Example 1
  ---------

  load heart
  y(y==0)                 = -1;

  T                       = 100;
  options.weaklearner     = 0;
  model                   = gentleboost_model(X , y , T , options);
  [yest , fx]             = gentleboost_predict(X , model , options);



 Author : Sébastien PARIS : sebastien.paris@lsis.org
 -------  Date : 01/27/2009

 Reference ""


*/


#include <time.h>
#include <math.h>
#include <mex.h>


#define sign(a) ((a) >= (0) ? (1.0) : (-1.0))

#define tiny -1e300


struct opts
{

  int   weaklearner;

  double epsi;


};



struct weak_learner
{

	double *featureIdx;

	double *th;

	double *a;

	double *b;

};



/* Function prototypes */



void  gentleboost_predict(double * , struct weak_learner  , struct opts  ,
		                  double * , double *,
		                   int  , int, int , int );



/*-------------------------------------------------------------------------------------------------------------- */



void mexFunction( int nlhs, mxArray *plhs[] , int nrhs, const mxArray *prhs[] )

{
	
	
    double *X ;
	
	struct weak_learner model;
	
	struct opts options = {0 , 1};
	
	double *yest , *fx;
	
	int d , N , T=100 , m=1;
	
	mxArray *mxtemp;
	
	double *tmp;
	
	int tempint;
	
	
	
    /* Input 1  */
	
	if( (mxGetNumberOfDimensions(prhs[0]) !=2) || !mxIsDouble(prhs[0]) )
	{
		
		mexErrMsgTxt("X must be (d x N) in double format");
		
	}
	
	
	X           = mxGetPr(prhs[0]);
	
	d           = mxGetM(prhs[0]);
	
	N           = mxGetN(prhs[0]);
	
	
	
	/* Input 2  */
	
	if ((nrhs > 1) && !mxIsEmpty(prhs[1]) && mxIsStruct(prhs[1]) )
		
	{
		
		
		mxtemp                    = mxGetField( prhs[1], 0, "featureIdx" );
		
		if(mxtemp != NULL)
		{
			
			T                     = mxGetM(mxtemp);

			m                     = mxGetN(mxtemp);
			
			model.featureIdx = mxGetPr(mxtemp);
			
		}
		
		
		mxtemp                    = mxGetField( prhs[1], 0, "th" );
		
		if(mxtemp != NULL)
		{
			
			model.th              = mxGetPr(mxtemp);
			  
		}
		
		
		mxtemp                    = mxGetField( prhs[1] , 0, "a" );
		
		if(mxtemp != NULL)
		{
			
			model.a               = mxGetPr(mxtemp);
			
		}
		
		mxtemp                    = mxGetField( prhs[1] , 0, "b" );
		
		if(mxtemp != NULL)
		{
			
			model.b               = mxGetPr(mxtemp);
			
		}
		
	}
	
	else
	{
		
		mexErrMsgTxt("model must be a structure");
		
	}
	
	
	/* Input 3  */
	
	
	
	if ((nrhs > 2) && !mxIsEmpty(prhs[2]) )
		
	{
		
		if(!mxIsStruct(prhs[2]) )
		{
			
			mexErrMsgTxt("options must be a structure");
			
		}
		
		mxtemp                            = mxGetField(prhs[2] , 0 , "weaklearner");
		
		if(mxtemp != NULL)
		{
			
			tmp                           = mxGetPr(mxtemp);
			
			tempint                       = (int) tmp[0];
			
			if(tempint < 0)
			{
				
				mexErrMsgTxt("weaklearner ={0,1}, force default to 0");	
				
				options.weaklearner       = 0;
				
			}
			else
			{
				
				options.weaklearner       = tempint;
				
			}
			
		}


		mxtemp                            = mxGetField(prhs[2] , 0 , "epsi");
		
		if(mxtemp != NULL)
		{
			
			
			tmp                           = mxGetPr(mxtemp);
			
			options.epsi                  = tmp[0];	
			
		}
		
	}
	
	
	/* Output 1  */
	
	
	
	plhs[0]             =  mxCreateNumericMatrix(1 , N, mxDOUBLE_CLASS, mxREAL);
	
	yest                =  mxGetPr(plhs[0]);
	
	/* Output 1  */
	
	
	plhs[1]             =  mxCreateNumericMatrix(m , N, mxDOUBLE_CLASS, mxREAL);
	
	fx                  =  mxGetPr(plhs[1]);

	

		/*------------------------ Main Call ----------------------------*/

	
	gentleboost_predict(X , model , options ,
		                yest , fx ,
		                d , N , T , m);
	
	
	
}


/*----------------------------------------------------------------------------------------------------------------------------------------- */

void  gentleboost_predict(double *X , struct weak_learner model , struct opts options ,
		                  double *yest, double *fx,
		                  int d , int N , int T , int m)


{


	int t , n , c , cT , nm;

	int indd , ind_maxi;

	double *featureIdx , *th , *a , *b;

	int weaklearner = options.weaklearner;

	double epsi = options.epsi;

	double sum , maxi;


	if(weaklearner == 0) /* Decision Stump */
	{
		
		featureIdx = model.featureIdx;
		th         = model.th;	
		a          = model.a;
		b          = model.b;
		
		if(m == 1)
		{
			
			indd      = 0;
			
			for(n = 0 ; n < N ; n++)
			{
				
				sum   = 0.0;
				
				for(t = 0 ; t < T ; t++)
				{
					
					sum    += (a[t]*( X[(int)featureIdx[t] - 1 + indd]>th[t] ) + b[t]);
					
				}
				
				fx[n]       = sum;
				
				yest[n]     = sign(sum);
				
				indd       += d;
				
			}	
		}
		
		else
		{
			
			indd      = 0;
			
			nm        = 0;
			
			for(n = 0 ; n < N ; n++)
			{
				cT        = 0;	
				
				maxi      = tiny;
				
				ind_maxi  = 0;
				
				for(c = 0 ; c < m ; c++)
				{
					sum   = 0.0;
					
					for(t = 0 + cT ; t < T + cT ; t++)
					{
						
						sum    += (a[t]*( X[(int)featureIdx[t] - 1 + indd] > th[t] ) + b[t]);
						
					}
					
					fx[c + nm] = sum;
					
					if(sum > maxi)
					{
						
						maxi     = sum;
						
						ind_maxi = c;
						
					}		
					
					cT         += T;
					
				}
				
				yest[n]       = ind_maxi;
				
				nm           += m;
				
				indd         += d;
				
				
			}			
			
		}
		
	}
	
	if(weaklearner == 1) /* Perceptron */
	{
		
		
		featureIdx = model.featureIdx;		
		a          = model.a;
		b          = model.b;
		
		if(m == 1)
		{
			
			indd       = 0;
			
			for(n = 0 ; n < N ; n++)
			{
				
				sum   = 0.0;
				
				for(t = 0 ; t < T ; t++)
				{
					
					sum    += ((2.0/(1.0 + exp(-2.0*epsi*(a[t]* X[(int)featureIdx[t] - 1 + indd] + b[t])))) - 1.0);
					
				}
				
				fx[n]       = sum;
				
				yest[n]     = sign(sum);
				
				indd       += d;
				
			}
			
		}
		else
		{
			
			indd       = 0;
			
			nm         = 0;
			
			for(n = 0 ; n < N ; n++)
			{
				
				cT        = 0;	
				
				maxi      = tiny;
				
				ind_maxi  = 0;
				
				for(c = 0 ; c < m ; c++)
				{
					
					sum        = 0.0;
					
					for(t = 0 + cT ; t < T + cT ; t++)
					{
						
						sum    += ((2.0/(1.0 + exp(-2.0*epsi*(a[t]*X[(int)featureIdx[t] - 1 + indd] + b[t] )))) - 1.0);
						
					}
					
					fx[c + nm] = sum;
					
					if(sum > maxi)
					{
						
						maxi     = sum;
						
						ind_maxi = c;
						
					}
					
					cT         += T;
					
				}
				
				yest[n]       = ind_maxi;
				
				nm           += m;
				
				indd         += d;
					
			}
			
		}	
	}
}

/*----------------------------------------------------------------------------------------------------------------------------------------- */

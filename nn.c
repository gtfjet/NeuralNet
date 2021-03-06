/*!
	G.N.N.N.U.
	General Neural Net with Newton Updates
*/

#include <stdio.h>
#include <math.h>
#define RAND_MAX 32767

/* Global Variables */
int net[] = {2,3,2};    //structure of neural net
int nNeurons, nWeights, nBiases, nLayers, nInputs, nOutputs, nTuners;

/* Sigmoid Function */
double sig(double t) {
	return 1.0/(1.0+exp(-t));
}

/* Get Size of Network */
void get_size() {
	nNeurons = 0;
	nWeights = 0;
	nLayers  = sizeof(net)/sizeof(int);
	nInputs  = net[0];
	nOutputs = net[nLayers-1];
	for(int i=0; i<nLayers; i++) {
		nNeurons += net[i];
		if(i>0) {
			nWeights += net[i]*net[i-1];
		}
	}
	nBiases = nNeurons-nInputs;
	nTuners = nWeights+nBiases;
	printf("%i,%i,%i,%i,%i\n",nLayers,nBiases,nNeurons,nWeights,nWeights+nBiases); 
}
	
/* Calculate Output using the Net */
void run(double* x, const double* w, const double* b) {
	double total;
	int p = 0;		//step thru w
	int q = 0;		//step thru x
	int r = 0;		//step thru b
	for(int i=1; i<nLayers; i++) {
		for(int j=0; j<net[i]; j++) {
			total = b[r];  		//start with bias
			r++;
			for(int k=0; k<net[i-1]; k++) {
				total += w[p]*x[q+k];
				p++;
			}
			x[q+net[i-1]+j] = tanh(total);
		}
		q += net[i-1];
	}
}

/* Calculate Square Error */ 
double get_error(const double* x, const double* y) {
	double err = 0;
	for(int i=0; i<nOutputs; i++) {
		err += (y[i]-x[nNeurons-nOutputs+i])*(y[i]-x[nNeurons-nOutputs+i])/2;
	}
	//printf("err=%f\n",err);
	return err;
}

/* Get Partials */
void get_partials(const double* x, const double *y, const double* w, double* dedw, double* dedb, double* dat) {
	int p = nWeights-1; 	//step backward thru w
	int q = nNeurons-1;		//step backward thru x
	int r;
	for(int i=0; i<nNeurons; i++) { dat[i]=0; } 
	for(int i=(nLayers-1); i>0; i--) {
		for(int j=0; j<net[i]; j++) {
			if(i==(nLayers-1)) {		//(q-j) is current node
				dat[q-j] = (x[q-j]-y[nOutputs-1-j])*(1-x[q-j]*x[q-j]); 
			} else {	
				dat[q-j] *= (1-x[q-j]*x[q-j]);  	
			}
			dedb[q-j-nInputs] = dat[q-j];
			for(int k=0; k<net[i-1]; k++) {
				r 		= q-k-net[i]; 		//other node
				dat[r] += dat[q-j]*w[p];
				dedw[p] = dat[q-j]*x[r];
				p--;
			}
		}
		q -= net[i];
	}
}

/* Solve Linear System (http://web.mit.edu/10.001/Web/Course_Notes/Gauss_Pivoting.c) */
void gauss(double **A, double *b, double *x, int n) {
	int i, j, k, m, rowx;
	double xfac, temp, amax;
	rowx = 0;					//keep count of the row interchanges
	for(k=0; k<n-1; k++) {
		amax = (double) fabs(A[k][k]);
		m = k;
		for(i=k+1; i<n; i++) {	//find the row with largest pivot
			xfac = (double) fabs(A[i][k]);
			if(xfac > amax) { amax = xfac; m=i; }
		}
		if(m != k) {			//row interchanges
			rowx++;
			temp = b[k];
			b[k] = b[m];
			b[m] = temp;
			for(j=k; j<n; j++) {
				temp    = A[k][j];
				A[k][j] = A[m][j];
				A[m][j] = temp;
			}
		}
		for(i=k+1; i<n; i++) {
			xfac = A[i][k]/A[k][k];
			for(j=k+1; j<n; j++) {
				A[i][j] = A[i][j]-xfac*A[k][j];
			}
			b[i] = b[i]-xfac*b[k];
		}
	}
	for(j=0; j<n; j++) {
		k=n-j-1;
		x[k] = b[k];
		for(i=k+1; i<n; i++) {
			x[k] = x[k]-A[k][i]*x[i];
		}
		x[k] = x[k]/A[k][k];
	}
}

/* Main Program */
void main() {
	int i, j, k;		//indices
	double* x;  		//pointer to neuron data
	double* y;			//pointer to outputs
	double* w;  		//pointer to weights
	double* b;			//pointer to biases	
	double* dedw;		//pointer to weight partials	
	double* dedb;		//pointer to bias partials	
	double* dat; 		//pointer to aux data
	double* de;			//pointer to delta error
	double* de_old;		//pointer to delta error
	double* dw;			//pointer to delta weight & biases
	double** A;			//pointer to sensitivity matrix
	double** A_old;		//pointer to sensitivity matrix
	double err, total;
	FILE* fin  = fopen("in.bin",  "rb");
	FILE* fout = fopen("out.bin", "rb");
	FILE* fp   = fopen("calc.bin","wb");
	
	/* Allocate Memory */
	get_size();
	x     = (double*) calloc(nNeurons,sizeof(double));
	y     = (double*) calloc(nOutputs,sizeof(double));
	w     = (double*) calloc(nWeights,sizeof(double));
	b     = (double*) calloc(nBiases, sizeof(double));
	dedw  = (double*) calloc(nWeights,sizeof(double));
	dedb  = (double*) calloc(nBiases, sizeof(double));
	dat   = (double*) calloc(nNeurons,sizeof(double));
	de    = (double*) calloc(nTuners, sizeof(double));
	de_old= (double*) calloc(nTuners, sizeof(double));
	dw    = (double*) calloc(nTuners, sizeof(double));
	A     = (double**)calloc(nTuners, sizeof(double*));
	A_old = (double**)calloc(nTuners, sizeof(double*));
	for(i=0; i<nTuners; i++) { 
		A[i]    = (double*) calloc(nTuners, sizeof(double));
		A_old[i]= (double*) calloc(nTuners, sizeof(double));
	} 
		
	/* Initialize Weights */
	srand(123456);
	for(i=0; i<nWeights; i++) {
		w[i] = (double) (rand()%10-5)/10.0;
	}
	for(i=0; i<nBiases; i++) {
		b[i] = (double) (rand()%10-5)/10.0;
	}

	/* Train the Net */ 
	for(k=0; k<3; k++) {
		for(i=0; i<nTuners; i++) {
			/* Get Input/Output */
			fread(x,sizeof(double),nInputs, fin);
			fread(y,sizeof(double),nOutputs,fout);
			
			/* Run Net and Calculate Error */
			run(x,w,b);
			err = get_error(x,y);		
			de[i] = -err;
			
			/* Populate Sensitivity matrix */
			get_partials(x,y,w,dedw,dedb,dat);
			for(j=0; j<nWeights; j++) { 
				A[i][j] = dedw[j];
			}
			for(j=0; j<nBiases; j++) { 
				A[i][nWeights+j] = dedb[j];
			}
		}
		
		/* Display Matrices */
		printf("A:\n");
		for(i=0; i<nTuners; i++) {
			for(j=0; j<nTuners; j++) {
				printf(",%.17g",A[i][j]);
			}
			printf(";");
		}
		printf("\n\nde:\n");
		for(i=0; i<nTuners; i++) { printf("%.17g;", de[i]); }
		printf("\n\n");

		/* Save original matrices, solve linear system */
		for(i=0; i<nTuners; i++) { 
			de_old[i] = de[i];
			memcpy(A_old[i],A[i],nTuners*sizeof(double)); 			
		}
		gauss(A,de,dw,nTuners);	
		//printf("\ndw:\n");
		//for(i=0; i<nTuners; i++) { printf("%.17g;", dw[i]); }
		
		/* Check Linear Solver */
		err=0;
		for(i=0; i<nTuners; i++) {
			total = 0;
			for(j=0; j<nTuners; j++) {
				total += A_old[i][j]*dw[j];
			}
			err += fabs(total-de_old[i]);
		}
		if(err>1e-5) {
			printf("\n\nError: residual:\n%.17g\n",err);
			goto stop;
		}
		
		/* Update Weights	*/
		for(i=0; i<nWeights; i++) {
			w[i] += dw[i];
		}
		for(i=0; i<nBiases; i++) {
			b[i] += dw[nWeights+i];
		}
		//getch();
		
		/* Rewind Streams */ 
		rewind(fin);
		rewind(fout);
	}
	
	/* Rerun the Net */
	printf("\n\n");
	for(i=0; i<nTuners; i++) {
		fread(x,sizeof(double),nInputs, fin);
		fread(y,sizeof(double),nOutputs,fout);
		run(x,w,b);
		err = get_error(x,y);
		de[i] = -err;
		fwrite(x+nNeurons-nOutputs,sizeof(double),nOutputs,fp);
	}
	printf("de_final:\n");
	for(i=0; i<nTuners; i++) { printf("%.17g;", de[i]); }
	printf("\n\n");
	
	/* Clean-up */
	stop:
	printf("\n\ndone!\n");
	free(x);
	free(y);
	free(w);
	free(b);
	free(dedw);
	free(dedb);
	free(dat);
	free(de);
	free(dw);
	for(i=0; i<(nWeights+nBiases); i++) { 
		free(A[i]);
	} 
	free(A);
	fclose(fin);
	fclose(fout);
	fclose(fp);
	getch();
}
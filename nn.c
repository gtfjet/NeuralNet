
#include <stdio.h>
#include <math.h>

double sig(double u) {
	return 1.0/(1.0 + exp(-u));	
}


/*  Helper to find index of element */
int FindIndex(double* x, va_list ap) {
  int index=0;
  for(int i=0;i<ndims;i++) {
    int j=va_arg(ap,int);
    Assert((j>=1)&&(j<=dim),"Index out of bounds");
    index=index*dim+(j-1);
  }
  return index;
}

/* Indexing to set/get elements */
DYMOLA_STATIC Real RealElement(const RealArray a,...) {
  SizeType index;
  va_list ap;
  va_start(ap,a);
  index=FindIndex(a.ndims,a.dims,ap);
  va_end(ap);
  return a.data[index];
}

/* Calculate error Partials recursively */
void get_partials(double *dedw) {
	
	for(int i=0; i<nNeurons; i++) {
		dedw[i] = -x[i];
		
		
		
		
		

		for(int j=1; j<(nLayers-1); j++) {
			dedw[i] *= w[s[j-1]*s[j]];
			get_partials();
		}				
	}		
}




void main() {
	int s[] = {2,4,3,2};  //define structure of neural net
	double* w;  		//pointer to weights
	double* x;  		//pointer to neuron data
	double* b;			//pointer to biases	
	double* dedw;		//pointer to partials	
	int nNeurons = 0;	
	int nWeights = 0;
	int nLayers  = sizeof(s)/sizeof(int);
	int nInputs  = s[0];
	int nOutputs = s[nLayers];
	double u1, u2, y1, y2, total, err1, err2;
	int p, q, r;

	/* Get Size of Network */
	for(int i=0; i<nLayers; i++) {
		nNeurons += s[i];
		if(i>0) {
			nWeights += s[i]*s[i-1];
		}
	}		
	printf("%i,%i,%i\n",nLayers,nNeurons,nWeights); 
	
	/* Allocate Memory */
	w = (double*) calloc(nWeights,sizeof(double));
	x = (double*) calloc(nNeurons,sizeof(double));
	b = (double*) calloc(nNeurons-nInputs,sizeof(double));
	dedw = (double*) calloc(nWeights,sizeof(double));
	
	
	/* Train the net */ 
	for(int n=0; n<=100; n++) {
		/* Calculate input/output CURRENTLY TWO OF EACH */
		u1 = (double) n/100;
		u2 = (double) n/10;
		y1 = sin(u1);
		y2 = cos(u1*u2);
		
		/* Calculate output using the net */
		p = 0; //step thru w
		q = 0; //step thru x
		r = 0; //step thru b
		for(int i=1; i<nLayers; i++) {
			for(int j=0; j<s[i]; j++) {
				total = b[r];  //start with bias
				r++;
				for(int k=0; k<s[i-1]; k++) {
					total += w[p]*x[q+k];
					p++;
				}
				x[q+s[i-1]+j] = tanh(total);
			}
			q += s[i-1];
		}
		
		/* Calculate SQUARE Error CURRENTLY TWO OUTPUTS*/ 
		err = (y1-x[nNeurons-2])*(y1-x[nNeurons-2])/2 + (y2-x[nNeurons-1])*(y2-x[nNeurons-1])/2;
		printf("%f,%f,%f,%f\n",u1,y1,x[nNeurons-2],err);	
		
		get_partials();
		
		/* Update weights using one-step Newton's */
		for(int i=0; i<nWeights; i++) {
			//w[i] -= err/dedw[i];
		}
	 
	}
	
	
	
	
	/* Free Memory */
	free(w);
	free(x);
	free(b);
	free(dedw);
	getch();
}


#include <stdio.h>
#include <math.h>

double sig(double u) {
	return 1.0/(1.0 + exp(-u));	
}

void main() {
	int s[] = {1,3,1};  //define structure of neural net
	double* w;  		//pointer to weights
	double* x;  		//pointer to neuron data
	double* b;			//pointer to biases	
	double* dedw;		//pointer to partials	
	int nNeurons = 0;	
	int nWeights = 0;
	int nLayers  = sizeof(s)/sizeof(int);
	int nInputs  = s[0];
	int nOutputs = s[nLayers];
	double u, y, total, err;
	int p, q, r;

	/* Get size of network */
	for(int i=0; i<nLayers; i++) {
		nNeurons += s[i];
		if(i>0) {
			nWeights += s[i]*s[i-1];
		}
	}		
	printf("%i,%i,%i\n",nLayers,nNeurons,nWeights); 
	
	/* Allocate memory */
	w = (double*) calloc(nWeights,sizeof(double));
	x = (double*) calloc(nNeurons,sizeof(double));
	b = (double*) calloc(nNeurons-s[0],sizeof(double));
	dedw = (double*) calloc(nWeights,sizeof(double));
	
	
	/* Train the net */ 
	for(int n=0; n<=1000; n++) {
		/* Calculate input/output CURRENTLY ONE OF EACH */
		u = (double) n/100;
		y = sin(u);
		
		/* Calculate output using the net */
		p = 0; //step thru w
		q = 0; //step thru x
		r = 0; //step thru b
		for(int i=1; i<nLayers; i++) {
			for(int j=0; j<s[i]; j++) {
				total = b[r];
				r++;
				for(int k=0; k<s[i-1]; k++) {
					total += w[p]*x[q+k];
					p++;
				}
				x[q+s[i-1]+j] = total;
			}
			q += s[i-1];
		}
		
		/* Calculate error */ 
		err = pow(y-x[nNeurons-1],2);
		err = sqrt(err);
		printf("%f,%f,%f,%f\n",u,y,x[nNeurons-1],err);	
		
		/* Calculate error partials recursively */
		for(int i=0; i<nWeights; i++) {
			dedw[i] = -x[i%nInputs];
			for(int j=1; j<(nLayers-1); j++) {
				dedw[i] *= w[s[j-1]*s[j]];
			}				
		}		
		
		/* Update weights using one-step Newton's */
		for(int i=0; i<nWeights; i++) {
			//w[i] -= err/dedw[i];
		}
	
	}
	
	
	
	
	
	free(w);
	free(x);
	free(b);
	free(dedw);
	getch();
}

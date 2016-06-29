
#include <stdio.h>
#include <math.h>

int net[] = {2,4,3,2};  //define structure of neural net
double* w;  		//pointer to weights
double* x;  		//pointer to neuron data
double* b;			//pointer to biases	
double* dedw;		//pointer to partials	
double* help

double sig(double u) {
	return 1.0/(1.0 + exp(-u));	
}

/*  Helper to find index of element */
int find_index(double* x, va_list ap) {
  int index=0;
  for(int i=0;i<ndims;i++) {
    int j=va_arg(ap,int);
    index=index*dim+(j-1);
  }
  return index;
}

/* Return zero based index to the layer */
int get_layer(int k) {
	int n = 0;
	int c = -1;
	while(k>c) {
		c += net[n];
		n++;
	}
	return(n-1);
}

/* Calculate error Partials recursively */
// void get_partials(int k, int total) {
	// int nLayers = sizeof(net)/sizeof(int);
	// int layer   = get_layer(k);
	// int total  *= (1-x[k]*x[k]);

	// for(int i=0; i<net[layer-1]; i++) {
		// get_partials(j, total*w[]);
	// }
	
		
		
		
		// dedw[k] = x[k];
		// for(int j=1; j<(nLayers-1); j++) {
			// dedw[i] *= w[net[j-1]*net[j]];

		// }				
	// }		
	
	
	
			// p = 0; //step thru w
		// q = 0; //step thru x
		// while(p<nWeights) {
			// total = 0;
			// for(int i=1; i<nLayers; i++) {
				// for(int j=0; j<net[i]; j++) {
					// total = x[q];
					// total *= 1-pow(x[q+net[i-1]],2);
					// total *= 
					
					
					// for(int k=0; k<net[i-1]; k++) {
						// total += w[p]*x[q+k];
						// p++;
					// }
					// x[q+net[i-1]+j] = tanh(total);
				// }
				// q += net[i-1];
			// }
		// }
// }




void main() {
	int nNeurons = 0;	
	int nWeights = 0;
	int nLayers  = sizeof(net)/sizeof(int);
	int nInputs  = net[0];
	int nOutputs = net[nLayers];
	double u1, u2, y1, y2, total, err1, err2;
	int p, q, r;

	/* Get Size of Network */
	for(int i=0; i<nLayers; i++) {
		nNeurons += net[i];
		if(i>0) {
			nWeights += net[i]*net[i-1];
		}
	}		
	printf("%i,%i,%i\n",nLayers,nNeurons,nWeights); 
	
	/* Allocate Memory */
	w    = (double*) calloc(nWeights,sizeof(double));
	x    = (double*) calloc(nNeurons,sizeof(double));
	b    = (double*) calloc(nNeurons-nInputs,sizeof(double));
	dedw = (double*) calloc(nWeights,sizeof(double));
	help = (double*) calloc(nNeurons,sizeof(double));
	
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
			for(int j=0; j<net[i]; j++) {
				total = b[r];  //start with bias
				r++;
				for(int k=0; k<net[i-1]; k++) {
					total += w[p]*x[q+k];
					p++;
				}
				x[q+net[i-1]+j] = tanh(total);
			}
			q += net[i-1];
		}
		
		/* Calculate SQUARE Error CURRENTLY TWO OUTPUTS */ 
		err = (y1-x[nNeurons-2])*(y1-x[nNeurons-2])/2 + (y2-x[nNeurons-1])*(y2-x[nNeurons-1])/2;
		printf("%f,%f,%f,%f\n",u1,y1,x[nNeurons-2],err);	
		
		/* Get Partials CURRENTLY TWO OUTPUTS */
		//get_partials(nNeurons-2,0);
		//get_partials(nNeurons-1,0);
		
		
		p = nWeights-1; //step backward thru w
		q = nNeurons-1;	//step backward thru x
		for(int i=(nLayers-1); i>0; i--) {
			for(int j=0; j<net[i]; j++) {
				help[q] = (1-x[q]*x[q]);
				for(int k=0; k<net[i-1]; k++) {
					dedw[p] = help[q]*x[q-k-net[i]];
					help[q-k-net[i]] = help[q]*w[p]*x[q-k-net[i]];
					p--;
				}
			}
			q -= net[i];
		}
		
		
		
		
		
		total = 0;
		total = (1-x[k]*x[k]);
		
		
		
		
		

		
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
	free(help);
	getch();
}

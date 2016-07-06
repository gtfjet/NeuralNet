/*!
	G.N.N.N.U.
	General Neural Net with Newton Updates
*/

#include <stdio.h>
#include <math.h>
#define RAND_MAX 32767

void main() {
	int i, j, k, n, p, q, r;    //indices
	int net[] = {2,5,5,2};     	//define structure of neural net
	double* w;  				//pointer to weights
	double* x;  				//pointer to neuron data
	double* b;					//pointer to biases	
	double* dedw;				//pointer to partials	
	double* dedb;				//pointer to biases partials	
	double* dat; 				//pointer to data
	int nNeurons = 0;
	int nWeights = 0;
	int nLayers  = sizeof(net)/sizeof(int);
	int nInputs  = net[0];
	int nOutputs = net[nLayers-1];
	int nBiases;
	double y[2], total, err;

	/* Get Size of Network */
	for(i=0; i<nLayers; i++) {
		nNeurons += net[i];
		if(i>0) {
			nWeights += net[i]*net[i-1];
		}
	}
	nBiases = nNeurons-nInputs;
	printf("%i,%i,%i,%i\n",nLayers,nBiases,nNeurons,nWeights); 
	
	/* Allocate Memory */
	w    = (double*) calloc(nWeights,sizeof(double));
	x    = (double*) calloc(nNeurons,sizeof(double));
	b    = (double*) calloc(nBiases, sizeof(double));
	dedb = (double*) calloc(nBiases, sizeof(double));
	dedw = (double*) calloc(nWeights,sizeof(double));
	dat  = (double*) calloc(nNeurons,sizeof(double));
	
	/* Initialize Weights */
	srand(123456);
	for(i=0; i<nWeights; i++) {
		w[i] = (double) (rand()%10 - 5)/100.0;
	}
	for(i=0; i<nBiases; i++) {
		b[i] = (double) (rand()%10 - 5)/100.0;
	}

	/* Train the Net */ 
	for(n=0; n<1000; n++) {
		/* Calculate input/output CURRENTLY TWO OF EACH */
		x[0] = (double) 0.5;
		x[1] = (double) 1.5;
		y[0] = sin(x[0]*x[1]);
		y[1] = cos(x[0]*x[1]);

		/* Calculate output using the net */
		p = 0;		//step thru w
		q = 0;		//step thru x
		r = 0;		//step thru b
		for(i=1; i<nLayers; i++) {
			for(j=0; j<net[i]; j++) {
				total = b[r];  		//start with bias
				r++;
				for(k=0; k<net[i-1]; k++) {
					total += w[p]*x[q+k];
					p++;
				}
				x[q+net[i-1]+j] = tanh(total);
			}
			q += net[i-1];
		}

		/* Calculate SQUARE Error CURRENTLY TWO OUTPUTS */ 
		err = (y[0]-x[nNeurons-2])*(y[0]-x[nNeurons-2])/2 
			+ (y[1]-x[nNeurons-1])*(y[1]-x[nNeurons-1])/2;
		
		/* Get Partials CURRENTLY TWO OUTPUTS */
		for(i=0; i<nNeurons; i++) { dat[i]=0; } 
		p = nWeights-1; 				//step backward thru w
		q = nNeurons-1;					//step backward thru x
		for(i=(nLayers-1); i>0; i--) {
			for(j=0; j<net[i]; j++) {
				if((q-j)==(nNeurons-1)) {		//(q-j) is current node
					dat[q-j] = (x[q-j]-y[1])*(1-x[q-j]*x[q-j]);  					
				} else if((q-j)==(nNeurons-2)) {
					dat[q-j] = (x[q-j]-y[0])*(1-x[q-j]*x[q-j]);  	
				} else {
					dat[q-j] *= (1-x[q-j]*x[q-j]);  	
				}
				dedb[q-j-nInputs] = dat[q-j];
				for(k=0; k<net[i-1]; k++) {
					r 		= q-k-net[i]; 		//other node
					dat[r] += dat[q-j]*w[p];
					dedw[p] = dat[q-j]*x[r];
					p--;
				}
			}
			q -= net[i];
		}		

		/* Update weights using one-step Newton's */
		for(i=0; i<nWeights; i++) {
			w[i] -= n*err*dedw[i];
		}
		for(i=0; i<nBiases; i++) {
			b[i] -= n*err*dedb[i];
		}
		printf("%f\n",err);	
	}
	printf("%f, %f, %f, %f\n\n",x[nNeurons-2],x[nNeurons-1],y[0],y[1]);
	
	/* Free Memory */
	free(w);
	free(x);
	free(b);
	free(dedw);
	free(dedb);
	free(dat);
	getch();
}
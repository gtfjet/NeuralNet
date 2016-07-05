/*!
	G.N.N.N.U.
	General Neural Net with Newton Updates
*/

#include <stdio.h>
#include <math.h>
#define RAND_MAX 32767

void main() {
	int i, j, k, n, p, q, r;  	//indices
	int net[] = {2,2,2};      	//define structure of neural net
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
		//w[i] = (double) 2*rand()/RAND_MAX - 1.0;
		w[i] = (double) (rand()%10 - 5)/10.0;
	}
	for(i=0; i<nBiases; i++) {
		b[i] = (double) (rand()%10 - 5)/10.0;
	}

	/* Train the Net */ 
	for(n=0; n<1000; n++) {
		/* Calculate input/output CURRENTLY TWO OF EACH */
		x[0] = (double) 3;
		x[1] = (double) 2;
		y[0] = x[0]*x[0];
		y[1] = x[0]*x[1];
		printf("\n\nx_0=%f; x_1=%f; y_0=%f; y_1=%f;\n\n",x[0],x[1],y[0],y[1]);

		/* Calculate output using the net */
		p = 0;		//step thru w
		q = 0;		//step thru x
		r = 0;		//step thru b
		for(i=1; i<nLayers; i++) {
			for(j=0; j<net[i]; j++) {
				total = b[r];  		//start with bias
				printf("b_%i=%f;\n",r,b[r]);
				r++;
				for(k=0; k<net[i-1]; k++) {
					total += w[p]*x[q+k];
					printf("w_%i=%f; ",p,w[p]);
					p++;
				}
				x[q+net[i-1]+j] = tanh(total);
				printf("\nx_%i=%f;\n",q+net[i-1]+j,x[q+net[i-1]+j]);
			}
			q += net[i-1];
		}

		/* Calculate SQUARE Error CURRENTLY TWO OUTPUTS */ 
		err = (y[0]-x[nNeurons-2])*(y[0]-x[nNeurons-2])/2 
			+ (y[1]-x[nNeurons-1])*(y[1]-x[nNeurons-1])/2;
		printf("\nerr=%f;\n\n",err);	
		
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
				printf("dedb_%i=%f;\n",q-j-nInputs,dedb[q-j-nInputs]);
				for(k=0; k<net[i-1]; k++) {
					r 		= q-k-net[i]; 		//other node
					dat[r] += dat[q-j]*w[p];
					dedw[p] = dat[q-j]*x[r];
					printf("dat_%i=%f; dedw_%i=%f;\n",r,dat[r],p,dedw[p]);
					p--;
				}
			}
			q -= net[i];
		}

		/* Update weights using one-step Newton's */
		for(i=0; i<nWeights; i++) {
			w[i] -= 0.001*err/dedw[i];
		}
		for(i=0; i<nBiases; i++) {
			b[i] -= 0.001*err/dedb[i];
		}
		getch();
	}
	
	/* Free Memory */
	free(w);
	free(x);
	free(b);
	free(dedw);
	free(dedb);
	free(dat);
	getch();
}
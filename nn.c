/*!
	G.N.N.N.U.
	General Neural Net with Newton Updates
*/

#define RAND_MAX 32767
#include <stdio.h>
#include <math.h>

void main() {
	int net[] = {2,4,3,2};  //define structure of neural net
	double* w;  		//pointer to weights
	double* x;  		//pointer to neuron data
	double* b;			//pointer to biases	
	double* dedw;		//pointer to partials	
	double* dat; 		//pointer to data
	int nNeurons = 0;	
	int nWeights = 0;
	int nLayers  = sizeof(net)/sizeof(int);
	int nInputs  = net[0];
	int nOutputs = net[nLayers];
	double y1, y2, total, err;
	int p, q, r;

	/* Get Size of Network */
	for(int i=0; i<nLayers; i++) {
		nNeurons += net[i];
		if(i>0) {
			nWeights += net[i]*net[i-1];
		}
	}		
	printf("Info: %i,%i,%i\n\n",nLayers,nNeurons,nWeights); 
	
	/* Allocate Memory */
	w    = (double*) calloc(nWeights,sizeof(double));
	x    = (double*) calloc(nNeurons,sizeof(double));
	b    = (double*) calloc(nNeurons-nInputs,sizeof(double));
	dedw = (double*) calloc(nWeights,sizeof(double));
	dat    = (double*) calloc(nNeurons,sizeof(double));
	
	/* Initialize weights */
	srand(0);
	for (int i=0; i<nWeights; i++) {
		w[i] = (double) rand()/RAND_MAX;
		printf("%f,",w[i]);
	}
	printf("\n\n");
	
	/* Train the net */ 
	for(int n=1; n<=3; n++) {
		/* Calculate input/output CURRENTLY TWO OF EACH */
		x[0] = (double) n/100;
		x[1] = (double) n/10;
		y1 = sin(x[0]);
		y2 = cos(x[0]*x[1]);
		printf("Inputs: %f, %f\n\n", x[0],x[1]);

		
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
					printf("w[%i]=%f, ",p, w[p]);
					p++;
				}
				printf("\nx[%i]=%f -> ", q+net[i-1]+j, x[q+net[i-1]+j]);
				x[q+net[i-1]+j] = tanh(total);
				printf("%f\n",x[q+net[i-1]+j]);
			}
			q += net[i-1];
		}

		/* Calculate SQUARE Error CURRENTLY TWO OUTPUTS */ 
		err = (y1-x[nNeurons-2])*(y1-x[nNeurons-2])/2 + (y2-x[nNeurons-1])*(y2-x[nNeurons-1])/2;
		//printf("%f,%f,%f,%f\n",u1,y1,x[nNeurons-2],err);	
		printf("\n");
		/* Get Partials CURRENTLY TWO OUTPUTS */
		p = nWeights-1; //step backward thru w
		q = nNeurons-1;	//step backward thru x
		for(int i=(nLayers-1); i>0; i--) {
			for(int j=0; j<net[i]; j++) {
				dat[q-j] = (1-x[q-j]*x[q-j]);  	//q-j is current node
				for(int k=0; k<net[i-1]; k++) {
					r 		= q-k-net[i]; 		//other node
					printf("index=%i, %i, p=%i \n", q-j, r, p);
					dat[r] += dat[q-j]*w[p];
					dedw[p] = dat[q-j]*x[r];
					printf("dat[%i]=%f, dedw[%i]=%f\n",r,dat[r],p,dedw[p]);
					p--;
				}
			}
			q -= net[i];
		}

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
	free(dat);
	getch();
}




/* Return zero based index to the layer 
int get_layer(int k) {
	int n = 0;
	int c = -1;
	while(k>c) {
		c += net[n];
		n++;
	}
	return(n-1);
}
*/
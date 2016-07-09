/*!
	G.N.N.N.U.
	General Neural Net with Newton Updates
*/

#include <stdio.h>
#include <math.h>
#define RAND_MAX 32767

/* Global Variables */
int net[] = {26,100,22};    //structure of neural net
int nNeurons, nWeights, nBiases, nLayers, nInputs, nOutputs;

/* Sigmoid Function */
float sig(float t) {
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
	printf("%i,%i,%i,%i\n",nLayers,nBiases,nNeurons,nWeights); 
}
	
/* Calculate Output using the Net */
void run(float* x, float* w, float* b) {
	int total;
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
			x[q+net[i-1]+j] = sig(total);
		}
		q += net[i-1];
	}
}

/* Calculate Square Error */ 
float get_error(float* x, float* y) {
	float err = 0;
	for(int i=0; i<nOutputs; i++) {
		err += (y[i]-x[nNeurons-nOutputs+i])*(y[i]-x[nNeurons-nOutputs+i])/2;
	}
	printf("%f\n",err);
	return err;
}

/* Get Partials */
void get_partials(float* x, float *y, float* w, float* dedw, float* dedb, float* dat) {
	int p = nWeights-1; 	//step backward thru w
	int q = nNeurons-1;		//step backward thru x
	int r;
	for(int i=0; i<nNeurons; i++) { dat[i]=0; } 
	for(int i=(nLayers-1); i>0; i--) {
		for(int j=0; j<net[i]; j++) {
			if(i==(nLayers-1)) {		//(q-j) is current node
				dat[q-j] = (x[q-j]-y[nOutputs-1-j])*(1-x[q-j])*x[q-j]; 
			} else {	
				dat[q-j] *= (1-x[q-j])*x[q-j];  	
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


void main() {
	int i, j, k;				//indices
	float* x;  					//pointer to neuron data
	float* y;					//pointer to outputs
	float* w;  					//pointer to weights
	float* b;					//pointer to biases	
	float* dedw;				//pointer to partials	
	float* dedb;				//pointer to biases partials	
	float* dat; 				//pointer to data
	float err;
	FILE* fin  = fopen("in.bin",  "rb");
	FILE* fout = fopen("out.bin", "rb");
	FILE* fp   = fopen("calc.bin","wb");
	
	/* Allocate Memory */
	get_size();
	x    = (float*) calloc(nNeurons,sizeof(float));
	y    = (float*) calloc(nOutputs,sizeof(float));
	w    = (float*) calloc(nWeights,sizeof(float));
	b    = (float*) calloc(nBiases, sizeof(float));
	dedw = (float*) calloc(nWeights,sizeof(float));
	dedb = (float*) calloc(nBiases, sizeof(float));
	dat  = (float*) calloc(nNeurons,sizeof(float));
		
	/* Initialize Weights */
	srand(123456);
	for(i=0; i<nWeights; i++) {
		w[i] = (float) (rand()%10 - 5)/100.0;
	}
	for(i=0; i<nBiases; i++) {
		b[i] = (float) (rand()%10 - 5)/100.0;
	}

	/* Train the Net */ 
	for(i=0; i<3335; i++) {
		/* Get Input/Output */
		fread(x,sizeof(float),nInputs, fin);
		fread(y,sizeof(float),nOutputs,fout);
		
		for(j=0; j<1000; j++) {
			run(x,w,b);
			err = get_error(x,y);		
			get_partials(x,y,w,dedw,dedb,dat);
				
			/* Update weights using one-step Newton's */
			for(k=0; k<nWeights; k++) {
				w[k] -= j*err*dedw[k];
			}
			for(k=0; k<nBiases; k++) {
				b[k] -= j*err*dedb[k];
			}	
		}
		fwrite(x+nNeurons-nOutputs,sizeof(float),nOutputs,fp);
	}
	
	/* ReRun the Net 
	rewind(fin);
	rewind(fout);
	for(i=0; i<3335; i++) {
		fread(x,sizeof(float),nInputs, fin);
		fread(y,sizeof(float),nOutputs,fout);
		run(x,w,b);
		//err = get_error(x,y);
		fwrite(x+nNeurons-nOutputs,sizeof(float),nOutputs,fp);
	}
	*/
	
	/* Free Memory */
	free(x);
	free(y);
	free(w);
	free(b);
	free(dedw);
	free(dedb);
	free(dat);
	fclose(fin);
	fclose(fout);
	fclose(fp);
	getch();
}
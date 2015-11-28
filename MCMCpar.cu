/*  Paraller Markov Chain Monte Carlo for Hard-Sphere-Particle Simulations on GPUs
    Numerical Precision: Single Point
    Author: Xin Yan
    Credit: Part of the algorithms in this program is adapted from Joshua A. Anderson et. al
            as in Anderson, J. A. et. al, Massively parallel Monte Carlo for many-particle 
            simulations on GPUs. Journal of Computational Physics 2013, 254, 27.
    Date: 12/06/2014
*/

#define FP float
#define NMAX 4 //max number of ptcs in a cell
#define HANDLE_ERROR(err) (HandleError(err, __FILE__, __LINE__))

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <math.h>
#include <assert.h>
#include <time.h>
#include <curand_kernel.h>
#include "/home/fas/hpcprog/ahs3/cpsc424/utils/timing/timing.h"

void genbox(struct ptc *rsys, int *n, FP diameter, FP w, int N, int m);
double MCsweep(struct ptc *rsys_d, int *n_d, struct ptc *rsys_update, int *n_update, int bdim, int m,\
int iter, curandState *state);
int randint(int n);
void timing(double* wcTime, double* cpuTime);

// Check for cuda errors
// credit: http://stackoverflow.com/questions/25702573
static void HandleError(cudaError_t err, const char *file, int line) {
    if (err != cudaSuccess) {
        printf("cudaError: %s in %s at line %d. Aborting...\n", cudaGetErrorString(err), file, line);
        exit(EXIT_FAILURE);
    }
}

// structure definition for particle position
struct ptc {
    FP x;
    FP y;
};

__device__ unsigned int curand (curandState_t *state); // declare state on glb mem
__device__ float curand_uniform (curandState_t *state); // declare state on glb mem

// function to calculate the index of a ptc in glb mem
__host__ __device__ int cellIdx(int m, int x, int y, int i) {
    int q;
    if (x%2) q = (x+m)/2;
    else q = x/2;
    return (i*m + y)*m + q;
}

// Sub-sweep kernel: Update cells in each set
__global__ void subsweep(struct ptc *rsys, int *n, int off_x, int off_y, int m, int iter, curandState *state) {
    // initialization
    int x, y; // cell indices
    int nptc; // number of particles in the current cell
    struct ptc rcell[NMAX]; // array for ptc coord in a cell
    struct ptc rmove; // new ptc coord after the move
    int overlap; // 0 for no overlap, 1 otherwise
    struct ptc vec; // vector pointing to the current neighbor
    unsigned int nrand; // random number
    int i, j, s, nb; // ptc index, sweep index, neighbor index
    int xnb, ynb; // x, y of neighbor cell
    int iptc, inb; // idx of ptc in glb mem
    struct ptc rnb;
    FP dist2; // distance between two ptcs
    int nnb; // number of ptcs in the neighboring cell
    int nblist[8][2] = {{-1,0}, {1,0}, {0,-1}, {0,1}, {-1,-1}, {1,-1}, {-1,1}, {1,1}};
    __shared__ FP diameter; // sphere diameter
    __shared__ int nsweeps; // max number of sub-sweeps per cell
    __shared__ FP d; // perturbation size
    __shared__ FP w; // cell width
    __shared__ float pi2; // 2*pi

    diameter = 1.0;
    nsweeps = 4;
    d = 0.16;
    w = 1.4142*diameter;
    pi2 = 6.28318530718;
    
    // load coord in glb mem to the shared mem
    x = 2*(blockDim.x*blockIdx.x + threadIdx.x) + off_x;
    y = 2*(blockDim.y*blockIdx.y + threadIdx.y) + off_y;
    // initialize ptc # in each cell
    nptc = n[y*m+x];
    if (nptc == 0) {
        return;
    }
    // initialize rcell to -10.
    for (i=0; i<NMAX; i++) {
        rcell[i].x = -10.;
        rcell[i].y = -10.;
    }
    // copy ptc in a cell from global memory to the register
    for (i=0; i<nptc; i++) {
        iptc = cellIdx(m, x, y, i); // call func cellIdx to calc iptc in glb mem
        rcell[i] = rsys[iptc];
    }

    // Fisher-Yates shuffling
    // initialize curand. Each thread will use the same seed for each iter but with diff seq. 
    curand_init(iter, y*m+x, 0, &state[y*m+x]); 
    // copy state to local memory
    curandState localState = state[y*m+x];
    // shuffle ptcs in the current cell
    for (i=nptc-1; i>0; i--) {
        nrand = curand(&localState)%(i+1); // Generate pseudo-rand unsigned ints from [0,i]
        struct ptc temp;
        temp = rcell[i];
        rcell[i] = rcell[nrand];
        rcell[nrand] = temp;
    }
    
    i = 0;
    for (s=0; s<nsweeps; s++) {
        // perturb the ptc
        float angrand = curand_uniform(&localState)*pi2; // gen rand number from [0,2*pi) 
        angrand = (float) cos(double(angrand));
        rmove.x = rcell[i].x + d * angrand;
        rmove.y = rcell[i].y + d * sqrt(1-angrand*angrand);
        overlap = 0;
        // check if moved out of the cell
        if ((rmove.x>0) && (rmove.y>0) && (rmove.x<=w) && (rmove.y<=w)) {
            // check for overlap within the cell
            for (j=0; j<nptc; j++) {
                if (i == j) continue;
                dist2 = (rmove.x - rcell[j].x)*(rmove.x - rcell[j].x);
                dist2 += (rmove.y - rcell[j].y)*(rmove.y - rcell[j].y);
                if (dist2 < diameter*diameter) {
                    overlap = 1;
                    break;
                }
            }
            // check for overlap with ptcs in neighboring cells
            for (nb=0; nb<8; nb++) {
                xnb = x + nblist[nb][0]; // indices of neighboring cells
                ynb = y + nblist[nb][1];
                if ((xnb<0) || (ynb<0) || (xnb>=m) || (ynb>=m)) continue;
                vec.x = nblist[nb][0]*w;
                vec.y = nblist[nb][1]*w;
                nnb = n[ynb*m + xnb];
                for (j=0; j<nnb; j++) {
                    inb = cellIdx(m, xnb, ynb, j); // call func cellIdx to calc inb in glb mem
                    rnb = rsys[inb];
                    dist2 = (rmove.x-rnb.x-vec.x)*(rmove.x-rnb.x-vec.x);
                    dist2 += (rmove.y-rnb.y-vec.y)*(rmove.y-rnb.y-vec.y);
                    if (dist2 < diameter*diameter) {
                        overlap = 1;
                        nb = 8;
                        j = nnb;
                        break;
                    }
                }
            }
            if (!overlap) {
                rcell[i] = rmove; // if rmove is still in the cell, accept the move
            }
        }
        i++;
        if (i == nptc) i = 0;
    }

    // copy state back to global mem
    state[y*m+x] = localState;

    for (i=0; i<nptc; i++) { // write updated cell info to the global memory
        iptc = cellIdx(m, x, y, i);
        rsys[iptc] = rcell[i];
    }

    return;
} // Done with subsweep kernel


// cell shift GPU kernel
__global__ void shift_cells(int fx, int fy, FP d, struct ptc *rsys, int *n, struct ptc *rsys_update, \
int *n_update, int m) {
    // initializations
    int x, y; // cell index
    int nptc, nnew; // ptc # in the current cell, ptc # in the cell after the shift
    struct ptc rcell[NMAX]; // ptc coords in the current cell after shift
    struct ptc rshft; // ptc coord after the shift
    struct ptc vec; // vector pointing to the neighbor
    int i; // ptc index
    int iptc, inb; // index of ptc in global memory 
    int xnb, ynb; // cell index in direction of f
    int nnb; // number of ptcs in the neighboring cell
    __shared__ FP diameter; // sphere diameter
    __shared__ FP w;

    diameter = 1.0;
    w = 1.4142*diameter;
    
    x = blockDim.x*blockIdx.x + threadIdx.x;
    y = blockDim.y*blockIdx.y + threadIdx.y;
    nptc = n[y*m+x];
    
    // initialize all ptc coord to -10 
    for (i=0; i<NMAX; i++) {
        rcell[i].x = -10.;
        rcell[i].y = -10.;
    }
    nnew = 0;
    // perform cell move 
    for (i=0; i<nptc; i++) {
        iptc = cellIdx(m, x, y, i);
        rshft = rsys[iptc];
        rshft.x -= fx*d;
        rshft.y -= fy*d;
        // update ptc that has remains in the current cell
        if ((rshft.x>0) && (rshft.y>0) && (rshft.x<=w) && (rshft.y<=w)) {
            rcell[nnew] = rshft;
            nnew++;
        }
    }

    // update ptc that moved into the current cell from neighboring cell
    xnb = (x+fx+m)%m;
    ynb = (y+fy+m)%m;
//    if ((xnb>=0) && (ynb>=0) && (xnb<m) && (ynb<m)) {
        vec.x = fx*w;
        vec.y = fy*w;
        nnb = n[ynb*m + xnb];
        for (i=0; i<nnb; i++) {
            inb = cellIdx(m, xnb, ynb, i);
            rshft = rsys[inb];
            rshft.x -= fx*d;
            rshft.y -= fy*d;
            if ((rshft.x<=0) || (rshft.y<=0) || (rshft.x>w) || (rshft.y>w)) {
                rshft.x += vec.x;
                rshft.y += vec.y;
                rcell[nnew] = rshft;
                nnew++;
            }
        }
//    }
    
    // update the coord and ptc # info to a new buffer
    n_update[y*m+x] = nnew; // update the ptc # to the new buffer
    for (i=0; i<nnew; i++) {
        iptc = cellIdx(m, x, y, i);
        rsys_update[iptc] = rcell[i];
    }
    return;
}


int main(int argc, char *argv[]) {
    // Declaration and Initialization
    int m, N; // number of cells in each dim, total ptc #
    const FP diameter = 1.; // diameter of the ptc sphere
    const FP w = 1.4142*diameter; // set the width of the cell to 2*sqrt(2)*diameter 
    int bdim; // blockDim
    struct ptc *rsys; // rsys on CPU host
    int *n; // n array on CPU host
    struct ptc *rsys_d, *rsys_update; // rsys on GPU glb mem
    int *n_d, *n_update; // n on GPU glb mem
    int sizer, sizen; // size of rsys and n in bytes
    struct ptc temp; // ptc generated by rng
    int x, y; // cell index
    int iter, maxiter; // the current MC sweeps and the max number of iterations
    int i;
    int gpucount;
    curandState *state;
    double wctime, totwctime; // timers

    //Read command line args 
    if (argc < 5) {
        printf("Error: Incorrect command line format. Aborting...\n");
        printf("Usage: MCMCpar <N> <m> <bdim> <maxiter>\n");
        exit(-1);
    }
    
    N = atoi(argv[1]);
    m = atoi(argv[2]);
    bdim = atoi(argv[3]);
    maxiter = atoi(argv[4]);

    if (N > m*m*NMAX) {
        printf("Error: Too many particles in the system. Aborting...\n");
        exit(-1);
    }

    // output basic simulation information
    printf("###### Markov Chain Monte Carlo Simulation on GPUs ######\n\n");
    printf("### Basic Info: \n");
    printf("# Total NO. of particles  = %d\n", N);
    printf("# NO. of cells in each dimension = %d\n", m);
    printf("# Number of Monte Carlo sweeps = %d\n", maxiter);
    // Check GPU device
    HANDLE_ERROR(cudaGetDeviceCount(&gpucount));
    printf("# GPU device count = %d.\n\n", gpucount);

    if (bdim*bdim > 1024) {
        printf("Error: Too many threads in a block. Aborting...\n");
        exit(-1);
    }

    // allocate n, rsys arrays and initialize them to zero
    int nsize = m*m;
    int rsyssize = nsize*NMAX;
    n = (int *) malloc(nsize*sizeof(int));
    rsys = (struct ptc *) malloc(rsyssize*sizeof(struct ptc));
    for (i=0; i<nsize; i++) n[i] = 0;
    for (i=0; i<rsyssize; i++) {
        rsys[i].x = -10.;
        rsys[i].y = -10.;
    }
    
    // generate initial simulation system
    printf("### Start generating intial simulation box...\n");
    genbox(rsys, n, diameter, w, N, m); 
    printf("# Box generation successfully finished!\n");

    int ncount = 0; // ptc counter
    for (y=0; y<m; y++) {
        for (x=0; x<m; x++) {
            ncount += n[m*y+x];
        }
    }
    printf("# %d particles generated in the system\n", ncount);
    // print initial particle positions
    printf("# Initial particle positions:\n");
    for (i=0; i<NMAX; i++) {
        for (y=0; y<m; y++) {
            for (x=0; x<m; x++) {
                temp = rsys[(i*m+y)*m+x];
                printf("%10.4f\t%10.4f\n",temp.x, temp.y);
            }
        }
    }
    printf("\n");

    // allocate and initialize global memory on device
    sizer = m * m * NMAX * sizeof(struct ptc);
    sizen = m * m * sizeof(int);
    HANDLE_ERROR(cudaMalloc((void **) &rsys_d, sizer));
    HANDLE_ERROR(cudaMemcpy(rsys_d, rsys, sizer, cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMalloc((void **) &rsys_update, sizer)); // allocate an update buffer
    HANDLE_ERROR(cudaMemcpy(rsys_update, rsys, sizer, cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMalloc((void **) &n_d, sizen));
    HANDLE_ERROR(cudaMemcpy(n_d, n, sizen, cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMalloc((void **) &n_update, sizen)); // allocate an update buffer
    HANDLE_ERROR(cudaMemcpy(n_update, n, sizen, cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMalloc((void **) &state, m*m*sizeof(curandState)));

// Loop for Monte Carlo sweep
    totwctime = 0.;
    printf("### Entering Monte Carlo sweep loops:\n");
    for (iter=0; iter<maxiter; iter++) {
        printf("# MC sweep iteration # %d\n", iter);
        wctime = MCsweep(rsys_d, n_d, rsys_update, n_update, bdim, m, iter, state);
        totwctime += wctime;
        printf("# Iteration # %d finished, total wctime: %f\n", iter, wctime);

        // Compute physical properties and output

    }
    printf("# Monte Carlo sweep finished in %f sec. Writing results to the output...\n\n", totwctime);

// Copy data from GPU glb mem back to CPU host
    if (maxiter%2) {
        HANDLE_ERROR(cudaMemcpy(rsys, rsys_update, sizer, cudaMemcpyDeviceToHost));
        HANDLE_ERROR(cudaMemcpy(n, n_update, sizen, cudaMemcpyDeviceToHost));
    } 
    else {
        HANDLE_ERROR(cudaMemcpy(rsys, rsys_d, sizer, cudaMemcpyDeviceToHost));
        HANDLE_ERROR(cudaMemcpy(n, n_d, sizen, cudaMemcpyDeviceToHost));
    }

    // print the coordinate of all particles after each MC sweep 
    ncount = 0; // ptc counter
    for (y=0; y<m; y++) {
        for (x=0; x<m; x++) {
            ncount += n[m*y+x];
        }
    }
    printf("# Final number of particles in the system: %d\n", ncount);
    printf("# Final position of particles:\n");
    for (i=0; i<NMAX; i++) {
        for (y=0; y<m; y++) {
            for (x=0; x<m; x++) {
                temp = rsys[(i*m+y)*m+x];
                printf("%10.4f\t%10.4f\n",temp.x, temp.y);
            }
        }
    }

// Free GPU glb mem
    HANDLE_ERROR(cudaFree(rsys_d)); 
    HANDLE_ERROR(cudaFree(rsys_update));
    HANDLE_ERROR(cudaFree(n_d)); 
    HANDLE_ERROR(cudaFree(n_update));
    HANDLE_ERROR(cudaFree(state));

    printf("\n### Successful Termination of the Markov Chain Monte Carlo Simulation!\n");
    return(0);
}


// set up a box of 8*8 cells, m should be multiples of 8, N should be multiples of m*m
void genbox(struct ptc *rsys, int *n, FP diameter, FP w, int N, int m) {
    // Declarations
    struct ptc test; // test particle
    int ptcbox = N*64/(m*m); // number of ptcs per box
    FP lb2 = diameter*diameter; // shortest allowed distance between two ptcs
    int nbox = 0; // actual number of ptcs in the box
    int success; // successfully generated a new ptc
    FP dist2; // distance^2 between two particles
    struct ptc vec;
    int nb, xnb, ynb, inb, nnb; // index of neighboring cell and total ptc # in nb cell
    struct ptc rnb; // ptc in nb cell
    int xbox, ybox; // the index of cell in box
    int idx; //idx in rsys
    int i, j, x, y;
    int nblist[8][2] = {{-1,0}, {1,0}, {0,-1}, {0,1}, {-1,-1}, {1,-1}, {-1,1}, {1,1}};
    int ncell;

    // loop over all cells to generation ptc positions
    for (i=0; i<NMAX; i++) {
        for (y=0; y<8; y++) {
            for (x=0; x<8; x++) {
                if (nbox >= ptcbox) { // enough ptc has been gen, break out of nested loops
                    x = y = 8;
                    i = NMAX;
                    break;
                }
                else {
                    success = 0;
                    while (!success) { // loop until successfully generated a new ptc 
                        test.x = (FP)rand()/(FP)RAND_MAX*w; // gen test ptc within [0,w)
                        test.y = (FP)rand()/(FP)RAND_MAX*w; // gen test ptc within [0,w)
                        // check for overlap within the cell
                        ncell = n[y*m+x];
                        if (ncell == 0) {
                            success = 1;
                        }
                        else {
                            for (j=0; j<ncell; j++) { //loop over all previously generated ptcs
                                idx = cellIdx(m, x, y, j); //very bad memory retrieving
                                dist2 = (test.x-rsys[idx].x)*(test.x-rsys[idx].x);
                                dist2 += (test.y-rsys[idx].y)*(test.y-rsys[idx].y);
                                if (dist2 < lb2) { //overlap
                                    success = 0; 
                                    break;
                                }
                                else {
                                    success = 1;
                                }
                            }
                        }
                        //if no overlap within the cell, check for overlap with neighbor cells
                        if (success) { 
                            for (nb=0; nb<8; nb++) {
                                xnb = (x + nblist[nb][0]+8)%8; // indices of neighboring cells
                                ynb = (y + nblist[nb][1]+8)%8;
                                vec.x = nblist[nb][0]*w;
                                vec.y = nblist[nb][1]*w;
                                nnb = n[ynb*m + xnb];
                                if (nnb == 0) continue;
                                for (j=0; j<nnb; j++) { // loop over all ptcs in a neighbor cell
                                    inb = cellIdx(m, xnb, ynb, j); // call func cellIdx to calc inb in glb mem
                                    rnb = rsys[inb];
                                    dist2 = (test.x-rnb.x-vec.x)*(test.x-rnb.x-vec.x);
                                    dist2 += (test.y-rnb.y-vec.y)*(test.y-rnb.y-vec.y);
                                    if (dist2 < lb2) {
                                        success = 0;
                                        nb = 8;
                                        j = nnb;
                                        break;
                                    }
                                }
                            }
                        }
                    }
                    // successful generation of a test ptc, store it in the host memory
                    idx = cellIdx(m, x, y, i);
                    rsys[idx] = test;
                    n[m*y+x]++;
                    nbox++;
                }
            }
        }
    }
                                
    int iptcbox, iptc;
   // replicate the 8*8 box to all other cells in the system
    for (i=0; i<NMAX; i++) {
        for (y=0; y<m; y++) {
            ybox = y%8;
            for (x=0; x<m; x++) {
                xbox = x%8;
                iptcbox = cellIdx(m, xbox, ybox, i);
                iptc = cellIdx(m, x, y, i);
                n[y*m+x] = n[ybox*m+xbox];
                rsys[iptc] = rsys[iptcbox];
            }
        }
    }
    return;
} // Done with genbox


// Monte Carlo Sweep
double MCsweep(struct ptc *rsys_d, int *n_d, struct ptc *rsys_update, int *n_update, int bdim, int m,\
int iter, curandState *state) {
    // initialization
    int chksets[] = {'a', 'b', 'c', 'd'}; // collection of checkerboard sets
    int set; // checkerboard set
    int i; // index
    unsigned int nrand; // random number
    int off_x, off_y; // cell index offset to the lower-leftmost active cell in the current set
    FP d; // cell shift distance
    int shftvec[4][2] = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}}; // unit vectors for cell shift
    int fx, fy; // vector to perform cell shift;
    int bx, by, gx, gy; // blockdim and griddim, bx=by=bdim
    const FP diameter = 1.0;
    const FP w = 1.4142*diameter;
    double start, end, cput;
    
    // start timing
    timing(&start, &cput);
    // Fisher-Yates shuffling
    srand(time(NULL));
    for (i=3; i>0; i--) {
        //pick a rand number in [0, i]
        nrand = randint(i+1);
        //swap chksets[i] and chksets[nrand]
        int temp;
        temp = chksets[nrand];
        chksets[nrand] = chksets[i];
        chksets[i] = temp;
    }
    
    // define grid and block size
    bx = by = bdim; // bx and by set from command line input 
    gx = gy = m/(bdim*2); // total # threads=m/2, each thread controls every other r/c of cells
    if (bx*gx < m/2) {
        printf("Error: number of threads in x dimension less than half the number of cells. \
Aborting...\n");
        exit(-1);
    }
    dim3 dimBlock(bx, by, 1);
    dim3 dimGrid(gx, gy, 1);
    printf("# sub-sweeps: Block x = %d, Block y = %d, Grid x = %d, Grid y = %d.\n", bx, by, gx, \
gy);
    
    // Loop over checkerboard sets
    for(i=0; i<4; i++) {
        set = chksets[i];
        switch(set) {
            case 'a': 
                off_x = 0; 
                off_y = 0;
                break;
            case 'b': 
                off_x = 1;
                off_y = 0;
                break;
            case 'c': 
                off_x = 0; 
                off_y = 1;
                break;
            case 'd': 
                off_x = 1; 
                off_y = 1;
                break;
            default: 
                printf("Error: set not in the checkerboard sets. Aborting...\n");
                exit(-1);
        }

        // Sub-sweep GPU kernel
        // need to swap buffer each iteration 
        if (iter%2) 
            subsweep<<<dimGrid, dimBlock>>>(rsys_update, n_update, off_x, off_y, m, iter, state);
        else
            subsweep<<<dimGrid, dimBlock>>>(rsys_d, n_d, off_x, off_y, m, iter, state);
        // synchronize all threads in the device
        HANDLE_ERROR(cudaDeviceSynchronize());
         
    } // Done with sub-sweeps
    
    // Shift cells 
    d = (float) rand()/(float)(RAND_MAX)*w/2.;//generate random floating point number [0, w/2.]
    nrand = randint(4); // randomly select a direction to perform cell shift
    fx = shftvec[nrand][0];
    fy = shftvec[nrand][1];
    gx = gy = m/bdim; // total # threads=m, each thread controls a cell
    if (bx*gx < m) {
        printf("Error: number of threads in x dimension less than the nubmer of the number of cells. \
Aborting...\n");
        exit(-1);
    }
    printf("# shift cells: Block x = %d, Block y = %d, Grid x = %d, Grid y = %d.\n", bx, by, gx, gy);
    dim3 dimGrid2(gx, gy, 1);
    // need to swap buffer each iteration 
    if (iter%2)
        shift_cells<<<dimGrid2, dimBlock>>>(fx, fy, d, rsys_update, n_update, rsys_d, n_d, m);
    else
        shift_cells<<<dimGrid2, dimBlock>>>(fx, fy, d, rsys_d, n_d, rsys_update, n_update, m);
    HANDLE_ERROR(cudaDeviceSynchronize());
    //end timing
    timing(&end, &cput);
    return(end - start);
} // Done with MC sweep


// random number generator, returns an integer in the range [0, n)
// credit: http://stackoverflow.com/questions/822323
int randint(int n) {
    if ((n-1) == RAND_MAX) {
        return rand();
    } else {
        // chop off all values that would cause skew
        long end = RAND_MAX / n;
        assert (end > 0L);
        end *= n;
        //ignore results from rand() that fall above the limit 
        int r;
        while ((r=rand()) >= end) ;
        return r%n; // obtain rand number that give uniform distribution
    }
}

// timer function
// credit: Dr. Andrew Sherman, Yale University
void timing(double* wcTime, double* cpuTime)
{
   struct timeval tp;
   struct rusage ruse;

   gettimeofday(&tp, NULL);
   *wcTime=(double) (tp.tv_sec + tp.tv_usec/1000000.0); 
  
   getrusage(RUSAGE_SELF, &ruse);
   *cpuTime=(double)(ruse.ru_utime.tv_sec+ruse.ru_utime.tv_usec / 1000000.0);
}


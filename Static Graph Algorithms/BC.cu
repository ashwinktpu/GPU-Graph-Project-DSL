//written by Rajesh

#include<stdio.h>
#include<stdlib.h>
#include<limits.h>
#include<cuda.h>
#include"graph.hpp"

#define cudaCheckError() {                                             \
 cudaError_t e=cudaGetLastError();                                     \
 if(e!=cudaSuccess) {                                                  \
   printf("Cuda failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(e)); \
   exit(0);                                                            \
 }                                                                     \
}

template <typename T>
__global__ void initKernel(unsigned V, T* init_array, T initVal)
{
  unsigned id = threadIdx.x + blockDim.x * blockIdx.x;
  if(id < V)
  {
    init_array[id]=initVal;
  }
}

/*
__global__ void inBFS_kernel(int * d_offset , int * d_edgeList , int* d_weight, int * d_level , int src ,int V, int MAX_VAL , bool * d_modified_prev,
bool * d_modified_next, bool * d_finished)
{
  unsigned int id = threadIdx.x + (blockDim.x * blockIdx.x);
  unsigned int v = id;
  if (id < V)
  {
    if (d_modified_prev[id] ){
      for (int edge = d_offset[id]; edge < d_offset[id+1]; edge ++)
      {
        int nbr = d_edgeList[edge] ;
        //int e = edge;

        int dist_new ;
        if(d_level[id] != MAX_VAL)
          dist_new = d_level[v] + 1;
        //~ int dist_new = d_level[v] + d_weight[e];
        if (d_level[nbr] > dist_new)
        //~ if (d_level[id] != MAX_VAL && d_level[nbr] > dist_new)
        {
          atomicMin(&d_level[nbr] , dist_new);
          d_modified_next[nbr]=true;
          d_finished[0] = false;
        }
      }
    }
  }

}
*/
__global__ void incHop(int* d_hops_from_source) {
    *d_hops_from_source = *d_hops_from_source + 1;
  }

  __global__ void bc_forward_pass(int* d_offset,int* d_edgeList,int* d_edgeLen, unsigned* d_sigma, int* d_level, int* d_hops_from_source, unsigned n, bool* d_finished) {
    unsigned u = blockIdx.x * blockDim.x + threadIdx.x;
    if(u >= n) return;

    // only processing the nodes at level '*d_hops_from_source' -- a level synchronous processing, though not work efficient
    if(d_level[u] == *d_hops_from_source) {
       unsigned end = d_offset[u+1];
       for(unsigned i = d_offset[u]; i < end; ++i) { // going over the neighbors of u
          unsigned v = d_edgeList[i];
          if(d_level[v] == -1) {  // v is seen for the first time
            d_level[v] = *d_hops_from_source + 1; // no atomics required since this is benign data race due to level synchronous implementation
            *d_finished = false;
          }
          if(d_level[v] == *d_hops_from_source + 1) { // 'v' is indeed the neighbor of u
            atomicAdd(&d_sigma[v], d_sigma[u]);
            //~ *d_finished = false;
          }
       }
    }
  }


  __global__ void bc_backward_pass(int* d_offset,int* d_edgeList,int* d_edgeLen, unsigned* d_sigma, double* d_delta, double* d_nodeBC, int* d_level, int* d_hops_from_source, unsigned n) {

    unsigned u = blockIdx.x * blockDim.x + threadIdx.x;
    if(u >= n) return;

    if(d_level[u] == *d_hops_from_source - 1) { // backward traversal of DAG, one level at a time

       unsigned end = d_offset[u+1];
       double sum = 0.0;
       for(unsigned i = d_offset[u]; i < end; ++i) { // going over the neighbors of u for which it is the predecessor in the DAG
          unsigned v = d_edgeList[i];
          if(d_level[v] == *d_hops_from_source) {
            sum += (1.0 * d_sigma[u]) / d_sigma[v] * (1.0 + d_delta[v]);
          }
       }

       d_delta[u] += sum;
    }

  }

  __global__ void accumulate_bc(double * d_delta, double* d_nodeBC, int* d_level, unsigned s, unsigned n) {

    unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid >= n || tid == s || d_level[tid] == -1) return;


    d_nodeBC[tid] += d_delta[tid]/2.0;

  }


__global__ void initialize(unsigned* d_sigma, double* d_delta, int* d_level, int* d_hops_from_source, unsigned s, unsigned n) {
    unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid < n) {
      d_level[tid] = -1;
      d_delta[tid] = 0.0;
      d_sigma[tid] = 0;

      if(tid == s) { // for the source
        d_level[tid] = 0;
        d_sigma[tid] = 1;
        *d_hops_from_source = 0;
      }
    }
  }

  void BC(int * OA , int * edgeList , int* edgeLength, int V, int E)
{

  int* d_offset;
  int* d_edgeList;
  int* d_edgeLen;  //why this for unweighted?


  bool* d_modified_prev; //might help acclerate fwd pass
  bool* d_modified_next;
  bool* d_finished;

  bool finished = false;


  unsigned int block_size = V;
  unsigned int num_blocks = 1;

  if(V > 1024){
    block_size = 1024;                        // at some point when sh-mem comes in, it should be 512
    num_blocks = (V+block_size-1)/block_size; // modified without ceil fun call
  }

  // VAR for BC
  double* d_delta ;
  double* d_nodeBC;
  unsigned* d_sigma;

  int* d_level;
  int* d_hops_from_source;
  int hops_from_source;

  cudaMalloc(&d_offset,sizeof(int) *(1+V));
  cudaMalloc(&d_edgeList,sizeof(int) *(E));
  cudaMalloc(&d_edgeLen,sizeof(int) *(E));
  cudaMalloc(&d_level,sizeof(int) *(V));

  cudaMalloc(&d_sigma,  sizeof(unsigned) * V);
  cudaMalloc(&d_delta,  sizeof(double)   * V);
  cudaMalloc(&d_nodeBC, sizeof(double)   * V);

  cudaMalloc(&d_modified_prev,sizeof(bool) *(V));
  cudaMalloc(&d_modified_next,sizeof(bool) *(V));
  cudaMalloc(&d_finished,sizeof(bool) *(1));

  cudaMalloc(&d_hops_from_source, sizeof(int));

  cudaEvent_t start, stop; ///TIMER START
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  float milliseconds = 0;
  cudaEventRecord(start,0);

  cudaMemcpy (d_offset, OA, sizeof(int) *(1+V) ,cudaMemcpyHostToDevice);
  cudaMemcpy (d_edgeList, edgeList, sizeof(int) *(E) ,cudaMemcpyHostToDevice);
  cudaMemcpy (d_edgeLen, edgeLength , sizeof(int) *(E) ,cudaMemcpyHostToDevice);
  //~ cudaMemcpy (d_level, dist, sizeof(int) *(V) ,cudaMemcpyHostToDevice);
  //~ cudaMemcpy (d_modified_prev, modified , sizeof(bool) *(V) ,cudaMemcpyHostToDevice);
  //~ cudaMemcpy (d_finished, finished , sizeof(bool) *(1) ,cudaMemcpyHostToDevice);

  int* level;
  level= (int*) malloc((V)*sizeof(int));

  for(unsigned src=0; src<V; ++src) {
    hops_from_source = 0; // keeps track of the number of hops from source in the current iteration.
    initialize<<<num_blocks, block_size>>>(d_sigma, d_delta, d_level, d_hops_from_source, src, V);
    long k=0;
    //FORWARD PASS
    do{

      finished=true;
      cudaMemcpy(d_finished,&finished, sizeof(bool) ,cudaMemcpyHostToDevice);

      //~ cudaMemset(d_finished,true,sizeof(bool));

      bc_forward_pass<<<num_blocks, block_size>>>(d_offset, d_edgeList,d_edgeLen, d_sigma, d_level, d_hops_from_source, V, d_finished);

      cudaDeviceSynchronize();

      ++hops_from_source; // updating the level to process in the next iteration
//     gpuErrchk(cudaMemcpy(d_hops_from_source, &hops_from_source, sizeof(hops_from_source), cudaMemcpyHostToDevice));
      incHop<<<1,1>>>(d_hops_from_source);

      k++;
      cudaMemcpy(&finished,d_finished, sizeof(bool) ,cudaMemcpyDeviceToHost);
      //~ std::cout<< "SRC:"<< src <<" Fin? "<< (finished?"True":"False") << '\n';
    }while(!finished);


    cudaMemcpy(level,d_level , sizeof(int) * (V), cudaMemcpyDeviceToHost);
    std::cout<< "SRC:"<< src << " iters:" << k << " Hops:"<<hops_from_source<< '\n';
    for (int i = 0; i <V; i++)
      printf("%d %d\n", i, level[i]);
    return;


    hops_from_source--;
    cudaMemcpy(d_hops_from_source, &hops_from_source, sizeof(hops_from_source), cudaMemcpyHostToDevice);

    //BACKWARD PASS
    while(hops_from_source > 1) {
      bc_backward_pass<<<num_blocks, block_size>>>(d_offset, d_edgeList,d_edgeLen, d_sigma, d_delta, d_nodeBC, d_level, d_hops_from_source, V);
      --hops_from_source;
      cudaMemcpy(d_hops_from_source, &hops_from_source, sizeof(hops_from_source), cudaMemcpyHostToDevice);
    }

    accumulate_bc<<<num_blocks, block_size>>>(d_delta, d_nodeBC, d_level, src, V);
    cudaDeviceSynchronize();
  }

  double* nodeBC;
  nodeBC = (double *)malloc( (V)*sizeof(double));

  cudaMemcpy(nodeBC,d_nodeBC , sizeof(double) * (V), cudaMemcpyDeviceToHost);

  cudaEventRecord(stop,0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&milliseconds, start, stop);
  printf("GPU Time: %.6f ms\n", milliseconds);
  //~ printf("GPU Time: %.6f ms \nIterations:%d\n", milliseconds);

   for (int i = 0; i <V; i++)
     printf("%d %lf\n", i, nodeBC[i]);

    cudaCheckError();


  //~ char *outputfilename = "output_generated.txt";
  //~ FILE *outputfilepointer;
  //~ outputfilepointer = fopen(outputfilename, "w");
  //~ for (int i = 0; i <V; i++)
  //~ {
    //~ fprintf(outputfilepointer, "%d  %d\n", i, dist[i]);
  //~ }
  //~ Let's add fclose!

}


// driver program to test above function
int main(int argc , char ** argv)
{
  graph G(argv[1]);
  G.parseGraph();

  int V = G.num_nodes();
//---------------------------------------//
  printf("#nodes:%d\n",V);
//-------------------------------------//
 int E = G.num_edges();

 //---------------------------------------//
  printf("#edges:%d\n",E);
//-------------------------------------//

  int* edgeLen = G.getEdgeLen();

  int *OA;
  int *edgeList;
  int *edgeLength;

   OA = (int *)malloc( (V+1)*sizeof(int));
   edgeList = (int *)malloc( (E)*sizeof(int));
   edgeLength = (int *)malloc( (E)*sizeof(int));

  for(int i=0; i<= V; i++) {
    int temp = G.indexofNodes[i];
    OA[i] = temp;
  }

  for(int i=0; i< E; i++) {
    int temp = G.edgeList[i];
    edgeList[i] = temp;
    temp = edgeLen[i];
    edgeLength[i] = temp;
  }


    //~ cudaEvent_t start, stop; // should not be here!
    //~ cudaEventCreate(&start);
    //~ cudaEventCreate(&stop);
    //~ float milliseconds = 0;
    //~ cudaEventRecord(start,0);

    BC(OA,edgeList, edgeLength, V,E);
    //~ cudaDeviceSynchronize();

    //~ cudaEventRecord(stop,0);
    //~ cudaEventSynchronize(stop);
    //~ cudaEventElapsedTime(&milliseconds, start, stop);
    //~ printf("Time taken by function to execute is: %.6f ms\n", milliseconds);


  return 0;

}


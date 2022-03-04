#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <cuda.h>
#include <climits>
#include "graph.hpp"

#define cudaCheckError() {                                             \
 cudaError_t e=cudaGetLastError();                                     \
 if(e!=cudaSuccess) {                                                  \
   printf("Cuda failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(e)); \
   exit(0);                                                            \
 }                                                                     \
}

// we need to push this inits to our library.cuda file
template <typename T>
__global__ void initKernel0(T* init_array, T id, T init_value) { // MOSTLY 1 thread kernel
  init_array[id]=init_value;
}


template <typename T>
__global__ void initKernel(unsigned V, T* init_array, T init_value) {
  unsigned id = threadIdx.x + blockDim.x * blockIdx.x;
  if(id < V) {
    init_array[id]=init_value;
  }
}

template <typename T1, typename T2>
__global__ void initKernel2( unsigned V, T1* init_array1, T1 init_value1, T2* init_array2, T2 init_value2)  {
  unsigned id = threadIdx.x + blockDim.x * blockIdx.x;
  if(id < V) {
    init_array1[id]=init_value1;
    init_array2[id]=init_value2;
  }
}

__global__ void Compute_SSSP_kernel(int * gpu_offset_array ,
  int * gpu_edge_list ,
  int* gpu_weight,
  int * gpu_dist,
  int V,
  bool * gpu_modified_prev,
  bool * gpu_modified_next,
  bool * gpu_finished)
{
  unsigned int id = threadIdx.x + (blockDim.x * blockIdx.x);
  unsigned int v = id;
  if (id < V)
  {
    if (gpu_modified_prev[id] ){

      for (int edge = gpu_offset_array[id]; edge < gpu_offset_array[id+1]; edge ++)
      {
        int nbr = gpu_edge_list[edge] ;
        int e = edge;
        int dist_new;
        if(gpu_dist[id] != INT_MAX)
          dist_new = gpu_dist[v] + gpu_weight[e];

        if (gpu_dist[nbr] > dist_new)
        {
          atomicMin(&gpu_dist[nbr] , dist_new);
          gpu_modified_next[nbr]=true;
          *gpu_finished = false ;
        }
      }
    }
  }

}
  void SSSP(int* offset_array , int* edge_list , int* cpu_edge_weight  , int src ,int V, int E , bool printAns)
{
  //CSR VARS
  int * gpu_offset_array;
  cudaMalloc(&gpu_offset_array,sizeof(int) *(1+V));
  int * gpu_edge_list;
  cudaMalloc(&gpu_edge_list,sizeof(int) *(E));
  int * gpu_edge_weight;
  cudaMalloc(&gpu_edge_weight,sizeof(int) *(E));

  // RESUT VAR
  int * gpu_dist;
  cudaMalloc(&gpu_dist,sizeof(int) *(V));

  // EXTRA VARS
  bool * gpu_modified_prev;
  cudaMalloc(&gpu_modified_prev,sizeof(bool) *(V));
  bool * gpu_modified_next;
  cudaMalloc(&gpu_modified_next,sizeof(bool) *(V));
  bool * gpu_finished;
  cudaMalloc(&gpu_finished,sizeof(bool) *(1));

  unsigned int block_size=V;
  unsigned int num_blocks=1;

  // Launch Config is ready!
  if ( V > 512 ) {
    block_size = 512;
    num_blocks = (V+block_size-1) / block_size; // avoid ceil fun call
  }
  std::cout<< "nBlock:" << num_blocks  << '\n';
  std::cout<< "threadsPerBlock:" << block_size  << '\n';
  // This comes from attach propoety
  //~ with two init1
  //~ initKernel<int> <<<num_blocks,block_size>>>(V,gpu_dist, INT_MAX);
  //~ initKernel<bool><<<num_blocks,block_size>>>(V,gpu_modified_prev, false);
  //~ with single init2
  initKernel2<int,bool> <<<num_blocks,block_size>>>(V,gpu_dist, INT_MAX,gpu_modified_prev, false);

  // This comes from DSL. Single thread kernel
  initKernel0<int> <<<1,1>>>(gpu_dist, src,0);
  initKernel0<bool> <<<1,1>>>(gpu_modified_prev, src,true);

  cudaEvent_t start, stop; ///TIMER START
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  float milliseconds = 0;
  cudaEventRecord(start,0);

  // CSR
  cudaMemcpy (gpu_offset_array  , offset_array    , sizeof(int)   *(1+V), cudaMemcpyHostToDevice);
  cudaMemcpy (gpu_edge_list     , edge_list       , sizeof(int)   *(E)  , cudaMemcpyHostToDevice);
  cudaMemcpy (gpu_edge_weight   , cpu_edge_weight , sizeof(int)   *(E)  , cudaMemcpyHostToDevice);


  // COMES FROM DSL so CPU VARIABLE
  bool* finished = new bool[1];
  *finished = false; // to kick start

  int k =0; /// We need it only for count iterations attached to FIXED pt
  while ( !(*finished) )
  {
    //~ finished[0]=true;   /// I guess  we do not need this line overwrritten in memcpy below
    initKernel<bool> <<< 1, 1>>>(1, gpu_finished, true);

    Compute_SSSP_kernel<<<num_blocks , block_size>>>(gpu_offset_array,gpu_edge_list, gpu_edge_weight ,gpu_dist, V , gpu_modified_prev, gpu_modified_next, gpu_finished);

    initKernel<bool><<<num_blocks,block_size>>>(V, gpu_modified_prev, false);

    cudaMemcpy(finished, gpu_finished,  sizeof(bool) *(1), cudaMemcpyDeviceToHost);

    bool *tempModPtr  = gpu_modified_next;
    gpu_modified_next = gpu_modified_prev;
    gpu_modified_prev = tempModPtr;

    ++k;
    if(k==V)    // NEED NOT GENERATE. DEBUG Only
    {
      std::cout<< "THIS SHOULD NEVER HAPPEN" << '\n';
      exit(0);
    }
  }

  //STOP TIMER
  cudaEventRecord(stop,0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&milliseconds, start, stop);
  printf("GPU Time: %.6f ms \nIterations:%d\n", milliseconds,k);

  cudaCheckError()

  // PRINT THE OUTPUT vars
  if(printAns) {
    int* dist=new int[V];
    cudaMemcpy(dist,gpu_dist , sizeof(int) * (V), cudaMemcpyDeviceToHost);
    for (int i = 0; i <V; i++) {
      printf("%d %d\n", i, dist[i]);
    }
  }

  //~ char *outputfilename = "output_generated.txt";
  //~ FILE *outputfilepointer;
  //~ outputfilepointer = fopen(outputfilename, "w");
  //~ for (int i = 0; i <V; i++)
  //~ {
    //~ fprintf(outputfilepointer, "%d  %d\n", i, dist[i]);
  //~ }

}


// driver program to test above function
int main(int argc , char ** argv)
{
  graph G(argv[1]);
  G.parseGraph();

  bool printAns =false;
  if(argc>2)
    printAns=true;

  int V = G.num_nodes();
//---------------------------------------//
  printf("#nodes:%d\n",V);
//-------------------------------------//
 int E = G.num_edges();

 //---------------------------------------//
  printf("#edges:%d\n",E);
//-------------------------------------//

  int* edge_weight = G.getEdgeLen();

  //~ int* dist;

  int src=0;

  int *offset_array;
  int *edge_list;
  int *cpu_edge_weight;


   offset_array = (int *)malloc( (V+1)*sizeof(int));
   edge_list = (int *)malloc( (E)*sizeof(int));
   cpu_edge_weight = (int *)malloc( (E)*sizeof(int));
   //~ dist = (int *)malloc( (V)*sizeof(int));

  for(int i=0; i<= V; i++) {
    int temp = G.indexofNodes[i];
    offset_array[i] = temp;
  }

  for(int i=0; i< E; i++) {
    int temp = G.edgeList[i];
    edge_list[i] = temp;
    temp = edge_weight[i];
    cpu_edge_weight[i] = temp;
  }

  //~ for(int i=0; i< E; i++) {
    //~ int temp = edge_weight[i];
    //~ cpu_edge_weight[i] = temp;
  //~ }


    //~ cudaEvent_t start, stop;
    //~ cudaEventCreate(&start);
    //~ cudaEventCreate(&stop);
    //~ float milliseconds = 0;
    //~ cudaEventRecord(start,0);

    SSSP(offset_array,edge_list, cpu_edge_weight ,src, V,E,printAns);
    //~ cudaDeviceSynchronize();

    //~ cudaEventRecord(stop,0);
    //~ cudaEventSynchronize(stop);
    //~ cudaEventElapsedTime(&milliseconds, start, stop);
    //~ printf("Time taken by function to execute is: %.6f ms\n", milliseconds);


  return 0;

}


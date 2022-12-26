#include <stdio.h>
#include <stdlib.h>
#include <limits.h>  //Please give a space before <  or "
#include <cuda.h>
#include <climits>
#include "graph.hpp"

// we need to push this to our library.cuda file
template <typename T>
__global__ void initKernel0(T* init_array, T id, T init_value) { // MOSTLY 1 thread kernel
  init_array[id]=init_value;
}
// we need to push this to our library.cuda file
template <typename T>
__global__ void initKernel1(unsigned V, T* init_array, T init_value) {
  unsigned id = threadIdx.x + blockDim.x * blockIdx.x;
  if(id < V) {
    init_array[id]=init_value;
  }
}
// we need to push this to our library.cuda file
template <typename T1, typename T2>
__global__ void initKernel2( unsigned V, T1* init_array1, T1 init_value1, T2* init_array2, T2 init_value2)  {
  unsigned id = threadIdx.x + blockDim.x * blockIdx.x;
  if(id < V) {
    init_array1[id]=init_value1;
    init_array2[id]=init_value2;
  }
}



__global__ void Compute_SSSP_kernel(int * gpu_offset_array , int * gpu_edge_list , int* gpu_weight, int * gpu_dist ,
  /* int src , */ // src is never used below
  int V,
  /* int MAX_VAL , */ //saved 4 Bytes per thread!!
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

        int dist_new ;
        if(gpu_dist[id] != INT_MAX)
          dist_new = gpu_dist[v] + gpu_weight[e];
        //~ int dist_new = gpu_dist[v] + gpu_weight[e];
        if (gpu_dist[nbr] > dist_new)
        //~ if (gpu_dist[id] != MAX_VAL && gpu_dist[nbr] > dist_new)
        {
          atomicMin(&gpu_dist[nbr] , dist_new);
          gpu_modified_next[nbr]=true;
          //~ gpu_finished[0] = false;
          *gpu_finished = false    ;      // nice if we use this way
        }
      }
    }
  }

}
  void SSSP(int* offset_array , int* edge_list , int* cpu_edge_weight  , int src ,int V, int E )
{
  //~ int MAX_VAL = 2147483647 ; // we do not need this variable!
  // G or CSR vars
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
   //~ if(V <= 1024)    // this can be do with single if
   //~ {
    //~ block_size  = V;
    //~ block_size  = 1; // why do we set block_size repeatedly
  //~ }
  //~ else
  //~ {
    //~ block_size = 1024;
    //~ num_blocks = ceil(((float)V) / block_size);
  //~ }

  // Launch Config is ready!
  if ( V > 512 ) {
    block_size = 512;
    num_blocks = (V+block_size-1) / block_size; // avoid ceil fun call
  }

  //~ int* dist=new int[V];
  //~ bool* modified=new bool[V];
  //~ for (int t = 0; t < V; t ++)
  //~ {
    //~ dist[t] = INT_MAX;
    //~ modified[t] = false;
  //~ }

  // This comes from attach propoety
  initKernel1<bool><<<num_blocks,block_size>>>(V,gpu_modified_prev, false);
  initKernel1<int> <<<num_blocks,block_size>>>(V,gpu_dist, false);


  //~ modified[src] = true;
  //~ dist[src] = 0;          // This comes from DSL
  initKernel0<int> <<<1,1>>>(gpu_dist, src,0);
  initKernel0<bool> <<<1,1>>>(gpu_modified_prev, src,true);

  cudaEvent_t start, stop; ///TIMER START
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  float milliseconds = 0;
  cudaEventRecord(start,0);

  cudaMemcpy (gpu_offset_array  , offset_array    , sizeof(int)   *(1+V), cudaMemcpyHostToDevice);
  cudaMemcpy (gpu_edge_list     , edge_list       , sizeof(int)   *(E)  , cudaMemcpyHostToDevice);
  cudaMemcpy (gpu_edge_weight   , cpu_edge_weight , sizeof(int)   *(E)  , cudaMemcpyHostToDevice);

  //~ cudaMemcpy (gpu_dist          , dist        , sizeof(int)   *(V)  , cudaMemcpyHostToDevice);  // i guess we do not have to do this
  //~ cudaMemcpy (gpu_finished      , finished    , sizeof(bool) *(1)   , cudaMemcpyHostToDevice); // This gets overwritten in fixed pt init kerkel1 line
  //~ cudaMemcpy (gpu_modified_prev , modified    , sizeof(bool) *(V)   , cudaMemcpyHostToDevice); // we do not need this


  // COMES FROM DSL so CPU VARIABLE
  bool* finished = new bool[1];
  *finished = false; // to kick start

  int k =0; /// We need it only for count iterations

  while ( *finished ) /// *finished should be good
  {
    //~ finished[0]=true;   /// I guess  we do not need this line overwrritten in memcpy below
    initKernel1<bool> <<< 1, 1>>>(1, gpu_finished, true);

    Compute_SSSP_kernel<<<num_blocks , block_size>>>(gpu_offset_array,gpu_edge_list, gpu_edge_weight ,gpu_dist,/*src,*/ V /*,MAX_VAL*/ , gpu_modified_prev, gpu_modified_next, gpu_finished);

    initKernel1<bool><<<num_blocks,block_size>>>(V, gpu_modified_prev, false);

    cudaMemcpy(finished, gpu_finished,  sizeof(bool) *(1), cudaMemcpyDeviceToHost); //added this.

    bool *tempModPtr  = gpu_modified_next;
    gpu_modified_next = gpu_modified_prev;
    gpu_modified_prev = tempModPtr;

    ++k;
    //~ if(k==V)
    //~ {
      //~ break;
    //~ }
  }

  //STOP TIMER
  cudaEventRecord(stop,0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&milliseconds, start, stop);
  printf("GPU Time: %.6f ms \nIterations:%d\n", milliseconds,k);

  // PRINT THE OUTPUT vars
  int* dist=new int[V];
  cudaMemcpy(dist,gpu_dist , sizeof(int) * (V), cudaMemcpyDeviceToHost);
  for (int i = 0; i <V; i++) {
    printf("%d %d\n", i, dist[i]);
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

    SSSP(offset_array,edge_list, cpu_edge_weight ,src, V,E);
    //~ cudaDeviceSynchronize();

    //~ cudaEventRecord(stop,0);
    //~ cudaEventSynchronize(stop);
    //~ cudaEventElapsedTime(&milliseconds, start, stop);
    //~ printf("Time taken by function to execute is: %.6f ms\n", milliseconds);


  return 0;

}


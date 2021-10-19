#include<stdio.h>
#include<stdlib.h>
#include<limits.h>
#include<cuda.h>
#include"graph.hpp"


template <typename T>
__global__ void initKernel(unsigned V, T* init_array, T initVal)
{
  unsigned id = threadIdx.x + blockDim.x * blockIdx.x;
  if(id < V)
  {
    init_array[id]=initVal;
  }
}
__global__ void Compute_SSSP_kernel(int * gpu_OA , int * gpu_edgeList , int* gpu_weight, int * gpu_dist , int src ,int V, int MAX_VAL , bool * gpu_modified_prev,
bool * gpu_modified_next, bool * gpu_finished)
{
  unsigned int id = threadIdx.x + (blockDim.x * blockIdx.x);
  unsigned int v = id;
  if (id < V)
  {
    if (gpu_modified_prev[id] ){
      for (int edge = gpu_OA[id]; edge < gpu_OA[id+1]; edge ++)
      {
        int nbr = gpu_edgeList[edge] ;
        int e = edge;

        int dist_new ;
        if(gpu_dist[id] != MAX_VAL)
          dist_new = gpu_dist[v] + gpu_weight[e];
        //~ int dist_new = gpu_dist[v] + gpu_weight[e];
        if (gpu_dist[nbr] > dist_new)
        //~ if (gpu_dist[id] != MAX_VAL && gpu_dist[nbr] > dist_new)
        {
          atomicMin(&gpu_dist[nbr] , dist_new);
          gpu_modified_next[nbr]=true;
          gpu_finished[0] = false;
        }
      }
    }
  }

}
  void SSSP(int * OA , int * edgeList , int* cpu_edgeLen  , int src ,int V, int E )
{
  int MAX_VAL = 2147483647 ;
  int * gpu_edgeList;
  int * gpu_edgeLen;
  int * gpu_dist;
  int * gpu_OA;
  bool * gpu_modified_prev;
  bool * gpu_modified_next;
  bool * gpu_finished;


  cudaMalloc(&gpu_OA,sizeof(int) *(1+V));
  cudaMalloc(&gpu_edgeList,sizeof(int) *(E));
  cudaMalloc(&gpu_edgeLen,sizeof(int) *(E));
  cudaMalloc(&gpu_dist,sizeof(int) *(V));
  cudaMalloc(&gpu_modified_prev,sizeof(bool) *(V));
  cudaMalloc(&gpu_modified_next,sizeof(bool) *(V));
  cudaMalloc(&gpu_finished,sizeof(bool) *(1));

  unsigned int block_size;
  unsigned int num_blocks;
   if(V <= 1024)
   {
    block_size  = V;
    block_size  = 1;
  }
  else
  {
    block_size = 1024;
    num_blocks = ceil(((float)V) / block_size);
  }
  bool * finished = new bool[1];
  finished[0] = false;
  int* dist=new int[V];
  bool* modified=new bool[V];
  for (int t = 0; t < V; t ++)
  {
    dist[t] = INT_MAX;
    modified[t] = false;
  }
  modified[src] = true;
  dist[src] = 0;

  cudaEvent_t start, stop; ///TIMER START
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  float milliseconds = 0;
  cudaEventRecord(start,0);

  cudaMemcpy (gpu_OA, OA, sizeof(int) *(1+V) ,cudaMemcpyHostToDevice);
  cudaMemcpy (gpu_edgeList, edgeList, sizeof(int) *(E) ,cudaMemcpyHostToDevice);
  cudaMemcpy (gpu_edgeLen, cpu_edgeLen , sizeof(int) *(E) ,cudaMemcpyHostToDevice);
  cudaMemcpy (gpu_dist, dist, sizeof(int) *(V) ,cudaMemcpyHostToDevice);
  cudaMemcpy (gpu_modified_prev, modified , sizeof(bool) *(V) ,cudaMemcpyHostToDevice);
  cudaMemcpy (gpu_finished, finished , sizeof(bool) *(1) ,cudaMemcpyHostToDevice);

  int k =0;

  while ( !finished[0] )
  {
    finished[0]=true;//ADDED
    initKernel<bool> <<< 1, 1>>>(1, gpu_finished, true);

    Compute_SSSP_kernel<<<num_blocks , block_size>>>(gpu_OA,gpu_edgeList, gpu_edgeLen ,gpu_dist,src, V ,MAX_VAL , gpu_modified_prev, gpu_modified_next, gpu_finished);
    //~ cudaDeviceSynchronize();
    initKernel<bool><<<num_blocks,block_size>>>(V, gpu_modified_prev, false);
    //~ cudaDeviceSynchronize();

    cudaMemcpy(finished, gpu_finished,  sizeof(bool) *(1), cudaMemcpyDeviceToHost); //added this.

    bool *tempModPtr  = gpu_modified_next;
    gpu_modified_next = gpu_modified_prev;
    gpu_modified_prev = tempModPtr;
    ++k;
    if(k==V)
    {
      break;
    }
  }



  cudaMemcpy(dist,gpu_dist , sizeof(int) * (V), cudaMemcpyDeviceToHost);

  cudaEventRecord(stop,0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&milliseconds, start, stop);
  printf("GPU Time: %.6f ms \nIterations:%d\n", milliseconds,k);

   for (int i = 0; i <V; i++)
   {
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

  int* edgeLen = G.getEdgeLen();

  //~ int* dist;

  int src=0;

  int *OA;
  int *edgeList;
  int *cpu_edgeLen;


   OA = (int *)malloc( (V+1)*sizeof(int));
   edgeList = (int *)malloc( (E)*sizeof(int));
   cpu_edgeLen = (int *)malloc( (E)*sizeof(int));
   //~ dist = (int *)malloc( (V)*sizeof(int));

  for(int i=0; i<= V; i++) {
    int temp = G.indexofNodes[i];
    OA[i] = temp;
  }

  for(int i=0; i< E; i++) {
    int temp = G.edgeList[i];
    edgeList[i] = temp;
  }

  for(int i=0; i< E; i++) {
    int temp = edgeLen[i];
    cpu_edgeLen[i] = temp;
  }


    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float milliseconds = 0;
    cudaEventRecord(start,0);

    SSSP(OA,edgeList, cpu_edgeLen ,src, V,E);
    //~ cudaDeviceSynchronize();

    //~ cudaEventRecord(stop,0);
    //~ cudaEventSynchronize(stop);
    //~ cudaEventElapsedTime(&milliseconds, start, stop);
    //~ printf("Time taken by function to execute is: %.6f ms\n", milliseconds);


  return 0;

}


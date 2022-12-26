/*

Tested for 2 million nodes


*/

#include <limits.h>
#include <stdio.h>
#include <cuda.h>
#include "graph.hpp"

#include <sys/time.h>


__global__ void SSSP_Kernel(int * gpu_OA , int * gpu_edgeList , int* weight, int * gpu_dist , int src ,int V, int MAX_VAL) {

unsigned int id = threadIdx.x + (blockDim.x * blockIdx.x);
 

 for (int edge = gpu_OA[id]; edge < gpu_OA[id+1]; edge ++) 
 {
          int nbr = gpu_edgeList[edge] ;
          int e = edge;
          int dist_new;
          
           dist_new = gpu_dist[id] + weight[e];
           
           
          
          if ( gpu_dist[id] != MAX_VAL  && gpu_dist[nbr] > dist_new)
          {
            //bool modified_new = true;
           // omp_set_lock(&(lock[nbr])) ;
           // if (gpu_dist[nbr] > dist_new)
            //{
            
               atomicMin(&gpu_dist[nbr] , dist_new);
               
              // printf("%d  %d\n", gpu_dist[nbr],dist_new);
             
             
              //gpu_dist[nbr] = dist_new;
              //modified[nbr] = modified_new;
            //}
            //omp_unset_lock(&(lock[nbr]));
            
          }
   }

} 

void SSSP(int * OA , int * edgeList , int* cpu_edgeLen , int * dist , int src ,int V, int E)
{
  int MAX_VAL = 2147483647 ;
  
  struct timeval t1, t2;
  
  clock_t start , end;
  double gpu_time_use;
  
  int *gpu_edgeList;
  int *gpu_edgeLen;
  int *gpu_dist;
  int *gpu_OA;

  cudaMalloc( &gpu_OA, sizeof(int) * (1+V) );
  cudaMalloc( &gpu_edgeList, sizeof(int) * (E) );
  cudaMalloc( &gpu_edgeLen, sizeof(int) * (E) );
  cudaMalloc( &gpu_dist, sizeof(int) * (V) );
  
  unsigned int block_size;
	unsigned int num_blocks;
 
 
  
  if(V <= 1024)
	{
		block_size = V;
		num_blocks = 1;
	}
	else
	{
		block_size = 1024;
		num_blocks = ceil(((float)V) / block_size);
			
	}
  
  bool* modified=new bool[V];
  
  

  for (int t = 0; t < V; t ++) 
 {
    dist[t] = INT_MAX;
    modified[t] = false;
  }
  
  
  modified[src] = true;
  dist[src] = 0;
  bool finished = false;
  
  cudaMemcpy(gpu_OA, OA, sizeof(int) * (1+V), cudaMemcpyHostToDevice);
  cudaMemcpy(gpu_edgeList, edgeList, sizeof(int) * (E), cudaMemcpyHostToDevice);
  cudaMemcpy(gpu_edgeLen,cpu_edgeLen , sizeof(int) * (E), cudaMemcpyHostToDevice);
  cudaMemcpy(gpu_dist,dist , sizeof(int) * (V), cudaMemcpyHostToDevice);
  
  
 // while ( !finished )
 // {
      //if (modified[id] == true ){
      //  modified[index] = false;
      
     // gettimeofday(&t1, 0);
      
      start = clock();
      
      for (int i = 0; i <= V - 1; i++) 
        {
            SSSP_Kernel<<<num_blocks , block_size>>>(gpu_OA,gpu_edgeList, gpu_edgeLen ,gpu_dist,src, V ,MAX_VAL);
            cudaDeviceSynchronize();
        }
        
        
      end = clock();
      //gettimeofday(&t2, 0);
      
      
     gpu_time_use = ((double)(end-start)) /CLOCKS_PER_SEC;

     printf("the time taken  is %f secs\n",gpu_time_use);
     
  
      
   // }
    //bool modified_fp = false ;
   // for (int v = 0; v < V; v ++) 
    //modified_fp = modified_fp || modified[v] ;
    //finished = !modified_fp ;
  //}
  
   cudaMemcpy(dist,gpu_dist , sizeof(int) * (V), cudaMemcpyDeviceToHost);
   
   
   printf("\n");
   
   
   for (int i = 0; i <V; i++)
   {
      printf("%d  %d\n", i, dist[i]);
   }
   
 
  char *outputfilename = "output.txt";
  FILE *outputfilepointer;
  outputfilepointer = fopen(outputfilename, "w");
  
  fprintf(outputfilepointer, "%d\n", gpu_time_use);


  for (int i = 0; i <V; i++)
  {
    fprintf(outputfilepointer, "%d  %d\n", i, dist[i]);
  }
 
  
}

// driver program to test above function
int main(int argc , char ** argv)
{
  graph G("/home/ashwina/email-Eu-core.txt");
  G.parseGraph();
   
  int V = G.num_nodes();
//---------------------------------------//   
   //printf("number of nodes =%d\n",V);
//-------------------------------------// 
 int E = G.num_edges();
 
 //---------------------------------------//   
  // printf("number of edges =%d\n",E);
//-------------------------------------// 
  
  int* edgeLen = G.getEdgeLen();
  
  int* dist;

  int src=0;
  
  int *OA;
  int *edgeList;
  int *cpu_edgeLen;
  
  
   OA = (int *)malloc( (V+1)*sizeof(int));
   edgeList = (int *)malloc( (E)*sizeof(int));
   cpu_edgeLen = (int *)malloc( (E)*sizeof(int));
   dist = (int *)malloc( (V)*sizeof(int));
  
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

  SSSP(OA,edgeList, cpu_edgeLen ,dist,src, V,E);
 

  return 0;

}

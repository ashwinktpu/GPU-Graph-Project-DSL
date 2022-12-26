#include<stdio.h>
#include<stdlib.h>
#include<limits.h>
#include<cmath>
#include<algorithm>
#include<cuda.h>
#include"graph.hpp"


__device__ int triangle_count = 0;

  __device__ bool check_if_nbr(int s, int d, int * gpu_OA, int *gpu_edgeList)   //we can move this to graph.hpp file
    {
      int startEdge=gpu_OA[s];
      int endEdge=gpu_OA[s+1]-1;


      if(gpu_edgeList[startEdge]==d)
          return true;
      if(gpu_edgeList[endEdge]==d)
         return true;   

       int mid = (startEdge+endEdge)/2;

      while(startEdge<=endEdge)
        {
       
          if(gpu_edgeList[mid]==d)
             return true;

          if(d<gpu_edgeList[mid])
             endEdge=mid-1;
          else
            startEdge=mid+1;   
          
          mid = (startEdge+endEdge)/2;

        }
      
      return false;

    }




__global__ void kernel(int * gpu_OA, int * gpu_edgeList, int V, int E) 

{
 
    unsigned int id = threadIdx.x + (blockDim.x * blockIdx.x);
      
     // printf("id = %d",id);
      
      for (int edge = gpu_OA[id]; edge < gpu_OA[id+1]; edge ++) 
    { 
      int u =  gpu_edgeList[edge] ;
      if (u < id )
      {
        for (int edge = gpu_OA[id]; edge <  gpu_OA[id+1]; edge ++) 
         { 
          int w = gpu_edgeList[edge] ;
          if (w > id )
          {
            if (check_if_nbr(u, w,gpu_OA,gpu_edgeList ) )
            {
              atomicAdd(&triangle_count ,1);
            }
          }
        }
      }
    }
    //printf("TC = %d",triangle_count);
}


void Compute_TC(int * OA, int * edgeList, int V, int E)
{
  
 // printf("hi from function\n");
   int *gpu_edgeList;
   int *gpu_OA;
  // printf("V inside fun =%d",V);
  
  cudaMalloc( &gpu_OA, sizeof(int) * (1+V) );
  cudaMalloc( &gpu_edgeList, sizeof(int) * (E) );
  
  
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
 
  cudaMemcpy(gpu_OA, OA, sizeof(int) * (1+V), cudaMemcpyHostToDevice);
  cudaMemcpy(gpu_edgeList, edgeList, sizeof(int) * (E), cudaMemcpyHostToDevice);
  
  printf("before kernel\n");
  
  
  kernel<<<num_blocks,block_size>>>(gpu_OA, gpu_edgeList, V, E );
  cudaDeviceSynchronize();

  printf("after kernel\n");
  int count;
  cudaMemcpyFromSymbol(&count, triangle_count, sizeof(int));
  
  printf("TC = %d",count);
 }



 int main()
{

  graph G("/home/ashwina/roadNet-CA.txt");  //this will be changed
  G.parseGraph();
  
  int V = G.num_nodes();
  
 // printf("number pf nodes =%d",V);
  
  int E = G.num_edges();
  
//  printf("number pf edges =%d",E);
  

  int *OA;
  int *edgeList;
  
  
   OA = (int *)malloc( (V+1)*sizeof(int));
   edgeList = (int *)malloc( (E)*sizeof(int));
  
    
  for(int i=0; i<= V; i++) {
    int temp = G.indexofNodes[i];
    OA[i] = temp;
  }
  
  for(int i=0; i< E; i++) {
    int temp = G.edgeList[i];
    edgeList[i] = temp;
  }
  
 
  
  Compute_TC(OA, edgeList, V,E);

}

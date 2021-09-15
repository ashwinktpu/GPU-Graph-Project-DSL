#include<stdio.h>
#include<stdlib.h>
#include<limits.h>
#include<cmath>
#include<omp.h>
#include"graph.hpp"

void __global__  Compute_PR_Kernel(int * gpu_rev_OA, int * gpu_OA, int * gpu_srcList , float * gpu_node_pr , int V, int E , float beta, float delta, int maxIter) 
{

      unsigned int id = threadIdx.x + (blockDim.x * blockIdx.x);
      float sum=0.0;
      int iterCount=0;
      float diff;
      if (id < V) 
      {

      for(int edge= gpu_rev_OA[id] ;edge< gpu_rev_OA[id+1];edge++)
         {
           int nbr=  gpu_srcList[edge];
           sum =sum + gpu_node_pr[nbr]/(gpu_OA[nbr+1]- gpu_OA[nbr]);
           
         }

         float val=(1-delta)/V + delta * sum;
         float temp = std::fabs(val-gpu_node_pr[id]);
         gpu_node_pr[id]=val;
         //printf("val = %f ",val);
}
}


void Compute_PR(int * rev_OA, int * OA, int * cpu_srcList , float * node_pr , int V, int E)
{
  
  int *gpu_rev_OA;
  int *gpu_srcList;
  int * gpu_OA;
  float * gpu_node_pr;
  
  
  cudaMalloc( &gpu_rev_OA, sizeof(int) * (1+V) ); //rev_OA
  cudaMalloc( &gpu_OA, sizeof(int) * (1+V) );   //OA
  cudaMalloc( &gpu_srcList, sizeof(int) * (E) ); //nbr
  cudaMalloc( &gpu_node_pr, sizeof(float) * (V) ); //output
  
  
  unsigned int block_size;
	unsigned int num_blocks;
 
   for(int i=0; i< V; i++)
     {
         node_pr[i]= 1.0/V;
     }
   
  
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
 
  cudaMemcpy(gpu_rev_OA, rev_OA, sizeof(int) * (1+V), cudaMemcpyHostToDevice);
  cudaMemcpy(gpu_OA, OA, sizeof(int) * (1+V), cudaMemcpyHostToDevice);
  cudaMemcpy(gpu_srcList, cpu_srcList, sizeof(int) * (E), cudaMemcpyHostToDevice);
  cudaMemcpy(gpu_node_pr, node_pr , sizeof(float) * (V), cudaMemcpyHostToDevice);
  
  
  
  float beta = 0.001;
  float delta = 0.85;
  int maxIter = 100;
  
  int iterCount=0;
  float diff;
  
  
 // cudaMemcpy(node_pr,gpu_node_pr , sizeof(float) * (V), cudaMemcpyDeviceToHost);
 
  do
  {
   diff=0.0;
  
  Compute_PR_Kernel<<<num_blocks , block_size>>>(gpu_rev_OA, gpu_OA, gpu_srcList, gpu_node_pr , V,E,  0.001, 0.85, 1);
 // diff=diff+temp;
  printf("iterCount %d diff %f\n",iterCount,diff);
  iterCount=iterCount+1;
  
  }while ((iterCount < maxIter));
  
  printf("iteration took %d\n",iterCount);
  
  
  
  
  cudaMemcpy(node_pr,gpu_node_pr , sizeof(float) * (V), cudaMemcpyDeviceToHost);
  printf("\n");
  
  for (int i = 0; i <V; i++)
   {
      printf("%d  %0.9lf\n", i, node_pr[i]);
   }
  
  //output
  char *outputfilename = "outputN.txt";
  FILE *outputfilepointer;
  outputfilepointer = fopen(outputfilename, "w");

  for (int i = 0; i < V; i++)
  {
    fprintf(outputfilepointer, "%d  %0.9lf\n", i, node_pr[i]);
  }
 }

 int main()
{

  graph G("/home/ashwina/cuda/final/input4.txt");
  G.parseGraph();
  
  int V = G.num_nodes();
  
  printf("number pf nodes =%d",V);
  
  int E = G.num_edges();
  
  printf("number pf edges =%d",E);
  
  

  float* node_pr;
  int *rev_OA;
  int *OA;
  int *cpu_srcList;
  
  
  
  node_pr = (float *)malloc( (V)*sizeof(float));
  rev_OA = (int *)malloc( (V+1)*sizeof(int));
  OA = (int *)malloc( (V+1)*sizeof(int));
  cpu_srcList = (int *)malloc( (E)*sizeof(int));
  
    
  for(int i=0; i<= V; i++) {
    int temp = G.rev_indexofNodes[i];
    rev_OA[i] = temp;
  }
  
  printf("\n");
  
   for(int i=0; i< E; i++) {
    int temp = G.srcList[i];
    cpu_srcList[i] = temp;
  }
  
 
  printf("\n");
  
   for(int i=0; i<= V; i++) {
    int temp = G.indexofNodes[i];
    OA[i] = temp;
  }
  
  Compute_PR(rev_OA, OA, cpu_srcList , node_pr , V, E);
}

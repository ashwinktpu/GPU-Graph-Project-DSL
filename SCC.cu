#include<stdio.h>
#include<cuda.h>


__global__ void trim1(const uint32_t *range, uint8_t *tags, const uint32_t *Fc, const uint32_t *Fr, const uint32_t *Bc, const uint32_t *Br, const uint32_t num_rows, bool volatile *terminate){

	uint32_t row = blockIdx.y * gridDim.x * blockDim.x + blockDim.x * blockIdx.x + threadIdx.x + 1;
	uint8_t myTag;

    if (row > num_rows || isRangeSet(myTag = tags[row]))
        return;

    uint32_t myRange = range[row];

	uint32_t cnt = Br[row + 1] - Br[row];
    const uint32_t *nbrs = &Bc[Br[row]];

	bool eliminate = true;
	for(uint32_t i = 0; i < cnt; i++){
	    uint32_t index = nbrs[i];

		if ( !isRangeSet(tags[index]) && range[index] == myRange){
			eliminate = false;
            break;
        }
	}

	if ( !eliminate ) {
		eliminate = true;
		cnt = Fr[row + 1] - Fr[row];
        nbrs = &Fc[Fr[row]];
			
		for(uint32_t i = 0; i < cnt; i++){
	        uint32_t index = nbrs[i];

			if ( !isRangeSet(tags[index]) && range[index] == myRange){
				eliminate = false;
                break;
            }
		}
	}

	if ( eliminate ) {
		rangeSet(&myTag);
        setTrim1(&myTag);
        tags[row] = myTag;
		*terminate = false;
	}
	return;
}


__global__ void fwd(const uint32_t *Fc, const uint32_t *Fr, const uint32_t *range, uint8_t *tags, const uint32_t num_rows, bool volatile *terminate){

	uint32_t row = blockIdx.y * gridDim.x * blockDim.x + blockDim.x * blockIdx.x + threadIdx.x + 1;
	uint8_t myTag;

    if (row > num_rows || isRangeSet(myTag = tags[row]) || isForwardPropagate(myTag) || !isForwardVisited(myTag))
        return;
	
    uint32_t myRange = range[row];
	uint32_t cnt = Fr[row + 1] - Fr[row];
    const uint32_t *nbrs = &Fc[Fr[row]];

	bool end = true;
	for ( uint32_t i = 0; i < cnt; i++ ) {
        uint32_t index = nbrs[i];
        uint8_t nbrTag = tags[index];

		if(isRangeSet(nbrTag) || isForwardVisited(nbrTag) || range[index] != myRange)
			continue;

		setForwardVisitedBit(&tags[index]);
		end = false;
	}
	setForwardPropagateBit(&tags[row]);
	if (!end)
		*terminate = false;
}




__global__ void bwd(const uint32_t *Bc, const uint32_t *Br, const uint32_t *range, uint8_t *tags, const uint32_t num_rows, bool volatile *terminate){

	uint32_t row = blockIdx.y * gridDim.x * blockDim.x + blockDim.x * blockIdx.x + threadIdx.x + 1;
	uint8_t myTag;

    if (row > num_rows || isRangeSet(myTag = tags[row]) || isBackwardPropagate(myTag) || !isBackwardVisited(myTag))
        return;

    uint32_t myRange = range[row];
	uint32_t cnt = Br[row + 1] - Br[row];
    const uint32_t *nbrs = &Bc[Br[row]];

	bool end = true;
	for ( uint32_t i = 0; i < cnt; i++ ) {
        uint32_t index = nbrs[i];
        uint8_t nbrTag = tags[index];

		if(isRangeSet(nbrTag) || isBackwardVisited(nbrTag) || range[index] != myRange )
			continue;

		setBackwardVisitedBit(&tags[index]);
		end = false;
	}
	setBackwardPropagateBit(&tags[row]);
	if (!end)
		*terminate = false;
}



void vHong(uint32_t CSize, uint32_t RSize, uint32_t *Fc, uint32_t *Fr, uint32_t * Bc, uint32_t * Br, bool t1, bool t2){
    //Set the device which exclusively used by this program
    cudaSetDevice(7);

    float sccTime=0;
    cudaEvent_t sccTimeStart, sccTimeStop;
    cudaEventCreate(&sccTimeStart);
    cudaEventCreate(&sccTimeStop);
    cudaEventRecord(sccTimeStart, 0);

//-----------GPU initialization---------------------------->
	uint32_t* d_Fr = NULL;
    uint32_t* d_Br = NULL;
	uint32_t* d_Fc = NULL;
    uint32_t* d_Bc = NULL;
    uint32_t* d_pivots = NULL;

	uint32_t* d_range = NULL;
    uint8_t* d_tags = NULL;
    uint8_t* tags = new uint8_t[RSize+1];

    bool volatile* d_terminatef = NULL;
    bool terminatef = false;

    bool volatile* d_terminateb = NULL;
    bool terminateb = false;

	int FWD_iterations = 0;
    int BWD_iterations = 0;
	uint32_t iterations = 0;
	int Trimm_iterations = 0;

    const uint32_t max_pivot_count = RSize;

	cudaError_t e1, e2, e3, e4, e5, e6, e7, e8;
	CUDA_SAFE_CALL( e1 = cudaMalloc( (void**) &d_Fc, CSize * sizeof(uint32_t) ));
	CUDA_SAFE_CALL( e2 = cudaMalloc( (void**) &d_Fr, (RSize + 2) * sizeof(uint32_t) ));
	CUDA_SAFE_CALL( e3 = cudaMalloc( (void**) &d_Bc, CSize * sizeof(uint32_t) ));
	CUDA_SAFE_CALL( e4 = cudaMalloc( (void**) &d_Br, (RSize + 2) * sizeof(uint32_t) ));
	CUDA_SAFE_CALL( e5 = cudaMalloc( (void**) &d_range,  (RSize + 1) * sizeof(uint32_t)));
    CUDA_SAFE_CALL( e5 = cudaMalloc( (void**) &d_tags,  (RSize + 1) * sizeof(uint8_t)));
    CUDA_SAFE_CALL( e6 = cudaMalloc( (void**) &d_pivots, max_pivot_count * sizeof(uint32_t) ));
    CUDA_SAFE_CALL( e7 = cudaMalloc( (void**) &d_terminatef, sizeof(bool) ));
    CUDA_SAFE_CALL( e8 = cudaMalloc( (void**) &d_terminateb, sizeof(bool) ));

	if (e1 == cudaErrorMemoryAllocation || e2 == cudaErrorMemoryAllocation ||
		e3 == cudaErrorMemoryAllocation || e4 == cudaErrorMemoryAllocation ||
		e5 == cudaErrorMemoryAllocation || e6 == cudaErrorMemoryAllocation ||
        e7 == cudaErrorMemoryAllocation || e8 == cudaErrorMemoryAllocation ) {
		throw "Error: Not enough memory on GPU\n";
	}

	CUDA_SAFE_CALL( cudaMemcpy( d_Fc, Fc, CSize * sizeof(uint32_t), cudaMemcpyHostToDevice ));
	CUDA_SAFE_CALL( cudaMemcpy( d_Fr, Fr, (RSize + 2) * sizeof(uint32_t), cudaMemcpyHostToDevice ));
	CUDA_SAFE_CALL( cudaMemcpy( d_Bc, Bc, CSize * sizeof(uint32_t), cudaMemcpyHostToDevice ));
	CUDA_SAFE_CALL( cudaMemcpy( d_Br, Br, (RSize + 2) * sizeof(uint32_t), cudaMemcpyHostToDevice ));
	
    CUDA_SAFE_CALL( cudaMemset( d_range, 0, (RSize + 1) * sizeof(uint32_t)));
    CUDA_SAFE_CALL( cudaMemset( d_tags, 0, (RSize + 1) * sizeof(uint8_t)));

    //for vertex-to-thread mapping
    dim3 grid;
    if((RSize + BLOCKSIZE - 1)/BLOCKSIZE > MaxXDimOfGrid) {
        int dim = ceill(sqrt(RSize / BLOCKSIZE));
        grid.x = dim;
        grid.y = dim;
        grid.z = 1;
    }else{
        grid.x = (RSize + BLOCKSIZE - 1)/BLOCKSIZE;
        grid.y = 1;
        grid.z = 1;
    }


	dim3 threads(BLOCKSIZE, 1, 1);


#ifdef _DEBUG
float pivotTime = 0, temp = 0, bTime = 0, trim1Time = 0, trim2Time = 0, updateTime = 0, wccTime = 0;
cudaEvent_t bTimeStart, bTimeStop, pivotTimeStart, pivotTimeStop, updateTimeStart, updateTimeStop;
cudaEvent_t trim1TimeStart, trim1TimeStop, trim2TimeStart, trim2TimeStop, wccTimeStart, wccTimeStop;

cudaEventCreate(&bTimeStart);
cudaEventCreate(&bTimeStop);

cudaEventCreate(&pivotTimeStart);
cudaEventCreate(&pivotTimeStop);

cudaEventCreate(&trim1TimeStart);
cudaEventCreate(&trim1TimeStop);

cudaEventCreate(&trim2TimeStart);
cudaEventCreate(&trim2TimeStop);

cudaEventCreate(&updateTimeStart);
cudaEventCreate(&updateTimeStop);

cudaEventCreate(&wccTimeStart);
cudaEventCreate(&wccTimeStop);
#endif


#ifdef _DEBUG
cudaEventRecord(trim1TimeStart, 0);
#endif

//-----------Trimming-------------------------------------->
        if(t1){
            do {
                Trimm_iterations++;
                CUDA_SAFE_CALL( cudaMemset( (void *)d_terminatef, true, sizeof(bool) ));
                trim1<<<grid, threads>>>( d_range, d_tags, d_Fc, d_Fr, d_Bc, d_Br, RSize, d_terminatef);
                CUDA_SAFE_CALL( cudaMemcpy( &terminatef, (const void *)d_terminatef, sizeof(bool), cudaMemcpyDeviceToHost ));
            } while (!terminatef);
        }

#ifdef _DEBUG
cudaEventRecord(trim1TimeStop, 0);
cudaEventSynchronize(trim1TimeStop);
cudaEventElapsedTime(&temp, trim1TimeStart, trim1TimeStop);
trim1Time+=temp;
#endif

//-----------Choose pivots--------------------------------->
#ifdef _DEBUG
cudaEventRecord(pivotTimeStart, 0);
#endif

        CUDA_SAFE_CALL( cudaMemset( d_pivots, 0, sizeof(uint32_t) ));
        pollForFirstPivot<<<grid, threads>>>( d_tags, RSize, d_pivots, d_Fr, d_Br);
        selectFirstPivot<<<grid, threads>>>( d_tags, RSize, d_pivots);

#ifdef _DEBUG
cudaEventRecord(pivotTimeStop, 0);
cudaEventSynchronize(pivotTimeStop);

cudaEventElapsedTime(&temp, pivotTimeStart, pivotTimeStop);
pivotTime+=temp;
#endif

#ifdef _DEBUG
cudaEventRecord(bTimeStart, 0);
#endif

        do{//Forward and Backward reachability
            FWD_iterations++;
            BWD_iterations++;

            CUDA_SAFE_CALL( cudaMemset( (void *)d_terminatef, true, sizeof(bool) ));
            CUDA_SAFE_CALL( cudaMemset((void *)d_terminateb, true, sizeof(bool) ));
            fwd<<<grid, threads>>>( d_Fc, d_Fr, d_range, d_tags, RSize, d_terminatef);
            bwd<<<grid, threads>>>( d_Bc, d_Br, d_range, d_tags, RSize, d_terminateb);
            CUDA_SAFE_CALL( cudaMemcpy( &terminatef, (const void *)d_terminatef, sizeof(bool), cudaMemcpyDeviceToHost ));
            CUDA_SAFE_CALL( cudaMemcpy( &terminateb, (const void *)d_terminateb, sizeof(bool), cudaMemcpyDeviceToHost ));
        }while(!terminatef && !terminateb);

        while(!terminatef){//Forward reachability
            FWD_iterations++;

            CUDA_SAFE_CALL( cudaMemset( (void *)d_terminatef, true, sizeof(bool) ));
            fwd<<<grid, threads>>>( d_Fc, d_Fr, d_range, d_tags, RSize, d_terminatef);
            CUDA_SAFE_CALL( cudaMemcpy( &terminatef, (const void *)d_terminatef, sizeof(bool), cudaMemcpyDeviceToHost ));
        }

         while(!terminateb){//Backward reachability
            BWD_iterations++;

            CUDA_SAFE_CALL( cudaMemset((void *)d_terminateb, true, sizeof(bool) ));
            bwd<<<grid, threads>>>( d_Bc, d_Br, d_range, d_tags, RSize, d_terminateb);
            CUDA_SAFE_CALL( cudaMemcpy( &terminateb, (const void *)d_terminateb, sizeof(bool), cudaMemcpyDeviceToHost ));
        }


#ifdef _DEBUG
cudaEventRecord(bTimeStop, 0);
cudaEventSynchronize(bTimeStop);

cudaEventElapsedTime(&temp, bTimeStart, bTimeStop);
bTime+=temp;
#endif

#ifdef _DEBUG
cudaEventRecord(updateTimeStart, 0);
#endif

        update<<<grid, threads>>>(d_range, d_tags, RSize, d_terminatef);

#ifdef _DEBUG
cudaEventRecord(updateTimeStop, 0);
cudaEventSynchronize(updateTimeStop);

cudaEventElapsedTime(&temp, updateTimeStart, updateTimeStop);
updateTime+=temp;
#endif

#ifdef _DEBUG
cudaEventRecord(trim1TimeStart, 0);
#endif

//-----------Trimming-------------------------------------->
        if(t1){
            do {
                Trimm_iterations++;
                CUDA_SAFE_CALL( cudaMemset( (void *)d_terminatef, true, sizeof(bool) ));
                trim1<<<grid, threads>>>( d_range, d_tags, d_Fc, d_Fr, d_Bc, d_Br, RSize, d_terminatef);
                CUDA_SAFE_CALL( cudaMemcpy( &terminatef, (const void *)d_terminatef, sizeof(bool), cudaMemcpyDeviceToHost ));
            } while (!terminatef);
        }

#ifdef _DEBUG
cudaEventRecord(trim1TimeStop, 0);
cudaEventSynchronize(trim1TimeStop);
cudaEventElapsedTime(&temp, trim1TimeStart, trim1TimeStop);
trim1Time+=temp;
#endif

#ifdef _DEBUG
cudaEventRecord(trim2TimeStart, 0);
#endif
        if(t2)
            trim2<<<grid, threads>>>( d_range, d_tags, d_Fc, d_Fr, d_Bc, d_Br, RSize);

#ifdef _DEBUG
cudaEventRecord(trim2TimeStop, 0);
cudaEventSynchronize(trim2TimeStop);
cudaEventElapsedTime(&temp, trim2TimeStart, trim2TimeStop);
trim2Time+=temp;
#endif


#ifdef _DEBUG
cudaEventRecord(trim1TimeStart, 0);
#endif

//-----------Trimming-------------------------------------->
        if(t1){
            do {
                Trimm_iterations++;
                CUDA_SAFE_CALL( cudaMemset( (void *)d_terminatef, true, sizeof(bool) ));
                trim1<<<grid, threads>>>( d_range, d_tags, d_Fc, d_Fr, d_Bc, d_Br, RSize, d_terminatef);
                CUDA_SAFE_CALL( cudaMemcpy( &terminatef, (const void *)d_terminatef, sizeof(bool), cudaMemcpyDeviceToHost ));
            } while (!terminatef);
        }

#ifdef _DEBUG
cudaEventRecord(trim1TimeStop, 0);
cudaEventSynchronize(trim1TimeStop);
cudaEventElapsedTime(&temp, trim1TimeStart, trim1TimeStop);
trim1Time+=temp;
#endif


#ifdef _DEBUG
cudaEventRecord(wccTimeStart, 0);
#endif

//Now WCC decomposition
    assignUniqueRange<<<grid, threads>>>(d_range, d_tags, RSize);

    do{
        CUDA_SAFE_CALL( cudaMemset((void *)d_terminatef, true, sizeof(bool) ));
        propagateRange1<<<grid, threads>>>( d_Fc, d_Fr, d_range, d_tags, RSize, d_terminatef);
        CUDA_SAFE_CALL( cudaMemcpy( &terminatef, (const void *)d_terminatef, sizeof(bool), cudaMemcpyDeviceToHost ));
        
        CUDA_SAFE_CALL( cudaMemset((void *)d_terminateb, true, sizeof(bool) ));
        propagateRange2<<<grid, threads>>>( d_range, d_tags, RSize, d_terminateb);
        CUDA_SAFE_CALL( cudaMemcpy( &terminateb, (const void *)d_terminateb, sizeof(bool), cudaMemcpyDeviceToHost ));
    }while(!terminatef || !terminateb);


#ifdef _DEBUG
cudaEventRecord(wccTimeStop, 0);
cudaEventSynchronize(wccTimeStop);
cudaEventElapsedTime(&temp, wccTimeStart, wccTimeStop);
wccTime+=temp;
#endif

//-----------Main algorithm-------------------------------->
	while ( true ) {
		iterations++;
        //cout<<"\nIteration : "<<iterations<<endl;

//-----------Choose pivots--------------------------------->
#ifdef _DEBUG
cudaEventRecord(pivotTimeStart, 0);
#endif

        CUDA_SAFE_CALL( cudaMemset( d_pivots, 0,  max_pivot_count * sizeof(uint32_t) ));
        pollForPivots<<<grid, threads>>>( d_range, d_tags, RSize, d_pivots, max_pivot_count, d_Fr, d_Br);
        selectPivots<<<grid, threads>>>( d_range, d_tags, RSize, d_pivots, max_pivot_count);

#ifdef _DEBUG
cudaEventRecord(pivotTimeStop, 0);
cudaEventSynchronize(pivotTimeStop);

cudaEventElapsedTime(&temp, pivotTimeStart, pivotTimeStop);
pivotTime+=temp;
#endif

#ifdef _DEBUG
cudaEventRecord(bTimeStart, 0);
#endif

        do{//Forward and Backward reachability
            FWD_iterations++;
            BWD_iterations++;

            CUDA_SAFE_CALL( cudaMemset( (void *)d_terminatef, true, sizeof(bool) ));
            CUDA_SAFE_CALL( cudaMemset((void *)d_terminateb, true, sizeof(bool) ));
            fwd<<<grid, threads>>>( d_Fc, d_Fr, d_range, d_tags, RSize, d_terminatef);
            bwd<<<grid, threads>>>( d_Bc, d_Br, d_range, d_tags, RSize, d_terminateb);
            CUDA_SAFE_CALL( cudaMemcpy( &terminatef, (const void *)d_terminatef, sizeof(bool), cudaMemcpyDeviceToHost ));
            CUDA_SAFE_CALL( cudaMemcpy( &terminateb, (const void *)d_terminateb, sizeof(bool), cudaMemcpyDeviceToHost ));
        }while(!terminatef && !terminateb);

        while(!terminatef){//Forward reachability
            FWD_iterations++;

            CUDA_SAFE_CALL( cudaMemset( (void *)d_terminatef, true, sizeof(bool) ));
            fwd<<<grid, threads>>>( d_Fc, d_Fr, d_range, d_tags, RSize, d_terminatef);
            CUDA_SAFE_CALL( cudaMemcpy( &terminatef, (const void *)d_terminatef, sizeof(bool), cudaMemcpyDeviceToHost ));
        }

         while(!terminateb){//Backward reachability
            BWD_iterations++;

            CUDA_SAFE_CALL( cudaMemset((void *)d_terminateb, true, sizeof(bool) ));
            bwd<<<grid, threads>>>( d_Bc, d_Br, d_range, d_tags, RSize, d_terminateb);
            CUDA_SAFE_CALL( cudaMemcpy( &terminateb, (const void *)d_terminateb, sizeof(bool), cudaMemcpyDeviceToHost ));
        }


#ifdef _DEBUG
cudaEventRecord(bTimeStop, 0);
cudaEventSynchronize(bTimeStop);

cudaEventElapsedTime(&temp, bTimeStart, bTimeStop);
bTime+=temp;
#endif

#ifdef _DEBUG
cudaEventRecord(updateTimeStart, 0);
#endif

        CUDA_SAFE_CALL( cudaMemset( (void *)d_terminatef, true, sizeof(bool) ));
        update<<<grid, threads>>>(d_range, d_tags, RSize, d_terminatef);
        CUDA_SAFE_CALL( cudaMemcpy( &terminatef, (const void *)d_terminatef, sizeof(bool), cudaMemcpyDeviceToHost ));
        if (terminatef)
            break; //only way out

#ifdef _DEBUG
cudaEventRecord(updateTimeStop, 0);
cudaEventSynchronize(updateTimeStop);

cudaEventElapsedTime(&temp, updateTimeStart, updateTimeStop);
updateTime+=temp;
#endif
	}
//<----------Main algorithm---------------------------------

    //SCC extraction
    CUDA_SAFE_CALL( cudaMemcpy(tags, d_tags, sizeof(uint8_t) * (RSize + 1), cudaMemcpyDeviceToHost ));
    uint32_t numberOf1Sccs = 0;
    uint32_t numberOf2Sccs = 0;
    uint32_t numberOfPivotSccs = 0;
    uint32_t numberOfSccs = 0;

    for(uint32_t i=1;i<=RSize;i++)
        if(isTrim1(tags[i]))
            numberOf1Sccs++;
        else if(isTrim2(tags[i]))
            numberOf2Sccs++;
        else if(isPivot(tags[i]))
            numberOfPivotSccs++;

    numberOfSccs = numberOf1Sccs + numberOf2Sccs + numberOfPivotSccs;

	cudaEventRecord(sccTimeStop, 0);
    cudaEventSynchronize(sccTimeStop);
    cudaEventElapsedTime(&sccTime, sccTimeStart, sccTimeStop);

    //printf(", %u, %d, %d, %d", iterations, FWD_iterations , BWD_iterations, Trimm_iterations);

#ifdef _DEBUG
printf(", %f", bTime);
printf(", %f", trim1Time);
printf(", %f", trim2Time);
printf(", %f", pivotTime);
printf(", %f", updateTime);
printf(", %f", wccTime);
#endif

    printf("\nNumber Of Sccs : %d", numberOfSccs);
    printf("\nTime : %f", sccTime );

	CUDA_SAFE_CALL( cudaFree( d_Fc ));
	CUDA_SAFE_CALL( cudaFree( d_Fr ));
	CUDA_SAFE_CALL( cudaFree( d_Bc ));
	CUDA_SAFE_CALL( cudaFree( d_Br ));
	CUDA_SAFE_CALL( cudaFree( d_range));
    CUDA_SAFE_CALL( cudaFree( d_tags));
	CUDA_SAFE_CALL( cudaFree( d_pivots ));
	CUDA_SAFE_CALL( cudaFree( (void *)d_terminatef));
    CUDA_SAFE_CALL( cudaFree( (void *)d_terminateb));

	cudaEventDestroy(sccTimeStart);
    cudaEventDestroy(sccTimeStop);

#ifdef _DEBUG
cudaEventDestroy(bTimeStart);
cudaEventDestroy(bTimeStop);
cudaEventDestroy(trim1TimeStart);
cudaEventDestroy(trim1TimeStop);
cudaEventDestroy(trim2TimeStart);
cudaEventDestroy(trim2TimeStop);
cudaEventDestroy(pivotTimeStart);
cudaEventDestroy(pivotTimeStop);
cudaEventDestroy(updateTimeStart);
cudaEventDestroy(updateTimeStop);
cudaEventDestroy(wccTimeStart);
cudaEventDestroy(wccTimeStop);
#endif

	return;
}


bool mycomp(pair<int, int> a, pair<int , int> b){
    return (a.first<b.first || (a.first==b.first && a.second<b.second));
}

void loadFullGraph(const char * filename, uint32_t * oCSize, uint32_t * oRSize, uint32_t ** oFc, uint32_t ** oFr, uint32_t ** oBc, uint32_t ** oBr){

  uint32_t  Edges = 0;
  uint32_t  Vertices = 0;
  char tmp[256];
  char tmp_c;
  uint32_t  tmp_i, from, to;

  // open the file
  filebuf fb;
  fb.open(filename,ios::in);
  if (!fb.is_open() )
  {
     printf("Error Reading graph file\n");
     return;
  }
  istream is(&fb);

  // ignore the header of the file
  for(uint32_t i = 0; i<=6; i++) 
    is.getline(tmp,256);

  //obtain the size of the graph (Edges, Vertices)
  is  >> Vertices >> Edges;

    vector<pair<int, int> > edgeList;
    pair<int, int> p;

    
    for(unsigned int k=0;k<Edges;k++){
        is >> p.first >> p.second ;
        edgeList.push_back(p);
    }
    
    sort(edgeList.begin(), edgeList.end(), mycomp);
    
  uint32_t  CSize = Edges;
  uint32_t  RSize = Vertices + 2;

  uint32_t* Fc = new uint32_t[CSize];
  uint32_t* Fr = new uint32_t[RSize];

  Fr[0] = 0;
  Fr[1] = 0;

  //obtain Fc, Fr
  uint32_t i = 1, j = 0;

  //cout<< "Reading the file" << endl;
  while(j < Edges){ 
    from = edgeList[j].first;
    to = edgeList[j].second;

     while(from > i)
     {
       Fr[ i + 1 ] = j;
       i++;
     }
     Fc[j] = to;
     j++;
  }

  //Fill up remaining indexes with M
  for(uint32_t k = i+1;k<RSize;k++)
    Fr[k] = j;

  //transposition
  uint32_t* Bc = new uint32_t[CSize];
  uint32_t* Br = new uint32_t[RSize];

  uint32_t * shift = new uint32_t [RSize];  

  uint32_t target_vertex = 0, source_vertex = 0;

  for(unsigned int i = 0; i < RSize; i++)
  {
    Br[i] = 0;
    shift[i] = 0;
  }

  for(unsigned int i = 0; i < CSize; i++)
  {
    Br[Fc[i] + 1]++;
  }

  for(unsigned int i = 0; i < RSize - 1; i++)
  {
    Br[i+1] = Br[i] + Br[i+1];
  }

  for(unsigned int i = 0; i < CSize; i++)
  {
    while(i >= Fr[target_vertex + 1])
    {
       target_vertex++;
    }
    source_vertex = Fc[i];
    Bc[ Br[source_vertex] + shift[source_vertex] ] = target_vertex;
    shift[source_vertex]++;
  }
  delete [] shift;

  *oCSize = Edges;
  *oRSize = Vertices;
  *oFc = Fc;
  *oFr = Fr;
  *oBc = Bc;
  *oBr = Br;
}

bool mycomp(pair<int, int> a, pair<int , int> b){
    return (a.first<b.first || (a.first==b.first && a.second<b.second));
}

void loadFullGraph(const char * filename, uint32_t * oCSize, uint32_t * oRSize, uint32_t ** oFc, uint32_t ** oFr, uint32_t ** oBc, uint32_t ** oBr){

  uint32_t  Edges = 0;
  uint32_t  Vertices = 0;
  char tmp[256];
  char tmp_c;
  uint32_t  tmp_i, from, to;

  // open the file
  filebuf fb;
  fb.open(filename,ios::in);
  if (!fb.is_open() )
  {
     printf("Error Reading graph file\n");
     return;
  }
  istream is(&fb);

  // ignore the header of the file
  for(uint32_t i = 0; i<=6; i++) 
    is.getline(tmp,256);

  //obtain the size of the graph (Edges, Vertices)
  is  >> Vertices >> Edges;

    vector<pair<int, int> > edgeList;
    pair<int, int> p;

    
    for(unsigned int k=0;k<Edges;k++){
        is >> p.first >> p.second ;
        edgeList.push_back(p);
    }
    
    sort(edgeList.begin(), edgeList.end(), mycomp);
    
  uint32_t  CSize = Edges;
  uint32_t  RSize = Vertices + 2;

  uint32_t* Fc = new uint32_t[CSize];
  uint32_t* Fr = new uint32_t[RSize];

  Fr[0] = 0;
  Fr[1] = 0;

  //obtain Fc, Fr
  uint32_t i = 1, j = 0;

  //cout<< "Reading the file" << endl;
  while(j < Edges){ 
    from = edgeList[j].first;
    to = edgeList[j].second;

     while(from > i)
     {
       Fr[ i + 1 ] = j;
       i++;
     }
     Fc[j] = to;
     j++;
  }

  //Fill up remaining indexes with M
  for(uint32_t k = i+1;k<RSize;k++)
    Fr[k] = j;

  //transposition
  uint32_t* Bc = new uint32_t[CSize];
  uint32_t* Br = new uint32_t[RSize];

  uint32_t * shift = new uint32_t [RSize];  

  uint32_t target_vertex = 0, source_vertex = 0;

  for(unsigned int i = 0; i < RSize; i++)
  {
    Br[i] = 0;
    shift[i] = 0;
  }

  for(unsigned int i = 0; i < CSize; i++)
  {
    Br[Fc[i] + 1]++;
  }

  for(unsigned int i = 0; i < RSize - 1; i++)
  {
    Br[i+1] = Br[i] + Br[i+1];
  }

  for(unsigned int i = 0; i < CSize; i++)
  {
    while(i >= Fr[target_vertex + 1])
    {
       target_vertex++;
    }
    source_vertex = Fc[i];
    Bc[ Br[source_vertex] + shift[source_vertex] ] = target_vertex;
    shift[source_vertex]++;
  }
  delete [] shift;

  *oCSize = Edges;
  *oRSize = Vertices;
  *oFc = Fc;
  *oFr = Fr;
  *oBc = Bc;
  *oBr = Br;
}

int main( int argc, char** argv ){
	
	
	// CSR representation 
    uint32_t CSize; // column arrays size
    uint32_t RSize; // range arrays size
    // Forwards arrays
    uint32_t *Fc = NULL; // forward columns
    uint32_t *Fr = NULL; // forward ranges
    // Backwards arrays
    uint32_t *Bc = NULL; // backward columns
    uint32_t *Br = NULL; // backward ranges
	
 loadFullGraph(file, &CSize, &RSize, &Fc, &Fr, &Bc, &Br );
	
    delete [] Fr;
    delete [] Fc;
    delete [] Br;
    delete [] Bc;
	
return 0;	
}



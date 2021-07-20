# GPU-Graph-Project-DSL

<h1>
  GunRock trace for roadNet CA
  </h1>
    
    <body>
      <div id = "first">
        Loading Matrix-market coordinate-formatted graph ...
  Reading meta data from /home/ashwina/gunrock/dataset/large/roadNet-CA/roadNet-CA.mtx.meta
  Reading edge lists from /home/ashwina/gunrock/dataset/large/roadNet-CA/roadNet-CA.mtx.coo_edge_pairs
Reading from /home/ashwina/gunrock/dataset/large/roadNet-CA/roadNet-CA.mtx.coo_edge_pairs, typeId = 262, targetId = 262, length = 2766607
  Assigning 1 to all 2766607 edges
  Substracting 1 from node Ids...
  Edge doubleing: 2766607 -> 5533214 edges
  graph loaded as COO in 0.238578s.
Converting 1971281 vertices, 5533214 directed edges ( ordered tuples) to CSR format...Done (0s).
Degree Histogram (1971281 vertices, 5533214 edges):
    Degree 0: 6075 (0.308175 %)
    Degree 2^0: 321027 (16.285197 %)
    Degree 2^1: 1176030 (59.658161 %)
    Degree 2^2: 468115 (23.746741 %)
    Degree 2^3: 34 (0.001725 %)

Computing reference value ...
__________________________
--------------------------
Run 0 elapsed: 241.523987 ms, src = 0
==============================================
64bit-VertexT=false 64bit-SizeT=false 64bit-ValueT=false undirected=false mark-pred=0 advance-mode=LB
Using advance mode LB
Using filter mode CULL
__________________________
--------------------------
Run 0 elapsed: 43.238878 ms, src = 0, #iterations = 555
Distance Validity: PASS
First 40 distances of the GPU result:
[0:0 1:1 2:1 3:2 4:3 5:3 6:2 7:14 8:15 9:13 10:12 11:11 12:10 13:9 14:10 15:11 16:11 17:10 18:11 19:12 20:13 21:14 22:14 23:13 24:14 25:14 26:15 27:14 28:15 29:13 30:14 31:15 32:16 33:16 34:17 35:62 36:63 37:64 38:65 39:66 ]
First 40 distances of the reference CPU result.
[0:0 1:1 2:1 3:2 4:3 5:3 6:2 7:14 8:15 9:13 10:12 11:11 12:10 13:9 14:10 15:11 16:11 17:10 18:11 19:12 20:13 21:14 22:14 23:13 24:14 25:14 26:15 27:14 28:15 29:13 30:14 31:15 32:16 33:16 34:17 35:62 36:63 37:64 38:65 39:66 ]

[sssp] finished.
 avg. elapsed: 43.238878 ms
 iterations: 555
 min. elapsed: 43.238878 ms
 max. elapsed: 43.238878 ms
 rate: 127.680833 MiEdges/s
 src: 0
 nodes_visited: 1957027
 edges_visited: 5520776
 load time: 297.184 ms
 preprocess time: 197.449000 ms
 postprocess time: 10.353804 ms
 total time: 251.209974 ms

      </div>
      
    </body>

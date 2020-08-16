## Affine Loop Interchange

The file AffineLoopInterchange.cpp implements a loop interchange pass in the 
affine dialect of MLIR. The pass deals only with rectangular loop nests that do 
not have any affine.if statements. For all other cases, it simply bails out 
without crashing. The pass is driven by an analytical cost model that optimizes 
for locality (spatial, temporal - both self and group) and parallelism for 
multicores to minimize the frequency of synchronization. It can handle both 
perfectly nested loop nests and imperfectly nested ones. The pass is triggered 
by using the command line flag -affine-loop-interchange.


### Design:
The pass starts by locating and converting all the imperfectly nested loop 
nests to perfectly nested ones. This process is described in the next section. 
Once all such loop nests are converted, it starts dealing with each one of them 
in turn.

 First, it groups all the affine.for ops in a loop nest in an 
llvm::SmallVector. Loop IDs are assigned to each loop induction variable in 
such a way that the outermost affine.for IV has an ID of 0. These IDs help 
calculate the access matrices for all memory accesses.

 Next, it iterates over all possible permutations of a given loop nest. For the 
permutations which do not violate any dependence constraints, it calculates the 
cost of spatial and temporal locality and a cost for parallelism. These costs 
are based on a measure of data reuse and the extent to which the loops can be 
parallelized. The one with the minimum cost is considered as the best 
permutation.

To evaluate these costs, we need affine access matrices of each memory access 
in the loop nest body and the loop-carried-dependence vector. We also need 
locality information for each loop. This locality information is needed to get 
an estimate on the total number of cache lines that will be accessed during the 
execution if the loop nest was arranged in this permutation.

 After the best permutation has been identified, it then calls the 
`permuteLoops` function declared in `LoopUtils.h` for carrying out the loop 
interchange.

Please note: In case of large loop nests having depth more than four, we 
consider permutations of only the four innermost for loops. This restriction is 
necessary to avoid long compilation times.

### Conversion from Imperfectly nested to Perfectly nested loop nest:
For each imperfectly nested loop nest, we first construct a for-loop-tree. This 
is used to capture the arrangement of loops in a given function such that all 
the loops in a loop nest are grouped together in this tree.

It then iterates over each loop nest and splits those loops which have more 
than one sibling at the same level. It does so by making separate copies of the 
parent loop for each child and then in each cloned parent copy, only one child 
is left and rest are removed. This effectively separates each child from its 
siblings and provides it with a copy of the parent.

### Cost Model:
The final cost of a given permutation consists of a combination of three 
separate costs. These are parallelism cost, spatial locality cost, and temporal 
reuse cost.

`FinalCost = 100* ParallelismCost + SpatialLocalityCost + TemporalReuseCost`

For the final cost, we assume that the cost involved with parallelism is 100x 
more expensive than those with locality costs. This assumption is reasonable 
since we know that synchronization operations require some form of main memory 
access, and memory accesses are approximately 100x slower than cache accesses. 

Thus, in a way, we evaluate the total cost of a permutation assuming the least 
costly synchronization mechanism which requires only one memory access per 
synchronization operation.

The individual costs are calculated as follows:

#### Spatial Locality Cost:
The spatial locality cost is based on the size of the locality of each loop. 
Some define locality as the number of cache lines accessed by this loop if it 
acted as the innermost loop. A method for calculating these values is given in 
a paper by Steve Carr et al. 
[!here](https://dl.acm.org/doi/abs/10.1145/195470.195557).

Once the locality information is available, we define the cost of a given 
permutation as the sum of the product of locality and the size of iteration 
sub-space defined by the parent loops in this permutation. 

That is, `SpatialCost = Sum_over_all_loops(locality of loop i * 
size of the iteration sub-space of parent loops)`

For a loopnest l1, l2, l3 this will be equal to 
SpatialCost = (locality_l3 * |l1| * |l2|)+ (locality_l2 * |l1|) + locality_l1, 
where |l| is the number of iterations of loop l.

This cost is appropriate because it calculates the overall number of cache 
lines accessed - a loop at level i executes n number of times, where n is the 
iteration sub-space size defined by the parent loops. In each execution, it 
will access its locality.

#### Temporal Reuse Cost:
The Temporal Reuse cost is based on the total reuse experienced by each memory 
access in a given permutation. For this, we check which loops (starting from 
the innermost loop) reuse given access. A loop is said to reuse access if the 
column corresponding to that loop's IV in the access matrix of that access is 
all zeros. 

This criterion is suitable since we know that if a column is all zeros, it 
means the memory access function is independent of this loop IV. Hence, the 
same value is accessed across all the loop iterations - the reuse count 
increases by a factor of O(n). Also, we will get reuse until we encounter a 
loop for which the respective column is non-zero. The reuse count stops there 
no matter if further loops have an all-zero column or not.

Based on the total reuse count, we assign a cost to the permutation such that 
the one having maximal temporal reuse receives a minimum cost.

This helps in the case when the spatial costs are the same. The permutation 
having high temporal reuse will be selected over the one having low temporal 
reuse.

#### Parallelism Cost:
For parallelism, we make use of the loop-carried-dependence vector. We define 
the cost of a loop as the size of iteration sub-space (product of iteration 
count of underlying loops) if the loop carries a dependency. Otherwise, the 
cost of this loop is zero.

The final cost of a given permutation is defined as the sum over all the 
individual loop costs. This approach works because it favors the permutations 
which have more number of free outer loops.


#### Conclusion:
The permutation to be selected depends on all - spatial locality, temporal 
reuse, and parallelism cost. The one with the minimum cost is selected as the 
final permutation. The performance of the model can be tuned using appropriate 
weights on the cost quantities. We go with a weight of 100 on parallelism, 1 on 
spatial and 1 on temporal costs.

For cases where there is no conflict between spatial reuse and parallelism, the 
approach will always produce the best permutation. For all other cases, the 
best permutation selected will depend upon the amount of locality reuse and 
amount of parallelism benefit.

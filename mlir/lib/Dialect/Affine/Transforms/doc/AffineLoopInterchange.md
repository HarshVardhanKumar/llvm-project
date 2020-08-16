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

 First, it stores all the affine.for ops in a loop nest in an 
llvm::SmallVector. It assigns loop IDs to each induction variable of the 
AffineForOp it encounters. The loop IDs are assigned in such a way that the 
outermost affine.for IV has an ID of 0. These IDs help calculate the access 
matrices for load/store ops.

 Next, it iterates over all possible permutations of a given loop nest. For 
those permutations which do not violate any dependence constraints, it 
calculates the cost of spatial and temporal locality and a cost for 
parallelism. These costs are based on a measure of data reuse and the extent to 
which the loops can be parallelized. The permutation with the minimum cost is 
considered as the best permutation.

To evaluate these costs, we need the affine access matrices of each load/store 
operation in the loop nest and the loop-carried-dependence vector. We also need 
a count of the number of cache lines accessed by each affine.for loop if it 
were considered the innermost loop. This cache access count information is 
needed to get an estimate on the total number of cache lines that will be 
accessed during the execution of the entire loop body if the loop nest was 
arranged in this permutation.

 After the best permutation has been identified, it then calls the 
`permuteLoops` function defined in `LoopUtils.h` for carrying out the loop 
interchange.

Please note: In case of large loop nests having depth more than four, we 
consider permutations of only the four innermost for loops. This restriction is 
necessary to avoid long compilation times.

### Conversion from Imperfectly nested to Perfectly nested loop nest:
For each imperfectly nested loop nest, we first construct a for-loop-tree. The 
for-loop-tree is used to capture the arrangement of for-loops in a function 
such that all the for loops in a loop nest are grouped in this tree.

It then iterates over each loop nest and splits those loops which have more 
than one sibling at the same level. It does so by making separate copies of the 
parent loop for each child and then in each cloned parent copy, removing all 
children other than one. This separates each child from its siblings and 
provides it with its copy of the parent.

### Cost Model:
The final cost of a given permutation consists of a combination of three 
separate costs. These are parallelism cost, spatial locality cost, and temporal 
reuse cost.

`FinalCost = 100* ParallelismCost + SpatialLocalityCost + TemporalReuseCost`

We assume that the cost involved with parallelism is 100x more expensive than 
those with locality costs. 

This assumption is reasonable since we know that synchronization operations 
require some form of main memory access, and memory accesses are approximately 
100x slower than cache accesses. Thus, in a way, we assume the least costly 
synchronization mechanism which requires only one memory access per 
synchronization operation.

The individual costs are calculated as follows:

#### Spatial Locality Cost:
The spatial locality cost is based on the size of the locality of each loop. 
Some define it as the number of cache lines accessed in this loop if it were 
the innermost loop. A method for calculating these values is given in a paper 
by Steve Carr et al. [!here](https://dl.acm.org/doi/abs/10.1145/195470.195557).

Once the locality information is available, we define the cost of a given 
permutation as the sum of the product of locality of each loop multiplied with 
the size of iteration sub-space defined by the loops above this loop in this 
permutation. 
That is, `SpatialCost = Sum_over_all_loops(locality of loop i * 
size of the iteration sub-space of all loops above loop i)`

For a loopnest l1, l2, l3 this will be equal to 
SpatialCost = (locality_l3 * |l1| * |l2|)+ (locality_l2 * |l1|) + locality_l1, 
where |l| is the number of iterations of loop l.

This cost is appropriate because it calculates the overall number of cache 
lines accessed due to this permutation - a loop at level i is called n number 
of times, where n is the iteration sub-space size defined by all the loops 
above this loop. In each call, this loop will access its locality.

#### Temporal Reuse Cost:
The Temporal Reuse cost is based on the total reuse experienced by each memory 
access in a given permutation. For each permutation, we check which loops 
(starting from the innermost loop) reuse given memory access. A loop is said to 
reuse memory access if the column corresponding to that loop's IV in the access 
matrix of that memory access is all zeros. 

This criterion is suitable since we know that if a column is all zeros, it 
means the memory access is independent of different values of the loop IV. 
Hence, the same value is accessed over the entire loop iteration - the reuse 
count increases by a factor of O(n). Also, we get reuse until we encounter a 
loop for which the column in the access matrix is non-zero. The reuse count 
stops there. No matter whether further loops have an all-zero column or not, 
the reuse will not take place.

Based on the total reuse count, we assign a cost to the permutation such that a 
permutation having maximal temporal reuse receives a minimum cost.

This approach works since across all the permutations, the one with maximum 
temporal reuse will receive a minimum cost - even if the spatial costs are the 
same, the permutation having high temporal reuse will be selected over the one 
having low temporal reuse.

#### Parallelism Cost:
For parallelism, we make use of the loop-carried-dependence vector. We define 
individual loop cost as the size of iteration sub-space (product of iteration 
count of underlying loops) if the loop carries a dependency. Otherwise, the 
individual loop cost for this loop is zero.

The final cost of a permutation is defined as the sum over all the individual 
loop costs. This approach works because it favors the permutations which have 
more number of free outer loops.


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

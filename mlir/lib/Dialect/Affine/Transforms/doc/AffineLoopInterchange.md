## Affine Loop Interchange

The file AffineLoopInterchange.cpp implements a loop interchange pass in the 
affine dialect of MLIR. The pass deals only with rectangular loopnests which do 
not have any affine.if statements. For all other cases, it simply bails out with-
out crashing. The pass is driven by an analytical cost model that optimizes for 
locality (spatial, temporal - both self and group) and parallelism for multicores
to minimize the frequence of synchronization. It is able to handle both perfectly
nested loop nests and imperfectly nested ones. The pass is triggered by using the
command line flag -affine-loop-interchange.


### Working:
The pass starts by locating and converting all the imperfectly nested loopnests 
to perfectly nested ones. This process is described in next section. Once all 
such loop nests are converted, it starts dealing with each one of them in turn:

*  First, it stores all the affine.for ops in a loopnest in a llvm::SmallVector.
It assigns loop IDs to each induction variable of the AffineForOp it encounters. 
The loop IDs are assigned in such a way that the outermost AffineForOp induction 
variable has an ID of 0. These IDs are helpful in calculating the access matrices 
for load/store ops.

* Next, it iterates over all the possible permutations of a loopnest to determine 
those permutations which do not violate any dependencies in the loop nest body. 
Such permutations are called `validPermutations`. If only one valid permutation 
is found, we do not need to proceed any further.

* In case more than one permutations are valid, we may need to consider each of 
those permutations to detect an optimal permutation. For this, we assign a cost to
each permutation based on a measure of locality and the extent to which the loop 
nest can be parallelized if this permutation is followed.We select the permutation 
with the minimum cost as the best permutation.

* The cost model needs few inputs to evaluate cost of each permutation - a list  
of affine access matrix of each load/store operation in the loopnest body. We 
also need a vector representing which loops in the loop nest carry dependences.

* Next we also need a count of the number of cache lines accessed by each 
affine.for loop if it were considered as the innermost loop.  This information 
is needed to get an estimate on the total number of cache lines accessed by the 
entire loop nest for a given permutation.

* Using all this information, it gets a cost for each permutation and selects 
the one with minimum cost as the final interchange permutation. It then calls the 
`permuteLoops` function for carrying out loop interchange.

### Conversion from Imperfectly nested to Perfectly nested loopnests:

For each imperfectly nested loopnest, we first construct a for-loop-tree. The 
for-loop-tree basically captures the arrangement of for-loops in the function 
such that all the for loops in a loop nest are grouped together in this tree.

It then iterates over each loopnest in this tree and makes a separate copy of 
each sibling (alongwith all the common parents) from rest of other siblings, 
one at a time. From the original copy, we erase this affine.for and from the new
copy we erase all other siblings.

### Cost Model:
The final cost of a permutation consists of a combination of three separate costs.
These are Parallelism cost, Spatial locality cost and Temporal reuse cost.

FinalCost = 100* ParallelismCost + SpatialLocalityCost + TemporalReuseCost

We assume that the cost involved with parallelism is 100x more expensive than 
those with locality cost. This assumption is fair since we know that synchronization 
operations are more costly on multicores due to cache coherence policies.

The costs are calculated as follows:

#### Spatial Locality Cost:

The Spatial locality cost is based on size of the locality of each 
forLoop. Some define it as number of cahce lines accessed in this for loop were 
the innermost loop. A method for calculating these values are given in a paper by
Steve Carr et al. [here](https://dl.acm.org/doi/abs/10.1145/195470.195557).

Once the locality information is available, we define the cost of a given 
permutation as the sum of product of locality of each loop multiplied with the 
size of iteration sub-space defined by the loops inside that loop in this
permutation. That is, `Cost_Permutation = Sum_over_all_loop(locality of loop i * 
Size of iteration sub-space of all loops which are defined inside loop i`

For a loopnest l1, l2, l3 this will be equal to 
Cost_l1_l2_l3 = locality_l1* number of iterations in l2* number of iterations in 
l3 + locality_l2 * number of iterations in l3 + locality_l3.

###### Please note that this is just a representative cost and not some 
absolute quantity.

#### Temporal Reuse Cost:

The Temporal Reuse cost is based on the sum total number of reuses experienced
by each memory access for a given permutation. For each permutation, we check 
for each loop (starting from the innermost loop) if the column corresponding to
that loop's IV is all zeros in the access matrix of any given load/store. If 
the column is all zeros, it means the same value is accessed over the entire 
loop iteration - hence the reuse count increases by a factor of O(n). The same 
procedure is repeated till we encounter a loop for which the column in access 
matrix is non-zero. The reuse count stops there. No matter whether further loops
have a all-zero column or not, the reuse will not take place once a non-zero 
column has been encountered.

Based on the total reuse count, we assign a cost to the permutation such that 
maximal temporal reuse corresponds to a minimum cost.

This approach works since over all the permutation, the one which allows for 
maximum temporal reuse will receive minimum cost - even if spatial costs are 
same, the permutation having high temporal reuse will be selected over the one 
having low temporal reuse.

#### Parallelism Cost:

For parallelism, we make use of loop-carried-dependence vector. We define 
individual loopcost as the size of iteration sub-space (product of iteration 
count of underlying loops) if the loop carries a dependency. Otherwise, the 
individual loopcost for this loop is zero.

The final cost of a permutation is defined as the sum over all the individual 
loopcosts.This approach works because it favours the permutations which have 
more number of free outer loops.


#### Conclusion:

Thus, the permutation to be selected as the best permutation is depends on all - 
spatial locality, temporal reuse and parallelism cost. The one with minimum cost
is selected as the final permutation. The performance of the model can be tuned 
using appropriate weights on the tree quantities. We go with a weight of 100 on 
parallelism, 1 on spatial and 1 on temporal costs.

However for cases which do not conflict between spatial reuse and parallelism, 
the approach will always produce the best permutation. For all other cases, the 
best permutation selected will depend upond the amount of locality reuse and 
amount of parallelism benefit.

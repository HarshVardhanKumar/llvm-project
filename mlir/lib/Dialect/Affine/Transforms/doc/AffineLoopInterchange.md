## Affine Loop Interchange

The file AffineLoopInterchange.cpp implements a loop interchange pass in affine dialect of MLIR. It deals only with rectangular loopnests which do not have any Affine-If statements. For all other cases, it bails out without crashing.

The pass is driven by an analytical cost model (described below) that optimizes for locality(spatial, temporal - both self and group) and parallelism for multicores so as to minimize the frequence of synchronization.

It is able to handle both perfectly nested loop nest and imperfectly nested ones. 

The pass is triggered by the command line flag -affine-loop-interchange.


### Working:
The pass starts with locating and converting imperfectly nested loopnest to perfectly nested loopnest. The process is described in next section.

Once all the loop nests are converted to perfectly nested loops, it starts dealing with each one of them in turn:
*  First, it detects a loopnest and stores all the AffineForOps involved in the loopnest in a llvm::SmallVector. Along the way, it assigns loop-id values to each induction variable of the AffineForOp it encounters. The loop-id values are assigned such that the outermost AffineForOp induction variable has the minimum loop-var-id The loop-var-id are helpful in calculating the access matrices of all the load/store operations at a later stage.

* Next, we iterate over all possible permutations of the loopnest to determine those permutations which do not violate any dependencies in the loopnest body. Such permutations are called "ValidPermutations". If only one valid permutation is found, we do not need to proceed any further.

* In case more than one permutations are valid, we may need to consider each of those permutations to detect the optimal permutation. For this, we iterate over all the valid permutations and assigning a cost for each permutation. Lateon, we select the permutation with minimum cost. The cost is assinged using a cost model.

* The cost model needs few inputs to evaluate cost of each permutation - We first need a list of Affine Access Matrix of each load/store operation in the loopnest body. After that, we need a list of all dependencies in the loopnest body and a list of all loop-carried dependence vectors.

* Next we also need a count of the different cache lines accessed by each AffineForOp loop in the loop if it were considered as the innermost loop. This information is needed to get a bound on the number of cache lines accessed by the entire loopnest in a given permutation.

* Using all this information, we get a cost for each permutation and select the permutation with minimum cost as the final interchange permutation. It then calls the "permuteLoops" function of mlir for loop interchange.

### Conversion from Imperfectly nested to Perfectly nested loopnests:

For each imperfectly nested loopnest, we first construct a forlooptree. The forlooptree basically captures the structure of forloops in the function.

It then iterates over each loopnest in the forlooptree and performs a loop-split wherever it finds siblings under a common parent forloop. After the split, each sibling is a separate loopnest with a copy of the common parent.

### Cost Model:
The cost model consists of calculating three separate costs - Parallelism cost, Spatial locality cost and Temporal reuse cost associated with each permutation. It then calculates a weighted average of these costs to get the final cost.

FinalCost = 100* ParallelismCost + SpatialLocalityCost + TemporalReuseCost

That is, we assume that the cost involved with parallelism are 100x more expensive than those with locality cost. This assumption is fair since we know that synchronization operations are more involved.

These costs are calculated as follows:

#### Spatial Locality Cost:

The Spatial locality cost is based on finding out the locality of each forloop in the loopnest. The "locality" here is defined as the number of cache lines accessed by that forloop if it were assumed to be the only forloop in the loopnest. A method for calculating these values are given in a paper by Steve Carr et al. [here](https://dl.acm.org/doi/abs/10.1145/195470.195557).

Once the locality information is available, we define the cost of a given permutation as the sum of product of locality of each loop multiplied with the size of iteration sub-space defined by all other loops inside that loop in the loopnest. 
That is, Cost_Permutation = Sum_over_all_loop(locality of loop i * Size of iteration sub-space of all loops which are defined inside loop i)

For a loopnest l1, l2, l3 this will be equal to Cost_l1_l2_l3 = locality_l1*number of iterations in l2* number of iterations in l3 + locality_l2*number of iterations in l3 + locality_l3.

###### Please note that this is just a representative cost and not some absolute quantity.

#### Temporal Reuse Cost:

The Temporal Reuse cost is based on the sum of total number of reuses experienced by each memory access for a given permutation.

For each permutation, we check for each forloop starting from the innermost loops if the column corresponding to that forloop-induction-var is all zeros in the access matrix of a given load/store. If the column is all zeros, it means the same value is accessed over the entire loop iteration - hence the reuse count increases by a factor of O(n). The same procedure is repeated till we encounter a forloop in the loopnest for which the column in access matrix is non-zero. The reuse count stops there. No matter whether further loops have a all zero column or not, the reuse will not take place once a non-zero column has been encountered.

Based on the total reuse count, we assign a cost to the permutation such that maximal temporal reuse corresponds to minimum cost.

This approach works since over all the permutation, the one which allows for maximum temporal reuse will receive minimum cost - even if spatial costs are same, the permutation having high temporal reuse will be selected over the one having low temporal reuse.

#### Parallelism Cost:

For parallelism, we make use of loop-carried-dependence vector. We define individual loopcost as the size of iteration sub-space (product of iteration count of underlying loops) defined by underlying loops inside this loop if the loop carries a dependency. Otherwise, the individual loopcost for this loop is zero.

The final cost of a permutation is defined as the sum over all the individual loopcosts.

This approach works because it favours the permutations which have more number of free outer loops. Please note that we favour permutations with low cost.


#### Conclusion:

Thus, which permutation will be selected as the optimal permutation is dependent on all - spatial locality, temporal reuse and parallelism cost. The one with minimum cost is selected as the final permutation. The performance of the model can be tuned using appropriate weights on the tree quantities. We go with a weight of 100 on parallelism, 1 on spatial and 1 on temporal costs.

However for cases which do not conflict between spatial reuse and parallelism, the approach will always produce the best permutation. For all other cases, the best permutation selected will depend upond the amount of locality reuse and amount of parallelism benefit.
 

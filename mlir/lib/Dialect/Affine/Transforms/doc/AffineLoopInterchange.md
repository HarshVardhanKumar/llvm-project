# Affine Loop Interchange


## Abstract
This patch implements a loop interchange pass in the affine dialect of MLIR 
codebase. The pass finds and permutes all the loop nests in the input mlir file 
to their best permutation. In such a loop permutation, the loop body will 
experience a higher level of data reuse and parallelism, or conversly a minimal 
number of synchronizations and cache accesses. The pass is driven by an 
analytical cost model as described in subsequent sections. The pass is able to 
handle both perfectly and imperfectly nested loop nests. Use the flag 
-affine-loop-interchange to trigger the pass.

## Design:
The pass is designed to work only for rectangular loop nests that do not have 
any if-conditional statements in the loop body - for such cases, the pass will 
simply bail out without doing anything. 

Before carrying out any interchange, the pass first detects and converts all 
the imperfectly nested loop nests to perfectly nested. Doing this allows a
regular implementation. However a possible drawback is that those loop nests
lose their structure. The procedure for carrying out this transformation is
explained in the coming sections.

After all the loop nests have been converted to perfect loop nest, the pass 
makes an estimate on the number of cycles needed to execute the entire loop 
body for each valid loop permutation. The particulars of this cost model is 
explained in the following section. After all the permutations are checked,
the one with the minimum estimate is selected for the final interchange. 

Please note: In case of large loop nests having depth more than four, we 
consider permutations of only the three innermost for loops. This restriction is 
necessary to avoid long compilation times.

The pass maintains few state variables during the time it processes each loop 
nest. These include the `loop carried dependence vector`, `loop iteration counts` 
vector, a `cache lines access count` map, a list of all load and store ops and a
vector of all loops in the current loop nest. All these are invariant with respect
to various loop permutations and storing them as state variables avoids the need
to recalculate them for every permutation.

## Cost Model:
The pass uses a very simple cost model. The cost of any loop permutation is the 
number of cycles estimated for the complete execution of the loop body.

Now, since the actual number of cycles will be machine dependent, the pass 
estimates this value in terms of number of cache accesses, considering each 
cache access taking only unit cycle. The final cost of each loop permutation is 
given by the sum of the costs due to synchronizations and costs due to cache 
accesses.

The cache lines are accessed during spatial reuse and due to *lack* of temporal 
reuse. In other words, the cost due of cache accesses can be further divided into
two categories: spatial cost and temporal cost. 

Using all this, we formulate the total cost of a loop permutation as:
Total cost = Cost due to syncs + Cost due to spatial reuse + Cost due to *lack* 
of temporal reuse. 

Now, for synchronization operations, the pass makes an humble assumption that 
each sync needs only one memory access on average, and thus each sync takes the 
same number of cycles needed for single memory access - approx 128 cycles. In 
other words, the cost of one sync is approx 128x the cost of one cache access. 

The expression for the total cost can thus be written as: 

`TotalCost = 128 x Number of syncs + Max number of cache accesses due to 
spatial reuse + Max number of cache accesses due to lack of temporal reuse.` 

The calculation for each of these values is discussed in coming sections.


## Imperfectly nested to Perfectly nested loop nest:
The pass deals with only one type of imperfect nest where multiple loops appear 
as children of some common parent. Other types of imperfect nests are not dealt 
with at the present.

For this kind of imperfect nests, the pass separates each such loop from the rest
of its siblings. Implementation wise, the pass clones the common parent-for each
sibling and then ensures that each clone has only one sibling left. Rest of the 
siblings. At the end, the original loop nest is replaced with a new loop nest 
where each parent has only one child loop.


## Calculating the number of cache accesses due to spatial reuse:
The pass uses an upper limit on the number of cache accesses while executing 
the loop body as one of the parameters in the cost function. For this, it needs 
the upper limits on the number of cache accesses made by every loop of the loop
nest.

This upper limit information is found using the method described in the paper 
['Compiler Optimizations for improving data 
locality'](https://dl.acm.org/doi/abs/10.1145/195470.195557) by Steve Carr et 
al.

After the upper limit values are obtained, the pass makes use of the fact that 
when a loop at depth `i` (having upper limit U) is executed n times, the total 
number of cache accesses made by it cannot exceed more than n x U. Thus, if the 
iteration vector space defined by its parent loops has a size S, then this loop 
will be executed exactly S times and the upper limit on the number of cache 
accesses made by this loop is S x U.

A sum of the upper limit for each loop gives an estimate on the upper limit for 
the entire permutation. Please note that while calculating this value, we do 
not consider any temporal reuse.


## Calculating the number of cache accesses due to lack of temporal reuse:
We know that if some memref undergoes temporal reuse, it can be referred from 
the registers directly without accessing the cache. Thus, a loop permutation 
exhibiting high amounts of temporal reuse requires less number of cache 
accesses and vice-versa.

To estimate the amount of temporal reuse, the pass makes use of the fact that 
if a memref's access function is independent of loopIVs of consecutive k 
innermost loops, then the memref will experience a temporal reuse of order 
O(n^k).

The pass starts with the innermost loop and checks for every loop if the memref's 
access matrix is independent of its IV till it finds the first loop whose loopIV 
value determines the memref's access function. If such a loop is found at depth k, 
then there is O(n^{L-k}) temporal reuse, where L is the max loop depth.

In other words, there is an upper limit of O(n^L)-O(n^{L-k}) cache access (ignoring 
spatial reuses).


## Number of Synchronizations needed:
To find out the number of synchronizations needed in a given loop permuation, 
the pass uses the fact that the location of the "first loop with a loop carried 
dependence" on the given permutation satisfies the loop dependence and therefore 
decides the number of syncs needed. If such a loop is found at depth i on a loop
nest of max depth L, then one needs O(n^{L-i}) synchronizations per iteration of
this loop.

The pass makes use of the loop-carried-dependence vector to find the location of
such a loop. The loop carried dependence vector itself is calculated by checking 
memref access dependence for each pair of memref accesses at each loop depth.

## Conclusion:
The permutation with the smallest final cost value is selected as the final loop 
permutation. The pass then calls the `permuteLoops()` method declared in 
`LoopUtils.h` to carry out the interchange.

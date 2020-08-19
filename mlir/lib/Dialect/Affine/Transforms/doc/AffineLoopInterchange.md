# Affine Loop Interchange

Author: Harsh Kumar

## Abstract

This patch implements a loop interchange pass in the affine dialect of MLIR 
codebase. The pass finds and permutes all the loop nests in an input mlir file 
to 
thier best permutation. In such a loop permutation, the loop body will 
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
the imperfectly nested loop nests to perfectly nested loop nests. Doing this 
allows a regular implementation. However a possible drawback is that those loop 
nests will lose their structure. The procedure for carrying out this 
transformation is explained in the next section.

After all the loop nests have been converted to perfect loop nest, the pass 
makes an estimate on the number of cycles needed to execute the entire loop 
body for each valid loop permutation. The particulars of this cost model is 
explained in the following section. After all the permutations are checked, the 
one with the minimum estimate is selected for the final interchange. 

Please note: In case of large loop nests having depth more than four, we 
consider permutations of only the four innermost for loops. This restriction is 
necessary to avoid long compilation times.

The pass maintains few state variables during the time it spends while 
processing each loop nest. These include the `loop carried dependence vector`, 
`loop iteration counts` vector, a `cache lines access count` map, a list of all 
load and store ops and a vector of all loops in the current loop nest. All 
these values are invariant with respect to various loop permutations and 
storing them as state variables avoids the need to recalculate them for every 
permutation.

## Cost Model:

The pass uses a very simple cost model. The cost of any loop permutation is the 
number of cycles estimated for the complete execution of the loop body.

Now, since the actual number of cycles will be machine dependent, the pass 
estimates this value in terms of number of cache accesses, considering each 
cache access taking only unit cycle. The final cost of each loop permutation is 
given by the sum of the costs due to synchronizations and costs due to cache 
accesses.

The cache lines are accessed during spatial reuse and due to *lack* of temporal 
reuse. That is, the cost due of cache accesses can be further divided into two 
categories: spatial cost and temporal cost. 

Final cost = Cost due to syncs + Cost due to spatial reuse + Cost due to *lack* 
of temporal reuse. 

Now, for synchronization operations, the pass makes an humble assumption that 
each sync needs only one memory access on average, and thus each sync takes the 
same number of cycles needed for single memory access - approx 128 cycles. That 
is, the cost due to one sync is approx 128x the cost of one cache access. 

The expression for final cost can thus be written as: 

`FinalCost = 128 x Number of syncs + Max number of cache accesses due to 
spatial reuse + Max number of cache accesses due to lack of temporal reuse.` 

The calculation for each of these values is discussed in coming sections.


## Imperfectly nested to Perfectly nested loop nest:

The pass deals with only one type of imperfect nest where multiple loops appear 
as children of some common parent. Other types of imperfect nests are not dealt 
with at present.

For the imperfect nests where multiple loops appear as siblings at some depth, 
the pass separates each such loop from the rest of its siblings. Implementation 
wise, the pass clones the common parent loop multiple times - one for each 
sibling and then ensures that each clone has only one sibling left. Rest of the 
siblings are removed from each clone. At the end, the original loop nest is 
replaced with a new loop nest where each parent has only one child loop.


#### Calculating the number of cache accesses due to spatial reuse:

The pass uses an upper limit on the number of cache accesses while executing 
the loop body as one of the parameters in the cost function. For this, it needs 
the information on upper limits on the number of cache accesses made by each 
loop of the loop nest.

This upper limit information is found using the method described in the paper 
['Compiler Optimizations for improving data 
locality'](https://dl.acm.org/doi/abs/10.1145/195470.195557) by Steve Carr et 
al.

After the upper limit values are obtained, the pass makes use of the fact that 
when a loop at depth `i` (having upper limit U) is executed n times, the total 
number of cache accesses made by it cannot exceed more than n x U. Also, if the 
iteration vector space defined by its parent loops has a size S, then this loop 
will be executed exactly S times. In other words, the upper limit on the number 
of cache accesses made by this loop is S x U.

A sum of the upper limit for each loop gives an estimate on the upper limit for 
the entire permutation. Please note that while calculating this value, we do 
not consider the existence of any temporal reuse.


#### Calculating the number of cache accesses due to lack of temporal reuse:

We know that if some memref undergoes temporal reuse, it can be referred from 
the registers directly without accessing the cache. Thus, a loop permutation 
exhibiting high amounts of temporal reuse requires less number of cache 
accesses and vice-versa.

To estimate the amount of temporal reuse, the pass makes use of the fact that 
if a memref's access function is independent of loopIVs of k innermost loops, 
then for all iterations of those loops, the memref experiences a temporal reuse 
of order O(n^k).

The pass starts with the innermost loop and checks for every loop if the 
memref's access matrix is independent of that loop's IV till it finds the first 
loop whose loopIV changes the memref's access function. If such a loop is found 
at depth k, it means there is O(n^{L-k}) temporal reuse, where L is the max 
loop depth.

It means, there is an upper limit of O(n^L)-O(n^{L-k}) cache access (ignoring 
spatial reuses).


#### Number of Synchronizations needed:

To find out the number of synchronizations needed in a given loop permuation, 
the pass uses the fact that the location of the "first loop with a loop carried 
dependence" on the given permutation decides the number of syncs needed. If 
such a loop is found at depth i on a loop nest of max depth L, then one needs 
O(n^{L-i}) synchronizations per iteration of this loop.

The pass makes use of the loop-carried-dependence vector to find the location 
of such a loop. The loop carried dependence vector is calculated by checking 
memref access dependence for each pair of memref accesses at each loop depth.

#### Conclusion:

The permutation with the smalles final cost value is selected as the final loop 
permutation. The pass then calls the `permuteLoops()` method declared in 
`LoopUtils.h` to carry out the interchange.

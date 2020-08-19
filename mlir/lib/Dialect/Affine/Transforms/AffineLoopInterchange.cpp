//===- AffineLoopInterchange.cpp - Pass to perform loop interchange-----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a loop interchange pass that optimizes for locality
// (spatial and temporal - both self and group) and parallelism for multicores,
// to minimize the frequency of synchronization. The pass works for both
// perfectly nested and imperfectly nested loops (any level of nesting). However
// in the presence of affine.if statements and/or non-rectangular iteration
// space, the pass simply bails out - leaving the original loop nest unchanged.
// The pass is triggered by the command line flag -affine-loop-interchange.
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"
#include "mlir/Analysis/AffineAnalysis.h"
#include "mlir/Analysis/Utils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/Transforms/LoopUtils.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVector.h"
#include <algorithm>
#include <cmath>
#include <numeric>
using namespace mlir;
namespace {
struct LoopInterchange : public AffineLoopInterchangeBase<LoopInterchange> {
  void runOnFunction() override;
  void handleImperfectlyNestedAffineLoops(Operation &funcOp);
  void runOnAffineLoopNest();

private:
  /// Default cache line size(in bytes). Useful for getting a measure of the
  /// locality of each loop in a given loop nest.
  constexpr static unsigned kCacheLineSize = 64;

  /// Default element size to be used if a memref does not have a static shape.
  constexpr static unsigned kDefaultEltSize = 8;

  bool isRectangularAffineForLoopNest();

  void getLoopCarriedDependenceVector();

  void getAllLoadStores();

  void getCacheLineAccessCounts(
      DenseMap<Operation *, SmallVector<SmallVector<int64_t, 4>, 4>>
          &loopAccessMatrices,
      DenseMap<Operation *, unsigned> &elementSizes);

  uint64_t getNumCacheLinesSpatialReuse(ArrayRef<unsigned> perm);

  uint64_t getNumSyncs(ArrayRef<unsigned> perm);

  uint64_t getNumCacheLinesTemporalReuse(
      ArrayRef<unsigned> permutation,
      DenseMap<Operation *, SmallVector<SmallVector<int64_t, 4>, 4>>
          &loopAccessMatrices,
      uint64_t maxTemporalReuse);

  bool getBestPermutation(DenseMap<Value, unsigned> &loopIndexMap,
                          SmallVector<unsigned, 4> &bestPerm);

  // Loop Carried Dependence vector. A 'true' at index 'i' means that the loop
  // at depth 'i' carries a dependence.
  SmallVector<bool, 4> loopCarriedDV;

  // Iteration count of each loop in the loop nest.
  SmallVector<unsigned, 4> loopIterationCounts;

  // The loop nest.
  SmallVector<AffineForOp, 4> loopVector;

  /// Number of cache lines accessed by each loop in the loop nest.
  DenseMap<const AffineForOp *, uint64_t> cacheLinesAccessCounts;

  // List of all load/store ops in the loop nest body.
  SmallVector<Operation *, 8> loadAndStoreOps;
};
} // namespace

/// Returns true if any affine-if op found in the loop nest rooted at `forOp`
static bool hasAffineIfStatement(AffineForOp &forOp) {
  auto walkResult =
      forOp.walk([&](AffineIfOp op) { return WalkResult::interrupt(); });
  return walkResult.wasInterrupted();
}

/// Checks if the given loop nest has a rectangular-shaped iteration space.
bool LoopInterchange::isRectangularAffineForLoopNest() {
  for (AffineForOp forOp : loopVector) {
    if (!forOp.hasConstantUpperBound() || !forOp.hasConstantLowerBound())
      return false;
  }
  return true;
}

/// Fills `row` with the coefficients of loopIVs in `expr`. Every value in 
/// `operands` should either be a loopIV or a terminal symbol.
static void prepareCoeffientRow(AffineExpr expr, ArrayRef<Value> operands,
                                DenseMap<Value, unsigned> &loopIndexMap,
                                SmallVector<int64_t, 4> &row) {
  // TODO: Implement support for terminal symbols.
  // The value at the last index of the `row` is an element of the constant
  // vector b.
  row.resize(loopIndexMap.size() + 1);
  switch (expr.getKind()) {
  case AffineExprKind::Add: {
    // Please note that in the case of an add operation, either both `lhs` and
    // `rhs` are dim exprs or the `lhs` is a dim expr and the `rhs` is a
    // constant expr.
    AffineBinaryOpExpr addExpr = expr.cast<AffineBinaryOpExpr>();
    AffineExpr lhs = addExpr.getLHS();
    AffineExpr rhs = addExpr.getRHS();
    unsigned lhsPosition = 0;
    unsigned rhsPosition = 0;
    if (lhs.isa<AffineDimExpr>()) {
      auto dimExpr = lhs.cast<AffineDimExpr>();
      lhsPosition = loopIndexMap[operands[dimExpr.getPosition()]];
    }
    // Update the loopIV only if it has not been encountered before. Please note
    // that it is possible that the same loopIV have been encountered before
    // while parsing other exprs. In that case, the appropriate coefficient is
    // already set.
    if (row[lhsPosition] == 0)
      row[lhsPosition] = 1;
    // The `rhs` may be a constant expr. In that case, no need to update the
    // `row`.
    bool isConstRhs = false;
    if (rhs.isa<AffineDimExpr>()) {
      auto dimExpr = rhs.cast<AffineDimExpr>();
      rhsPosition = loopIndexMap[operands[dimExpr.getPosition()]];
    } else if (rhs.isa<AffineConstantExpr>()) {
      row.back() += rhs.cast<AffineConstantExpr>().getValue();
      isConstRhs = true;
    }
    if (row[rhsPosition] == 0 && !isConstRhs)
      row[rhsPosition] = 1;
    break;
  }
  case AffineExprKind::Mul: {
    AffineBinaryOpExpr mulExpr = expr.cast<AffineBinaryOpExpr>();
    AffineExpr lhs = mulExpr.getLHS();
    AffineExpr rhs = mulExpr.getRHS();
    unsigned dimIdPos = 0;
    // In the case of a mul expr, the lhs can only be a dim expr and the rhs can
    // only be a constant expr.
    if (lhs.isa<AffineDimExpr>()) {
      auto dim = lhs.cast<AffineDimExpr>();
      dimIdPos = loopIndexMap[operands[dim.getPosition()]];
    }
    if (rhs.isa<AffineConstantExpr>()) {
      row[dimIdPos] = rhs.cast<AffineConstantExpr>().getValue();
    }
    break;
  }
  case AffineExprKind::DimId: {
    // This takes care of the cases like A[i] where i is a loopIV. Since it is
    // not a binary expr, there is no lhs/rhs.
    auto dimExpr = expr.cast<AffineDimExpr>();
    row[loopIndexMap[operands[dimExpr.getPosition()]]] = 1;
    break;
  }
  case AffineExprKind::CeilDiv:
  case AffineExprKind::FloorDiv:
  case AffineExprKind::Mod: {
    // Even though exprs like CeilDiv/FloorDiv and Mod can be considered as
    // binary exprs, the `rhs` in these exprs is always a constant as per the
    // rules of AffineExpr. These constant values do not make part of either
    // the vector-b or the matrix A. Thus, we don't need to care about `rhs`
    // in these cases.
    auto modExpr = expr.cast<AffineBinaryOpExpr>();
    AffineExpr lhs = modExpr.getLHS();
    if (lhs.isa<AffineDimExpr>()) {
      auto dimExpr = lhs.cast<AffineDimExpr>();
      row[loopIndexMap[operands[dimExpr.getPosition()]]] = 1;
    }
  }
  }
}

/// Populates `loopAccessMatrices` with the access matrices (A|b) of all load 
/// and store ops in the loop body. Please note that each affine access can be 
/// represented as a linear system Ax+b (A is the affine access matrix, x is the 
/// vector of loopIVs and b is the constant-term vector). `loopIndexMap` holds 
/// depth locations of each loopIV in the original loop order.
static void getAffineAccessMatrices(
    ArrayRef<Operation *> loadAndStoreOps,
    DenseMap<Value, unsigned> &loopIndexMap,
    DenseMap<Operation *, SmallVector<SmallVector<int64_t, 4>, 4>>
        &loopAccessMatrices) {

  for (unsigned i = 0; i < loadAndStoreOps.size(); ++i) {
    Operation *srcOp = loadAndStoreOps[i];
    MemRefAccess srcAccess(srcOp);
    AffineMap map;
    if (auto loadOp = dyn_cast<AffineLoadOp>(srcOp))
      map = loadOp.getAffineMap();
    else if (auto storeOp = dyn_cast<AffineStoreOp>(srcOp))
      map = storeOp.getAffineMap();
    SmallVector<Value, 8> operands(srcAccess.indices.begin(),
                                   srcAccess.indices.end());
    fullyComposeAffineMapAndOperands(&map, &operands);
    map = simplifyAffineMap(map);
    canonicalizeMapAndOperands(&map, &operands);
    ArrayRef<AffineExpr> mapResults = map.getResults();
    loopAccessMatrices[srcOp].resize(mapResults.size());
    for (unsigned l = 0; l < mapResults.size(); l++) {
      // Parse the l-th map result(access expr for the l-th dim of this memref)
      // to get the l-th row of this op's access matrix.
      AffineExpr mapResult = mapResults[l];
      // Check if the `mapResult` is a constant expr. If yes, there is no need
      // to walk it. Instead, add the value to the constant b-vector element and
      // leave the row unchanged. The last column of an access matrix stores the
      // b-vector.
      if (mapResult.isa<AffineConstantExpr>()) {
        auto constExpr = mapResult.cast<AffineConstantExpr>();
        loopAccessMatrices[srcOp][l].back() = constExpr.getValue();
      } else {
        mapResult.walk([&](AffineExpr expr) {
          // Each expr can in turn be a combination of many sub expressions.
          // Walk each of these sub-exprs to fully parse the `mapResult`.
          prepareCoeffientRow(expr, operands, loopIndexMap,
                              loopAccessMatrices[srcOp][l]);
        });
      }
    }
  }
}

/// Separates the last sibling loop from its fellow siblings. After separation,
/// it receives a copy of the common parent independent from its other siblings.
/// A loop nest like: \code
///     parent{forOpA, forOpB, lastSibling}
/// \endcode
/// becomes
/// \code
///     parent{lastSibling}, parent{forOpA, forOpB}
/// \endcode
static void separateSiblingLoops(AffineForOp &parentForOp,
                                 SmallVector<AffineForOp, 4> &siblings) {

  OpBuilder builder(parentForOp.getOperation()->getBlock(),
                    std::next(Block::iterator(parentForOp)));
  AffineForOp copyParentForOp = cast<AffineForOp>(builder.clone(*parentForOp));
  // We need `siblings` as a SmallVector. We cannot use an ArrayRef here because
  // that would make each element in `siblings` a 'const' and this would prevent
  // us from calling getOperation() method.

  // We always separate the last sibling from the group. For this we'll need the
  // order in which all the siblings are arranged. We need this order to compare
  // loops with their cloned copy in `copyParentForOp`. Comparision using the
  // AffineForOp.getOperation() method does not work in this case.
  AffineForOp lastSibling = siblings.back();
  unsigned lastSiblingPosition = 0;
  llvm::SmallSet<unsigned, 8> siblingsIndices;
  unsigned siblingIndex = 0;
  parentForOp.getOperation()->walk([&](AffineForOp op) {
    siblingIndex++;
    if (op.getOperation() == lastSibling.getOperation())
      lastSiblingPosition = siblingIndex;
    for (unsigned i = 0; i < siblings.size(); i++)
      if (op.getOperation() == siblings[i].getOperation())
        siblingsIndices.insert(siblingIndex);
  });
  // Walk the cloned copy to erase all the other siblings.
  siblingIndex = 0;
  copyParentForOp.getOperation()->walk([&](AffineForOp op) {
    siblingIndex++;
    if (siblingIndex != lastSiblingPosition &&
        siblingsIndices.count(siblingIndex))
      op.getOperation()->erase();
  });
  // Erase the `lastSibling` from the the original copy.
  lastSibling.getOperation()->erase();
}

/// Deals with imperfect loop nests where multiple loops appear as children
/// of some common parent loop. Converts all such imperfectly nested loops
/// in `funcOp` to perfectly nested ones by separating each sibling at a
/// time. That is, if two or more loops are present as siblings at some depth,
/// it will separate each of those siblings such that there is no common 
/// parent left in the new structure. Each sibling receives a separate copy
/// of the common parent. This process is repeated until each parent has only 
/// one child left.
void LoopInterchange::handleImperfectlyNestedAffineLoops(Operation &funcOp) {
  // Store the arrangement of all the for-loops in the `funcOp` body in a tree
  // structure. This makes storing the parent-child relationship an easy task. 
  DenseMap<Operation *, SmallVector<AffineForOp, 4>> forTree;
  // A helper map for the `forTree`. Since `AffineForOp` cannot act as a for
  // a DenseMap, we've to use a map to convert to and from an affine.for to an
  // Operation* and vice-versa.
  DenseMap<Operation *, AffineForOp> forOperations;

  // Stop splitting when each parent has only one child left.
  bool oneChild = false;
  while (!oneChild) {
    oneChild = true;
    // Walk the function to create a tree of affine.for operations.
    funcOp.walk([&](AffineForOp op) {
      if (op.getParentOp()->getName().getStringRef() == "affine.for")
        forTree[op.getOperation()->getParentOp()].push_back(op);
      forOperations[op.getOperation()] = op;
    });
    // Separate one sibling at a time.
    for (auto &parentChildrenPair : forTree) {
      // This loop nest has no sibling problem. Check the next loop nest.
      if (parentChildrenPair.second.size() < 2)
        continue;
      oneChild = false;
      separateSiblingLoops(forOperations[parentChildrenPair.first],
                           parentChildrenPair.second);
      // We need to walk the function again to create a new `forTree` since the
      // structure of the loop nests within the `funcOp` body has changed after
      // the separation.
      break;
    }
    forTree.clear();
    forOperations.clear();
  }
  return;
}

/// Scans the loop nest to collect all the load and store ops. The list
/// of all such ops is maintained in the private member `loadAndStoreOps`.
void LoopInterchange::getAllLoadStores() {
  loopVector[0].getOperation()->walk([&](Operation *op) {
    if (isa<AffineLoadOp>(op) || isa<AffineStoreOp>(op)) {
      loadAndStoreOps.push_back(op);
    }
  });
}

/// Fills `elementSizes` with the size of the element types of all the memrefs
/// in the loop nest body. These are later used to check whether or not two
/// accesses are within a cacheLineSize/elementSize distance apart for a 
/// successful reuse.
static void getElementSizes(ArrayRef<Operation *> loadAndStoreOps,
                            unsigned defaultElementSize,
                            DenseMap<Operation *, unsigned> &elementSizes) {

  MemRefType memRefType;
  for (Operation *op : loadAndStoreOps) {
    if (isa<AffineLoadOp>(op)) {
      memRefType = cast<AffineLoadOp>(*op).getMemRefType();
    } else if (isa<AffineStoreOp>(op)) {
      memRefType = cast<AffineStoreOp>(*op).getMemRefType();
    }
    elementSizes[op] = memRefType.hasStaticShape()
                           ? getMemRefSizeInBytes(memRefType).getValue() /
                                 memRefType.getNumElements()
                           : defaultElementSize;
  }
}

/// Calculates the loop-carried-dependence vector for the given loop nest. A value
/// `true` at the i-th index means there is a loop carried dependence at depth i.
void LoopInterchange::getLoopCarriedDependenceVector() {

  // `loopCarriedDV` should have one entry for each loop.
  loopCarriedDV.resize(loopVector.size());
  for (unsigned i = 0; i < loadAndStoreOps.size(); ++i) {
    Operation *srcOp = loadAndStoreOps[i];
    for (unsigned j = 0; j < loadAndStoreOps.size(); ++j) {
      Operation *dstOp = loadAndStoreOps[j];
      for (unsigned depth = 1; depth <= loopVector.size() + 1; ++depth) {
        MemRefAccess srcAccess(srcOp), dstAccess(dstOp);
        FlatAffineConstraints dependenceConstraints;
        SmallVector<DependenceComponent, 2> depComps;
        DependenceResult result = checkMemrefAccessDependence(
            srcAccess, dstAccess, depth, &dependenceConstraints, &depComps);
        if (hasDependence(result)) {
          for (unsigned i = 0; i < depComps.size(); i++) {
            DependenceComponent depComp = depComps[i];
            if (depComp.lb.getValue() != 0 || depComp.ub.getValue() != 0)
              loopCarriedDV[i] = true;
          }
          // Dependence found. No need to check further.
          break;
        }
      }
    }
  }
}

/// Calculates the number of synchronizations needed in this loop permutation.
/// Those permutations having dependence satisfied on inner loops require
/// relatively less number of synchronizations.
uint64_t LoopInterchange::getNumSyncs(ArrayRef<unsigned> perm) {
  uint64_t totalSyncs = 1;
  // Depth at which dependence is satisfied.
  unsigned depDepth = 0;
  for (unsigned i = 0; i < perm.size(); i++) {
    if (!loopCarriedDV[perm[i]])
      continue;
    depDepth = i;
    break;
  }
  for (unsigned j = depDepth + 1; j < perm.size(); j++)
    totalSyncs *= loopIterationCounts[perm[j]];
  return totalSyncs;
}

/// Calculates an upper bound on the number of cache lines accessed in this
/// loop permutation considering only the temporal (and no spatial) reuse of
/// memrefs. A smaller value returned signifies a larger temporal reuse. The 
/// param `maxPossibleReuse` denotes the upper limit on the temporal reuse count.
///  Often it should be equal to the iteration space size of the given loop nest.
uint64_t LoopInterchange::getNumCacheLinesTemporalReuse(
    ArrayRef<unsigned> permutation,
    DenseMap<Operation *, SmallVector<SmallVector<int64_t, 4>, 4>>
        &loopAccessMatrices,
    uint64_t maxPossibleReuse) {

  // Initially, assume no temporal reuse. The cost then represents the fact
  // that we need to access the cache everytime. 
  uint64_t cost = maxPossibleReuse;

  // Start with the innermost loop to check if the access matrix of an op has
  // all zeros in the respective loopIV column. If yes, there is a O(n) reuse.
  // The reuse gets multiplied everytime until the first loop with no reuse is
  // encountered.
  uint64_t actualReuse = 1;
  for (auto &accessMatrixOpPair : loopAccessMatrices) {
    actualReuse = 1;
    SmallVector<SmallVector<int64_t, 4>, 4> accessMatrix =
        accessMatrixOpPair.second;
    for (int i = permutation.size() - 1; i >= 0; i--) {
      bool isColumnAllZeros = true;
      for (SmallVector<int64_t, 4> &row : accessMatrix) {
        if (row[permutation[i]] != 0) {
          isColumnAllZeros = false;
          break;
        }
      }
      if (!isColumnAllZeros)
        break;
      actualReuse *= loopIterationCounts[permutation[i]];
    }
    // An increase in the temporal reuse decreases the cost.
    cost -= actualReuse;
  }
  return cost;
}

/// Removes `dstOp` from its current reference group and places it in the
/// `srcOp's reference group. Updates the `groupId` to reflect these changes.
static void insertIntoReferenceGroup(
    Operation *srcOp, Operation *dstOp,
    DenseMap<Operation *, unsigned> &groupId,
    SmallVector<llvm::SmallSet<Operation *, 8>, 8> &referenceGroups) {
  referenceGroups[groupId[srcOp]].insert(
      referenceGroups[groupId[dstOp]].begin(),
      referenceGroups[groupId[dstOp]].end());
  referenceGroups.erase(referenceGroups.begin() + groupId[dstOp]);
  groupId[dstOp] = groupId[srcOp];
}

/// Groups ops in `loadAndStoreOps` into `referenceGroups` based on whether or
/// not they exhibit group-temporal or group-spatial reuse with respect to the
/// loop present at depth `innermostIndex`.
///
/// Please refer to the paper 'Compiler optimizations for improving data
/// locality' by Steve Carr et. al for a detailed description.
/// https://dl.acm.org/doi/abs/10.1145/195470.195557
static void buildReferenceGroups(
    ArrayRef<Operation *> loadAndStoreOps,
    DenseMap<Operation *, SmallVector<SmallVector<int64_t, 4>, 4>>
        &loopAccessMatrices,
    DenseMap<Operation *, unsigned> &elementSizes, unsigned cacheLineSize,
    unsigned maxLoopDepth, unsigned innermostIndex,
    SmallVector<llvm::SmallSet<Operation *, 8>, 8> &referenceGroups) {

  // Two accesses ref1 and ref2 belong to the same reference group with respect
  // to a loop if :
  // Criteria 1: There exists a dependence l and
  //     1.1 l is a loop-independent dependence or
  //     1.2 l's component for the loop is a small constant d (|d|<=2) and all
  //     other entries are zero.
  // OR
  // Criteria 2: ref1 and ref2 refer to the same array and differ by at most d1
  // in the last subscript dimension, where d1 <= cache line size in terms of
  // the array elements. All other subscripts must be identical.
  //
  // We start with all the accesses having their own group. Thus, if an access
  // is not a part of any group-reuse, it still has it's own group. This counts
  // as a self-spatial reuse.
  referenceGroups.resize(loadAndStoreOps.size());
  // Since we emulate groups using a SmallVector, the `groupID` is used to track
  // the insertions and deletions among the `referenceGroups`.
  DenseMap<Operation *, unsigned> groupId;
  // Initialize each op to its own referenceGroup.
  for (unsigned i = 0; i < loadAndStoreOps.size(); i++) {
    groupId[loadAndStoreOps[i]] = i;
    referenceGroups[i].insert(loadAndStoreOps[i]);
  }
  for (unsigned i = 0; i < loadAndStoreOps.size(); ++i) {
    Operation *srcOp = loadAndStoreOps[i];
    MemRefAccess srcAccess(srcOp);
    for (unsigned j = i + 1; j < loadAndStoreOps.size(); ++j) {
      Operation *dstOp = loadAndStoreOps[j];
      MemRefAccess dstAccess(dstOp);
      if (srcOp == dstOp)
        continue;
      // Criteria 2: Both the ops should access the same memref and both should
      // have the same matrix A. Also, their constant-vectors b should vary only
      // at the last index. Please note that the last column of an acces matrix
      // is the constant-vector b.
      if (srcAccess.memref == dstAccess.memref) {
        bool onlyLastIndexVaries = true;
        for (unsigned row = 0;
             onlyLastIndexVaries && row < loopAccessMatrices[srcOp].size();
             row++) {
          for (unsigned col = 0; col < loopAccessMatrices[srcOp][0].size();
               col++) {
            if (loopAccessMatrices[srcOp][row][col] !=
                    loopAccessMatrices[dstOp][row][col] &&
                (row != loopAccessMatrices[srcOp].size() - 1 ||
                 col != loopAccessMatrices[srcOp][0].size() - 1)) {
              // If the two access matrices vary at any position other than the
              // last row and the last column (the last index of the b-vector),
              // then these ops cannot be grouped together.
              onlyLastIndexVaries = false;
              break;
            }
          }
        }
        if (!onlyLastIndexVaries)
          continue;
        // Even if only the last index of vector b varies, the difference should
        // be less than the cacheLineSize/elementSize for a successful reuse.
        unsigned elementSize = elementSizes[srcOp];
        if (!(abs(loopAccessMatrices[srcOp].back().back() -
                  loopAccessMatrices[dstOp].back().back()) <=
              cacheLineSize / elementSize))
          continue;
        insertIntoReferenceGroup(srcOp, dstOp, groupId, referenceGroups);
      } else {
        // Criteria 1
        for (unsigned depth = 1; depth <= maxLoopDepth + 1; depth++) {
          FlatAffineConstraints dependenceConstraints;
          SmallVector<DependenceComponent, 2> depComps;
          DependenceResult result = checkMemrefAccessDependence(
              srcAccess, dstAccess, depth, &dependenceConstraints, &depComps);
          if (!hasDependence(result))
            continue;
          // Criteria 1.1
          if (depth == maxLoopDepth + 1) {
            // Loop-independent dependence. Put both ops in the same group.
            insertIntoReferenceGroup(srcOp, dstOp, groupId, referenceGroups);
          } else {
            // Criteria 1.2
            // Search for a dependence at depths other than innermostIndex. All
            // other entries should be zero.
            bool hasDependence = false;
            for (unsigned i = 0; i <= maxLoopDepth; i++) {
              if ((depComps[i].lb.getValue() != 0) && (i != innermostIndex)) {
                hasDependence = true;
                break;
              }
            }
            if (hasDependence)
              continue;
            else if (abs(depComps[innermostIndex].lb.getValue()) <= 2)
              insertIntoReferenceGroup(srcOp, dstOp, groupId, referenceGroups);
          }
        }
      }
    }
  }
}

/// Calculates the number of cache lines accessed by each loop of the loop nest
/// if it was the innermost loop. Final values are stored in the private member
/// `cacheLinesAccessCounts`.
///
/// Please refer to the paper 'Compiler optimizations for improving data
/// locality' by Steve Carr et. al for a detailed description.
/// https://dl.acm.org/doi/abs/10.1145/195470.195557
void LoopInterchange::getCacheLineAccessCounts(
    DenseMap<Operation *, SmallVector<SmallVector<int64_t, 4>, 4>>
        &loopAccessMatrices,
    DenseMap<Operation *, unsigned> &elementSizes) {

  // Groups of affine.load/store ops exhibiting group spatial/temporal reuse.
  SmallVector<llvm::SmallSet<Operation *, 8>, 8> refGroups;
  for (unsigned innerloop = 0; innerloop < loopVector.size(); innerloop++) {
    AffineForOp forOp = loopVector[innerloop];
    buildReferenceGroups(loadAndStoreOps, loopAccessMatrices, elementSizes,
                         kCacheLineSize, loopVector.size(), innerloop + 1,
                         refGroups);
    unsigned step = forOp.getStep();
    uint64_t trip =
        (forOp.getConstantUpperBound() - forOp.getConstantLowerBound()) / step +
        1;
    // Represents total number of cache lines accessed during execution of the
    // loop body with this loop acting as the innermost loop.
    uint64_t totalCacheLineCount = 0;
    for (llvm::SmallSet<Operation *, 8> group : refGroups) {
      Operation *op = *group.begin();
      ArrayRef<SmallVector<int64_t, 4>> accessMatrix = loopAccessMatrices[op];
      unsigned stride = step * accessMatrix.back()[innerloop];
      unsigned numEltPerCacheLine = kCacheLineSize / elementSizes[op];
      // Number of cache lines this affine.for op accesses executing this `op`
      // `1` for loop-invariant references,
      // `trip/(cacheLineSize/stride)` for consecutive accesses,
      // `trip` for non-reuse.

      // Start by assuming no-reuse.
      uint64_t cacheLinesForThisOp = trip;
      // Test if this group is loop invariant or loop consecutive. In either
      // case, there is a reuse and hence the number of cache lines accessed
      // will be less than the iteration count (trip) of the loop.
      bool isLoopInvariant = true;
      bool isConsecutive = true;
      for (unsigned j = 0; j < accessMatrix.size(); j++) {
        if (accessMatrix[j][innerloop] != 0) {
          isLoopInvariant = false;
          if (j != accessMatrix.size() - 1)
            isConsecutive = false;
        }
      }
      if (isLoopInvariant)
        cacheLinesForThisOp = 1;
      else if (stride < numEltPerCacheLine && isConsecutive) {
        cacheLinesForThisOp = (trip * stride) / numEltPerCacheLine;
      }
      totalCacheLineCount += cacheLinesForThisOp;
    }
    cacheLinesAccessCounts[&loopVector[innerloop]] = totalCacheLineCount;
  }
}

/// Calculates an upper bound on the number of cache lines accessed in this 
/// loop permutation during the entire execution considering only the spatial
/// (no temporal) reuse of memrefs. A lower value returned implies a better
/// spatial reuse.
uint64_t LoopInterchange::getNumCacheLinesSpatialReuse(ArrayRef<unsigned> perm) {
  uint64_t totalCLAccessed = 0;
  uint64_t iterSubSpaceSize = 1;
  for (int i = 0; i < perm.size(); i++) {
    unsigned numCLThisLoop =
        cacheLinesAccessCounts[&loopVector[perm[i]]];
    // A loop at depth `i` executes `iterSubSpaceSize` number of times. Its each
    // execution consists of `loopIterationCounts[i]` number of iterations and 
    // `numCLThisLoop` number of cache accesses.
    totalCLAccessed += numCLThisLoop * iterSubSpaceSize;
    iterSubSpaceSize *= loopIterationCounts[perm[i]];
  }
  return totalCLAccessed;
}

/// Fills the `bestPerm` with the loop permutation which requires the minimal
/// number of syncs and cache accesses. `loopIndexMap` holds index values for
/// each loopIV in the original loop order. Returns false if the original loop
/// order is found to be the optimal loop permutation.
bool LoopInterchange::getBestPermutation(
    DenseMap<Value, unsigned> &loopIndexMap,
    SmallVector<unsigned, 4> &bestPerm) {
  uint64_t minCost = UINT64_MAX;

  getAllLoadStores();
  DenseMap<Operation *, unsigned> elementSizes;
  getElementSizes(loadAndStoreOps, kDefaultEltSize, elementSizes);
  DenseMap<Operation *, SmallVector<SmallVector<int64_t, 4>, 4>>
      loopAccessMatrices;
  getAffineAccessMatrices(loadAndStoreOps, loopIndexMap, loopAccessMatrices);
  getLoopCarriedDependenceVector();
  getCacheLineAccessCounts(loopAccessMatrices, elementSizes);
  // Calculate the upper limit on the number of temporal reuse counts. This
  // is invariant across all permutations of a given loop nest.
  uint64_t maxTemporalReuse = 1;
  for (auto loopIterC : loopIterationCounts) {
    maxTemporalReuse *= loopIterC;
  }

  // Start testing each loop permutation.
  SmallVector<unsigned, 4> permutation(loopVector.size());
  std::iota(permutation.begin(), permutation.end(), 0);
  unsigned permIndex = 0;
  unsigned bestPermIndex = 0;
  SmallVector<AffineForOp, 4> perfectLoopNest;
  getPerfectlyNestedLoops(perfectLoopNest, loopVector[0]);
  // We make a tradeoff here. For large loop nests having depth more than four, 
  // we permute only the innermost three loops. This tradeoff is necessary because 
  // larger loop nests have an extremely large number of permutations.
  while (std::next_permutation(permutation.size() <5  ? permutation.begin()
                                                      : permutation.end() - 3,
                               permutation.end())) {
    permIndex++;
    if (isValidLoopInterchangePermutation(perfectLoopNest, permutation)) {
      uint64_t numSyncs = getNumSyncs(permutation);
      uint64_t numCLSpatial = getNumCacheLinesSpatialReuse(permutation);
      uint64_t numCLTemporal = getNumCacheLinesTemporalReuse(
          permutation, loopAccessMatrices, maxTemporalReuse);
      // Assumption: Each sync. needs only one memory access. Since num of cycles
      // for one memory access is approx 100x the cycles needed for one cache
      // access, we asssume that cost due to one sync. operation is approx 100x 
      // (we assume 128) than one cache access.
      uint64_t cost = (numSyncs << 7) + numCLSpatial + numCLTemporal;
      if (cost < minCost) {
        minCost = cost;
        bestPermIndex = permIndex;
      }
    }
  }
  // If the original permutation is indeed the best permutation, return false.
  if (!bestPermIndex)
    return false;
  // Iterate again till we get to the best permutation. Since we are working
  // with only the innermost three loops, this does not cost much.
  std::iota(permutation.begin(), permutation.end(), 0);
  while (std::next_permutation(permutation.size() <5 ? permutation.begin()
                                                      : permutation.end() - 3,
                               permutation.end()),
         --bestPermIndex)
    ;
  //`permuteLoops()` (called in next step) maps loop 'i' to the location
  //bestPerm[i]. But here, permutation[i] maps to a loop at depth i. So, we
  // have to reassemble the values in `bestPerm` as required by `permuteLoops`.
  bestPerm.resize(loopVector.size());
  for (unsigned i = 0; i < bestPerm.size(); i++)
    bestPerm[permutation[i]] = i;
}

/// Finds and interchanges the current loop nest to its best possible permutation
/// in order to minimize the number of syncs and the cache accesses. This method 
/// calls the `permuteLoops` method declared in the LoopUtils.h file.
void LoopInterchange::runOnAffineLoopNest() {
  // With a postorder traversal, the affine.for ops are pushed to the `loopVector`
  // in reverse order. We need to reverse this order again to arrange them in the
  // original loop nest order.
  std::reverse(loopVector.begin(), loopVector.end());

  // A map to hold depth indices for all loop IVs of the loop nest. The loopIV
  // for a loop at depth i receives the value i. The map is used to get the
  // locations for each loop IV in rows of the access matrix.
  DenseMap<Value, unsigned> loopIndexMap;
  unsigned loopIndex = 0;
  for (auto &op : loopVector) {
    Value indVar = op.getInductionVar();
    loopIndexMap[indVar] = loopIndex++;
    loopIterationCounts.push_back(
        (op.getConstantUpperBound() - op.getConstantLowerBound()) /
        op.getStep());
  }
  SmallVector<unsigned, 4> bestPermutation;
  // The method `bestPermutation` returns false when the original permutation is
  // the best permutation.
  if (!getBestPermutation(loopIndexMap, bestPermutation))
    return;
  else
    permuteLoops(MutableArrayRef<AffineForOp>(loopVector), bestPermutation);
}

void LoopInterchange::runOnFunction() {
  Operation *function = getFunction().getOperation();
  handleImperfectlyNestedAffineLoops(*function);

  (*function).walk([&](AffineForOp op) {
    loopVector.push_back(op);

    // Check if `op` is the root of some loop nest.
    if ((op.getParentOp()->getName().getStringRef().str() == "func")) {
      // The loop nest should not have any affine-if op and should have a
      // rectangular-shaped iteration space.
      if (!hasAffineIfStatement(op) && isRectangularAffineForLoopNest()) {
        runOnAffineLoopNest();
      }
      // Clear the state for the next loop nest.
      loopVector.clear();
      loopCarriedDV.clear();
      loopIterationCounts.clear();
      cacheLinesAccessCounts.clear();
      loadAndStoreOps.clear();
    }
  });
}

std::unique_ptr<OperationPass<FuncOp>> mlir::createAffineLoopInterchangePass() {
  return std::make_unique<LoopInterchange>();
}

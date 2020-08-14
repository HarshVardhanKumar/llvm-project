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
// so as to minimize the frequence of synchronization. The pass works for both
// perfectly nested and implerfectly nested loops (any level of nesting).
// However in presence of affine.if statements and/or non-rectangular iteration
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

using namespace mlir;
namespace {
struct LoopInterchange : public AffineLoopInterchangeBase<LoopInterchange> {
  void runOnFunction() override;
  void handleImperfectlyNestedAffineLoops(Operation &funcOp);
};
}

/// Returns True if any affine.if op found in the loop nest rooted at `forOp`
static bool hasAffineIfStatement(AffineForOp &forOp) {
  auto walkResult =
      forOp.walk([&](AffineIfOp op) { return WalkResult::interrupt(); });
  return walkResult.wasInterrupted();
}

/// Checks if this `loopNest` has a rectangular shaped iteration space.
static bool isRectangularAffineForLoopNest(ArrayRef<AffineForOp> loopNest) {
  for (AffineForOp forOp : loopNest) {
    if (!forOp.hasConstantUpperBound() || !forOp.hasConstantLowerBound())
      return false;
  }
  return true;
}

/// Returns true if this entire column of `matrix` is zero.
static bool checkColumnIsZero(ArrayRef<SmallVector<int64_t, 4>> matrix,
                              unsigned column) {
  for (const SmallVector<int64_t, 4> &row : matrix) {
    if (row[column] != 0)
      return false;
  }
  return true;
}

/// Fill `row` with the coefficients of loopIVs in `expr`. Any constant terms
/// encountered in `expr` are added to `constantVectorValue`. Every value in
/// `operands` should be a loopIV or a terminal symbol.
static void prepareCoeffientRow(AffineExpr &expr, SmallVector<int64_t, 4> &row,
                                int64_t &constantVectorValue,
                                ArrayRef<Value> operands,
                                DenseMap<Value, unsigned> &loopIndexMap) {
  // TODO: Implement support for terminal symbols in `expr`.
  switch (expr.getKind()) {
  case AffineExprKind::Add: {
    // Is this sub-expr a constant? If yes, no need to modify `row`. Start by
    // assuming the sub-expr is not a constant.
    bool isConstSubExpr = false;
    AffineBinaryOpExpr addExpr = expr.cast<AffineBinaryOpExpr>();
    AffineExpr lhs = addExpr.getLHS();
    AffineExpr rhs = addExpr.getRHS();
    unsigned lhsPosition = 0;
    unsigned rhsPosition = 0;
    // Parse LHS part of affine access expr.
    switch (lhs.getKind()) {
    case AffineExprKind::DimId: {
      auto dim = lhs.cast<AffineDimExpr>();
      lhsPosition = loopIndexMap[operands[dim.getPosition()]];
      break;
    }
    case AffineExprKind::Constant: {
      constantVectorValue += rhs.cast<AffineConstantExpr>().getValue();
      isConstSubExpr = true;
    }
    }
    if (row[lhsPosition] == 0 && !isConstSubExpr)
      row[lhsPosition] = 1;
    // Parse RHS part of affine access expr. Again assume this sub-expr is also
    // not a constant expr.
    isConstSubExpr = false;
    switch (rhs.getKind()) {
    case AffineExprKind::DimId: {
      auto dimExpr = rhs.cast<AffineDimExpr>();
      rhsPosition = loopIndexMap[operands[dimExpr.getPosition()]];
      break;
    }
    case AffineExprKind::Constant: {
      // If the RHS is a constant, add this constant to B[l].
      constantVectorValue += rhs.cast<AffineConstantExpr>().getValue();
      isConstSubExpr = true;
    }
    }
    if (row[rhsPosition] == 0 && !isConstSubExpr)
      row[rhsPosition] = 1;
    break;
  }
  case AffineExprKind::Mul: {
    AffineBinaryOpExpr mulExpr = expr.cast<AffineBinaryOpExpr>();
    AffineExpr lhs = mulExpr.getLHS();
    AffineExpr rhs = mulExpr.getRHS();
    unsigned dimIdPos = 0;
    // Parse LHS part of affine binary expr.
    if (lhs.isa<AffineDimExpr>()) {
      auto dim = lhs.cast<AffineDimExpr>();
      dimIdPos = loopIndexMap[operands[dim.getPosition()]];
    }
    // RHS in this case should always be a constant.
    if (rhs.isa<AffineConstantExpr>()) {
      row[dimIdPos] = rhs.cast<AffineConstantExpr>().getValue();
    }
    break;
  }
  case AffineExprKind::DimId: {
    auto dim = expr.cast<AffineDimExpr>();
    row[loopIndexMap[operands[dim.getPosition()]]] = 1;
    constantVectorValue += 0;
    break;
  }
  case AffineExprKind::CeilDiv:
  case AffineExprKind::FloorDiv:
  case AffineExprKind::Mod: {
    auto modExpr = expr.cast<AffineBinaryOpExpr>();
    AffineExpr lhs = modExpr.getLHS();
    AffineExpr rhs = modExpr.getRHS();
    if (lhs.isa<AffineDimExpr>()) {
      auto dim = lhs.cast<AffineDimExpr>();
      row[loopIndexMap[operands[dim.getPosition()]]] = 1;
    }
    // RHS in this case is always a constant or a symbol. For a constant, we
    // don't need to modify the access matrix.
  }
  }
}

/// For a memref access function AX+B, it calculates both A and B and stores
/// to `loopAccessMatrices` (collection of As) and `constVector`(collection of
/// Bs). The param `loopIndexMap` is used to locate position for coefficients of
/// loopIVs in each row of matrix A.
static void getAffineAccessMatrices(
    AffineForOp &rootForOp, SmallVector<Operation *, 8> &loadAndStoreOps,
    DenseMap<Value, unsigned> &loopIndexMap,
    DenseMap<Operation *, SmallVector<SmallVector<int64_t, 4>, 4>>
        &loopAccessMatrices,
    DenseMap<Operation *, SmallVector<int64_t, 4>> &constVector,
    unsigned AffineForOpLoopNestSize) {

  unsigned numOps = loadAndStoreOps.size();
  for (unsigned i = 0; i < numOps; ++i) {
    Operation *srcOp = loadAndStoreOps[i];
    MemRefAccess srcAccess(srcOp);
    AffineMap map;
    if (isa<AffineLoadOp>(srcOp))
      map = cast<AffineLoadOp>(srcOp).getAffineMap();
    else if (isa<AffineStoreOp>(srcOp))
      map = cast<AffineStoreOp>(srcOp).getAffineMap();
    SmallVector<Value, 8> operands(srcAccess.indices.begin(),
                                   srcAccess.indices.end());
    fullyComposeAffineMapAndOperands(&map, &operands);
    map = simplifyAffineMap(map);
    canonicalizeMapAndOperands(&map, &operands);
    // Parse each map result to construct access matrices.
    ArrayRef<AffineExpr> mapResults = map.getResults();
    // Number of rows in an accessMatrix = Number of dimensions in the memref
    // object. Number of Columns = (noDimIDs + noSymbols).
    loopAccessMatrices[srcOp].resize(mapResults.size());
    constVector[srcOp].resize(mapResults.size());
    for (unsigned l = 0; l < mapResults.size(); l++) {
      // Parse the l-th map result(access expr for l-th dim of this memref) to
      // get the l-th row of this access matrix.
      AffineExpr mapResult = mapResults[l];
      loopAccessMatrices[srcOp][l].resize(std::max(
          AffineForOpLoopNestSize, map.getNumDims() + map.getNumSymbols()));
      // Check if `mapResult` is a constant expr. If it is a constant expr, no
      // need to walk it. Instead, add the value to constVector and leave
      // the row vector to be zeroes.
      if (mapResult.isa<AffineConstantExpr>()) {
        auto constExpr = mapResult.cast<AffineConstantExpr>();
        constVector[srcOp][l] = constExpr.getValue();
      } else {
        // Start parsing the mapResult.
        mapResult.walk([&](AffineExpr expr) {
          // Each expr can in turn be a combination of many affine expressions.
          prepareCoeffientRow(expr, loopAccessMatrices[srcOp][l],
                              constVector[srcOp][l], operands, loopIndexMap);
        });
      }
    }
  }
}

/// Separates `forOpA` from its siblings. After the separation,`forOpA` receives
/// a copy of its parent independent from other siblings. A loop nest such as:
/// \code
///     parent{forOpA, forOpB, forOpC}
/// \endcode
/// becomes
/// \code
///     parent{forOpA}, parent{forOpB, forOpC}
/// \endcode
static void separateSiblingAffineForOps(AffineForOp &parentForOp,
                                        AffineForOp &forOpA,
                                        SmallVector<AffineForOp, 4> &siblings) {

  OpBuilder builder(parentForOp.getOperation()->getBlock(),
                    std::next(Block::iterator(parentForOp)));
  AffineForOp copyParentForOp = cast<AffineForOp>(builder.clone(*parentForOp));

  // Note the order in which `forOpA` and all other siblings are visited. We
  // need this order to compare affine.for ops within `parentForOp` with their
  // copy in `copyParentForOp`. Comparing forOp.getOperation() does not work in
  // that case.
  unsigned forOpAPosition = 0;
  llvm::SmallSet<unsigned, 8> siblingsIndices;
  unsigned index = 0;
  parentForOp.getOperation()->walk([&](AffineForOp op) {
    index++;
    if (op.getOperation() == forOpA.getOperation())
      forOpAPosition = index;
    for (unsigned i = 0; i < siblings.size(); i++)
      if (op.getOperation() == siblings[i].getOperation())
        siblingsIndices.insert(index);
  });
  // Walk the copy of parentOp to erase all siblings other than `forOpA`.
  index = 0;
  copyParentForOp.getOperation()->walk([&](AffineForOp op) {
    index++;
    if (index != forOpAPosition && siblingsIndices.count(index))
      op.getOperation()->erase();
  });
  // Erase `forOpA` from the original copy.
  forOpA.getOperation()->erase();
}

/// Converts all imperfectly nested loop nests in `funcOp` to perfectly 
/// nested loop nests by loop splitting.
void LoopInterchange::handleImperfectlyNestedAffineLoops(Operation &funcOp) {
  SmallVector<AffineForOp, 4> loopNest;
  DenseMap<Operation *, SmallVector<AffineForOp, 4>> forTree;
  DenseMap<Operation *, AffineForOp> forOperations;

  // Stop splitting when each parent has only one child left.
  bool oneChild = false;
  while (!oneChild) {
    oneChild = true;
    // Walk the function to create a tree of affine.for operations.
    funcOp.walk([&](AffineForOp op) {
      loopNest.push_back(op);
      if (op.getParentOp()->getName().getStringRef() == "affine.for")
        forTree[op.getOperation()->getParentOp()].push_back(op);

      forOperations[op.getOperation()] = op;
    });
    // Separate one of the sibling at a time.
    for (auto &loopNest : forTree) {
      // This loop nest has no sibling problem. Check the next loop nest.
      if (loopNest.second.size() < 2)
        continue;
      oneChild = false;
      separateSiblingAffineForOps(forOperations[loopNest.first],
                                  loopNest.second.back(), loopNest.second);
      // We need to walk the function again since the structure of loop nests
      // within the funcOp body has changed.
      break;
    }
    loopNest.clear();
    forTree.clear();
    forOperations.clear();
  }
  return;
}

/// Scans the loop nest rooted at `rootForOp` and collects all affine.load and
/// affine.store ops. Fills `loadAndStoreOps` with all such ops.
static void getAllLoadStores(AffineForOp rootForOp,
                             SmallVector<Operation *, 8> &loadAndStoreOps) {
  rootForOp.getOperation()->walk([&](Operation *op) {
    if (isa<AffineLoadOp>(op) || isa<AffineStoreOp>(op)) {
      loadAndStoreOps.push_back(op);
    }
  });
}

/// Fills `elementsSize` with size of element types of respective memrefs
/// accessed by respective ops in `loadAndStoreOps`. These will be used to 
/// check if two accesses are within a cache_line_size/element_size distance
/// apart for a useful locality.
static void getElementsSize(SmallVector<Operation *, 8> &loadAndStoreOps,
                            DenseMap<Operation *, unsigned> &elementsSize) {
  MemRefType memrefType;
  for (Operation *op : loadAndStoreOps) {
    if (isa<AffineLoadOp>(op)) {
      AffineLoadOp loadOp = cast<AffineLoadOp>(*op);
      memrefType = loadOp.getMemRefType();
    } else if (isa<AffineStoreOp>(op)) {
      AffineStoreOp storeOp = cast<AffineStoreOp>(*op);
      memrefType = storeOp.getMemRefType();
    }
    // If the memref has a static shape, obtain the element size. Otherwise
    // consider a default value.
    elementsSize[op] = memrefType.hasStaticShape()
                           ? getMemRefSizeInBytes(memrefType).getValue() /
                                 memrefType.getNumElements()
                           : 8;
  }
}

/// Fills `validPermutations` with all valid loop interchange permutations of the
/// loop nest rooted at `rootForOp`.
/// [Theorem] A permutation of the loops in a perfect nest is legal if and only
/// if the direction matrix, after the same permutation is applied to its
/// columns, has no ">" direction as the leftmost non-"=" direction in any row.
static void getAllValidLoopInterchangePermutations(
    AffineForOp &rootForOp, unsigned noLoopsInNest,
    SmallVector<SmallVector<unsigned, 4>, 8> &validPermutations) {

  SmallVector<unsigned, 4> permutation;
  for (unsigned i = 0; i < noLoopsInNest; i++)
    permutation.push_back(i);

  validPermutations.push_back(permutation);
  SmallVector<AffineForOp, 4> perfectLoopNest;
  getPerfectlyNestedLoops(perfectLoopNest, rootForOp);
  while (std::next_permutation(permutation.begin(), permutation.end()))
    if (isValidLoopInterchangePermutation(perfectLoopNest, permutation))
      validPermutations.push_back(permutation);
}

/// Calculates loop-carried-dependence vector for this loop nest rooted at
/// `rootForOp`. Result is stored in  `loopCarriedDependenceVector`.
static void getLoopCarriedDependenceVector(
    AffineForOp &rootForOp, ArrayRef<Operation *> loadAndStoreOps,
    SmallVector<bool, 4> &loopCarriedDependenceVector, unsigned loopNestSize) {

  unsigned numOps = loadAndStoreOps.size();
  // Resize `loopCarriedDependenceVector` to fit entire loop nest.
  loopCarriedDependenceVector.resize(loopNestSize);
  for (unsigned i = 0; i < numOps; ++i) {
    Operation *srcOp = loadAndStoreOps[i];
    for (unsigned j = 0; j < numOps; ++j) {
      Operation *dstOp = loadAndStoreOps[j];
      for (unsigned depth = 1; depth <= loopNestSize + 1; ++depth) {
        MemRefAccess srcAccess(srcOp), dstAccess(dstOp);
        FlatAffineConstraints dependenceConstraints;
        SmallVector<DependenceComponent, 2> depComps;
        DependenceResult result = checkMemrefAccessDependence(
            srcAccess, dstAccess, depth, &dependenceConstraints, &depComps);
        if (hasDependence(result)) {
          for (unsigned i = 0; i < depComps.size(); i++) {
            DependenceComponent depComp = depComps[i];
            if (depComp.lb.getValue() != 0 || depComp.ub.getValue() != 0)
              loopCarriedDependenceVector[i] = true;
          }
          // Dependence found. No need to check further.
          break;
        }
      }
    }
  }
}

/// Calculates a representative cost of this permutation for parallelism on
/// multicores. A permutation having more free outer loops gets a smaller cost.
static uint64_t getParallelismCost(ArrayRef<unsigned> perm,
                                   ArrayRef<bool> loopCarriedDV,
                                   ArrayRef<unsigned> iterCounts) {
  uint64_t totalcost = 0;
  uint64_t thisLoopcost = 1;
  for (unsigned i = 0; i < perm.size(); i++) {
    if (!loopCarriedDV[perm[i]])
      continue;
    thisLoopcost = 1;
    for (unsigned j = i + 1; j < perm.size(); j++)
      thisLoopcost *= iterCounts[perm[j]];
    totalcost += thisLoopcost;
  }
  return totalcost;
}

/// Removes `dstOp` from its current group and inserts it into `srcOp's`
/// group. Updates `groupId` to reflect the changes.
static void insertIntoReferenceGroup(
    SmallVector<llvm::SmallSet<Operation *, 8>, 8> &referenceGroups,
    DenseMap<Operation *, unsigned> &groupId, Operation *srcOp,
    Operation *dstOp) {
  referenceGroups[groupId[srcOp]].insert(
      referenceGroups[groupId[dstOp]].begin(),
      referenceGroups[groupId[dstOp]].end());
  referenceGroups.erase(referenceGroups.begin() + groupId[dstOp]);
  // Insert operation results in same group-id for both instructions.
  groupId[dstOp] = groupId[srcOp];
}

/// Groups ops in `loadAndStoreOps` into `referenceGroups` based on whether or
/// not they exibit group-temporal or group-spatial reuse with respect to an
/// affine.for op present at `innermostIndex` in the original loop nest.
///
/// Please refer Steve Carr et. al for a detailed description.
/// https://dl.acm.org/doi/abs/10.1145/195470.195557
static void buildReferenceGroups(
    SmallVector<Operation *, 8> &loadAndStoreOps,
    DenseMap<Operation *, SmallVector<SmallVector<int64_t, 4>, 4>>
        &loopAccessMatrices,
    DenseMap<Operation *, SmallVector<int64_t, 4>> &constVector,
    DenseMap<Operation *, unsigned> &elementsSize, unsigned maxDepth,
    unsigned innermostIndex,
    SmallVector<llvm::SmallSet<Operation *, 8>, 8> &referenceGroups) {

  // Two accesses ref1 and ref2 belong to same reference group with respect
  // to a loop if :
  // Criteria 1: There exists a dependence l and
  //     1.1 l is a loop-independent dependence or
  //     1.2 l's component for `forOp` is a small constant d (|d|<=2) and all
  //     other entries are zero.
  // OR
  // Criteria 2: ref1 and ref2 refer to the same array and differ by at most d1
  // in the last subscript dimension, where d1 <= cache line size in terms of
  // array elements. All other subscripts must be identical.
  //
  // This implementation starts with all accesses having their own group. Thus,
  // if an access is not part of any group-reuse, it still has it's own group.
  // Doing this takes care of self-spatial reuse.
  constexpr unsigned cache_line_size = 64;
  unsigned numOps = loadAndStoreOps.size();
  referenceGroups.resize(numOps);
  // Since we emulate groups using SmallVector, `groupID` is used to track
  // insertions/deletions among `referenceGroups`.
  DenseMap<Operation *, unsigned> groupId;
  // Initialize each op to its own referenceGroup.
  for (unsigned i = 0; i < loadAndStoreOps.size(); i++) {
    groupId[loadAndStoreOps[i]] = i;
    referenceGroups[i].insert(loadAndStoreOps[i]);
  }
  for (unsigned i = 0; i < numOps; ++i) {
    Operation *srcOp = loadAndStoreOps[i];
    MemRefAccess srcAccess(srcOp);
    for (unsigned j = i + 1; j < numOps; ++j) {
      Operation *dstOp = loadAndStoreOps[j];
      MemRefAccess dstAccess(dstOp);
      if (srcOp == dstOp)
        continue;
      // Criteria 2
      if (srcAccess.memref == dstAccess.memref) {
        // For two memref accesses Ax+B, matrix A of both accesses should be
        // equal. Only B should differ in last index.
        if (loopAccessMatrices[srcOp] != loopAccessMatrices[dstOp])
          continue;
        SmallVector<int64_t, 4> srcOpCV = constVector[srcOp];
        SmallVector<int64_t, 4> destOpCV = constVector[dstOp];
        bool onlyLastIndexVaries = true;
        for (unsigned i = 0; i < srcOpCV.size() - 1; i++) {
          if ((srcOpCV[i] != destOpCV[i])) {
            onlyLastIndexVaries = false;
            break;
          }
        }
        if (!onlyLastIndexVaries)
          continue;
        // Difference in values in last index should be less than
        // cache_line_size/elementSize for a useful locality.
        unsigned elementSize = elementsSize[srcOp];
        if (!(abs(srcOpCV.back() - destOpCV.back()) <=
              cache_line_size / elementSize))
          continue;
        // Insert `dstOp` into the group of `srcOp`.
        insertIntoReferenceGroup(referenceGroups, groupId, srcOp, dstOp);
      } else {
        // Criteria 1
        for (unsigned depth = 1; depth <= maxDepth + 1; depth++) {
          FlatAffineConstraints dependenceConstraints;
          SmallVector<DependenceComponent, 2> depComps;
          DependenceResult result = checkMemrefAccessDependence(
              srcAccess, dstAccess, depth, &dependenceConstraints, &depComps);
          if (!hasDependence(result))
            continue;
          // Criteria 1.1
          if (depth == maxDepth + 1) {
            // Loop-independent dependence. Put both ops in same group.
            insertIntoReferenceGroup(referenceGroups, groupId, srcOp, dstOp);
          } else {
            // Criteria 1.2
            // Search for dependence at depths other than innermostIndex. All
            // entries should be zero.
            bool hasDependence = false;
            for (unsigned i = 0; i <= maxDepth; i++) {
              if ((depComps[i].lb.getValue() != 0) && (i != innermostIndex)) {
                hasDependence = true;
                break;
              }
            }
            if (hasDependence)
              continue;
            else if (abs(depComps[innermostIndex].lb.getValue()) <= 2)
              insertIntoReferenceGroup(referenceGroups, groupId, srcOp, dstOp);
          }
        }
      }
    }
  }
}

/// Calculates number of cache lines accessed by each affine.for op in the
/// `loopNest` if that affine.for is considered the innermost loop. Final
/// values are stored in 'cacheLinesAccessCounts'.
///
/// Please refer Steve Carr et. al for a detailed description.
/// https://dl.acm.org/doi/abs/10.1145/195470.195557
static void getCacheLineAccessCounts(
    ArrayRef<AffineForOp> loopNest,
    SmallVector<Operation *, 8> &loadAndStoreOps,
    DenseMap<Operation *, SmallVector<SmallVector<int64_t, 4>, 4>>
        &loopAccessMatrices,
    DenseMap<Operation *, SmallVector<int64_t, 4>> &constVector,
    DenseMap<Operation *, unsigned> &elementsSize,
    DenseMap<const AffineForOp *, long double> &cacheLinesAccessCounts) {

  unsigned loopNestSize = loopNest.size();
  // Group of affine.load/store ops exibiting group spatial/temporal reuse.
  SmallVector<llvm::SmallSet<Operation *, 8>, 8> refGroups;
  for (unsigned innerloop = 0; innerloop < loopNestSize; innerloop++) {
    AffineForOp forOp = loopNest[innerloop];
    // Build reference groups considering each `forOp` to be the innermost loop.
    buildReferenceGroups(loadAndStoreOps, loopAccessMatrices, constVector,
                         elementsSize, loopNestSize, innerloop + 1, refGroups);
    unsigned step = forOp.getStep();
    uint64_t trip =
        (forOp.getConstantUpperBound() - forOp.getConstantLowerBound()) / step +
        1;
    // `totalCacheLineCount` represents overall number of cache lines accessed
    // with this loop as the inner most loop.
    uint64_t totalCacheLineCount = 0;
    for (llvm::SmallSet<Operation *, 8> group : refGroups) {
      Operation *op = *group.begin();
      ArrayRef<SmallVector<int64_t, 4>> accessMatrix = loopAccessMatrices[op];
      unsigned stride = step * accessMatrix.back()[innerloop];

      // Number of cache lines this affine.for op accesses executing this op
      // `1` for loop-invariant references,
      // `trip/(cache_line_size/stride)` for consecutive accesses,
      // `trip` for non-reuse.

      // Start by assuming no-reuse.
      uint64_t cacheLinesForThisOp = trip;
      // Test if this group is loop invariant or loop consecutive. In either of
      // these cases, there is a reuse and hence number of cache lines accessed
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
      else if (stride < 8 && isConsecutive) {
        // Assumption: cache_line_size/element size = 8.
        // TODO: Implement for actual size of the elements in memref.
        cacheLinesForThisOp = (trip * stride) / 8;
      }
      totalCacheLineCount += cacheLinesForThisOp;
    }
    cacheLinesAccessCounts[&loopNest[innerloop]] = totalCacheLineCount;
  }
}

/// Calculates a representative temporal reuse cost for a given permutation of
/// the loop nest. A lower value returned means higher temporal reuse.
static uint64_t getTemporalReuseCost(
    ArrayRef<unsigned> permutation, ArrayRef<unsigned> loopIterationCounts,
    DenseMap<Operation *, SmallVector<SmallVector<int64_t, 4>, 4>>
        &loopAccessMatrices) {
  uint64_t cost = 1;
  // Initially we assume no temporal reuse, hence the cost is a big value
  // (arbitrary chosen to be the size of the entire iteration space of the loop
  // nest).
  for (unsigned iterCount : loopIterationCounts)
    cost *= iterCount;

  // Start from innermost loop to check if the access matrix for an op has all
  // zeros in the respective column. If yes, there is a O(n) reuse. The reuse
  // gets multiplied for each loop index until the first loop index with no
  // reuse is encountered.
  uint64_t temporalReuse = 1;
  for (auto &accessMatrixOpPair : loopAccessMatrices) {
    temporalReuse = 1;
    SmallVector<SmallVector<int64_t, 4>, 4> accessMatrix =
        accessMatrixOpPair.second;
    for (int i = permutation.size() - 1; i >= 0; i--) {
      if (!checkColumnIsZero(accessMatrix, permutation[i]))
        break;
      temporalReuse *= loopIterationCounts[permutation[i]];
    }
    // Increase in temporalReuse decreases the cost.
    cost -= temporalReuse;
  }
  return cost;
}

/// Calculates a representative cost of this permutation considering spatial
/// locality. Lower cost implies better spatial reuse. Uses the fact that loops
/// with smaller locality at inner positions promote more reuse.
static uint64_t getSpatialLocalityCost(
    ArrayRef<unsigned> perm, SmallVector<AffineForOp, 4> &loopNest,
    DenseMap<const AffineForOp *, long double> cacheLineCounts,
    ArrayRef<unsigned> loopIterCounts) {

  uint64_t auxiliaryCost = 0;
  // Product of iteration count of inner loops.
  uint64_t iterSubSpaceSize = 1;
  // Maximum cache lines accessed by any affine.for op in the loop nest. Helpful
  // in calculating sentinel - see below.
  unsigned maxCacheLinesAccessed = 0;
  for (int i = perm.size() - 1; i >= 0; i--) {
    // Number of cache lines accessed by this loop at index i
    unsigned numberCacheLinesAccessed = cacheLineCounts[&loopNest[perm[i]]];
    if (numberCacheLinesAccessed > maxCacheLinesAccessed)
      maxCacheLinesAccessed = numberCacheLinesAccessed;
    auxiliaryCost += numberCacheLinesAccessed * iterSubSpaceSize;
    iterSubSpaceSize *= loopIterCounts[perm[i]];
  }

  // The optimal permutation is one in which the affine.for ops are arranged in
  // descending order of their cache line access counts from left to right.

  // However such a permutation will have the maximum 'auxiliaryCost' value. But
  // we want a minimum cost for such a permutation.

  // For this, we subtract the `auxiliaryCost` value from a sentinel value. We
  // define sentinel value as follows: sentinel =
  // Sum_for_all_loops(maxCacheLineAccessed * iterSubSpaceSize of each loop).
  // Since the sentinel does not depend on any permutation of the loop nest, we
  // can be sure that the cost obtained is consistent across permutations.
  uint64_t sentinel = 0;
  iterSubSpaceSize = 1;
  for (unsigned iterCount : loopIterCounts) {
    sentinel += maxCacheLinesAccessed * iterSubSpaceSize;
    iterSubSpaceSize *= iterCount;
  }
  return sentinel - auxiliaryCost;
}

/// Fills `bestPerm` with the optimal permutation considering both the locality
/// cost and the parallelism cost. If the current permutation is the best
/// permutation for this loop nest, the method returns false.
static bool getBestPermutation(SmallVector<AffineForOp, 4> &loopNest,
                               SmallVector<unsigned, 4> &loopIterationCounts,
                               DenseMap<Value, unsigned> &loopIndexMap,
                               SmallVector<unsigned, 4> &bestPerm) {
  uint64_t minCost = UINT64_MAX;
  SmallVector<Operation *, 8> loadAndStoreOps;
  // Get all affine.load and affine.store ops.
  getAllLoadStores(loopNest[0], loadAndStoreOps);

  DenseMap<Operation *, unsigned> elementsSize;
  // Get size of elements (in bytes) in each memref access. Later to be used
  // to build reference groups.
  getElementsSize(loadAndStoreOps, elementsSize);

  // Now calculate affine access matrices for all load/store ops in this loop
  // nest. The access matrices are needed to get both temporal reuse cost and
  // spatial reuse cost.

  // For each memref access function Ax+B, this is  the collection of all A's
  // indexed by their respective affine.load/affine.store op.
  DenseMap<Operation *, SmallVector<SmallVector<int64_t, 4>, 4>>
      loopAccessMatrices;
  // For a memref access function Ax+B, this represents B.
  DenseMap<Operation *, SmallVector<int64_t, 4>> constVector;
  // Get affine access matrices for all load/stores.
  getAffineAccessMatrices(loopNest[0], loadAndStoreOps, loopIndexMap,
                          loopAccessMatrices, constVector, loopNest.size());

  // Loop Carried Dependence vector. A 'true' at index 'i' means loop at depth
  // 'i' carries a dependence. This is useful in calculating parallelism cost
  // for each permutation.
  SmallVector<bool, 4> loopCarriedDV;
  getLoopCarriedDependenceVector(loopNest[0], loadAndStoreOps, loopCarriedDV,
                                 loopNest.size());

  // Number of cache lines accessed (locality) by each affine.for op if it was
  // the innermost loop.
  DenseMap<const AffineForOp *, long double> cacheLinesAccessCounts;
  // Get locality information for each affine.forop in the loop nest. This will
  // be useful in calculating spatial locality cost for each permutation.
  getCacheLineAccessCounts(loopNest, loadAndStoreOps, loopAccessMatrices,
                           constVector, elementsSize, cacheLinesAccessCounts);

  // Get all permutations which do not violate any dependence constraints.
  SmallVector<SmallVector<unsigned, 4>, 8> validPerms;
  getAllValidLoopInterchangePermutations(loopNest[0], loopNest.size(),
                                         validPerms);

  // Return false to indicate that no other valid permutation exists.
  if (validPerms.size() < 2)
    return false;
  unsigned bestPermIndex = 0;
  for (unsigned i = 0; i < validPerms.size(); i++) {
    uint64_t parallelCost =
        getParallelismCost(validPerms[i], loopCarriedDV, loopIterationCounts);
    uint64_t spatialCost = getSpatialLocalityCost(
        validPerms[i], loopNest, cacheLinesAccessCounts, loopIterationCounts);
    uint64_t temporalCost = getTemporalReuseCost(
        validPerms[i], loopIterationCounts, loopAccessMatrices);

    // Assumption: costs due to parallelism (synchronization) are 100x
    // more expensive than those due to locality.
    uint64_t cost = 100 * parallelCost + spatialCost + temporalCost;
    if (cost < minCost) {
      minCost = cost;
      bestPermIndex = i;
    }
  }

  // Check if best permutation is the original permutation. In that
  // case, return false.
  if (!bestPermIndex)
    return false;

  //`permuteLoops()` (called in next step) maps loop 'i' to location
  // bestPerm[i]. But here validPerms[bestPermIndex][i] maps to loop
  // at depth 'i'. Hence, we need to reassemble values in bestPerm[i]
  // as required for `permuteLoops()`.
  bestPerm.resize(loopNest.size());
  for (unsigned i = 0; i < bestPerm.size(); i++)
    bestPerm[validPerms[bestPermIndex][i]] = i;
}

/// Finds and permutes the `loopVector` to the best possible permutation
/// considering locality and parallelism of this loop nest.
void runOnAffineLoopNest(SmallVector<AffineForOp, 4> &loopVector) {

  // With a postorder traversal, affine.forops in `loopVector`
  // are pushed in reverse order. We need to reverse this order to
  // arrange them in loop nest order.
  std::reverse(loopVector.begin(), loopVector.end());

  // A map to hold index values for all affine.for IVs in the loop nest.
  // The map serves two purposes - It helps to maintain these index values
  // across various permutations and it is used to get locations for each IV
  // Value object in a coefficient row of the access matrix.
  DenseMap<Value, unsigned> loopIndexMap;
  unsigned loopIndex = 0;
  // Iteration count for each affine.for in this loop nest.
  SmallVector<unsigned, 4> loopIterationCounts;
  for (auto &op : loopVector) {
    // Assign IDs to loop variables - used later to populate access-matrix
    // for each memref.
    Value indVar = op.getInductionVar();
    loopIndexMap[indVar] = loopIndex++;
    // Populate `loopIterationCounts` of each for loop in loop nest.
    loopIterationCounts.push_back(
        (op.getConstantUpperBound() - op.getConstantLowerBound()) /
        op.getStep());
  }
  SmallVector<unsigned, 4> bestPermutation;
  // `bestPermutation` returns false when the original permutation is the best
  // permutation.
  if (!getBestPermutation(loopVector, loopIterationCounts, loopIndexMap,
                          bestPermutation))
    return;
  else
    permuteLoops(MutableArrayRef<AffineForOp>(loopVector), bestPermutation);
}

void LoopInterchange::runOnFunction() {
  Operation *function = getFunction().getOperation();
  handleImperfectlyNestedAffineLoops(*function);

  SmallVector<AffineForOp, 4> loopVector;
  (*function).walk([&](AffineForOp op) {
    loopVector.push_back(op);

    // Check if `op` is a root of some loop nest.
    if ((op.getParentOp()->getName().getStringRef().str() == "func")) {
      // The loop nest should not have any affine.if op and should have
      // rectangular shaped iteration space.
      if (!hasAffineIfStatement(op) &&
          isRectangularAffineForLoopNest(loopVector)) {
        runOnAffineLoopNest(loopVector);
        // Clear for next loop nest.
        loopVector.clear();
      }
    }
  });
}

std::unique_ptr<OperationPass<FuncOp>> mlir::createAffineLoopInterchangePass() {
  return std::make_unique<LoopInterchange>();
}

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
// perfectly nested and implerfectly nested loops (any level of nesting). However
// in presence of affine.if statements and/or non-rectangular iteration space,
// the pass simply bails out - leaving the original loop nest unchanged.
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

protected:
  void getAffineAccessMatrices(
      AffineForOp &rootForOp,
      SmallVector<Operation*,8> &loadAndStoreOps,
      DenseMap<Value, int64_t> &loopVarIdMap,
      unsigned AffineForOpLoopNestSize
      );

  void getAffineLoopCarriedDependencies(AffineForOp &rootForOp,
      ArrayRef<Operation*> loadAndStoreOps,
      unsigned maxLoopDepth);

  void getElementsSize(AffineForOp &rootForOp);

  void buildReferenceGroups(AffineForOp innermostForOp, 
      SmallVector<Operation*,8> &loadAndStoreOps,
      unsigned maxDepth, unsigned forOpLevelInOriginalLoopNest);
  
  void getCacheLineAccessCounts(
      ArrayRef<AffineForOp> AffineLoopNest,
      SmallVector<Operation*,8>&loadAndStoreOps);
  
  long double getTemporalReuseCost(ArrayRef<unsigned> permutation);
  
  SmallVector<unsigned,0> getBestPermutation(
      SmallVector<AffineForOp,0> &AffineLoopNest,
      DenseMap<Value, int64_t>&loopVarIdMap);
  
  void clear(SmallVector<AffineForOp, 0> &AffineForOpLoopNest,
             DenseMap<Value, int64_t> &loopVarIdMap, 
      int *loopVarIdCount);

  constexpr static int cache_line_size = 64;

  // For each MemRef access function Ax+B, it is
  // the collection of A's indexed by respective
  // load/store operation.
  DenseMap<Operation *, SmallVector<std::vector<int64_t>, 0>>
      loopAccessMatrices;

  // For a MemRef access function Ax+B, this represents B.
  DenseMap<Operation *, std::vector<int64_t>>
      constVector; 

  // Element size in a given MemRef.
  DenseMap<Operation *, unsigned>
      dataSize; 

  // Iteration count for each affine.for in the loop nest.
  SmallVector<unsigned, 2>
      loopIterationCounts; 

  // Loop Carried Dependency vector. A 'true'
  // at index i means loop at depth i carries a dependence.
  SmallVector<bool,2>
      loopCarriedDependenceVector; 

  // Number of cache lines accessed (locality) by each
  // affine.for if it was the innermost loop in the loop nest.
  DenseMap<const AffineForOp *, long double>
      numberOfCacheLinesAccessed; 

  // affine.load/affine.store ops exibiting group-spatial/group-temporal 
  // reuse.
  SmallVector<llvm::SmallSet<Operation *,8>,8> referenceGroups;
};
}

/// Returns True if any affine.if op found in the loop nest rooted 
/// at `forOp`
static bool hasAffineIfStatement(AffineForOp &forOp) {
  auto walkResult =
      forOp.walk([&](AffineIfOp op) { return WalkResult::interrupt(); });
  return walkResult.wasInterrupted();
}

/// Checks if this loop nest has a rectangular shaped iteration
/// space.
static bool isRectangularAffineForLoopNest(
    ArrayRef<AffineForOp> loopNest) {
  for (AffineForOp forOp : loopNest) {
    if (!forOp.hasConstantUpperBound() || !forOp.hasConstantLowerBound())
      return false;
  }
  return true;
}

/// Returns true if this entire column of `matrix` is zero.
static bool checkColumnIsZero(ArrayRef<std::vector<int64_t>> matrix,
                       unsigned column) {
  for (std::vector<int64_t> row : matrix) {
    if (row[column] != 0)
      return false;
  }
  return true;
}

/// For a MemRef access function AX+B, it calculates both A and B and 
/// assigns these values to protected members `loopAccessMatrices`(A) and 
/// `constVector`(B). The param `loopVarIdMap` is used to locate position
/// for coefficients of X in each row of matrix A.
void LoopInterchange::getAffineAccessMatrices(
    AffineForOp &rootForOp,
    SmallVector<Operation*,8> &loadAndStoreOps,
    DenseMap<Value, int64_t> &loopVarIdMap,
    unsigned AffineForOpLoopNestSize) {
  
  this->loopAccessMatrices.clear();
  this->constVector.clear();

  unsigned numOps = loadAndStoreOps.size();
  for (unsigned i = 0; (i < numOps); ++i) {
    Operation *srcOp = loadAndStoreOps[i];
    MemRefAccess srcAccess(srcOp);
    AffineMap map;
    if (AffineLoadOp loadOp = dyn_cast<AffineLoadOp>(srcAccess.opInst))
      map = loadOp.getAffineMap();
    else if (AffineStoreOp storeOp = dyn_cast<AffineStoreOp>(srcAccess.opInst))
      map = storeOp.getAffineMap();
    SmallVector<Value, 8> operands(srcAccess.indices.begin(),
                                   srcAccess.indices.end());

    fullyComposeAffineMapAndOperands(&map, &operands);
    map = simplifyAffineMap(map);
    canonicalizeMapAndOperands(&map, &operands);

    // Parse each map result to construct access matrices.
    ArrayRef<AffineExpr> mapResults = map.getResults();

    uint64_t noDims = map.getNumDims();
    uint64_t noSymbols = map.getNumSymbols();

    // Number of rows in an accessMatrix = Number of dimensions 
    // in the MemRef object.

    // Number of Columns = (noDims + noSymbols).

    SmallVector<std::vector<int64_t>, 0> accessMatrix;
    std::vector<int64_t> constVector(mapResults.size());

    for (unsigned l = 0; l < mapResults.size(); l++) {
      
      // Parse the l-th map result (access expr for l-th dimension of this MemRef) 
      // to get the l-th row of this access matrix.
      AffineExpr mapResult = mapResults[l]; 

      // A Row of Matrix A.
      std::vector<int64_t> Row(std::max(
          AffineForOpLoopNestSize,
          unsigned(noDims + noSymbols))); 

      // Check if `mapResult` is not a constant expr.
      // If it is a constant expr like A[5], no need to walk it.
      // Instead, push the value in constVector and
      // leave the Row matrix to be a vector of zeroes.

      if (mapResult.getKind() == AffineExprKind::Constant) {
        auto constant = mapResult.cast<AffineConstantExpr>();
        constVector[l] = (constant.getValue());
      } else {

          // Start parsing the mapResult.
        mapResult.walk([&](AffineExpr expr) {

          // Each `expr` can in turn be a combination of other affine expressions.
          switch (expr.getKind()) {
          case AffineExprKind::Add: {

            // Does this row need a modification?
            // Please note that the Row should not
            // be modified in case of a constant affine expr.
            bool modifyRow = true; 

            AffineBinaryOpExpr addExpr = expr.cast<AffineBinaryOpExpr>();
            AffineExpr lhs = addExpr.getLHS();
            AffineExpr rhs = addExpr.getRHS();
            unsigned lhsPosition = 0;
            unsigned rhsPosition = 0;

            // Parse LHS part of affine access expr.
            AffineExprKind lhskind = lhs.getKind();
            if (lhskind == AffineExprKind::DimId) {
              auto dim = lhs.cast<AffineDimExpr>();
              lhsPosition = loopVarIdMap[operands[dim.getPosition()]];
            } else if (lhskind == AffineExprKind::SymbolId) {
              auto symbol = lhs.cast<AffineSymbolExpr>();
              lhsPosition = noDims + symbol.getPosition();
            } else if (lhskind == AffineExprKind::Constant) {
              int cons = rhs.cast<AffineConstantExpr>().getValue();
              constVector.push_back(cons);
              modifyRow = false;
            }

            if (Row[lhsPosition] == 0 && modifyRow)
              Row[lhsPosition] = 1;

            // Parse RHS part of affine access expr.
            AffineExprKind rhskind = rhs.getKind();
            modifyRow = true;

            if (rhskind == AffineExprKind::DimId) {
              auto dim = rhs.cast<AffineDimExpr>();
              rhsPosition = dim.getPosition();
              rhsPosition = loopVarIdMap[operands[rhsPosition]];
            } else if (rhskind == AffineExprKind::SymbolId) {
              auto symbol = rhs.cast<AffineSymbolExpr>();
              rhsPosition = symbol.getPosition();
            } else if (rhskind == AffineExprKind::Constant) {
              int64_t cons = rhs.cast<AffineConstantExpr>().getValue();
              // If the RHS is a constant, add this constant to B[l].
              constVector[l] += cons;
              modifyRow = false;
            }
            if (Row[rhsPosition] == 0 && modifyRow)
              Row[rhsPosition] = 1;

            break;
          }

          case AffineExprKind::Mul: {
            AffineBinaryOpExpr mulExpr = expr.cast<AffineBinaryOpExpr>();
            AffineExpr lhs = mulExpr.getLHS();
            AffineExpr rhs = mulExpr.getRHS();
            unsigned position = 0;

            // Parse LHS part of affine binary expr.
            switch (lhs.getKind()) {
            case AffineExprKind::DimId: {
              auto dim = lhs.cast<AffineDimExpr>();
              position = loopVarIdMap[operands[dim.getPosition()]];
              break;
            }
            case AffineExprKind::SymbolId: {
              auto symbol = lhs.cast<AffineSymbolExpr>();
              position = noDims + symbol.getPosition();
              break;
            }
            }

            // Parse RHS part of affine binary expr.
            switch (rhs.getKind()) {
            case AffineExprKind::Constant: {
              auto constant = rhs.cast<AffineConstantExpr>();
              Row[position] = constant.getValue();
              break;
            }
            }
            break;
          }

          case AffineExprKind::DimId: {
            auto dim = expr.cast<AffineDimExpr>();
            Row[loopVarIdMap[operands[dim.getPosition()]]] = 1;
            constVector[l] += 0;
            break;
          }
          case AffineExprKind::SymbolId: {
            auto symbol = expr.cast<AffineDimExpr>();
            Row[loopVarIdMap[operands[symbol.getPosition()]]] = 1;
            constVector[l] += 0;
            break;
          }

          case AffineExprKind::CeilDiv:
          case AffineExprKind::FloorDiv:
          case AffineExprKind::Mod: {
            auto dim = expr.cast<AffineBinaryOpExpr>();
            AffineExpr lhs = dim.getLHS();
            // RHS in this case is always a contant or a symbol.
            AffineExpr rhs = dim.getRHS();
            switch (lhs.getKind()) {
            case AffineExprKind::DimId: {
              auto dim = lhs.cast<AffineDimExpr>();
              Row[loopVarIdMap[operands[dim.getPosition()]]] = 1;
              break;
            }
            case AffineExprKind::SymbolId: {
              auto symbol = lhs.cast<AffineSymbolExpr>();
              Row[noDims + symbol.getPosition()] = 1;
              break;
            }
            }
            switch (rhs.getKind()) {
            case AffineExprKind::SymbolId: {
              auto symbol = rhs.cast<AffineSymbolExpr>();
              Row[noDims + symbol.getPosition()] = 1;
              break;
            }
            }
          }
          }
        });
      }
      accessMatrix.push_back(Row);
    }

    this->loopAccessMatrices[srcOp] = accessMatrix;
    this->constVector[srcOp] = constVector;
  }
}

/// Separates `forOpA` from its siblings. `forOpA` and all affine.for ops in
/// `siblings` have a common parent `parentForOp`.
static void separateSiblingAffineForOps(AffineForOp &parentForOp, 
    AffineForOp &forOpA,SmallVector<AffineForOp,0> &siblings) {

  OpBuilder builder(parentForOp.getOperation()->getBlock(),
                    std::next(Block::iterator(parentForOp)));
  AffineForOp copyParentForOp = cast<AffineForOp>(builder.clone(*parentForOp));

  // Note the order in which `forOpA` and all other siblings are
  // visited. We need this order to compare affine.for ops within
  // `parentForOp` with their copy in `copyParentForOp`.
  // Comparing forOp.getOperation() does not work in that case.
  int forOpAPosition = 0;
  llvm::SmallSet<unsigned,8> siblingsIndices;
  int parentPosition = 0;
  int index = 0;
  
  parentForOp.getOperation()->walk([&](AffineForOp op) {
    index++;
    if (op.getOperation() == forOpA.getOperation()) {
      // Note the position of `forOpA` in walk order.
      forOpAPosition = index; 
    }
    for (unsigned i = 0; i<siblings.size(); i++) {
      if (op.getOperation() == siblings[i].getOperation()) {
        siblingsIndices.insert(index);
      }
    }
  });

  // Walk the copy of parentOp to erase all siblings other than `forOpA`.
  index = 0;
  copyParentForOp.getOperation()->walk([&](AffineForOp op) {
    index++;
    if (index != forOpAPosition && siblingsIndices.count(index)) {
      op.getOperation()->erase();
    }
  });

  // Erase `forOpA` from the original copy.
  forOpA.getOperation()->erase();
}

/// Converts all imperfectly nested loop nests in the body of `funcOp`
/// to perfectly nested loop nests using loop-splitting. 
void LoopInterchange::handleImperfectlyNestedAffineLoops(
    Operation &funcOp) {

  SmallVector<AffineForOp, 0> AffineForOpLoopNest;
  DenseMap<Operation *, SmallVector<AffineForOp, 0>> forTree;
  DenseMap<Operation *, AffineForOp> forOperations;

  // Stop splitting when each parent has only one child left.
  bool oneChild = false;
  while (!oneChild) {
    AffineForOpLoopNest.clear();
    forTree.clear();
    forOperations.clear();
    oneChild = true;

    // Walk the function to create a tree of affine.for operations.
    funcOp.walk([&](AffineForOp op) {
      AffineForOpLoopNest.push_back(op);
      if (op.getParentOp()->getName().getStringRef() == "affine.for") {
        forTree[op.getOperation()->getParentOp()].push_back(op);
      }
      forOperations[op.getOperation()] = op;
    });
    
    // Separate one of the sibling at a time.
    for (auto loopNest : forTree) {
        // This loop nest has no sibling problem.
        // Check the next loop nest.
      if (loopNest.second.size() < 2)
        continue;
      oneChild = false;
      separateSiblingAffineForOps(forOperations[loopNest.first],
                                  loopNest.second[loopNest.second.size()-1], 
                                  loopNest.second);
      // We need to walk the function again. The loop nests have changed.
      break;
    }
  }
  return;
}

/// Populates `loadAndStoreOps` with a list of all affine.load/affine.store
/// ops in the loop nest rooted at `rootForOp`.
static void getAllLoadStores(
    AffineForOp rootForOp, 
    SmallVector<Operation *, 8> &loadAndStoreOps) {
  rootForOp.getOperation()->walk([&](Operation *op) {
    if (isa<AffineLoadOp>(op) || isa<AffineStoreOp>(op)) {
      MemRefType memref;
      if (isa<AffineLoadOp>(op)) {
        AffineLoadOp loadOp = dyn_cast<AffineLoadOp>(*op);
        memref = loadOp.getMemRefType();
      } else {
        AffineStoreOp storeOp = dyn_cast<AffineStoreOp>(*op);
        memref = storeOp.getMemRefType();
      }
      loadAndStoreOps.push_back(op);
    }
  });
}

/// Finds out element size of all MemRefs in loop nest and assigns to
/// protected `dataSize` property.
void LoopInterchange::getElementsSize(AffineForOp &rootForOp) {
  bool isLoadStore = false;
  rootForOp.getOperation()->walk([&](Operation *op) {
    isLoadStore = false;
    MemRefType memref;
    if (isa<AffineLoadOp>(op)) {
      AffineLoadOp loadOp = dyn_cast<AffineLoadOp>(*op);
      memref = loadOp.getMemRefType();
      isLoadStore = true;
    } else if (isa<AffineStoreOp>(op)) {
      AffineStoreOp storeOp = dyn_cast<AffineStoreOp>(*op);
      memref = storeOp.getMemRefType();
      isLoadStore = true;
    }
    if (isLoadStore) {
      auto elementType = memref.getElementType();

      unsigned sizeInBits = 0;
      if (elementType.isIntOrFloat()) {
        sizeInBits = elementType.getIntOrFloatBitWidth();
      }
      // Store dataSize. Later on to be used to check whether two accesses are
      // within a cache-line/dataSize distance apart for a useful locality.
      this->dataSize[op] = llvm::divideCeil(sizeInBits, 8);
    }
  });
}

/// Get all permutations of the loop nest rooted at `rootForOp`. Store them
/// in `validPermutations` as a list.
void getAllValidPermutations(
    AffineForOp &rootForop, int noLoopsInNest,
    SmallVector<SmallVector<unsigned, 0>, 0> &validPermutations) {

  SmallVector<unsigned, 0> permutation;
  for (int i = 0; i < noLoopsInNest; i++)
    permutation.push_back(i);

  validPermutations.push_back(permutation);
  SmallVector<AffineForOp, 0> perfectLoopNest;
  getPerfectlyNestedLoops(perfectLoopNest, rootForop);

  while (std::next_permutation(permutation.begin(), permutation.end())) {
    if (isValidLoopInterchangePermutation(
            perfectLoopNest,
            permutation))
      validPermutations.push_back(permutation);
  }
}

/// Checks which loops in the loop nest have loop-carried-dependencies.
void LoopInterchange::getAffineLoopCarriedDependencies(
    AffineForOp &rootForOp,
    ArrayRef<Operation *> loadAndStoreOps,
    unsigned maxLoopDepth) {
  
    SmallVector<std::vector<int64_t>,0> dependenceMatrix;

  unsigned numOps = loadAndStoreOps.size();
  for (unsigned i = 0; i < numOps; ++i) {
    Operation *srcOp = loadAndStoreOps[i];
    for (unsigned j = 0; j < numOps; ++j) {
      Operation *dstOp = loadAndStoreOps[j];
      for (unsigned depth = 1; depth <= maxLoopDepth + 1; ++depth) {
        MemRefAccess srcAccess(srcOp);
        MemRefAccess dstAccess(dstOp);
        FlatAffineConstraints dependenceConstraints;
        SmallVector<DependenceComponent, 2> depComps;
        DependenceResult result = checkMemrefAccessDependence(
            srcAccess, dstAccess, depth, &dependenceConstraints, &depComps);

        if (hasDependence(result)) {
          std::vector<int64_t> components;

          // Average upper and lower bounds of each dependence component. We don't 
          // need actual upper/lower bound - just a value which suggests if 
          // there is any dependency or not.
          for (DependenceComponent depComp : depComps) {
            
            // both upper and lower bound have same signs
            if (depComp.lb.getValue() * depComp.ub.getValue() >= 0) { 
              components.push_back(
                  (depComp.lb.getValue() + depComp.ub.getValue()) / 2);
            } else {
              if (depComp.lb.getValue() < 0 && depComp.ub.getValue() > 0) 
                components.push_back(depComp.lb.getValue());
            }
          }
          dependenceMatrix.push_back(components);
          break;
        }
      }
    }
  }

  if (!dependenceMatrix.size()) {
      // Create a dummy all zeros dependency matrix. We need at least
      // one row in dependency matrix for computing loop-carried-depend-
      // dence vector in next block.
    std::vector<int64_t> allZeros(maxLoopDepth);
      dependenceMatrix.push_back(allZeros);
  }

  // If an entire column i is zeroes, there is no dependence on loop at 
  // depth i.
  this->loopCarriedDependenceVector.resize((dependenceMatrix)[0].size());
  for (unsigned i = 0; i < ((dependenceMatrix)[0].size()); i++) {
    if (!checkColumnIsZero(dependenceMatrix, i)) {
      this->loopCarriedDependenceVector[i] = true;
    }
  }
}

/// Calculates a representative cost of a permutation regarding parallelism on
/// multicores. A permutation having more free outer loops gets a smaller cost.
uint64_t
getParallelismCost(ArrayRef<unsigned> permutation,
                   SmallVector<bool,2> &loopCarriedDependenceVector,
                   SmallVector<unsigned, 2> &iterationCountVector) {
  unsigned totalcost = 0;
  for (unsigned i = 0; i < permutation.size(); i++) {
    if (!loopCarriedDependenceVector[permutation[i]])
      continue;
    int individualLoopcost = 1;
    for (unsigned j = i + 1; j < permutation.size(); j++) {
      individualLoopcost *= iterationCountVector[permutation[j]];
    }
    totalcost += individualLoopcost;
  }
  return totalcost;
}

/// Builds reference groups to calculate group-reuse. Two references are in the
/// same reference group if they exibit group-temporal or group-spatial reuse with
/// respect to a given loop.
///
/// Please refer Steve Carr et. al for detailed descriptions.
/// https://dl.acm.org/doi/abs/10.1145/195470.195557
///
/// Two accesses ref1 and ref2 belong to same reference group with respect 
/// to a loop if :
/// 1. There exists a dependency l and
///     1.1 l is a loop-independent dependence or
///     1.2 l's component for ForOp is a small constant d (|d|<=2) and all other
///     entries are zero.
/// OR
/// 2. ref1 and ref2 refer to the same array and differ by at most d1 in the
/// last subscript dimension, where d1 <= cache line size in terms of array
/// elements. All other subscripts must be identical.
///
/// This implementation starts with all accesses having their own group. Thus, 
/// if an access is not part of any group-reuse, it still has it's own group. 
/// This takes care of self-spatial reuse.
void LoopInterchange::buildReferenceGroups(
    AffineForOp ForOp, SmallVector<Operation*,8> &loadAndStoreOps, 
    unsigned maxDepth,
    unsigned forOpLevelInOriginalLoopNest) {

  unsigned numOps = loadAndStoreOps.size();

  SmallVector<llvm::SmallSet<Operation *,8>,8> referenceGroups;
  referenceGroups.resize(numOps);

  // Each Operation* is assigned a group ID. This ID is used to track the
  // insertions/deletions among referenceGroups.
  DenseMap<Operation *, int>
      groupId; 

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

      // Test for criteria 2 for reference groups.
      if (srcAccess.memref == dstAccess.memref) {
        // Check if both access the same array. If yes, then
        // for an access function Ax+B, we want the matrix A of both
        // accesses to be equal. Only B should differ in last dimension.
        if (this->loopAccessMatrices[srcOp] !=
            this->loopAccessMatrices[dstOp])
          continue; 
        std::vector<int64_t> srcOpConstantVector =
            this->constVector[srcOp];
        std::vector<int64_t> destOpConstantVector =
            this->constVector[dstOp];

        bool onlyLastIndexVaries = true;
        
        for (unsigned i = 0; i < srcOpConstantVector.size() - 1; i++) {
          if ((srcOpConstantVector[i] != destOpConstantVector[i])) {
            onlyLastIndexVaries = false;
            break;
          }
        }

        if (!onlyLastIndexVaries)
          continue;
        // Difference in last index should be less than cache_line_size/dataSize 
        // for a useful locality.
        unsigned dataSize = this->dataSize[srcOp];

        if (!(abs(srcOpConstantVector[srcOpConstantVector.size() - 1] -
                  destOpConstantVector[destOpConstantVector.size() - 1]) <=
              cache_line_size / dataSize))
          continue;
        referenceGroups[groupId[srcOp]].insert(
            referenceGroups[groupId[dstOp]].begin(),
            referenceGroups[groupId[dstOp]].end());
        referenceGroups.erase(referenceGroups.begin() + groupId[dstOp]);
        // Insert operation results in same group-id for
        // both instructions.
        groupId[dstOp] = groupId[srcOp]; 
      } else {

        // Test for criteria 1 for reference groups.
        for (unsigned depth = 1; depth <= maxDepth + 1; depth++) {
          FlatAffineConstraints dependenceConstraints;
          SmallVector<DependenceComponent, 2> depComps;
          DependenceResult result = checkMemrefAccessDependence(
              srcAccess, dstAccess, depth, &dependenceConstraints, &depComps);

          if (!hasDependence(result))
            continue;
          
          if (depth == maxDepth + 1) {
            // There is a loop-independent dependence. This imples both 
            // instructions belong to the same reference group.
            referenceGroups[groupId[srcOp]].insert(
                referenceGroups[groupId[dstOp]].begin(),
                referenceGroups[groupId[dstOp]].end());
            referenceGroups.erase(referenceGroups.begin() + groupId[dstOp]);
            groupId[dstOp] = groupId[srcOp];
          } else {
            // Search for dependence at depths other than level of this
            // `forOp` (considered as innermost loop). All entries other than 
            // at the level of this `forOp` should be zero.
            bool hasDependency = false;
            for (unsigned i = 0; i <= maxDepth; i++) {
              if ((depComps[i].lb.getValue() != 0) &&
                  (i != forOpLevelInOriginalLoopNest)) {
                hasDependency = true;
                break;
              }
            }
            if (hasDependency)
              continue;
            if (abs(depComps[forOpLevelInOriginalLoopNest].lb.getValue()) <=
                2) {
              referenceGroups[groupId[srcOp]].insert(
                  referenceGroups[groupId[dstOp]].begin(),
                  referenceGroups[groupId[dstOp]].end());
              referenceGroups.erase(referenceGroups.begin() +
                                    groupId[dstOp]);
              groupId[dstOp] = groupId[srcOp];
            }
          }
        }
      }
    }
  }
  this->referenceGroups = referenceGroups;
}

/// Calculates number of cache lines accessed by each affine.for in the
/// `AffineForOpLoopNest` each of those were the innermost loop.
///
/// Please refer Steve Carr et. al for detailed descriptions.
/// https://dl.acm.org/doi/abs/10.1145/195470.195557
void LoopInterchange::getCacheLineAccessCounts(
  ArrayRef<AffineForOp> AffineForOpLoopNest,
    SmallVector<Operation*, 8> &loadAndStoreOps) {
    
  unsigned loopNestSize = AffineForOpLoopNest.size();
  
  for (unsigned innerloop = 0; innerloop < loopNestSize;
       innerloop++) {

    AffineForOp forOp = AffineForOpLoopNest[innerloop];
    // Build reference groups considering each `forOp` to be the innermost
    // loop.
    buildReferenceGroups(forOp, loadAndStoreOps, loopNestSize, innerloop+1);

    float step = forOp.getStep();
    float trip =
        (forOp.getConstantUpperBound() - forOp.getConstantLowerBound()) / step +
        1;
    long double cacheLineCount = 0;

    for (llvm::SmallSet<Operation *,8> group : this->referenceGroups) {
      Operation *op = *group.begin();
      ArrayRef<std::vector<int64_t>> accessMatrix = 
          this->loopAccessMatrices[op];
      float stride = step * 
          accessMatrix[accessMatrix.size() - 1][innerloop];

      // `cacheLinesForThisOperation` represents the locality of this affine.for
      // due to this op. It is the number of cache lines that this affine.for
      // uses in this operation :
      // "1" for loop-invariant references,
      // "trip/(cache_line_size/stride)" for consecutive accesses,
      // "trip" for non-consecutive references.
      double cacheLinesForThisOperation = trip;

      bool isLoopInvariant = true;
      bool isConsecutive = true;

      for (unsigned j = 0; j < accessMatrix.size(); j++) {
        if (accessMatrix[j][innerloop] != 0) {
          isLoopInvariant = false;
          if (j != accessMatrix.size() - 1)
            isConsecutive = false;
        }
      }

      if (isLoopInvariant) {
        cacheLinesForThisOperation = 1;
      } else if (stride < 8 &&
                 isConsecutive) { 
        // Assumption: cache_line_size/dataSize = 8.
        // TODO: Implement for actual dataSize of the elements in MemRef.
        cacheLinesForThisOperation = (trip * stride) / 8;
      }

      // `cacheLineCount` represents overall number of cache lines 
      // accessed with this loop as the innermost loop.
      cacheLineCount += cacheLinesForThisOperation;
    }
    this->numberOfCacheLinesAccessed[&AffineForOpLoopNest[innerloop]] =
        cacheLineCount;
  }
}

/// Calculates a representative temporal reuse cost for a given permutation of
/// the loop nest. A low value returned means high temporal reuse.
long double LoopInterchange::getTemporalReuseCost(
    ArrayRef<unsigned> permutation) {
  // Initially we assume the cost for no temporal reuse is a big value
  // (arbitrary chosen to be the size of iteration space of loop nest).
  long double cost = 1;
  for (unsigned iterationCount : this->loopIterationCounts)
    cost *= iterationCount;

  for (auto accessMatrixOpPair : this->loopAccessMatrices) {
    long double temporalReuse = 1;
    SmallVector<std::vector<int64_t>,0> accessMatrix = accessMatrixOpPair.second;
    for (int i = permutation.size() - 1; i >= 0; i--) {
      if (!checkColumnIsZero(accessMatrix, permutation[i])) {
        break;
      }
      temporalReuse *= this->loopIterationCounts[permutation[i]];
    }
    // Increasing temporalReuse decreases the cost.
    cost -= temporalReuse; 
  }
  return cost;
}

/// Calculates a representative cost of this permutation regarding spatial
/// locality. Lower cost implies this permutation promotes better spatial 
/// reuse. Use the fact that loops with smaller locality at inner positions 
/// promote more reuse.
long double
getSpatialLocalityCost(ArrayRef<unsigned> permutation,
                       SmallVector<AffineForOp,0> &AffineForOpLoopNest,
                       DenseMap<const AffineForOp *, 
                       long double> cacheLineCount,
                       ArrayRef<unsigned>loopIterationCounts) {

  // Overall number of cache lines accessed during execution of the
  // entire iteration space.
  long double auxiliaryCost = 0;
  
  // Size of the iteration sub-space defined of other loops inside
  // this loop in the loop nest - product of iteration counts of inner loops.
  long double iterationSubSpaceSize = 1;

  // Maximum cache lines accessed by any affine.for in the
  // loop nest. Helpful in calculating sentinel.
  unsigned maxCacheLinesAccessed = 0; 

  for (int i = permutation.size() - 1; i >= 0; i--) {
      // Number of cache lines accessed by this loop at index i
    unsigned numberCacheLinesAccessed =
        cacheLineCount[&AffineForOpLoopNest[permutation[i]]];
    if (numberCacheLinesAccessed > maxCacheLinesAccessed)
      maxCacheLinesAccessed = numberCacheLinesAccessed;

    auxiliaryCost += numberCacheLinesAccessed * iterationSubSpaceSize;
    iterationSubSpaceSize *= loopIterationCounts[permutation[i]];
  }

  // The optimal permutation is one in which the affine.for ops are arranged
  // in descending order of their individual cache line access counts from
  // left to right.

  // However such a permutation will have maximum 'auxiliaryCost' value. But we
  // want a minimum cost for such a permutation.

  // To reverse the effect, we subtract the auxiliaryCost value from a sentinel
  // value(maximum possible spatial locality cost). We define sentinel value as 
  // follows: sentinel = Sum_for_all_loops(maxCacheLineAccessed * 
  // iterationSubSpaceSize of each loop).

  long double sentinel = 0;
  iterationSubSpaceSize = 1;
  for (int loopIterationCount : loopIterationCounts) {
    sentinel += maxCacheLinesAccessed * iterationSubSpaceSize;
    iterationSubSpaceSize *= loopIterationCount;
  }
  return sentinel - auxiliaryCost;
}

/// Returns the optimal permutation considering both - the locality cost
/// as well as the parallelism cost. To compare different permutations,
/// it calcuates a weighted-average cost of each permutation.
///
/// WeightedCost = 100*parallelismCost + spatialLocalityCost + 
/// temporalLocalityCost. 
///
/// Assumption: Cost exprienced due to synchronizations are 100x expensive
/// than those due to locality. The permutation which has the lowest overall 
/// cost is returned.
SmallVector<unsigned,0> LoopInterchange::getBestPermutation(
    SmallVector<AffineForOp,0> &AffineForOpLoopNest,
    DenseMap<Value, int64_t>&loopVarIdMap) {

  SmallVector<unsigned, 0> bestPermutation;
  SmallVector<Operation *, 8> loadAndStoreOps;
  long double mincost = LONG_MAX;

  // Get a list of all affine.load/affine.store ops in the loop nest
  getAllLoadStores(AffineForOpLoopNest[0], loadAndStoreOps);

  // Get size of elements in each MemRef access. Later to be used
  // in building reference groups for spatial locality cost.
  getElementsSize(AffineForOpLoopNest[0]);

  // Get Affine access matrices for all load/stores in the loop nest body.
  getAffineAccessMatrices(AffineForOpLoopNest[0], loadAndStoreOps,
                          loopVarIdMap, AffineForOpLoopNest.size());

  // Get loop carried dependency vector for all affine.forops in the loop nest.
  // This is useful in calculating the parallelism cost for each permutation.
  getAffineLoopCarriedDependencies(AffineForOpLoopNest[0], 
      loadAndStoreOps, AffineForOpLoopNest.size());

  // Get locality information for each affine.forop in the loop nest.
  // This will be useful in calculating spatial locality cost for each
  // permutation.
  getCacheLineAccessCounts(AffineForOpLoopNest, loadAndStoreOps);
  
  // Get a list of all loop nest permutations which do not violate
  // any dependency constraints in the loop body.
  SmallVector<SmallVector<unsigned, 0>, 0> validPermutations;
  getAllValidPermutations(AffineForOpLoopNest[0], AffineForOpLoopNest.size(), 
      validPermutations);

  // Return size 0 SmallVector to indicate that no interchange is needed.
  if (validPermutations.size() <= 1) {
    return bestPermutation;
  }

  for (auto permutation : validPermutations) {
    int64_t parallelCost =
        getParallelismCost(permutation, this->loopCarriedDependenceVector,
                           this->loopIterationCounts);
    long double spatialLocalityCost = getSpatialLocalityCost(
        permutation, AffineForOpLoopNest, this->numberOfCacheLinesAccessed,
        this->loopIterationCounts);
    long double temporalReuseCost = getTemporalReuseCost(permutation);
    long double cost =
        100 * parallelCost + spatialLocalityCost + temporalReuseCost;
    if (cost < mincost) {
      mincost = cost;
      bestPermutation = permutation;
    }
  }

  //`permuteLoops()` (called in next step) maps loop 'i' to location
  // bestPermutation[i]. But we assume here that bestPermutation[i] 
  // maps to loop at depth 'i'. Hence, we need to prepare a new custom 
  // `returnPermutation` according to requirements of `permuteLoops()`.
  SmallVector<unsigned, 0> returnPermutation;
  returnPermutation.resize(bestPermutation.size());
  for (unsigned i = 0; i < bestPermutation.size(); i++)
    returnPermutation[bestPermutation[i]] = i;
  return returnPermutation;
}

/// Resets all the protected data members alongwith the provided arguments.
void LoopInterchange::clear(
    SmallVector<AffineForOp, 0> &AffineForOpLoopNest,
    DenseMap<Value, int64_t> &loopVarIdMap, int *loopVarIdCount) {
  AffineForOpLoopNest.clear();
  loopVarIdMap.clear();
  *loopVarIdCount = 0;
  this->loopIterationCounts.clear();
  this->loopCarriedDependenceVector.clear();
  this->numberOfCacheLinesAccessed.clear();
  this->referenceGroups.clear();
  this->dataSize.clear();
  this->loopAccessMatrices.clear();
  this->constVector.clear();
}

void LoopInterchange::runOnFunction() {
  SmallVector<AffineForOp, 0> AffineForOpLoopNest;

  // A map to hold IDs for all affine.for induction vars.
  DenseMap<Value, int64_t> loopVarIdMap;
  int loopVarIdCount = 0;

  Operation *function = getFunction().getOperation();

  handleImperfectlyNestedAffineLoops(*function);

  (*function).walk([&](AffineForOp op) {
    AffineForOpLoopNest.push_back(op);

    if (op.hasConstantUpperBound() && op.hasConstantLowerBound()) {
      this->loopIterationCounts.push_back(
          (op.getConstantUpperBound() - op.getConstantLowerBound()) /
          op.getStep());
    }

    // Assign IDs to loop variables - used later to populate
    // accessMatrix of each MemRef access.
    Value loopVar = op.getInductionVar();
    loopVarIdMap[loopVar] = loopVarIdCount;
    loopVarIdCount++;

    // Test if op is a root of some loop nest.
    if ((op.getParentOp()->getName().getStringRef().str() == "func")) {
      // With a postorder traversal, affine.forops in `AffineForOpLoopNest`
      // are arranged in reverse order.
      // Need to reverse to arrange in loop nest order.
      std::reverse(AffineForOpLoopNest.begin(), AffineForOpLoopNest.end());

      // The loop nest should not have any affine.if op and should have a
      // rectangular shaped iteration space.
      if (!hasAffineIfStatement(op) &&
          isRectangularAffineForLoopNest(AffineForOpLoopNest)) {

        // With a postorder traversal, each affine.forop gets an id in 
        // reverse order. We need to reverse the order so that the topmost 
        // loop induction var has an id = 0.
        for (auto loopVarIdPair : loopVarIdMap)
          loopVarIdMap[loopVarIdPair.first] =
              loopVarIdCount - 1 - loopVarIdPair.second;

        SmallVector<unsigned, 0> bestPermutation = getBestPermutation(
           AffineForOpLoopNest, loopVarIdMap);
        
        // `bestPermutation` has a size = 0 when there is no need of 
        // interchange.
        if (!bestPermutation.size()){
          // No need for interchange. Go to next loop nest.
          clear(AffineForOpLoopNest, loopVarIdMap, &loopVarIdCount);
        } else {

          // Finally, permute the loop nest to best permutation.
          permuteLoops(MutableArrayRef<AffineForOp>(AffineForOpLoopNest), 
              bestPermutation);
        }
      }
      // Clear all the variables for next loop nest.
      clear(AffineForOpLoopNest, loopVarIdMap, &loopVarIdCount);
    }
  }); 
}

std::unique_ptr<OperationPass<FuncOp>> mlir::createAffineLoopInterchangePass() {
  return std::make_unique<LoopInterchange>();
}

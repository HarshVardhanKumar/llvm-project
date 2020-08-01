//===- affineLoopInterchange.cpp - Code to perform loop interchange-----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"
#include "mlir/Analysis/AffineAnalysis.h"
#include "mlir/Analysis/AffineStructures.h"
#include "mlir/Analysis/LoopAnalysis.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Analysis/Utils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/IR/AffineValueMap.h"
#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BlockSupport.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Transforms/LoopUtils.h"
#include "mlir/Transforms/Utils.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <set>
#include <unordered_map>

#define cache_line_size 64

using namespace mlir;

namespace {
struct LoopInterchange : public AffineLoopInterchangeBase<LoopInterchange> {
  void runOnFunction() override;
  void handleImperfectlyNestedAffineLoops(Operation &funcOp);

protected:
  void getAffineAccessMatrices(
      AffineForOp rootForOp,
      llvm::SmallVector<AffineForOp, 0> &AffineForOpLoopNest,
      llvm::DenseMap<Value, int64_t> &loopVarIdMap);
  void getDependencesPresentInAffineLoopNest(
      AffineForOp forOp, unsigned maxLoopDepth,
      std::vector<llvm::SmallVector<int64_t, 8>> *depCompsMatrix);
  void getAffineLoopCarriedDependencies(
      std::vector<llvm::SmallVector<int64_t, 8>> *dependenceMatrix);
  void buildReferenceGroups(AffineForOp innermostForOp, unsigned maxDepth,
                            unsigned forOpLevelInOriginalLoopNest);
  void getCacheLineAccessCounts(
      llvm::SmallVector<AffineForOp, 0> &AffineForOpLoopNest);
  long double getTemporalReuseCost(llvm::SmallVector<unsigned, 0> &permutation);
  llvm::SmallVector<unsigned, 0> getBestPermutation(
      llvm::SmallVector<llvm::SmallVector<unsigned, 0>, 0> &validPermutations,
      llvm::SmallVector<AffineForOp, 0> &AffineForOpLoopNest);
  void clear(llvm::SmallVector<AffineForOp, 0> &AffineForOpLoopNest,
             llvm::DenseMap<Value, int64_t> &loopVarIdMap, int *loopVarIdCount);

  llvm::DenseMap<Operation *, llvm::SmallVector<std::vector<int64_t>, 0>>
      loopAccessMatrices; // For each memory access Ax+B in the loopnest, it is
                          // the collection of A's indexed by respective
                          // load/store operation
  llvm::DenseMap<Operation *, std::vector<int64_t>>
      constVector; // For a memory access Ax+B, this represents B
  llvm::DenseMap<Operation *, unsigned>
      dataSize; // The size of data in memref access
  llvm::SmallVector<unsigned, 0>
      loopIterationCounts; // Iteration counts for each AffineForOp
  std::vector<int>
      loopCarriedDependenceVector; // Loop Carried Dependency vector. A value of
                                   // "1" at index i represents there is a
                                   // dependency carried on the loop at depth i
                                   // in the loopnest
  llvm::DenseMap<AffineForOp *, long double>
      numberOfCacheLinesAccessed; // Number of cache lines accessed by each
                                  // AffineForOp if they were the innermost and
                                  // only loop in the loopnest
  std::vector<std::set<Operation *>> referenceGroups;
};
} // namespace

/// <summary>
/// Performs a walk over the loopnest rooted at "forOp" to check if there are
/// any intermediate "Affine-if" statements in the loopnest.
/// </summary>
/// <param name="forOp"></param>
/// <returns>Returns True if an "Affine-if" statement found.</returns>
bool hasAffineIfStatement(AffineForOp forOp) {
  int iffound = 0;
  forOp.walk([&](Operation *op) {
    if (isa<AffineIfOp>(op))
      iffound = 1;
  });
  return iffound;
}

/// <summary>
/// Tests if a given AffineForOp loopnest has a rectangular shaped iteration
/// space.
/// </summary>
/// <param name="loopnest"></param>
/// <returns>Returns True if each AffineForOp in the loopnest has a constant
/// upper and lower bound. Returns false otherwise.</returns>
bool isRectangularAffineForLoopNest(
    llvm::SmallVector<AffineForOp, 0> &loopnest) {
  for (auto a : loopnest) {
    if (!a.hasConstantUpperBound() || !a.hasConstantLowerBound())
      return false;
  }
  return true;
}

/// <summary>
/// Calculates Access Matrix for each memory access operation within the
/// loopnest rooted at rootForOp by parsing the memory access expressions. For a
/// memory access Ax+B, it calculates both A and B and populates the protected
/// "loopAccessMatrices" and "constVector" properties of LoopInterchange object.
/// </summary>
/// <param name="rootForOp">Root of the loopnest</param>
/// <param name="AffineForOpLoopNest"></param>
/// <param name="loopVarIdMap">The map maintains an Id to each loop induction
/// variable</param>
void LoopInterchange::getAffineAccessMatrices(
    AffineForOp rootForOp,
    llvm::SmallVector<AffineForOp, 0> &AffineForOpLoopNest,
    llvm::DenseMap<Value, int64_t> &loopVarIdMap) {
  // for the mod / ceildiv / floordiv operations, consider them similar to
  // access to dimensions.
  this->loopAccessMatrices.clear();
  this->constVector.clear(); // For Ax+B, this represents B
  SmallVector<Operation *, 8> loadAndStoreOpInsts;
  rootForOp.getOperation()->walk([&](Operation *opInst) {
    if (isa<AffineLoadOp>(opInst) || isa<AffineStoreOp>(opInst)) {
      loadAndStoreOpInsts.push_back(opInst);
    }
  });

  unsigned numOps = loadAndStoreOpInsts.size();
  for (unsigned i = 0; (i < numOps); ++i) {
    auto *srcOpInst = loadAndStoreOpInsts[i];
    MemRefAccess srcAccess(srcOpInst);
    AffineMap map;
    if (auto loadOp = dyn_cast<AffineLoadOp>(srcAccess.opInst))
      map = loadOp.getAffineMap();
    else if (auto storeOp = dyn_cast<AffineStoreOp>(srcAccess.opInst))
      map = storeOp.getAffineMap();
    SmallVector<Value, 8> operands(srcAccess.indices.begin(),
                                   srcAccess.indices.end());

    fullyComposeAffineMapAndOperands(&map, &operands);
    map = simplifyAffineMap(map);
    canonicalizeMapAndOperands(&map, &operands);

    auto mapResults = map.getResults();

    uint64_t noDims = map.getNumDims();
    uint64_t noSymbols = map.getNumSymbols();

    // #Rows in each accessMatrix: #Dimensions of the MemRef object
    // #Columns in each accessMatrix: (noDims + noSymbols)

    llvm::SmallVector<std::vector<int64_t>, 0> accessMatrix;
    std::vector<int64_t> constVector(mapResults.size());

    // accesssing the affine operations.
    // all for a single instruction

    for (unsigned l = 0; l < mapResults.size(); l++) {
      // mapResults has a size equal to the number of dimensions of the MemRef
      // object being accessed.
      AffineExpr mapResult = mapResults[l]; // expression in the l-th dimension
                                            // of the MemRef object

      std::vector<int64_t> Row(std::max(
          AffineForOpLoopNest.size(),
          noDims + noSymbols)); // Represents each row of Matrix A in Ax+B

      // check if mapResult is not a constant expr.
      // If it is constant expr like A[5], then no need to walk it.
      // Instead simply push the value in constVector and
      // leave the Row matrix to be a vector of zeroes.

      if (mapResult.getKind() == AffineExprKind::Constant) {
        auto constant = mapResult.cast<AffineConstantExpr>();
        constVector[l] = (constant.getValue());
      } else {
        // constructing one row of access matrix
        mapResult.walk([&](AffineExpr expr) {
          // all these expr are sub-level expressions inside the mapResult
          // expression mapResult = (expr op expr) op expr ...

          switch (expr.getKind()) {

            // For an AffineExpression a op b, both a and b themselves can be
            // expressions. However if either is a constant, it will always be
            // on the right.

          case AffineExprKind::Add: {
            bool modifyRow =
                true; // used to indicate whether or not to modify the Row -
                      // intitially all zeros. Please note that the Row will not
                      // be modified in case of a constant
            AffineBinaryOpExpr op = expr.cast<AffineBinaryOpExpr>();
            auto lhs = op.getLHS();
            auto rhs = op.getRHS();
            unsigned lhsPosition = 0;
            unsigned rhsPosition = 0;

            auto lhskind = lhs.getKind();

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

            // If there is a non-zero coefficient, it means that the symbol has
            // some coefficient x associated with some sub-expr.
            // A[ x*i + ...] = [x ...][a ..]^T
            // In that case, no need to reset/add the value on the access matrix

            if (Row[lhsPosition] == 0 && modifyRow)
              Row[lhsPosition] = 1;

            int64_t cons = 0;
            auto rhskind = rhs.getKind();
            modifyRow = true;

            if (rhskind == AffineExprKind::DimId) {
              auto dim = rhs.cast<AffineDimExpr>();
              rhsPosition = dim.getPosition();
              rhsPosition = loopVarIdMap[operands[rhsPosition]];
            } else if (rhskind == AffineExprKind::SymbolId) {
              auto symbol = rhs.cast<AffineSymbolExpr>();
              rhsPosition = symbol.getPosition();
            } else if (rhskind == AffineExprKind::Constant) {
              cons = rhs.cast<AffineConstantExpr>().getValue();
              // if the rhs is a constant, then simply add this constant to
              // this dimension.
              constVector[l] += cons;
              modifyRow = false;
            }

            if (Row[rhsPosition] == 0 && modifyRow)
              Row[rhsPosition] = 1;

            break;
          }

          case AffineExprKind::Mul: {
            AffineBinaryOpExpr op = expr.cast<AffineBinaryOpExpr>();
            auto lhs = op.getLHS();
            auto rhs = op.getRHS();
            unsigned position = 0;
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
            default: {
            }
            }
            switch (rhs.getKind()) {
            case AffineExprKind::Constant: {
              auto constant = rhs.cast<AffineConstantExpr>();
              Row[position] = constant.getValue();
              break;
            }
            default: {
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
            auto lhs = dim.getLHS();
            // rhs is always a contant or symbol
            auto rhs = dim.getRHS();
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
            default: {
            }
            }
            switch (rhs.getKind()) {
            case AffineExprKind::SymbolId: {
              auto symbol = rhs.cast<AffineSymbolExpr>();
              Row[noDims + symbol.getPosition()] = 1;
              break;
            }
            default: {
            }
            }
          }
          default: {
          }
          }
        });
      }
      accessMatrix.push_back(Row);
    }

    this->loopAccessMatrices[srcOpInst] = accessMatrix;
    this->constVector[srcOpInst] = constVector;
  }
}

/// <summary>
/// Checks if an entire column of a 2D matrix is zero
/// </summary>
/// <param name="accessMatrix"></param>
/// <param name="column"></param>
/// <returns>True if the column is all zeros. False otherwise.</returns>
bool checkColumnIsZero(llvm::SmallVector<std::vector<int64_t>, 0> &accessMatrix,
                       unsigned column) {
  for (auto a : accessMatrix) {
    if (a[column] != 0)
      return false;
  }
  return true;
}

/// <summary>
/// Splits a parent AffineForOp having two sibling loops into two different
/// loopnests. Each new loopnest consists of a copy of parent for each sibling
/// loop. A loopnest such as : parent{
///     sibling1{}
///     sibling2{}
/// }
/// is converted to
///
/// parent{
///     sibling1
/// }
/// parent{
///     sibling2
/// }
/// </summary>
/// <param name="parentOp">The parent-loop</param>
/// <param name="forOpA">First Sibling loop</param>
/// <param name="forOpB">Second Sibling loop</param>
void splitSiblingAffineForLoops(AffineForOp parentForOp, AffineForOp forOpA,
                                AffineForOp forOpB) {

  OpBuilder builder(parentForOp.getOperation()->getBlock(),
                    std::next(Block::iterator(parentForOp)));
  auto copyParentForOp = cast<AffineForOp>(builder.clone(*parentForOp));

  int forOpAPosition = 0, forOpBPosition = 0;
  int parentPos = 0;
  int index = 0;
  parentForOp.getOperation()->walk([&](AffineForOp op) {
    index++;
    if (op.getOperation() == forOpA.getOperation()) {
      forOpAPosition =
          index; // note the position of first sibling in walk order
    }
    if (parentForOp.getOperation() == op.getOperation()) {
      parentPos = index; // note the position of parent in walk order
    }
  });
  copyParentForOp.getOperation()->walk([&](AffineForOp op) {
    forOpBPosition++;
    // find and erase the second sibling from the cloned copy
    if ((forOpBPosition != forOpAPosition) && (forOpBPosition != parentPos)) {
      op.getOperation()->erase();
    }
  });
  forOpA.getOperation()->erase();
}

/// <summary>
/// Converts imperfectly nested AffineForLoops to perfectly nested
/// AffineForLoops by splitting the initial loopnest. Each innermost sibling in
/// the original loopnest gets a copy of all the common parents.
/// </summary>
/// <param name="funcOp">The function which contains one or more imperfectly
/// nested loops</param>
void LoopInterchange::handleImperfectlyNestedAffineLoops(
    Operation &funcOp) { // converts the imperfectly nested loop nest to a
                         // perfectly nested loop nest by loop splitting.
  llvm::SmallVector<AffineForOp, 0> AffineForOpLoopNest;
  llvm::SmallVector<Operation *, 0> opvector;
  llvm::DenseMap<Operation *, llvm::SmallVector<AffineForOp, 0>> fortree;
  llvm::DenseMap<Operation *, AffineForOp> foroperation;

  // walk the function to create a tree of affine for operations.
  funcOp.walk([&](AffineForOp op) {
    AffineForOpLoopNest.push_back(op);
    if (op.getParentOp()->getName().getStringRef() == "affine.for")
      fortree[op.getOperation()->getParentOp()].push_back(op);
    foroperation[op.getOperation()] = op;
  });

  // for each loopNest in the tree of AffineForOps, split the parent
  // AffineForOps such that each lowest level sibling has its own copy of the
  // common parents.
  for (auto loopnest : fortree) {
    for (auto i = loopnest.second.size() - 1; i > 0; i--) {
      splitSiblingAffineForLoops(foroperation[loopnest.first],
                                 loopnest.second[i], loopnest.second[i - 1]);
    }
  }
  return;
}

/// <summary>
/// Get all permutations of the AffineForOps in the loopnest which do not
/// violate any dependency constraints.
/// </summary>
/// <param name="noLoopsInNest"></param>
/// <param name="forop">The first AffineForOp operation in the loopnest</param>
/// <param name="validPermutations"></param>
void getAllValidPermutations(
    int noLoopsInNest, AffineForOp forop,
    llvm::SmallVector<llvm::SmallVector<unsigned, 0>, 0> &validPermutations) {
  llvm::SmallVector<unsigned, 0> permutation;
  for (int i = 0; i < noLoopsInNest; i++)
    permutation.push_back(i);
  validPermutations.push_back(permutation);
  SmallVector<AffineForOp, 0> perfectloopnest;
  getPerfectlyNestedLoops(perfectloopnest, forop);

  while (std::next_permutation(permutation.begin(), permutation.end())) {
    if (isValidLoopInterchangePermutation(
            ArrayRef<AffineForOp>(perfectloopnest),
            ArrayRef<unsigned>(permutation)))
      validPermutations.push_back(permutation);
  }
}

/// <summary>
/// Populates the "dependenceMatrix" parameter with the dependencies present in
/// the loopnest rooted at "forOp". Each value in the dependency vector is an
/// "average" distance vector rather than a set of upper/lower bounds. Thus,
/// effectively, each value represents a direction rather than distance at each
/// loop depth.
/// </summary>
/// <param name="forOp"></param>
/// <param name="maxLoopDepth"></param>
/// <param name="dependenceMatrix"></param>
void LoopInterchange::getDependencesPresentInAffineLoopNest(
    AffineForOp forOp, unsigned maxLoopDepth,
    std::vector<llvm::SmallVector<int64_t, 8>> *dependenceMatrix) {
  SmallVector<Operation *, 8> loadAndStoreOpInsts;
  forOp.getOperation()->walk([&](Operation *opInst) {
    if (isa<AffineLoadOp>(opInst) || isa<AffineStoreOp>(opInst))
      loadAndStoreOpInsts.push_back(opInst);
  });
  unsigned numOps = loadAndStoreOpInsts.size();
  for (unsigned i = 0; i < numOps; ++i) {
    auto *srcOpInst = loadAndStoreOpInsts[i];
    for (unsigned j = 0; j < numOps; ++j) {
      auto *dstOpInst = loadAndStoreOpInsts[j];
      for (unsigned depth = 1; depth <= maxLoopDepth + 1; ++depth) {
        MemRefAccess srcAccess(srcOpInst);
        MemRefAccess dstAccess(dstOpInst);
        FlatAffineConstraints dependenceConstraints;
        SmallVector<DependenceComponent, 2> depComps;
        DependenceResult result = checkMemrefAccessDependence(
            srcAccess, dstAccess, depth, &dependenceConstraints, &depComps);

        if (mlir::hasDependence(result)) {
          llvm::SmallVector<int64_t, 8> components;

          // averaging the upper and lower bounds on dependenceComponents
          for (auto depComp : depComps) {
            if (depComp.lb.getValue() * depComp.ub.getValue() >=
                0) { // both upper and lower bound have same signs
              components.push_back(
                  (depComp.lb.getValue() + depComp.ub.getValue()) / 2);
            } else {
              if (depComp.lb.getValue() < 0 &&
                  depComp.ub.getValue() >
                      0) // upper and lower bounds have different signs.
                components.push_back(depComp.lb.getValue());
            }
          }
          dependenceMatrix->push_back(components);
          break;
        }
      }
    }
  }
  if (dependenceMatrix->size() == 0) {
    // create a zero vector
    llvm::SmallVector<int64_t, 8> loopDepVector(maxLoopDepth);
    dependenceMatrix->push_back(loopDepVector);
  }
}

/// <summary>
/// It calculates the loop-carried-dependency vector using the parameter
/// dependenceMatrix and assigns it to the private "loopCarriedDependencyVector"
/// property of LoopInterchange object. A value of 1 at index i represents that
/// the AffineForOp loop at depth i has a loop-carried-dependency.
/// </summary>
/// <param name="dependenceMatrix">Contains the direction vector (as calculated
/// using getDependencesPresentInAffineLoopNest method) of dependencies at each
/// loop depth</param>
void LoopInterchange::getAffineLoopCarriedDependencies(
    std::vector<llvm::SmallVector<int64_t, 8>> *dependenceMatrix) {
  std::vector<int> depVec((*dependenceMatrix)[0].size());
  for (unsigned i = 0; i < ((*dependenceMatrix)[0].size()); i++) {
    for (unsigned j = 0; j < dependenceMatrix->size(); j++) {
      if (((*dependenceMatrix)[j][i]) == 0)
        continue;
      depVec[i] = 1;
    }
  }
  this->loopCarriedDependenceVector = depVec;
}

/// <summary>
/// Calculates a representative cost of a permutation for parallelism on
/// multicores. The cost is not absolute in any sense, but it gives a measure of
/// relative closeness of this permutation to optimal permutation. The
/// permutation which having more number of free outer loops gets smaller cost.
/// </summary>
/// <param name="permutation"></param>
/// <param name="loopCarriedDependenceVector"></param>
/// <param name="iterationCountVector">Iteration counts for each AffinForOp
/// loop</param> <returns></returns>
uint64_t
getParallelismCost(llvm::SmallVector<unsigned, 0> &permutation,
                   std::vector<int> &loopCarriedDependenceVector,
                   llvm::SmallVector<unsigned, 0> &iterationCountVector) {
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

/// <summary>
/// Builds Reference groups to calculate group-reuse. Two references are in the
/// same reference group if they exibit group-temporal or group-spatial reuse.
/// Based on a paper by Steve Carr et. al
/// https://dl.acm.org/doi/abs/10.1145/195470.195557
///
/// Two accesses ref1 and ref2 belong to same reference group with respect to
/// loop ForOp if :
/// 1. There exists a dependency l and
///     1.1 l is a loop-independent dependence or
///     1.2 l's component for ForOp is a small constant d (|d|<=2) and all other
///     entries are zero.
/// OR
/// 2. ref1 and ref2 refer to the same array and differ by at most d1 in the
/// last subscript dimension, where d1 <= cache line size in terms of array
/// elements. All other subscripts must be identical.
///
/// Our implementation starts with all accesses having their own group before
/// they are grouped with other accesses. Thus, if an access is not part of a
/// group-reuse, it still has it's own group. This takes care of self-spatial
/// reuse.
/// </summary>
/// <param name="ForOp"></param>
/// <param name="maxDepth"></param>
/// <param name="forOpLevelInOriginalLoopNest">The level of this ForOp in the
/// original loopnest. For the analysis we consider this loop as the innermost
/// loop</param>
void LoopInterchange::buildReferenceGroups(
    AffineForOp ForOp, unsigned maxDepth,
    unsigned forOpLevelInOriginalLoopNest) {

  SmallVector<Operation *, 8> loadAndStoreOpInsts;

  ForOp.getOperation()->walk([&](Operation *op) {
    if (isa<AffineLoadOp>(op) || isa<AffineStoreOp>(op)) {
      MemRefType memref;
      if (isa<AffineLoadOp>(op)) {
        AffineLoadOp loadOp = dyn_cast<AffineLoadOp>(*op);
        memref = loadOp.getMemRefType();
      } else {
        AffineStoreOp storeOp = dyn_cast<AffineStoreOp>(*op);
        memref = storeOp.getMemRefType();
      }

      // get size of data
      auto elementType = memref.getElementType();

      unsigned sizeInBits = 0;
      if (elementType.isIntOrFloat()) {
        sizeInBits = elementType.getIntOrFloatBitWidth();
      }

      this->dataSize[op] = llvm::divideCeil(
          sizeInBits,
          8); // save dataSize. Lateron, to be used to check if two accesses are
              // within a cache-line/dataSize distance apart.
      loadAndStoreOpInsts.push_back(op);
    }
  });

  unsigned numOps = loadAndStoreOpInsts.size();

  std::vector<std::set<Operation *>> referenceGroups(numOps);
  llvm::DenseMap<Operation *, int>
      groupId; // Each Operation* is assigned a group-id. Used to track the
               // insertions/deletions among referenceGroups.
  for (unsigned i = 0; i < loadAndStoreOpInsts.size(); i++) {
    groupId[loadAndStoreOpInsts[i]] = i;
    referenceGroups[i].insert(loadAndStoreOpInsts[i]);
  }

  // now test for dependences
  for (unsigned i = 0; i < numOps; ++i) {
    auto *srcOpInst = loadAndStoreOpInsts[i];
    MemRefAccess srcAccess(srcOpInst);
    for (unsigned j = i + 1; j < numOps; ++j) {
      auto *dstOpInst = loadAndStoreOpInsts[j];
      MemRefAccess dstAccess(dstOpInst);
      if (srcOpInst == dstOpInst)
        continue;
      if (srcAccess.memref == dstAccess.memref) {
        // check if two instructions access the same array and their access
        // matrix varies only in the last dimension of the array by a constant.
        if (this->loopAccessMatrices[srcOpInst] !=
            this->loopAccessMatrices[dstOpInst])
          continue; // For Ax+B, we want the matrix A to be same for both
                    // accesses. Only B should vary.
        std::vector<int64_t> srcInstConstantVector =
            this->constVector[srcOpInst];
        std::vector<int64_t> destInstConstantVector =
            this->constVector[dstOpInst];

        bool onlyLastIndexVaries = true;
        // all the values should be same except the last value
        for (unsigned i = 0; i < srcInstConstantVector.size() - 1; i++) {
          if ((srcInstConstantVector[i] != destInstConstantVector[i])) {
            onlyLastIndexVaries = false;
            break;
          }
        }

        if (!onlyLastIndexVaries)
          continue;
        // test for the last index. The difference should be
        // constant and less than the cache_line_size/dataSize for a usable
        // locality.
        unsigned dataSize = this->dataSize[srcOpInst];

        if (!(abs(srcInstConstantVector[srcInstConstantVector.size() - 1] -
                  destInstConstantVector[destInstConstantVector.size() - 1]) <=
              cache_line_size / dataSize))
          continue;
        referenceGroups[groupId[srcOpInst]].insert(
            referenceGroups[groupId[dstOpInst]].begin(),
            referenceGroups[groupId[dstOpInst]].end());
        referenceGroups.erase(referenceGroups.begin() + groupId[dstOpInst]);
        groupId[dstOpInst] =
            groupId[srcOpInst]; // insert operation results in same group-id for
                                // both instructions.
      } else {
        for (unsigned depth = 1; depth <= maxDepth + 1; depth++) {
          FlatAffineConstraints dependenceConstraints;
          SmallVector<DependenceComponent, 2> depComps;
          DependenceResult result = checkMemrefAccessDependence(
              srcAccess, dstAccess, depth, &dependenceConstraints, &depComps);
          if (!mlir::hasDependence(result))
            continue;
          if (depth == maxDepth + 1) {
            // there is a loop-independent dependence -> Both instructions
            // belong to the same group
            referenceGroups[groupId[srcOpInst]].insert(
                referenceGroups[groupId[dstOpInst]].begin(),
                referenceGroups[groupId[dstOpInst]].end());
            referenceGroups.erase(referenceGroups.begin() + groupId[dstOpInst]);
            groupId[dstOpInst] = groupId[srcOpInst];
          } else {
            // search for dependence values at depths other than innermost loop
            // level. All entries other than at the level of this
            // forOp(forOpLevelInOriginalLoopNest) should be zero.
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
              referenceGroups[groupId[srcOpInst]].insert(
                  referenceGroups[groupId[dstOpInst]].begin(),
                  referenceGroups[groupId[dstOpInst]].end());
              referenceGroups.erase(referenceGroups.begin() +
                                    groupId[dstOpInst]);
              groupId[dstOpInst] = groupId[srcOpInst];
            }
          }
        }
      }
    }
  }
  this->referenceGroups = referenceGroups;
}

/// <summary>
/// Calculates number of cache-lines accessed by each AffineForOp in the
/// AffineForOpLoopNest if it were the innermost and only for loop.
///
/// Based on a paper by Steve Carr et al
/// https://dl.acm.org/doi/abs/10.1145/195470.195557
/// </summary>
/// <param name="AffineForOpLoopNest"></param>
void LoopInterchange::getCacheLineAccessCounts(
    llvm::SmallVector<AffineForOp, 0> &AffineForOpLoopNest) {
  // get total no. of  points in iteration space of the loopnest
  long double totaliterations = 1;
  for (auto a : this->loopIterationCounts)
    totaliterations *= a;

  for (unsigned innerloop = 0; innerloop < AffineForOpLoopNest.size();
       innerloop++) {
    AffineForOp forop = AffineForOpLoopNest[innerloop];
    float step = forop.getStep();
    float trip =
        (forop.getConstantUpperBound() - forop.getConstantLowerBound()) / step +
        1;
    long double cacheLineCount = 0;
    for (auto group : this->referenceGroups) {
      Operation *op = *group.begin();
      auto accessMatrix = this->loopAccessMatrices[op];
      float stride = step * accessMatrix[accessMatrix.size() - 1][innerloop];
      // cacheLinesForThisOperation represents the locality of the forOp due to
      // this operation. That is, it is the number of cache lines that forop
      // uses in this operation -> 1 for loop-invariant references,
      // trip/(cache_line_size/stride) for consecutive accesses, trip for
      // non-consecutive references
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
                 isConsecutive) { // assume that cache_line_size/dataSize = 8
        cacheLinesForThisOperation = (trip * stride) / 8;
      }

      // loopcost represents the no. of cache lines accessed with this loop as
      // the inner and only loop.
      cacheLineCount += cacheLinesForThisOperation;
    }
    this->numberOfCacheLinesAccessed[&AffineForOpLoopNest[innerloop]] =
        cacheLineCount;
  }
}

/// <summary>
/// Calculates a representative temporal reuse cost for a given permutation of
/// AffineForOp loopnest. A low value returned means high temporal reuse.
/// </summary>
/// <param name="permutation"></param>
/// <param name="AffineForOpLoopNest"></param>
/// <returns></returns>
long double LoopInterchange::getTemporalReuseCost(
    llvm::SmallVector<unsigned, 0> &permutation) {
  // Initially we assume the cost for no temporal reuse is a big value
  // (arbitrary chosen to be the size of iteration space of loopnest)
  long double cost = 1;
  for (auto iterationCount : this->loopIterationCounts)
    cost *= iterationCount;

  for (auto accessMatrixOpPair : this->loopAccessMatrices) {
    long double temporalReuse = 1;
    auto accessMatrix = accessMatrixOpPair.second;
    for (int i = permutation.size() - 1; i >= 0; i--) {
      if (!checkColumnIsZero(accessMatrix, permutation[i])) {
        break;
      }
      temporalReuse *= this->loopIterationCounts[permutation[i]];
    }
    cost -= temporalReuse; // Increasing temporalReuse decreases the cost
  }
  return cost;
}

/// <summary>
/// Calculates a representative cost of a permutation for spatial locality.
/// The cost is not absolute in any sense, but gives a comparative measure of
/// how close this permutation is to optimal permutation. Lower cost implies the
/// permutation is closer to optimal permutation. Optimal Permutation is the
/// permutation in which AffineForOps are arranged in a descending order of
/// cache-line-access-counts. It is based on the fact "If loop l promotes more
/// reuse than loop l1 when both are considered as innermost loops, l will
/// promote more reuse than l1 at any outer loop position"
/// </summary>
/// <param name="permutation"></param>
/// <param name="AffineForOpLoopNest"></param>
/// <param name="loopcost"></param>
/// <returns>A representative cost of each permutation for spatial
/// locality</returns>
long double
getSpatialLocalityCost(llvm::SmallVector<unsigned, 0> &permutation,
                       llvm::SmallVector<AffineForOp, 0> &AffineForOpLoopNest,
                       llvm::DenseMap<AffineForOp *, long double> loopcost,
                       llvm::SmallVector<unsigned, 0> &loopIterationCounts) {
  long double auxiliaryCost = 0;
  long double iterationSubSpaceSize =
      1; // size of the iteration sub-space defined by all other loops inside
         // this loop in the loopnest
  unsigned maxCacheLinesAccessed =
      0; // used to find out the maximum cache lines accessed by any loop in the
         // loopnest. Helpful while calculating sentinel.

  for (int i = permutation.size() - 1; i >= 0; i--) {
    unsigned numberCacheLinesAccessed =
        loopcost[&AffineForOpLoopNest[permutation[i]]];
    if (numberCacheLinesAccessed > maxCacheLinesAccessed)
      maxCacheLinesAccessed = numberCacheLinesAccessed;

    auxiliaryCost += numberCacheLinesAccessed * iterationSubSpaceSize;
    iterationSubSpaceSize *= loopIterationCounts[permutation[i]];
  }

  // The optimal permutation is one in which the loops are arranged in
  // descending order in terms of their individual cache line access counts from
  // left to right. Thus the rightmost loop should be the one having minimum
  // cacheline access count and the leftone should have maximum cacheline access
  // count.

  // However such a permutation will have maximum auxiliaryCost value.
  // To reverse the effect, we subtract the auxiliaryCost value from a sentinel
  // value. We define sentinel value as follows: sentinel =
  // Sum_for_all_loops(maxCacheLineAccessed * iterationSubSpaceSize of each
  // loop)

  long double sentinel = 0;
  iterationSubSpaceSize = 1;
  for (int loopIterationCount : loopIterationCounts) {
    sentinel += maxCacheLinesAccessed * iterationSubSpaceSize;
    iterationSubSpaceSize *= loopIterationCount;
  }

  return sentinel - auxiliaryCost;
}

/// <summary>
/// Returns the optimal permutation considering both - the cost exprienced due
/// to locality as well as parallelism. It calcuates a weighted-average cost for
/// each permutation. WeightedCost = 100 * parallelismCost + spatialLocalityCost
/// + temporalLocalityCost. It inherently makes an assumption that cost
/// exprienced due to synchronizations are 100x expensive than those of
/// locality. The permutation which has lowest overall cost is considered the
/// best permutation.
/// </summary>
/// <param name="validPermutations"></param>
/// <param name="AffineForOpLoopNest"></param>
/// <returns></returns>
llvm::SmallVector<unsigned, 0> LoopInterchange::getBestPermutation(
    llvm::SmallVector<llvm::SmallVector<unsigned, 0>, 0> &validPermutations,
    llvm::SmallVector<AffineForOp, 0> &AffineForOpLoopNest) {
  llvm::SmallVector<unsigned, 0> bestPermutation;
  long double mincost = LONG_MAX;
  for (llvm::SmallVector<unsigned, 0> permutation : validPermutations) {
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
  llvm::SmallVector<unsigned, 0> retPermutation;
  retPermutation.resize(bestPermutation.size());
  for (unsigned i = 0; i < bestPermutation.size(); i++)
    retPermutation[bestPermutation[i]] = i;
  return retPermutation;
}

/// <summary>
/// Resets all the protected data members as well as provided arguments.
/// </summary>
/// <param name="AffineForOpLoopNest"></param>
/// <param name="loopVarIdMap"></param>
/// <param name="loopVarIdCount"></param>
void LoopInterchange::clear(
    llvm::SmallVector<AffineForOp, 0> &AffineForOpLoopNest,
    llvm::DenseMap<Value, int64_t> &loopVarIdMap, int *loopVarIdCount) {
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
  llvm::SmallVector<AffineForOp, 0> AffineForOpLoopNest;

  // used to hold the id for the for-op induction vars and other variables
  // passed in the access function.
  llvm::DenseMap<Value, int64_t> loopVarIdMap;
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

    // assign an id to the loop variables - later to be used to calculate
    // accessMatrices for each memref access operation. The loop id has a direct
    // correspondence with the SymbolId while parsing AffineExpressions in the
    // affine access function
    Value loopVar = op.getInductionVar();
    loopVarIdMap[loopVar] = loopVarIdCount;
    loopVarIdCount++;

    // test if op is a root AffineForOp
    if ((op.getParentOp()->getName().getStringRef().str() == "func")) {
      std::reverse(AffineForOpLoopNest.begin(), AffineForOpLoopNest.end());

      // the loop nest should not have any if statement and should be
      // rectangular shaped in iteration space
      if (!hasAffineIfStatement(op) &&
          isRectangularAffineForLoopNest(AffineForOpLoopNest)) {

        // reverse the order of loop var id such that the topmost loop induction
        // var has an id = 0
        for (auto loopVarIdPair : loopVarIdMap)
          loopVarIdMap[loopVarIdPair.first] =
              loopVarIdCount - 1 - loopVarIdPair.second;

        // get a list of all valid permutations of the loopnest
        llvm::SmallVector<llvm::SmallVector<unsigned, 0>, 0> validPermutations;
        getAllValidPermutations(AffineForOpLoopNest.size(), op,
                                validPermutations);

        if (validPermutations.size() <= 1) {
          // Only one valid permutation. Go to next loopnest.
          clear(AffineForOpLoopNest, loopVarIdMap, &loopVarIdCount);
        } else {
          getAffineAccessMatrices(op, AffineForOpLoopNest, loopVarIdMap);

          std::vector<llvm::SmallVector<int64_t, 8>> dependenceMatrix;
          getDependencesPresentInAffineLoopNest(op, AffineForOpLoopNest.size(),
                                                &dependenceMatrix);
          getAffineLoopCarriedDependencies(&dependenceMatrix);

          buildReferenceGroups(op, AffineForOpLoopNest.size(), 1);
          getCacheLineAccessCounts(AffineForOpLoopNest);

          permuteLoops(MutableArrayRef<AffineForOp>(AffineForOpLoopNest),
                       llvm::ArrayRef<unsigned>(getBestPermutation(
                           validPermutations, AffineForOpLoopNest)));
        }
      }
      // clear all the state variables for next loopnest.
      clear(AffineForOpLoopNest, loopVarIdMap, &loopVarIdCount);
    }
  });
}

std::unique_ptr<OperationPass<FuncOp>> mlir::createAffineLoopInterchangePass() {
  return std::make_unique<LoopInterchange>();
}

// NOTE: Assertions have been autogenerated by utils/update_cc_test_checks.py
// RUN: %clang_cc1 -triple thumbv8.1m.main-arm-none-eabi \
// RUN:   -target-feature +cdecp0 -target-feature +cdecp1 \
// RUN:   -mfloat-abi hard -O0 -disable-O0-optnone \
// RUN:   -S -emit-llvm -o - %s | opt -S -mem2reg | FileCheck %s

#include <arm_cde.h>

// CHECK-LABEL: @test_cx1(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[TMP0:%.*]] = call i32 @llvm.arm.cde.cx1(i32 0, i32 123)
// CHECK-NEXT:    ret i32 [[TMP0]]
//
uint32_t test_cx1() {
  return __arm_cx1(0, 123);
}
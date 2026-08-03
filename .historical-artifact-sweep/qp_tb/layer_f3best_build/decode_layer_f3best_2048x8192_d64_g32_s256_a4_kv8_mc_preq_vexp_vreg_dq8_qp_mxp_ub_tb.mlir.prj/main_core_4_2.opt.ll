; ModuleID = 'decode_layer_f3best_2048x8192_d64_g32_s256_a4_kv8_mc_preq_vexp_vreg_dq8_qp_mxp_ub_tb.mlir.prj\main_core_4_2.peanohack.ll'
source_filename = "LLVMDialectModule"
target datalayout = "e-m:e-p:20:32-i1:8:32-i8:8:32-i16:16:32-i32:32:32-f32:32:32-i64:32-f64:32-a:0:32-n32"
target triple = "aie2p"

@c2_uni_partial = external global [128 x i8]
@c2_silu = external global [1024 x bfloat]
@c2_up = external global [1024 x bfloat]
@c2_gate = external global [1024 x bfloat]
@c2_P = external global [2320 x bfloat]
@c2_O = external global [2048 x bfloat]
@c2_Q = external global [322 x bfloat]
@c2_B2 = external global [2320 x bfloat]
@c2_B1 = external global [2320 x bfloat]
@c2_B0 = external global [2320 x bfloat]
@c2_A1 = external global [4608 x i8]
@c2_A0 = external global [4608 x i8]

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn
declare void @llvm.aie2p.acquire(i32, i32) #0

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn
declare void @llvm.aie2p.release(i32, i32) #0

declare void @_ha_noop() local_unnamed_addr

declare void @rope_bundled(ptr, ptr, ptr, i32) local_unnamed_addr

declare void @generic_bcast_gemv_bf16_q(i32, ptr, ptr, ptr, i32, ptr) local_unnamed_addr

declare void @generic_bcast_gemv_bf16_o(i32, ptr, ptr, ptr, i32, ptr) local_unnamed_addr

declare void @generic_bcast_gemv_bf16_g(i32, ptr, ptr, ptr, i32, ptr) local_unnamed_addr

declare void @generic_bcast_gemv_bf16_d(i32, ptr, ptr, ptr, i32, ptr) local_unnamed_addr

declare void @layer_fused_silu_mul_explicit_bf16(ptr, ptr, ptr, i32) local_unnamed_addr

define void @core_4_2() local_unnamed_addr {
  br label %1

1:                                                ; preds = %0, %80
  %2 = phi i64 [ 0, %0 ], [ %81, %80 ]
  tail call void @_ha_noop()
  tail call void @llvm.aie2p.acquire(i32 51, i32 -1)
  tail call void @llvm.aie2p.acquire(i32 52, i32 -1)
  br label %3

3:                                                ; preds = %3, %1
  %4 = phi i64 [ 0, %1 ], [ %16, %3 ]
  tail call void @llvm.aie2p.acquire(i32 49, i32 -1)
  %5 = trunc nuw i64 %4 to i32
  tail call void @generic_bcast_gemv_bf16_q(i32 %5, ptr nonnull @c2_A0, ptr nonnull @c2_B0, ptr nonnull @c2_uni_partial, i32 8, ptr nonnull @c2_Q)
  tail call void @llvm.aie2p.release(i32 48, i32 1)
  tail call void @llvm.aie2p.acquire(i32 59, i32 -1)
  %6 = or disjoint i32 %5, 1
  tail call void @generic_bcast_gemv_bf16_q(i32 %6, ptr nonnull @c2_A1, ptr nonnull @c2_B0, ptr nonnull @c2_uni_partial, i32 8, ptr nonnull @c2_Q)
  tail call void @llvm.aie2p.release(i32 58, i32 1)
  tail call void @llvm.aie2p.acquire(i32 49, i32 -1)
  %7 = trunc i64 %4 to i32
  %8 = or disjoint i32 %7, 2
  tail call void @generic_bcast_gemv_bf16_q(i32 %8, ptr nonnull @c2_A0, ptr nonnull @c2_B0, ptr nonnull @c2_uni_partial, i32 8, ptr nonnull @c2_Q)
  tail call void @llvm.aie2p.release(i32 48, i32 1)
  tail call void @llvm.aie2p.acquire(i32 59, i32 -1)
  %9 = or disjoint i32 %7, 3
  tail call void @generic_bcast_gemv_bf16_q(i32 %9, ptr nonnull @c2_A1, ptr nonnull @c2_B0, ptr nonnull @c2_uni_partial, i32 8, ptr nonnull @c2_Q)
  tail call void @llvm.aie2p.release(i32 58, i32 1)
  tail call void @llvm.aie2p.acquire(i32 49, i32 -1)
  %10 = trunc i64 %4 to i32
  %11 = or disjoint i32 %10, 4
  tail call void @generic_bcast_gemv_bf16_q(i32 %11, ptr nonnull @c2_A0, ptr nonnull @c2_B0, ptr nonnull @c2_uni_partial, i32 8, ptr nonnull @c2_Q)
  tail call void @llvm.aie2p.release(i32 48, i32 1)
  tail call void @llvm.aie2p.acquire(i32 59, i32 -1)
  %12 = or disjoint i32 %10, 5
  tail call void @generic_bcast_gemv_bf16_q(i32 %12, ptr nonnull @c2_A1, ptr nonnull @c2_B0, ptr nonnull @c2_uni_partial, i32 8, ptr nonnull @c2_Q)
  tail call void @llvm.aie2p.release(i32 58, i32 1)
  %13 = or disjoint i64 %4, 6
  tail call void @llvm.aie2p.acquire(i32 49, i32 -1)
  %14 = trunc nuw i64 %13 to i32
  tail call void @generic_bcast_gemv_bf16_q(i32 %14, ptr nonnull @c2_A0, ptr nonnull @c2_B0, ptr nonnull @c2_uni_partial, i32 8, ptr nonnull @c2_Q)
  tail call void @llvm.aie2p.release(i32 48, i32 1)
  tail call void @llvm.aie2p.acquire(i32 59, i32 -1)
  %15 = or disjoint i32 %14, 1
  tail call void @generic_bcast_gemv_bf16_q(i32 %15, ptr nonnull @c2_A1, ptr nonnull @c2_B0, ptr nonnull @c2_uni_partial, i32 8, ptr nonnull @c2_Q)
  tail call void @llvm.aie2p.release(i32 58, i32 1)
  %16 = add nuw nsw i64 %4, 8
  %17 = icmp samesign ult i64 %13, 62
  br i1 %17, label %3, label %18

18:                                               ; preds = %3
  tail call void @rope_bundled(ptr nonnull @c2_Q, ptr nonnull @c2_B0, ptr nonnull @c2_Q, i32 256)
  tail call void @llvm.aie2p.release(i32 53, i32 1)
  tail call void @llvm.aie2p.release(i32 50, i32 1)
  tail call void @llvm.aie2p.acquire(i32 61, i32 -1)
  tail call void @llvm.aie2p.acquire(i32 54, i32 -1)
  br label %19

19:                                               ; preds = %19, %18
  %20 = phi i64 [ 0, %18 ], [ %32, %19 ]
  tail call void @llvm.aie2p.acquire(i32 49, i32 -1)
  %21 = trunc nuw i64 %20 to i32
  tail call void @generic_bcast_gemv_bf16_o(i32 %21, ptr nonnull @c2_A0, ptr nonnull @c2_B1, ptr nonnull @c2_uni_partial, i32 8, ptr nonnull @c2_O)
  tail call void @llvm.aie2p.release(i32 48, i32 1)
  tail call void @llvm.aie2p.acquire(i32 59, i32 -1)
  %22 = or disjoint i32 %21, 1
  tail call void @generic_bcast_gemv_bf16_o(i32 %22, ptr nonnull @c2_A1, ptr nonnull @c2_B1, ptr nonnull @c2_uni_partial, i32 8, ptr nonnull @c2_O)
  tail call void @llvm.aie2p.release(i32 58, i32 1)
  tail call void @llvm.aie2p.acquire(i32 49, i32 -1)
  %23 = trunc i64 %20 to i32
  %24 = or disjoint i32 %23, 2
  tail call void @generic_bcast_gemv_bf16_o(i32 %24, ptr nonnull @c2_A0, ptr nonnull @c2_B1, ptr nonnull @c2_uni_partial, i32 8, ptr nonnull @c2_O)
  tail call void @llvm.aie2p.release(i32 48, i32 1)
  tail call void @llvm.aie2p.acquire(i32 59, i32 -1)
  %25 = or disjoint i32 %23, 3
  tail call void @generic_bcast_gemv_bf16_o(i32 %25, ptr nonnull @c2_A1, ptr nonnull @c2_B1, ptr nonnull @c2_uni_partial, i32 8, ptr nonnull @c2_O)
  tail call void @llvm.aie2p.release(i32 58, i32 1)
  tail call void @llvm.aie2p.acquire(i32 49, i32 -1)
  %26 = trunc i64 %20 to i32
  %27 = or disjoint i32 %26, 4
  tail call void @generic_bcast_gemv_bf16_o(i32 %27, ptr nonnull @c2_A0, ptr nonnull @c2_B1, ptr nonnull @c2_uni_partial, i32 8, ptr nonnull @c2_O)
  tail call void @llvm.aie2p.release(i32 48, i32 1)
  tail call void @llvm.aie2p.acquire(i32 59, i32 -1)
  %28 = or disjoint i32 %26, 5
  tail call void @generic_bcast_gemv_bf16_o(i32 %28, ptr nonnull @c2_A1, ptr nonnull @c2_B1, ptr nonnull @c2_uni_partial, i32 8, ptr nonnull @c2_O)
  tail call void @llvm.aie2p.release(i32 58, i32 1)
  %29 = or disjoint i64 %20, 6
  tail call void @llvm.aie2p.acquire(i32 49, i32 -1)
  %30 = trunc nuw i64 %29 to i32
  tail call void @generic_bcast_gemv_bf16_o(i32 %30, ptr nonnull @c2_A0, ptr nonnull @c2_B1, ptr nonnull @c2_uni_partial, i32 8, ptr nonnull @c2_O)
  tail call void @llvm.aie2p.release(i32 48, i32 1)
  tail call void @llvm.aie2p.acquire(i32 59, i32 -1)
  %31 = or disjoint i32 %30, 1
  tail call void @generic_bcast_gemv_bf16_o(i32 %31, ptr nonnull @c2_A1, ptr nonnull @c2_B1, ptr nonnull @c2_uni_partial, i32 8, ptr nonnull @c2_O)
  tail call void @llvm.aie2p.release(i32 58, i32 1)
  %32 = add nuw nsw i64 %20, 8
  %33 = icmp samesign ult i64 %29, 62
  br i1 %33, label %19, label %34

34:                                               ; preds = %19
  tail call void @llvm.aie2p.release(i32 60, i32 1)
  tail call void @llvm.aie2p.release(i32 55, i32 1)
  tail call void @llvm.aie2p.acquire(i32 63, i32 -1)
  br label %35

35:                                               ; preds = %35, %34
  %36 = phi i64 [ 0, %34 ], [ %48, %35 ]
  tail call void @llvm.aie2p.acquire(i32 49, i32 -1)
  %37 = trunc nuw i64 %36 to i32
  tail call void @generic_bcast_gemv_bf16_g(i32 %37, ptr nonnull @c2_A0, ptr nonnull @c2_B2, ptr nonnull @c2_uni_partial, i32 8, ptr nonnull @c2_gate)
  tail call void @llvm.aie2p.release(i32 48, i32 1)
  tail call void @llvm.aie2p.acquire(i32 59, i32 -1)
  %38 = or disjoint i32 %37, 1
  tail call void @generic_bcast_gemv_bf16_g(i32 %38, ptr nonnull @c2_A1, ptr nonnull @c2_B2, ptr nonnull @c2_uni_partial, i32 8, ptr nonnull @c2_gate)
  tail call void @llvm.aie2p.release(i32 58, i32 1)
  tail call void @llvm.aie2p.acquire(i32 49, i32 -1)
  %39 = trunc i64 %36 to i32
  %40 = or disjoint i32 %39, 2
  tail call void @generic_bcast_gemv_bf16_g(i32 %40, ptr nonnull @c2_A0, ptr nonnull @c2_B2, ptr nonnull @c2_uni_partial, i32 8, ptr nonnull @c2_gate)
  tail call void @llvm.aie2p.release(i32 48, i32 1)
  tail call void @llvm.aie2p.acquire(i32 59, i32 -1)
  %41 = or disjoint i32 %39, 3
  tail call void @generic_bcast_gemv_bf16_g(i32 %41, ptr nonnull @c2_A1, ptr nonnull @c2_B2, ptr nonnull @c2_uni_partial, i32 8, ptr nonnull @c2_gate)
  tail call void @llvm.aie2p.release(i32 58, i32 1)
  tail call void @llvm.aie2p.acquire(i32 49, i32 -1)
  %42 = trunc i64 %36 to i32
  %43 = or disjoint i32 %42, 4
  tail call void @generic_bcast_gemv_bf16_g(i32 %43, ptr nonnull @c2_A0, ptr nonnull @c2_B2, ptr nonnull @c2_uni_partial, i32 8, ptr nonnull @c2_gate)
  tail call void @llvm.aie2p.release(i32 48, i32 1)
  tail call void @llvm.aie2p.acquire(i32 59, i32 -1)
  %44 = or disjoint i32 %42, 5
  tail call void @generic_bcast_gemv_bf16_g(i32 %44, ptr nonnull @c2_A1, ptr nonnull @c2_B2, ptr nonnull @c2_uni_partial, i32 8, ptr nonnull @c2_gate)
  tail call void @llvm.aie2p.release(i32 58, i32 1)
  %45 = or disjoint i64 %36, 6
  tail call void @llvm.aie2p.acquire(i32 49, i32 -1)
  %46 = trunc nuw i64 %45 to i32
  tail call void @generic_bcast_gemv_bf16_g(i32 %46, ptr nonnull @c2_A0, ptr nonnull @c2_B2, ptr nonnull @c2_uni_partial, i32 8, ptr nonnull @c2_gate)
  tail call void @llvm.aie2p.release(i32 48, i32 1)
  tail call void @llvm.aie2p.acquire(i32 59, i32 -1)
  %47 = or disjoint i32 %46, 1
  tail call void @generic_bcast_gemv_bf16_g(i32 %47, ptr nonnull @c2_A1, ptr nonnull @c2_B2, ptr nonnull @c2_uni_partial, i32 8, ptr nonnull @c2_gate)
  tail call void @llvm.aie2p.release(i32 58, i32 1)
  %48 = add nuw nsw i64 %36, 8
  %49 = icmp samesign ult i64 %45, 254
  br i1 %49, label %35, label %.preheader

.preheader:                                       ; preds = %35, %.preheader
  %50 = phi i64 [ %62, %.preheader ], [ 0, %35 ]
  tail call void @llvm.aie2p.acquire(i32 49, i32 -1)
  %51 = trunc nuw i64 %50 to i32
  tail call void @generic_bcast_gemv_bf16_g(i32 %51, ptr nonnull @c2_A0, ptr nonnull @c2_B2, ptr nonnull @c2_uni_partial, i32 8, ptr nonnull @c2_up)
  tail call void @llvm.aie2p.release(i32 48, i32 1)
  tail call void @llvm.aie2p.acquire(i32 59, i32 -1)
  %52 = or disjoint i32 %51, 1
  tail call void @generic_bcast_gemv_bf16_g(i32 %52, ptr nonnull @c2_A1, ptr nonnull @c2_B2, ptr nonnull @c2_uni_partial, i32 8, ptr nonnull @c2_up)
  tail call void @llvm.aie2p.release(i32 58, i32 1)
  tail call void @llvm.aie2p.acquire(i32 49, i32 -1)
  %53 = trunc i64 %50 to i32
  %54 = or disjoint i32 %53, 2
  tail call void @generic_bcast_gemv_bf16_g(i32 %54, ptr nonnull @c2_A0, ptr nonnull @c2_B2, ptr nonnull @c2_uni_partial, i32 8, ptr nonnull @c2_up)
  tail call void @llvm.aie2p.release(i32 48, i32 1)
  tail call void @llvm.aie2p.acquire(i32 59, i32 -1)
  %55 = or disjoint i32 %53, 3
  tail call void @generic_bcast_gemv_bf16_g(i32 %55, ptr nonnull @c2_A1, ptr nonnull @c2_B2, ptr nonnull @c2_uni_partial, i32 8, ptr nonnull @c2_up)
  tail call void @llvm.aie2p.release(i32 58, i32 1)
  tail call void @llvm.aie2p.acquire(i32 49, i32 -1)
  %56 = trunc i64 %50 to i32
  %57 = or disjoint i32 %56, 4
  tail call void @generic_bcast_gemv_bf16_g(i32 %57, ptr nonnull @c2_A0, ptr nonnull @c2_B2, ptr nonnull @c2_uni_partial, i32 8, ptr nonnull @c2_up)
  tail call void @llvm.aie2p.release(i32 48, i32 1)
  tail call void @llvm.aie2p.acquire(i32 59, i32 -1)
  %58 = or disjoint i32 %56, 5
  tail call void @generic_bcast_gemv_bf16_g(i32 %58, ptr nonnull @c2_A1, ptr nonnull @c2_B2, ptr nonnull @c2_uni_partial, i32 8, ptr nonnull @c2_up)
  tail call void @llvm.aie2p.release(i32 58, i32 1)
  %59 = or disjoint i64 %50, 6
  tail call void @llvm.aie2p.acquire(i32 49, i32 -1)
  %60 = trunc nuw i64 %59 to i32
  tail call void @generic_bcast_gemv_bf16_g(i32 %60, ptr nonnull @c2_A0, ptr nonnull @c2_B2, ptr nonnull @c2_uni_partial, i32 8, ptr nonnull @c2_up)
  tail call void @llvm.aie2p.release(i32 48, i32 1)
  tail call void @llvm.aie2p.acquire(i32 59, i32 -1)
  %61 = or disjoint i32 %60, 1
  tail call void @generic_bcast_gemv_bf16_g(i32 %61, ptr nonnull @c2_A1, ptr nonnull @c2_B2, ptr nonnull @c2_uni_partial, i32 8, ptr nonnull @c2_up)
  tail call void @llvm.aie2p.release(i32 58, i32 1)
  %62 = add nuw nsw i64 %50, 8
  %63 = icmp samesign ult i64 %59, 254
  br i1 %63, label %.preheader, label %64

64:                                               ; preds = %.preheader
  tail call void @llvm.aie2p.release(i32 62, i32 1)
  tail call void @layer_fused_silu_mul_explicit_bf16(ptr nonnull @c2_gate, ptr nonnull @c2_up, ptr nonnull @c2_silu, i32 1024)
  tail call void @llvm.aie2p.acquire(i32 56, i32 -1)
  br label %65

65:                                               ; preds = %65, %64
  %66 = phi i64 [ 0, %64 ], [ %78, %65 ]
  tail call void @llvm.aie2p.acquire(i32 49, i32 -1)
  %67 = trunc nuw i64 %66 to i32
  tail call void @generic_bcast_gemv_bf16_d(i32 %67, ptr nonnull @c2_A0, ptr nonnull @c2_silu, ptr nonnull @c2_uni_partial, i32 4, ptr nonnull @c2_P)
  tail call void @llvm.aie2p.release(i32 48, i32 1)
  tail call void @llvm.aie2p.acquire(i32 59, i32 -1)
  %68 = or disjoint i32 %67, 1
  tail call void @generic_bcast_gemv_bf16_d(i32 %68, ptr nonnull @c2_A1, ptr nonnull @c2_silu, ptr nonnull @c2_uni_partial, i32 4, ptr nonnull @c2_P)
  tail call void @llvm.aie2p.release(i32 58, i32 1)
  tail call void @llvm.aie2p.acquire(i32 49, i32 -1)
  %69 = trunc i64 %66 to i32
  %70 = or disjoint i32 %69, 2
  tail call void @generic_bcast_gemv_bf16_d(i32 %70, ptr nonnull @c2_A0, ptr nonnull @c2_silu, ptr nonnull @c2_uni_partial, i32 4, ptr nonnull @c2_P)
  tail call void @llvm.aie2p.release(i32 48, i32 1)
  tail call void @llvm.aie2p.acquire(i32 59, i32 -1)
  %71 = or disjoint i32 %69, 3
  tail call void @generic_bcast_gemv_bf16_d(i32 %71, ptr nonnull @c2_A1, ptr nonnull @c2_silu, ptr nonnull @c2_uni_partial, i32 4, ptr nonnull @c2_P)
  tail call void @llvm.aie2p.release(i32 58, i32 1)
  tail call void @llvm.aie2p.acquire(i32 49, i32 -1)
  %72 = trunc i64 %66 to i32
  %73 = or disjoint i32 %72, 4
  tail call void @generic_bcast_gemv_bf16_d(i32 %73, ptr nonnull @c2_A0, ptr nonnull @c2_silu, ptr nonnull @c2_uni_partial, i32 4, ptr nonnull @c2_P)
  tail call void @llvm.aie2p.release(i32 48, i32 1)
  tail call void @llvm.aie2p.acquire(i32 59, i32 -1)
  %74 = or disjoint i32 %72, 5
  tail call void @generic_bcast_gemv_bf16_d(i32 %74, ptr nonnull @c2_A1, ptr nonnull @c2_silu, ptr nonnull @c2_uni_partial, i32 4, ptr nonnull @c2_P)
  tail call void @llvm.aie2p.release(i32 58, i32 1)
  %75 = or disjoint i64 %66, 6
  tail call void @llvm.aie2p.acquire(i32 49, i32 -1)
  %76 = trunc nuw i64 %75 to i32
  tail call void @generic_bcast_gemv_bf16_d(i32 %76, ptr nonnull @c2_A0, ptr nonnull @c2_silu, ptr nonnull @c2_uni_partial, i32 4, ptr nonnull @c2_P)
  tail call void @llvm.aie2p.release(i32 48, i32 1)
  tail call void @llvm.aie2p.acquire(i32 59, i32 -1)
  %77 = or disjoint i32 %76, 1
  tail call void @generic_bcast_gemv_bf16_d(i32 %77, ptr nonnull @c2_A1, ptr nonnull @c2_silu, ptr nonnull @c2_uni_partial, i32 4, ptr nonnull @c2_P)
  tail call void @llvm.aie2p.release(i32 58, i32 1)
  %78 = add nuw nsw i64 %66, 8
  %79 = icmp samesign ult i64 %75, 254
  br i1 %79, label %65, label %80

80:                                               ; preds = %65
  tail call void @llvm.aie2p.release(i32 57, i32 1)
  %81 = add nuw nsw i64 %2, 1
  %.not = icmp eq i64 %81, 9223372036854775807
  br i1 %.not, label %82, label %1

82:                                               ; preds = %80
  ret void
}

attributes #0 = { mustprogress nocallback nofree nosync nounwind willreturn }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}

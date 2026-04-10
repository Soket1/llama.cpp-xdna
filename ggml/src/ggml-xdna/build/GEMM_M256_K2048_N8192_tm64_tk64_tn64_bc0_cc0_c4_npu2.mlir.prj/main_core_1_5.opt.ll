; ModuleID = 'GEMM_M256_K2048_N8192_tm64_tk64_tn64_bc0_cc0_c4_npu2.mlir.prj/main_core_1_5.peanohack.ll'
source_filename = "LLVMDialectModule"
target datalayout = "e-m:e-p:20:32-i1:8:32-i8:8:32-i16:16:32-i32:32:32-f32:32:32-i64:32-f64:32-a:0:32-n32"
target triple = "aie2p"

@_anonymous13 = external local_unnamed_addr global [3 x i32]
@rtp3_1 = local_unnamed_addr global [2 x i32] zeroinitializer
@A_L2L1_3_1_cons_buff_1 = external global [64 x [64 x bfloat]]
@A_L2L1_3_1_cons_buff_0 = external global [64 x [64 x bfloat]]
@B_L2L1_1_3_cons_buff_1 = external global [64 x [64 x bfloat]]
@B_L2L1_1_3_cons_buff_0 = external global [64 x [64 x bfloat]]
@C_L1L2_1_3_buff_1 = external global [64 x [64 x bfloat]]
@C_L1L2_1_3_buff_0 = external global [64 x [64 x bfloat]]

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn
declare void @llvm.aie2p.acquire(i32, i32) #0

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn
declare void @llvm.aie2p.release(i32, i32) #0

declare void @zero_bf16(ptr) local_unnamed_addr

declare void @matmul_bf16_bf16(ptr, ptr, ptr) local_unnamed_addr

define void @core_1_5() local_unnamed_addr {
  store i32 0, ptr @_anonymous13, align 4
  store i32 0, ptr getelementptr inbounds nuw (i8, ptr @_anonymous13, i20 4), align 4
  store i32 0, ptr getelementptr inbounds nuw (i8, ptr @_anonymous13, i20 8), align 4
  br label %1

1:                                                ; preds = %0, %._crit_edge8
  %2 = phi i64 [ 0, %0 ], [ %35, %._crit_edge8 ]
  tail call void @llvm.aie2p.acquire(i32 48, i32 1)
  %3 = load i32, ptr @rtp3_1, align 4
  %4 = load i32, ptr getelementptr inbounds nuw (i8, ptr @rtp3_1, i20 4), align 4
  %5 = sext i32 %4 to i64
  %6 = sext i32 %3 to i64
  %7 = icmp sgt i32 %4, 0
  br i1 %7, label %.lr.ph7, label %._crit_edge8

.lr.ph7:                                          ; preds = %1
  %8 = icmp sgt i32 %3, 0
  br label %9

9:                                                ; preds = %.lr.ph7, %._crit_edge
  %10 = phi i64 [ 0, %.lr.ph7 ], [ %33, %._crit_edge ]
  tail call void @llvm.aie2p.acquire(i32 53, i32 -1)
  %11 = load i32, ptr @_anonymous13, align 4
  %cond = icmp eq i32 %11, 1
  %spec.select = select i1 %cond, ptr @C_L1L2_1_3_buff_1, ptr @C_L1L2_1_3_buff_0
  tail call void @zero_bf16(ptr nonnull %spec.select)
  br i1 %8, label %.lr.ph, label %._crit_edge

.lr.ph:                                           ; preds = %9, %.lr.ph
  %12 = phi i64 [ %26, %.lr.ph ], [ 0, %9 ]
  tail call void @llvm.aie2p.acquire(i32 50, i32 -1)
  %13 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous13, i20 4), align 4
  %cond1 = icmp eq i32 %13, 1
  %spec.select5 = select i1 %cond1, ptr @A_L2L1_3_1_cons_buff_1, ptr @A_L2L1_3_1_cons_buff_0
  tail call void @llvm.aie2p.acquire(i32 52, i32 -1)
  %14 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous13, i20 8), align 4
  %cond2 = icmp eq i32 %14, 1
  %15 = select i1 %cond2, ptr @B_L2L1_1_3_cons_buff_1, ptr @B_L2L1_1_3_cons_buff_0
  tail call void @matmul_bf16_bf16(ptr nonnull %spec.select5, ptr nonnull %15, ptr nonnull %spec.select)
  tail call void @llvm.aie2p.release(i32 49, i32 1)
  %16 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous13, i20 4), align 4
  %17 = add i32 %16, 1
  %18 = icmp sgt i32 %17, 1
  %19 = add i32 %16, -1
  %20 = select i1 %18, i32 %19, i32 %17
  store i32 %20, ptr getelementptr inbounds nuw (i8, ptr @_anonymous13, i20 4), align 4
  tail call void @llvm.aie2p.release(i32 51, i32 1)
  %21 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous13, i20 8), align 4
  %22 = add i32 %21, 1
  %23 = icmp sgt i32 %22, 1
  %24 = add i32 %21, -1
  %25 = select i1 %23, i32 %24, i32 %22
  store i32 %25, ptr getelementptr inbounds nuw (i8, ptr @_anonymous13, i20 8), align 4
  %26 = add nuw nsw i64 %12, 1
  %27 = icmp slt i64 %26, %6
  br i1 %27, label %.lr.ph, label %._crit_edge

._crit_edge:                                      ; preds = %.lr.ph, %9
  tail call void @llvm.aie2p.release(i32 54, i32 1)
  %28 = load i32, ptr @_anonymous13, align 4
  %29 = add i32 %28, 1
  %30 = icmp sgt i32 %29, 1
  %31 = add i32 %28, -1
  %32 = select i1 %30, i32 %31, i32 %29
  store i32 %32, ptr @_anonymous13, align 4
  %33 = add nuw nsw i64 %10, 1
  %34 = icmp slt i64 %33, %5
  br i1 %34, label %9, label %._crit_edge8

._crit_edge8:                                     ; preds = %._crit_edge, %1
  %35 = add nuw nsw i64 %2, 1
  %.not = icmp eq i64 %35, 9223372036854775807
  br i1 %.not, label %36, label %1

36:                                               ; preds = %._crit_edge8
  ret void
}

attributes #0 = { mustprogress nocallback nofree nosync nounwind willreturn }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}

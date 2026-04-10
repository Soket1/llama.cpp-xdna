; ModuleID = 'GEMM_M256_K256_N256_tm64_tk64_tn64_bc0_cc0_c4_npu2.mlir.prj/main_core_1_4.peanohack.ll'
source_filename = "LLVMDialectModule"
target datalayout = "e-m:e-p:20:32-i1:8:32-i8:8:32-i16:16:32-i32:32:32-f32:32:32-i64:32-f64:32-a:0:32-n32"
target triple = "aie2p"

@_anonymous9 = external local_unnamed_addr global [3 x i32]
@acc_buffer_2_1 = external global [64 x [64 x float]]
@rtp2_1 = local_unnamed_addr global [2 x i32] zeroinitializer
@A_L2L1_2_1_cons_buff_1 = external global [64 x [64 x bfloat]]
@A_L2L1_2_1_cons_buff_0 = external global [64 x [64 x bfloat]]
@B_L2L1_1_2_cons_buff_1 = external global [64 x [64 x bfloat]]
@B_L2L1_1_2_cons_buff_0 = external global [64 x [64 x bfloat]]
@C_L1L2_1_2_buff_0 = external global [64 x [64 x bfloat]]

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn
declare void @llvm.aie2p.acquire(i32, i32) #0

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn
declare void @llvm.aie2p.release(i32, i32) #0

declare void @zero_f32(ptr) local_unnamed_addr

declare void @matmul_bf16_f32(ptr, ptr, ptr) local_unnamed_addr

declare void @convert_copy_f32_to_bf16(ptr, ptr, i32) local_unnamed_addr

define void @core_1_4() local_unnamed_addr {
  store i32 0, ptr @_anonymous9, align 4
  store i32 0, ptr getelementptr inbounds nuw (i8, ptr @_anonymous9, i20 4), align 4
  store i32 0, ptr getelementptr inbounds nuw (i8, ptr @_anonymous9, i20 8), align 4
  br label %1

1:                                                ; preds = %0, %._crit_edge6
  %2 = phi i64 [ 0, %0 ], [ %33, %._crit_edge6 ]
  tail call void @llvm.aie2p.acquire(i32 48, i32 1)
  %3 = load i32, ptr @rtp2_1, align 4
  %4 = load i32, ptr getelementptr inbounds nuw (i8, ptr @rtp2_1, i20 4), align 4
  %5 = sext i32 %4 to i64
  %6 = sext i32 %3 to i64
  %7 = icmp sgt i32 %4, 0
  br i1 %7, label %.lr.ph5, label %._crit_edge6

.lr.ph5:                                          ; preds = %1
  %8 = icmp sgt i32 %3, 0
  br label %9

9:                                                ; preds = %.lr.ph5, %._crit_edge
  %10 = phi i64 [ 0, %.lr.ph5 ], [ %31, %._crit_edge ]
  tail call void @zero_f32(ptr nonnull @acc_buffer_2_1)
  br i1 %8, label %.lr.ph, label %._crit_edge

.lr.ph:                                           ; preds = %9, %.lr.ph
  %11 = phi i64 [ %25, %.lr.ph ], [ 0, %9 ]
  tail call void @llvm.aie2p.acquire(i32 50, i32 -1)
  %12 = load i32, ptr @_anonymous9, align 4
  %cond = icmp eq i32 %12, 1
  %spec.select = select i1 %cond, ptr @A_L2L1_2_1_cons_buff_1, ptr @A_L2L1_2_1_cons_buff_0
  tail call void @llvm.aie2p.acquire(i32 52, i32 -1)
  %13 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous9, i20 4), align 4
  %cond1 = icmp eq i32 %13, 1
  %14 = select i1 %cond1, ptr @B_L2L1_1_2_cons_buff_1, ptr @B_L2L1_1_2_cons_buff_0
  tail call void @matmul_bf16_f32(ptr nonnull %spec.select, ptr nonnull %14, ptr nonnull @acc_buffer_2_1)
  tail call void @llvm.aie2p.release(i32 49, i32 1)
  %15 = load i32, ptr @_anonymous9, align 4
  %16 = add i32 %15, 1
  %17 = icmp sgt i32 %16, 1
  %18 = add i32 %15, -1
  %19 = select i1 %17, i32 %18, i32 %16
  store i32 %19, ptr @_anonymous9, align 4
  tail call void @llvm.aie2p.release(i32 51, i32 1)
  %20 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous9, i20 4), align 4
  %21 = add i32 %20, 1
  %22 = icmp sgt i32 %21, 1
  %23 = add i32 %20, -1
  %24 = select i1 %22, i32 %23, i32 %21
  store i32 %24, ptr getelementptr inbounds nuw (i8, ptr @_anonymous9, i20 4), align 4
  %25 = add nuw nsw i64 %11, 1
  %26 = icmp slt i64 %25, %6
  br i1 %26, label %.lr.ph, label %._crit_edge

._crit_edge:                                      ; preds = %.lr.ph, %9
  tail call void @llvm.aie2p.acquire(i32 53, i32 -1)
  tail call void @convert_copy_f32_to_bf16(ptr nonnull @acc_buffer_2_1, ptr nonnull @C_L1L2_1_2_buff_0, i32 4096)
  tail call void @llvm.aie2p.release(i32 54, i32 1)
  %27 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous9, i20 8), align 4
  %28 = icmp ugt i32 %27, 2147483646
  %29 = zext i1 %28 to i32
  %30 = add i32 %27, %29
  store i32 %30, ptr getelementptr inbounds nuw (i8, ptr @_anonymous9, i20 8), align 4
  %31 = add nuw nsw i64 %10, 1
  %32 = icmp slt i64 %31, %5
  br i1 %32, label %9, label %._crit_edge6

._crit_edge6:                                     ; preds = %._crit_edge, %1
  %33 = add nuw nsw i64 %2, 1
  %.not = icmp eq i64 %33, 9223372036854775807
  br i1 %.not, label %34, label %1

34:                                               ; preds = %._crit_edge6
  ret void
}

attributes #0 = { mustprogress nocallback nofree nosync nounwind willreturn }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}

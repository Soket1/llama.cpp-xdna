; ModuleID = 'decode_layer_f3best_2048x8192_d64_g32_s256_a4_kv8_mc_preq_vexp_vreg_dq8_qp_mxp_tb.mlir.prj\main_core_4_4.peanohack.ll'
source_filename = "LLVMDialectModule"
target datalayout = "e-m:e-p:20:32-i1:8:32-i8:8:32-i16:16:32-i32:32:32-f32:32:32-i64:32-f64:32-a:0:32-n32"
target triple = "aie2p"

@nm_gain = external global [2048 x bfloat]
@nm_FN = external global [2320 x bfloat]
@nm_F = external global [2320 x bfloat]
@nm_R = external global [2048 x bfloat]
@nm_O = external global [2048 x bfloat]

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn
declare void @llvm.aie2p.acquire(i32, i32) #0

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn
declare void @llvm.aie2p.release(i32, i32) #0

declare void @layer_fused_add_bf16(ptr, ptr, ptr, i32) local_unnamed_addr

declare void @layer_fused_rms_norm2_bf16(ptr, ptr, ptr, i32) local_unnamed_addr

define void @core_4_4() local_unnamed_addr {
  br label %1

1:                                                ; preds = %0, %1
  %2 = phi i64 [ 0, %0 ], [ %3, %1 ]
  tail call void @llvm.aie2p.acquire(i32 49, i32 -1)
  tail call void @llvm.aie2p.acquire(i32 51, i32 -1)
  tail call void @llvm.aie2p.acquire(i32 55, i32 -1)
  tail call void @llvm.aie2p.acquire(i32 52, i32 -1)
  tail call void @llvm.aie2p.acquire(i32 56, i32 -1)
  tail call void @layer_fused_add_bf16(ptr nonnull @nm_O, ptr nonnull @nm_R, ptr nonnull @nm_F, i32 2048)
  tail call void @layer_fused_rms_norm2_bf16(ptr nonnull @nm_F, ptr nonnull @nm_gain, ptr nonnull @nm_FN, i32 2048)
  tail call void @llvm.aie2p.release(i32 48, i32 1)
  tail call void @llvm.aie2p.release(i32 50, i32 1)
  tail call void @llvm.aie2p.release(i32 54, i32 1)
  tail call void @llvm.aie2p.release(i32 53, i32 1)
  tail call void @llvm.aie2p.release(i32 57, i32 1)
  %3 = add nuw nsw i64 %2, 1
  %.not = icmp eq i64 %3, 9223372036854775807
  br i1 %.not, label %4, label %1

4:                                                ; preds = %1
  ret void
}

attributes #0 = { mustprogress nocallback nofree nosync nounwind willreturn }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}

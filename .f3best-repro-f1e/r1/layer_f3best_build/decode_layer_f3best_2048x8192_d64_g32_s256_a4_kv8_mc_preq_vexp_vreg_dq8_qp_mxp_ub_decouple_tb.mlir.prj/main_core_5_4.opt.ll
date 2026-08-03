; ModuleID = 'decode_layer_f3best_2048x8192_d64_g32_s256_a4_kv8_mc_preq_vexp_vreg_dq8_qp_mxp_ub_decouple_tb.mlir.prj\main_core_5_4.peanohack.ll'
source_filename = "LLVMDialectModule"
target datalayout = "e-m:e-p:20:32-i1:8:32-i8:8:32-i16:16:32-i32:32:32-f32:32:32-i64:32-f64:32-a:0:32-n32"
target triple = "aie2p"

@mx_o = external global [2320 x bfloat]
@mx_f = external global [2320 x bfloat]
@mx_a = external global [2320 x bfloat]
@mx_x = external global [2320 x bfloat]

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn
declare void @llvm.aie2p.acquire(i32, i32) #0

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn
declare void @llvm.aie2p.release(i32, i32) #0

declare void @attn_copy_bf16(ptr, ptr, i32) local_unnamed_addr

define void @core_5_4() local_unnamed_addr {
  br label %1

1:                                                ; preds = %0, %1
  %2 = phi i64 [ 0, %0 ], [ %3, %1 ]
  tail call void @llvm.aie2p.acquire(i32 49, i32 -1)
  tail call void @llvm.aie2p.acquire(i32 54, i32 -1)
  tail call void @attn_copy_bf16(ptr nonnull @mx_x, ptr nonnull @mx_o, i32 2320)
  tail call void @llvm.aie2p.release(i32 48, i32 1)
  tail call void @llvm.aie2p.release(i32 55, i32 1)
  tail call void @llvm.aie2p.acquire(i32 51, i32 -1)
  tail call void @llvm.aie2p.acquire(i32 54, i32 -1)
  tail call void @attn_copy_bf16(ptr nonnull @mx_a, ptr nonnull @mx_o, i32 2048)
  tail call void @llvm.aie2p.release(i32 50, i32 1)
  tail call void @llvm.aie2p.release(i32 55, i32 1)
  tail call void @llvm.aie2p.acquire(i32 53, i32 -1)
  tail call void @llvm.aie2p.acquire(i32 54, i32 -1)
  tail call void @attn_copy_bf16(ptr nonnull @mx_f, ptr nonnull @mx_o, i32 2048)
  tail call void @llvm.aie2p.release(i32 52, i32 1)
  tail call void @llvm.aie2p.release(i32 55, i32 1)
  %3 = add nuw nsw i64 %2, 1
  %.not = icmp eq i64 %3, 9223372036854775807
  br i1 %.not, label %4, label %1

4:                                                ; preds = %1
  ret void
}

attributes #0 = { mustprogress nocallback nofree nosync nounwind willreturn }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}

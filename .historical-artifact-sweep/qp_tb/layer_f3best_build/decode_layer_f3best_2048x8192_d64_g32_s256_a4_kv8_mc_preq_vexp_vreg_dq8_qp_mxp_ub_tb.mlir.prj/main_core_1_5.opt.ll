; ModuleID = 'decode_layer_f3best_2048x8192_d64_g32_s256_a4_kv8_mc_preq_vexp_vreg_dq8_qp_mxp_ub_tb.mlir.prj\main_core_1_5.peanohack.ll'
source_filename = "LLVMDialectModule"
target datalayout = "e-m:e-p:20:32-i1:8:32-i8:8:32-i16:16:32-i32:32:32-f32:32:32-i64:32-f64:32-a:0:32-n32"
target triple = "aie2p"

@va5_Of = external global [256 x bfloat]
@va5_V = external global [8192 x bfloat]
@va5_Iv = external global [520 x bfloat]

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn
declare void @llvm.aie2p.acquire(i32, i32) #0

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn
declare void @llvm.aie2p.release(i32, i32) #0

declare void @flowkv_value_init_bf16(i32, i32) local_unnamed_addr

declare void @flowkv_value_accum_bf16(ptr, ptr, i32, i32, i32) local_unnamed_addr

declare void @flowkv_value_normalize_bf16(ptr, i32, i32) local_unnamed_addr

define void @core_1_5() local_unnamed_addr {
  br label %1

1:                                                ; preds = %0, %1
  %2 = phi i64 [ 0, %0 ], [ %3, %1 ]
  tail call void @flowkv_value_init_bf16(i32 4, i32 64)
  tail call void @llvm.aie2p.acquire(i32 49, i32 -1)
  tail call void @llvm.aie2p.acquire(i32 51, i32 -1)
  tail call void @flowkv_value_accum_bf16(ptr nonnull @va5_Iv, ptr nonnull @va5_V, i32 4, i32 64, i32 128)
  tail call void @llvm.aie2p.release(i32 48, i32 1)
  tail call void @llvm.aie2p.release(i32 50, i32 1)
  tail call void @llvm.aie2p.acquire(i32 49, i32 -1)
  tail call void @llvm.aie2p.acquire(i32 51, i32 -1)
  tail call void @flowkv_value_accum_bf16(ptr nonnull @va5_Iv, ptr nonnull @va5_V, i32 4, i32 64, i32 128)
  tail call void @llvm.aie2p.release(i32 48, i32 1)
  tail call void @llvm.aie2p.release(i32 50, i32 1)
  tail call void @llvm.aie2p.acquire(i32 52, i32 -1)
  tail call void @flowkv_value_normalize_bf16(ptr nonnull @va5_Of, i32 4, i32 64)
  tail call void @llvm.aie2p.release(i32 53, i32 1)
  %3 = add nuw nsw i64 %2, 1
  %.not = icmp eq i64 %3, 9223372036854775807
  br i1 %.not, label %4, label %1

4:                                                ; preds = %1
  ret void
}

attributes #0 = { mustprogress nocallback nofree nosync nounwind willreturn }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}

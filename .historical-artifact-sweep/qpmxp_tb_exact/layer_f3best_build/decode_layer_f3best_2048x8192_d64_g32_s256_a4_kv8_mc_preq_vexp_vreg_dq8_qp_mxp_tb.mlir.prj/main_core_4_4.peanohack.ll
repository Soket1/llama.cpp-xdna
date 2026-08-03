; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target triple = "aie2p"

@mx_o = external global [2320 x bfloat]
@mx_f = external global [2320 x bfloat]
@mx_a = external global [2320 x bfloat]
@mx_x = external global [2320 x bfloat]
@nm_gain = external global [2048 x bfloat]
@nm_FN = external global [2320 x bfloat]
@nm_F = external global [2320 x bfloat]
@nm_R = external global [2048 x bfloat]
@nm_O = external global [2048 x bfloat]
@op_O = external global [2048 x bfloat]
@rl_A = external global [2048 x bfloat]
@Vhi_buf = external global [65536 x bfloat]
@Vlo_buf = external global [65536 x bfloat]
@Khi_buf = external global [65536 x bfloat]
@Klo_buf = external global [65536 x bfloat]
@oK_buf = external global [2048 x bfloat]
@oJ_buf = external global [2048 x bfloat]
@jB_buf = external global [2048 x bfloat]
@jA_buf = external global [2048 x bfloat]
@va7_Of = external global [256 x bfloat]
@va7_V = external global [8192 x bfloat]
@va7_Iv = external global [520 x bfloat]
@va6_Of = external global [256 x bfloat]
@va6_V = external global [8192 x bfloat]
@va6_Iv = external global [520 x bfloat]
@va5_Of = external global [256 x bfloat]
@va5_V = external global [8192 x bfloat]
@va5_Iv = external global [520 x bfloat]
@va4_Of = external global [256 x bfloat]
@va4_V = external global [8192 x bfloat]
@va4_Iv = external global [520 x bfloat]
@va3_Of = external global [256 x bfloat]
@va3_V = external global [8192 x bfloat]
@va3_Iv = external global [520 x bfloat]
@va2_Of = external global [256 x bfloat]
@va2_V = external global [8192 x bfloat]
@va2_Iv = external global [520 x bfloat]
@va1_Of = external global [256 x bfloat]
@va1_V = external global [8192 x bfloat]
@va1_Iv = external global [520 x bfloat]
@va0_Of = external global [256 x bfloat]
@va0_V = external global [8192 x bfloat]
@va0_Iv = external global [520 x bfloat]
@sc7_Oh = external global [256 x bfloat]
@sc7_It = external global [520 x bfloat]
@sc7_K = external global [8192 x bfloat]
@sc7_Qs = external global [322 x bfloat]
@sc6_Oh = external global [256 x bfloat]
@sc6_It = external global [520 x bfloat]
@sc6_K = external global [8192 x bfloat]
@sc6_Qs = external global [322 x bfloat]
@sc5_Oh = external global [256 x bfloat]
@sc5_It = external global [520 x bfloat]
@sc5_K = external global [8192 x bfloat]
@sc5_Qs = external global [322 x bfloat]
@sc4_Oh = external global [256 x bfloat]
@sc4_It = external global [520 x bfloat]
@sc4_K = external global [8192 x bfloat]
@sc4_Qs = external global [322 x bfloat]
@sc3_Oh = external global [256 x bfloat]
@sc3_It = external global [520 x bfloat]
@sc3_K = external global [8192 x bfloat]
@sc3_Qs = external global [322 x bfloat]
@sc2_Oh = external global [256 x bfloat]
@sc2_It = external global [520 x bfloat]
@sc2_K = external global [8192 x bfloat]
@sc2_Qs = external global [322 x bfloat]
@sc1_Oh = external global [256 x bfloat]
@sc1_It = external global [520 x bfloat]
@sc1_K = external global [8192 x bfloat]
@sc1_Qs = external global [322 x bfloat]
@sc0_Oh = external global [256 x bfloat]
@sc0_It = external global [520 x bfloat]
@sc0_K = external global [8192 x bfloat]
@sc0_Qs = external global [322 x bfloat]
@c7_P = external global [2320 x bfloat]
@c7_O = external global [2048 x bfloat]
@c7_Q = external global [322 x bfloat]
@c7_B2 = external global [2320 x bfloat]
@c7_B1 = external global [2320 x bfloat]
@c7_B0 = external global [2320 x bfloat]
@c7_A1 = external global [4608 x i8]
@c7_A0 = external global [4608 x i8]
@c6_P = external global [2320 x bfloat]
@c6_O = external global [2048 x bfloat]
@c6_Q = external global [322 x bfloat]
@c6_B2 = external global [2320 x bfloat]
@c6_B1 = external global [2320 x bfloat]
@c6_B0 = external global [2320 x bfloat]
@c6_A1 = external global [4608 x i8]
@c6_A0 = external global [4608 x i8]
@c5_P = external global [2320 x bfloat]
@c5_O = external global [2048 x bfloat]
@c5_Q = external global [322 x bfloat]
@c5_B2 = external global [2320 x bfloat]
@c5_B1 = external global [2320 x bfloat]
@c5_B0 = external global [2320 x bfloat]
@c5_A1 = external global [4608 x i8]
@c5_A0 = external global [4608 x i8]
@c4_P = external global [2320 x bfloat]
@c4_O = external global [2048 x bfloat]
@c4_Q = external global [322 x bfloat]
@c4_B2 = external global [2320 x bfloat]
@c4_B1 = external global [2320 x bfloat]
@c4_B0 = external global [2320 x bfloat]
@c4_A1 = external global [4608 x i8]
@c4_A0 = external global [4608 x i8]
@c3_P = external global [2320 x bfloat]
@c3_O = external global [2048 x bfloat]
@c3_Q = external global [322 x bfloat]
@c3_B2 = external global [2320 x bfloat]
@c3_B1 = external global [2320 x bfloat]
@c3_B0 = external global [2320 x bfloat]
@c3_A1 = external global [4608 x i8]
@c3_A0 = external global [4608 x i8]
@c2_P = external global [2320 x bfloat]
@c2_O = external global [2048 x bfloat]
@c2_Q = external global [322 x bfloat]
@c2_B2 = external global [2320 x bfloat]
@c2_B1 = external global [2320 x bfloat]
@c2_B0 = external global [2320 x bfloat]
@c2_A1 = external global [4608 x i8]
@c2_A0 = external global [4608 x i8]
@c1_P = external global [2320 x bfloat]
@c1_O = external global [2048 x bfloat]
@c1_Q = external global [322 x bfloat]
@c1_B2 = external global [2320 x bfloat]
@c1_B1 = external global [2320 x bfloat]
@c1_B0 = external global [2320 x bfloat]
@c1_A1 = external global [4608 x i8]
@c1_A0 = external global [4608 x i8]
@c0_P = external global [2320 x bfloat]
@c0_O = external global [2048 x bfloat]
@c0_Q = external global [322 x bfloat]
@c0_B2 = external global [2320 x bfloat]
@c0_B1 = external global [2320 x bfloat]
@c0_B0 = external global [2320 x bfloat]
@c0_A1 = external global [4608 x i8]
@c0_A0 = external global [4608 x i8]

declare void @debug_i32(i32)

; Unknown intrinsic
declare void @llvm.aie2p.event(i32)

; Unknown intrinsic
declare void @llvm.aie2p.put.ms(i32, i32)

; Unknown intrinsic
declare { i32, i32 } @llvm.aie2p.get.ss()

; Unknown intrinsic
declare void @llvm.aie2p.mcd.write.vec(<16 x i32>, i32)

; Unknown intrinsic
declare <16 x i32> @llvm.aie2p.scd.read.vec(i32)

; Unknown intrinsic
declare void @llvm.aie2p.acquire(i32, i32)

; Unknown intrinsic
declare void @llvm.aie2p.release(i32, i32)

; Unknown intrinsic
declare void @llvm.aie2p.set.ctrl.reg(i32, i32)

declare void @_ha_noop()

declare void @fused_dequant_matvec_v2_bf16(i32, i32, ptr, ptr, ptr)

declare void @rope_bundled(ptr, ptr, ptr, i32)

declare void @layer_fused_gate_up_bf16(i32, i32, ptr, ptr, i32)

declare void @layer_fused_qkv_bcast_bf16(i32, i32, ptr, ptr, ptr)

declare void @layer_fused_oproj_bcast_bf16(i32, i32, ptr, ptr, ptr)

declare void @layer_fused_gate_up_bcast_bf16(i32, i32, ptr, ptr, i32)

declare void @layer_fused_silu_mul_static_bf16(i32)

declare void @layer_fused_down_v2_x4_bf16(i32, i32, i32, ptr, ptr)

declare void @layer_fused_down_bcast_bf16(i32, i32, ptr, ptr)

declare void @attn_copy_bf16(ptr, ptr, i32)

declare void @oproj_matvec_v2_bf16(i32, i32, ptr, ptr, ptr)

declare void @layer_fused_add_bf16(ptr, ptr, ptr, i32)

declare void @layer_fused_rms_norm2_bf16(ptr, ptr, ptr, i32)

declare void @flowkv_score_init_bf16(i32)

declare void @flowkv_score_rope_q_bf16(ptr, i32, i32)

declare void @flowkv_score_chunk_bf16(ptr, ptr, ptr, i32, i32, i32)

declare void @flowkv_value_init_bf16(i32, i32)

declare void @flowkv_value_accum_bf16(ptr, ptr, i32, i32, i32)

declare void @flowkv_value_normalize_bf16(ptr, i32, i32)

define void @core_4_4() {
  br label %1

1:                                                ; preds = %4, %0
  %2 = phi i64 [ %5, %4 ], [ 0, %0 ]
  %3 = icmp slt i64 %2, 9223372036854775807
  br i1 %3, label %4, label %6

4:                                                ; preds = %1
  call void @llvm.aie2p.acquire(i32 49, i32 -1)
  call void @llvm.aie2p.acquire(i32 51, i32 -1)
  call void @llvm.aie2p.acquire(i32 55, i32 -1)
  call void @llvm.aie2p.acquire(i32 52, i32 -1)
  call void @llvm.aie2p.acquire(i32 56, i32 -1)
  call void @layer_fused_add_bf16(ptr @nm_O, ptr @nm_R, ptr @nm_F, i32 2048)
  call void @layer_fused_rms_norm2_bf16(ptr @nm_F, ptr @nm_gain, ptr @nm_FN, i32 2048)
  call void @llvm.aie2p.release(i32 48, i32 1)
  call void @llvm.aie2p.release(i32 50, i32 1)
  call void @llvm.aie2p.release(i32 54, i32 1)
  call void @llvm.aie2p.release(i32 53, i32 1)
  call void @llvm.aie2p.release(i32 57, i32 1)
  %5 = add i64 %2, 1
  br label %1

6:                                                ; preds = %1
  ret void
}

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}

module {
  aie.device(npu2) {
    %tile_2_2 = aie.tile(2, 2) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 27>}
    %tile_3_2 = aie.tile(3, 2) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 27>}
    %tile_4_2 = aie.tile(4, 2) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 27>}
    %tile_5_2 = aie.tile(5, 2) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 27>}
    %tile_2_3 = aie.tile(2, 3) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 29>}
    %tile_3_3 = aie.tile(3, 3) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 29>}
    %tile_4_3 = aie.tile(4, 3) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 29>}
    %tile_5_3 = aie.tile(5, 3) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 29>}
    %tile_0_2 = aie.tile(0, 2) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 27>}
    %tile_1_2 = aie.tile(1, 2) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 27>}
    %tile_6_2 = aie.tile(6, 2) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 27>}
    %tile_7_2 = aie.tile(7, 2) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 27>}
    %tile_0_3 = aie.tile(0, 3) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 29>}
    %tile_1_3 = aie.tile(1, 3) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 29>}
    %tile_6_3 = aie.tile(6, 3) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 29>}
    %tile_7_3 = aie.tile(7, 3) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 29>}
    %tile_0_4 = aie.tile(0, 4) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 30>}
    %tile_1_4 = aie.tile(1, 4) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 30>}
    %tile_6_4 = aie.tile(6, 4) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 30>}
    %tile_7_4 = aie.tile(7, 4) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 30>}
    %tile_0_5 = aie.tile(0, 5) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 31>}
    %tile_1_5 = aie.tile(1, 5) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 31>}
    %tile_6_5 = aie.tile(6, 5) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 31>}
    %tile_7_5 = aie.tile(7, 5) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 31>}
    %mem_tile_2_1 = aie.tile(2, 1) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 26>}
    %mem_tile_3_1 = aie.tile(3, 1) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 26>}
    %mem_tile_0_1 = aie.tile(0, 1) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 26>}
    %mem_tile_1_1 = aie.tile(1, 1) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 26>}
    %mem_tile_6_1 = aie.tile(6, 1) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 26>}
    %mem_tile_7_1 = aie.tile(7, 1) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 26>}
    %mem_tile_4_1 = aie.tile(4, 1) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 26>}
    %mem_tile_5_1 = aie.tile(5, 1) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 26>}
    %tile_2_4 = aie.tile(2, 4) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 30>}
    %tile_3_4 = aie.tile(3, 4) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 30>}
    %tile_4_4 = aie.tile(4, 4) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 30>}
    %tile_5_4 = aie.tile(5, 4) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 30>}
    %shim_noc_tile_0_0 = aie.tile(0, 0) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 15>}
    %shim_noc_tile_1_0 = aie.tile(1, 0) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 15>}
    %shim_noc_tile_2_0 = aie.tile(2, 0) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 15>}
    %shim_noc_tile_3_0 = aie.tile(3, 0) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 15>}
    %shim_noc_tile_4_0 = aie.tile(4, 0) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 15>}
    %shim_noc_tile_5_0 = aie.tile(5, 0) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 15>}
    %shim_noc_tile_6_0 = aie.tile(6, 0) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 15>}
    %shim_noc_tile_7_0 = aie.tile(7, 0) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 15>}
    func.func private @_ha_noop() attributes {link_with = "layer_fused_bcast_kc256.o"}
    func.func private @fused_dequant_matvec_v2_bf16(i32, i32, memref<4608xi8>, memref<2320xbf16>, memref<322xbf16>) attributes {link_with = "fused_dequant_gemv_v2_signed_2048k_g32.o"}
    func.func private @rope_bundled(memref<322xbf16>, memref<2320xbf16>, memref<322xbf16>, i32) attributes {link_with = "rope_il.o"}
    func.func private @layer_fused_gate_up_bf16(i32, i32, memref<4608xi8>, memref<2320xbf16>, i32) attributes {link_with = "layer_fused_relay.o"}
    func.func private @generic_bcast_gemv_bf16_q(i32, memref<4608xi8>, memref<2320xbf16>, memref<128xi8>, i32, memref<322xbf16>) attributes {link_with = "layer_fused_unified_bcast.o"}
    func.func private @generic_bcast_gemv_bf16_o(i32, memref<4608xi8>, memref<2320xbf16>, memref<128xi8>, i32, memref<2048xbf16>) attributes {link_with = "layer_fused_unified_bcast.o"}
    func.func private @generic_bcast_gemv_bf16_g(i32, memref<4608xi8>, memref<2320xbf16>, memref<128xi8>, i32, memref<1024xbf16>) attributes {link_with = "layer_fused_unified_bcast.o"}
    func.func private @generic_bcast_gemv_bf16_d(i32, memref<4608xi8>, memref<1024xbf16>, memref<128xi8>, i32, memref<2320xbf16>) attributes {link_with = "layer_fused_unified_bcast.o"}
    func.func private @layer_fused_silu_mul_explicit_bf16(memref<1024xbf16>, memref<1024xbf16>, memref<1024xbf16>, i32) attributes {link_with = "layer_fused_relay.o"}
    func.func private @layer_fused_down_v2_x4_bf16(i32, i32, i32, memref<4608xi8>, memref<2320xbf16>) attributes {link_with = "layer_fused_relay.o"}
    func.func private @attn_copy_bf16(memref<2320xbf16>, memref<2320xbf16>, i32) attributes {link_with = "attn_concat.o"}
    func.func private @oproj_matvec_v2_bf16(i32, i32, memref<4608xi8>, memref<2320xbf16>, memref<2048xbf16>) attributes {link_with = "fused_dequant_gemv_v2_oproj_signed_2048k_g32.o"}
    func.func private @layer_fused_add_bf16(memref<2048xbf16>, memref<2048xbf16>, memref<2320xbf16>, i32) attributes {link_with = "layer_fused_relay.o"}
    func.func private @layer_fused_rms_norm2_bf16(memref<2320xbf16>, memref<2048xbf16>, memref<2320xbf16>, i32) attributes {link_with = "layer_fused_relay.o"}
    func.func private @flowkv_score_init_bf16(i32) attributes {link_with = "flowkv_64d_h4_c256.o"}
    func.func private @flowkv_score_rope_q_bf16(memref<322xbf16>, i32, i32) attributes {link_with = "flowkv_64d_h4_c256.o"}
    func.func private @flowkv_score_chunk_bf16(memref<322xbf16>, memref<8192xbf16>, memref<520xbf16>, i32, i32, i32) attributes {link_with = "flowkv_64d_h4_c256.o"}
    func.func private @flowkv_value_init_bf16(i32, i32) attributes {link_with = "flowkv_64d_h4_c256.o"}
    func.func private @flowkv_value_accum_bf16(memref<520xbf16>, memref<8192xbf16>, i32, i32, i32) attributes {link_with = "flowkv_64d_h4_c256.o"}
    func.func private @flowkv_value_normalize_bf16(memref<256xbf16>, i32, i32) attributes {link_with = "flowkv_64d_h4_c256.o"}
    aie.flow(%shim_noc_tile_0_0, DMA : 1, %tile_5_4, DMA : 0)
    aie.packet_flow(0) {
      aie.packet_source<%tile_2_4, DMA : 0>
      aie.packet_dest<%tile_5_4, DMA : 1>
    }
    aie.packet_flow(1) {
      aie.packet_source<%tile_4_4, DMA : 0>
      aie.packet_dest<%tile_5_4, DMA : 1>
    }
    aie.flow(%shim_noc_tile_0_0, DMA : 0, %tile_2_2, DMA : 0)
    aie.flow(%tile_5_4, DMA : 0, %tile_2_2, DMA : 1)
    aie.flow(%tile_2_2, DMA : 0, %shim_noc_tile_0_0, DMA : 0)
    aie.flow(%tile_2_2, DMA : 1, %tile_0_2, DMA : 0)
    aie.flow(%tile_0_2, DMA : 0, %tile_0_4, DMA : 0)
    aie.flow(%tile_0_2, DMA : 1, %mem_tile_4_1, DMA : 0)
    aie.flow(%shim_noc_tile_1_0, DMA : 0, %tile_3_2, DMA : 0)
    aie.flow(%tile_5_4, DMA : 0, %tile_3_2, DMA : 1)
    aie.flow(%tile_3_2, DMA : 0, %shim_noc_tile_1_0, DMA : 0)
    aie.flow(%tile_3_2, DMA : 1, %tile_1_2, DMA : 0)
    aie.flow(%tile_1_2, DMA : 0, %tile_1_4, DMA : 0)
    aie.flow(%tile_1_2, DMA : 1, %mem_tile_4_1, DMA : 1)
    aie.flow(%shim_noc_tile_2_0, DMA : 0, %tile_4_2, DMA : 0)
    aie.flow(%tile_5_4, DMA : 0, %tile_4_2, DMA : 1)
    aie.flow(%tile_4_2, DMA : 0, %shim_noc_tile_2_0, DMA : 0)
    aie.flow(%tile_4_2, DMA : 1, %tile_6_2, DMA : 0)
    aie.flow(%tile_6_2, DMA : 0, %tile_6_4, DMA : 0)
    aie.flow(%tile_6_2, DMA : 1, %mem_tile_4_1, DMA : 2)
    aie.flow(%shim_noc_tile_3_0, DMA : 0, %tile_5_2, DMA : 0)
    aie.flow(%tile_5_4, DMA : 0, %tile_5_2, DMA : 1)
    aie.flow(%tile_5_2, DMA : 0, %shim_noc_tile_3_0, DMA : 0)
    aie.flow(%tile_5_2, DMA : 1, %tile_7_2, DMA : 0)
    aie.flow(%tile_7_2, DMA : 0, %tile_7_4, DMA : 0)
    aie.flow(%tile_7_2, DMA : 1, %mem_tile_4_1, DMA : 3)
    aie.flow(%shim_noc_tile_4_0, DMA : 0, %tile_2_3, DMA : 0)
    aie.flow(%tile_5_4, DMA : 0, %tile_2_3, DMA : 1)
    aie.flow(%tile_2_3, DMA : 0, %shim_noc_tile_4_0, DMA : 0)
    aie.flow(%tile_2_3, DMA : 1, %tile_0_3, DMA : 0)
    aie.flow(%tile_0_3, DMA : 0, %tile_0_5, DMA : 0)
    aie.flow(%tile_0_3, DMA : 1, %mem_tile_5_1, DMA : 0)
    aie.flow(%shim_noc_tile_5_0, DMA : 0, %tile_3_3, DMA : 0)
    aie.flow(%tile_5_4, DMA : 0, %tile_3_3, DMA : 1)
    aie.flow(%tile_3_3, DMA : 0, %shim_noc_tile_5_0, DMA : 0)
    aie.flow(%tile_3_3, DMA : 1, %tile_1_3, DMA : 0)
    aie.flow(%tile_1_3, DMA : 0, %tile_1_5, DMA : 0)
    aie.flow(%tile_1_3, DMA : 1, %mem_tile_5_1, DMA : 1)
    aie.flow(%shim_noc_tile_6_0, DMA : 0, %tile_4_3, DMA : 0)
    aie.flow(%tile_5_4, DMA : 0, %tile_4_3, DMA : 1)
    aie.flow(%tile_4_3, DMA : 0, %shim_noc_tile_6_0, DMA : 0)
    aie.flow(%tile_4_3, DMA : 1, %tile_6_3, DMA : 0)
    aie.flow(%tile_6_3, DMA : 0, %tile_6_5, DMA : 0)
    aie.flow(%tile_6_3, DMA : 1, %mem_tile_5_1, DMA : 2)
    aie.flow(%shim_noc_tile_7_0, DMA : 0, %tile_5_3, DMA : 0)
    aie.flow(%tile_5_4, DMA : 0, %tile_5_3, DMA : 1)
    aie.flow(%tile_5_3, DMA : 0, %shim_noc_tile_7_0, DMA : 0)
    aie.flow(%tile_5_3, DMA : 1, %tile_7_3, DMA : 0)
    aie.flow(%tile_7_3, DMA : 0, %tile_7_5, DMA : 0)
    aie.flow(%tile_7_3, DMA : 1, %mem_tile_5_1, DMA : 3)
    aie.flow(%shim_noc_tile_3_0, DMA : 1, %mem_tile_0_1, DMA : 0)
    aie.flow(%shim_noc_tile_4_0, DMA : 1, %mem_tile_1_1, DMA : 0)
    aie.flow(%shim_noc_tile_5_0, DMA : 1, %mem_tile_6_1, DMA : 0)
    aie.flow(%shim_noc_tile_6_0, DMA : 1, %mem_tile_7_1, DMA : 0)
    aie.flow(%mem_tile_0_1, DMA : 0, %tile_0_2, DMA : 1)
    aie.flow(%mem_tile_1_1, DMA : 0, %tile_0_3, DMA : 1)
    aie.flow(%mem_tile_6_1, DMA : 0, %tile_0_4, DMA : 1)
    aie.flow(%mem_tile_7_1, DMA : 0, %tile_0_5, DMA : 1)
    aie.flow(%mem_tile_0_1, DMA : 1, %tile_1_2, DMA : 1)
    aie.flow(%mem_tile_1_1, DMA : 1, %tile_1_3, DMA : 1)
    aie.flow(%mem_tile_6_1, DMA : 1, %tile_1_4, DMA : 1)
    aie.flow(%mem_tile_7_1, DMA : 1, %tile_1_5, DMA : 1)
    aie.flow(%mem_tile_0_1, DMA : 2, %tile_6_2, DMA : 1)
    aie.flow(%mem_tile_1_1, DMA : 2, %tile_6_3, DMA : 1)
    aie.flow(%mem_tile_6_1, DMA : 2, %tile_6_4, DMA : 1)
    aie.flow(%mem_tile_7_1, DMA : 2, %tile_6_5, DMA : 1)
    aie.flow(%mem_tile_0_1, DMA : 3, %tile_7_2, DMA : 1)
    aie.flow(%mem_tile_1_1, DMA : 3, %tile_7_3, DMA : 1)
    aie.flow(%mem_tile_6_1, DMA : 3, %tile_7_4, DMA : 1)
    aie.flow(%mem_tile_7_1, DMA : 3, %tile_7_5, DMA : 1)
    aie.flow(%tile_0_4, DMA : 0, %mem_tile_2_1, DMA : 0)
    aie.flow(%tile_0_5, DMA : 0, %mem_tile_3_1, DMA : 0)
    aie.flow(%tile_1_4, DMA : 0, %mem_tile_2_1, DMA : 1)
    aie.flow(%tile_1_5, DMA : 0, %mem_tile_3_1, DMA : 1)
    aie.flow(%tile_6_4, DMA : 0, %mem_tile_2_1, DMA : 2)
    aie.flow(%tile_6_5, DMA : 0, %mem_tile_3_1, DMA : 2)
    aie.flow(%tile_7_4, DMA : 0, %mem_tile_2_1, DMA : 3)
    aie.flow(%tile_7_5, DMA : 0, %mem_tile_3_1, DMA : 3)
    aie.flow(%mem_tile_2_1, DMA : 0, %tile_2_4, DMA : 0)
    aie.flow(%mem_tile_3_1, DMA : 0, %tile_2_4, DMA : 1)
    aie.flow(%mem_tile_4_1, DMA : 0, %tile_3_4, DMA : 0)
    aie.flow(%mem_tile_5_1, DMA : 0, %tile_3_4, DMA : 1)
    aie.flow(%tile_3_4, DMA : 0, %tile_4_4, DMA : 0)
    aie.flow(%shim_noc_tile_2_0, DMA : 1, %tile_4_4, DMA : 1)
    aie.flow(%tile_4_4, DMA : 1, %shim_noc_tile_4_0, DMA : 1)
    %c0_A0 = aie.buffer(%tile_2_2) {address = 5664 : i32, mem_bank = 0 : i32, sym_name = "c0_A0"} : memref<4608xi8> 
    %c0_A1 = aie.buffer(%tile_2_2) {address = 21024 : i32, mem_bank = 1 : i32, sym_name = "c0_A1"} : memref<4608xi8> 
    %c0_B0 = aie.buffer(%tile_2_2) {address = 1024 : i32, mem_bank = 0 : i32, sym_name = "c0_B0"} : memref<2320xbf16> 
    %c0_B1 = aie.buffer(%tile_2_2) {address = 16384 : i32, mem_bank = 1 : i32, sym_name = "c0_B1"} : memref<2320xbf16> 
    %c0_B2 = aie.buffer(%tile_2_2) {address = 32768 : i32, mem_bank = 2 : i32, sym_name = "c0_B2"} : memref<2320xbf16> 
    %c0_Q = aie.buffer(%tile_2_2) {address = 41504 : i32, mem_bank = 2 : i32, sym_name = "c0_Q"} : memref<322xbf16> 
    %c0_O = aie.buffer(%tile_2_2) {address = 37408 : i32, mem_bank = 2 : i32, sym_name = "c0_O"} : memref<2048xbf16> 
    %c0_P = aie.buffer(%tile_2_2) {address = 49152 : i32, mem_bank = 3 : i32, sym_name = "c0_P"} : memref<2320xbf16> 
    %c0_A0p = aie.lock(%tile_2_2, 0) {init = 1 : i32, sym_name = "c0_A0p"}
    %c0_A0c = aie.lock(%tile_2_2, 1) {init = 0 : i32, sym_name = "c0_A0c"}
    %c0_B0p = aie.lock(%tile_2_2, 2) {init = 1 : i32, sym_name = "c0_B0p"}
    %c0_B0c = aie.lock(%tile_2_2, 3) {init = 0 : i32, sym_name = "c0_B0c"}
    %c0_Qp = aie.lock(%tile_2_2, 4) {init = 1 : i32, sym_name = "c0_Qp"}
    %c0_Qc = aie.lock(%tile_2_2, 5) {init = 0 : i32, sym_name = "c0_Qc"}
    %c0_Op = aie.lock(%tile_2_2, 6) {init = 1 : i32, sym_name = "c0_Op"}
    %c0_Oc = aie.lock(%tile_2_2, 7) {init = 0 : i32, sym_name = "c0_Oc"}
    %c0_Pp = aie.lock(%tile_2_2, 8) {init = 1 : i32, sym_name = "c0_Pp"}
    %c0_Pc = aie.lock(%tile_2_2, 9) {init = 0 : i32, sym_name = "c0_Pc"}
    %c0_A1p = aie.lock(%tile_2_2, 10) {init = 1 : i32, sym_name = "c0_A1p"}
    %c0_A1c = aie.lock(%tile_2_2, 11) {init = 0 : i32, sym_name = "c0_A1c"}
    %c0_B1p = aie.lock(%tile_2_2, 12) {init = 1 : i32, sym_name = "c0_B1p"}
    %c0_B1c = aie.lock(%tile_2_2, 13) {init = 0 : i32, sym_name = "c0_B1c"}
    %c0_B2p = aie.lock(%tile_2_2, 14) {init = 1 : i32, sym_name = "c0_B2p"}
    %c0_B2c = aie.lock(%tile_2_2, 15) {init = 0 : i32, sym_name = "c0_B2c"}
    %c0_gate = aie.buffer(%tile_2_2) {address = 53792 : i32, mem_bank = 3 : i32, sym_name = "c0_gate"} : memref<1024xbf16> 
    %c0_up = aie.buffer(%tile_2_2) {address = 10272 : i32, mem_bank = 0 : i32, sym_name = "c0_up"} : memref<1024xbf16> 
    %c0_silu = aie.buffer(%tile_2_2) {address = 25632 : i32, mem_bank = 1 : i32, sym_name = "c0_silu"} : memref<1024xbf16> 
    %c0_uni_partial = aie.buffer(%tile_2_2) {address = 55840 : i32, mem_bank = 3 : i32, sym_name = "c0_uni_partial"} : memref<128xi8> 
    %core_2_2 = aie.core(%tile_2_2) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      %c64 = arith.constant 64 : index
      %c256 = arith.constant 256 : index
      %c4_i32 = arith.constant 4 : i32
      %c8_i32 = arith.constant 8 : i32
      %c256_i32 = arith.constant 256 : i32
      %c1024_i32 = arith.constant 1024 : i32
      cf.br ^bb1(%c0 : index)
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb17
      %1 = arith.cmpi slt, %0, %c9223372036854775807 : index
      cf.cond_br %1, ^bb2, ^bb18
    ^bb2:  // pred: ^bb1
      func.call @_ha_noop() : () -> ()
      aie.use_lock(%c0_B0c, AcquireGreaterEqual, 1)
      aie.use_lock(%c0_Qp, AcquireGreaterEqual, 1)
      cf.br ^bb3(%c0 : index)
    ^bb3(%2: index):  // 2 preds: ^bb2, ^bb4
      %3 = arith.cmpi slt, %2, %c64 : index
      cf.cond_br %3, ^bb4, ^bb5
    ^bb4:  // pred: ^bb3
      aie.use_lock(%c0_A0c, AcquireGreaterEqual, 1)
      %4 = arith.index_cast %2 : index to i32
      func.call @generic_bcast_gemv_bf16_q(%4, %c0_A0, %c0_B0, %c0_uni_partial, %c8_i32, %c0_Q) : (i32, memref<4608xi8>, memref<2320xbf16>, memref<128xi8>, i32, memref<322xbf16>) -> ()
      aie.use_lock(%c0_A0p, Release, 1)
      aie.use_lock(%c0_A1c, AcquireGreaterEqual, 1)
      %5 = arith.addi %2, %c1 : index
      %6 = arith.index_cast %5 : index to i32
      func.call @generic_bcast_gemv_bf16_q(%6, %c0_A1, %c0_B0, %c0_uni_partial, %c8_i32, %c0_Q) : (i32, memref<4608xi8>, memref<2320xbf16>, memref<128xi8>, i32, memref<322xbf16>) -> ()
      aie.use_lock(%c0_A1p, Release, 1)
      %7 = arith.addi %2, %c2 : index
      cf.br ^bb3(%7 : index)
    ^bb5:  // pred: ^bb3
      func.call @rope_bundled(%c0_Q, %c0_B0, %c0_Q, %c256_i32) : (memref<322xbf16>, memref<2320xbf16>, memref<322xbf16>, i32) -> ()
      aie.use_lock(%c0_Qc, Release, 1)
      aie.use_lock(%c0_B0p, Release, 1)
      aie.use_lock(%c0_B1c, AcquireGreaterEqual, 1)
      aie.use_lock(%c0_Op, AcquireGreaterEqual, 1)
      cf.br ^bb6(%c0 : index)
    ^bb6(%8: index):  // 2 preds: ^bb5, ^bb7
      %9 = arith.cmpi slt, %8, %c64 : index
      cf.cond_br %9, ^bb7, ^bb8
    ^bb7:  // pred: ^bb6
      aie.use_lock(%c0_A0c, AcquireGreaterEqual, 1)
      %10 = arith.index_cast %8 : index to i32
      func.call @generic_bcast_gemv_bf16_o(%10, %c0_A0, %c0_B1, %c0_uni_partial, %c8_i32, %c0_O) : (i32, memref<4608xi8>, memref<2320xbf16>, memref<128xi8>, i32, memref<2048xbf16>) -> ()
      aie.use_lock(%c0_A0p, Release, 1)
      aie.use_lock(%c0_A1c, AcquireGreaterEqual, 1)
      %11 = arith.addi %8, %c1 : index
      %12 = arith.index_cast %11 : index to i32
      func.call @generic_bcast_gemv_bf16_o(%12, %c0_A1, %c0_B1, %c0_uni_partial, %c8_i32, %c0_O) : (i32, memref<4608xi8>, memref<2320xbf16>, memref<128xi8>, i32, memref<2048xbf16>) -> ()
      aie.use_lock(%c0_A1p, Release, 1)
      %13 = arith.addi %8, %c2 : index
      cf.br ^bb6(%13 : index)
    ^bb8:  // pred: ^bb6
      aie.use_lock(%c0_B1p, Release, 1)
      aie.use_lock(%c0_Oc, Release, 1)
      aie.use_lock(%c0_B2c, AcquireGreaterEqual, 1)
      cf.br ^bb9(%c0 : index)
    ^bb9(%14: index):  // 2 preds: ^bb8, ^bb10
      %15 = arith.cmpi slt, %14, %c256 : index
      cf.cond_br %15, ^bb10, ^bb11
    ^bb10:  // pred: ^bb9
      aie.use_lock(%c0_A0c, AcquireGreaterEqual, 1)
      %16 = arith.index_cast %14 : index to i32
      func.call @generic_bcast_gemv_bf16_g(%16, %c0_A0, %c0_B2, %c0_uni_partial, %c8_i32, %c0_gate) : (i32, memref<4608xi8>, memref<2320xbf16>, memref<128xi8>, i32, memref<1024xbf16>) -> ()
      aie.use_lock(%c0_A0p, Release, 1)
      aie.use_lock(%c0_A1c, AcquireGreaterEqual, 1)
      %17 = arith.addi %14, %c1 : index
      %18 = arith.index_cast %17 : index to i32
      func.call @generic_bcast_gemv_bf16_g(%18, %c0_A1, %c0_B2, %c0_uni_partial, %c8_i32, %c0_gate) : (i32, memref<4608xi8>, memref<2320xbf16>, memref<128xi8>, i32, memref<1024xbf16>) -> ()
      aie.use_lock(%c0_A1p, Release, 1)
      %19 = arith.addi %14, %c2 : index
      cf.br ^bb9(%19 : index)
    ^bb11:  // pred: ^bb9
      cf.br ^bb12(%c0 : index)
    ^bb12(%20: index):  // 2 preds: ^bb11, ^bb13
      %21 = arith.cmpi slt, %20, %c256 : index
      cf.cond_br %21, ^bb13, ^bb14
    ^bb13:  // pred: ^bb12
      aie.use_lock(%c0_A0c, AcquireGreaterEqual, 1)
      %22 = arith.index_cast %20 : index to i32
      func.call @generic_bcast_gemv_bf16_g(%22, %c0_A0, %c0_B2, %c0_uni_partial, %c8_i32, %c0_up) : (i32, memref<4608xi8>, memref<2320xbf16>, memref<128xi8>, i32, memref<1024xbf16>) -> ()
      aie.use_lock(%c0_A0p, Release, 1)
      aie.use_lock(%c0_A1c, AcquireGreaterEqual, 1)
      %23 = arith.addi %20, %c1 : index
      %24 = arith.index_cast %23 : index to i32
      func.call @generic_bcast_gemv_bf16_g(%24, %c0_A1, %c0_B2, %c0_uni_partial, %c8_i32, %c0_up) : (i32, memref<4608xi8>, memref<2320xbf16>, memref<128xi8>, i32, memref<1024xbf16>) -> ()
      aie.use_lock(%c0_A1p, Release, 1)
      %25 = arith.addi %20, %c2 : index
      cf.br ^bb12(%25 : index)
    ^bb14:  // pred: ^bb12
      aie.use_lock(%c0_B2p, Release, 1)
      func.call @layer_fused_silu_mul_explicit_bf16(%c0_gate, %c0_up, %c0_silu, %c1024_i32) : (memref<1024xbf16>, memref<1024xbf16>, memref<1024xbf16>, i32) -> ()
      aie.use_lock(%c0_Pp, AcquireGreaterEqual, 1)
      cf.br ^bb15(%c0 : index)
    ^bb15(%26: index):  // 2 preds: ^bb14, ^bb16
      %27 = arith.cmpi slt, %26, %c256 : index
      cf.cond_br %27, ^bb16, ^bb17
    ^bb16:  // pred: ^bb15
      aie.use_lock(%c0_A0c, AcquireGreaterEqual, 1)
      %28 = arith.index_cast %26 : index to i32
      func.call @generic_bcast_gemv_bf16_d(%28, %c0_A0, %c0_silu, %c0_uni_partial, %c4_i32, %c0_P) : (i32, memref<4608xi8>, memref<1024xbf16>, memref<128xi8>, i32, memref<2320xbf16>) -> ()
      aie.use_lock(%c0_A0p, Release, 1)
      aie.use_lock(%c0_A1c, AcquireGreaterEqual, 1)
      %29 = arith.addi %26, %c1 : index
      %30 = arith.index_cast %29 : index to i32
      func.call @generic_bcast_gemv_bf16_d(%30, %c0_A1, %c0_silu, %c0_uni_partial, %c4_i32, %c0_P) : (i32, memref<4608xi8>, memref<1024xbf16>, memref<128xi8>, i32, memref<2320xbf16>) -> ()
      aie.use_lock(%c0_A1p, Release, 1)
      %31 = arith.addi %26, %c2 : index
      cf.br ^bb15(%31 : index)
    ^bb17:  // pred: ^bb15
      aie.use_lock(%c0_Pc, Release, 1)
      %32 = arith.addi %0, %c1 : index
      cf.br ^bb1(%32 : index)
    ^bb18:  // pred: ^bb1
      aie.end
    } {link_files = ["layer_fused_bcast_kc256.o", "layer_fused_unified_bcast.o", "rope_il.o", "layer_fused_relay.o"]}
    %mem_2_2 = aie.mem(%tile_2_2) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%c0_A0p, AcquireGreaterEqual, 1)
      aie.dma_bd(%c0_A0 : memref<4608xi8>, 0, 4608) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%c0_A0c, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%c0_A1p, AcquireGreaterEqual, 1)
      aie.dma_bd(%c0_A1 : memref<4608xi8>, 0, 4608) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%c0_A1c, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb7)
    ^bb4:  // 2 preds: ^bb3, ^bb6
      aie.use_lock(%c0_B0p, AcquireGreaterEqual, 1)
      aie.dma_bd(%c0_B0 : memref<2320xbf16>, 0, 2320) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%c0_B0c, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%c0_B1p, AcquireGreaterEqual, 1)
      aie.dma_bd(%c0_B1 : memref<2320xbf16>, 0, 2320) {bd_id = 3 : i32, next_bd_id = 4 : i32}
      aie.use_lock(%c0_B1c, Release, 1)
      aie.next_bd ^bb6
    ^bb6:  // pred: ^bb5
      aie.use_lock(%c0_B2p, AcquireGreaterEqual, 1)
      aie.dma_bd(%c0_B2 : memref<2320xbf16>, 0, 2320) {bd_id = 4 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%c0_B2c, Release, 1)
      aie.next_bd ^bb4
    ^bb7:  // pred: ^bb3
      %2 = aie.dma_start(MM2S, 0, ^bb8, ^bb9)
    ^bb8:  // 2 preds: ^bb7, ^bb8
      aie.use_lock(%c0_Pc, AcquireGreaterEqual, 1)
      aie.dma_bd(%c0_P : memref<2320xbf16>, 0, 2048) {bd_id = 5 : i32, next_bd_id = 5 : i32}
      aie.use_lock(%c0_Pp, Release, 1)
      aie.next_bd ^bb8
    ^bb9:  // pred: ^bb7
      %3 = aie.dma_start(MM2S, 1, ^bb10, ^bb12)
    ^bb10:  // 2 preds: ^bb9, ^bb11
      aie.use_lock(%c0_Qc, AcquireGreaterEqual, 1)
      aie.dma_bd(%c0_Q : memref<322xbf16>, 0, 322) {bd_id = 6 : i32, next_bd_id = 7 : i32}
      aie.use_lock(%c0_Qp, Release, 1)
      aie.next_bd ^bb11
    ^bb11:  // pred: ^bb10
      aie.use_lock(%c0_Oc, AcquireGreaterEqual, 1)
      aie.dma_bd(%c0_O : memref<2048xbf16>, 0, 256) {bd_id = 7 : i32, next_bd_id = 6 : i32}
      aie.use_lock(%c0_Op, Release, 1)
      aie.next_bd ^bb10
    ^bb12:  // pred: ^bb9
      aie.end
    }
    %c1_A0 = aie.buffer(%tile_3_2) {address = 5664 : i32, mem_bank = 0 : i32, sym_name = "c1_A0"} : memref<4608xi8> 
    %c1_A1 = aie.buffer(%tile_3_2) {address = 21024 : i32, mem_bank = 1 : i32, sym_name = "c1_A1"} : memref<4608xi8> 
    %c1_B0 = aie.buffer(%tile_3_2) {address = 1024 : i32, mem_bank = 0 : i32, sym_name = "c1_B0"} : memref<2320xbf16> 
    %c1_B1 = aie.buffer(%tile_3_2) {address = 16384 : i32, mem_bank = 1 : i32, sym_name = "c1_B1"} : memref<2320xbf16> 
    %c1_B2 = aie.buffer(%tile_3_2) {address = 32768 : i32, mem_bank = 2 : i32, sym_name = "c1_B2"} : memref<2320xbf16> 
    %c1_Q = aie.buffer(%tile_3_2) {address = 41504 : i32, mem_bank = 2 : i32, sym_name = "c1_Q"} : memref<322xbf16> 
    %c1_O = aie.buffer(%tile_3_2) {address = 37408 : i32, mem_bank = 2 : i32, sym_name = "c1_O"} : memref<2048xbf16> 
    %c1_P = aie.buffer(%tile_3_2) {address = 49152 : i32, mem_bank = 3 : i32, sym_name = "c1_P"} : memref<2320xbf16> 
    %c1_A0p = aie.lock(%tile_3_2, 0) {init = 1 : i32, sym_name = "c1_A0p"}
    %c1_A0c = aie.lock(%tile_3_2, 1) {init = 0 : i32, sym_name = "c1_A0c"}
    %c1_B0p = aie.lock(%tile_3_2, 2) {init = 1 : i32, sym_name = "c1_B0p"}
    %c1_B0c = aie.lock(%tile_3_2, 3) {init = 0 : i32, sym_name = "c1_B0c"}
    %c1_Qp = aie.lock(%tile_3_2, 4) {init = 1 : i32, sym_name = "c1_Qp"}
    %c1_Qc = aie.lock(%tile_3_2, 5) {init = 0 : i32, sym_name = "c1_Qc"}
    %c1_Op = aie.lock(%tile_3_2, 6) {init = 1 : i32, sym_name = "c1_Op"}
    %c1_Oc = aie.lock(%tile_3_2, 7) {init = 0 : i32, sym_name = "c1_Oc"}
    %c1_Pp = aie.lock(%tile_3_2, 8) {init = 1 : i32, sym_name = "c1_Pp"}
    %c1_Pc = aie.lock(%tile_3_2, 9) {init = 0 : i32, sym_name = "c1_Pc"}
    %c1_A1p = aie.lock(%tile_3_2, 10) {init = 1 : i32, sym_name = "c1_A1p"}
    %c1_A1c = aie.lock(%tile_3_2, 11) {init = 0 : i32, sym_name = "c1_A1c"}
    %c1_B1p = aie.lock(%tile_3_2, 12) {init = 1 : i32, sym_name = "c1_B1p"}
    %c1_B1c = aie.lock(%tile_3_2, 13) {init = 0 : i32, sym_name = "c1_B1c"}
    %c1_B2p = aie.lock(%tile_3_2, 14) {init = 1 : i32, sym_name = "c1_B2p"}
    %c1_B2c = aie.lock(%tile_3_2, 15) {init = 0 : i32, sym_name = "c1_B2c"}
    %c1_gate = aie.buffer(%tile_3_2) {address = 53792 : i32, mem_bank = 3 : i32, sym_name = "c1_gate"} : memref<1024xbf16> 
    %c1_up = aie.buffer(%tile_3_2) {address = 10272 : i32, mem_bank = 0 : i32, sym_name = "c1_up"} : memref<1024xbf16> 
    %c1_silu = aie.buffer(%tile_3_2) {address = 25632 : i32, mem_bank = 1 : i32, sym_name = "c1_silu"} : memref<1024xbf16> 
    %c1_uni_partial = aie.buffer(%tile_3_2) {address = 55840 : i32, mem_bank = 3 : i32, sym_name = "c1_uni_partial"} : memref<128xi8> 
    %core_3_2 = aie.core(%tile_3_2) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      %c64 = arith.constant 64 : index
      %c256 = arith.constant 256 : index
      %c4_i32 = arith.constant 4 : i32
      %c8_i32 = arith.constant 8 : i32
      %c256_i32 = arith.constant 256 : i32
      %c1024_i32 = arith.constant 1024 : i32
      cf.br ^bb1(%c0 : index)
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb17
      %1 = arith.cmpi slt, %0, %c9223372036854775807 : index
      cf.cond_br %1, ^bb2, ^bb18
    ^bb2:  // pred: ^bb1
      func.call @_ha_noop() : () -> ()
      aie.use_lock(%c1_B0c, AcquireGreaterEqual, 1)
      aie.use_lock(%c1_Qp, AcquireGreaterEqual, 1)
      cf.br ^bb3(%c0 : index)
    ^bb3(%2: index):  // 2 preds: ^bb2, ^bb4
      %3 = arith.cmpi slt, %2, %c64 : index
      cf.cond_br %3, ^bb4, ^bb5
    ^bb4:  // pred: ^bb3
      aie.use_lock(%c1_A0c, AcquireGreaterEqual, 1)
      %4 = arith.index_cast %2 : index to i32
      func.call @generic_bcast_gemv_bf16_q(%4, %c1_A0, %c1_B0, %c1_uni_partial, %c8_i32, %c1_Q) : (i32, memref<4608xi8>, memref<2320xbf16>, memref<128xi8>, i32, memref<322xbf16>) -> ()
      aie.use_lock(%c1_A0p, Release, 1)
      aie.use_lock(%c1_A1c, AcquireGreaterEqual, 1)
      %5 = arith.addi %2, %c1 : index
      %6 = arith.index_cast %5 : index to i32
      func.call @generic_bcast_gemv_bf16_q(%6, %c1_A1, %c1_B0, %c1_uni_partial, %c8_i32, %c1_Q) : (i32, memref<4608xi8>, memref<2320xbf16>, memref<128xi8>, i32, memref<322xbf16>) -> ()
      aie.use_lock(%c1_A1p, Release, 1)
      %7 = arith.addi %2, %c2 : index
      cf.br ^bb3(%7 : index)
    ^bb5:  // pred: ^bb3
      func.call @rope_bundled(%c1_Q, %c1_B0, %c1_Q, %c256_i32) : (memref<322xbf16>, memref<2320xbf16>, memref<322xbf16>, i32) -> ()
      aie.use_lock(%c1_Qc, Release, 1)
      aie.use_lock(%c1_B0p, Release, 1)
      aie.use_lock(%c1_B1c, AcquireGreaterEqual, 1)
      aie.use_lock(%c1_Op, AcquireGreaterEqual, 1)
      cf.br ^bb6(%c0 : index)
    ^bb6(%8: index):  // 2 preds: ^bb5, ^bb7
      %9 = arith.cmpi slt, %8, %c64 : index
      cf.cond_br %9, ^bb7, ^bb8
    ^bb7:  // pred: ^bb6
      aie.use_lock(%c1_A0c, AcquireGreaterEqual, 1)
      %10 = arith.index_cast %8 : index to i32
      func.call @generic_bcast_gemv_bf16_o(%10, %c1_A0, %c1_B1, %c1_uni_partial, %c8_i32, %c1_O) : (i32, memref<4608xi8>, memref<2320xbf16>, memref<128xi8>, i32, memref<2048xbf16>) -> ()
      aie.use_lock(%c1_A0p, Release, 1)
      aie.use_lock(%c1_A1c, AcquireGreaterEqual, 1)
      %11 = arith.addi %8, %c1 : index
      %12 = arith.index_cast %11 : index to i32
      func.call @generic_bcast_gemv_bf16_o(%12, %c1_A1, %c1_B1, %c1_uni_partial, %c8_i32, %c1_O) : (i32, memref<4608xi8>, memref<2320xbf16>, memref<128xi8>, i32, memref<2048xbf16>) -> ()
      aie.use_lock(%c1_A1p, Release, 1)
      %13 = arith.addi %8, %c2 : index
      cf.br ^bb6(%13 : index)
    ^bb8:  // pred: ^bb6
      aie.use_lock(%c1_B1p, Release, 1)
      aie.use_lock(%c1_Oc, Release, 1)
      aie.use_lock(%c1_B2c, AcquireGreaterEqual, 1)
      cf.br ^bb9(%c0 : index)
    ^bb9(%14: index):  // 2 preds: ^bb8, ^bb10
      %15 = arith.cmpi slt, %14, %c256 : index
      cf.cond_br %15, ^bb10, ^bb11
    ^bb10:  // pred: ^bb9
      aie.use_lock(%c1_A0c, AcquireGreaterEqual, 1)
      %16 = arith.index_cast %14 : index to i32
      func.call @generic_bcast_gemv_bf16_g(%16, %c1_A0, %c1_B2, %c1_uni_partial, %c8_i32, %c1_gate) : (i32, memref<4608xi8>, memref<2320xbf16>, memref<128xi8>, i32, memref<1024xbf16>) -> ()
      aie.use_lock(%c1_A0p, Release, 1)
      aie.use_lock(%c1_A1c, AcquireGreaterEqual, 1)
      %17 = arith.addi %14, %c1 : index
      %18 = arith.index_cast %17 : index to i32
      func.call @generic_bcast_gemv_bf16_g(%18, %c1_A1, %c1_B2, %c1_uni_partial, %c8_i32, %c1_gate) : (i32, memref<4608xi8>, memref<2320xbf16>, memref<128xi8>, i32, memref<1024xbf16>) -> ()
      aie.use_lock(%c1_A1p, Release, 1)
      %19 = arith.addi %14, %c2 : index
      cf.br ^bb9(%19 : index)
    ^bb11:  // pred: ^bb9
      cf.br ^bb12(%c0 : index)
    ^bb12(%20: index):  // 2 preds: ^bb11, ^bb13
      %21 = arith.cmpi slt, %20, %c256 : index
      cf.cond_br %21, ^bb13, ^bb14
    ^bb13:  // pred: ^bb12
      aie.use_lock(%c1_A0c, AcquireGreaterEqual, 1)
      %22 = arith.index_cast %20 : index to i32
      func.call @generic_bcast_gemv_bf16_g(%22, %c1_A0, %c1_B2, %c1_uni_partial, %c8_i32, %c1_up) : (i32, memref<4608xi8>, memref<2320xbf16>, memref<128xi8>, i32, memref<1024xbf16>) -> ()
      aie.use_lock(%c1_A0p, Release, 1)
      aie.use_lock(%c1_A1c, AcquireGreaterEqual, 1)
      %23 = arith.addi %20, %c1 : index
      %24 = arith.index_cast %23 : index to i32
      func.call @generic_bcast_gemv_bf16_g(%24, %c1_A1, %c1_B2, %c1_uni_partial, %c8_i32, %c1_up) : (i32, memref<4608xi8>, memref<2320xbf16>, memref<128xi8>, i32, memref<1024xbf16>) -> ()
      aie.use_lock(%c1_A1p, Release, 1)
      %25 = arith.addi %20, %c2 : index
      cf.br ^bb12(%25 : index)
    ^bb14:  // pred: ^bb12
      aie.use_lock(%c1_B2p, Release, 1)
      func.call @layer_fused_silu_mul_explicit_bf16(%c1_gate, %c1_up, %c1_silu, %c1024_i32) : (memref<1024xbf16>, memref<1024xbf16>, memref<1024xbf16>, i32) -> ()
      aie.use_lock(%c1_Pp, AcquireGreaterEqual, 1)
      cf.br ^bb15(%c0 : index)
    ^bb15(%26: index):  // 2 preds: ^bb14, ^bb16
      %27 = arith.cmpi slt, %26, %c256 : index
      cf.cond_br %27, ^bb16, ^bb17
    ^bb16:  // pred: ^bb15
      aie.use_lock(%c1_A0c, AcquireGreaterEqual, 1)
      %28 = arith.index_cast %26 : index to i32
      func.call @generic_bcast_gemv_bf16_d(%28, %c1_A0, %c1_silu, %c1_uni_partial, %c4_i32, %c1_P) : (i32, memref<4608xi8>, memref<1024xbf16>, memref<128xi8>, i32, memref<2320xbf16>) -> ()
      aie.use_lock(%c1_A0p, Release, 1)
      aie.use_lock(%c1_A1c, AcquireGreaterEqual, 1)
      %29 = arith.addi %26, %c1 : index
      %30 = arith.index_cast %29 : index to i32
      func.call @generic_bcast_gemv_bf16_d(%30, %c1_A1, %c1_silu, %c1_uni_partial, %c4_i32, %c1_P) : (i32, memref<4608xi8>, memref<1024xbf16>, memref<128xi8>, i32, memref<2320xbf16>) -> ()
      aie.use_lock(%c1_A1p, Release, 1)
      %31 = arith.addi %26, %c2 : index
      cf.br ^bb15(%31 : index)
    ^bb17:  // pred: ^bb15
      aie.use_lock(%c1_Pc, Release, 1)
      %32 = arith.addi %0, %c1 : index
      cf.br ^bb1(%32 : index)
    ^bb18:  // pred: ^bb1
      aie.end
    } {link_files = ["layer_fused_bcast_kc256.o", "layer_fused_unified_bcast.o", "rope_il.o", "layer_fused_relay.o"]}
    %mem_3_2 = aie.mem(%tile_3_2) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%c1_A0p, AcquireGreaterEqual, 1)
      aie.dma_bd(%c1_A0 : memref<4608xi8>, 0, 4608) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%c1_A0c, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%c1_A1p, AcquireGreaterEqual, 1)
      aie.dma_bd(%c1_A1 : memref<4608xi8>, 0, 4608) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%c1_A1c, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb7)
    ^bb4:  // 2 preds: ^bb3, ^bb6
      aie.use_lock(%c1_B0p, AcquireGreaterEqual, 1)
      aie.dma_bd(%c1_B0 : memref<2320xbf16>, 0, 2320) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%c1_B0c, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%c1_B1p, AcquireGreaterEqual, 1)
      aie.dma_bd(%c1_B1 : memref<2320xbf16>, 0, 2320) {bd_id = 3 : i32, next_bd_id = 4 : i32}
      aie.use_lock(%c1_B1c, Release, 1)
      aie.next_bd ^bb6
    ^bb6:  // pred: ^bb5
      aie.use_lock(%c1_B2p, AcquireGreaterEqual, 1)
      aie.dma_bd(%c1_B2 : memref<2320xbf16>, 0, 2320) {bd_id = 4 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%c1_B2c, Release, 1)
      aie.next_bd ^bb4
    ^bb7:  // pred: ^bb3
      %2 = aie.dma_start(MM2S, 0, ^bb8, ^bb9)
    ^bb8:  // 2 preds: ^bb7, ^bb8
      aie.use_lock(%c1_Pc, AcquireGreaterEqual, 1)
      aie.dma_bd(%c1_P : memref<2320xbf16>, 0, 2048) {bd_id = 5 : i32, next_bd_id = 5 : i32}
      aie.use_lock(%c1_Pp, Release, 1)
      aie.next_bd ^bb8
    ^bb9:  // pred: ^bb7
      %3 = aie.dma_start(MM2S, 1, ^bb10, ^bb12)
    ^bb10:  // 2 preds: ^bb9, ^bb11
      aie.use_lock(%c1_Qc, AcquireGreaterEqual, 1)
      aie.dma_bd(%c1_Q : memref<322xbf16>, 0, 322) {bd_id = 6 : i32, next_bd_id = 7 : i32}
      aie.use_lock(%c1_Qp, Release, 1)
      aie.next_bd ^bb11
    ^bb11:  // pred: ^bb10
      aie.use_lock(%c1_Oc, AcquireGreaterEqual, 1)
      aie.dma_bd(%c1_O : memref<2048xbf16>, 0, 256) {bd_id = 7 : i32, next_bd_id = 6 : i32}
      aie.use_lock(%c1_Op, Release, 1)
      aie.next_bd ^bb10
    ^bb12:  // pred: ^bb9
      aie.end
    }
    %c2_A0 = aie.buffer(%tile_4_2) {address = 5664 : i32, mem_bank = 0 : i32, sym_name = "c2_A0"} : memref<4608xi8> 
    %c2_A1 = aie.buffer(%tile_4_2) {address = 21024 : i32, mem_bank = 1 : i32, sym_name = "c2_A1"} : memref<4608xi8> 
    %c2_B0 = aie.buffer(%tile_4_2) {address = 1024 : i32, mem_bank = 0 : i32, sym_name = "c2_B0"} : memref<2320xbf16> 
    %c2_B1 = aie.buffer(%tile_4_2) {address = 16384 : i32, mem_bank = 1 : i32, sym_name = "c2_B1"} : memref<2320xbf16> 
    %c2_B2 = aie.buffer(%tile_4_2) {address = 32768 : i32, mem_bank = 2 : i32, sym_name = "c2_B2"} : memref<2320xbf16> 
    %c2_Q = aie.buffer(%tile_4_2) {address = 41504 : i32, mem_bank = 2 : i32, sym_name = "c2_Q"} : memref<322xbf16> 
    %c2_O = aie.buffer(%tile_4_2) {address = 37408 : i32, mem_bank = 2 : i32, sym_name = "c2_O"} : memref<2048xbf16> 
    %c2_P = aie.buffer(%tile_4_2) {address = 49152 : i32, mem_bank = 3 : i32, sym_name = "c2_P"} : memref<2320xbf16> 
    %c2_A0p = aie.lock(%tile_4_2, 0) {init = 1 : i32, sym_name = "c2_A0p"}
    %c2_A0c = aie.lock(%tile_4_2, 1) {init = 0 : i32, sym_name = "c2_A0c"}
    %c2_B0p = aie.lock(%tile_4_2, 2) {init = 1 : i32, sym_name = "c2_B0p"}
    %c2_B0c = aie.lock(%tile_4_2, 3) {init = 0 : i32, sym_name = "c2_B0c"}
    %c2_Qp = aie.lock(%tile_4_2, 4) {init = 1 : i32, sym_name = "c2_Qp"}
    %c2_Qc = aie.lock(%tile_4_2, 5) {init = 0 : i32, sym_name = "c2_Qc"}
    %c2_Op = aie.lock(%tile_4_2, 6) {init = 1 : i32, sym_name = "c2_Op"}
    %c2_Oc = aie.lock(%tile_4_2, 7) {init = 0 : i32, sym_name = "c2_Oc"}
    %c2_Pp = aie.lock(%tile_4_2, 8) {init = 1 : i32, sym_name = "c2_Pp"}
    %c2_Pc = aie.lock(%tile_4_2, 9) {init = 0 : i32, sym_name = "c2_Pc"}
    %c2_A1p = aie.lock(%tile_4_2, 10) {init = 1 : i32, sym_name = "c2_A1p"}
    %c2_A1c = aie.lock(%tile_4_2, 11) {init = 0 : i32, sym_name = "c2_A1c"}
    %c2_B1p = aie.lock(%tile_4_2, 12) {init = 1 : i32, sym_name = "c2_B1p"}
    %c2_B1c = aie.lock(%tile_4_2, 13) {init = 0 : i32, sym_name = "c2_B1c"}
    %c2_B2p = aie.lock(%tile_4_2, 14) {init = 1 : i32, sym_name = "c2_B2p"}
    %c2_B2c = aie.lock(%tile_4_2, 15) {init = 0 : i32, sym_name = "c2_B2c"}
    %c2_gate = aie.buffer(%tile_4_2) {address = 53792 : i32, mem_bank = 3 : i32, sym_name = "c2_gate"} : memref<1024xbf16> 
    %c2_up = aie.buffer(%tile_4_2) {address = 10272 : i32, mem_bank = 0 : i32, sym_name = "c2_up"} : memref<1024xbf16> 
    %c2_silu = aie.buffer(%tile_4_2) {address = 25632 : i32, mem_bank = 1 : i32, sym_name = "c2_silu"} : memref<1024xbf16> 
    %c2_uni_partial = aie.buffer(%tile_4_2) {address = 55840 : i32, mem_bank = 3 : i32, sym_name = "c2_uni_partial"} : memref<128xi8> 
    %core_4_2 = aie.core(%tile_4_2) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      %c64 = arith.constant 64 : index
      %c256 = arith.constant 256 : index
      %c4_i32 = arith.constant 4 : i32
      %c8_i32 = arith.constant 8 : i32
      %c256_i32 = arith.constant 256 : i32
      %c1024_i32 = arith.constant 1024 : i32
      cf.br ^bb1(%c0 : index)
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb17
      %1 = arith.cmpi slt, %0, %c9223372036854775807 : index
      cf.cond_br %1, ^bb2, ^bb18
    ^bb2:  // pred: ^bb1
      func.call @_ha_noop() : () -> ()
      aie.use_lock(%c2_B0c, AcquireGreaterEqual, 1)
      aie.use_lock(%c2_Qp, AcquireGreaterEqual, 1)
      cf.br ^bb3(%c0 : index)
    ^bb3(%2: index):  // 2 preds: ^bb2, ^bb4
      %3 = arith.cmpi slt, %2, %c64 : index
      cf.cond_br %3, ^bb4, ^bb5
    ^bb4:  // pred: ^bb3
      aie.use_lock(%c2_A0c, AcquireGreaterEqual, 1)
      %4 = arith.index_cast %2 : index to i32
      func.call @generic_bcast_gemv_bf16_q(%4, %c2_A0, %c2_B0, %c2_uni_partial, %c8_i32, %c2_Q) : (i32, memref<4608xi8>, memref<2320xbf16>, memref<128xi8>, i32, memref<322xbf16>) -> ()
      aie.use_lock(%c2_A0p, Release, 1)
      aie.use_lock(%c2_A1c, AcquireGreaterEqual, 1)
      %5 = arith.addi %2, %c1 : index
      %6 = arith.index_cast %5 : index to i32
      func.call @generic_bcast_gemv_bf16_q(%6, %c2_A1, %c2_B0, %c2_uni_partial, %c8_i32, %c2_Q) : (i32, memref<4608xi8>, memref<2320xbf16>, memref<128xi8>, i32, memref<322xbf16>) -> ()
      aie.use_lock(%c2_A1p, Release, 1)
      %7 = arith.addi %2, %c2 : index
      cf.br ^bb3(%7 : index)
    ^bb5:  // pred: ^bb3
      func.call @rope_bundled(%c2_Q, %c2_B0, %c2_Q, %c256_i32) : (memref<322xbf16>, memref<2320xbf16>, memref<322xbf16>, i32) -> ()
      aie.use_lock(%c2_Qc, Release, 1)
      aie.use_lock(%c2_B0p, Release, 1)
      aie.use_lock(%c2_B1c, AcquireGreaterEqual, 1)
      aie.use_lock(%c2_Op, AcquireGreaterEqual, 1)
      cf.br ^bb6(%c0 : index)
    ^bb6(%8: index):  // 2 preds: ^bb5, ^bb7
      %9 = arith.cmpi slt, %8, %c64 : index
      cf.cond_br %9, ^bb7, ^bb8
    ^bb7:  // pred: ^bb6
      aie.use_lock(%c2_A0c, AcquireGreaterEqual, 1)
      %10 = arith.index_cast %8 : index to i32
      func.call @generic_bcast_gemv_bf16_o(%10, %c2_A0, %c2_B1, %c2_uni_partial, %c8_i32, %c2_O) : (i32, memref<4608xi8>, memref<2320xbf16>, memref<128xi8>, i32, memref<2048xbf16>) -> ()
      aie.use_lock(%c2_A0p, Release, 1)
      aie.use_lock(%c2_A1c, AcquireGreaterEqual, 1)
      %11 = arith.addi %8, %c1 : index
      %12 = arith.index_cast %11 : index to i32
      func.call @generic_bcast_gemv_bf16_o(%12, %c2_A1, %c2_B1, %c2_uni_partial, %c8_i32, %c2_O) : (i32, memref<4608xi8>, memref<2320xbf16>, memref<128xi8>, i32, memref<2048xbf16>) -> ()
      aie.use_lock(%c2_A1p, Release, 1)
      %13 = arith.addi %8, %c2 : index
      cf.br ^bb6(%13 : index)
    ^bb8:  // pred: ^bb6
      aie.use_lock(%c2_B1p, Release, 1)
      aie.use_lock(%c2_Oc, Release, 1)
      aie.use_lock(%c2_B2c, AcquireGreaterEqual, 1)
      cf.br ^bb9(%c0 : index)
    ^bb9(%14: index):  // 2 preds: ^bb8, ^bb10
      %15 = arith.cmpi slt, %14, %c256 : index
      cf.cond_br %15, ^bb10, ^bb11
    ^bb10:  // pred: ^bb9
      aie.use_lock(%c2_A0c, AcquireGreaterEqual, 1)
      %16 = arith.index_cast %14 : index to i32
      func.call @generic_bcast_gemv_bf16_g(%16, %c2_A0, %c2_B2, %c2_uni_partial, %c8_i32, %c2_gate) : (i32, memref<4608xi8>, memref<2320xbf16>, memref<128xi8>, i32, memref<1024xbf16>) -> ()
      aie.use_lock(%c2_A0p, Release, 1)
      aie.use_lock(%c2_A1c, AcquireGreaterEqual, 1)
      %17 = arith.addi %14, %c1 : index
      %18 = arith.index_cast %17 : index to i32
      func.call @generic_bcast_gemv_bf16_g(%18, %c2_A1, %c2_B2, %c2_uni_partial, %c8_i32, %c2_gate) : (i32, memref<4608xi8>, memref<2320xbf16>, memref<128xi8>, i32, memref<1024xbf16>) -> ()
      aie.use_lock(%c2_A1p, Release, 1)
      %19 = arith.addi %14, %c2 : index
      cf.br ^bb9(%19 : index)
    ^bb11:  // pred: ^bb9
      cf.br ^bb12(%c0 : index)
    ^bb12(%20: index):  // 2 preds: ^bb11, ^bb13
      %21 = arith.cmpi slt, %20, %c256 : index
      cf.cond_br %21, ^bb13, ^bb14
    ^bb13:  // pred: ^bb12
      aie.use_lock(%c2_A0c, AcquireGreaterEqual, 1)
      %22 = arith.index_cast %20 : index to i32
      func.call @generic_bcast_gemv_bf16_g(%22, %c2_A0, %c2_B2, %c2_uni_partial, %c8_i32, %c2_up) : (i32, memref<4608xi8>, memref<2320xbf16>, memref<128xi8>, i32, memref<1024xbf16>) -> ()
      aie.use_lock(%c2_A0p, Release, 1)
      aie.use_lock(%c2_A1c, AcquireGreaterEqual, 1)
      %23 = arith.addi %20, %c1 : index
      %24 = arith.index_cast %23 : index to i32
      func.call @generic_bcast_gemv_bf16_g(%24, %c2_A1, %c2_B2, %c2_uni_partial, %c8_i32, %c2_up) : (i32, memref<4608xi8>, memref<2320xbf16>, memref<128xi8>, i32, memref<1024xbf16>) -> ()
      aie.use_lock(%c2_A1p, Release, 1)
      %25 = arith.addi %20, %c2 : index
      cf.br ^bb12(%25 : index)
    ^bb14:  // pred: ^bb12
      aie.use_lock(%c2_B2p, Release, 1)
      func.call @layer_fused_silu_mul_explicit_bf16(%c2_gate, %c2_up, %c2_silu, %c1024_i32) : (memref<1024xbf16>, memref<1024xbf16>, memref<1024xbf16>, i32) -> ()
      aie.use_lock(%c2_Pp, AcquireGreaterEqual, 1)
      cf.br ^bb15(%c0 : index)
    ^bb15(%26: index):  // 2 preds: ^bb14, ^bb16
      %27 = arith.cmpi slt, %26, %c256 : index
      cf.cond_br %27, ^bb16, ^bb17
    ^bb16:  // pred: ^bb15
      aie.use_lock(%c2_A0c, AcquireGreaterEqual, 1)
      %28 = arith.index_cast %26 : index to i32
      func.call @generic_bcast_gemv_bf16_d(%28, %c2_A0, %c2_silu, %c2_uni_partial, %c4_i32, %c2_P) : (i32, memref<4608xi8>, memref<1024xbf16>, memref<128xi8>, i32, memref<2320xbf16>) -> ()
      aie.use_lock(%c2_A0p, Release, 1)
      aie.use_lock(%c2_A1c, AcquireGreaterEqual, 1)
      %29 = arith.addi %26, %c1 : index
      %30 = arith.index_cast %29 : index to i32
      func.call @generic_bcast_gemv_bf16_d(%30, %c2_A1, %c2_silu, %c2_uni_partial, %c4_i32, %c2_P) : (i32, memref<4608xi8>, memref<1024xbf16>, memref<128xi8>, i32, memref<2320xbf16>) -> ()
      aie.use_lock(%c2_A1p, Release, 1)
      %31 = arith.addi %26, %c2 : index
      cf.br ^bb15(%31 : index)
    ^bb17:  // pred: ^bb15
      aie.use_lock(%c2_Pc, Release, 1)
      %32 = arith.addi %0, %c1 : index
      cf.br ^bb1(%32 : index)
    ^bb18:  // pred: ^bb1
      aie.end
    } {link_files = ["layer_fused_bcast_kc256.o", "layer_fused_unified_bcast.o", "rope_il.o", "layer_fused_relay.o"]}
    %mem_4_2 = aie.mem(%tile_4_2) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%c2_A0p, AcquireGreaterEqual, 1)
      aie.dma_bd(%c2_A0 : memref<4608xi8>, 0, 4608) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%c2_A0c, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%c2_A1p, AcquireGreaterEqual, 1)
      aie.dma_bd(%c2_A1 : memref<4608xi8>, 0, 4608) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%c2_A1c, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb7)
    ^bb4:  // 2 preds: ^bb3, ^bb6
      aie.use_lock(%c2_B0p, AcquireGreaterEqual, 1)
      aie.dma_bd(%c2_B0 : memref<2320xbf16>, 0, 2320) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%c2_B0c, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%c2_B1p, AcquireGreaterEqual, 1)
      aie.dma_bd(%c2_B1 : memref<2320xbf16>, 0, 2320) {bd_id = 3 : i32, next_bd_id = 4 : i32}
      aie.use_lock(%c2_B1c, Release, 1)
      aie.next_bd ^bb6
    ^bb6:  // pred: ^bb5
      aie.use_lock(%c2_B2p, AcquireGreaterEqual, 1)
      aie.dma_bd(%c2_B2 : memref<2320xbf16>, 0, 2320) {bd_id = 4 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%c2_B2c, Release, 1)
      aie.next_bd ^bb4
    ^bb7:  // pred: ^bb3
      %2 = aie.dma_start(MM2S, 0, ^bb8, ^bb9)
    ^bb8:  // 2 preds: ^bb7, ^bb8
      aie.use_lock(%c2_Pc, AcquireGreaterEqual, 1)
      aie.dma_bd(%c2_P : memref<2320xbf16>, 0, 2048) {bd_id = 5 : i32, next_bd_id = 5 : i32}
      aie.use_lock(%c2_Pp, Release, 1)
      aie.next_bd ^bb8
    ^bb9:  // pred: ^bb7
      %3 = aie.dma_start(MM2S, 1, ^bb10, ^bb12)
    ^bb10:  // 2 preds: ^bb9, ^bb11
      aie.use_lock(%c2_Qc, AcquireGreaterEqual, 1)
      aie.dma_bd(%c2_Q : memref<322xbf16>, 0, 322) {bd_id = 6 : i32, next_bd_id = 7 : i32}
      aie.use_lock(%c2_Qp, Release, 1)
      aie.next_bd ^bb11
    ^bb11:  // pred: ^bb10
      aie.use_lock(%c2_Oc, AcquireGreaterEqual, 1)
      aie.dma_bd(%c2_O : memref<2048xbf16>, 0, 256) {bd_id = 7 : i32, next_bd_id = 6 : i32}
      aie.use_lock(%c2_Op, Release, 1)
      aie.next_bd ^bb10
    ^bb12:  // pred: ^bb9
      aie.end
    }
    %c3_A0 = aie.buffer(%tile_5_2) {address = 5664 : i32, mem_bank = 0 : i32, sym_name = "c3_A0"} : memref<4608xi8> 
    %c3_A1 = aie.buffer(%tile_5_2) {address = 21024 : i32, mem_bank = 1 : i32, sym_name = "c3_A1"} : memref<4608xi8> 
    %c3_B0 = aie.buffer(%tile_5_2) {address = 1024 : i32, mem_bank = 0 : i32, sym_name = "c3_B0"} : memref<2320xbf16> 
    %c3_B1 = aie.buffer(%tile_5_2) {address = 16384 : i32, mem_bank = 1 : i32, sym_name = "c3_B1"} : memref<2320xbf16> 
    %c3_B2 = aie.buffer(%tile_5_2) {address = 32768 : i32, mem_bank = 2 : i32, sym_name = "c3_B2"} : memref<2320xbf16> 
    %c3_Q = aie.buffer(%tile_5_2) {address = 41504 : i32, mem_bank = 2 : i32, sym_name = "c3_Q"} : memref<322xbf16> 
    %c3_O = aie.buffer(%tile_5_2) {address = 37408 : i32, mem_bank = 2 : i32, sym_name = "c3_O"} : memref<2048xbf16> 
    %c3_P = aie.buffer(%tile_5_2) {address = 49152 : i32, mem_bank = 3 : i32, sym_name = "c3_P"} : memref<2320xbf16> 
    %c3_A0p = aie.lock(%tile_5_2, 0) {init = 1 : i32, sym_name = "c3_A0p"}
    %c3_A0c = aie.lock(%tile_5_2, 1) {init = 0 : i32, sym_name = "c3_A0c"}
    %c3_B0p = aie.lock(%tile_5_2, 2) {init = 1 : i32, sym_name = "c3_B0p"}
    %c3_B0c = aie.lock(%tile_5_2, 3) {init = 0 : i32, sym_name = "c3_B0c"}
    %c3_Qp = aie.lock(%tile_5_2, 4) {init = 1 : i32, sym_name = "c3_Qp"}
    %c3_Qc = aie.lock(%tile_5_2, 5) {init = 0 : i32, sym_name = "c3_Qc"}
    %c3_Op = aie.lock(%tile_5_2, 6) {init = 1 : i32, sym_name = "c3_Op"}
    %c3_Oc = aie.lock(%tile_5_2, 7) {init = 0 : i32, sym_name = "c3_Oc"}
    %c3_Pp = aie.lock(%tile_5_2, 8) {init = 1 : i32, sym_name = "c3_Pp"}
    %c3_Pc = aie.lock(%tile_5_2, 9) {init = 0 : i32, sym_name = "c3_Pc"}
    %c3_A1p = aie.lock(%tile_5_2, 10) {init = 1 : i32, sym_name = "c3_A1p"}
    %c3_A1c = aie.lock(%tile_5_2, 11) {init = 0 : i32, sym_name = "c3_A1c"}
    %c3_B1p = aie.lock(%tile_5_2, 12) {init = 1 : i32, sym_name = "c3_B1p"}
    %c3_B1c = aie.lock(%tile_5_2, 13) {init = 0 : i32, sym_name = "c3_B1c"}
    %c3_B2p = aie.lock(%tile_5_2, 14) {init = 1 : i32, sym_name = "c3_B2p"}
    %c3_B2c = aie.lock(%tile_5_2, 15) {init = 0 : i32, sym_name = "c3_B2c"}
    %c3_gate = aie.buffer(%tile_5_2) {address = 53792 : i32, mem_bank = 3 : i32, sym_name = "c3_gate"} : memref<1024xbf16> 
    %c3_up = aie.buffer(%tile_5_2) {address = 10272 : i32, mem_bank = 0 : i32, sym_name = "c3_up"} : memref<1024xbf16> 
    %c3_silu = aie.buffer(%tile_5_2) {address = 25632 : i32, mem_bank = 1 : i32, sym_name = "c3_silu"} : memref<1024xbf16> 
    %c3_uni_partial = aie.buffer(%tile_5_2) {address = 55840 : i32, mem_bank = 3 : i32, sym_name = "c3_uni_partial"} : memref<128xi8> 
    %core_5_2 = aie.core(%tile_5_2) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      %c64 = arith.constant 64 : index
      %c256 = arith.constant 256 : index
      %c4_i32 = arith.constant 4 : i32
      %c8_i32 = arith.constant 8 : i32
      %c256_i32 = arith.constant 256 : i32
      %c1024_i32 = arith.constant 1024 : i32
      cf.br ^bb1(%c0 : index)
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb17
      %1 = arith.cmpi slt, %0, %c9223372036854775807 : index
      cf.cond_br %1, ^bb2, ^bb18
    ^bb2:  // pred: ^bb1
      func.call @_ha_noop() : () -> ()
      aie.use_lock(%c3_B0c, AcquireGreaterEqual, 1)
      aie.use_lock(%c3_Qp, AcquireGreaterEqual, 1)
      cf.br ^bb3(%c0 : index)
    ^bb3(%2: index):  // 2 preds: ^bb2, ^bb4
      %3 = arith.cmpi slt, %2, %c64 : index
      cf.cond_br %3, ^bb4, ^bb5
    ^bb4:  // pred: ^bb3
      aie.use_lock(%c3_A0c, AcquireGreaterEqual, 1)
      %4 = arith.index_cast %2 : index to i32
      func.call @generic_bcast_gemv_bf16_q(%4, %c3_A0, %c3_B0, %c3_uni_partial, %c8_i32, %c3_Q) : (i32, memref<4608xi8>, memref<2320xbf16>, memref<128xi8>, i32, memref<322xbf16>) -> ()
      aie.use_lock(%c3_A0p, Release, 1)
      aie.use_lock(%c3_A1c, AcquireGreaterEqual, 1)
      %5 = arith.addi %2, %c1 : index
      %6 = arith.index_cast %5 : index to i32
      func.call @generic_bcast_gemv_bf16_q(%6, %c3_A1, %c3_B0, %c3_uni_partial, %c8_i32, %c3_Q) : (i32, memref<4608xi8>, memref<2320xbf16>, memref<128xi8>, i32, memref<322xbf16>) -> ()
      aie.use_lock(%c3_A1p, Release, 1)
      %7 = arith.addi %2, %c2 : index
      cf.br ^bb3(%7 : index)
    ^bb5:  // pred: ^bb3
      func.call @rope_bundled(%c3_Q, %c3_B0, %c3_Q, %c256_i32) : (memref<322xbf16>, memref<2320xbf16>, memref<322xbf16>, i32) -> ()
      aie.use_lock(%c3_Qc, Release, 1)
      aie.use_lock(%c3_B0p, Release, 1)
      aie.use_lock(%c3_B1c, AcquireGreaterEqual, 1)
      aie.use_lock(%c3_Op, AcquireGreaterEqual, 1)
      cf.br ^bb6(%c0 : index)
    ^bb6(%8: index):  // 2 preds: ^bb5, ^bb7
      %9 = arith.cmpi slt, %8, %c64 : index
      cf.cond_br %9, ^bb7, ^bb8
    ^bb7:  // pred: ^bb6
      aie.use_lock(%c3_A0c, AcquireGreaterEqual, 1)
      %10 = arith.index_cast %8 : index to i32
      func.call @generic_bcast_gemv_bf16_o(%10, %c3_A0, %c3_B1, %c3_uni_partial, %c8_i32, %c3_O) : (i32, memref<4608xi8>, memref<2320xbf16>, memref<128xi8>, i32, memref<2048xbf16>) -> ()
      aie.use_lock(%c3_A0p, Release, 1)
      aie.use_lock(%c3_A1c, AcquireGreaterEqual, 1)
      %11 = arith.addi %8, %c1 : index
      %12 = arith.index_cast %11 : index to i32
      func.call @generic_bcast_gemv_bf16_o(%12, %c3_A1, %c3_B1, %c3_uni_partial, %c8_i32, %c3_O) : (i32, memref<4608xi8>, memref<2320xbf16>, memref<128xi8>, i32, memref<2048xbf16>) -> ()
      aie.use_lock(%c3_A1p, Release, 1)
      %13 = arith.addi %8, %c2 : index
      cf.br ^bb6(%13 : index)
    ^bb8:  // pred: ^bb6
      aie.use_lock(%c3_B1p, Release, 1)
      aie.use_lock(%c3_Oc, Release, 1)
      aie.use_lock(%c3_B2c, AcquireGreaterEqual, 1)
      cf.br ^bb9(%c0 : index)
    ^bb9(%14: index):  // 2 preds: ^bb8, ^bb10
      %15 = arith.cmpi slt, %14, %c256 : index
      cf.cond_br %15, ^bb10, ^bb11
    ^bb10:  // pred: ^bb9
      aie.use_lock(%c3_A0c, AcquireGreaterEqual, 1)
      %16 = arith.index_cast %14 : index to i32
      func.call @generic_bcast_gemv_bf16_g(%16, %c3_A0, %c3_B2, %c3_uni_partial, %c8_i32, %c3_gate) : (i32, memref<4608xi8>, memref<2320xbf16>, memref<128xi8>, i32, memref<1024xbf16>) -> ()
      aie.use_lock(%c3_A0p, Release, 1)
      aie.use_lock(%c3_A1c, AcquireGreaterEqual, 1)
      %17 = arith.addi %14, %c1 : index
      %18 = arith.index_cast %17 : index to i32
      func.call @generic_bcast_gemv_bf16_g(%18, %c3_A1, %c3_B2, %c3_uni_partial, %c8_i32, %c3_gate) : (i32, memref<4608xi8>, memref<2320xbf16>, memref<128xi8>, i32, memref<1024xbf16>) -> ()
      aie.use_lock(%c3_A1p, Release, 1)
      %19 = arith.addi %14, %c2 : index
      cf.br ^bb9(%19 : index)
    ^bb11:  // pred: ^bb9
      cf.br ^bb12(%c0 : index)
    ^bb12(%20: index):  // 2 preds: ^bb11, ^bb13
      %21 = arith.cmpi slt, %20, %c256 : index
      cf.cond_br %21, ^bb13, ^bb14
    ^bb13:  // pred: ^bb12
      aie.use_lock(%c3_A0c, AcquireGreaterEqual, 1)
      %22 = arith.index_cast %20 : index to i32
      func.call @generic_bcast_gemv_bf16_g(%22, %c3_A0, %c3_B2, %c3_uni_partial, %c8_i32, %c3_up) : (i32, memref<4608xi8>, memref<2320xbf16>, memref<128xi8>, i32, memref<1024xbf16>) -> ()
      aie.use_lock(%c3_A0p, Release, 1)
      aie.use_lock(%c3_A1c, AcquireGreaterEqual, 1)
      %23 = arith.addi %20, %c1 : index
      %24 = arith.index_cast %23 : index to i32
      func.call @generic_bcast_gemv_bf16_g(%24, %c3_A1, %c3_B2, %c3_uni_partial, %c8_i32, %c3_up) : (i32, memref<4608xi8>, memref<2320xbf16>, memref<128xi8>, i32, memref<1024xbf16>) -> ()
      aie.use_lock(%c3_A1p, Release, 1)
      %25 = arith.addi %20, %c2 : index
      cf.br ^bb12(%25 : index)
    ^bb14:  // pred: ^bb12
      aie.use_lock(%c3_B2p, Release, 1)
      func.call @layer_fused_silu_mul_explicit_bf16(%c3_gate, %c3_up, %c3_silu, %c1024_i32) : (memref<1024xbf16>, memref<1024xbf16>, memref<1024xbf16>, i32) -> ()
      aie.use_lock(%c3_Pp, AcquireGreaterEqual, 1)
      cf.br ^bb15(%c0 : index)
    ^bb15(%26: index):  // 2 preds: ^bb14, ^bb16
      %27 = arith.cmpi slt, %26, %c256 : index
      cf.cond_br %27, ^bb16, ^bb17
    ^bb16:  // pred: ^bb15
      aie.use_lock(%c3_A0c, AcquireGreaterEqual, 1)
      %28 = arith.index_cast %26 : index to i32
      func.call @generic_bcast_gemv_bf16_d(%28, %c3_A0, %c3_silu, %c3_uni_partial, %c4_i32, %c3_P) : (i32, memref<4608xi8>, memref<1024xbf16>, memref<128xi8>, i32, memref<2320xbf16>) -> ()
      aie.use_lock(%c3_A0p, Release, 1)
      aie.use_lock(%c3_A1c, AcquireGreaterEqual, 1)
      %29 = arith.addi %26, %c1 : index
      %30 = arith.index_cast %29 : index to i32
      func.call @generic_bcast_gemv_bf16_d(%30, %c3_A1, %c3_silu, %c3_uni_partial, %c4_i32, %c3_P) : (i32, memref<4608xi8>, memref<1024xbf16>, memref<128xi8>, i32, memref<2320xbf16>) -> ()
      aie.use_lock(%c3_A1p, Release, 1)
      %31 = arith.addi %26, %c2 : index
      cf.br ^bb15(%31 : index)
    ^bb17:  // pred: ^bb15
      aie.use_lock(%c3_Pc, Release, 1)
      %32 = arith.addi %0, %c1 : index
      cf.br ^bb1(%32 : index)
    ^bb18:  // pred: ^bb1
      aie.end
    } {link_files = ["layer_fused_bcast_kc256.o", "layer_fused_unified_bcast.o", "rope_il.o", "layer_fused_relay.o"]}
    %mem_5_2 = aie.mem(%tile_5_2) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%c3_A0p, AcquireGreaterEqual, 1)
      aie.dma_bd(%c3_A0 : memref<4608xi8>, 0, 4608) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%c3_A0c, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%c3_A1p, AcquireGreaterEqual, 1)
      aie.dma_bd(%c3_A1 : memref<4608xi8>, 0, 4608) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%c3_A1c, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb7)
    ^bb4:  // 2 preds: ^bb3, ^bb6
      aie.use_lock(%c3_B0p, AcquireGreaterEqual, 1)
      aie.dma_bd(%c3_B0 : memref<2320xbf16>, 0, 2320) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%c3_B0c, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%c3_B1p, AcquireGreaterEqual, 1)
      aie.dma_bd(%c3_B1 : memref<2320xbf16>, 0, 2320) {bd_id = 3 : i32, next_bd_id = 4 : i32}
      aie.use_lock(%c3_B1c, Release, 1)
      aie.next_bd ^bb6
    ^bb6:  // pred: ^bb5
      aie.use_lock(%c3_B2p, AcquireGreaterEqual, 1)
      aie.dma_bd(%c3_B2 : memref<2320xbf16>, 0, 2320) {bd_id = 4 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%c3_B2c, Release, 1)
      aie.next_bd ^bb4
    ^bb7:  // pred: ^bb3
      %2 = aie.dma_start(MM2S, 0, ^bb8, ^bb9)
    ^bb8:  // 2 preds: ^bb7, ^bb8
      aie.use_lock(%c3_Pc, AcquireGreaterEqual, 1)
      aie.dma_bd(%c3_P : memref<2320xbf16>, 0, 2048) {bd_id = 5 : i32, next_bd_id = 5 : i32}
      aie.use_lock(%c3_Pp, Release, 1)
      aie.next_bd ^bb8
    ^bb9:  // pred: ^bb7
      %3 = aie.dma_start(MM2S, 1, ^bb10, ^bb12)
    ^bb10:  // 2 preds: ^bb9, ^bb11
      aie.use_lock(%c3_Qc, AcquireGreaterEqual, 1)
      aie.dma_bd(%c3_Q : memref<322xbf16>, 0, 322) {bd_id = 6 : i32, next_bd_id = 7 : i32}
      aie.use_lock(%c3_Qp, Release, 1)
      aie.next_bd ^bb11
    ^bb11:  // pred: ^bb10
      aie.use_lock(%c3_Oc, AcquireGreaterEqual, 1)
      aie.dma_bd(%c3_O : memref<2048xbf16>, 0, 256) {bd_id = 7 : i32, next_bd_id = 6 : i32}
      aie.use_lock(%c3_Op, Release, 1)
      aie.next_bd ^bb10
    ^bb12:  // pred: ^bb9
      aie.end
    }
    %c4_A0 = aie.buffer(%tile_2_3) {address = 5664 : i32, mem_bank = 0 : i32, sym_name = "c4_A0"} : memref<4608xi8> 
    %c4_A1 = aie.buffer(%tile_2_3) {address = 21024 : i32, mem_bank = 1 : i32, sym_name = "c4_A1"} : memref<4608xi8> 
    %c4_B0 = aie.buffer(%tile_2_3) {address = 1024 : i32, mem_bank = 0 : i32, sym_name = "c4_B0"} : memref<2320xbf16> 
    %c4_B1 = aie.buffer(%tile_2_3) {address = 16384 : i32, mem_bank = 1 : i32, sym_name = "c4_B1"} : memref<2320xbf16> 
    %c4_B2 = aie.buffer(%tile_2_3) {address = 32768 : i32, mem_bank = 2 : i32, sym_name = "c4_B2"} : memref<2320xbf16> 
    %c4_Q = aie.buffer(%tile_2_3) {address = 41504 : i32, mem_bank = 2 : i32, sym_name = "c4_Q"} : memref<322xbf16> 
    %c4_O = aie.buffer(%tile_2_3) {address = 37408 : i32, mem_bank = 2 : i32, sym_name = "c4_O"} : memref<2048xbf16> 
    %c4_P = aie.buffer(%tile_2_3) {address = 49152 : i32, mem_bank = 3 : i32, sym_name = "c4_P"} : memref<2320xbf16> 
    %c4_A0p = aie.lock(%tile_2_3, 0) {init = 1 : i32, sym_name = "c4_A0p"}
    %c4_A0c = aie.lock(%tile_2_3, 1) {init = 0 : i32, sym_name = "c4_A0c"}
    %c4_B0p = aie.lock(%tile_2_3, 2) {init = 1 : i32, sym_name = "c4_B0p"}
    %c4_B0c = aie.lock(%tile_2_3, 3) {init = 0 : i32, sym_name = "c4_B0c"}
    %c4_Qp = aie.lock(%tile_2_3, 4) {init = 1 : i32, sym_name = "c4_Qp"}
    %c4_Qc = aie.lock(%tile_2_3, 5) {init = 0 : i32, sym_name = "c4_Qc"}
    %c4_Op = aie.lock(%tile_2_3, 6) {init = 1 : i32, sym_name = "c4_Op"}
    %c4_Oc = aie.lock(%tile_2_3, 7) {init = 0 : i32, sym_name = "c4_Oc"}
    %c4_Pp = aie.lock(%tile_2_3, 8) {init = 1 : i32, sym_name = "c4_Pp"}
    %c4_Pc = aie.lock(%tile_2_3, 9) {init = 0 : i32, sym_name = "c4_Pc"}
    %c4_A1p = aie.lock(%tile_2_3, 10) {init = 1 : i32, sym_name = "c4_A1p"}
    %c4_A1c = aie.lock(%tile_2_3, 11) {init = 0 : i32, sym_name = "c4_A1c"}
    %c4_B1p = aie.lock(%tile_2_3, 12) {init = 1 : i32, sym_name = "c4_B1p"}
    %c4_B1c = aie.lock(%tile_2_3, 13) {init = 0 : i32, sym_name = "c4_B1c"}
    %c4_B2p = aie.lock(%tile_2_3, 14) {init = 1 : i32, sym_name = "c4_B2p"}
    %c4_B2c = aie.lock(%tile_2_3, 15) {init = 0 : i32, sym_name = "c4_B2c"}
    %c4_gate = aie.buffer(%tile_2_3) {address = 53792 : i32, mem_bank = 3 : i32, sym_name = "c4_gate"} : memref<1024xbf16> 
    %c4_up = aie.buffer(%tile_2_3) {address = 10272 : i32, mem_bank = 0 : i32, sym_name = "c4_up"} : memref<1024xbf16> 
    %c4_silu = aie.buffer(%tile_2_3) {address = 25632 : i32, mem_bank = 1 : i32, sym_name = "c4_silu"} : memref<1024xbf16> 
    %c4_uni_partial = aie.buffer(%tile_2_3) {address = 55840 : i32, mem_bank = 3 : i32, sym_name = "c4_uni_partial"} : memref<128xi8> 
    %core_2_3 = aie.core(%tile_2_3) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      %c64 = arith.constant 64 : index
      %c256 = arith.constant 256 : index
      %c4_i32 = arith.constant 4 : i32
      %c8_i32 = arith.constant 8 : i32
      %c256_i32 = arith.constant 256 : i32
      %c1024_i32 = arith.constant 1024 : i32
      cf.br ^bb1(%c0 : index)
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb17
      %1 = arith.cmpi slt, %0, %c9223372036854775807 : index
      cf.cond_br %1, ^bb2, ^bb18
    ^bb2:  // pred: ^bb1
      func.call @_ha_noop() : () -> ()
      aie.use_lock(%c4_B0c, AcquireGreaterEqual, 1)
      aie.use_lock(%c4_Qp, AcquireGreaterEqual, 1)
      cf.br ^bb3(%c0 : index)
    ^bb3(%2: index):  // 2 preds: ^bb2, ^bb4
      %3 = arith.cmpi slt, %2, %c64 : index
      cf.cond_br %3, ^bb4, ^bb5
    ^bb4:  // pred: ^bb3
      aie.use_lock(%c4_A0c, AcquireGreaterEqual, 1)
      %4 = arith.index_cast %2 : index to i32
      func.call @generic_bcast_gemv_bf16_q(%4, %c4_A0, %c4_B0, %c4_uni_partial, %c8_i32, %c4_Q) : (i32, memref<4608xi8>, memref<2320xbf16>, memref<128xi8>, i32, memref<322xbf16>) -> ()
      aie.use_lock(%c4_A0p, Release, 1)
      aie.use_lock(%c4_A1c, AcquireGreaterEqual, 1)
      %5 = arith.addi %2, %c1 : index
      %6 = arith.index_cast %5 : index to i32
      func.call @generic_bcast_gemv_bf16_q(%6, %c4_A1, %c4_B0, %c4_uni_partial, %c8_i32, %c4_Q) : (i32, memref<4608xi8>, memref<2320xbf16>, memref<128xi8>, i32, memref<322xbf16>) -> ()
      aie.use_lock(%c4_A1p, Release, 1)
      %7 = arith.addi %2, %c2 : index
      cf.br ^bb3(%7 : index)
    ^bb5:  // pred: ^bb3
      func.call @rope_bundled(%c4_Q, %c4_B0, %c4_Q, %c256_i32) : (memref<322xbf16>, memref<2320xbf16>, memref<322xbf16>, i32) -> ()
      aie.use_lock(%c4_Qc, Release, 1)
      aie.use_lock(%c4_B0p, Release, 1)
      aie.use_lock(%c4_B1c, AcquireGreaterEqual, 1)
      aie.use_lock(%c4_Op, AcquireGreaterEqual, 1)
      cf.br ^bb6(%c0 : index)
    ^bb6(%8: index):  // 2 preds: ^bb5, ^bb7
      %9 = arith.cmpi slt, %8, %c64 : index
      cf.cond_br %9, ^bb7, ^bb8
    ^bb7:  // pred: ^bb6
      aie.use_lock(%c4_A0c, AcquireGreaterEqual, 1)
      %10 = arith.index_cast %8 : index to i32
      func.call @generic_bcast_gemv_bf16_o(%10, %c4_A0, %c4_B1, %c4_uni_partial, %c8_i32, %c4_O) : (i32, memref<4608xi8>, memref<2320xbf16>, memref<128xi8>, i32, memref<2048xbf16>) -> ()
      aie.use_lock(%c4_A0p, Release, 1)
      aie.use_lock(%c4_A1c, AcquireGreaterEqual, 1)
      %11 = arith.addi %8, %c1 : index
      %12 = arith.index_cast %11 : index to i32
      func.call @generic_bcast_gemv_bf16_o(%12, %c4_A1, %c4_B1, %c4_uni_partial, %c8_i32, %c4_O) : (i32, memref<4608xi8>, memref<2320xbf16>, memref<128xi8>, i32, memref<2048xbf16>) -> ()
      aie.use_lock(%c4_A1p, Release, 1)
      %13 = arith.addi %8, %c2 : index
      cf.br ^bb6(%13 : index)
    ^bb8:  // pred: ^bb6
      aie.use_lock(%c4_B1p, Release, 1)
      aie.use_lock(%c4_Oc, Release, 1)
      aie.use_lock(%c4_B2c, AcquireGreaterEqual, 1)
      cf.br ^bb9(%c0 : index)
    ^bb9(%14: index):  // 2 preds: ^bb8, ^bb10
      %15 = arith.cmpi slt, %14, %c256 : index
      cf.cond_br %15, ^bb10, ^bb11
    ^bb10:  // pred: ^bb9
      aie.use_lock(%c4_A0c, AcquireGreaterEqual, 1)
      %16 = arith.index_cast %14 : index to i32
      func.call @generic_bcast_gemv_bf16_g(%16, %c4_A0, %c4_B2, %c4_uni_partial, %c8_i32, %c4_gate) : (i32, memref<4608xi8>, memref<2320xbf16>, memref<128xi8>, i32, memref<1024xbf16>) -> ()
      aie.use_lock(%c4_A0p, Release, 1)
      aie.use_lock(%c4_A1c, AcquireGreaterEqual, 1)
      %17 = arith.addi %14, %c1 : index
      %18 = arith.index_cast %17 : index to i32
      func.call @generic_bcast_gemv_bf16_g(%18, %c4_A1, %c4_B2, %c4_uni_partial, %c8_i32, %c4_gate) : (i32, memref<4608xi8>, memref<2320xbf16>, memref<128xi8>, i32, memref<1024xbf16>) -> ()
      aie.use_lock(%c4_A1p, Release, 1)
      %19 = arith.addi %14, %c2 : index
      cf.br ^bb9(%19 : index)
    ^bb11:  // pred: ^bb9
      cf.br ^bb12(%c0 : index)
    ^bb12(%20: index):  // 2 preds: ^bb11, ^bb13
      %21 = arith.cmpi slt, %20, %c256 : index
      cf.cond_br %21, ^bb13, ^bb14
    ^bb13:  // pred: ^bb12
      aie.use_lock(%c4_A0c, AcquireGreaterEqual, 1)
      %22 = arith.index_cast %20 : index to i32
      func.call @generic_bcast_gemv_bf16_g(%22, %c4_A0, %c4_B2, %c4_uni_partial, %c8_i32, %c4_up) : (i32, memref<4608xi8>, memref<2320xbf16>, memref<128xi8>, i32, memref<1024xbf16>) -> ()
      aie.use_lock(%c4_A0p, Release, 1)
      aie.use_lock(%c4_A1c, AcquireGreaterEqual, 1)
      %23 = arith.addi %20, %c1 : index
      %24 = arith.index_cast %23 : index to i32
      func.call @generic_bcast_gemv_bf16_g(%24, %c4_A1, %c4_B2, %c4_uni_partial, %c8_i32, %c4_up) : (i32, memref<4608xi8>, memref<2320xbf16>, memref<128xi8>, i32, memref<1024xbf16>) -> ()
      aie.use_lock(%c4_A1p, Release, 1)
      %25 = arith.addi %20, %c2 : index
      cf.br ^bb12(%25 : index)
    ^bb14:  // pred: ^bb12
      aie.use_lock(%c4_B2p, Release, 1)
      func.call @layer_fused_silu_mul_explicit_bf16(%c4_gate, %c4_up, %c4_silu, %c1024_i32) : (memref<1024xbf16>, memref<1024xbf16>, memref<1024xbf16>, i32) -> ()
      aie.use_lock(%c4_Pp, AcquireGreaterEqual, 1)
      cf.br ^bb15(%c0 : index)
    ^bb15(%26: index):  // 2 preds: ^bb14, ^bb16
      %27 = arith.cmpi slt, %26, %c256 : index
      cf.cond_br %27, ^bb16, ^bb17
    ^bb16:  // pred: ^bb15
      aie.use_lock(%c4_A0c, AcquireGreaterEqual, 1)
      %28 = arith.index_cast %26 : index to i32
      func.call @generic_bcast_gemv_bf16_d(%28, %c4_A0, %c4_silu, %c4_uni_partial, %c4_i32, %c4_P) : (i32, memref<4608xi8>, memref<1024xbf16>, memref<128xi8>, i32, memref<2320xbf16>) -> ()
      aie.use_lock(%c4_A0p, Release, 1)
      aie.use_lock(%c4_A1c, AcquireGreaterEqual, 1)
      %29 = arith.addi %26, %c1 : index
      %30 = arith.index_cast %29 : index to i32
      func.call @generic_bcast_gemv_bf16_d(%30, %c4_A1, %c4_silu, %c4_uni_partial, %c4_i32, %c4_P) : (i32, memref<4608xi8>, memref<1024xbf16>, memref<128xi8>, i32, memref<2320xbf16>) -> ()
      aie.use_lock(%c4_A1p, Release, 1)
      %31 = arith.addi %26, %c2 : index
      cf.br ^bb15(%31 : index)
    ^bb17:  // pred: ^bb15
      aie.use_lock(%c4_Pc, Release, 1)
      %32 = arith.addi %0, %c1 : index
      cf.br ^bb1(%32 : index)
    ^bb18:  // pred: ^bb1
      aie.end
    } {link_files = ["layer_fused_bcast_kc256.o", "layer_fused_unified_bcast.o", "rope_il.o", "layer_fused_relay.o"]}
    %mem_2_3 = aie.mem(%tile_2_3) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%c4_A0p, AcquireGreaterEqual, 1)
      aie.dma_bd(%c4_A0 : memref<4608xi8>, 0, 4608) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%c4_A0c, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%c4_A1p, AcquireGreaterEqual, 1)
      aie.dma_bd(%c4_A1 : memref<4608xi8>, 0, 4608) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%c4_A1c, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb7)
    ^bb4:  // 2 preds: ^bb3, ^bb6
      aie.use_lock(%c4_B0p, AcquireGreaterEqual, 1)
      aie.dma_bd(%c4_B0 : memref<2320xbf16>, 0, 2320) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%c4_B0c, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%c4_B1p, AcquireGreaterEqual, 1)
      aie.dma_bd(%c4_B1 : memref<2320xbf16>, 0, 2320) {bd_id = 3 : i32, next_bd_id = 4 : i32}
      aie.use_lock(%c4_B1c, Release, 1)
      aie.next_bd ^bb6
    ^bb6:  // pred: ^bb5
      aie.use_lock(%c4_B2p, AcquireGreaterEqual, 1)
      aie.dma_bd(%c4_B2 : memref<2320xbf16>, 0, 2320) {bd_id = 4 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%c4_B2c, Release, 1)
      aie.next_bd ^bb4
    ^bb7:  // pred: ^bb3
      %2 = aie.dma_start(MM2S, 0, ^bb8, ^bb9)
    ^bb8:  // 2 preds: ^bb7, ^bb8
      aie.use_lock(%c4_Pc, AcquireGreaterEqual, 1)
      aie.dma_bd(%c4_P : memref<2320xbf16>, 0, 2048) {bd_id = 5 : i32, next_bd_id = 5 : i32}
      aie.use_lock(%c4_Pp, Release, 1)
      aie.next_bd ^bb8
    ^bb9:  // pred: ^bb7
      %3 = aie.dma_start(MM2S, 1, ^bb10, ^bb12)
    ^bb10:  // 2 preds: ^bb9, ^bb11
      aie.use_lock(%c4_Qc, AcquireGreaterEqual, 1)
      aie.dma_bd(%c4_Q : memref<322xbf16>, 0, 322) {bd_id = 6 : i32, next_bd_id = 7 : i32}
      aie.use_lock(%c4_Qp, Release, 1)
      aie.next_bd ^bb11
    ^bb11:  // pred: ^bb10
      aie.use_lock(%c4_Oc, AcquireGreaterEqual, 1)
      aie.dma_bd(%c4_O : memref<2048xbf16>, 0, 256) {bd_id = 7 : i32, next_bd_id = 6 : i32}
      aie.use_lock(%c4_Op, Release, 1)
      aie.next_bd ^bb10
    ^bb12:  // pred: ^bb9
      aie.end
    }
    %c5_A0 = aie.buffer(%tile_3_3) {address = 5664 : i32, mem_bank = 0 : i32, sym_name = "c5_A0"} : memref<4608xi8> 
    %c5_A1 = aie.buffer(%tile_3_3) {address = 21024 : i32, mem_bank = 1 : i32, sym_name = "c5_A1"} : memref<4608xi8> 
    %c5_B0 = aie.buffer(%tile_3_3) {address = 1024 : i32, mem_bank = 0 : i32, sym_name = "c5_B0"} : memref<2320xbf16> 
    %c5_B1 = aie.buffer(%tile_3_3) {address = 16384 : i32, mem_bank = 1 : i32, sym_name = "c5_B1"} : memref<2320xbf16> 
    %c5_B2 = aie.buffer(%tile_3_3) {address = 32768 : i32, mem_bank = 2 : i32, sym_name = "c5_B2"} : memref<2320xbf16> 
    %c5_Q = aie.buffer(%tile_3_3) {address = 41504 : i32, mem_bank = 2 : i32, sym_name = "c5_Q"} : memref<322xbf16> 
    %c5_O = aie.buffer(%tile_3_3) {address = 37408 : i32, mem_bank = 2 : i32, sym_name = "c5_O"} : memref<2048xbf16> 
    %c5_P = aie.buffer(%tile_3_3) {address = 49152 : i32, mem_bank = 3 : i32, sym_name = "c5_P"} : memref<2320xbf16> 
    %c5_A0p = aie.lock(%tile_3_3, 0) {init = 1 : i32, sym_name = "c5_A0p"}
    %c5_A0c = aie.lock(%tile_3_3, 1) {init = 0 : i32, sym_name = "c5_A0c"}
    %c5_B0p = aie.lock(%tile_3_3, 2) {init = 1 : i32, sym_name = "c5_B0p"}
    %c5_B0c = aie.lock(%tile_3_3, 3) {init = 0 : i32, sym_name = "c5_B0c"}
    %c5_Qp = aie.lock(%tile_3_3, 4) {init = 1 : i32, sym_name = "c5_Qp"}
    %c5_Qc = aie.lock(%tile_3_3, 5) {init = 0 : i32, sym_name = "c5_Qc"}
    %c5_Op = aie.lock(%tile_3_3, 6) {init = 1 : i32, sym_name = "c5_Op"}
    %c5_Oc = aie.lock(%tile_3_3, 7) {init = 0 : i32, sym_name = "c5_Oc"}
    %c5_Pp = aie.lock(%tile_3_3, 8) {init = 1 : i32, sym_name = "c5_Pp"}
    %c5_Pc = aie.lock(%tile_3_3, 9) {init = 0 : i32, sym_name = "c5_Pc"}
    %c5_A1p = aie.lock(%tile_3_3, 10) {init = 1 : i32, sym_name = "c5_A1p"}
    %c5_A1c = aie.lock(%tile_3_3, 11) {init = 0 : i32, sym_name = "c5_A1c"}
    %c5_B1p = aie.lock(%tile_3_3, 12) {init = 1 : i32, sym_name = "c5_B1p"}
    %c5_B1c = aie.lock(%tile_3_3, 13) {init = 0 : i32, sym_name = "c5_B1c"}
    %c5_B2p = aie.lock(%tile_3_3, 14) {init = 1 : i32, sym_name = "c5_B2p"}
    %c5_B2c = aie.lock(%tile_3_3, 15) {init = 0 : i32, sym_name = "c5_B2c"}
    %c5_gate = aie.buffer(%tile_3_3) {address = 53792 : i32, mem_bank = 3 : i32, sym_name = "c5_gate"} : memref<1024xbf16> 
    %c5_up = aie.buffer(%tile_3_3) {address = 10272 : i32, mem_bank = 0 : i32, sym_name = "c5_up"} : memref<1024xbf16> 
    %c5_silu = aie.buffer(%tile_3_3) {address = 25632 : i32, mem_bank = 1 : i32, sym_name = "c5_silu"} : memref<1024xbf16> 
    %c5_uni_partial = aie.buffer(%tile_3_3) {address = 55840 : i32, mem_bank = 3 : i32, sym_name = "c5_uni_partial"} : memref<128xi8> 
    %core_3_3 = aie.core(%tile_3_3) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      %c64 = arith.constant 64 : index
      %c256 = arith.constant 256 : index
      %c4_i32 = arith.constant 4 : i32
      %c8_i32 = arith.constant 8 : i32
      %c256_i32 = arith.constant 256 : i32
      %c1024_i32 = arith.constant 1024 : i32
      cf.br ^bb1(%c0 : index)
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb17
      %1 = arith.cmpi slt, %0, %c9223372036854775807 : index
      cf.cond_br %1, ^bb2, ^bb18
    ^bb2:  // pred: ^bb1
      func.call @_ha_noop() : () -> ()
      aie.use_lock(%c5_B0c, AcquireGreaterEqual, 1)
      aie.use_lock(%c5_Qp, AcquireGreaterEqual, 1)
      cf.br ^bb3(%c0 : index)
    ^bb3(%2: index):  // 2 preds: ^bb2, ^bb4
      %3 = arith.cmpi slt, %2, %c64 : index
      cf.cond_br %3, ^bb4, ^bb5
    ^bb4:  // pred: ^bb3
      aie.use_lock(%c5_A0c, AcquireGreaterEqual, 1)
      %4 = arith.index_cast %2 : index to i32
      func.call @generic_bcast_gemv_bf16_q(%4, %c5_A0, %c5_B0, %c5_uni_partial, %c8_i32, %c5_Q) : (i32, memref<4608xi8>, memref<2320xbf16>, memref<128xi8>, i32, memref<322xbf16>) -> ()
      aie.use_lock(%c5_A0p, Release, 1)
      aie.use_lock(%c5_A1c, AcquireGreaterEqual, 1)
      %5 = arith.addi %2, %c1 : index
      %6 = arith.index_cast %5 : index to i32
      func.call @generic_bcast_gemv_bf16_q(%6, %c5_A1, %c5_B0, %c5_uni_partial, %c8_i32, %c5_Q) : (i32, memref<4608xi8>, memref<2320xbf16>, memref<128xi8>, i32, memref<322xbf16>) -> ()
      aie.use_lock(%c5_A1p, Release, 1)
      %7 = arith.addi %2, %c2 : index
      cf.br ^bb3(%7 : index)
    ^bb5:  // pred: ^bb3
      func.call @rope_bundled(%c5_Q, %c5_B0, %c5_Q, %c256_i32) : (memref<322xbf16>, memref<2320xbf16>, memref<322xbf16>, i32) -> ()
      aie.use_lock(%c5_Qc, Release, 1)
      aie.use_lock(%c5_B0p, Release, 1)
      aie.use_lock(%c5_B1c, AcquireGreaterEqual, 1)
      aie.use_lock(%c5_Op, AcquireGreaterEqual, 1)
      cf.br ^bb6(%c0 : index)
    ^bb6(%8: index):  // 2 preds: ^bb5, ^bb7
      %9 = arith.cmpi slt, %8, %c64 : index
      cf.cond_br %9, ^bb7, ^bb8
    ^bb7:  // pred: ^bb6
      aie.use_lock(%c5_A0c, AcquireGreaterEqual, 1)
      %10 = arith.index_cast %8 : index to i32
      func.call @generic_bcast_gemv_bf16_o(%10, %c5_A0, %c5_B1, %c5_uni_partial, %c8_i32, %c5_O) : (i32, memref<4608xi8>, memref<2320xbf16>, memref<128xi8>, i32, memref<2048xbf16>) -> ()
      aie.use_lock(%c5_A0p, Release, 1)
      aie.use_lock(%c5_A1c, AcquireGreaterEqual, 1)
      %11 = arith.addi %8, %c1 : index
      %12 = arith.index_cast %11 : index to i32
      func.call @generic_bcast_gemv_bf16_o(%12, %c5_A1, %c5_B1, %c5_uni_partial, %c8_i32, %c5_O) : (i32, memref<4608xi8>, memref<2320xbf16>, memref<128xi8>, i32, memref<2048xbf16>) -> ()
      aie.use_lock(%c5_A1p, Release, 1)
      %13 = arith.addi %8, %c2 : index
      cf.br ^bb6(%13 : index)
    ^bb8:  // pred: ^bb6
      aie.use_lock(%c5_B1p, Release, 1)
      aie.use_lock(%c5_Oc, Release, 1)
      aie.use_lock(%c5_B2c, AcquireGreaterEqual, 1)
      cf.br ^bb9(%c0 : index)
    ^bb9(%14: index):  // 2 preds: ^bb8, ^bb10
      %15 = arith.cmpi slt, %14, %c256 : index
      cf.cond_br %15, ^bb10, ^bb11
    ^bb10:  // pred: ^bb9
      aie.use_lock(%c5_A0c, AcquireGreaterEqual, 1)
      %16 = arith.index_cast %14 : index to i32
      func.call @generic_bcast_gemv_bf16_g(%16, %c5_A0, %c5_B2, %c5_uni_partial, %c8_i32, %c5_gate) : (i32, memref<4608xi8>, memref<2320xbf16>, memref<128xi8>, i32, memref<1024xbf16>) -> ()
      aie.use_lock(%c5_A0p, Release, 1)
      aie.use_lock(%c5_A1c, AcquireGreaterEqual, 1)
      %17 = arith.addi %14, %c1 : index
      %18 = arith.index_cast %17 : index to i32
      func.call @generic_bcast_gemv_bf16_g(%18, %c5_A1, %c5_B2, %c5_uni_partial, %c8_i32, %c5_gate) : (i32, memref<4608xi8>, memref<2320xbf16>, memref<128xi8>, i32, memref<1024xbf16>) -> ()
      aie.use_lock(%c5_A1p, Release, 1)
      %19 = arith.addi %14, %c2 : index
      cf.br ^bb9(%19 : index)
    ^bb11:  // pred: ^bb9
      cf.br ^bb12(%c0 : index)
    ^bb12(%20: index):  // 2 preds: ^bb11, ^bb13
      %21 = arith.cmpi slt, %20, %c256 : index
      cf.cond_br %21, ^bb13, ^bb14
    ^bb13:  // pred: ^bb12
      aie.use_lock(%c5_A0c, AcquireGreaterEqual, 1)
      %22 = arith.index_cast %20 : index to i32
      func.call @generic_bcast_gemv_bf16_g(%22, %c5_A0, %c5_B2, %c5_uni_partial, %c8_i32, %c5_up) : (i32, memref<4608xi8>, memref<2320xbf16>, memref<128xi8>, i32, memref<1024xbf16>) -> ()
      aie.use_lock(%c5_A0p, Release, 1)
      aie.use_lock(%c5_A1c, AcquireGreaterEqual, 1)
      %23 = arith.addi %20, %c1 : index
      %24 = arith.index_cast %23 : index to i32
      func.call @generic_bcast_gemv_bf16_g(%24, %c5_A1, %c5_B2, %c5_uni_partial, %c8_i32, %c5_up) : (i32, memref<4608xi8>, memref<2320xbf16>, memref<128xi8>, i32, memref<1024xbf16>) -> ()
      aie.use_lock(%c5_A1p, Release, 1)
      %25 = arith.addi %20, %c2 : index
      cf.br ^bb12(%25 : index)
    ^bb14:  // pred: ^bb12
      aie.use_lock(%c5_B2p, Release, 1)
      func.call @layer_fused_silu_mul_explicit_bf16(%c5_gate, %c5_up, %c5_silu, %c1024_i32) : (memref<1024xbf16>, memref<1024xbf16>, memref<1024xbf16>, i32) -> ()
      aie.use_lock(%c5_Pp, AcquireGreaterEqual, 1)
      cf.br ^bb15(%c0 : index)
    ^bb15(%26: index):  // 2 preds: ^bb14, ^bb16
      %27 = arith.cmpi slt, %26, %c256 : index
      cf.cond_br %27, ^bb16, ^bb17
    ^bb16:  // pred: ^bb15
      aie.use_lock(%c5_A0c, AcquireGreaterEqual, 1)
      %28 = arith.index_cast %26 : index to i32
      func.call @generic_bcast_gemv_bf16_d(%28, %c5_A0, %c5_silu, %c5_uni_partial, %c4_i32, %c5_P) : (i32, memref<4608xi8>, memref<1024xbf16>, memref<128xi8>, i32, memref<2320xbf16>) -> ()
      aie.use_lock(%c5_A0p, Release, 1)
      aie.use_lock(%c5_A1c, AcquireGreaterEqual, 1)
      %29 = arith.addi %26, %c1 : index
      %30 = arith.index_cast %29 : index to i32
      func.call @generic_bcast_gemv_bf16_d(%30, %c5_A1, %c5_silu, %c5_uni_partial, %c4_i32, %c5_P) : (i32, memref<4608xi8>, memref<1024xbf16>, memref<128xi8>, i32, memref<2320xbf16>) -> ()
      aie.use_lock(%c5_A1p, Release, 1)
      %31 = arith.addi %26, %c2 : index
      cf.br ^bb15(%31 : index)
    ^bb17:  // pred: ^bb15
      aie.use_lock(%c5_Pc, Release, 1)
      %32 = arith.addi %0, %c1 : index
      cf.br ^bb1(%32 : index)
    ^bb18:  // pred: ^bb1
      aie.end
    } {link_files = ["layer_fused_bcast_kc256.o", "layer_fused_unified_bcast.o", "rope_il.o", "layer_fused_relay.o"]}
    %mem_3_3 = aie.mem(%tile_3_3) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%c5_A0p, AcquireGreaterEqual, 1)
      aie.dma_bd(%c5_A0 : memref<4608xi8>, 0, 4608) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%c5_A0c, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%c5_A1p, AcquireGreaterEqual, 1)
      aie.dma_bd(%c5_A1 : memref<4608xi8>, 0, 4608) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%c5_A1c, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb7)
    ^bb4:  // 2 preds: ^bb3, ^bb6
      aie.use_lock(%c5_B0p, AcquireGreaterEqual, 1)
      aie.dma_bd(%c5_B0 : memref<2320xbf16>, 0, 2320) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%c5_B0c, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%c5_B1p, AcquireGreaterEqual, 1)
      aie.dma_bd(%c5_B1 : memref<2320xbf16>, 0, 2320) {bd_id = 3 : i32, next_bd_id = 4 : i32}
      aie.use_lock(%c5_B1c, Release, 1)
      aie.next_bd ^bb6
    ^bb6:  // pred: ^bb5
      aie.use_lock(%c5_B2p, AcquireGreaterEqual, 1)
      aie.dma_bd(%c5_B2 : memref<2320xbf16>, 0, 2320) {bd_id = 4 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%c5_B2c, Release, 1)
      aie.next_bd ^bb4
    ^bb7:  // pred: ^bb3
      %2 = aie.dma_start(MM2S, 0, ^bb8, ^bb9)
    ^bb8:  // 2 preds: ^bb7, ^bb8
      aie.use_lock(%c5_Pc, AcquireGreaterEqual, 1)
      aie.dma_bd(%c5_P : memref<2320xbf16>, 0, 2048) {bd_id = 5 : i32, next_bd_id = 5 : i32}
      aie.use_lock(%c5_Pp, Release, 1)
      aie.next_bd ^bb8
    ^bb9:  // pred: ^bb7
      %3 = aie.dma_start(MM2S, 1, ^bb10, ^bb12)
    ^bb10:  // 2 preds: ^bb9, ^bb11
      aie.use_lock(%c5_Qc, AcquireGreaterEqual, 1)
      aie.dma_bd(%c5_Q : memref<322xbf16>, 0, 322) {bd_id = 6 : i32, next_bd_id = 7 : i32}
      aie.use_lock(%c5_Qp, Release, 1)
      aie.next_bd ^bb11
    ^bb11:  // pred: ^bb10
      aie.use_lock(%c5_Oc, AcquireGreaterEqual, 1)
      aie.dma_bd(%c5_O : memref<2048xbf16>, 0, 256) {bd_id = 7 : i32, next_bd_id = 6 : i32}
      aie.use_lock(%c5_Op, Release, 1)
      aie.next_bd ^bb10
    ^bb12:  // pred: ^bb9
      aie.end
    }
    %c6_A0 = aie.buffer(%tile_4_3) {address = 5664 : i32, mem_bank = 0 : i32, sym_name = "c6_A0"} : memref<4608xi8> 
    %c6_A1 = aie.buffer(%tile_4_3) {address = 21024 : i32, mem_bank = 1 : i32, sym_name = "c6_A1"} : memref<4608xi8> 
    %c6_B0 = aie.buffer(%tile_4_3) {address = 1024 : i32, mem_bank = 0 : i32, sym_name = "c6_B0"} : memref<2320xbf16> 
    %c6_B1 = aie.buffer(%tile_4_3) {address = 16384 : i32, mem_bank = 1 : i32, sym_name = "c6_B1"} : memref<2320xbf16> 
    %c6_B2 = aie.buffer(%tile_4_3) {address = 32768 : i32, mem_bank = 2 : i32, sym_name = "c6_B2"} : memref<2320xbf16> 
    %c6_Q = aie.buffer(%tile_4_3) {address = 41504 : i32, mem_bank = 2 : i32, sym_name = "c6_Q"} : memref<322xbf16> 
    %c6_O = aie.buffer(%tile_4_3) {address = 37408 : i32, mem_bank = 2 : i32, sym_name = "c6_O"} : memref<2048xbf16> 
    %c6_P = aie.buffer(%tile_4_3) {address = 49152 : i32, mem_bank = 3 : i32, sym_name = "c6_P"} : memref<2320xbf16> 
    %c6_A0p = aie.lock(%tile_4_3, 0) {init = 1 : i32, sym_name = "c6_A0p"}
    %c6_A0c = aie.lock(%tile_4_3, 1) {init = 0 : i32, sym_name = "c6_A0c"}
    %c6_B0p = aie.lock(%tile_4_3, 2) {init = 1 : i32, sym_name = "c6_B0p"}
    %c6_B0c = aie.lock(%tile_4_3, 3) {init = 0 : i32, sym_name = "c6_B0c"}
    %c6_Qp = aie.lock(%tile_4_3, 4) {init = 1 : i32, sym_name = "c6_Qp"}
    %c6_Qc = aie.lock(%tile_4_3, 5) {init = 0 : i32, sym_name = "c6_Qc"}
    %c6_Op = aie.lock(%tile_4_3, 6) {init = 1 : i32, sym_name = "c6_Op"}
    %c6_Oc = aie.lock(%tile_4_3, 7) {init = 0 : i32, sym_name = "c6_Oc"}
    %c6_Pp = aie.lock(%tile_4_3, 8) {init = 1 : i32, sym_name = "c6_Pp"}
    %c6_Pc = aie.lock(%tile_4_3, 9) {init = 0 : i32, sym_name = "c6_Pc"}
    %c6_A1p = aie.lock(%tile_4_3, 10) {init = 1 : i32, sym_name = "c6_A1p"}
    %c6_A1c = aie.lock(%tile_4_3, 11) {init = 0 : i32, sym_name = "c6_A1c"}
    %c6_B1p = aie.lock(%tile_4_3, 12) {init = 1 : i32, sym_name = "c6_B1p"}
    %c6_B1c = aie.lock(%tile_4_3, 13) {init = 0 : i32, sym_name = "c6_B1c"}
    %c6_B2p = aie.lock(%tile_4_3, 14) {init = 1 : i32, sym_name = "c6_B2p"}
    %c6_B2c = aie.lock(%tile_4_3, 15) {init = 0 : i32, sym_name = "c6_B2c"}
    %c6_gate = aie.buffer(%tile_4_3) {address = 53792 : i32, mem_bank = 3 : i32, sym_name = "c6_gate"} : memref<1024xbf16> 
    %c6_up = aie.buffer(%tile_4_3) {address = 10272 : i32, mem_bank = 0 : i32, sym_name = "c6_up"} : memref<1024xbf16> 
    %c6_silu = aie.buffer(%tile_4_3) {address = 25632 : i32, mem_bank = 1 : i32, sym_name = "c6_silu"} : memref<1024xbf16> 
    %c6_uni_partial = aie.buffer(%tile_4_3) {address = 55840 : i32, mem_bank = 3 : i32, sym_name = "c6_uni_partial"} : memref<128xi8> 
    %core_4_3 = aie.core(%tile_4_3) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      %c64 = arith.constant 64 : index
      %c256 = arith.constant 256 : index
      %c4_i32 = arith.constant 4 : i32
      %c8_i32 = arith.constant 8 : i32
      %c256_i32 = arith.constant 256 : i32
      %c1024_i32 = arith.constant 1024 : i32
      cf.br ^bb1(%c0 : index)
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb17
      %1 = arith.cmpi slt, %0, %c9223372036854775807 : index
      cf.cond_br %1, ^bb2, ^bb18
    ^bb2:  // pred: ^bb1
      func.call @_ha_noop() : () -> ()
      aie.use_lock(%c6_B0c, AcquireGreaterEqual, 1)
      aie.use_lock(%c6_Qp, AcquireGreaterEqual, 1)
      cf.br ^bb3(%c0 : index)
    ^bb3(%2: index):  // 2 preds: ^bb2, ^bb4
      %3 = arith.cmpi slt, %2, %c64 : index
      cf.cond_br %3, ^bb4, ^bb5
    ^bb4:  // pred: ^bb3
      aie.use_lock(%c6_A0c, AcquireGreaterEqual, 1)
      %4 = arith.index_cast %2 : index to i32
      func.call @generic_bcast_gemv_bf16_q(%4, %c6_A0, %c6_B0, %c6_uni_partial, %c8_i32, %c6_Q) : (i32, memref<4608xi8>, memref<2320xbf16>, memref<128xi8>, i32, memref<322xbf16>) -> ()
      aie.use_lock(%c6_A0p, Release, 1)
      aie.use_lock(%c6_A1c, AcquireGreaterEqual, 1)
      %5 = arith.addi %2, %c1 : index
      %6 = arith.index_cast %5 : index to i32
      func.call @generic_bcast_gemv_bf16_q(%6, %c6_A1, %c6_B0, %c6_uni_partial, %c8_i32, %c6_Q) : (i32, memref<4608xi8>, memref<2320xbf16>, memref<128xi8>, i32, memref<322xbf16>) -> ()
      aie.use_lock(%c6_A1p, Release, 1)
      %7 = arith.addi %2, %c2 : index
      cf.br ^bb3(%7 : index)
    ^bb5:  // pred: ^bb3
      func.call @rope_bundled(%c6_Q, %c6_B0, %c6_Q, %c256_i32) : (memref<322xbf16>, memref<2320xbf16>, memref<322xbf16>, i32) -> ()
      aie.use_lock(%c6_Qc, Release, 1)
      aie.use_lock(%c6_B0p, Release, 1)
      aie.use_lock(%c6_B1c, AcquireGreaterEqual, 1)
      aie.use_lock(%c6_Op, AcquireGreaterEqual, 1)
      cf.br ^bb6(%c0 : index)
    ^bb6(%8: index):  // 2 preds: ^bb5, ^bb7
      %9 = arith.cmpi slt, %8, %c64 : index
      cf.cond_br %9, ^bb7, ^bb8
    ^bb7:  // pred: ^bb6
      aie.use_lock(%c6_A0c, AcquireGreaterEqual, 1)
      %10 = arith.index_cast %8 : index to i32
      func.call @generic_bcast_gemv_bf16_o(%10, %c6_A0, %c6_B1, %c6_uni_partial, %c8_i32, %c6_O) : (i32, memref<4608xi8>, memref<2320xbf16>, memref<128xi8>, i32, memref<2048xbf16>) -> ()
      aie.use_lock(%c6_A0p, Release, 1)
      aie.use_lock(%c6_A1c, AcquireGreaterEqual, 1)
      %11 = arith.addi %8, %c1 : index
      %12 = arith.index_cast %11 : index to i32
      func.call @generic_bcast_gemv_bf16_o(%12, %c6_A1, %c6_B1, %c6_uni_partial, %c8_i32, %c6_O) : (i32, memref<4608xi8>, memref<2320xbf16>, memref<128xi8>, i32, memref<2048xbf16>) -> ()
      aie.use_lock(%c6_A1p, Release, 1)
      %13 = arith.addi %8, %c2 : index
      cf.br ^bb6(%13 : index)
    ^bb8:  // pred: ^bb6
      aie.use_lock(%c6_B1p, Release, 1)
      aie.use_lock(%c6_Oc, Release, 1)
      aie.use_lock(%c6_B2c, AcquireGreaterEqual, 1)
      cf.br ^bb9(%c0 : index)
    ^bb9(%14: index):  // 2 preds: ^bb8, ^bb10
      %15 = arith.cmpi slt, %14, %c256 : index
      cf.cond_br %15, ^bb10, ^bb11
    ^bb10:  // pred: ^bb9
      aie.use_lock(%c6_A0c, AcquireGreaterEqual, 1)
      %16 = arith.index_cast %14 : index to i32
      func.call @generic_bcast_gemv_bf16_g(%16, %c6_A0, %c6_B2, %c6_uni_partial, %c8_i32, %c6_gate) : (i32, memref<4608xi8>, memref<2320xbf16>, memref<128xi8>, i32, memref<1024xbf16>) -> ()
      aie.use_lock(%c6_A0p, Release, 1)
      aie.use_lock(%c6_A1c, AcquireGreaterEqual, 1)
      %17 = arith.addi %14, %c1 : index
      %18 = arith.index_cast %17 : index to i32
      func.call @generic_bcast_gemv_bf16_g(%18, %c6_A1, %c6_B2, %c6_uni_partial, %c8_i32, %c6_gate) : (i32, memref<4608xi8>, memref<2320xbf16>, memref<128xi8>, i32, memref<1024xbf16>) -> ()
      aie.use_lock(%c6_A1p, Release, 1)
      %19 = arith.addi %14, %c2 : index
      cf.br ^bb9(%19 : index)
    ^bb11:  // pred: ^bb9
      cf.br ^bb12(%c0 : index)
    ^bb12(%20: index):  // 2 preds: ^bb11, ^bb13
      %21 = arith.cmpi slt, %20, %c256 : index
      cf.cond_br %21, ^bb13, ^bb14
    ^bb13:  // pred: ^bb12
      aie.use_lock(%c6_A0c, AcquireGreaterEqual, 1)
      %22 = arith.index_cast %20 : index to i32
      func.call @generic_bcast_gemv_bf16_g(%22, %c6_A0, %c6_B2, %c6_uni_partial, %c8_i32, %c6_up) : (i32, memref<4608xi8>, memref<2320xbf16>, memref<128xi8>, i32, memref<1024xbf16>) -> ()
      aie.use_lock(%c6_A0p, Release, 1)
      aie.use_lock(%c6_A1c, AcquireGreaterEqual, 1)
      %23 = arith.addi %20, %c1 : index
      %24 = arith.index_cast %23 : index to i32
      func.call @generic_bcast_gemv_bf16_g(%24, %c6_A1, %c6_B2, %c6_uni_partial, %c8_i32, %c6_up) : (i32, memref<4608xi8>, memref<2320xbf16>, memref<128xi8>, i32, memref<1024xbf16>) -> ()
      aie.use_lock(%c6_A1p, Release, 1)
      %25 = arith.addi %20, %c2 : index
      cf.br ^bb12(%25 : index)
    ^bb14:  // pred: ^bb12
      aie.use_lock(%c6_B2p, Release, 1)
      func.call @layer_fused_silu_mul_explicit_bf16(%c6_gate, %c6_up, %c6_silu, %c1024_i32) : (memref<1024xbf16>, memref<1024xbf16>, memref<1024xbf16>, i32) -> ()
      aie.use_lock(%c6_Pp, AcquireGreaterEqual, 1)
      cf.br ^bb15(%c0 : index)
    ^bb15(%26: index):  // 2 preds: ^bb14, ^bb16
      %27 = arith.cmpi slt, %26, %c256 : index
      cf.cond_br %27, ^bb16, ^bb17
    ^bb16:  // pred: ^bb15
      aie.use_lock(%c6_A0c, AcquireGreaterEqual, 1)
      %28 = arith.index_cast %26 : index to i32
      func.call @generic_bcast_gemv_bf16_d(%28, %c6_A0, %c6_silu, %c6_uni_partial, %c4_i32, %c6_P) : (i32, memref<4608xi8>, memref<1024xbf16>, memref<128xi8>, i32, memref<2320xbf16>) -> ()
      aie.use_lock(%c6_A0p, Release, 1)
      aie.use_lock(%c6_A1c, AcquireGreaterEqual, 1)
      %29 = arith.addi %26, %c1 : index
      %30 = arith.index_cast %29 : index to i32
      func.call @generic_bcast_gemv_bf16_d(%30, %c6_A1, %c6_silu, %c6_uni_partial, %c4_i32, %c6_P) : (i32, memref<4608xi8>, memref<1024xbf16>, memref<128xi8>, i32, memref<2320xbf16>) -> ()
      aie.use_lock(%c6_A1p, Release, 1)
      %31 = arith.addi %26, %c2 : index
      cf.br ^bb15(%31 : index)
    ^bb17:  // pred: ^bb15
      aie.use_lock(%c6_Pc, Release, 1)
      %32 = arith.addi %0, %c1 : index
      cf.br ^bb1(%32 : index)
    ^bb18:  // pred: ^bb1
      aie.end
    } {link_files = ["layer_fused_bcast_kc256.o", "layer_fused_unified_bcast.o", "rope_il.o", "layer_fused_relay.o"]}
    %mem_4_3 = aie.mem(%tile_4_3) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%c6_A0p, AcquireGreaterEqual, 1)
      aie.dma_bd(%c6_A0 : memref<4608xi8>, 0, 4608) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%c6_A0c, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%c6_A1p, AcquireGreaterEqual, 1)
      aie.dma_bd(%c6_A1 : memref<4608xi8>, 0, 4608) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%c6_A1c, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb7)
    ^bb4:  // 2 preds: ^bb3, ^bb6
      aie.use_lock(%c6_B0p, AcquireGreaterEqual, 1)
      aie.dma_bd(%c6_B0 : memref<2320xbf16>, 0, 2320) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%c6_B0c, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%c6_B1p, AcquireGreaterEqual, 1)
      aie.dma_bd(%c6_B1 : memref<2320xbf16>, 0, 2320) {bd_id = 3 : i32, next_bd_id = 4 : i32}
      aie.use_lock(%c6_B1c, Release, 1)
      aie.next_bd ^bb6
    ^bb6:  // pred: ^bb5
      aie.use_lock(%c6_B2p, AcquireGreaterEqual, 1)
      aie.dma_bd(%c6_B2 : memref<2320xbf16>, 0, 2320) {bd_id = 4 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%c6_B2c, Release, 1)
      aie.next_bd ^bb4
    ^bb7:  // pred: ^bb3
      %2 = aie.dma_start(MM2S, 0, ^bb8, ^bb9)
    ^bb8:  // 2 preds: ^bb7, ^bb8
      aie.use_lock(%c6_Pc, AcquireGreaterEqual, 1)
      aie.dma_bd(%c6_P : memref<2320xbf16>, 0, 2048) {bd_id = 5 : i32, next_bd_id = 5 : i32}
      aie.use_lock(%c6_Pp, Release, 1)
      aie.next_bd ^bb8
    ^bb9:  // pred: ^bb7
      %3 = aie.dma_start(MM2S, 1, ^bb10, ^bb12)
    ^bb10:  // 2 preds: ^bb9, ^bb11
      aie.use_lock(%c6_Qc, AcquireGreaterEqual, 1)
      aie.dma_bd(%c6_Q : memref<322xbf16>, 0, 322) {bd_id = 6 : i32, next_bd_id = 7 : i32}
      aie.use_lock(%c6_Qp, Release, 1)
      aie.next_bd ^bb11
    ^bb11:  // pred: ^bb10
      aie.use_lock(%c6_Oc, AcquireGreaterEqual, 1)
      aie.dma_bd(%c6_O : memref<2048xbf16>, 0, 256) {bd_id = 7 : i32, next_bd_id = 6 : i32}
      aie.use_lock(%c6_Op, Release, 1)
      aie.next_bd ^bb10
    ^bb12:  // pred: ^bb9
      aie.end
    }
    %c7_A0 = aie.buffer(%tile_5_3) {address = 5664 : i32, mem_bank = 0 : i32, sym_name = "c7_A0"} : memref<4608xi8> 
    %c7_A1 = aie.buffer(%tile_5_3) {address = 21024 : i32, mem_bank = 1 : i32, sym_name = "c7_A1"} : memref<4608xi8> 
    %c7_B0 = aie.buffer(%tile_5_3) {address = 1024 : i32, mem_bank = 0 : i32, sym_name = "c7_B0"} : memref<2320xbf16> 
    %c7_B1 = aie.buffer(%tile_5_3) {address = 16384 : i32, mem_bank = 1 : i32, sym_name = "c7_B1"} : memref<2320xbf16> 
    %c7_B2 = aie.buffer(%tile_5_3) {address = 32768 : i32, mem_bank = 2 : i32, sym_name = "c7_B2"} : memref<2320xbf16> 
    %c7_Q = aie.buffer(%tile_5_3) {address = 41504 : i32, mem_bank = 2 : i32, sym_name = "c7_Q"} : memref<322xbf16> 
    %c7_O = aie.buffer(%tile_5_3) {address = 37408 : i32, mem_bank = 2 : i32, sym_name = "c7_O"} : memref<2048xbf16> 
    %c7_P = aie.buffer(%tile_5_3) {address = 49152 : i32, mem_bank = 3 : i32, sym_name = "c7_P"} : memref<2320xbf16> 
    %c7_A0p = aie.lock(%tile_5_3, 0) {init = 1 : i32, sym_name = "c7_A0p"}
    %c7_A0c = aie.lock(%tile_5_3, 1) {init = 0 : i32, sym_name = "c7_A0c"}
    %c7_B0p = aie.lock(%tile_5_3, 2) {init = 1 : i32, sym_name = "c7_B0p"}
    %c7_B0c = aie.lock(%tile_5_3, 3) {init = 0 : i32, sym_name = "c7_B0c"}
    %c7_Qp = aie.lock(%tile_5_3, 4) {init = 1 : i32, sym_name = "c7_Qp"}
    %c7_Qc = aie.lock(%tile_5_3, 5) {init = 0 : i32, sym_name = "c7_Qc"}
    %c7_Op = aie.lock(%tile_5_3, 6) {init = 1 : i32, sym_name = "c7_Op"}
    %c7_Oc = aie.lock(%tile_5_3, 7) {init = 0 : i32, sym_name = "c7_Oc"}
    %c7_Pp = aie.lock(%tile_5_3, 8) {init = 1 : i32, sym_name = "c7_Pp"}
    %c7_Pc = aie.lock(%tile_5_3, 9) {init = 0 : i32, sym_name = "c7_Pc"}
    %c7_A1p = aie.lock(%tile_5_3, 10) {init = 1 : i32, sym_name = "c7_A1p"}
    %c7_A1c = aie.lock(%tile_5_3, 11) {init = 0 : i32, sym_name = "c7_A1c"}
    %c7_B1p = aie.lock(%tile_5_3, 12) {init = 1 : i32, sym_name = "c7_B1p"}
    %c7_B1c = aie.lock(%tile_5_3, 13) {init = 0 : i32, sym_name = "c7_B1c"}
    %c7_B2p = aie.lock(%tile_5_3, 14) {init = 1 : i32, sym_name = "c7_B2p"}
    %c7_B2c = aie.lock(%tile_5_3, 15) {init = 0 : i32, sym_name = "c7_B2c"}
    %c7_gate = aie.buffer(%tile_5_3) {address = 53792 : i32, mem_bank = 3 : i32, sym_name = "c7_gate"} : memref<1024xbf16> 
    %c7_up = aie.buffer(%tile_5_3) {address = 10272 : i32, mem_bank = 0 : i32, sym_name = "c7_up"} : memref<1024xbf16> 
    %c7_silu = aie.buffer(%tile_5_3) {address = 25632 : i32, mem_bank = 1 : i32, sym_name = "c7_silu"} : memref<1024xbf16> 
    %c7_uni_partial = aie.buffer(%tile_5_3) {address = 55840 : i32, mem_bank = 3 : i32, sym_name = "c7_uni_partial"} : memref<128xi8> 
    %core_5_3 = aie.core(%tile_5_3) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      %c64 = arith.constant 64 : index
      %c256 = arith.constant 256 : index
      %c4_i32 = arith.constant 4 : i32
      %c8_i32 = arith.constant 8 : i32
      %c256_i32 = arith.constant 256 : i32
      %c1024_i32 = arith.constant 1024 : i32
      cf.br ^bb1(%c0 : index)
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb17
      %1 = arith.cmpi slt, %0, %c9223372036854775807 : index
      cf.cond_br %1, ^bb2, ^bb18
    ^bb2:  // pred: ^bb1
      func.call @_ha_noop() : () -> ()
      aie.use_lock(%c7_B0c, AcquireGreaterEqual, 1)
      aie.use_lock(%c7_Qp, AcquireGreaterEqual, 1)
      cf.br ^bb3(%c0 : index)
    ^bb3(%2: index):  // 2 preds: ^bb2, ^bb4
      %3 = arith.cmpi slt, %2, %c64 : index
      cf.cond_br %3, ^bb4, ^bb5
    ^bb4:  // pred: ^bb3
      aie.use_lock(%c7_A0c, AcquireGreaterEqual, 1)
      %4 = arith.index_cast %2 : index to i32
      func.call @generic_bcast_gemv_bf16_q(%4, %c7_A0, %c7_B0, %c7_uni_partial, %c8_i32, %c7_Q) : (i32, memref<4608xi8>, memref<2320xbf16>, memref<128xi8>, i32, memref<322xbf16>) -> ()
      aie.use_lock(%c7_A0p, Release, 1)
      aie.use_lock(%c7_A1c, AcquireGreaterEqual, 1)
      %5 = arith.addi %2, %c1 : index
      %6 = arith.index_cast %5 : index to i32
      func.call @generic_bcast_gemv_bf16_q(%6, %c7_A1, %c7_B0, %c7_uni_partial, %c8_i32, %c7_Q) : (i32, memref<4608xi8>, memref<2320xbf16>, memref<128xi8>, i32, memref<322xbf16>) -> ()
      aie.use_lock(%c7_A1p, Release, 1)
      %7 = arith.addi %2, %c2 : index
      cf.br ^bb3(%7 : index)
    ^bb5:  // pred: ^bb3
      func.call @rope_bundled(%c7_Q, %c7_B0, %c7_Q, %c256_i32) : (memref<322xbf16>, memref<2320xbf16>, memref<322xbf16>, i32) -> ()
      aie.use_lock(%c7_Qc, Release, 1)
      aie.use_lock(%c7_B0p, Release, 1)
      aie.use_lock(%c7_B1c, AcquireGreaterEqual, 1)
      aie.use_lock(%c7_Op, AcquireGreaterEqual, 1)
      cf.br ^bb6(%c0 : index)
    ^bb6(%8: index):  // 2 preds: ^bb5, ^bb7
      %9 = arith.cmpi slt, %8, %c64 : index
      cf.cond_br %9, ^bb7, ^bb8
    ^bb7:  // pred: ^bb6
      aie.use_lock(%c7_A0c, AcquireGreaterEqual, 1)
      %10 = arith.index_cast %8 : index to i32
      func.call @generic_bcast_gemv_bf16_o(%10, %c7_A0, %c7_B1, %c7_uni_partial, %c8_i32, %c7_O) : (i32, memref<4608xi8>, memref<2320xbf16>, memref<128xi8>, i32, memref<2048xbf16>) -> ()
      aie.use_lock(%c7_A0p, Release, 1)
      aie.use_lock(%c7_A1c, AcquireGreaterEqual, 1)
      %11 = arith.addi %8, %c1 : index
      %12 = arith.index_cast %11 : index to i32
      func.call @generic_bcast_gemv_bf16_o(%12, %c7_A1, %c7_B1, %c7_uni_partial, %c8_i32, %c7_O) : (i32, memref<4608xi8>, memref<2320xbf16>, memref<128xi8>, i32, memref<2048xbf16>) -> ()
      aie.use_lock(%c7_A1p, Release, 1)
      %13 = arith.addi %8, %c2 : index
      cf.br ^bb6(%13 : index)
    ^bb8:  // pred: ^bb6
      aie.use_lock(%c7_B1p, Release, 1)
      aie.use_lock(%c7_Oc, Release, 1)
      aie.use_lock(%c7_B2c, AcquireGreaterEqual, 1)
      cf.br ^bb9(%c0 : index)
    ^bb9(%14: index):  // 2 preds: ^bb8, ^bb10
      %15 = arith.cmpi slt, %14, %c256 : index
      cf.cond_br %15, ^bb10, ^bb11
    ^bb10:  // pred: ^bb9
      aie.use_lock(%c7_A0c, AcquireGreaterEqual, 1)
      %16 = arith.index_cast %14 : index to i32
      func.call @generic_bcast_gemv_bf16_g(%16, %c7_A0, %c7_B2, %c7_uni_partial, %c8_i32, %c7_gate) : (i32, memref<4608xi8>, memref<2320xbf16>, memref<128xi8>, i32, memref<1024xbf16>) -> ()
      aie.use_lock(%c7_A0p, Release, 1)
      aie.use_lock(%c7_A1c, AcquireGreaterEqual, 1)
      %17 = arith.addi %14, %c1 : index
      %18 = arith.index_cast %17 : index to i32
      func.call @generic_bcast_gemv_bf16_g(%18, %c7_A1, %c7_B2, %c7_uni_partial, %c8_i32, %c7_gate) : (i32, memref<4608xi8>, memref<2320xbf16>, memref<128xi8>, i32, memref<1024xbf16>) -> ()
      aie.use_lock(%c7_A1p, Release, 1)
      %19 = arith.addi %14, %c2 : index
      cf.br ^bb9(%19 : index)
    ^bb11:  // pred: ^bb9
      cf.br ^bb12(%c0 : index)
    ^bb12(%20: index):  // 2 preds: ^bb11, ^bb13
      %21 = arith.cmpi slt, %20, %c256 : index
      cf.cond_br %21, ^bb13, ^bb14
    ^bb13:  // pred: ^bb12
      aie.use_lock(%c7_A0c, AcquireGreaterEqual, 1)
      %22 = arith.index_cast %20 : index to i32
      func.call @generic_bcast_gemv_bf16_g(%22, %c7_A0, %c7_B2, %c7_uni_partial, %c8_i32, %c7_up) : (i32, memref<4608xi8>, memref<2320xbf16>, memref<128xi8>, i32, memref<1024xbf16>) -> ()
      aie.use_lock(%c7_A0p, Release, 1)
      aie.use_lock(%c7_A1c, AcquireGreaterEqual, 1)
      %23 = arith.addi %20, %c1 : index
      %24 = arith.index_cast %23 : index to i32
      func.call @generic_bcast_gemv_bf16_g(%24, %c7_A1, %c7_B2, %c7_uni_partial, %c8_i32, %c7_up) : (i32, memref<4608xi8>, memref<2320xbf16>, memref<128xi8>, i32, memref<1024xbf16>) -> ()
      aie.use_lock(%c7_A1p, Release, 1)
      %25 = arith.addi %20, %c2 : index
      cf.br ^bb12(%25 : index)
    ^bb14:  // pred: ^bb12
      aie.use_lock(%c7_B2p, Release, 1)
      func.call @layer_fused_silu_mul_explicit_bf16(%c7_gate, %c7_up, %c7_silu, %c1024_i32) : (memref<1024xbf16>, memref<1024xbf16>, memref<1024xbf16>, i32) -> ()
      aie.use_lock(%c7_Pp, AcquireGreaterEqual, 1)
      cf.br ^bb15(%c0 : index)
    ^bb15(%26: index):  // 2 preds: ^bb14, ^bb16
      %27 = arith.cmpi slt, %26, %c256 : index
      cf.cond_br %27, ^bb16, ^bb17
    ^bb16:  // pred: ^bb15
      aie.use_lock(%c7_A0c, AcquireGreaterEqual, 1)
      %28 = arith.index_cast %26 : index to i32
      func.call @generic_bcast_gemv_bf16_d(%28, %c7_A0, %c7_silu, %c7_uni_partial, %c4_i32, %c7_P) : (i32, memref<4608xi8>, memref<1024xbf16>, memref<128xi8>, i32, memref<2320xbf16>) -> ()
      aie.use_lock(%c7_A0p, Release, 1)
      aie.use_lock(%c7_A1c, AcquireGreaterEqual, 1)
      %29 = arith.addi %26, %c1 : index
      %30 = arith.index_cast %29 : index to i32
      func.call @generic_bcast_gemv_bf16_d(%30, %c7_A1, %c7_silu, %c7_uni_partial, %c4_i32, %c7_P) : (i32, memref<4608xi8>, memref<1024xbf16>, memref<128xi8>, i32, memref<2320xbf16>) -> ()
      aie.use_lock(%c7_A1p, Release, 1)
      %31 = arith.addi %26, %c2 : index
      cf.br ^bb15(%31 : index)
    ^bb17:  // pred: ^bb15
      aie.use_lock(%c7_Pc, Release, 1)
      %32 = arith.addi %0, %c1 : index
      cf.br ^bb1(%32 : index)
    ^bb18:  // pred: ^bb1
      aie.end
    } {link_files = ["layer_fused_bcast_kc256.o", "layer_fused_unified_bcast.o", "rope_il.o", "layer_fused_relay.o"]}
    %mem_5_3 = aie.mem(%tile_5_3) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%c7_A0p, AcquireGreaterEqual, 1)
      aie.dma_bd(%c7_A0 : memref<4608xi8>, 0, 4608) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%c7_A0c, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%c7_A1p, AcquireGreaterEqual, 1)
      aie.dma_bd(%c7_A1 : memref<4608xi8>, 0, 4608) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%c7_A1c, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb7)
    ^bb4:  // 2 preds: ^bb3, ^bb6
      aie.use_lock(%c7_B0p, AcquireGreaterEqual, 1)
      aie.dma_bd(%c7_B0 : memref<2320xbf16>, 0, 2320) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%c7_B0c, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%c7_B1p, AcquireGreaterEqual, 1)
      aie.dma_bd(%c7_B1 : memref<2320xbf16>, 0, 2320) {bd_id = 3 : i32, next_bd_id = 4 : i32}
      aie.use_lock(%c7_B1c, Release, 1)
      aie.next_bd ^bb6
    ^bb6:  // pred: ^bb5
      aie.use_lock(%c7_B2p, AcquireGreaterEqual, 1)
      aie.dma_bd(%c7_B2 : memref<2320xbf16>, 0, 2320) {bd_id = 4 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%c7_B2c, Release, 1)
      aie.next_bd ^bb4
    ^bb7:  // pred: ^bb3
      %2 = aie.dma_start(MM2S, 0, ^bb8, ^bb9)
    ^bb8:  // 2 preds: ^bb7, ^bb8
      aie.use_lock(%c7_Pc, AcquireGreaterEqual, 1)
      aie.dma_bd(%c7_P : memref<2320xbf16>, 0, 2048) {bd_id = 5 : i32, next_bd_id = 5 : i32}
      aie.use_lock(%c7_Pp, Release, 1)
      aie.next_bd ^bb8
    ^bb9:  // pred: ^bb7
      %3 = aie.dma_start(MM2S, 1, ^bb10, ^bb12)
    ^bb10:  // 2 preds: ^bb9, ^bb11
      aie.use_lock(%c7_Qc, AcquireGreaterEqual, 1)
      aie.dma_bd(%c7_Q : memref<322xbf16>, 0, 322) {bd_id = 6 : i32, next_bd_id = 7 : i32}
      aie.use_lock(%c7_Qp, Release, 1)
      aie.next_bd ^bb11
    ^bb11:  // pred: ^bb10
      aie.use_lock(%c7_Oc, AcquireGreaterEqual, 1)
      aie.dma_bd(%c7_O : memref<2048xbf16>, 0, 256) {bd_id = 7 : i32, next_bd_id = 6 : i32}
      aie.use_lock(%c7_Op, Release, 1)
      aie.next_bd ^bb10
    ^bb12:  // pred: ^bb9
      aie.end
    }
    %sc0_Qs = aie.buffer(%tile_0_2) {address = 49152 : i32, mem_bank = 3 : i32, sym_name = "sc0_Qs"} : memref<322xbf16> 
    %sc0_K = aie.buffer(%tile_0_2) {address = 16384 : i32, mem_bank = 1 : i32, sym_name = "sc0_K"} : memref<8192xbf16> 
    %sc0_It = aie.buffer(%tile_0_2) {address = 32768 : i32, mem_bank = 2 : i32, sym_name = "sc0_It"} : memref<520xbf16> 
    %sc0_Oh = aie.buffer(%tile_0_2) {address = 1024 : i32, mem_bank = 0 : i32, sym_name = "sc0_Oh"} : memref<256xbf16> 
    %sc0_Qp = aie.lock(%tile_0_2, 0) {init = 1 : i32, sym_name = "sc0_Qp"}
    %sc0_Qc = aie.lock(%tile_0_2, 1) {init = 0 : i32, sym_name = "sc0_Qc"}
    %sc0_Kp = aie.lock(%tile_0_2, 2) {init = 1 : i32, sym_name = "sc0_Kp"}
    %sc0_Kc = aie.lock(%tile_0_2, 3) {init = 0 : i32, sym_name = "sc0_Kc"}
    %sc0_Ip = aie.lock(%tile_0_2, 4) {init = 1 : i32, sym_name = "sc0_Ip"}
    %sc0_Ic = aie.lock(%tile_0_2, 5) {init = 0 : i32, sym_name = "sc0_Ic"}
    %sc0_Ohp = aie.lock(%tile_0_2, 6) {init = 1 : i32, sym_name = "sc0_Ohp"}
    %sc0_Ohc = aie.lock(%tile_0_2, 7) {init = 0 : i32, sym_name = "sc0_Ohc"}
    %sc0_Ohdp = aie.lock(%tile_0_2, 8) {init = 1 : i32, sym_name = "sc0_Ohdp"}
    %sc0_Ohdc = aie.lock(%tile_0_2, 9) {init = 0 : i32, sym_name = "sc0_Ohdc"}
    %core_0_2 = aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      %c4_i32 = arith.constant 4 : i32
      %c64_i32 = arith.constant 64 : i32
      %c2 = arith.constant 2 : index
      %c128_i32 = arith.constant 128 : i32
      cf.br ^bb1(%c0 : index)
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb5
      %1 = arith.cmpi slt, %0, %c9223372036854775807 : index
      cf.cond_br %1, ^bb2, ^bb6
    ^bb2:  // pred: ^bb1
      func.call @flowkv_score_init_bf16(%c4_i32) : (i32) -> ()
      aie.use_lock(%sc0_Qc, AcquireGreaterEqual, 1)
      func.call @flowkv_score_rope_q_bf16(%sc0_Qs, %c4_i32, %c64_i32) : (memref<322xbf16>, i32, i32) -> ()
      cf.br ^bb3(%c0 : index)
    ^bb3(%2: index):  // 2 preds: ^bb2, ^bb4
      %3 = arith.cmpi slt, %2, %c2 : index
      cf.cond_br %3, ^bb4, ^bb5
    ^bb4:  // pred: ^bb3
      aie.use_lock(%sc0_Kc, AcquireGreaterEqual, 1)
      aie.use_lock(%sc0_Ip, AcquireGreaterEqual, 1)
      func.call @flowkv_score_chunk_bf16(%sc0_Qs, %sc0_K, %sc0_It, %c4_i32, %c64_i32, %c128_i32) : (memref<322xbf16>, memref<8192xbf16>, memref<520xbf16>, i32, i32, i32) -> ()
      aie.use_lock(%sc0_Kp, Release, 1)
      aie.use_lock(%sc0_Ic, Release, 1)
      %4 = arith.addi %2, %c1 : index
      cf.br ^bb3(%4 : index)
    ^bb5:  // pred: ^bb3
      aie.use_lock(%sc0_Qp, Release, 1)
      aie.use_lock(%sc0_Ohc, AcquireGreaterEqual, 1)
      aie.use_lock(%sc0_Ohdp, AcquireGreaterEqual, 1)
      aie.use_lock(%sc0_Ohp, Release, 1)
      aie.use_lock(%sc0_Ohdc, Release, 1)
      %5 = arith.addi %0, %c1 : index
      cf.br ^bb1(%5 : index)
    ^bb6:  // pred: ^bb1
      aie.end
    } {link_files = ["flowkv_64d_h4_c256.o"]}
    %mem_0_2 = aie.mem(%tile_0_2) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%sc0_Qp, AcquireGreaterEqual, 1)
      aie.dma_bd(%sc0_Qs : memref<322xbf16>, 0, 322) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%sc0_Qc, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%sc0_Ohp, AcquireGreaterEqual, 1)
      aie.dma_bd(%sc0_Oh : memref<256xbf16>, 0, 256) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%sc0_Ohc, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb5)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%sc0_Kp, AcquireGreaterEqual, 1)
      aie.dma_bd(%sc0_K : memref<8192xbf16>, 0, 8192) {bd_id = 2 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%sc0_Kc, Release, 1)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb3
      %2 = aie.dma_start(MM2S, 0, ^bb6, ^bb7)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      aie.use_lock(%sc0_Ic, AcquireGreaterEqual, 1)
      aie.dma_bd(%sc0_It : memref<520xbf16>, 0, 520) {bd_id = 3 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%sc0_Ip, Release, 1)
      aie.next_bd ^bb6
    ^bb7:  // pred: ^bb5
      %3 = aie.dma_start(MM2S, 1, ^bb8, ^bb9)
    ^bb8:  // 2 preds: ^bb7, ^bb8
      aie.use_lock(%sc0_Ohdc, AcquireGreaterEqual, 1)
      aie.dma_bd(%sc0_Oh : memref<256xbf16>, 0, 256) {bd_id = 4 : i32, next_bd_id = 4 : i32}
      aie.use_lock(%sc0_Ohdp, Release, 1)
      aie.next_bd ^bb8
    ^bb9:  // pred: ^bb7
      aie.end
    }
    %sc1_Qs = aie.buffer(%tile_1_2) {address = 49152 : i32, mem_bank = 3 : i32, sym_name = "sc1_Qs"} : memref<322xbf16> 
    %sc1_K = aie.buffer(%tile_1_2) {address = 16384 : i32, mem_bank = 1 : i32, sym_name = "sc1_K"} : memref<8192xbf16> 
    %sc1_It = aie.buffer(%tile_1_2) {address = 32768 : i32, mem_bank = 2 : i32, sym_name = "sc1_It"} : memref<520xbf16> 
    %sc1_Oh = aie.buffer(%tile_1_2) {address = 1024 : i32, mem_bank = 0 : i32, sym_name = "sc1_Oh"} : memref<256xbf16> 
    %sc1_Qp = aie.lock(%tile_1_2, 0) {init = 1 : i32, sym_name = "sc1_Qp"}
    %sc1_Qc = aie.lock(%tile_1_2, 1) {init = 0 : i32, sym_name = "sc1_Qc"}
    %sc1_Kp = aie.lock(%tile_1_2, 2) {init = 1 : i32, sym_name = "sc1_Kp"}
    %sc1_Kc = aie.lock(%tile_1_2, 3) {init = 0 : i32, sym_name = "sc1_Kc"}
    %sc1_Ip = aie.lock(%tile_1_2, 4) {init = 1 : i32, sym_name = "sc1_Ip"}
    %sc1_Ic = aie.lock(%tile_1_2, 5) {init = 0 : i32, sym_name = "sc1_Ic"}
    %sc1_Ohp = aie.lock(%tile_1_2, 6) {init = 1 : i32, sym_name = "sc1_Ohp"}
    %sc1_Ohc = aie.lock(%tile_1_2, 7) {init = 0 : i32, sym_name = "sc1_Ohc"}
    %sc1_Ohdp = aie.lock(%tile_1_2, 8) {init = 1 : i32, sym_name = "sc1_Ohdp"}
    %sc1_Ohdc = aie.lock(%tile_1_2, 9) {init = 0 : i32, sym_name = "sc1_Ohdc"}
    %core_1_2 = aie.core(%tile_1_2) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      %c4_i32 = arith.constant 4 : i32
      %c64_i32 = arith.constant 64 : i32
      %c2 = arith.constant 2 : index
      %c128_i32 = arith.constant 128 : i32
      cf.br ^bb1(%c0 : index)
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb5
      %1 = arith.cmpi slt, %0, %c9223372036854775807 : index
      cf.cond_br %1, ^bb2, ^bb6
    ^bb2:  // pred: ^bb1
      func.call @flowkv_score_init_bf16(%c4_i32) : (i32) -> ()
      aie.use_lock(%sc1_Qc, AcquireGreaterEqual, 1)
      func.call @flowkv_score_rope_q_bf16(%sc1_Qs, %c4_i32, %c64_i32) : (memref<322xbf16>, i32, i32) -> ()
      cf.br ^bb3(%c0 : index)
    ^bb3(%2: index):  // 2 preds: ^bb2, ^bb4
      %3 = arith.cmpi slt, %2, %c2 : index
      cf.cond_br %3, ^bb4, ^bb5
    ^bb4:  // pred: ^bb3
      aie.use_lock(%sc1_Kc, AcquireGreaterEqual, 1)
      aie.use_lock(%sc1_Ip, AcquireGreaterEqual, 1)
      func.call @flowkv_score_chunk_bf16(%sc1_Qs, %sc1_K, %sc1_It, %c4_i32, %c64_i32, %c128_i32) : (memref<322xbf16>, memref<8192xbf16>, memref<520xbf16>, i32, i32, i32) -> ()
      aie.use_lock(%sc1_Kp, Release, 1)
      aie.use_lock(%sc1_Ic, Release, 1)
      %4 = arith.addi %2, %c1 : index
      cf.br ^bb3(%4 : index)
    ^bb5:  // pred: ^bb3
      aie.use_lock(%sc1_Qp, Release, 1)
      aie.use_lock(%sc1_Ohc, AcquireGreaterEqual, 1)
      aie.use_lock(%sc1_Ohdp, AcquireGreaterEqual, 1)
      aie.use_lock(%sc1_Ohp, Release, 1)
      aie.use_lock(%sc1_Ohdc, Release, 1)
      %5 = arith.addi %0, %c1 : index
      cf.br ^bb1(%5 : index)
    ^bb6:  // pred: ^bb1
      aie.end
    } {link_files = ["flowkv_64d_h4_c256.o"]}
    %mem_1_2 = aie.mem(%tile_1_2) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%sc1_Qp, AcquireGreaterEqual, 1)
      aie.dma_bd(%sc1_Qs : memref<322xbf16>, 0, 322) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%sc1_Qc, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%sc1_Ohp, AcquireGreaterEqual, 1)
      aie.dma_bd(%sc1_Oh : memref<256xbf16>, 0, 256) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%sc1_Ohc, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb5)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%sc1_Kp, AcquireGreaterEqual, 1)
      aie.dma_bd(%sc1_K : memref<8192xbf16>, 0, 8192) {bd_id = 2 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%sc1_Kc, Release, 1)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb3
      %2 = aie.dma_start(MM2S, 0, ^bb6, ^bb7)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      aie.use_lock(%sc1_Ic, AcquireGreaterEqual, 1)
      aie.dma_bd(%sc1_It : memref<520xbf16>, 0, 520) {bd_id = 3 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%sc1_Ip, Release, 1)
      aie.next_bd ^bb6
    ^bb7:  // pred: ^bb5
      %3 = aie.dma_start(MM2S, 1, ^bb8, ^bb9)
    ^bb8:  // 2 preds: ^bb7, ^bb8
      aie.use_lock(%sc1_Ohdc, AcquireGreaterEqual, 1)
      aie.dma_bd(%sc1_Oh : memref<256xbf16>, 0, 256) {bd_id = 4 : i32, next_bd_id = 4 : i32}
      aie.use_lock(%sc1_Ohdp, Release, 1)
      aie.next_bd ^bb8
    ^bb9:  // pred: ^bb7
      aie.end
    }
    %sc2_Qs = aie.buffer(%tile_6_2) {address = 49152 : i32, mem_bank = 3 : i32, sym_name = "sc2_Qs"} : memref<322xbf16> 
    %sc2_K = aie.buffer(%tile_6_2) {address = 16384 : i32, mem_bank = 1 : i32, sym_name = "sc2_K"} : memref<8192xbf16> 
    %sc2_It = aie.buffer(%tile_6_2) {address = 32768 : i32, mem_bank = 2 : i32, sym_name = "sc2_It"} : memref<520xbf16> 
    %sc2_Oh = aie.buffer(%tile_6_2) {address = 1024 : i32, mem_bank = 0 : i32, sym_name = "sc2_Oh"} : memref<256xbf16> 
    %sc2_Qp = aie.lock(%tile_6_2, 0) {init = 1 : i32, sym_name = "sc2_Qp"}
    %sc2_Qc = aie.lock(%tile_6_2, 1) {init = 0 : i32, sym_name = "sc2_Qc"}
    %sc2_Kp = aie.lock(%tile_6_2, 2) {init = 1 : i32, sym_name = "sc2_Kp"}
    %sc2_Kc = aie.lock(%tile_6_2, 3) {init = 0 : i32, sym_name = "sc2_Kc"}
    %sc2_Ip = aie.lock(%tile_6_2, 4) {init = 1 : i32, sym_name = "sc2_Ip"}
    %sc2_Ic = aie.lock(%tile_6_2, 5) {init = 0 : i32, sym_name = "sc2_Ic"}
    %sc2_Ohp = aie.lock(%tile_6_2, 6) {init = 1 : i32, sym_name = "sc2_Ohp"}
    %sc2_Ohc = aie.lock(%tile_6_2, 7) {init = 0 : i32, sym_name = "sc2_Ohc"}
    %sc2_Ohdp = aie.lock(%tile_6_2, 8) {init = 1 : i32, sym_name = "sc2_Ohdp"}
    %sc2_Ohdc = aie.lock(%tile_6_2, 9) {init = 0 : i32, sym_name = "sc2_Ohdc"}
    %core_6_2 = aie.core(%tile_6_2) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      %c4_i32 = arith.constant 4 : i32
      %c64_i32 = arith.constant 64 : i32
      %c2 = arith.constant 2 : index
      %c128_i32 = arith.constant 128 : i32
      cf.br ^bb1(%c0 : index)
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb5
      %1 = arith.cmpi slt, %0, %c9223372036854775807 : index
      cf.cond_br %1, ^bb2, ^bb6
    ^bb2:  // pred: ^bb1
      func.call @flowkv_score_init_bf16(%c4_i32) : (i32) -> ()
      aie.use_lock(%sc2_Qc, AcquireGreaterEqual, 1)
      func.call @flowkv_score_rope_q_bf16(%sc2_Qs, %c4_i32, %c64_i32) : (memref<322xbf16>, i32, i32) -> ()
      cf.br ^bb3(%c0 : index)
    ^bb3(%2: index):  // 2 preds: ^bb2, ^bb4
      %3 = arith.cmpi slt, %2, %c2 : index
      cf.cond_br %3, ^bb4, ^bb5
    ^bb4:  // pred: ^bb3
      aie.use_lock(%sc2_Kc, AcquireGreaterEqual, 1)
      aie.use_lock(%sc2_Ip, AcquireGreaterEqual, 1)
      func.call @flowkv_score_chunk_bf16(%sc2_Qs, %sc2_K, %sc2_It, %c4_i32, %c64_i32, %c128_i32) : (memref<322xbf16>, memref<8192xbf16>, memref<520xbf16>, i32, i32, i32) -> ()
      aie.use_lock(%sc2_Kp, Release, 1)
      aie.use_lock(%sc2_Ic, Release, 1)
      %4 = arith.addi %2, %c1 : index
      cf.br ^bb3(%4 : index)
    ^bb5:  // pred: ^bb3
      aie.use_lock(%sc2_Qp, Release, 1)
      aie.use_lock(%sc2_Ohc, AcquireGreaterEqual, 1)
      aie.use_lock(%sc2_Ohdp, AcquireGreaterEqual, 1)
      aie.use_lock(%sc2_Ohp, Release, 1)
      aie.use_lock(%sc2_Ohdc, Release, 1)
      %5 = arith.addi %0, %c1 : index
      cf.br ^bb1(%5 : index)
    ^bb6:  // pred: ^bb1
      aie.end
    } {link_files = ["flowkv_64d_h4_c256.o"]}
    %mem_6_2 = aie.mem(%tile_6_2) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%sc2_Qp, AcquireGreaterEqual, 1)
      aie.dma_bd(%sc2_Qs : memref<322xbf16>, 0, 322) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%sc2_Qc, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%sc2_Ohp, AcquireGreaterEqual, 1)
      aie.dma_bd(%sc2_Oh : memref<256xbf16>, 0, 256) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%sc2_Ohc, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb5)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%sc2_Kp, AcquireGreaterEqual, 1)
      aie.dma_bd(%sc2_K : memref<8192xbf16>, 0, 8192) {bd_id = 2 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%sc2_Kc, Release, 1)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb3
      %2 = aie.dma_start(MM2S, 0, ^bb6, ^bb7)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      aie.use_lock(%sc2_Ic, AcquireGreaterEqual, 1)
      aie.dma_bd(%sc2_It : memref<520xbf16>, 0, 520) {bd_id = 3 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%sc2_Ip, Release, 1)
      aie.next_bd ^bb6
    ^bb7:  // pred: ^bb5
      %3 = aie.dma_start(MM2S, 1, ^bb8, ^bb9)
    ^bb8:  // 2 preds: ^bb7, ^bb8
      aie.use_lock(%sc2_Ohdc, AcquireGreaterEqual, 1)
      aie.dma_bd(%sc2_Oh : memref<256xbf16>, 0, 256) {bd_id = 4 : i32, next_bd_id = 4 : i32}
      aie.use_lock(%sc2_Ohdp, Release, 1)
      aie.next_bd ^bb8
    ^bb9:  // pred: ^bb7
      aie.end
    }
    %sc3_Qs = aie.buffer(%tile_7_2) {address = 49152 : i32, mem_bank = 3 : i32, sym_name = "sc3_Qs"} : memref<322xbf16> 
    %sc3_K = aie.buffer(%tile_7_2) {address = 16384 : i32, mem_bank = 1 : i32, sym_name = "sc3_K"} : memref<8192xbf16> 
    %sc3_It = aie.buffer(%tile_7_2) {address = 32768 : i32, mem_bank = 2 : i32, sym_name = "sc3_It"} : memref<520xbf16> 
    %sc3_Oh = aie.buffer(%tile_7_2) {address = 1024 : i32, mem_bank = 0 : i32, sym_name = "sc3_Oh"} : memref<256xbf16> 
    %sc3_Qp = aie.lock(%tile_7_2, 0) {init = 1 : i32, sym_name = "sc3_Qp"}
    %sc3_Qc = aie.lock(%tile_7_2, 1) {init = 0 : i32, sym_name = "sc3_Qc"}
    %sc3_Kp = aie.lock(%tile_7_2, 2) {init = 1 : i32, sym_name = "sc3_Kp"}
    %sc3_Kc = aie.lock(%tile_7_2, 3) {init = 0 : i32, sym_name = "sc3_Kc"}
    %sc3_Ip = aie.lock(%tile_7_2, 4) {init = 1 : i32, sym_name = "sc3_Ip"}
    %sc3_Ic = aie.lock(%tile_7_2, 5) {init = 0 : i32, sym_name = "sc3_Ic"}
    %sc3_Ohp = aie.lock(%tile_7_2, 6) {init = 1 : i32, sym_name = "sc3_Ohp"}
    %sc3_Ohc = aie.lock(%tile_7_2, 7) {init = 0 : i32, sym_name = "sc3_Ohc"}
    %sc3_Ohdp = aie.lock(%tile_7_2, 8) {init = 1 : i32, sym_name = "sc3_Ohdp"}
    %sc3_Ohdc = aie.lock(%tile_7_2, 9) {init = 0 : i32, sym_name = "sc3_Ohdc"}
    %core_7_2 = aie.core(%tile_7_2) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      %c4_i32 = arith.constant 4 : i32
      %c64_i32 = arith.constant 64 : i32
      %c2 = arith.constant 2 : index
      %c128_i32 = arith.constant 128 : i32
      cf.br ^bb1(%c0 : index)
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb5
      %1 = arith.cmpi slt, %0, %c9223372036854775807 : index
      cf.cond_br %1, ^bb2, ^bb6
    ^bb2:  // pred: ^bb1
      func.call @flowkv_score_init_bf16(%c4_i32) : (i32) -> ()
      aie.use_lock(%sc3_Qc, AcquireGreaterEqual, 1)
      func.call @flowkv_score_rope_q_bf16(%sc3_Qs, %c4_i32, %c64_i32) : (memref<322xbf16>, i32, i32) -> ()
      cf.br ^bb3(%c0 : index)
    ^bb3(%2: index):  // 2 preds: ^bb2, ^bb4
      %3 = arith.cmpi slt, %2, %c2 : index
      cf.cond_br %3, ^bb4, ^bb5
    ^bb4:  // pred: ^bb3
      aie.use_lock(%sc3_Kc, AcquireGreaterEqual, 1)
      aie.use_lock(%sc3_Ip, AcquireGreaterEqual, 1)
      func.call @flowkv_score_chunk_bf16(%sc3_Qs, %sc3_K, %sc3_It, %c4_i32, %c64_i32, %c128_i32) : (memref<322xbf16>, memref<8192xbf16>, memref<520xbf16>, i32, i32, i32) -> ()
      aie.use_lock(%sc3_Kp, Release, 1)
      aie.use_lock(%sc3_Ic, Release, 1)
      %4 = arith.addi %2, %c1 : index
      cf.br ^bb3(%4 : index)
    ^bb5:  // pred: ^bb3
      aie.use_lock(%sc3_Qp, Release, 1)
      aie.use_lock(%sc3_Ohc, AcquireGreaterEqual, 1)
      aie.use_lock(%sc3_Ohdp, AcquireGreaterEqual, 1)
      aie.use_lock(%sc3_Ohp, Release, 1)
      aie.use_lock(%sc3_Ohdc, Release, 1)
      %5 = arith.addi %0, %c1 : index
      cf.br ^bb1(%5 : index)
    ^bb6:  // pred: ^bb1
      aie.end
    } {link_files = ["flowkv_64d_h4_c256.o"]}
    %mem_7_2 = aie.mem(%tile_7_2) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%sc3_Qp, AcquireGreaterEqual, 1)
      aie.dma_bd(%sc3_Qs : memref<322xbf16>, 0, 322) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%sc3_Qc, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%sc3_Ohp, AcquireGreaterEqual, 1)
      aie.dma_bd(%sc3_Oh : memref<256xbf16>, 0, 256) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%sc3_Ohc, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb5)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%sc3_Kp, AcquireGreaterEqual, 1)
      aie.dma_bd(%sc3_K : memref<8192xbf16>, 0, 8192) {bd_id = 2 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%sc3_Kc, Release, 1)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb3
      %2 = aie.dma_start(MM2S, 0, ^bb6, ^bb7)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      aie.use_lock(%sc3_Ic, AcquireGreaterEqual, 1)
      aie.dma_bd(%sc3_It : memref<520xbf16>, 0, 520) {bd_id = 3 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%sc3_Ip, Release, 1)
      aie.next_bd ^bb6
    ^bb7:  // pred: ^bb5
      %3 = aie.dma_start(MM2S, 1, ^bb8, ^bb9)
    ^bb8:  // 2 preds: ^bb7, ^bb8
      aie.use_lock(%sc3_Ohdc, AcquireGreaterEqual, 1)
      aie.dma_bd(%sc3_Oh : memref<256xbf16>, 0, 256) {bd_id = 4 : i32, next_bd_id = 4 : i32}
      aie.use_lock(%sc3_Ohdp, Release, 1)
      aie.next_bd ^bb8
    ^bb9:  // pred: ^bb7
      aie.end
    }
    %sc4_Qs = aie.buffer(%tile_0_3) {address = 49152 : i32, mem_bank = 3 : i32, sym_name = "sc4_Qs"} : memref<322xbf16> 
    %sc4_K = aie.buffer(%tile_0_3) {address = 16384 : i32, mem_bank = 1 : i32, sym_name = "sc4_K"} : memref<8192xbf16> 
    %sc4_It = aie.buffer(%tile_0_3) {address = 32768 : i32, mem_bank = 2 : i32, sym_name = "sc4_It"} : memref<520xbf16> 
    %sc4_Oh = aie.buffer(%tile_0_3) {address = 1024 : i32, mem_bank = 0 : i32, sym_name = "sc4_Oh"} : memref<256xbf16> 
    %sc4_Qp = aie.lock(%tile_0_3, 0) {init = 1 : i32, sym_name = "sc4_Qp"}
    %sc4_Qc = aie.lock(%tile_0_3, 1) {init = 0 : i32, sym_name = "sc4_Qc"}
    %sc4_Kp = aie.lock(%tile_0_3, 2) {init = 1 : i32, sym_name = "sc4_Kp"}
    %sc4_Kc = aie.lock(%tile_0_3, 3) {init = 0 : i32, sym_name = "sc4_Kc"}
    %sc4_Ip = aie.lock(%tile_0_3, 4) {init = 1 : i32, sym_name = "sc4_Ip"}
    %sc4_Ic = aie.lock(%tile_0_3, 5) {init = 0 : i32, sym_name = "sc4_Ic"}
    %sc4_Ohp = aie.lock(%tile_0_3, 6) {init = 1 : i32, sym_name = "sc4_Ohp"}
    %sc4_Ohc = aie.lock(%tile_0_3, 7) {init = 0 : i32, sym_name = "sc4_Ohc"}
    %sc4_Ohdp = aie.lock(%tile_0_3, 8) {init = 1 : i32, sym_name = "sc4_Ohdp"}
    %sc4_Ohdc = aie.lock(%tile_0_3, 9) {init = 0 : i32, sym_name = "sc4_Ohdc"}
    %core_0_3 = aie.core(%tile_0_3) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      %c4_i32 = arith.constant 4 : i32
      %c64_i32 = arith.constant 64 : i32
      %c2 = arith.constant 2 : index
      %c128_i32 = arith.constant 128 : i32
      cf.br ^bb1(%c0 : index)
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb5
      %1 = arith.cmpi slt, %0, %c9223372036854775807 : index
      cf.cond_br %1, ^bb2, ^bb6
    ^bb2:  // pred: ^bb1
      func.call @flowkv_score_init_bf16(%c4_i32) : (i32) -> ()
      aie.use_lock(%sc4_Qc, AcquireGreaterEqual, 1)
      func.call @flowkv_score_rope_q_bf16(%sc4_Qs, %c4_i32, %c64_i32) : (memref<322xbf16>, i32, i32) -> ()
      cf.br ^bb3(%c0 : index)
    ^bb3(%2: index):  // 2 preds: ^bb2, ^bb4
      %3 = arith.cmpi slt, %2, %c2 : index
      cf.cond_br %3, ^bb4, ^bb5
    ^bb4:  // pred: ^bb3
      aie.use_lock(%sc4_Kc, AcquireGreaterEqual, 1)
      aie.use_lock(%sc4_Ip, AcquireGreaterEqual, 1)
      func.call @flowkv_score_chunk_bf16(%sc4_Qs, %sc4_K, %sc4_It, %c4_i32, %c64_i32, %c128_i32) : (memref<322xbf16>, memref<8192xbf16>, memref<520xbf16>, i32, i32, i32) -> ()
      aie.use_lock(%sc4_Kp, Release, 1)
      aie.use_lock(%sc4_Ic, Release, 1)
      %4 = arith.addi %2, %c1 : index
      cf.br ^bb3(%4 : index)
    ^bb5:  // pred: ^bb3
      aie.use_lock(%sc4_Qp, Release, 1)
      aie.use_lock(%sc4_Ohc, AcquireGreaterEqual, 1)
      aie.use_lock(%sc4_Ohdp, AcquireGreaterEqual, 1)
      aie.use_lock(%sc4_Ohp, Release, 1)
      aie.use_lock(%sc4_Ohdc, Release, 1)
      %5 = arith.addi %0, %c1 : index
      cf.br ^bb1(%5 : index)
    ^bb6:  // pred: ^bb1
      aie.end
    } {link_files = ["flowkv_64d_h4_c256.o"]}
    %mem_0_3 = aie.mem(%tile_0_3) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%sc4_Qp, AcquireGreaterEqual, 1)
      aie.dma_bd(%sc4_Qs : memref<322xbf16>, 0, 322) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%sc4_Qc, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%sc4_Ohp, AcquireGreaterEqual, 1)
      aie.dma_bd(%sc4_Oh : memref<256xbf16>, 0, 256) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%sc4_Ohc, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb5)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%sc4_Kp, AcquireGreaterEqual, 1)
      aie.dma_bd(%sc4_K : memref<8192xbf16>, 0, 8192) {bd_id = 2 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%sc4_Kc, Release, 1)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb3
      %2 = aie.dma_start(MM2S, 0, ^bb6, ^bb7)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      aie.use_lock(%sc4_Ic, AcquireGreaterEqual, 1)
      aie.dma_bd(%sc4_It : memref<520xbf16>, 0, 520) {bd_id = 3 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%sc4_Ip, Release, 1)
      aie.next_bd ^bb6
    ^bb7:  // pred: ^bb5
      %3 = aie.dma_start(MM2S, 1, ^bb8, ^bb9)
    ^bb8:  // 2 preds: ^bb7, ^bb8
      aie.use_lock(%sc4_Ohdc, AcquireGreaterEqual, 1)
      aie.dma_bd(%sc4_Oh : memref<256xbf16>, 0, 256) {bd_id = 4 : i32, next_bd_id = 4 : i32}
      aie.use_lock(%sc4_Ohdp, Release, 1)
      aie.next_bd ^bb8
    ^bb9:  // pred: ^bb7
      aie.end
    }
    %sc5_Qs = aie.buffer(%tile_1_3) {address = 49152 : i32, mem_bank = 3 : i32, sym_name = "sc5_Qs"} : memref<322xbf16> 
    %sc5_K = aie.buffer(%tile_1_3) {address = 16384 : i32, mem_bank = 1 : i32, sym_name = "sc5_K"} : memref<8192xbf16> 
    %sc5_It = aie.buffer(%tile_1_3) {address = 32768 : i32, mem_bank = 2 : i32, sym_name = "sc5_It"} : memref<520xbf16> 
    %sc5_Oh = aie.buffer(%tile_1_3) {address = 1024 : i32, mem_bank = 0 : i32, sym_name = "sc5_Oh"} : memref<256xbf16> 
    %sc5_Qp = aie.lock(%tile_1_3, 0) {init = 1 : i32, sym_name = "sc5_Qp"}
    %sc5_Qc = aie.lock(%tile_1_3, 1) {init = 0 : i32, sym_name = "sc5_Qc"}
    %sc5_Kp = aie.lock(%tile_1_3, 2) {init = 1 : i32, sym_name = "sc5_Kp"}
    %sc5_Kc = aie.lock(%tile_1_3, 3) {init = 0 : i32, sym_name = "sc5_Kc"}
    %sc5_Ip = aie.lock(%tile_1_3, 4) {init = 1 : i32, sym_name = "sc5_Ip"}
    %sc5_Ic = aie.lock(%tile_1_3, 5) {init = 0 : i32, sym_name = "sc5_Ic"}
    %sc5_Ohp = aie.lock(%tile_1_3, 6) {init = 1 : i32, sym_name = "sc5_Ohp"}
    %sc5_Ohc = aie.lock(%tile_1_3, 7) {init = 0 : i32, sym_name = "sc5_Ohc"}
    %sc5_Ohdp = aie.lock(%tile_1_3, 8) {init = 1 : i32, sym_name = "sc5_Ohdp"}
    %sc5_Ohdc = aie.lock(%tile_1_3, 9) {init = 0 : i32, sym_name = "sc5_Ohdc"}
    %core_1_3 = aie.core(%tile_1_3) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      %c4_i32 = arith.constant 4 : i32
      %c64_i32 = arith.constant 64 : i32
      %c2 = arith.constant 2 : index
      %c128_i32 = arith.constant 128 : i32
      cf.br ^bb1(%c0 : index)
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb5
      %1 = arith.cmpi slt, %0, %c9223372036854775807 : index
      cf.cond_br %1, ^bb2, ^bb6
    ^bb2:  // pred: ^bb1
      func.call @flowkv_score_init_bf16(%c4_i32) : (i32) -> ()
      aie.use_lock(%sc5_Qc, AcquireGreaterEqual, 1)
      func.call @flowkv_score_rope_q_bf16(%sc5_Qs, %c4_i32, %c64_i32) : (memref<322xbf16>, i32, i32) -> ()
      cf.br ^bb3(%c0 : index)
    ^bb3(%2: index):  // 2 preds: ^bb2, ^bb4
      %3 = arith.cmpi slt, %2, %c2 : index
      cf.cond_br %3, ^bb4, ^bb5
    ^bb4:  // pred: ^bb3
      aie.use_lock(%sc5_Kc, AcquireGreaterEqual, 1)
      aie.use_lock(%sc5_Ip, AcquireGreaterEqual, 1)
      func.call @flowkv_score_chunk_bf16(%sc5_Qs, %sc5_K, %sc5_It, %c4_i32, %c64_i32, %c128_i32) : (memref<322xbf16>, memref<8192xbf16>, memref<520xbf16>, i32, i32, i32) -> ()
      aie.use_lock(%sc5_Kp, Release, 1)
      aie.use_lock(%sc5_Ic, Release, 1)
      %4 = arith.addi %2, %c1 : index
      cf.br ^bb3(%4 : index)
    ^bb5:  // pred: ^bb3
      aie.use_lock(%sc5_Qp, Release, 1)
      aie.use_lock(%sc5_Ohc, AcquireGreaterEqual, 1)
      aie.use_lock(%sc5_Ohdp, AcquireGreaterEqual, 1)
      aie.use_lock(%sc5_Ohp, Release, 1)
      aie.use_lock(%sc5_Ohdc, Release, 1)
      %5 = arith.addi %0, %c1 : index
      cf.br ^bb1(%5 : index)
    ^bb6:  // pred: ^bb1
      aie.end
    } {link_files = ["flowkv_64d_h4_c256.o"]}
    %mem_1_3 = aie.mem(%tile_1_3) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%sc5_Qp, AcquireGreaterEqual, 1)
      aie.dma_bd(%sc5_Qs : memref<322xbf16>, 0, 322) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%sc5_Qc, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%sc5_Ohp, AcquireGreaterEqual, 1)
      aie.dma_bd(%sc5_Oh : memref<256xbf16>, 0, 256) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%sc5_Ohc, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb5)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%sc5_Kp, AcquireGreaterEqual, 1)
      aie.dma_bd(%sc5_K : memref<8192xbf16>, 0, 8192) {bd_id = 2 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%sc5_Kc, Release, 1)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb3
      %2 = aie.dma_start(MM2S, 0, ^bb6, ^bb7)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      aie.use_lock(%sc5_Ic, AcquireGreaterEqual, 1)
      aie.dma_bd(%sc5_It : memref<520xbf16>, 0, 520) {bd_id = 3 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%sc5_Ip, Release, 1)
      aie.next_bd ^bb6
    ^bb7:  // pred: ^bb5
      %3 = aie.dma_start(MM2S, 1, ^bb8, ^bb9)
    ^bb8:  // 2 preds: ^bb7, ^bb8
      aie.use_lock(%sc5_Ohdc, AcquireGreaterEqual, 1)
      aie.dma_bd(%sc5_Oh : memref<256xbf16>, 0, 256) {bd_id = 4 : i32, next_bd_id = 4 : i32}
      aie.use_lock(%sc5_Ohdp, Release, 1)
      aie.next_bd ^bb8
    ^bb9:  // pred: ^bb7
      aie.end
    }
    %sc6_Qs = aie.buffer(%tile_6_3) {address = 49152 : i32, mem_bank = 3 : i32, sym_name = "sc6_Qs"} : memref<322xbf16> 
    %sc6_K = aie.buffer(%tile_6_3) {address = 16384 : i32, mem_bank = 1 : i32, sym_name = "sc6_K"} : memref<8192xbf16> 
    %sc6_It = aie.buffer(%tile_6_3) {address = 32768 : i32, mem_bank = 2 : i32, sym_name = "sc6_It"} : memref<520xbf16> 
    %sc6_Oh = aie.buffer(%tile_6_3) {address = 1024 : i32, mem_bank = 0 : i32, sym_name = "sc6_Oh"} : memref<256xbf16> 
    %sc6_Qp = aie.lock(%tile_6_3, 0) {init = 1 : i32, sym_name = "sc6_Qp"}
    %sc6_Qc = aie.lock(%tile_6_3, 1) {init = 0 : i32, sym_name = "sc6_Qc"}
    %sc6_Kp = aie.lock(%tile_6_3, 2) {init = 1 : i32, sym_name = "sc6_Kp"}
    %sc6_Kc = aie.lock(%tile_6_3, 3) {init = 0 : i32, sym_name = "sc6_Kc"}
    %sc6_Ip = aie.lock(%tile_6_3, 4) {init = 1 : i32, sym_name = "sc6_Ip"}
    %sc6_Ic = aie.lock(%tile_6_3, 5) {init = 0 : i32, sym_name = "sc6_Ic"}
    %sc6_Ohp = aie.lock(%tile_6_3, 6) {init = 1 : i32, sym_name = "sc6_Ohp"}
    %sc6_Ohc = aie.lock(%tile_6_3, 7) {init = 0 : i32, sym_name = "sc6_Ohc"}
    %sc6_Ohdp = aie.lock(%tile_6_3, 8) {init = 1 : i32, sym_name = "sc6_Ohdp"}
    %sc6_Ohdc = aie.lock(%tile_6_3, 9) {init = 0 : i32, sym_name = "sc6_Ohdc"}
    %core_6_3 = aie.core(%tile_6_3) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      %c4_i32 = arith.constant 4 : i32
      %c64_i32 = arith.constant 64 : i32
      %c2 = arith.constant 2 : index
      %c128_i32 = arith.constant 128 : i32
      cf.br ^bb1(%c0 : index)
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb5
      %1 = arith.cmpi slt, %0, %c9223372036854775807 : index
      cf.cond_br %1, ^bb2, ^bb6
    ^bb2:  // pred: ^bb1
      func.call @flowkv_score_init_bf16(%c4_i32) : (i32) -> ()
      aie.use_lock(%sc6_Qc, AcquireGreaterEqual, 1)
      func.call @flowkv_score_rope_q_bf16(%sc6_Qs, %c4_i32, %c64_i32) : (memref<322xbf16>, i32, i32) -> ()
      cf.br ^bb3(%c0 : index)
    ^bb3(%2: index):  // 2 preds: ^bb2, ^bb4
      %3 = arith.cmpi slt, %2, %c2 : index
      cf.cond_br %3, ^bb4, ^bb5
    ^bb4:  // pred: ^bb3
      aie.use_lock(%sc6_Kc, AcquireGreaterEqual, 1)
      aie.use_lock(%sc6_Ip, AcquireGreaterEqual, 1)
      func.call @flowkv_score_chunk_bf16(%sc6_Qs, %sc6_K, %sc6_It, %c4_i32, %c64_i32, %c128_i32) : (memref<322xbf16>, memref<8192xbf16>, memref<520xbf16>, i32, i32, i32) -> ()
      aie.use_lock(%sc6_Kp, Release, 1)
      aie.use_lock(%sc6_Ic, Release, 1)
      %4 = arith.addi %2, %c1 : index
      cf.br ^bb3(%4 : index)
    ^bb5:  // pred: ^bb3
      aie.use_lock(%sc6_Qp, Release, 1)
      aie.use_lock(%sc6_Ohc, AcquireGreaterEqual, 1)
      aie.use_lock(%sc6_Ohdp, AcquireGreaterEqual, 1)
      aie.use_lock(%sc6_Ohp, Release, 1)
      aie.use_lock(%sc6_Ohdc, Release, 1)
      %5 = arith.addi %0, %c1 : index
      cf.br ^bb1(%5 : index)
    ^bb6:  // pred: ^bb1
      aie.end
    } {link_files = ["flowkv_64d_h4_c256.o"]}
    %mem_6_3 = aie.mem(%tile_6_3) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%sc6_Qp, AcquireGreaterEqual, 1)
      aie.dma_bd(%sc6_Qs : memref<322xbf16>, 0, 322) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%sc6_Qc, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%sc6_Ohp, AcquireGreaterEqual, 1)
      aie.dma_bd(%sc6_Oh : memref<256xbf16>, 0, 256) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%sc6_Ohc, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb5)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%sc6_Kp, AcquireGreaterEqual, 1)
      aie.dma_bd(%sc6_K : memref<8192xbf16>, 0, 8192) {bd_id = 2 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%sc6_Kc, Release, 1)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb3
      %2 = aie.dma_start(MM2S, 0, ^bb6, ^bb7)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      aie.use_lock(%sc6_Ic, AcquireGreaterEqual, 1)
      aie.dma_bd(%sc6_It : memref<520xbf16>, 0, 520) {bd_id = 3 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%sc6_Ip, Release, 1)
      aie.next_bd ^bb6
    ^bb7:  // pred: ^bb5
      %3 = aie.dma_start(MM2S, 1, ^bb8, ^bb9)
    ^bb8:  // 2 preds: ^bb7, ^bb8
      aie.use_lock(%sc6_Ohdc, AcquireGreaterEqual, 1)
      aie.dma_bd(%sc6_Oh : memref<256xbf16>, 0, 256) {bd_id = 4 : i32, next_bd_id = 4 : i32}
      aie.use_lock(%sc6_Ohdp, Release, 1)
      aie.next_bd ^bb8
    ^bb9:  // pred: ^bb7
      aie.end
    }
    %sc7_Qs = aie.buffer(%tile_7_3) {address = 49152 : i32, mem_bank = 3 : i32, sym_name = "sc7_Qs"} : memref<322xbf16> 
    %sc7_K = aie.buffer(%tile_7_3) {address = 16384 : i32, mem_bank = 1 : i32, sym_name = "sc7_K"} : memref<8192xbf16> 
    %sc7_It = aie.buffer(%tile_7_3) {address = 32768 : i32, mem_bank = 2 : i32, sym_name = "sc7_It"} : memref<520xbf16> 
    %sc7_Oh = aie.buffer(%tile_7_3) {address = 1024 : i32, mem_bank = 0 : i32, sym_name = "sc7_Oh"} : memref<256xbf16> 
    %sc7_Qp = aie.lock(%tile_7_3, 0) {init = 1 : i32, sym_name = "sc7_Qp"}
    %sc7_Qc = aie.lock(%tile_7_3, 1) {init = 0 : i32, sym_name = "sc7_Qc"}
    %sc7_Kp = aie.lock(%tile_7_3, 2) {init = 1 : i32, sym_name = "sc7_Kp"}
    %sc7_Kc = aie.lock(%tile_7_3, 3) {init = 0 : i32, sym_name = "sc7_Kc"}
    %sc7_Ip = aie.lock(%tile_7_3, 4) {init = 1 : i32, sym_name = "sc7_Ip"}
    %sc7_Ic = aie.lock(%tile_7_3, 5) {init = 0 : i32, sym_name = "sc7_Ic"}
    %sc7_Ohp = aie.lock(%tile_7_3, 6) {init = 1 : i32, sym_name = "sc7_Ohp"}
    %sc7_Ohc = aie.lock(%tile_7_3, 7) {init = 0 : i32, sym_name = "sc7_Ohc"}
    %sc7_Ohdp = aie.lock(%tile_7_3, 8) {init = 1 : i32, sym_name = "sc7_Ohdp"}
    %sc7_Ohdc = aie.lock(%tile_7_3, 9) {init = 0 : i32, sym_name = "sc7_Ohdc"}
    %core_7_3 = aie.core(%tile_7_3) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      %c4_i32 = arith.constant 4 : i32
      %c64_i32 = arith.constant 64 : i32
      %c2 = arith.constant 2 : index
      %c128_i32 = arith.constant 128 : i32
      cf.br ^bb1(%c0 : index)
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb5
      %1 = arith.cmpi slt, %0, %c9223372036854775807 : index
      cf.cond_br %1, ^bb2, ^bb6
    ^bb2:  // pred: ^bb1
      func.call @flowkv_score_init_bf16(%c4_i32) : (i32) -> ()
      aie.use_lock(%sc7_Qc, AcquireGreaterEqual, 1)
      func.call @flowkv_score_rope_q_bf16(%sc7_Qs, %c4_i32, %c64_i32) : (memref<322xbf16>, i32, i32) -> ()
      cf.br ^bb3(%c0 : index)
    ^bb3(%2: index):  // 2 preds: ^bb2, ^bb4
      %3 = arith.cmpi slt, %2, %c2 : index
      cf.cond_br %3, ^bb4, ^bb5
    ^bb4:  // pred: ^bb3
      aie.use_lock(%sc7_Kc, AcquireGreaterEqual, 1)
      aie.use_lock(%sc7_Ip, AcquireGreaterEqual, 1)
      func.call @flowkv_score_chunk_bf16(%sc7_Qs, %sc7_K, %sc7_It, %c4_i32, %c64_i32, %c128_i32) : (memref<322xbf16>, memref<8192xbf16>, memref<520xbf16>, i32, i32, i32) -> ()
      aie.use_lock(%sc7_Kp, Release, 1)
      aie.use_lock(%sc7_Ic, Release, 1)
      %4 = arith.addi %2, %c1 : index
      cf.br ^bb3(%4 : index)
    ^bb5:  // pred: ^bb3
      aie.use_lock(%sc7_Qp, Release, 1)
      aie.use_lock(%sc7_Ohc, AcquireGreaterEqual, 1)
      aie.use_lock(%sc7_Ohdp, AcquireGreaterEqual, 1)
      aie.use_lock(%sc7_Ohp, Release, 1)
      aie.use_lock(%sc7_Ohdc, Release, 1)
      %5 = arith.addi %0, %c1 : index
      cf.br ^bb1(%5 : index)
    ^bb6:  // pred: ^bb1
      aie.end
    } {link_files = ["flowkv_64d_h4_c256.o"]}
    %mem_7_3 = aie.mem(%tile_7_3) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%sc7_Qp, AcquireGreaterEqual, 1)
      aie.dma_bd(%sc7_Qs : memref<322xbf16>, 0, 322) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%sc7_Qc, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%sc7_Ohp, AcquireGreaterEqual, 1)
      aie.dma_bd(%sc7_Oh : memref<256xbf16>, 0, 256) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%sc7_Ohc, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb5)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%sc7_Kp, AcquireGreaterEqual, 1)
      aie.dma_bd(%sc7_K : memref<8192xbf16>, 0, 8192) {bd_id = 2 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%sc7_Kc, Release, 1)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb3
      %2 = aie.dma_start(MM2S, 0, ^bb6, ^bb7)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      aie.use_lock(%sc7_Ic, AcquireGreaterEqual, 1)
      aie.dma_bd(%sc7_It : memref<520xbf16>, 0, 520) {bd_id = 3 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%sc7_Ip, Release, 1)
      aie.next_bd ^bb6
    ^bb7:  // pred: ^bb5
      %3 = aie.dma_start(MM2S, 1, ^bb8, ^bb9)
    ^bb8:  // 2 preds: ^bb7, ^bb8
      aie.use_lock(%sc7_Ohdc, AcquireGreaterEqual, 1)
      aie.dma_bd(%sc7_Oh : memref<256xbf16>, 0, 256) {bd_id = 4 : i32, next_bd_id = 4 : i32}
      aie.use_lock(%sc7_Ohdp, Release, 1)
      aie.next_bd ^bb8
    ^bb9:  // pred: ^bb7
      aie.end
    }
    %va0_Iv = aie.buffer(%tile_0_4) {address = 32768 : i32, mem_bank = 2 : i32, sym_name = "va0_Iv"} : memref<520xbf16> 
    %va0_V = aie.buffer(%tile_0_4) {address = 16384 : i32, mem_bank = 1 : i32, sym_name = "va0_V"} : memref<8192xbf16> 
    %va0_Of = aie.buffer(%tile_0_4) {address = 49152 : i32, mem_bank = 3 : i32, sym_name = "va0_Of"} : memref<256xbf16> 
    %va0_Ip = aie.lock(%tile_0_4, 0) {init = 1 : i32, sym_name = "va0_Ip"}
    %va0_Ic = aie.lock(%tile_0_4, 1) {init = 0 : i32, sym_name = "va0_Ic"}
    %va0_Vp = aie.lock(%tile_0_4, 2) {init = 1 : i32, sym_name = "va0_Vp"}
    %va0_Vc = aie.lock(%tile_0_4, 3) {init = 0 : i32, sym_name = "va0_Vc"}
    %va0_Op = aie.lock(%tile_0_4, 4) {init = 1 : i32, sym_name = "va0_Op"}
    %va0_Oc = aie.lock(%tile_0_4, 5) {init = 0 : i32, sym_name = "va0_Oc"}
    %core_0_4 = aie.core(%tile_0_4) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      %c4_i32 = arith.constant 4 : i32
      %c64_i32 = arith.constant 64 : i32
      %c2 = arith.constant 2 : index
      %c128_i32 = arith.constant 128 : i32
      cf.br ^bb1(%c0 : index)
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb5
      %1 = arith.cmpi slt, %0, %c9223372036854775807 : index
      cf.cond_br %1, ^bb2, ^bb6
    ^bb2:  // pred: ^bb1
      func.call @flowkv_value_init_bf16(%c4_i32, %c64_i32) : (i32, i32) -> ()
      cf.br ^bb3(%c0 : index)
    ^bb3(%2: index):  // 2 preds: ^bb2, ^bb4
      %3 = arith.cmpi slt, %2, %c2 : index
      cf.cond_br %3, ^bb4, ^bb5
    ^bb4:  // pred: ^bb3
      aie.use_lock(%va0_Ic, AcquireGreaterEqual, 1)
      aie.use_lock(%va0_Vc, AcquireGreaterEqual, 1)
      func.call @flowkv_value_accum_bf16(%va0_Iv, %va0_V, %c4_i32, %c64_i32, %c128_i32) : (memref<520xbf16>, memref<8192xbf16>, i32, i32, i32) -> ()
      aie.use_lock(%va0_Ip, Release, 1)
      aie.use_lock(%va0_Vp, Release, 1)
      %4 = arith.addi %2, %c1 : index
      cf.br ^bb3(%4 : index)
    ^bb5:  // pred: ^bb3
      aie.use_lock(%va0_Op, AcquireGreaterEqual, 1)
      func.call @flowkv_value_normalize_bf16(%va0_Of, %c4_i32, %c64_i32) : (memref<256xbf16>, i32, i32) -> ()
      aie.use_lock(%va0_Oc, Release, 1)
      %5 = arith.addi %0, %c1 : index
      cf.br ^bb1(%5 : index)
    ^bb6:  // pred: ^bb1
      aie.end
    } {link_files = ["flowkv_64d_h4_c256.o"]}
    %mem_0_4 = aie.mem(%tile_0_4) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb2)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%va0_Ip, AcquireGreaterEqual, 1)
      aie.dma_bd(%va0_Iv : memref<520xbf16>, 0, 520) {bd_id = 0 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%va0_Ic, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb3, ^bb4)
    ^bb3:  // 2 preds: ^bb2, ^bb3
      aie.use_lock(%va0_Vp, AcquireGreaterEqual, 1)
      aie.dma_bd(%va0_V : memref<8192xbf16>, 0, 8192) {bd_id = 1 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%va0_Vc, Release, 1)
      aie.next_bd ^bb3
    ^bb4:  // pred: ^bb2
      %2 = aie.dma_start(MM2S, 0, ^bb5, ^bb6)
    ^bb5:  // 2 preds: ^bb4, ^bb5
      aie.use_lock(%va0_Oc, AcquireGreaterEqual, 1)
      aie.dma_bd(%va0_Of : memref<256xbf16>, 0, 256) {bd_id = 2 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%va0_Op, Release, 1)
      aie.next_bd ^bb5
    ^bb6:  // pred: ^bb4
      aie.end
    }
    %va1_Iv = aie.buffer(%tile_1_4) {address = 32768 : i32, mem_bank = 2 : i32, sym_name = "va1_Iv"} : memref<520xbf16> 
    %va1_V = aie.buffer(%tile_1_4) {address = 16384 : i32, mem_bank = 1 : i32, sym_name = "va1_V"} : memref<8192xbf16> 
    %va1_Of = aie.buffer(%tile_1_4) {address = 49152 : i32, mem_bank = 3 : i32, sym_name = "va1_Of"} : memref<256xbf16> 
    %va1_Ip = aie.lock(%tile_1_4, 0) {init = 1 : i32, sym_name = "va1_Ip"}
    %va1_Ic = aie.lock(%tile_1_4, 1) {init = 0 : i32, sym_name = "va1_Ic"}
    %va1_Vp = aie.lock(%tile_1_4, 2) {init = 1 : i32, sym_name = "va1_Vp"}
    %va1_Vc = aie.lock(%tile_1_4, 3) {init = 0 : i32, sym_name = "va1_Vc"}
    %va1_Op = aie.lock(%tile_1_4, 4) {init = 1 : i32, sym_name = "va1_Op"}
    %va1_Oc = aie.lock(%tile_1_4, 5) {init = 0 : i32, sym_name = "va1_Oc"}
    %core_1_4 = aie.core(%tile_1_4) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      %c4_i32 = arith.constant 4 : i32
      %c64_i32 = arith.constant 64 : i32
      %c2 = arith.constant 2 : index
      %c128_i32 = arith.constant 128 : i32
      cf.br ^bb1(%c0 : index)
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb5
      %1 = arith.cmpi slt, %0, %c9223372036854775807 : index
      cf.cond_br %1, ^bb2, ^bb6
    ^bb2:  // pred: ^bb1
      func.call @flowkv_value_init_bf16(%c4_i32, %c64_i32) : (i32, i32) -> ()
      cf.br ^bb3(%c0 : index)
    ^bb3(%2: index):  // 2 preds: ^bb2, ^bb4
      %3 = arith.cmpi slt, %2, %c2 : index
      cf.cond_br %3, ^bb4, ^bb5
    ^bb4:  // pred: ^bb3
      aie.use_lock(%va1_Ic, AcquireGreaterEqual, 1)
      aie.use_lock(%va1_Vc, AcquireGreaterEqual, 1)
      func.call @flowkv_value_accum_bf16(%va1_Iv, %va1_V, %c4_i32, %c64_i32, %c128_i32) : (memref<520xbf16>, memref<8192xbf16>, i32, i32, i32) -> ()
      aie.use_lock(%va1_Ip, Release, 1)
      aie.use_lock(%va1_Vp, Release, 1)
      %4 = arith.addi %2, %c1 : index
      cf.br ^bb3(%4 : index)
    ^bb5:  // pred: ^bb3
      aie.use_lock(%va1_Op, AcquireGreaterEqual, 1)
      func.call @flowkv_value_normalize_bf16(%va1_Of, %c4_i32, %c64_i32) : (memref<256xbf16>, i32, i32) -> ()
      aie.use_lock(%va1_Oc, Release, 1)
      %5 = arith.addi %0, %c1 : index
      cf.br ^bb1(%5 : index)
    ^bb6:  // pred: ^bb1
      aie.end
    } {link_files = ["flowkv_64d_h4_c256.o"]}
    %mem_1_4 = aie.mem(%tile_1_4) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb2)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%va1_Ip, AcquireGreaterEqual, 1)
      aie.dma_bd(%va1_Iv : memref<520xbf16>, 0, 520) {bd_id = 0 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%va1_Ic, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb3, ^bb4)
    ^bb3:  // 2 preds: ^bb2, ^bb3
      aie.use_lock(%va1_Vp, AcquireGreaterEqual, 1)
      aie.dma_bd(%va1_V : memref<8192xbf16>, 0, 8192) {bd_id = 1 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%va1_Vc, Release, 1)
      aie.next_bd ^bb3
    ^bb4:  // pred: ^bb2
      %2 = aie.dma_start(MM2S, 0, ^bb5, ^bb6)
    ^bb5:  // 2 preds: ^bb4, ^bb5
      aie.use_lock(%va1_Oc, AcquireGreaterEqual, 1)
      aie.dma_bd(%va1_Of : memref<256xbf16>, 0, 256) {bd_id = 2 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%va1_Op, Release, 1)
      aie.next_bd ^bb5
    ^bb6:  // pred: ^bb4
      aie.end
    }
    %va2_Iv = aie.buffer(%tile_6_4) {address = 32768 : i32, mem_bank = 2 : i32, sym_name = "va2_Iv"} : memref<520xbf16> 
    %va2_V = aie.buffer(%tile_6_4) {address = 16384 : i32, mem_bank = 1 : i32, sym_name = "va2_V"} : memref<8192xbf16> 
    %va2_Of = aie.buffer(%tile_6_4) {address = 49152 : i32, mem_bank = 3 : i32, sym_name = "va2_Of"} : memref<256xbf16> 
    %va2_Ip = aie.lock(%tile_6_4, 0) {init = 1 : i32, sym_name = "va2_Ip"}
    %va2_Ic = aie.lock(%tile_6_4, 1) {init = 0 : i32, sym_name = "va2_Ic"}
    %va2_Vp = aie.lock(%tile_6_4, 2) {init = 1 : i32, sym_name = "va2_Vp"}
    %va2_Vc = aie.lock(%tile_6_4, 3) {init = 0 : i32, sym_name = "va2_Vc"}
    %va2_Op = aie.lock(%tile_6_4, 4) {init = 1 : i32, sym_name = "va2_Op"}
    %va2_Oc = aie.lock(%tile_6_4, 5) {init = 0 : i32, sym_name = "va2_Oc"}
    %core_6_4 = aie.core(%tile_6_4) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      %c4_i32 = arith.constant 4 : i32
      %c64_i32 = arith.constant 64 : i32
      %c2 = arith.constant 2 : index
      %c128_i32 = arith.constant 128 : i32
      cf.br ^bb1(%c0 : index)
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb5
      %1 = arith.cmpi slt, %0, %c9223372036854775807 : index
      cf.cond_br %1, ^bb2, ^bb6
    ^bb2:  // pred: ^bb1
      func.call @flowkv_value_init_bf16(%c4_i32, %c64_i32) : (i32, i32) -> ()
      cf.br ^bb3(%c0 : index)
    ^bb3(%2: index):  // 2 preds: ^bb2, ^bb4
      %3 = arith.cmpi slt, %2, %c2 : index
      cf.cond_br %3, ^bb4, ^bb5
    ^bb4:  // pred: ^bb3
      aie.use_lock(%va2_Ic, AcquireGreaterEqual, 1)
      aie.use_lock(%va2_Vc, AcquireGreaterEqual, 1)
      func.call @flowkv_value_accum_bf16(%va2_Iv, %va2_V, %c4_i32, %c64_i32, %c128_i32) : (memref<520xbf16>, memref<8192xbf16>, i32, i32, i32) -> ()
      aie.use_lock(%va2_Ip, Release, 1)
      aie.use_lock(%va2_Vp, Release, 1)
      %4 = arith.addi %2, %c1 : index
      cf.br ^bb3(%4 : index)
    ^bb5:  // pred: ^bb3
      aie.use_lock(%va2_Op, AcquireGreaterEqual, 1)
      func.call @flowkv_value_normalize_bf16(%va2_Of, %c4_i32, %c64_i32) : (memref<256xbf16>, i32, i32) -> ()
      aie.use_lock(%va2_Oc, Release, 1)
      %5 = arith.addi %0, %c1 : index
      cf.br ^bb1(%5 : index)
    ^bb6:  // pred: ^bb1
      aie.end
    } {link_files = ["flowkv_64d_h4_c256.o"]}
    %mem_6_4 = aie.mem(%tile_6_4) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb2)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%va2_Ip, AcquireGreaterEqual, 1)
      aie.dma_bd(%va2_Iv : memref<520xbf16>, 0, 520) {bd_id = 0 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%va2_Ic, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb3, ^bb4)
    ^bb3:  // 2 preds: ^bb2, ^bb3
      aie.use_lock(%va2_Vp, AcquireGreaterEqual, 1)
      aie.dma_bd(%va2_V : memref<8192xbf16>, 0, 8192) {bd_id = 1 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%va2_Vc, Release, 1)
      aie.next_bd ^bb3
    ^bb4:  // pred: ^bb2
      %2 = aie.dma_start(MM2S, 0, ^bb5, ^bb6)
    ^bb5:  // 2 preds: ^bb4, ^bb5
      aie.use_lock(%va2_Oc, AcquireGreaterEqual, 1)
      aie.dma_bd(%va2_Of : memref<256xbf16>, 0, 256) {bd_id = 2 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%va2_Op, Release, 1)
      aie.next_bd ^bb5
    ^bb6:  // pred: ^bb4
      aie.end
    }
    %va3_Iv = aie.buffer(%tile_7_4) {address = 32768 : i32, mem_bank = 2 : i32, sym_name = "va3_Iv"} : memref<520xbf16> 
    %va3_V = aie.buffer(%tile_7_4) {address = 16384 : i32, mem_bank = 1 : i32, sym_name = "va3_V"} : memref<8192xbf16> 
    %va3_Of = aie.buffer(%tile_7_4) {address = 49152 : i32, mem_bank = 3 : i32, sym_name = "va3_Of"} : memref<256xbf16> 
    %va3_Ip = aie.lock(%tile_7_4, 0) {init = 1 : i32, sym_name = "va3_Ip"}
    %va3_Ic = aie.lock(%tile_7_4, 1) {init = 0 : i32, sym_name = "va3_Ic"}
    %va3_Vp = aie.lock(%tile_7_4, 2) {init = 1 : i32, sym_name = "va3_Vp"}
    %va3_Vc = aie.lock(%tile_7_4, 3) {init = 0 : i32, sym_name = "va3_Vc"}
    %va3_Op = aie.lock(%tile_7_4, 4) {init = 1 : i32, sym_name = "va3_Op"}
    %va3_Oc = aie.lock(%tile_7_4, 5) {init = 0 : i32, sym_name = "va3_Oc"}
    %core_7_4 = aie.core(%tile_7_4) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      %c4_i32 = arith.constant 4 : i32
      %c64_i32 = arith.constant 64 : i32
      %c2 = arith.constant 2 : index
      %c128_i32 = arith.constant 128 : i32
      cf.br ^bb1(%c0 : index)
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb5
      %1 = arith.cmpi slt, %0, %c9223372036854775807 : index
      cf.cond_br %1, ^bb2, ^bb6
    ^bb2:  // pred: ^bb1
      func.call @flowkv_value_init_bf16(%c4_i32, %c64_i32) : (i32, i32) -> ()
      cf.br ^bb3(%c0 : index)
    ^bb3(%2: index):  // 2 preds: ^bb2, ^bb4
      %3 = arith.cmpi slt, %2, %c2 : index
      cf.cond_br %3, ^bb4, ^bb5
    ^bb4:  // pred: ^bb3
      aie.use_lock(%va3_Ic, AcquireGreaterEqual, 1)
      aie.use_lock(%va3_Vc, AcquireGreaterEqual, 1)
      func.call @flowkv_value_accum_bf16(%va3_Iv, %va3_V, %c4_i32, %c64_i32, %c128_i32) : (memref<520xbf16>, memref<8192xbf16>, i32, i32, i32) -> ()
      aie.use_lock(%va3_Ip, Release, 1)
      aie.use_lock(%va3_Vp, Release, 1)
      %4 = arith.addi %2, %c1 : index
      cf.br ^bb3(%4 : index)
    ^bb5:  // pred: ^bb3
      aie.use_lock(%va3_Op, AcquireGreaterEqual, 1)
      func.call @flowkv_value_normalize_bf16(%va3_Of, %c4_i32, %c64_i32) : (memref<256xbf16>, i32, i32) -> ()
      aie.use_lock(%va3_Oc, Release, 1)
      %5 = arith.addi %0, %c1 : index
      cf.br ^bb1(%5 : index)
    ^bb6:  // pred: ^bb1
      aie.end
    } {link_files = ["flowkv_64d_h4_c256.o"]}
    %mem_7_4 = aie.mem(%tile_7_4) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb2)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%va3_Ip, AcquireGreaterEqual, 1)
      aie.dma_bd(%va3_Iv : memref<520xbf16>, 0, 520) {bd_id = 0 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%va3_Ic, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb3, ^bb4)
    ^bb3:  // 2 preds: ^bb2, ^bb3
      aie.use_lock(%va3_Vp, AcquireGreaterEqual, 1)
      aie.dma_bd(%va3_V : memref<8192xbf16>, 0, 8192) {bd_id = 1 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%va3_Vc, Release, 1)
      aie.next_bd ^bb3
    ^bb4:  // pred: ^bb2
      %2 = aie.dma_start(MM2S, 0, ^bb5, ^bb6)
    ^bb5:  // 2 preds: ^bb4, ^bb5
      aie.use_lock(%va3_Oc, AcquireGreaterEqual, 1)
      aie.dma_bd(%va3_Of : memref<256xbf16>, 0, 256) {bd_id = 2 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%va3_Op, Release, 1)
      aie.next_bd ^bb5
    ^bb6:  // pred: ^bb4
      aie.end
    }
    %va4_Iv = aie.buffer(%tile_0_5) {address = 32768 : i32, mem_bank = 2 : i32, sym_name = "va4_Iv"} : memref<520xbf16> 
    %va4_V = aie.buffer(%tile_0_5) {address = 16384 : i32, mem_bank = 1 : i32, sym_name = "va4_V"} : memref<8192xbf16> 
    %va4_Of = aie.buffer(%tile_0_5) {address = 49152 : i32, mem_bank = 3 : i32, sym_name = "va4_Of"} : memref<256xbf16> 
    %va4_Ip = aie.lock(%tile_0_5, 0) {init = 1 : i32, sym_name = "va4_Ip"}
    %va4_Ic = aie.lock(%tile_0_5, 1) {init = 0 : i32, sym_name = "va4_Ic"}
    %va4_Vp = aie.lock(%tile_0_5, 2) {init = 1 : i32, sym_name = "va4_Vp"}
    %va4_Vc = aie.lock(%tile_0_5, 3) {init = 0 : i32, sym_name = "va4_Vc"}
    %va4_Op = aie.lock(%tile_0_5, 4) {init = 1 : i32, sym_name = "va4_Op"}
    %va4_Oc = aie.lock(%tile_0_5, 5) {init = 0 : i32, sym_name = "va4_Oc"}
    %core_0_5 = aie.core(%tile_0_5) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      %c4_i32 = arith.constant 4 : i32
      %c64_i32 = arith.constant 64 : i32
      %c2 = arith.constant 2 : index
      %c128_i32 = arith.constant 128 : i32
      cf.br ^bb1(%c0 : index)
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb5
      %1 = arith.cmpi slt, %0, %c9223372036854775807 : index
      cf.cond_br %1, ^bb2, ^bb6
    ^bb2:  // pred: ^bb1
      func.call @flowkv_value_init_bf16(%c4_i32, %c64_i32) : (i32, i32) -> ()
      cf.br ^bb3(%c0 : index)
    ^bb3(%2: index):  // 2 preds: ^bb2, ^bb4
      %3 = arith.cmpi slt, %2, %c2 : index
      cf.cond_br %3, ^bb4, ^bb5
    ^bb4:  // pred: ^bb3
      aie.use_lock(%va4_Ic, AcquireGreaterEqual, 1)
      aie.use_lock(%va4_Vc, AcquireGreaterEqual, 1)
      func.call @flowkv_value_accum_bf16(%va4_Iv, %va4_V, %c4_i32, %c64_i32, %c128_i32) : (memref<520xbf16>, memref<8192xbf16>, i32, i32, i32) -> ()
      aie.use_lock(%va4_Ip, Release, 1)
      aie.use_lock(%va4_Vp, Release, 1)
      %4 = arith.addi %2, %c1 : index
      cf.br ^bb3(%4 : index)
    ^bb5:  // pred: ^bb3
      aie.use_lock(%va4_Op, AcquireGreaterEqual, 1)
      func.call @flowkv_value_normalize_bf16(%va4_Of, %c4_i32, %c64_i32) : (memref<256xbf16>, i32, i32) -> ()
      aie.use_lock(%va4_Oc, Release, 1)
      %5 = arith.addi %0, %c1 : index
      cf.br ^bb1(%5 : index)
    ^bb6:  // pred: ^bb1
      aie.end
    } {link_files = ["flowkv_64d_h4_c256.o"]}
    %mem_0_5 = aie.mem(%tile_0_5) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb2)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%va4_Ip, AcquireGreaterEqual, 1)
      aie.dma_bd(%va4_Iv : memref<520xbf16>, 0, 520) {bd_id = 0 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%va4_Ic, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb3, ^bb4)
    ^bb3:  // 2 preds: ^bb2, ^bb3
      aie.use_lock(%va4_Vp, AcquireGreaterEqual, 1)
      aie.dma_bd(%va4_V : memref<8192xbf16>, 0, 8192) {bd_id = 1 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%va4_Vc, Release, 1)
      aie.next_bd ^bb3
    ^bb4:  // pred: ^bb2
      %2 = aie.dma_start(MM2S, 0, ^bb5, ^bb6)
    ^bb5:  // 2 preds: ^bb4, ^bb5
      aie.use_lock(%va4_Oc, AcquireGreaterEqual, 1)
      aie.dma_bd(%va4_Of : memref<256xbf16>, 0, 256) {bd_id = 2 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%va4_Op, Release, 1)
      aie.next_bd ^bb5
    ^bb6:  // pred: ^bb4
      aie.end
    }
    %va5_Iv = aie.buffer(%tile_1_5) {address = 32768 : i32, mem_bank = 2 : i32, sym_name = "va5_Iv"} : memref<520xbf16> 
    %va5_V = aie.buffer(%tile_1_5) {address = 16384 : i32, mem_bank = 1 : i32, sym_name = "va5_V"} : memref<8192xbf16> 
    %va5_Of = aie.buffer(%tile_1_5) {address = 49152 : i32, mem_bank = 3 : i32, sym_name = "va5_Of"} : memref<256xbf16> 
    %va5_Ip = aie.lock(%tile_1_5, 0) {init = 1 : i32, sym_name = "va5_Ip"}
    %va5_Ic = aie.lock(%tile_1_5, 1) {init = 0 : i32, sym_name = "va5_Ic"}
    %va5_Vp = aie.lock(%tile_1_5, 2) {init = 1 : i32, sym_name = "va5_Vp"}
    %va5_Vc = aie.lock(%tile_1_5, 3) {init = 0 : i32, sym_name = "va5_Vc"}
    %va5_Op = aie.lock(%tile_1_5, 4) {init = 1 : i32, sym_name = "va5_Op"}
    %va5_Oc = aie.lock(%tile_1_5, 5) {init = 0 : i32, sym_name = "va5_Oc"}
    %core_1_5 = aie.core(%tile_1_5) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      %c4_i32 = arith.constant 4 : i32
      %c64_i32 = arith.constant 64 : i32
      %c2 = arith.constant 2 : index
      %c128_i32 = arith.constant 128 : i32
      cf.br ^bb1(%c0 : index)
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb5
      %1 = arith.cmpi slt, %0, %c9223372036854775807 : index
      cf.cond_br %1, ^bb2, ^bb6
    ^bb2:  // pred: ^bb1
      func.call @flowkv_value_init_bf16(%c4_i32, %c64_i32) : (i32, i32) -> ()
      cf.br ^bb3(%c0 : index)
    ^bb3(%2: index):  // 2 preds: ^bb2, ^bb4
      %3 = arith.cmpi slt, %2, %c2 : index
      cf.cond_br %3, ^bb4, ^bb5
    ^bb4:  // pred: ^bb3
      aie.use_lock(%va5_Ic, AcquireGreaterEqual, 1)
      aie.use_lock(%va5_Vc, AcquireGreaterEqual, 1)
      func.call @flowkv_value_accum_bf16(%va5_Iv, %va5_V, %c4_i32, %c64_i32, %c128_i32) : (memref<520xbf16>, memref<8192xbf16>, i32, i32, i32) -> ()
      aie.use_lock(%va5_Ip, Release, 1)
      aie.use_lock(%va5_Vp, Release, 1)
      %4 = arith.addi %2, %c1 : index
      cf.br ^bb3(%4 : index)
    ^bb5:  // pred: ^bb3
      aie.use_lock(%va5_Op, AcquireGreaterEqual, 1)
      func.call @flowkv_value_normalize_bf16(%va5_Of, %c4_i32, %c64_i32) : (memref<256xbf16>, i32, i32) -> ()
      aie.use_lock(%va5_Oc, Release, 1)
      %5 = arith.addi %0, %c1 : index
      cf.br ^bb1(%5 : index)
    ^bb6:  // pred: ^bb1
      aie.end
    } {link_files = ["flowkv_64d_h4_c256.o"]}
    %mem_1_5 = aie.mem(%tile_1_5) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb2)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%va5_Ip, AcquireGreaterEqual, 1)
      aie.dma_bd(%va5_Iv : memref<520xbf16>, 0, 520) {bd_id = 0 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%va5_Ic, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb3, ^bb4)
    ^bb3:  // 2 preds: ^bb2, ^bb3
      aie.use_lock(%va5_Vp, AcquireGreaterEqual, 1)
      aie.dma_bd(%va5_V : memref<8192xbf16>, 0, 8192) {bd_id = 1 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%va5_Vc, Release, 1)
      aie.next_bd ^bb3
    ^bb4:  // pred: ^bb2
      %2 = aie.dma_start(MM2S, 0, ^bb5, ^bb6)
    ^bb5:  // 2 preds: ^bb4, ^bb5
      aie.use_lock(%va5_Oc, AcquireGreaterEqual, 1)
      aie.dma_bd(%va5_Of : memref<256xbf16>, 0, 256) {bd_id = 2 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%va5_Op, Release, 1)
      aie.next_bd ^bb5
    ^bb6:  // pred: ^bb4
      aie.end
    }
    %va6_Iv = aie.buffer(%tile_6_5) {address = 32768 : i32, mem_bank = 2 : i32, sym_name = "va6_Iv"} : memref<520xbf16> 
    %va6_V = aie.buffer(%tile_6_5) {address = 16384 : i32, mem_bank = 1 : i32, sym_name = "va6_V"} : memref<8192xbf16> 
    %va6_Of = aie.buffer(%tile_6_5) {address = 49152 : i32, mem_bank = 3 : i32, sym_name = "va6_Of"} : memref<256xbf16> 
    %va6_Ip = aie.lock(%tile_6_5, 0) {init = 1 : i32, sym_name = "va6_Ip"}
    %va6_Ic = aie.lock(%tile_6_5, 1) {init = 0 : i32, sym_name = "va6_Ic"}
    %va6_Vp = aie.lock(%tile_6_5, 2) {init = 1 : i32, sym_name = "va6_Vp"}
    %va6_Vc = aie.lock(%tile_6_5, 3) {init = 0 : i32, sym_name = "va6_Vc"}
    %va6_Op = aie.lock(%tile_6_5, 4) {init = 1 : i32, sym_name = "va6_Op"}
    %va6_Oc = aie.lock(%tile_6_5, 5) {init = 0 : i32, sym_name = "va6_Oc"}
    %core_6_5 = aie.core(%tile_6_5) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      %c4_i32 = arith.constant 4 : i32
      %c64_i32 = arith.constant 64 : i32
      %c2 = arith.constant 2 : index
      %c128_i32 = arith.constant 128 : i32
      cf.br ^bb1(%c0 : index)
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb5
      %1 = arith.cmpi slt, %0, %c9223372036854775807 : index
      cf.cond_br %1, ^bb2, ^bb6
    ^bb2:  // pred: ^bb1
      func.call @flowkv_value_init_bf16(%c4_i32, %c64_i32) : (i32, i32) -> ()
      cf.br ^bb3(%c0 : index)
    ^bb3(%2: index):  // 2 preds: ^bb2, ^bb4
      %3 = arith.cmpi slt, %2, %c2 : index
      cf.cond_br %3, ^bb4, ^bb5
    ^bb4:  // pred: ^bb3
      aie.use_lock(%va6_Ic, AcquireGreaterEqual, 1)
      aie.use_lock(%va6_Vc, AcquireGreaterEqual, 1)
      func.call @flowkv_value_accum_bf16(%va6_Iv, %va6_V, %c4_i32, %c64_i32, %c128_i32) : (memref<520xbf16>, memref<8192xbf16>, i32, i32, i32) -> ()
      aie.use_lock(%va6_Ip, Release, 1)
      aie.use_lock(%va6_Vp, Release, 1)
      %4 = arith.addi %2, %c1 : index
      cf.br ^bb3(%4 : index)
    ^bb5:  // pred: ^bb3
      aie.use_lock(%va6_Op, AcquireGreaterEqual, 1)
      func.call @flowkv_value_normalize_bf16(%va6_Of, %c4_i32, %c64_i32) : (memref<256xbf16>, i32, i32) -> ()
      aie.use_lock(%va6_Oc, Release, 1)
      %5 = arith.addi %0, %c1 : index
      cf.br ^bb1(%5 : index)
    ^bb6:  // pred: ^bb1
      aie.end
    } {link_files = ["flowkv_64d_h4_c256.o"]}
    %mem_6_5 = aie.mem(%tile_6_5) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb2)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%va6_Ip, AcquireGreaterEqual, 1)
      aie.dma_bd(%va6_Iv : memref<520xbf16>, 0, 520) {bd_id = 0 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%va6_Ic, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb3, ^bb4)
    ^bb3:  // 2 preds: ^bb2, ^bb3
      aie.use_lock(%va6_Vp, AcquireGreaterEqual, 1)
      aie.dma_bd(%va6_V : memref<8192xbf16>, 0, 8192) {bd_id = 1 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%va6_Vc, Release, 1)
      aie.next_bd ^bb3
    ^bb4:  // pred: ^bb2
      %2 = aie.dma_start(MM2S, 0, ^bb5, ^bb6)
    ^bb5:  // 2 preds: ^bb4, ^bb5
      aie.use_lock(%va6_Oc, AcquireGreaterEqual, 1)
      aie.dma_bd(%va6_Of : memref<256xbf16>, 0, 256) {bd_id = 2 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%va6_Op, Release, 1)
      aie.next_bd ^bb5
    ^bb6:  // pred: ^bb4
      aie.end
    }
    %va7_Iv = aie.buffer(%tile_7_5) {address = 32768 : i32, mem_bank = 2 : i32, sym_name = "va7_Iv"} : memref<520xbf16> 
    %va7_V = aie.buffer(%tile_7_5) {address = 16384 : i32, mem_bank = 1 : i32, sym_name = "va7_V"} : memref<8192xbf16> 
    %va7_Of = aie.buffer(%tile_7_5) {address = 49152 : i32, mem_bank = 3 : i32, sym_name = "va7_Of"} : memref<256xbf16> 
    %va7_Ip = aie.lock(%tile_7_5, 0) {init = 1 : i32, sym_name = "va7_Ip"}
    %va7_Ic = aie.lock(%tile_7_5, 1) {init = 0 : i32, sym_name = "va7_Ic"}
    %va7_Vp = aie.lock(%tile_7_5, 2) {init = 1 : i32, sym_name = "va7_Vp"}
    %va7_Vc = aie.lock(%tile_7_5, 3) {init = 0 : i32, sym_name = "va7_Vc"}
    %va7_Op = aie.lock(%tile_7_5, 4) {init = 1 : i32, sym_name = "va7_Op"}
    %va7_Oc = aie.lock(%tile_7_5, 5) {init = 0 : i32, sym_name = "va7_Oc"}
    %core_7_5 = aie.core(%tile_7_5) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      %c4_i32 = arith.constant 4 : i32
      %c64_i32 = arith.constant 64 : i32
      %c2 = arith.constant 2 : index
      %c128_i32 = arith.constant 128 : i32
      cf.br ^bb1(%c0 : index)
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb5
      %1 = arith.cmpi slt, %0, %c9223372036854775807 : index
      cf.cond_br %1, ^bb2, ^bb6
    ^bb2:  // pred: ^bb1
      func.call @flowkv_value_init_bf16(%c4_i32, %c64_i32) : (i32, i32) -> ()
      cf.br ^bb3(%c0 : index)
    ^bb3(%2: index):  // 2 preds: ^bb2, ^bb4
      %3 = arith.cmpi slt, %2, %c2 : index
      cf.cond_br %3, ^bb4, ^bb5
    ^bb4:  // pred: ^bb3
      aie.use_lock(%va7_Ic, AcquireGreaterEqual, 1)
      aie.use_lock(%va7_Vc, AcquireGreaterEqual, 1)
      func.call @flowkv_value_accum_bf16(%va7_Iv, %va7_V, %c4_i32, %c64_i32, %c128_i32) : (memref<520xbf16>, memref<8192xbf16>, i32, i32, i32) -> ()
      aie.use_lock(%va7_Ip, Release, 1)
      aie.use_lock(%va7_Vp, Release, 1)
      %4 = arith.addi %2, %c1 : index
      cf.br ^bb3(%4 : index)
    ^bb5:  // pred: ^bb3
      aie.use_lock(%va7_Op, AcquireGreaterEqual, 1)
      func.call @flowkv_value_normalize_bf16(%va7_Of, %c4_i32, %c64_i32) : (memref<256xbf16>, i32, i32) -> ()
      aie.use_lock(%va7_Oc, Release, 1)
      %5 = arith.addi %0, %c1 : index
      cf.br ^bb1(%5 : index)
    ^bb6:  // pred: ^bb1
      aie.end
    } {link_files = ["flowkv_64d_h4_c256.o"]}
    %mem_7_5 = aie.mem(%tile_7_5) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb2)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%va7_Ip, AcquireGreaterEqual, 1)
      aie.dma_bd(%va7_Iv : memref<520xbf16>, 0, 520) {bd_id = 0 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%va7_Ic, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb3, ^bb4)
    ^bb3:  // 2 preds: ^bb2, ^bb3
      aie.use_lock(%va7_Vp, AcquireGreaterEqual, 1)
      aie.dma_bd(%va7_V : memref<8192xbf16>, 0, 8192) {bd_id = 1 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%va7_Vc, Release, 1)
      aie.next_bd ^bb3
    ^bb4:  // pred: ^bb2
      %2 = aie.dma_start(MM2S, 0, ^bb5, ^bb6)
    ^bb5:  // 2 preds: ^bb4, ^bb5
      aie.use_lock(%va7_Oc, AcquireGreaterEqual, 1)
      aie.dma_bd(%va7_Of : memref<256xbf16>, 0, 256) {bd_id = 2 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%va7_Op, Release, 1)
      aie.next_bd ^bb5
    ^bb6:  // pred: ^bb4
      aie.end
    }
    %jA_buf = aie.buffer(%mem_tile_2_1) {address = 0 : i32, mem_bank = 0 : i32, sym_name = "jA_buf"} : memref<2048xbf16> 
    %jA_p0 = aie.lock(%mem_tile_2_1, 0) {init = 1 : i32, sym_name = "jA_p0"}
    %jA_c0 = aie.lock(%mem_tile_2_1, 1) {init = 0 : i32, sym_name = "jA_c0"}
    %jA_p1 = aie.lock(%mem_tile_2_1, 2) {init = 1 : i32, sym_name = "jA_p1"}
    %jA_c1 = aie.lock(%mem_tile_2_1, 3) {init = 0 : i32, sym_name = "jA_c1"}
    %jA_p2 = aie.lock(%mem_tile_2_1, 4) {init = 1 : i32, sym_name = "jA_p2"}
    %jA_c2 = aie.lock(%mem_tile_2_1, 5) {init = 0 : i32, sym_name = "jA_c2"}
    %jA_p3 = aie.lock(%mem_tile_2_1, 6) {init = 1 : i32, sym_name = "jA_p3"}
    %jA_c3 = aie.lock(%mem_tile_2_1, 7) {init = 0 : i32, sym_name = "jA_c3"}
    %memtile_dma_2_1 = aie.memtile_dma(%mem_tile_2_1) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb2)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%jA_p0, AcquireGreaterEqual, 1)
      aie.dma_bd(%jA_buf : memref<2048xbf16>, 0, 256) {bd_id = 0 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%jA_c0, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb3, ^bb4)
    ^bb3:  // 2 preds: ^bb2, ^bb3
      aie.use_lock(%jA_p1, AcquireGreaterEqual, 1)
      aie.dma_bd(%jA_buf : memref<2048xbf16>, 256, 256) {bd_id = 24 : i32, next_bd_id = 24 : i32}
      aie.use_lock(%jA_c1, Release, 1)
      aie.next_bd ^bb3
    ^bb4:  // pred: ^bb2
      %2 = aie.dma_start(S2MM, 2, ^bb5, ^bb6)
    ^bb5:  // 2 preds: ^bb4, ^bb5
      aie.use_lock(%jA_p2, AcquireGreaterEqual, 1)
      aie.dma_bd(%jA_buf : memref<2048xbf16>, 512, 256) {bd_id = 1 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%jA_c2, Release, 1)
      aie.next_bd ^bb5
    ^bb6:  // pred: ^bb4
      %3 = aie.dma_start(S2MM, 3, ^bb7, ^bb8)
    ^bb7:  // 2 preds: ^bb6, ^bb7
      aie.use_lock(%jA_p3, AcquireGreaterEqual, 1)
      aie.dma_bd(%jA_buf : memref<2048xbf16>, 768, 256) {bd_id = 25 : i32, next_bd_id = 25 : i32}
      aie.use_lock(%jA_c3, Release, 1)
      aie.next_bd ^bb7
    ^bb8:  // pred: ^bb6
      %4 = aie.dma_start(MM2S, 0, ^bb9, ^bb13)
    ^bb9:  // 2 preds: ^bb8, ^bb12
      aie.use_lock(%jA_c0, AcquireGreaterEqual, 1)
      aie.dma_bd(%jA_buf : memref<2048xbf16>, 0, 256) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%jA_p0, Release, 1)
      aie.next_bd ^bb10
    ^bb10:  // pred: ^bb9
      aie.use_lock(%jA_c1, AcquireGreaterEqual, 1)
      aie.dma_bd(%jA_buf : memref<2048xbf16>, 256, 256) {bd_id = 3 : i32, next_bd_id = 4 : i32}
      aie.use_lock(%jA_p1, Release, 1)
      aie.next_bd ^bb11
    ^bb11:  // pred: ^bb10
      aie.use_lock(%jA_c2, AcquireGreaterEqual, 1)
      aie.dma_bd(%jA_buf : memref<2048xbf16>, 512, 256) {bd_id = 4 : i32, next_bd_id = 5 : i32}
      aie.use_lock(%jA_p2, Release, 1)
      aie.next_bd ^bb12
    ^bb12:  // pred: ^bb11
      aie.use_lock(%jA_c3, AcquireGreaterEqual, 1)
      aie.dma_bd(%jA_buf : memref<2048xbf16>, 768, 256) {bd_id = 5 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%jA_p3, Release, 1)
      aie.next_bd ^bb9
    ^bb13:  // pred: ^bb8
      aie.end
    }
    %jB_buf = aie.buffer(%mem_tile_3_1) {address = 0 : i32, mem_bank = 0 : i32, sym_name = "jB_buf"} : memref<2048xbf16> 
    %jB_p0 = aie.lock(%mem_tile_3_1, 0) {init = 1 : i32, sym_name = "jB_p0"}
    %jB_c0 = aie.lock(%mem_tile_3_1, 1) {init = 0 : i32, sym_name = "jB_c0"}
    %jB_p1 = aie.lock(%mem_tile_3_1, 2) {init = 1 : i32, sym_name = "jB_p1"}
    %jB_c1 = aie.lock(%mem_tile_3_1, 3) {init = 0 : i32, sym_name = "jB_c1"}
    %jB_p2 = aie.lock(%mem_tile_3_1, 4) {init = 1 : i32, sym_name = "jB_p2"}
    %jB_c2 = aie.lock(%mem_tile_3_1, 5) {init = 0 : i32, sym_name = "jB_c2"}
    %jB_p3 = aie.lock(%mem_tile_3_1, 6) {init = 1 : i32, sym_name = "jB_p3"}
    %jB_c3 = aie.lock(%mem_tile_3_1, 7) {init = 0 : i32, sym_name = "jB_c3"}
    %memtile_dma_3_1 = aie.memtile_dma(%mem_tile_3_1) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb2)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%jB_p0, AcquireGreaterEqual, 1)
      aie.dma_bd(%jB_buf : memref<2048xbf16>, 0, 256) {bd_id = 0 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%jB_c0, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb3, ^bb4)
    ^bb3:  // 2 preds: ^bb2, ^bb3
      aie.use_lock(%jB_p1, AcquireGreaterEqual, 1)
      aie.dma_bd(%jB_buf : memref<2048xbf16>, 256, 256) {bd_id = 24 : i32, next_bd_id = 24 : i32}
      aie.use_lock(%jB_c1, Release, 1)
      aie.next_bd ^bb3
    ^bb4:  // pred: ^bb2
      %2 = aie.dma_start(S2MM, 2, ^bb5, ^bb6)
    ^bb5:  // 2 preds: ^bb4, ^bb5
      aie.use_lock(%jB_p2, AcquireGreaterEqual, 1)
      aie.dma_bd(%jB_buf : memref<2048xbf16>, 512, 256) {bd_id = 1 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%jB_c2, Release, 1)
      aie.next_bd ^bb5
    ^bb6:  // pred: ^bb4
      %3 = aie.dma_start(S2MM, 3, ^bb7, ^bb8)
    ^bb7:  // 2 preds: ^bb6, ^bb7
      aie.use_lock(%jB_p3, AcquireGreaterEqual, 1)
      aie.dma_bd(%jB_buf : memref<2048xbf16>, 768, 256) {bd_id = 25 : i32, next_bd_id = 25 : i32}
      aie.use_lock(%jB_c3, Release, 1)
      aie.next_bd ^bb7
    ^bb8:  // pred: ^bb6
      %4 = aie.dma_start(MM2S, 0, ^bb9, ^bb13)
    ^bb9:  // 2 preds: ^bb8, ^bb12
      aie.use_lock(%jB_c0, AcquireGreaterEqual, 1)
      aie.dma_bd(%jB_buf : memref<2048xbf16>, 0, 256) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%jB_p0, Release, 1)
      aie.next_bd ^bb10
    ^bb10:  // pred: ^bb9
      aie.use_lock(%jB_c1, AcquireGreaterEqual, 1)
      aie.dma_bd(%jB_buf : memref<2048xbf16>, 256, 256) {bd_id = 3 : i32, next_bd_id = 4 : i32}
      aie.use_lock(%jB_p1, Release, 1)
      aie.next_bd ^bb11
    ^bb11:  // pred: ^bb10
      aie.use_lock(%jB_c2, AcquireGreaterEqual, 1)
      aie.dma_bd(%jB_buf : memref<2048xbf16>, 512, 256) {bd_id = 4 : i32, next_bd_id = 5 : i32}
      aie.use_lock(%jB_p2, Release, 1)
      aie.next_bd ^bb12
    ^bb12:  // pred: ^bb11
      aie.use_lock(%jB_c3, AcquireGreaterEqual, 1)
      aie.dma_bd(%jB_buf : memref<2048xbf16>, 768, 256) {bd_id = 5 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%jB_p3, Release, 1)
      aie.next_bd ^bb9
    ^bb13:  // pred: ^bb8
      aie.end
    }
    %oJ_buf = aie.buffer(%mem_tile_4_1) {address = 0 : i32, mem_bank = 0 : i32, sym_name = "oJ_buf"} : memref<2048xbf16> 
    %oJ_p0 = aie.lock(%mem_tile_4_1, 0) {init = 1 : i32, sym_name = "oJ_p0"}
    %oJ_c0 = aie.lock(%mem_tile_4_1, 1) {init = 0 : i32, sym_name = "oJ_c0"}
    %oJ_p1 = aie.lock(%mem_tile_4_1, 2) {init = 1 : i32, sym_name = "oJ_p1"}
    %oJ_c1 = aie.lock(%mem_tile_4_1, 3) {init = 0 : i32, sym_name = "oJ_c1"}
    %oJ_p2 = aie.lock(%mem_tile_4_1, 4) {init = 1 : i32, sym_name = "oJ_p2"}
    %oJ_c2 = aie.lock(%mem_tile_4_1, 5) {init = 0 : i32, sym_name = "oJ_c2"}
    %oJ_p3 = aie.lock(%mem_tile_4_1, 6) {init = 1 : i32, sym_name = "oJ_p3"}
    %oJ_c3 = aie.lock(%mem_tile_4_1, 7) {init = 0 : i32, sym_name = "oJ_c3"}
    %memtile_dma_4_1 = aie.memtile_dma(%mem_tile_4_1) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb2)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%oJ_p0, AcquireGreaterEqual, 1)
      aie.dma_bd(%oJ_buf : memref<2048xbf16>, 0, 256) {bd_id = 0 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%oJ_c0, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb3, ^bb4)
    ^bb3:  // 2 preds: ^bb2, ^bb3
      aie.use_lock(%oJ_p1, AcquireGreaterEqual, 1)
      aie.dma_bd(%oJ_buf : memref<2048xbf16>, 256, 256) {bd_id = 24 : i32, next_bd_id = 24 : i32}
      aie.use_lock(%oJ_c1, Release, 1)
      aie.next_bd ^bb3
    ^bb4:  // pred: ^bb2
      %2 = aie.dma_start(S2MM, 2, ^bb5, ^bb6)
    ^bb5:  // 2 preds: ^bb4, ^bb5
      aie.use_lock(%oJ_p2, AcquireGreaterEqual, 1)
      aie.dma_bd(%oJ_buf : memref<2048xbf16>, 512, 256) {bd_id = 1 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%oJ_c2, Release, 1)
      aie.next_bd ^bb5
    ^bb6:  // pred: ^bb4
      %3 = aie.dma_start(S2MM, 3, ^bb7, ^bb8)
    ^bb7:  // 2 preds: ^bb6, ^bb7
      aie.use_lock(%oJ_p3, AcquireGreaterEqual, 1)
      aie.dma_bd(%oJ_buf : memref<2048xbf16>, 768, 256) {bd_id = 25 : i32, next_bd_id = 25 : i32}
      aie.use_lock(%oJ_c3, Release, 1)
      aie.next_bd ^bb7
    ^bb8:  // pred: ^bb6
      %4 = aie.dma_start(MM2S, 0, ^bb9, ^bb13)
    ^bb9:  // 2 preds: ^bb8, ^bb12
      aie.use_lock(%oJ_c0, AcquireGreaterEqual, 1)
      aie.dma_bd(%oJ_buf : memref<2048xbf16>, 0, 256) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%oJ_p0, Release, 1)
      aie.next_bd ^bb10
    ^bb10:  // pred: ^bb9
      aie.use_lock(%oJ_c1, AcquireGreaterEqual, 1)
      aie.dma_bd(%oJ_buf : memref<2048xbf16>, 256, 256) {bd_id = 3 : i32, next_bd_id = 4 : i32}
      aie.use_lock(%oJ_p1, Release, 1)
      aie.next_bd ^bb11
    ^bb11:  // pred: ^bb10
      aie.use_lock(%oJ_c2, AcquireGreaterEqual, 1)
      aie.dma_bd(%oJ_buf : memref<2048xbf16>, 512, 256) {bd_id = 4 : i32, next_bd_id = 5 : i32}
      aie.use_lock(%oJ_p2, Release, 1)
      aie.next_bd ^bb12
    ^bb12:  // pred: ^bb11
      aie.use_lock(%oJ_c3, AcquireGreaterEqual, 1)
      aie.dma_bd(%oJ_buf : memref<2048xbf16>, 768, 256) {bd_id = 5 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%oJ_p3, Release, 1)
      aie.next_bd ^bb9
    ^bb13:  // pred: ^bb8
      aie.end
    }
    %oK_buf = aie.buffer(%mem_tile_5_1) {address = 0 : i32, mem_bank = 0 : i32, sym_name = "oK_buf"} : memref<2048xbf16> 
    %oK_p0 = aie.lock(%mem_tile_5_1, 0) {init = 1 : i32, sym_name = "oK_p0"}
    %oK_c0 = aie.lock(%mem_tile_5_1, 1) {init = 0 : i32, sym_name = "oK_c0"}
    %oK_p1 = aie.lock(%mem_tile_5_1, 2) {init = 1 : i32, sym_name = "oK_p1"}
    %oK_c1 = aie.lock(%mem_tile_5_1, 3) {init = 0 : i32, sym_name = "oK_c1"}
    %oK_p2 = aie.lock(%mem_tile_5_1, 4) {init = 1 : i32, sym_name = "oK_p2"}
    %oK_c2 = aie.lock(%mem_tile_5_1, 5) {init = 0 : i32, sym_name = "oK_c2"}
    %oK_p3 = aie.lock(%mem_tile_5_1, 6) {init = 1 : i32, sym_name = "oK_p3"}
    %oK_c3 = aie.lock(%mem_tile_5_1, 7) {init = 0 : i32, sym_name = "oK_c3"}
    %memtile_dma_5_1 = aie.memtile_dma(%mem_tile_5_1) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb2)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%oK_p0, AcquireGreaterEqual, 1)
      aie.dma_bd(%oK_buf : memref<2048xbf16>, 0, 256) {bd_id = 0 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%oK_c0, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb3, ^bb4)
    ^bb3:  // 2 preds: ^bb2, ^bb3
      aie.use_lock(%oK_p1, AcquireGreaterEqual, 1)
      aie.dma_bd(%oK_buf : memref<2048xbf16>, 256, 256) {bd_id = 24 : i32, next_bd_id = 24 : i32}
      aie.use_lock(%oK_c1, Release, 1)
      aie.next_bd ^bb3
    ^bb4:  // pred: ^bb2
      %2 = aie.dma_start(S2MM, 2, ^bb5, ^bb6)
    ^bb5:  // 2 preds: ^bb4, ^bb5
      aie.use_lock(%oK_p2, AcquireGreaterEqual, 1)
      aie.dma_bd(%oK_buf : memref<2048xbf16>, 512, 256) {bd_id = 1 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%oK_c2, Release, 1)
      aie.next_bd ^bb5
    ^bb6:  // pred: ^bb4
      %3 = aie.dma_start(S2MM, 3, ^bb7, ^bb8)
    ^bb7:  // 2 preds: ^bb6, ^bb7
      aie.use_lock(%oK_p3, AcquireGreaterEqual, 1)
      aie.dma_bd(%oK_buf : memref<2048xbf16>, 768, 256) {bd_id = 25 : i32, next_bd_id = 25 : i32}
      aie.use_lock(%oK_c3, Release, 1)
      aie.next_bd ^bb7
    ^bb8:  // pred: ^bb6
      %4 = aie.dma_start(MM2S, 0, ^bb9, ^bb13)
    ^bb9:  // 2 preds: ^bb8, ^bb12
      aie.use_lock(%oK_c0, AcquireGreaterEqual, 1)
      aie.dma_bd(%oK_buf : memref<2048xbf16>, 0, 256) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%oK_p0, Release, 1)
      aie.next_bd ^bb10
    ^bb10:  // pred: ^bb9
      aie.use_lock(%oK_c1, AcquireGreaterEqual, 1)
      aie.dma_bd(%oK_buf : memref<2048xbf16>, 256, 256) {bd_id = 3 : i32, next_bd_id = 4 : i32}
      aie.use_lock(%oK_p1, Release, 1)
      aie.next_bd ^bb11
    ^bb11:  // pred: ^bb10
      aie.use_lock(%oK_c2, AcquireGreaterEqual, 1)
      aie.dma_bd(%oK_buf : memref<2048xbf16>, 512, 256) {bd_id = 4 : i32, next_bd_id = 5 : i32}
      aie.use_lock(%oK_p2, Release, 1)
      aie.next_bd ^bb12
    ^bb12:  // pred: ^bb11
      aie.use_lock(%oK_c3, AcquireGreaterEqual, 1)
      aie.dma_bd(%oK_buf : memref<2048xbf16>, 768, 256) {bd_id = 5 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%oK_p3, Release, 1)
      aie.next_bd ^bb9
    ^bb13:  // pred: ^bb8
      aie.end
    }
    %Klo_buf = aie.buffer(%mem_tile_0_1) {address = 0 : i32, sym_name = "Klo_buf"} : memref<65536xbf16> 
    %Klo_p0 = aie.lock(%mem_tile_0_1, 0) {init = 1 : i32, sym_name = "Klo_p0"}
    %Klo_c0 = aie.lock(%mem_tile_0_1, 1) {init = 0 : i32, sym_name = "Klo_c0"}
    %Klo_p1 = aie.lock(%mem_tile_0_1, 2) {init = 1 : i32, sym_name = "Klo_p1"}
    %Klo_c1 = aie.lock(%mem_tile_0_1, 3) {init = 0 : i32, sym_name = "Klo_c1"}
    %Klo_p2 = aie.lock(%mem_tile_0_1, 4) {init = 1 : i32, sym_name = "Klo_p2"}
    %Klo_c2 = aie.lock(%mem_tile_0_1, 5) {init = 0 : i32, sym_name = "Klo_c2"}
    %Klo_p3 = aie.lock(%mem_tile_0_1, 6) {init = 1 : i32, sym_name = "Klo_p3"}
    %Klo_c3 = aie.lock(%mem_tile_0_1, 7) {init = 0 : i32, sym_name = "Klo_c3"}
    %memtile_dma_0_1 = aie.memtile_dma(%mem_tile_0_1) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb5)
    ^bb1:  // 2 preds: ^bb0, ^bb4
      aie.use_lock(%Klo_p0, AcquireGreaterEqual, 1)
      aie.dma_bd(%Klo_buf : memref<65536xbf16>, 0, 16384) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%Klo_c0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%Klo_p1, AcquireGreaterEqual, 1)
      aie.dma_bd(%Klo_buf : memref<65536xbf16>, 16384, 16384) {bd_id = 1 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%Klo_c1, Release, 1)
      aie.next_bd ^bb3
    ^bb3:  // pred: ^bb2
      aie.use_lock(%Klo_p2, AcquireGreaterEqual, 1)
      aie.dma_bd(%Klo_buf : memref<65536xbf16>, 32768, 16384) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%Klo_c2, Release, 1)
      aie.next_bd ^bb4
    ^bb4:  // pred: ^bb3
      aie.use_lock(%Klo_p3, AcquireGreaterEqual, 1)
      aie.dma_bd(%Klo_buf : memref<65536xbf16>, 49152, 16384) {bd_id = 3 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%Klo_c3, Release, 1)
      aie.next_bd ^bb1
    ^bb5:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 0, ^bb6, ^bb7)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      aie.use_lock(%Klo_c0, AcquireGreaterEqual, 1)
      aie.dma_bd(%Klo_buf : memref<65536xbf16>, 0, 16384) {bd_id = 4 : i32, next_bd_id = 4 : i32}
      aie.use_lock(%Klo_p0, Release, 1)
      aie.next_bd ^bb6
    ^bb7:  // pred: ^bb5
      %2 = aie.dma_start(MM2S, 1, ^bb8, ^bb9)
    ^bb8:  // 2 preds: ^bb7, ^bb8
      aie.use_lock(%Klo_c1, AcquireGreaterEqual, 1)
      aie.dma_bd(%Klo_buf : memref<65536xbf16>, 16384, 16384) {bd_id = 24 : i32, next_bd_id = 24 : i32}
      aie.use_lock(%Klo_p1, Release, 1)
      aie.next_bd ^bb8
    ^bb9:  // pred: ^bb7
      %3 = aie.dma_start(MM2S, 2, ^bb10, ^bb11)
    ^bb10:  // 2 preds: ^bb9, ^bb10
      aie.use_lock(%Klo_c2, AcquireGreaterEqual, 1)
      aie.dma_bd(%Klo_buf : memref<65536xbf16>, 32768, 16384) {bd_id = 5 : i32, next_bd_id = 5 : i32}
      aie.use_lock(%Klo_p2, Release, 1)
      aie.next_bd ^bb10
    ^bb11:  // pred: ^bb9
      %4 = aie.dma_start(MM2S, 3, ^bb12, ^bb13)
    ^bb12:  // 2 preds: ^bb11, ^bb12
      aie.use_lock(%Klo_c3, AcquireGreaterEqual, 1)
      aie.dma_bd(%Klo_buf : memref<65536xbf16>, 49152, 16384) {bd_id = 25 : i32, next_bd_id = 25 : i32}
      aie.use_lock(%Klo_p3, Release, 1)
      aie.next_bd ^bb12
    ^bb13:  // pred: ^bb11
      aie.end
    }
    %Khi_buf = aie.buffer(%mem_tile_1_1) {address = 0 : i32, sym_name = "Khi_buf"} : memref<65536xbf16> 
    %Khi_p0 = aie.lock(%mem_tile_1_1, 0) {init = 1 : i32, sym_name = "Khi_p0"}
    %Khi_c0 = aie.lock(%mem_tile_1_1, 1) {init = 0 : i32, sym_name = "Khi_c0"}
    %Khi_p1 = aie.lock(%mem_tile_1_1, 2) {init = 1 : i32, sym_name = "Khi_p1"}
    %Khi_c1 = aie.lock(%mem_tile_1_1, 3) {init = 0 : i32, sym_name = "Khi_c1"}
    %Khi_p2 = aie.lock(%mem_tile_1_1, 4) {init = 1 : i32, sym_name = "Khi_p2"}
    %Khi_c2 = aie.lock(%mem_tile_1_1, 5) {init = 0 : i32, sym_name = "Khi_c2"}
    %Khi_p3 = aie.lock(%mem_tile_1_1, 6) {init = 1 : i32, sym_name = "Khi_p3"}
    %Khi_c3 = aie.lock(%mem_tile_1_1, 7) {init = 0 : i32, sym_name = "Khi_c3"}
    %memtile_dma_1_1 = aie.memtile_dma(%mem_tile_1_1) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb5)
    ^bb1:  // 2 preds: ^bb0, ^bb4
      aie.use_lock(%Khi_p0, AcquireGreaterEqual, 1)
      aie.dma_bd(%Khi_buf : memref<65536xbf16>, 0, 16384) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%Khi_c0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%Khi_p1, AcquireGreaterEqual, 1)
      aie.dma_bd(%Khi_buf : memref<65536xbf16>, 16384, 16384) {bd_id = 1 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%Khi_c1, Release, 1)
      aie.next_bd ^bb3
    ^bb3:  // pred: ^bb2
      aie.use_lock(%Khi_p2, AcquireGreaterEqual, 1)
      aie.dma_bd(%Khi_buf : memref<65536xbf16>, 32768, 16384) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%Khi_c2, Release, 1)
      aie.next_bd ^bb4
    ^bb4:  // pred: ^bb3
      aie.use_lock(%Khi_p3, AcquireGreaterEqual, 1)
      aie.dma_bd(%Khi_buf : memref<65536xbf16>, 49152, 16384) {bd_id = 3 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%Khi_c3, Release, 1)
      aie.next_bd ^bb1
    ^bb5:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 0, ^bb6, ^bb7)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      aie.use_lock(%Khi_c0, AcquireGreaterEqual, 1)
      aie.dma_bd(%Khi_buf : memref<65536xbf16>, 0, 16384) {bd_id = 4 : i32, next_bd_id = 4 : i32}
      aie.use_lock(%Khi_p0, Release, 1)
      aie.next_bd ^bb6
    ^bb7:  // pred: ^bb5
      %2 = aie.dma_start(MM2S, 1, ^bb8, ^bb9)
    ^bb8:  // 2 preds: ^bb7, ^bb8
      aie.use_lock(%Khi_c1, AcquireGreaterEqual, 1)
      aie.dma_bd(%Khi_buf : memref<65536xbf16>, 16384, 16384) {bd_id = 24 : i32, next_bd_id = 24 : i32}
      aie.use_lock(%Khi_p1, Release, 1)
      aie.next_bd ^bb8
    ^bb9:  // pred: ^bb7
      %3 = aie.dma_start(MM2S, 2, ^bb10, ^bb11)
    ^bb10:  // 2 preds: ^bb9, ^bb10
      aie.use_lock(%Khi_c2, AcquireGreaterEqual, 1)
      aie.dma_bd(%Khi_buf : memref<65536xbf16>, 32768, 16384) {bd_id = 5 : i32, next_bd_id = 5 : i32}
      aie.use_lock(%Khi_p2, Release, 1)
      aie.next_bd ^bb10
    ^bb11:  // pred: ^bb9
      %4 = aie.dma_start(MM2S, 3, ^bb12, ^bb13)
    ^bb12:  // 2 preds: ^bb11, ^bb12
      aie.use_lock(%Khi_c3, AcquireGreaterEqual, 1)
      aie.dma_bd(%Khi_buf : memref<65536xbf16>, 49152, 16384) {bd_id = 25 : i32, next_bd_id = 25 : i32}
      aie.use_lock(%Khi_p3, Release, 1)
      aie.next_bd ^bb12
    ^bb13:  // pred: ^bb11
      aie.end
    }
    %Vlo_buf = aie.buffer(%mem_tile_6_1) {address = 0 : i32, sym_name = "Vlo_buf"} : memref<65536xbf16> 
    %Vlo_p0 = aie.lock(%mem_tile_6_1, 0) {init = 1 : i32, sym_name = "Vlo_p0"}
    %Vlo_c0 = aie.lock(%mem_tile_6_1, 1) {init = 0 : i32, sym_name = "Vlo_c0"}
    %Vlo_p1 = aie.lock(%mem_tile_6_1, 2) {init = 1 : i32, sym_name = "Vlo_p1"}
    %Vlo_c1 = aie.lock(%mem_tile_6_1, 3) {init = 0 : i32, sym_name = "Vlo_c1"}
    %Vlo_p2 = aie.lock(%mem_tile_6_1, 4) {init = 1 : i32, sym_name = "Vlo_p2"}
    %Vlo_c2 = aie.lock(%mem_tile_6_1, 5) {init = 0 : i32, sym_name = "Vlo_c2"}
    %Vlo_p3 = aie.lock(%mem_tile_6_1, 6) {init = 1 : i32, sym_name = "Vlo_p3"}
    %Vlo_c3 = aie.lock(%mem_tile_6_1, 7) {init = 0 : i32, sym_name = "Vlo_c3"}
    %memtile_dma_6_1 = aie.memtile_dma(%mem_tile_6_1) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb5)
    ^bb1:  // 2 preds: ^bb0, ^bb4
      aie.use_lock(%Vlo_p0, AcquireGreaterEqual, 1)
      aie.dma_bd(%Vlo_buf : memref<65536xbf16>, 0, 16384) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%Vlo_c0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%Vlo_p1, AcquireGreaterEqual, 1)
      aie.dma_bd(%Vlo_buf : memref<65536xbf16>, 16384, 16384) {bd_id = 1 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%Vlo_c1, Release, 1)
      aie.next_bd ^bb3
    ^bb3:  // pred: ^bb2
      aie.use_lock(%Vlo_p2, AcquireGreaterEqual, 1)
      aie.dma_bd(%Vlo_buf : memref<65536xbf16>, 32768, 16384) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%Vlo_c2, Release, 1)
      aie.next_bd ^bb4
    ^bb4:  // pred: ^bb3
      aie.use_lock(%Vlo_p3, AcquireGreaterEqual, 1)
      aie.dma_bd(%Vlo_buf : memref<65536xbf16>, 49152, 16384) {bd_id = 3 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%Vlo_c3, Release, 1)
      aie.next_bd ^bb1
    ^bb5:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 0, ^bb6, ^bb7)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      aie.use_lock(%Vlo_c0, AcquireGreaterEqual, 1)
      aie.dma_bd(%Vlo_buf : memref<65536xbf16>, 0, 16384) {bd_id = 4 : i32, next_bd_id = 4 : i32}
      aie.use_lock(%Vlo_p0, Release, 1)
      aie.next_bd ^bb6
    ^bb7:  // pred: ^bb5
      %2 = aie.dma_start(MM2S, 1, ^bb8, ^bb9)
    ^bb8:  // 2 preds: ^bb7, ^bb8
      aie.use_lock(%Vlo_c1, AcquireGreaterEqual, 1)
      aie.dma_bd(%Vlo_buf : memref<65536xbf16>, 16384, 16384) {bd_id = 24 : i32, next_bd_id = 24 : i32}
      aie.use_lock(%Vlo_p1, Release, 1)
      aie.next_bd ^bb8
    ^bb9:  // pred: ^bb7
      %3 = aie.dma_start(MM2S, 2, ^bb10, ^bb11)
    ^bb10:  // 2 preds: ^bb9, ^bb10
      aie.use_lock(%Vlo_c2, AcquireGreaterEqual, 1)
      aie.dma_bd(%Vlo_buf : memref<65536xbf16>, 32768, 16384) {bd_id = 5 : i32, next_bd_id = 5 : i32}
      aie.use_lock(%Vlo_p2, Release, 1)
      aie.next_bd ^bb10
    ^bb11:  // pred: ^bb9
      %4 = aie.dma_start(MM2S, 3, ^bb12, ^bb13)
    ^bb12:  // 2 preds: ^bb11, ^bb12
      aie.use_lock(%Vlo_c3, AcquireGreaterEqual, 1)
      aie.dma_bd(%Vlo_buf : memref<65536xbf16>, 49152, 16384) {bd_id = 25 : i32, next_bd_id = 25 : i32}
      aie.use_lock(%Vlo_p3, Release, 1)
      aie.next_bd ^bb12
    ^bb13:  // pred: ^bb11
      aie.end
    }
    %Vhi_buf = aie.buffer(%mem_tile_7_1) {address = 0 : i32, sym_name = "Vhi_buf"} : memref<65536xbf16> 
    %Vhi_p0 = aie.lock(%mem_tile_7_1, 0) {init = 1 : i32, sym_name = "Vhi_p0"}
    %Vhi_c0 = aie.lock(%mem_tile_7_1, 1) {init = 0 : i32, sym_name = "Vhi_c0"}
    %Vhi_p1 = aie.lock(%mem_tile_7_1, 2) {init = 1 : i32, sym_name = "Vhi_p1"}
    %Vhi_c1 = aie.lock(%mem_tile_7_1, 3) {init = 0 : i32, sym_name = "Vhi_c1"}
    %Vhi_p2 = aie.lock(%mem_tile_7_1, 4) {init = 1 : i32, sym_name = "Vhi_p2"}
    %Vhi_c2 = aie.lock(%mem_tile_7_1, 5) {init = 0 : i32, sym_name = "Vhi_c2"}
    %Vhi_p3 = aie.lock(%mem_tile_7_1, 6) {init = 1 : i32, sym_name = "Vhi_p3"}
    %Vhi_c3 = aie.lock(%mem_tile_7_1, 7) {init = 0 : i32, sym_name = "Vhi_c3"}
    %memtile_dma_7_1 = aie.memtile_dma(%mem_tile_7_1) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb5)
    ^bb1:  // 2 preds: ^bb0, ^bb4
      aie.use_lock(%Vhi_p0, AcquireGreaterEqual, 1)
      aie.dma_bd(%Vhi_buf : memref<65536xbf16>, 0, 16384) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%Vhi_c0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%Vhi_p1, AcquireGreaterEqual, 1)
      aie.dma_bd(%Vhi_buf : memref<65536xbf16>, 16384, 16384) {bd_id = 1 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%Vhi_c1, Release, 1)
      aie.next_bd ^bb3
    ^bb3:  // pred: ^bb2
      aie.use_lock(%Vhi_p2, AcquireGreaterEqual, 1)
      aie.dma_bd(%Vhi_buf : memref<65536xbf16>, 32768, 16384) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%Vhi_c2, Release, 1)
      aie.next_bd ^bb4
    ^bb4:  // pred: ^bb3
      aie.use_lock(%Vhi_p3, AcquireGreaterEqual, 1)
      aie.dma_bd(%Vhi_buf : memref<65536xbf16>, 49152, 16384) {bd_id = 3 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%Vhi_c3, Release, 1)
      aie.next_bd ^bb1
    ^bb5:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 0, ^bb6, ^bb7)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      aie.use_lock(%Vhi_c0, AcquireGreaterEqual, 1)
      aie.dma_bd(%Vhi_buf : memref<65536xbf16>, 0, 16384) {bd_id = 4 : i32, next_bd_id = 4 : i32}
      aie.use_lock(%Vhi_p0, Release, 1)
      aie.next_bd ^bb6
    ^bb7:  // pred: ^bb5
      %2 = aie.dma_start(MM2S, 1, ^bb8, ^bb9)
    ^bb8:  // 2 preds: ^bb7, ^bb8
      aie.use_lock(%Vhi_c1, AcquireGreaterEqual, 1)
      aie.dma_bd(%Vhi_buf : memref<65536xbf16>, 16384, 16384) {bd_id = 24 : i32, next_bd_id = 24 : i32}
      aie.use_lock(%Vhi_p1, Release, 1)
      aie.next_bd ^bb8
    ^bb9:  // pred: ^bb7
      %3 = aie.dma_start(MM2S, 2, ^bb10, ^bb11)
    ^bb10:  // 2 preds: ^bb9, ^bb10
      aie.use_lock(%Vhi_c2, AcquireGreaterEqual, 1)
      aie.dma_bd(%Vhi_buf : memref<65536xbf16>, 32768, 16384) {bd_id = 5 : i32, next_bd_id = 5 : i32}
      aie.use_lock(%Vhi_p2, Release, 1)
      aie.next_bd ^bb10
    ^bb11:  // pred: ^bb9
      %4 = aie.dma_start(MM2S, 3, ^bb12, ^bb13)
    ^bb12:  // 2 preds: ^bb11, ^bb12
      aie.use_lock(%Vhi_c3, AcquireGreaterEqual, 1)
      aie.dma_bd(%Vhi_buf : memref<65536xbf16>, 49152, 16384) {bd_id = 25 : i32, next_bd_id = 25 : i32}
      aie.use_lock(%Vhi_p3, Release, 1)
      aie.next_bd ^bb12
    ^bb13:  // pred: ^bb11
      aie.end
    }
    %rl_A = aie.buffer(%tile_2_4) {address = 1024 : i32, mem_bank = 0 : i32, sym_name = "rl_A"} : memref<2048xbf16> 
    %rl_p0 = aie.lock(%tile_2_4, 0) {init = 1 : i32, sym_name = "rl_p0"}
    %rl_c0 = aie.lock(%tile_2_4, 1) {init = 0 : i32, sym_name = "rl_c0"}
    %rl_p1 = aie.lock(%tile_2_4, 2) {init = 1 : i32, sym_name = "rl_p1"}
    %rl_c1 = aie.lock(%tile_2_4, 3) {init = 0 : i32, sym_name = "rl_c1"}
    %rl_op = aie.lock(%tile_2_4, 4) {init = 1 : i32, sym_name = "rl_op"}
    %rl_oc = aie.lock(%tile_2_4, 5) {init = 0 : i32, sym_name = "rl_oc"}
    %core_2_4 = aie.core(%tile_2_4) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      cf.br ^bb1(%c0 : index)
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb2
      %1 = arith.cmpi slt, %0, %c9223372036854775807 : index
      cf.cond_br %1, ^bb2, ^bb3
    ^bb2:  // pred: ^bb1
      aie.use_lock(%rl_c0, AcquireGreaterEqual, 1)
      aie.use_lock(%rl_c1, AcquireGreaterEqual, 1)
      aie.use_lock(%rl_op, AcquireGreaterEqual, 1)
      aie.use_lock(%rl_p0, Release, 1)
      aie.use_lock(%rl_p1, Release, 1)
      aie.use_lock(%rl_oc, Release, 1)
      %2 = arith.addi %0, %c1 : index
      cf.br ^bb1(%2 : index)
    ^bb3:  // pred: ^bb1
      aie.end
    }
    %mem_2_4 = aie.mem(%tile_2_4) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb2)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%rl_p0, AcquireGreaterEqual, 1)
      aie.dma_bd(%rl_A : memref<2048xbf16>, 0, 1024) {bd_id = 0 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%rl_c0, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb3, ^bb4)
    ^bb3:  // 2 preds: ^bb2, ^bb3
      aie.use_lock(%rl_p1, AcquireGreaterEqual, 1)
      aie.dma_bd(%rl_A : memref<2048xbf16>, 1024, 1024) {bd_id = 1 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%rl_c1, Release, 1)
      aie.next_bd ^bb3
    ^bb4:  // pred: ^bb2
      %2 = aie.dma_start(MM2S, 0, ^bb5, ^bb6)
    ^bb5:  // 2 preds: ^bb4, ^bb5
      aie.use_lock(%rl_oc, AcquireGreaterEqual, 1)
      aie.dma_bd_packet(0, 0)
      aie.dma_bd(%rl_A : memref<2048xbf16>, 0, 2048) {bd_id = 2 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%rl_op, Release, 1)
      aie.next_bd ^bb5
    ^bb6:  // pred: ^bb4
      aie.end
    }
    %op_O = aie.buffer(%tile_3_4) {address = 1024 : i32, mem_bank = 0 : i32, sym_name = "op_O"} : memref<2048xbf16> 
    %op_p0 = aie.lock(%tile_3_4, 0) {init = 1 : i32, sym_name = "op_p0"}
    %op_c0 = aie.lock(%tile_3_4, 1) {init = 0 : i32, sym_name = "op_c0"}
    %op_p1 = aie.lock(%tile_3_4, 2) {init = 1 : i32, sym_name = "op_p1"}
    %op_c1 = aie.lock(%tile_3_4, 3) {init = 0 : i32, sym_name = "op_c1"}
    %op_op = aie.lock(%tile_3_4, 4) {init = 1 : i32, sym_name = "op_op"}
    %op_oc = aie.lock(%tile_3_4, 5) {init = 0 : i32, sym_name = "op_oc"}
    %core_3_4 = aie.core(%tile_3_4) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      cf.br ^bb1(%c0 : index)
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb2
      %1 = arith.cmpi slt, %0, %c9223372036854775807 : index
      cf.cond_br %1, ^bb2, ^bb3
    ^bb2:  // pred: ^bb1
      aie.use_lock(%op_c0, AcquireGreaterEqual, 1)
      aie.use_lock(%op_c1, AcquireGreaterEqual, 1)
      aie.use_lock(%op_op, AcquireGreaterEqual, 1)
      aie.use_lock(%op_p0, Release, 1)
      aie.use_lock(%op_p1, Release, 1)
      aie.use_lock(%op_oc, Release, 1)
      %2 = arith.addi %0, %c1 : index
      cf.br ^bb1(%2 : index)
    ^bb3:  // pred: ^bb1
      aie.end
    }
    %mem_3_4 = aie.mem(%tile_3_4) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb2)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%op_p0, AcquireGreaterEqual, 1)
      aie.dma_bd(%op_O : memref<2048xbf16>, 0, 1024) {bd_id = 0 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%op_c0, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb3, ^bb4)
    ^bb3:  // 2 preds: ^bb2, ^bb3
      aie.use_lock(%op_p1, AcquireGreaterEqual, 1)
      aie.dma_bd(%op_O : memref<2048xbf16>, 1024, 1024) {bd_id = 1 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%op_c1, Release, 1)
      aie.next_bd ^bb3
    ^bb4:  // pred: ^bb2
      %2 = aie.dma_start(MM2S, 0, ^bb5, ^bb6)
    ^bb5:  // 2 preds: ^bb4, ^bb5
      aie.use_lock(%op_oc, AcquireGreaterEqual, 1)
      aie.dma_bd(%op_O : memref<2048xbf16>, 0, 2048) {bd_id = 2 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%op_op, Release, 1)
      aie.next_bd ^bb5
    ^bb6:  // pred: ^bb4
      aie.end
    }
    %nm_O = aie.buffer(%tile_4_4) {address = 32768 : i32, mem_bank = 2 : i32, sym_name = "nm_O"} : memref<2048xbf16> 
    %nm_R = aie.buffer(%tile_4_4) {address = 49152 : i32, mem_bank = 3 : i32, sym_name = "nm_R"} : memref<2048xbf16> 
    %nm_F = aie.buffer(%tile_4_4) {address = 1024 : i32, mem_bank = 0 : i32, sym_name = "nm_F"} : memref<2320xbf16> 
    %nm_FN = aie.buffer(%tile_4_4) {address = 16384 : i32, mem_bank = 1 : i32, sym_name = "nm_FN"} : memref<2320xbf16> 
    %nm_gain = aie.buffer(%tile_4_4) {address = 5664 : i32, mem_bank = 0 : i32, sym_name = "nm_gain"} : memref<2048xbf16> 
    %nm_Op = aie.lock(%tile_4_4, 0) {init = 1 : i32, sym_name = "nm_Op"}
    %nm_Oc = aie.lock(%tile_4_4, 1) {init = 0 : i32, sym_name = "nm_Oc"}
    %nm_Rp = aie.lock(%tile_4_4, 2) {init = 1 : i32, sym_name = "nm_Rp"}
    %nm_Rc = aie.lock(%tile_4_4, 3) {init = 0 : i32, sym_name = "nm_Rc"}
    %nm_Fp = aie.lock(%tile_4_4, 4) {init = 1 : i32, sym_name = "nm_Fp"}
    %nm_Fc = aie.lock(%tile_4_4, 5) {init = 0 : i32, sym_name = "nm_Fc"}
    %nm_Gp = aie.lock(%tile_4_4, 6) {init = 1 : i32, sym_name = "nm_Gp"}
    %nm_Gc = aie.lock(%tile_4_4, 7) {init = 0 : i32, sym_name = "nm_Gc"}
    %nm_FNp = aie.lock(%tile_4_4, 8) {init = 1 : i32, sym_name = "nm_FNp"}
    %nm_FNc = aie.lock(%tile_4_4, 9) {init = 0 : i32, sym_name = "nm_FNc"}
    %core_4_4 = aie.core(%tile_4_4) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      %c2048_i32 = arith.constant 2048 : i32
      cf.br ^bb1(%c0 : index)
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb2
      %1 = arith.cmpi slt, %0, %c9223372036854775807 : index
      cf.cond_br %1, ^bb2, ^bb3
    ^bb2:  // pred: ^bb1
      aie.use_lock(%nm_Oc, AcquireGreaterEqual, 1)
      aie.use_lock(%nm_Rc, AcquireGreaterEqual, 1)
      aie.use_lock(%nm_Gc, AcquireGreaterEqual, 1)
      aie.use_lock(%nm_Fp, AcquireGreaterEqual, 1)
      aie.use_lock(%nm_FNp, AcquireGreaterEqual, 1)
      func.call @layer_fused_add_bf16(%nm_O, %nm_R, %nm_F, %c2048_i32) : (memref<2048xbf16>, memref<2048xbf16>, memref<2320xbf16>, i32) -> ()
      func.call @layer_fused_rms_norm2_bf16(%nm_F, %nm_gain, %nm_FN, %c2048_i32) : (memref<2320xbf16>, memref<2048xbf16>, memref<2320xbf16>, i32) -> ()
      aie.use_lock(%nm_Op, Release, 1)
      aie.use_lock(%nm_Rp, Release, 1)
      aie.use_lock(%nm_Gp, Release, 1)
      aie.use_lock(%nm_Fc, Release, 1)
      aie.use_lock(%nm_FNc, Release, 1)
      %2 = arith.addi %0, %c1 : index
      cf.br ^bb1(%2 : index)
    ^bb3:  // pred: ^bb1
      aie.end
    } {link_files = ["layer_fused_relay.o"]}
    %mem_4_4 = aie.mem(%tile_4_4) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb2)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%nm_Op, AcquireGreaterEqual, 1)
      aie.dma_bd(%nm_O : memref<2048xbf16>, 0, 2048) {bd_id = 0 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%nm_Oc, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb3, ^bb5)
    ^bb3:  // 2 preds: ^bb2, ^bb4
      aie.use_lock(%nm_Rp, AcquireGreaterEqual, 1)
      aie.dma_bd(%nm_R : memref<2048xbf16>, 0, 2048) {bd_id = 1 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%nm_Rc, Release, 1)
      aie.next_bd ^bb4
    ^bb4:  // pred: ^bb3
      aie.use_lock(%nm_Gp, AcquireGreaterEqual, 1)
      aie.dma_bd(%nm_gain : memref<2048xbf16>, 0, 2048) {bd_id = 2 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%nm_Gc, Release, 1)
      aie.next_bd ^bb3
    ^bb5:  // pred: ^bb2
      %2 = aie.dma_start(MM2S, 0, ^bb6, ^bb7)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      aie.use_lock(%nm_FNc, AcquireGreaterEqual, 1)
      aie.dma_bd_packet(0, 1)
      aie.dma_bd(%nm_FN : memref<2320xbf16>, 0, 2048) {bd_id = 3 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%nm_FNp, Release, 1)
      aie.next_bd ^bb6
    ^bb7:  // pred: ^bb5
      %3 = aie.dma_start(MM2S, 1, ^bb8, ^bb9)
    ^bb8:  // 2 preds: ^bb7, ^bb8
      aie.use_lock(%nm_Fc, AcquireGreaterEqual, 1)
      aie.dma_bd(%nm_F : memref<2320xbf16>, 0, 2048) {bd_id = 4 : i32, next_bd_id = 4 : i32}
      aie.use_lock(%nm_Fp, Release, 1)
      aie.next_bd ^bb8
    ^bb9:  // pred: ^bb7
      aie.end
    }
    %mx_x = aie.buffer(%tile_5_4) {address = 1024 : i32, mem_bank = 0 : i32, sym_name = "mx_x"} : memref<2320xbf16> 
    %mx_a = aie.buffer(%tile_5_4) {address = 16384 : i32, mem_bank = 1 : i32, sym_name = "mx_a"} : memref<2320xbf16> 
    %mx_f = aie.buffer(%tile_5_4) {address = 32768 : i32, mem_bank = 2 : i32, sym_name = "mx_f"} : memref<2320xbf16> 
    %mx_o = aie.buffer(%tile_5_4) {address = 49152 : i32, mem_bank = 3 : i32, sym_name = "mx_o"} : memref<2320xbf16> 
    %mx_xp = aie.lock(%tile_5_4, 0) {init = 1 : i32, sym_name = "mx_xp"}
    %mx_xc = aie.lock(%tile_5_4, 1) {init = 0 : i32, sym_name = "mx_xc"}
    %mx_ap = aie.lock(%tile_5_4, 2) {init = 1 : i32, sym_name = "mx_ap"}
    %mx_ac = aie.lock(%tile_5_4, 3) {init = 0 : i32, sym_name = "mx_ac"}
    %mx_fp = aie.lock(%tile_5_4, 4) {init = 1 : i32, sym_name = "mx_fp"}
    %mx_fc = aie.lock(%tile_5_4, 5) {init = 0 : i32, sym_name = "mx_fc"}
    %mx_op = aie.lock(%tile_5_4, 6) {init = 1 : i32, sym_name = "mx_op"}
    %mx_oc = aie.lock(%tile_5_4, 7) {init = 0 : i32, sym_name = "mx_oc"}
    %core_5_4 = aie.core(%tile_5_4) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      %c2320_i32 = arith.constant 2320 : i32
      %c2048_i32 = arith.constant 2048 : i32
      cf.br ^bb1(%c0 : index)
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb2
      %1 = arith.cmpi slt, %0, %c9223372036854775807 : index
      cf.cond_br %1, ^bb2, ^bb3
    ^bb2:  // pred: ^bb1
      aie.use_lock(%mx_xc, AcquireGreaterEqual, 1)
      aie.use_lock(%mx_op, AcquireGreaterEqual, 1)
      func.call @attn_copy_bf16(%mx_x, %mx_o, %c2320_i32) : (memref<2320xbf16>, memref<2320xbf16>, i32) -> ()
      aie.use_lock(%mx_xp, Release, 1)
      aie.use_lock(%mx_oc, Release, 1)
      aie.use_lock(%mx_ac, AcquireGreaterEqual, 1)
      aie.use_lock(%mx_op, AcquireGreaterEqual, 1)
      func.call @attn_copy_bf16(%mx_a, %mx_o, %c2048_i32) : (memref<2320xbf16>, memref<2320xbf16>, i32) -> ()
      aie.use_lock(%mx_ap, Release, 1)
      aie.use_lock(%mx_oc, Release, 1)
      aie.use_lock(%mx_fc, AcquireGreaterEqual, 1)
      aie.use_lock(%mx_op, AcquireGreaterEqual, 1)
      func.call @attn_copy_bf16(%mx_f, %mx_o, %c2048_i32) : (memref<2320xbf16>, memref<2320xbf16>, i32) -> ()
      aie.use_lock(%mx_fp, Release, 1)
      aie.use_lock(%mx_oc, Release, 1)
      %2 = arith.addi %0, %c1 : index
      cf.br ^bb1(%2 : index)
    ^bb3:  // pred: ^bb1
      aie.end
    } {link_files = ["attn_concat.o"]}
    %mem_5_4 = aie.mem(%tile_5_4) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb2)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%mx_xp, AcquireGreaterEqual, 1)
      aie.dma_bd(%mx_x : memref<2320xbf16>, 0, 2320) {bd_id = 0 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%mx_xc, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb3, ^bb5)
    ^bb3:  // 2 preds: ^bb2, ^bb4
      aie.use_lock(%mx_ap, AcquireGreaterEqual, 1)
      aie.dma_bd_packet(0, 0)
      aie.dma_bd(%mx_a : memref<2320xbf16>, 0, 2048) {bd_id = 1 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%mx_ac, Release, 1)
      aie.next_bd ^bb4
    ^bb4:  // pred: ^bb3
      aie.use_lock(%mx_fp, AcquireGreaterEqual, 1)
      aie.dma_bd_packet(0, 1)
      aie.dma_bd(%mx_f : memref<2320xbf16>, 0, 2048) {bd_id = 2 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%mx_fc, Release, 1)
      aie.next_bd ^bb3
    ^bb5:  // pred: ^bb2
      %2 = aie.dma_start(MM2S, 0, ^bb6, ^bb7)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      aie.use_lock(%mx_oc, AcquireGreaterEqual, 1)
      aie.dma_bd(%mx_o : memref<2320xbf16>, 0, 2320) {bd_id = 3 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%mx_op, Release, 1)
      aie.next_bd ^bb6
    ^bb7:  // pred: ^bb5
      aie.end
    }
    aie.shim_dma_allocation @X_alloc(%shim_noc_tile_0_0, MM2S, 1)
    aie.shim_dma_allocation @R_alloc(%shim_noc_tile_2_0, MM2S, 1)
    aie.shim_dma_allocation @Klo_alloc(%shim_noc_tile_3_0, MM2S, 1)
    aie.shim_dma_allocation @Khi_alloc(%shim_noc_tile_4_0, MM2S, 1)
    aie.shim_dma_allocation @Vlo_alloc(%shim_noc_tile_5_0, MM2S, 1)
    aie.shim_dma_allocation @Vhi_alloc(%shim_noc_tile_6_0, MM2S, 1)
    aie.shim_dma_allocation @S_alloc(%shim_noc_tile_4_0, S2MM, 1)
    aie.shim_dma_allocation @A0(%shim_noc_tile_0_0, MM2S, 0)
    aie.shim_dma_allocation @P0(%shim_noc_tile_0_0, S2MM, 0)
    aie.shim_dma_allocation @A1(%shim_noc_tile_1_0, MM2S, 0)
    aie.shim_dma_allocation @P1(%shim_noc_tile_1_0, S2MM, 0)
    aie.shim_dma_allocation @A2(%shim_noc_tile_2_0, MM2S, 0)
    aie.shim_dma_allocation @P2(%shim_noc_tile_2_0, S2MM, 0)
    aie.shim_dma_allocation @A3(%shim_noc_tile_3_0, MM2S, 0)
    aie.shim_dma_allocation @P3(%shim_noc_tile_3_0, S2MM, 0)
    aie.shim_dma_allocation @A4(%shim_noc_tile_4_0, MM2S, 0)
    aie.shim_dma_allocation @P4(%shim_noc_tile_4_0, S2MM, 0)
    aie.shim_dma_allocation @A5(%shim_noc_tile_5_0, MM2S, 0)
    aie.shim_dma_allocation @P5(%shim_noc_tile_5_0, S2MM, 0)
    aie.shim_dma_allocation @A6(%shim_noc_tile_6_0, MM2S, 0)
    aie.shim_dma_allocation @P6(%shim_noc_tile_6_0, S2MM, 0)
    aie.shim_dma_allocation @A7(%shim_noc_tile_7_0, MM2S, 0)
    aie.shim_dma_allocation @P7(%shim_noc_tile_7_0, S2MM, 0)
    aie.runtime_sequence(%arg0: memref<18432xbf16>, %arg1: memref<6416xbf16>, %arg2: memref<33030144xi8>, %arg3: memref<2359296xi8>, %arg4: memref<262144xbf16>) {
      %0 = aiex.dma_configure_task_for @X_alloc {
        aie.dma_bd(%arg1 : memref<6416xbf16>, 0, 2320, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 2320, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%0)
      %1 = aiex.dma_configure_task_for @R_alloc {
        aie.dma_bd(%arg1 : memref<6416xbf16>, 2320, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 4096, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%1)
      %2 = aiex.dma_configure_task_for @Klo_alloc {
        aie.dma_bd(%arg4 : memref<262144xbf16>, 0, 65536, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 65536, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%2)
      %3 = aiex.dma_configure_task_for @Khi_alloc {
        aie.dma_bd(%arg4 : memref<262144xbf16>, 65536, 65536, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 65536, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%3)
      %4 = aiex.dma_configure_task_for @Vlo_alloc {
        aie.dma_bd(%arg4 : memref<262144xbf16>, 131072, 65536, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 65536, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%4)
      %5 = aiex.dma_configure_task_for @Vhi_alloc {
        aie.dma_bd(%arg4 : memref<262144xbf16>, 196608, 65536, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 65536, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%5)
      %6 = aiex.dma_configure_task_for @A0 {
        aie.dma_bd(%arg2 : memref<33030144xi8>, 0, 4128768, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 4128768, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%6)
      %7 = aiex.dma_configure_task_for @P0 {
        aie.dma_bd(%arg0 : memref<18432xbf16>, 0, 2048, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 2048, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%7)
      %8 = aiex.dma_configure_task_for @A1 {
        aie.dma_bd(%arg2 : memref<33030144xi8>, 4128768, 4128768, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 4128768, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%8)
      %9 = aiex.dma_configure_task_for @P1 {
        aie.dma_bd(%arg0 : memref<18432xbf16>, 2048, 2048, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 2048, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%9)
      %10 = aiex.dma_configure_task_for @A2 {
        aie.dma_bd(%arg2 : memref<33030144xi8>, 8257536, 4128768, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 4128768, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%10)
      %11 = aiex.dma_configure_task_for @P2 {
        aie.dma_bd(%arg0 : memref<18432xbf16>, 4096, 2048, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 2048, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%11)
      %12 = aiex.dma_configure_task_for @A3 {
        aie.dma_bd(%arg2 : memref<33030144xi8>, 12386304, 4128768, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 4128768, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%12)
      %13 = aiex.dma_configure_task_for @P3 {
        aie.dma_bd(%arg0 : memref<18432xbf16>, 6144, 2048, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 2048, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%13)
      %14 = aiex.dma_configure_task_for @A4 {
        aie.dma_bd(%arg2 : memref<33030144xi8>, 16515072, 4128768, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 4128768, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%14)
      %15 = aiex.dma_configure_task_for @P4 {
        aie.dma_bd(%arg0 : memref<18432xbf16>, 8192, 2048, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 2048, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%15)
      %16 = aiex.dma_configure_task_for @A5 {
        aie.dma_bd(%arg2 : memref<33030144xi8>, 20643840, 4128768, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 4128768, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%16)
      %17 = aiex.dma_configure_task_for @P5 {
        aie.dma_bd(%arg0 : memref<18432xbf16>, 10240, 2048, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 2048, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%17)
      %18 = aiex.dma_configure_task_for @A6 {
        aie.dma_bd(%arg2 : memref<33030144xi8>, 24772608, 4128768, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 4128768, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%18)
      %19 = aiex.dma_configure_task_for @P6 {
        aie.dma_bd(%arg0 : memref<18432xbf16>, 12288, 2048, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 2048, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%19)
      %20 = aiex.dma_configure_task_for @A7 {
        aie.dma_bd(%arg2 : memref<33030144xi8>, 28901376, 4128768, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 4128768, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%20)
      %21 = aiex.dma_configure_task_for @P7 {
        aie.dma_bd(%arg0 : memref<18432xbf16>, 14336, 2048, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 2048, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%21)
      %22 = aiex.dma_configure_task_for @S_alloc {
        aie.dma_bd(%arg0 : memref<18432xbf16>, 16384, 2048, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 2048, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%22)
      aiex.dma_await_task(%7)
      aiex.dma_await_task(%9)
      aiex.dma_await_task(%11)
      aiex.dma_await_task(%13)
      aiex.dma_await_task(%15)
      aiex.dma_await_task(%17)
      aiex.dma_await_task(%19)
      aiex.dma_await_task(%21)
      aiex.dma_await_task(%22)
    }
    aie.packet_flow(15) {
      aie.packet_source<%shim_noc_tile_0_0, TileControl : 0>
      aie.packet_dest<%shim_noc_tile_0_0, South : 0>
    } {keep_pkt_header = true, priority_route = true}
    aie.packet_flow(15) {
      aie.packet_source<%shim_noc_tile_1_0, TileControl : 0>
      aie.packet_dest<%shim_noc_tile_1_0, South : 0>
    } {keep_pkt_header = true, priority_route = true}
    aie.packet_flow(15) {
      aie.packet_source<%shim_noc_tile_2_0, TileControl : 0>
      aie.packet_dest<%shim_noc_tile_2_0, South : 0>
    } {keep_pkt_header = true, priority_route = true}
    aie.packet_flow(15) {
      aie.packet_source<%shim_noc_tile_3_0, TileControl : 0>
      aie.packet_dest<%shim_noc_tile_3_0, South : 0>
    } {keep_pkt_header = true, priority_route = true}
    aie.packet_flow(15) {
      aie.packet_source<%shim_noc_tile_4_0, TileControl : 0>
      aie.packet_dest<%shim_noc_tile_4_0, South : 0>
    } {keep_pkt_header = true, priority_route = true}
    aie.packet_flow(15) {
      aie.packet_source<%shim_noc_tile_5_0, TileControl : 0>
      aie.packet_dest<%shim_noc_tile_5_0, South : 0>
    } {keep_pkt_header = true, priority_route = true}
    aie.packet_flow(15) {
      aie.packet_source<%shim_noc_tile_6_0, TileControl : 0>
      aie.packet_dest<%shim_noc_tile_6_0, South : 0>
    } {keep_pkt_header = true, priority_route = true}
    aie.packet_flow(15) {
      aie.packet_source<%shim_noc_tile_7_0, TileControl : 0>
      aie.packet_dest<%shim_noc_tile_7_0, South : 0>
    } {keep_pkt_header = true, priority_route = true}
  }
}

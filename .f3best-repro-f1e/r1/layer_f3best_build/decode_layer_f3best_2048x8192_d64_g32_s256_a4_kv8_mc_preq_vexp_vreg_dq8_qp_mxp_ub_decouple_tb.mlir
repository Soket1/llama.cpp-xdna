module {
  aie.device(npu2) {
    %t0 = aie.tile(2, 2)
    %t1 = aie.tile(3, 2)
    %t2 = aie.tile(4, 2)
    %t3 = aie.tile(5, 2)
    %t4 = aie.tile(2, 3)
    %t5 = aie.tile(3, 3)
    %t6 = aie.tile(4, 3)
    %t7 = aie.tile(5, 3)
    %sc0 = aie.tile(0, 2)
    %sc1 = aie.tile(1, 2)
    %sc2 = aie.tile(6, 2)
    %sc3 = aie.tile(7, 2)
    %sc4 = aie.tile(0, 3)
    %sc5 = aie.tile(1, 3)
    %sc6 = aie.tile(6, 3)
    %sc7 = aie.tile(7, 3)
    %va0 = aie.tile(0, 4)
    %va1 = aie.tile(1, 4)
    %va2 = aie.tile(6, 4)
    %va3 = aie.tile(7, 4)
    %va4 = aie.tile(0, 5)
    %va5 = aie.tile(1, 5)
    %va6 = aie.tile(6, 5)
    %va7 = aie.tile(7, 5)
    %jA = aie.tile(2, 1)
    %jB = aie.tile(3, 1)
    %Klo = aie.tile(0, 1)
    %Khi = aie.tile(1, 1)
    %Vlo = aie.tile(6, 1)
    %Vhi = aie.tile(7, 1)
    %oJ = aie.tile(4, 1)
    %oK = aie.tile(5, 1)
    %rl = aie.tile(2, 4)
    %op = aie.tile(3, 4)
    %nm = aie.tile(4, 4)
    %mx = aie.tile(5, 4)
    %sh0 = aie.tile(0, 0)
    %sh1 = aie.tile(1, 0)
    %sh2 = aie.tile(2, 0)
    %sh3 = aie.tile(3, 0)
    %sh4 = aie.tile(4, 0)
    %sh5 = aie.tile(5, 0)
    %sh6 = aie.tile(6, 0)
    %sh7 = aie.tile(7, 0)

    func.func private @_ha_noop() -> () attributes {link_with = "layer_fused_bcast_kc256.o"}
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

    aie.flow(%sh0, DMA : 1, %mx, DMA : 0)   // x_bundle -> mux S2MM0
    aie.packet_flow(0) { aie.packet_source<%rl, DMA : 0> aie.packet_dest<%mx, DMA : 1> }   // attn_out -> mux S2MM1
    aie.packet_flow(1) { aie.packet_source<%nm, DMA : 0> aie.packet_dest<%mx, DMA : 1> }   // ffn_in -> mux S2MM1
    aie.flow(%sh0, DMA : 0, %Klo, DMA : 4)   // A0 shim -> Klo (edge relay)
    aie.flow(%Klo, DMA : 4, %t0, DMA : 0)   // A0 Klo -> center
    aie.flow(%mx, DMA : 0, %t0, DMA : 1)   // mux 3-bcast -> center 0
    aie.flow(%t0, DMA : 0, %sh0, DMA : 0)   // Pf0 -> drain (circuit)
    aie.flow(%t0, DMA : 1, %sc0, DMA : 0)   // [Qi,O_h]0 2-BD -> score0 (circuit)
    aie.flow(%sc0, DMA : 0, %va0, DMA : 0)   // inter0 -> value0
    aie.flow(%sc0, DMA : 1, %oJ, DMA : 0)   // O_h0 relayed by score -> oJ[0]
    aie.flow(%sh1, DMA : 0, %Khi, DMA : 4)   // A1 shim -> Khi (edge relay)
    aie.flow(%Khi, DMA : 4, %t1, DMA : 0)   // A1 Khi -> center
    aie.flow(%mx, DMA : 0, %t1, DMA : 1)   // mux 3-bcast -> center 1
    aie.flow(%t1, DMA : 0, %sh1, DMA : 0)   // Pf1 -> drain (circuit)
    aie.flow(%t1, DMA : 1, %sc1, DMA : 0)   // [Qi,O_h]1 2-BD -> score1 (circuit)
    aie.flow(%sc1, DMA : 0, %va1, DMA : 0)   // inter1 -> value1
    aie.flow(%sc1, DMA : 1, %oJ, DMA : 1)   // O_h1 relayed by score -> oJ[1]
    aie.flow(%sh2, DMA : 0, %t2, DMA : 0)   // A2
    aie.flow(%mx, DMA : 0, %t2, DMA : 1)   // mux 3-bcast -> center 2
    aie.flow(%t2, DMA : 0, %sh2, DMA : 0)   // Pf2 -> drain (circuit)
    aie.flow(%t2, DMA : 1, %sc2, DMA : 0)   // [Qi,O_h]2 2-BD -> score2 (circuit)
    aie.flow(%sc2, DMA : 0, %va2, DMA : 0)   // inter2 -> value2
    aie.flow(%sc2, DMA : 1, %oJ, DMA : 2)   // O_h2 relayed by score -> oJ[2]
    aie.flow(%sh3, DMA : 0, %t3, DMA : 0)   // A3
    aie.flow(%mx, DMA : 0, %t3, DMA : 1)   // mux 3-bcast -> center 3
    aie.flow(%t3, DMA : 0, %sh3, DMA : 0)   // Pf3 -> drain (circuit)
    aie.flow(%t3, DMA : 1, %sc3, DMA : 0)   // [Qi,O_h]3 2-BD -> score3 (circuit)
    aie.flow(%sc3, DMA : 0, %va3, DMA : 0)   // inter3 -> value3
    aie.flow(%sc3, DMA : 1, %oJ, DMA : 3)   // O_h3 relayed by score -> oJ[3]
    aie.flow(%sh4, DMA : 0, %t4, DMA : 0)   // A4
    aie.flow(%mx, DMA : 0, %t4, DMA : 1)   // mux 3-bcast -> center 4
    aie.flow(%t4, DMA : 0, %sh4, DMA : 0)   // Pf4 -> drain (circuit)
    aie.flow(%t4, DMA : 1, %sc4, DMA : 0)   // [Qi,O_h]4 2-BD -> score4 (circuit)
    aie.flow(%sc4, DMA : 0, %va4, DMA : 0)   // inter4 -> value4
    aie.flow(%sc4, DMA : 1, %oK, DMA : 0)   // O_h4 relayed by score -> oK[0]
    aie.flow(%sh5, DMA : 0, %t5, DMA : 0)   // A5
    aie.flow(%mx, DMA : 0, %t5, DMA : 1)   // mux 3-bcast -> center 5
    aie.flow(%t5, DMA : 0, %sh5, DMA : 0)   // Pf5 -> drain (circuit)
    aie.flow(%t5, DMA : 1, %sc5, DMA : 0)   // [Qi,O_h]5 2-BD -> score5 (circuit)
    aie.flow(%sc5, DMA : 0, %va5, DMA : 0)   // inter5 -> value5
    aie.flow(%sc5, DMA : 1, %oK, DMA : 1)   // O_h5 relayed by score -> oK[1]
    aie.flow(%sh6, DMA : 0, %Vlo, DMA : 4)   // A6 shim -> Vlo (edge relay)
    aie.flow(%Vlo, DMA : 4, %t6, DMA : 0)   // A6 Vlo -> center
    aie.flow(%mx, DMA : 0, %t6, DMA : 1)   // mux 3-bcast -> center 6
    aie.flow(%t6, DMA : 0, %sh6, DMA : 0)   // Pf6 -> drain (circuit)
    aie.flow(%t6, DMA : 1, %sc6, DMA : 0)   // [Qi,O_h]6 2-BD -> score6 (circuit)
    aie.flow(%sc6, DMA : 0, %va6, DMA : 0)   // inter6 -> value6
    aie.flow(%sc6, DMA : 1, %oK, DMA : 2)   // O_h6 relayed by score -> oK[2]
    aie.flow(%sh7, DMA : 0, %Vhi, DMA : 4)   // A7 shim -> Vhi (edge relay)
    aie.flow(%Vhi, DMA : 4, %t7, DMA : 0)   // A7 Vhi -> center
    aie.flow(%mx, DMA : 0, %t7, DMA : 1)   // mux 3-bcast -> center 7
    aie.flow(%t7, DMA : 0, %sh7, DMA : 0)   // Pf7 -> drain (circuit)
    aie.flow(%t7, DMA : 1, %sc7, DMA : 0)   // [Qi,O_h]7 2-BD -> score7 (circuit)
    aie.flow(%sc7, DMA : 0, %va7, DMA : 0)   // inter7 -> value7
    aie.flow(%sc7, DMA : 1, %oK, DMA : 3)   // O_h7 relayed by score -> oK[3]
    aie.flow(%sh3, DMA : 1, %Klo, DMA : 0)   // K_lo -> Klo split
    aie.flow(%sh4, DMA : 1, %Khi, DMA : 0)   // K_hi -> Khi split
    aie.flow(%sh5, DMA : 1, %Vlo, DMA : 0)   // V_lo -> Vlo split
    aie.flow(%sh6, DMA : 1, %Vhi, DMA : 0)   // V_hi -> Vhi split
    aie.flow(%Klo, DMA : 0, %sc0, DMA : 1)   // K[0] -> score0
    aie.flow(%Khi, DMA : 0, %sc4, DMA : 1)   // K[4] -> score4
    aie.flow(%Vlo, DMA : 0, %va0, DMA : 1)   // V[0] -> value0
    aie.flow(%Vhi, DMA : 0, %va4, DMA : 1)   // V[4] -> value4
    aie.flow(%Klo, DMA : 1, %sc1, DMA : 1)   // K[1] -> score1
    aie.flow(%Khi, DMA : 1, %sc5, DMA : 1)   // K[5] -> score5
    aie.flow(%Vlo, DMA : 1, %va1, DMA : 1)   // V[1] -> value1
    aie.flow(%Vhi, DMA : 1, %va5, DMA : 1)   // V[5] -> value5
    aie.flow(%Klo, DMA : 2, %sc2, DMA : 1)   // K[2] -> score2
    aie.flow(%Khi, DMA : 2, %sc6, DMA : 1)   // K[6] -> score6
    aie.flow(%Vlo, DMA : 2, %va2, DMA : 1)   // V[2] -> value2
    aie.flow(%Vhi, DMA : 2, %va6, DMA : 1)   // V[6] -> value6
    aie.flow(%Klo, DMA : 3, %sc3, DMA : 1)   // K[3] -> score3
    aie.flow(%Khi, DMA : 3, %sc7, DMA : 1)   // K[7] -> score7
    aie.flow(%Vlo, DMA : 3, %va3, DMA : 1)   // V[3] -> value3
    aie.flow(%Vhi, DMA : 3, %va7, DMA : 1)   // V[7] -> value7
    aie.flow(%va0, DMA : 0, %jA, DMA : 0)   // Of0 -> joinA
    aie.flow(%va4, DMA : 0, %jB, DMA : 0)   // Of4 -> joinB
    aie.flow(%va1, DMA : 0, %jA, DMA : 1)   // Of1 -> joinA
    aie.flow(%va5, DMA : 0, %jB, DMA : 1)   // Of5 -> joinB
    aie.flow(%va2, DMA : 0, %jA, DMA : 2)   // Of2 -> joinA
    aie.flow(%va6, DMA : 0, %jB, DMA : 2)   // Of6 -> joinB
    aie.flow(%va3, DMA : 0, %jA, DMA : 3)   // Of3 -> joinA
    aie.flow(%va7, DMA : 0, %jB, DMA : 3)   // Of7 -> joinB
    aie.flow(%jA, DMA : 0, %rl, DMA : 0)   // attn Half0 -> relay
    aie.flow(%jB, DMA : 0, %rl, DMA : 1)   // attn Half1 -> relay
    aie.flow(%oJ, DMA : 0, %op, DMA : 0)   // O Half0 -> orelay
    aie.flow(%oK, DMA : 0, %op, DMA : 1)   // O Half1 -> orelay
    aie.flow(%op, DMA : 0, %nm, DMA : 0)   // O -> ANM
    aie.flow(%sh2, DMA : 1, %nm, DMA : 1)   // resid+gain -> ANM (2-BD on S2MM1)
    aie.flow(%nm, DMA : 1, %sh4, DMA : 1)   // s = O+resid (attn-residual) -> arg0 tail


    %c0_A0 = aie.buffer(%t0) {sym_name = "c0_A0"} : memref<4608xi8>
    %c0_A1 = aie.buffer(%t0) {sym_name = "c0_A1"} : memref<4608xi8>
    %c0_B0 = aie.buffer(%t0) {sym_name = "c0_B0"} : memref<2320xbf16>
    %c0_B1 = aie.buffer(%t0) {sym_name = "c0_B1"} : memref<2320xbf16>
    %c0_B2 = aie.buffer(%t0) {sym_name = "c0_B2"} : memref<2320xbf16>
    %c0_Q = aie.buffer(%t0) {sym_name = "c0_Q"} : memref<322xbf16>
    %c0_O = aie.buffer(%t0) {sym_name = "c0_O"} : memref<2048xbf16>
    %c0_P = aie.buffer(%t0) {sym_name = "c0_P"} : memref<2320xbf16>
    %c0_A0p = aie.lock(%t0, 0) {init = 1 : i32, sym_name = "c0_A0p"}
    %c0_A0c = aie.lock(%t0, 1) {init = 0 : i32, sym_name = "c0_A0c"}
    %c0_B0p = aie.lock(%t0, 2) {init = 1 : i32, sym_name = "c0_B0p"}
    %c0_B0c = aie.lock(%t0, 3) {init = 0 : i32, sym_name = "c0_B0c"}
    %c0_Qp = aie.lock(%t0, 4) {init = 1 : i32, sym_name = "c0_Qp"}
    %c0_Qc = aie.lock(%t0, 5) {init = 0 : i32, sym_name = "c0_Qc"}
    %c0_Op = aie.lock(%t0, 6) {init = 1 : i32, sym_name = "c0_Op"}
    %c0_Oc = aie.lock(%t0, 7) {init = 0 : i32, sym_name = "c0_Oc"}
    %c0_Pp = aie.lock(%t0, 8) {init = 1 : i32, sym_name = "c0_Pp"}
    %c0_Pc = aie.lock(%t0, 9) {init = 0 : i32, sym_name = "c0_Pc"}
    %c0_A1p = aie.lock(%t0, 10) {init = 1 : i32, sym_name = "c0_A1p"}
    %c0_A1c = aie.lock(%t0, 11) {init = 0 : i32, sym_name = "c0_A1c"}
    %c0_B1p = aie.lock(%t0, 12) {init = 1 : i32, sym_name = "c0_B1p"}
    %c0_B1c = aie.lock(%t0, 13) {init = 0 : i32, sym_name = "c0_B1c"}
    %c0_B2p = aie.lock(%t0, 14) {init = 1 : i32, sym_name = "c0_B2p"}
    %c0_B2c = aie.lock(%t0, 15) {init = 0 : i32, sym_name = "c0_B2c"}
    %c0_gate = aie.buffer(%t0) {sym_name = "c0_gate"} : memref<1024xbf16>
    %c0_up   = aie.buffer(%t0) {sym_name = "c0_up"}   : memref<1024xbf16>
    %c0_silu = aie.buffer(%t0) {sym_name = "c0_silu"} : memref<1024xbf16>
    %c0_uni_partial = aie.buffer(%t0) {sym_name = "c0_uni_partial"} : memref<128xi8>
    %core0 = aie.core(%t0) {
      %c0 = arith.constant 0 : index
      %cN = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      %c64 = arith.constant 64 : index
      %cF = arith.constant 256 : index
      %c4 = arith.constant 4 : i32
      %c8 = arith.constant 8 : i32
      %qr = arith.constant 256 : i32
      %hc = arith.constant 1024 : i32
      scf.for %tok = %c0 to %cN step %c1 {
        func.call @_ha_noop() : () -> ()
        // ---- phase 1: Q-GEMV + rope (B0 = x_bundle, nchunk=8) ----
        aie.use_lock(%c0_B0c, AcquireGreaterEqual, 1)
        aie.use_lock(%c0_Qp, AcquireGreaterEqual, 1)
        scf.for %j = %c0 to %c64 step %c2 {
          aie.use_lock(%c0_A0c, AcquireGreaterEqual, 1)
          %ji0 = arith.index_cast %j : index to i32
          func.call @generic_bcast_gemv_bf16_q(%ji0, %c0_A0, %c0_B0, %c0_uni_partial, %c8, %c0_Q) : (i32, memref<4608xi8>, memref<2320xbf16>, memref<128xi8>, i32, memref<322xbf16>) -> ()
          aie.use_lock(%c0_A0p, Release, 1)
          aie.use_lock(%c0_A1c, AcquireGreaterEqual, 1)
          %j1 = arith.addi %j, %c1 : index
          %ji1 = arith.index_cast %j1 : index to i32
          func.call @generic_bcast_gemv_bf16_q(%ji1, %c0_A1, %c0_B0, %c0_uni_partial, %c8, %c0_Q) : (i32, memref<4608xi8>, memref<2320xbf16>, memref<128xi8>, i32, memref<322xbf16>) -> ()
          aie.use_lock(%c0_A1p, Release, 1)
        }
        func.call @rope_bundled(%c0_Q, %c0_B0, %c0_Q, %qr) : (memref<322xbf16>, memref<2320xbf16>, memref<322xbf16>, i32) -> ()
        aie.use_lock(%c0_Qc, Release, 1)
        aie.use_lock(%c0_B0p, Release, 1)
        // ---- phase 2: O-proj (B1 = attn_out, nchunk=8) ----
        aie.use_lock(%c0_B1c, AcquireGreaterEqual, 1)
        aie.use_lock(%c0_Op, AcquireGreaterEqual, 1)
        scf.for %j = %c0 to %c64 step %c2 {
          aie.use_lock(%c0_A0c, AcquireGreaterEqual, 1)
          %ji0 = arith.index_cast %j : index to i32
          func.call @generic_bcast_gemv_bf16_o(%ji0, %c0_A0, %c0_B1, %c0_uni_partial, %c8, %c0_O) : (i32, memref<4608xi8>, memref<2320xbf16>, memref<128xi8>, i32, memref<2048xbf16>) -> ()
          aie.use_lock(%c0_A0p, Release, 1)
          aie.use_lock(%c0_A1c, AcquireGreaterEqual, 1)
          %j1 = arith.addi %j, %c1 : index
          %ji1 = arith.index_cast %j1 : index to i32
          func.call @generic_bcast_gemv_bf16_o(%ji1, %c0_A1, %c0_B1, %c0_uni_partial, %c8, %c0_O) : (i32, memref<4608xi8>, memref<2320xbf16>, memref<128xi8>, i32, memref<2048xbf16>) -> ()
          aie.use_lock(%c0_A1p, Release, 1)
        }
        aie.use_lock(%c0_B1p, Release, 1)
        aie.use_lock(%c0_Oc, Release, 1)
        // ---- phase 3: FFN (B2 = ffn_in) ----
        // 3a: gate GEMV (nchunk=8, output -> gate_buf)
        aie.use_lock(%c0_B2c, AcquireGreaterEqual, 1)
        scf.for %j = %c0 to %cF step %c2 {
          aie.use_lock(%c0_A0c, AcquireGreaterEqual, 1)
          %ji0 = arith.index_cast %j : index to i32
          func.call @generic_bcast_gemv_bf16_g(%ji0, %c0_A0, %c0_B2, %c0_uni_partial, %c8, %c0_gate) : (i32, memref<4608xi8>, memref<2320xbf16>, memref<128xi8>, i32, memref<1024xbf16>) -> ()
          aie.use_lock(%c0_A0p, Release, 1)
          aie.use_lock(%c0_A1c, AcquireGreaterEqual, 1)
          %j1 = arith.addi %j, %c1 : index
          %ji1 = arith.index_cast %j1 : index to i32
          func.call @generic_bcast_gemv_bf16_g(%ji1, %c0_A1, %c0_B2, %c0_uni_partial, %c8, %c0_gate) : (i32, memref<4608xi8>, memref<2320xbf16>, memref<128xi8>, i32, memref<1024xbf16>) -> ()
          aie.use_lock(%c0_A1p, Release, 1)
        }
        // 3b: up GEMV (nchunk=8, output -> up_buf)
        scf.for %j = %c0 to %cF step %c2 {
          aie.use_lock(%c0_A0c, AcquireGreaterEqual, 1)
          %ji0 = arith.index_cast %j : index to i32
          func.call @generic_bcast_gemv_bf16_g(%ji0, %c0_A0, %c0_B2, %c0_uni_partial, %c8, %c0_up) : (i32, memref<4608xi8>, memref<2320xbf16>, memref<128xi8>, i32, memref<1024xbf16>) -> ()
          aie.use_lock(%c0_A0p, Release, 1)
          aie.use_lock(%c0_A1c, AcquireGreaterEqual, 1)
          %j1 = arith.addi %j, %c1 : index
          %ji1 = arith.index_cast %j1 : index to i32
          func.call @generic_bcast_gemv_bf16_g(%ji1, %c0_A1, %c0_B2, %c0_uni_partial, %c8, %c0_up) : (i32, memref<4608xi8>, memref<2320xbf16>, memref<128xi8>, i32, memref<1024xbf16>) -> ()
          aie.use_lock(%c0_A1p, Release, 1)
        }
        aie.use_lock(%c0_B2p, Release, 1)
        // 3c: SiLU(gate) * up -> silu_buf (explicit pointers)
        func.call @layer_fused_silu_mul_explicit_bf16(%c0_gate, %c0_up, %c0_silu, %hc) : (memref<1024xbf16>, memref<1024xbf16>, memref<1024xbf16>, i32) -> ()
        // 3d: down GEMV (nchunk=4, activation = silu_buf)
        aie.use_lock(%c0_Pp, AcquireGreaterEqual, 1)
        scf.for %j = %c0 to %cF step %c2 {
          aie.use_lock(%c0_A0c, AcquireGreaterEqual, 1)
          %ji0 = arith.index_cast %j : index to i32
          func.call @generic_bcast_gemv_bf16_d(%ji0, %c0_A0, %c0_silu, %c0_uni_partial, %c4, %c0_P) : (i32, memref<4608xi8>, memref<1024xbf16>, memref<128xi8>, i32, memref<2320xbf16>) -> ()
          aie.use_lock(%c0_A0p, Release, 1)
          aie.use_lock(%c0_A1c, AcquireGreaterEqual, 1)
          %j1 = arith.addi %j, %c1 : index
          %ji1 = arith.index_cast %j1 : index to i32
          func.call @generic_bcast_gemv_bf16_d(%ji1, %c0_A1, %c0_silu, %c0_uni_partial, %c4, %c0_P) : (i32, memref<4608xi8>, memref<1024xbf16>, memref<128xi8>, i32, memref<2320xbf16>) -> ()
          aie.use_lock(%c0_A1p, Release, 1)
        }
        aie.use_lock(%c0_Pc, Release, 1)
      }
      aie.end
    }
    %mem_t0 = aie.mem(%t0) {
      %s0 = aie.dma_start(S2MM, 0, ^a00, ^bs0)
    ^a00:
      aie.use_lock(%c0_A0p, AcquireGreaterEqual, 1)
      aie.dma_bd(%c0_A0 : memref<4608xi8>, 0, 4608)
      aie.use_lock(%c0_A0c, Release, 1)
      aie.next_bd ^a10
    ^a10:
      aie.use_lock(%c0_A1p, AcquireGreaterEqual, 1)
      aie.dma_bd(%c0_A1 : memref<4608xi8>, 0, 4608)
      aie.use_lock(%c0_A1c, Release, 1)
      aie.next_bd ^a00
    ^bs0:
      %s1 = aie.dma_start(S2MM, 1, ^b00, ^m00)
    ^b00:
      aie.use_lock(%c0_B0p, AcquireGreaterEqual, 1)
      aie.dma_bd(%c0_B0 : memref<2320xbf16>, 0, 2320)
      aie.use_lock(%c0_B0c, Release, 1)
      aie.next_bd ^b10
    ^b10:
      aie.use_lock(%c0_B1p, AcquireGreaterEqual, 1)
      aie.dma_bd(%c0_B1 : memref<2320xbf16>, 0, 2320)
      aie.use_lock(%c0_B1c, Release, 1)
      aie.next_bd ^b20
    ^b20:
      aie.use_lock(%c0_B2p, AcquireGreaterEqual, 1)
      aie.dma_bd(%c0_B2 : memref<2320xbf16>, 0, 2320)
      aie.use_lock(%c0_B2c, Release, 1)
      aie.next_bd ^b00
    ^m00:
      %m0 = aie.dma_start(MM2S, 0, ^pf0, ^qo0)
    ^pf0:
      aie.use_lock(%c0_Pc, AcquireGreaterEqual, 1)
      aie.dma_bd(%c0_P : memref<2320xbf16>, 0, 2048)
      aie.use_lock(%c0_Pp, Release, 1)
      aie.next_bd ^pf0
    ^qo0:
      %m1 = aie.dma_start(MM2S, 1, ^qi0, ^e0)
    ^qi0:
      aie.use_lock(%c0_Qc, AcquireGreaterEqual, 1)
      aie.dma_bd(%c0_Q : memref<322xbf16>, 0, 322)
      aie.use_lock(%c0_Qp, Release, 1)
      aie.next_bd ^oh0
    ^oh0:
      aie.use_lock(%c0_Oc, AcquireGreaterEqual, 1)
      aie.dma_bd(%c0_O : memref<2048xbf16>, 0, 256)
      aie.use_lock(%c0_Op, Release, 1)
      aie.next_bd ^qi0
    ^e0:
      aie.end
    }
    %c1_A0 = aie.buffer(%t1) {sym_name = "c1_A0"} : memref<4608xi8>
    %c1_A1 = aie.buffer(%t1) {sym_name = "c1_A1"} : memref<4608xi8>
    %c1_B0 = aie.buffer(%t1) {sym_name = "c1_B0"} : memref<2320xbf16>
    %c1_B1 = aie.buffer(%t1) {sym_name = "c1_B1"} : memref<2320xbf16>
    %c1_B2 = aie.buffer(%t1) {sym_name = "c1_B2"} : memref<2320xbf16>
    %c1_Q = aie.buffer(%t1) {sym_name = "c1_Q"} : memref<322xbf16>
    %c1_O = aie.buffer(%t1) {sym_name = "c1_O"} : memref<2048xbf16>
    %c1_P = aie.buffer(%t1) {sym_name = "c1_P"} : memref<2320xbf16>
    %c1_A0p = aie.lock(%t1, 0) {init = 1 : i32, sym_name = "c1_A0p"}
    %c1_A0c = aie.lock(%t1, 1) {init = 0 : i32, sym_name = "c1_A0c"}
    %c1_B0p = aie.lock(%t1, 2) {init = 1 : i32, sym_name = "c1_B0p"}
    %c1_B0c = aie.lock(%t1, 3) {init = 0 : i32, sym_name = "c1_B0c"}
    %c1_Qp = aie.lock(%t1, 4) {init = 1 : i32, sym_name = "c1_Qp"}
    %c1_Qc = aie.lock(%t1, 5) {init = 0 : i32, sym_name = "c1_Qc"}
    %c1_Op = aie.lock(%t1, 6) {init = 1 : i32, sym_name = "c1_Op"}
    %c1_Oc = aie.lock(%t1, 7) {init = 0 : i32, sym_name = "c1_Oc"}
    %c1_Pp = aie.lock(%t1, 8) {init = 1 : i32, sym_name = "c1_Pp"}
    %c1_Pc = aie.lock(%t1, 9) {init = 0 : i32, sym_name = "c1_Pc"}
    %c1_A1p = aie.lock(%t1, 10) {init = 1 : i32, sym_name = "c1_A1p"}
    %c1_A1c = aie.lock(%t1, 11) {init = 0 : i32, sym_name = "c1_A1c"}
    %c1_B1p = aie.lock(%t1, 12) {init = 1 : i32, sym_name = "c1_B1p"}
    %c1_B1c = aie.lock(%t1, 13) {init = 0 : i32, sym_name = "c1_B1c"}
    %c1_B2p = aie.lock(%t1, 14) {init = 1 : i32, sym_name = "c1_B2p"}
    %c1_B2c = aie.lock(%t1, 15) {init = 0 : i32, sym_name = "c1_B2c"}
    %c1_gate = aie.buffer(%t1) {sym_name = "c1_gate"} : memref<1024xbf16>
    %c1_up   = aie.buffer(%t1) {sym_name = "c1_up"}   : memref<1024xbf16>
    %c1_silu = aie.buffer(%t1) {sym_name = "c1_silu"} : memref<1024xbf16>
    %c1_uni_partial = aie.buffer(%t1) {sym_name = "c1_uni_partial"} : memref<128xi8>
    %core1 = aie.core(%t1) {
      %c0 = arith.constant 0 : index
      %cN = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      %c64 = arith.constant 64 : index
      %cF = arith.constant 256 : index
      %c4 = arith.constant 4 : i32
      %c8 = arith.constant 8 : i32
      %qr = arith.constant 256 : i32
      %hc = arith.constant 1024 : i32
      scf.for %tok = %c0 to %cN step %c1 {
        func.call @_ha_noop() : () -> ()
        // ---- phase 1: Q-GEMV + rope (B0 = x_bundle, nchunk=8) ----
        aie.use_lock(%c1_B0c, AcquireGreaterEqual, 1)
        aie.use_lock(%c1_Qp, AcquireGreaterEqual, 1)
        scf.for %j = %c0 to %c64 step %c2 {
          aie.use_lock(%c1_A0c, AcquireGreaterEqual, 1)
          %ji0 = arith.index_cast %j : index to i32
          func.call @generic_bcast_gemv_bf16_q(%ji0, %c1_A0, %c1_B0, %c1_uni_partial, %c8, %c1_Q) : (i32, memref<4608xi8>, memref<2320xbf16>, memref<128xi8>, i32, memref<322xbf16>) -> ()
          aie.use_lock(%c1_A0p, Release, 1)
          aie.use_lock(%c1_A1c, AcquireGreaterEqual, 1)
          %j1 = arith.addi %j, %c1 : index
          %ji1 = arith.index_cast %j1 : index to i32
          func.call @generic_bcast_gemv_bf16_q(%ji1, %c1_A1, %c1_B0, %c1_uni_partial, %c8, %c1_Q) : (i32, memref<4608xi8>, memref<2320xbf16>, memref<128xi8>, i32, memref<322xbf16>) -> ()
          aie.use_lock(%c1_A1p, Release, 1)
        }
        func.call @rope_bundled(%c1_Q, %c1_B0, %c1_Q, %qr) : (memref<322xbf16>, memref<2320xbf16>, memref<322xbf16>, i32) -> ()
        aie.use_lock(%c1_Qc, Release, 1)
        aie.use_lock(%c1_B0p, Release, 1)
        // ---- phase 2: O-proj (B1 = attn_out, nchunk=8) ----
        aie.use_lock(%c1_B1c, AcquireGreaterEqual, 1)
        aie.use_lock(%c1_Op, AcquireGreaterEqual, 1)
        scf.for %j = %c0 to %c64 step %c2 {
          aie.use_lock(%c1_A0c, AcquireGreaterEqual, 1)
          %ji0 = arith.index_cast %j : index to i32
          func.call @generic_bcast_gemv_bf16_o(%ji0, %c1_A0, %c1_B1, %c1_uni_partial, %c8, %c1_O) : (i32, memref<4608xi8>, memref<2320xbf16>, memref<128xi8>, i32, memref<2048xbf16>) -> ()
          aie.use_lock(%c1_A0p, Release, 1)
          aie.use_lock(%c1_A1c, AcquireGreaterEqual, 1)
          %j1 = arith.addi %j, %c1 : index
          %ji1 = arith.index_cast %j1 : index to i32
          func.call @generic_bcast_gemv_bf16_o(%ji1, %c1_A1, %c1_B1, %c1_uni_partial, %c8, %c1_O) : (i32, memref<4608xi8>, memref<2320xbf16>, memref<128xi8>, i32, memref<2048xbf16>) -> ()
          aie.use_lock(%c1_A1p, Release, 1)
        }
        aie.use_lock(%c1_B1p, Release, 1)
        aie.use_lock(%c1_Oc, Release, 1)
        // ---- phase 3: FFN (B2 = ffn_in) ----
        // 3a: gate GEMV (nchunk=8, output -> gate_buf)
        aie.use_lock(%c1_B2c, AcquireGreaterEqual, 1)
        scf.for %j = %c0 to %cF step %c2 {
          aie.use_lock(%c1_A0c, AcquireGreaterEqual, 1)
          %ji0 = arith.index_cast %j : index to i32
          func.call @generic_bcast_gemv_bf16_g(%ji0, %c1_A0, %c1_B2, %c1_uni_partial, %c8, %c1_gate) : (i32, memref<4608xi8>, memref<2320xbf16>, memref<128xi8>, i32, memref<1024xbf16>) -> ()
          aie.use_lock(%c1_A0p, Release, 1)
          aie.use_lock(%c1_A1c, AcquireGreaterEqual, 1)
          %j1 = arith.addi %j, %c1 : index
          %ji1 = arith.index_cast %j1 : index to i32
          func.call @generic_bcast_gemv_bf16_g(%ji1, %c1_A1, %c1_B2, %c1_uni_partial, %c8, %c1_gate) : (i32, memref<4608xi8>, memref<2320xbf16>, memref<128xi8>, i32, memref<1024xbf16>) -> ()
          aie.use_lock(%c1_A1p, Release, 1)
        }
        // 3b: up GEMV (nchunk=8, output -> up_buf)
        scf.for %j = %c0 to %cF step %c2 {
          aie.use_lock(%c1_A0c, AcquireGreaterEqual, 1)
          %ji0 = arith.index_cast %j : index to i32
          func.call @generic_bcast_gemv_bf16_g(%ji0, %c1_A0, %c1_B2, %c1_uni_partial, %c8, %c1_up) : (i32, memref<4608xi8>, memref<2320xbf16>, memref<128xi8>, i32, memref<1024xbf16>) -> ()
          aie.use_lock(%c1_A0p, Release, 1)
          aie.use_lock(%c1_A1c, AcquireGreaterEqual, 1)
          %j1 = arith.addi %j, %c1 : index
          %ji1 = arith.index_cast %j1 : index to i32
          func.call @generic_bcast_gemv_bf16_g(%ji1, %c1_A1, %c1_B2, %c1_uni_partial, %c8, %c1_up) : (i32, memref<4608xi8>, memref<2320xbf16>, memref<128xi8>, i32, memref<1024xbf16>) -> ()
          aie.use_lock(%c1_A1p, Release, 1)
        }
        aie.use_lock(%c1_B2p, Release, 1)
        // 3c: SiLU(gate) * up -> silu_buf (explicit pointers)
        func.call @layer_fused_silu_mul_explicit_bf16(%c1_gate, %c1_up, %c1_silu, %hc) : (memref<1024xbf16>, memref<1024xbf16>, memref<1024xbf16>, i32) -> ()
        // 3d: down GEMV (nchunk=4, activation = silu_buf)
        aie.use_lock(%c1_Pp, AcquireGreaterEqual, 1)
        scf.for %j = %c0 to %cF step %c2 {
          aie.use_lock(%c1_A0c, AcquireGreaterEqual, 1)
          %ji0 = arith.index_cast %j : index to i32
          func.call @generic_bcast_gemv_bf16_d(%ji0, %c1_A0, %c1_silu, %c1_uni_partial, %c4, %c1_P) : (i32, memref<4608xi8>, memref<1024xbf16>, memref<128xi8>, i32, memref<2320xbf16>) -> ()
          aie.use_lock(%c1_A0p, Release, 1)
          aie.use_lock(%c1_A1c, AcquireGreaterEqual, 1)
          %j1 = arith.addi %j, %c1 : index
          %ji1 = arith.index_cast %j1 : index to i32
          func.call @generic_bcast_gemv_bf16_d(%ji1, %c1_A1, %c1_silu, %c1_uni_partial, %c4, %c1_P) : (i32, memref<4608xi8>, memref<1024xbf16>, memref<128xi8>, i32, memref<2320xbf16>) -> ()
          aie.use_lock(%c1_A1p, Release, 1)
        }
        aie.use_lock(%c1_Pc, Release, 1)
      }
      aie.end
    }
    %mem_t1 = aie.mem(%t1) {
      %s0 = aie.dma_start(S2MM, 0, ^a01, ^bs1)
    ^a01:
      aie.use_lock(%c1_A0p, AcquireGreaterEqual, 1)
      aie.dma_bd(%c1_A0 : memref<4608xi8>, 0, 4608)
      aie.use_lock(%c1_A0c, Release, 1)
      aie.next_bd ^a11
    ^a11:
      aie.use_lock(%c1_A1p, AcquireGreaterEqual, 1)
      aie.dma_bd(%c1_A1 : memref<4608xi8>, 0, 4608)
      aie.use_lock(%c1_A1c, Release, 1)
      aie.next_bd ^a01
    ^bs1:
      %s1 = aie.dma_start(S2MM, 1, ^b01, ^m01)
    ^b01:
      aie.use_lock(%c1_B0p, AcquireGreaterEqual, 1)
      aie.dma_bd(%c1_B0 : memref<2320xbf16>, 0, 2320)
      aie.use_lock(%c1_B0c, Release, 1)
      aie.next_bd ^b11
    ^b11:
      aie.use_lock(%c1_B1p, AcquireGreaterEqual, 1)
      aie.dma_bd(%c1_B1 : memref<2320xbf16>, 0, 2320)
      aie.use_lock(%c1_B1c, Release, 1)
      aie.next_bd ^b21
    ^b21:
      aie.use_lock(%c1_B2p, AcquireGreaterEqual, 1)
      aie.dma_bd(%c1_B2 : memref<2320xbf16>, 0, 2320)
      aie.use_lock(%c1_B2c, Release, 1)
      aie.next_bd ^b01
    ^m01:
      %m0 = aie.dma_start(MM2S, 0, ^pf1, ^qo1)
    ^pf1:
      aie.use_lock(%c1_Pc, AcquireGreaterEqual, 1)
      aie.dma_bd(%c1_P : memref<2320xbf16>, 0, 2048)
      aie.use_lock(%c1_Pp, Release, 1)
      aie.next_bd ^pf1
    ^qo1:
      %m1 = aie.dma_start(MM2S, 1, ^qi1, ^e1)
    ^qi1:
      aie.use_lock(%c1_Qc, AcquireGreaterEqual, 1)
      aie.dma_bd(%c1_Q : memref<322xbf16>, 0, 322)
      aie.use_lock(%c1_Qp, Release, 1)
      aie.next_bd ^oh1
    ^oh1:
      aie.use_lock(%c1_Oc, AcquireGreaterEqual, 1)
      aie.dma_bd(%c1_O : memref<2048xbf16>, 0, 256)
      aie.use_lock(%c1_Op, Release, 1)
      aie.next_bd ^qi1
    ^e1:
      aie.end
    }
    %c2_A0 = aie.buffer(%t2) {sym_name = "c2_A0"} : memref<4608xi8>
    %c2_A1 = aie.buffer(%t2) {sym_name = "c2_A1"} : memref<4608xi8>
    %c2_B0 = aie.buffer(%t2) {sym_name = "c2_B0"} : memref<2320xbf16>
    %c2_B1 = aie.buffer(%t2) {sym_name = "c2_B1"} : memref<2320xbf16>
    %c2_B2 = aie.buffer(%t2) {sym_name = "c2_B2"} : memref<2320xbf16>
    %c2_Q = aie.buffer(%t2) {sym_name = "c2_Q"} : memref<322xbf16>
    %c2_O = aie.buffer(%t2) {sym_name = "c2_O"} : memref<2048xbf16>
    %c2_P = aie.buffer(%t2) {sym_name = "c2_P"} : memref<2320xbf16>
    %c2_A0p = aie.lock(%t2, 0) {init = 1 : i32, sym_name = "c2_A0p"}
    %c2_A0c = aie.lock(%t2, 1) {init = 0 : i32, sym_name = "c2_A0c"}
    %c2_B0p = aie.lock(%t2, 2) {init = 1 : i32, sym_name = "c2_B0p"}
    %c2_B0c = aie.lock(%t2, 3) {init = 0 : i32, sym_name = "c2_B0c"}
    %c2_Qp = aie.lock(%t2, 4) {init = 1 : i32, sym_name = "c2_Qp"}
    %c2_Qc = aie.lock(%t2, 5) {init = 0 : i32, sym_name = "c2_Qc"}
    %c2_Op = aie.lock(%t2, 6) {init = 1 : i32, sym_name = "c2_Op"}
    %c2_Oc = aie.lock(%t2, 7) {init = 0 : i32, sym_name = "c2_Oc"}
    %c2_Pp = aie.lock(%t2, 8) {init = 1 : i32, sym_name = "c2_Pp"}
    %c2_Pc = aie.lock(%t2, 9) {init = 0 : i32, sym_name = "c2_Pc"}
    %c2_A1p = aie.lock(%t2, 10) {init = 1 : i32, sym_name = "c2_A1p"}
    %c2_A1c = aie.lock(%t2, 11) {init = 0 : i32, sym_name = "c2_A1c"}
    %c2_B1p = aie.lock(%t2, 12) {init = 1 : i32, sym_name = "c2_B1p"}
    %c2_B1c = aie.lock(%t2, 13) {init = 0 : i32, sym_name = "c2_B1c"}
    %c2_B2p = aie.lock(%t2, 14) {init = 1 : i32, sym_name = "c2_B2p"}
    %c2_B2c = aie.lock(%t2, 15) {init = 0 : i32, sym_name = "c2_B2c"}
    %c2_gate = aie.buffer(%t2) {sym_name = "c2_gate"} : memref<1024xbf16>
    %c2_up   = aie.buffer(%t2) {sym_name = "c2_up"}   : memref<1024xbf16>
    %c2_silu = aie.buffer(%t2) {sym_name = "c2_silu"} : memref<1024xbf16>
    %c2_uni_partial = aie.buffer(%t2) {sym_name = "c2_uni_partial"} : memref<128xi8>
    %core2 = aie.core(%t2) {
      %c0 = arith.constant 0 : index
      %cN = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      %c64 = arith.constant 64 : index
      %cF = arith.constant 256 : index
      %c4 = arith.constant 4 : i32
      %c8 = arith.constant 8 : i32
      %qr = arith.constant 256 : i32
      %hc = arith.constant 1024 : i32
      scf.for %tok = %c0 to %cN step %c1 {
        func.call @_ha_noop() : () -> ()
        // ---- phase 1: Q-GEMV + rope (B0 = x_bundle, nchunk=8) ----
        aie.use_lock(%c2_B0c, AcquireGreaterEqual, 1)
        aie.use_lock(%c2_Qp, AcquireGreaterEqual, 1)
        scf.for %j = %c0 to %c64 step %c2 {
          aie.use_lock(%c2_A0c, AcquireGreaterEqual, 1)
          %ji0 = arith.index_cast %j : index to i32
          func.call @generic_bcast_gemv_bf16_q(%ji0, %c2_A0, %c2_B0, %c2_uni_partial, %c8, %c2_Q) : (i32, memref<4608xi8>, memref<2320xbf16>, memref<128xi8>, i32, memref<322xbf16>) -> ()
          aie.use_lock(%c2_A0p, Release, 1)
          aie.use_lock(%c2_A1c, AcquireGreaterEqual, 1)
          %j1 = arith.addi %j, %c1 : index
          %ji1 = arith.index_cast %j1 : index to i32
          func.call @generic_bcast_gemv_bf16_q(%ji1, %c2_A1, %c2_B0, %c2_uni_partial, %c8, %c2_Q) : (i32, memref<4608xi8>, memref<2320xbf16>, memref<128xi8>, i32, memref<322xbf16>) -> ()
          aie.use_lock(%c2_A1p, Release, 1)
        }
        func.call @rope_bundled(%c2_Q, %c2_B0, %c2_Q, %qr) : (memref<322xbf16>, memref<2320xbf16>, memref<322xbf16>, i32) -> ()
        aie.use_lock(%c2_Qc, Release, 1)
        aie.use_lock(%c2_B0p, Release, 1)
        // ---- phase 2: O-proj (B1 = attn_out, nchunk=8) ----
        aie.use_lock(%c2_B1c, AcquireGreaterEqual, 1)
        aie.use_lock(%c2_Op, AcquireGreaterEqual, 1)
        scf.for %j = %c0 to %c64 step %c2 {
          aie.use_lock(%c2_A0c, AcquireGreaterEqual, 1)
          %ji0 = arith.index_cast %j : index to i32
          func.call @generic_bcast_gemv_bf16_o(%ji0, %c2_A0, %c2_B1, %c2_uni_partial, %c8, %c2_O) : (i32, memref<4608xi8>, memref<2320xbf16>, memref<128xi8>, i32, memref<2048xbf16>) -> ()
          aie.use_lock(%c2_A0p, Release, 1)
          aie.use_lock(%c2_A1c, AcquireGreaterEqual, 1)
          %j1 = arith.addi %j, %c1 : index
          %ji1 = arith.index_cast %j1 : index to i32
          func.call @generic_bcast_gemv_bf16_o(%ji1, %c2_A1, %c2_B1, %c2_uni_partial, %c8, %c2_O) : (i32, memref<4608xi8>, memref<2320xbf16>, memref<128xi8>, i32, memref<2048xbf16>) -> ()
          aie.use_lock(%c2_A1p, Release, 1)
        }
        aie.use_lock(%c2_B1p, Release, 1)
        aie.use_lock(%c2_Oc, Release, 1)
        // ---- phase 3: FFN (B2 = ffn_in) ----
        // 3a: gate GEMV (nchunk=8, output -> gate_buf)
        aie.use_lock(%c2_B2c, AcquireGreaterEqual, 1)
        scf.for %j = %c0 to %cF step %c2 {
          aie.use_lock(%c2_A0c, AcquireGreaterEqual, 1)
          %ji0 = arith.index_cast %j : index to i32
          func.call @generic_bcast_gemv_bf16_g(%ji0, %c2_A0, %c2_B2, %c2_uni_partial, %c8, %c2_gate) : (i32, memref<4608xi8>, memref<2320xbf16>, memref<128xi8>, i32, memref<1024xbf16>) -> ()
          aie.use_lock(%c2_A0p, Release, 1)
          aie.use_lock(%c2_A1c, AcquireGreaterEqual, 1)
          %j1 = arith.addi %j, %c1 : index
          %ji1 = arith.index_cast %j1 : index to i32
          func.call @generic_bcast_gemv_bf16_g(%ji1, %c2_A1, %c2_B2, %c2_uni_partial, %c8, %c2_gate) : (i32, memref<4608xi8>, memref<2320xbf16>, memref<128xi8>, i32, memref<1024xbf16>) -> ()
          aie.use_lock(%c2_A1p, Release, 1)
        }
        // 3b: up GEMV (nchunk=8, output -> up_buf)
        scf.for %j = %c0 to %cF step %c2 {
          aie.use_lock(%c2_A0c, AcquireGreaterEqual, 1)
          %ji0 = arith.index_cast %j : index to i32
          func.call @generic_bcast_gemv_bf16_g(%ji0, %c2_A0, %c2_B2, %c2_uni_partial, %c8, %c2_up) : (i32, memref<4608xi8>, memref<2320xbf16>, memref<128xi8>, i32, memref<1024xbf16>) -> ()
          aie.use_lock(%c2_A0p, Release, 1)
          aie.use_lock(%c2_A1c, AcquireGreaterEqual, 1)
          %j1 = arith.addi %j, %c1 : index
          %ji1 = arith.index_cast %j1 : index to i32
          func.call @generic_bcast_gemv_bf16_g(%ji1, %c2_A1, %c2_B2, %c2_uni_partial, %c8, %c2_up) : (i32, memref<4608xi8>, memref<2320xbf16>, memref<128xi8>, i32, memref<1024xbf16>) -> ()
          aie.use_lock(%c2_A1p, Release, 1)
        }
        aie.use_lock(%c2_B2p, Release, 1)
        // 3c: SiLU(gate) * up -> silu_buf (explicit pointers)
        func.call @layer_fused_silu_mul_explicit_bf16(%c2_gate, %c2_up, %c2_silu, %hc) : (memref<1024xbf16>, memref<1024xbf16>, memref<1024xbf16>, i32) -> ()
        // 3d: down GEMV (nchunk=4, activation = silu_buf)
        aie.use_lock(%c2_Pp, AcquireGreaterEqual, 1)
        scf.for %j = %c0 to %cF step %c2 {
          aie.use_lock(%c2_A0c, AcquireGreaterEqual, 1)
          %ji0 = arith.index_cast %j : index to i32
          func.call @generic_bcast_gemv_bf16_d(%ji0, %c2_A0, %c2_silu, %c2_uni_partial, %c4, %c2_P) : (i32, memref<4608xi8>, memref<1024xbf16>, memref<128xi8>, i32, memref<2320xbf16>) -> ()
          aie.use_lock(%c2_A0p, Release, 1)
          aie.use_lock(%c2_A1c, AcquireGreaterEqual, 1)
          %j1 = arith.addi %j, %c1 : index
          %ji1 = arith.index_cast %j1 : index to i32
          func.call @generic_bcast_gemv_bf16_d(%ji1, %c2_A1, %c2_silu, %c2_uni_partial, %c4, %c2_P) : (i32, memref<4608xi8>, memref<1024xbf16>, memref<128xi8>, i32, memref<2320xbf16>) -> ()
          aie.use_lock(%c2_A1p, Release, 1)
        }
        aie.use_lock(%c2_Pc, Release, 1)
      }
      aie.end
    }
    %mem_t2 = aie.mem(%t2) {
      %s0 = aie.dma_start(S2MM, 0, ^a02, ^bs2)
    ^a02:
      aie.use_lock(%c2_A0p, AcquireGreaterEqual, 1)
      aie.dma_bd(%c2_A0 : memref<4608xi8>, 0, 4608)
      aie.use_lock(%c2_A0c, Release, 1)
      aie.next_bd ^a12
    ^a12:
      aie.use_lock(%c2_A1p, AcquireGreaterEqual, 1)
      aie.dma_bd(%c2_A1 : memref<4608xi8>, 0, 4608)
      aie.use_lock(%c2_A1c, Release, 1)
      aie.next_bd ^a02
    ^bs2:
      %s1 = aie.dma_start(S2MM, 1, ^b02, ^m02)
    ^b02:
      aie.use_lock(%c2_B0p, AcquireGreaterEqual, 1)
      aie.dma_bd(%c2_B0 : memref<2320xbf16>, 0, 2320)
      aie.use_lock(%c2_B0c, Release, 1)
      aie.next_bd ^b12
    ^b12:
      aie.use_lock(%c2_B1p, AcquireGreaterEqual, 1)
      aie.dma_bd(%c2_B1 : memref<2320xbf16>, 0, 2320)
      aie.use_lock(%c2_B1c, Release, 1)
      aie.next_bd ^b22
    ^b22:
      aie.use_lock(%c2_B2p, AcquireGreaterEqual, 1)
      aie.dma_bd(%c2_B2 : memref<2320xbf16>, 0, 2320)
      aie.use_lock(%c2_B2c, Release, 1)
      aie.next_bd ^b02
    ^m02:
      %m0 = aie.dma_start(MM2S, 0, ^pf2, ^qo2)
    ^pf2:
      aie.use_lock(%c2_Pc, AcquireGreaterEqual, 1)
      aie.dma_bd(%c2_P : memref<2320xbf16>, 0, 2048)
      aie.use_lock(%c2_Pp, Release, 1)
      aie.next_bd ^pf2
    ^qo2:
      %m1 = aie.dma_start(MM2S, 1, ^qi2, ^e2)
    ^qi2:
      aie.use_lock(%c2_Qc, AcquireGreaterEqual, 1)
      aie.dma_bd(%c2_Q : memref<322xbf16>, 0, 322)
      aie.use_lock(%c2_Qp, Release, 1)
      aie.next_bd ^oh2
    ^oh2:
      aie.use_lock(%c2_Oc, AcquireGreaterEqual, 1)
      aie.dma_bd(%c2_O : memref<2048xbf16>, 0, 256)
      aie.use_lock(%c2_Op, Release, 1)
      aie.next_bd ^qi2
    ^e2:
      aie.end
    }
    %c3_A0 = aie.buffer(%t3) {sym_name = "c3_A0"} : memref<4608xi8>
    %c3_A1 = aie.buffer(%t3) {sym_name = "c3_A1"} : memref<4608xi8>
    %c3_B0 = aie.buffer(%t3) {sym_name = "c3_B0"} : memref<2320xbf16>
    %c3_B1 = aie.buffer(%t3) {sym_name = "c3_B1"} : memref<2320xbf16>
    %c3_B2 = aie.buffer(%t3) {sym_name = "c3_B2"} : memref<2320xbf16>
    %c3_Q = aie.buffer(%t3) {sym_name = "c3_Q"} : memref<322xbf16>
    %c3_O = aie.buffer(%t3) {sym_name = "c3_O"} : memref<2048xbf16>
    %c3_P = aie.buffer(%t3) {sym_name = "c3_P"} : memref<2320xbf16>
    %c3_A0p = aie.lock(%t3, 0) {init = 1 : i32, sym_name = "c3_A0p"}
    %c3_A0c = aie.lock(%t3, 1) {init = 0 : i32, sym_name = "c3_A0c"}
    %c3_B0p = aie.lock(%t3, 2) {init = 1 : i32, sym_name = "c3_B0p"}
    %c3_B0c = aie.lock(%t3, 3) {init = 0 : i32, sym_name = "c3_B0c"}
    %c3_Qp = aie.lock(%t3, 4) {init = 1 : i32, sym_name = "c3_Qp"}
    %c3_Qc = aie.lock(%t3, 5) {init = 0 : i32, sym_name = "c3_Qc"}
    %c3_Op = aie.lock(%t3, 6) {init = 1 : i32, sym_name = "c3_Op"}
    %c3_Oc = aie.lock(%t3, 7) {init = 0 : i32, sym_name = "c3_Oc"}
    %c3_Pp = aie.lock(%t3, 8) {init = 1 : i32, sym_name = "c3_Pp"}
    %c3_Pc = aie.lock(%t3, 9) {init = 0 : i32, sym_name = "c3_Pc"}
    %c3_A1p = aie.lock(%t3, 10) {init = 1 : i32, sym_name = "c3_A1p"}
    %c3_A1c = aie.lock(%t3, 11) {init = 0 : i32, sym_name = "c3_A1c"}
    %c3_B1p = aie.lock(%t3, 12) {init = 1 : i32, sym_name = "c3_B1p"}
    %c3_B1c = aie.lock(%t3, 13) {init = 0 : i32, sym_name = "c3_B1c"}
    %c3_B2p = aie.lock(%t3, 14) {init = 1 : i32, sym_name = "c3_B2p"}
    %c3_B2c = aie.lock(%t3, 15) {init = 0 : i32, sym_name = "c3_B2c"}
    %c3_gate = aie.buffer(%t3) {sym_name = "c3_gate"} : memref<1024xbf16>
    %c3_up   = aie.buffer(%t3) {sym_name = "c3_up"}   : memref<1024xbf16>
    %c3_silu = aie.buffer(%t3) {sym_name = "c3_silu"} : memref<1024xbf16>
    %c3_uni_partial = aie.buffer(%t3) {sym_name = "c3_uni_partial"} : memref<128xi8>
    %core3 = aie.core(%t3) {
      %c0 = arith.constant 0 : index
      %cN = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      %c64 = arith.constant 64 : index
      %cF = arith.constant 256 : index
      %c4 = arith.constant 4 : i32
      %c8 = arith.constant 8 : i32
      %qr = arith.constant 256 : i32
      %hc = arith.constant 1024 : i32
      scf.for %tok = %c0 to %cN step %c1 {
        func.call @_ha_noop() : () -> ()
        // ---- phase 1: Q-GEMV + rope (B0 = x_bundle, nchunk=8) ----
        aie.use_lock(%c3_B0c, AcquireGreaterEqual, 1)
        aie.use_lock(%c3_Qp, AcquireGreaterEqual, 1)
        scf.for %j = %c0 to %c64 step %c2 {
          aie.use_lock(%c3_A0c, AcquireGreaterEqual, 1)
          %ji0 = arith.index_cast %j : index to i32
          func.call @generic_bcast_gemv_bf16_q(%ji0, %c3_A0, %c3_B0, %c3_uni_partial, %c8, %c3_Q) : (i32, memref<4608xi8>, memref<2320xbf16>, memref<128xi8>, i32, memref<322xbf16>) -> ()
          aie.use_lock(%c3_A0p, Release, 1)
          aie.use_lock(%c3_A1c, AcquireGreaterEqual, 1)
          %j1 = arith.addi %j, %c1 : index
          %ji1 = arith.index_cast %j1 : index to i32
          func.call @generic_bcast_gemv_bf16_q(%ji1, %c3_A1, %c3_B0, %c3_uni_partial, %c8, %c3_Q) : (i32, memref<4608xi8>, memref<2320xbf16>, memref<128xi8>, i32, memref<322xbf16>) -> ()
          aie.use_lock(%c3_A1p, Release, 1)
        }
        func.call @rope_bundled(%c3_Q, %c3_B0, %c3_Q, %qr) : (memref<322xbf16>, memref<2320xbf16>, memref<322xbf16>, i32) -> ()
        aie.use_lock(%c3_Qc, Release, 1)
        aie.use_lock(%c3_B0p, Release, 1)
        // ---- phase 2: O-proj (B1 = attn_out, nchunk=8) ----
        aie.use_lock(%c3_B1c, AcquireGreaterEqual, 1)
        aie.use_lock(%c3_Op, AcquireGreaterEqual, 1)
        scf.for %j = %c0 to %c64 step %c2 {
          aie.use_lock(%c3_A0c, AcquireGreaterEqual, 1)
          %ji0 = arith.index_cast %j : index to i32
          func.call @generic_bcast_gemv_bf16_o(%ji0, %c3_A0, %c3_B1, %c3_uni_partial, %c8, %c3_O) : (i32, memref<4608xi8>, memref<2320xbf16>, memref<128xi8>, i32, memref<2048xbf16>) -> ()
          aie.use_lock(%c3_A0p, Release, 1)
          aie.use_lock(%c3_A1c, AcquireGreaterEqual, 1)
          %j1 = arith.addi %j, %c1 : index
          %ji1 = arith.index_cast %j1 : index to i32
          func.call @generic_bcast_gemv_bf16_o(%ji1, %c3_A1, %c3_B1, %c3_uni_partial, %c8, %c3_O) : (i32, memref<4608xi8>, memref<2320xbf16>, memref<128xi8>, i32, memref<2048xbf16>) -> ()
          aie.use_lock(%c3_A1p, Release, 1)
        }
        aie.use_lock(%c3_B1p, Release, 1)
        aie.use_lock(%c3_Oc, Release, 1)
        // ---- phase 3: FFN (B2 = ffn_in) ----
        // 3a: gate GEMV (nchunk=8, output -> gate_buf)
        aie.use_lock(%c3_B2c, AcquireGreaterEqual, 1)
        scf.for %j = %c0 to %cF step %c2 {
          aie.use_lock(%c3_A0c, AcquireGreaterEqual, 1)
          %ji0 = arith.index_cast %j : index to i32
          func.call @generic_bcast_gemv_bf16_g(%ji0, %c3_A0, %c3_B2, %c3_uni_partial, %c8, %c3_gate) : (i32, memref<4608xi8>, memref<2320xbf16>, memref<128xi8>, i32, memref<1024xbf16>) -> ()
          aie.use_lock(%c3_A0p, Release, 1)
          aie.use_lock(%c3_A1c, AcquireGreaterEqual, 1)
          %j1 = arith.addi %j, %c1 : index
          %ji1 = arith.index_cast %j1 : index to i32
          func.call @generic_bcast_gemv_bf16_g(%ji1, %c3_A1, %c3_B2, %c3_uni_partial, %c8, %c3_gate) : (i32, memref<4608xi8>, memref<2320xbf16>, memref<128xi8>, i32, memref<1024xbf16>) -> ()
          aie.use_lock(%c3_A1p, Release, 1)
        }
        // 3b: up GEMV (nchunk=8, output -> up_buf)
        scf.for %j = %c0 to %cF step %c2 {
          aie.use_lock(%c3_A0c, AcquireGreaterEqual, 1)
          %ji0 = arith.index_cast %j : index to i32
          func.call @generic_bcast_gemv_bf16_g(%ji0, %c3_A0, %c3_B2, %c3_uni_partial, %c8, %c3_up) : (i32, memref<4608xi8>, memref<2320xbf16>, memref<128xi8>, i32, memref<1024xbf16>) -> ()
          aie.use_lock(%c3_A0p, Release, 1)
          aie.use_lock(%c3_A1c, AcquireGreaterEqual, 1)
          %j1 = arith.addi %j, %c1 : index
          %ji1 = arith.index_cast %j1 : index to i32
          func.call @generic_bcast_gemv_bf16_g(%ji1, %c3_A1, %c3_B2, %c3_uni_partial, %c8, %c3_up) : (i32, memref<4608xi8>, memref<2320xbf16>, memref<128xi8>, i32, memref<1024xbf16>) -> ()
          aie.use_lock(%c3_A1p, Release, 1)
        }
        aie.use_lock(%c3_B2p, Release, 1)
        // 3c: SiLU(gate) * up -> silu_buf (explicit pointers)
        func.call @layer_fused_silu_mul_explicit_bf16(%c3_gate, %c3_up, %c3_silu, %hc) : (memref<1024xbf16>, memref<1024xbf16>, memref<1024xbf16>, i32) -> ()
        // 3d: down GEMV (nchunk=4, activation = silu_buf)
        aie.use_lock(%c3_Pp, AcquireGreaterEqual, 1)
        scf.for %j = %c0 to %cF step %c2 {
          aie.use_lock(%c3_A0c, AcquireGreaterEqual, 1)
          %ji0 = arith.index_cast %j : index to i32
          func.call @generic_bcast_gemv_bf16_d(%ji0, %c3_A0, %c3_silu, %c3_uni_partial, %c4, %c3_P) : (i32, memref<4608xi8>, memref<1024xbf16>, memref<128xi8>, i32, memref<2320xbf16>) -> ()
          aie.use_lock(%c3_A0p, Release, 1)
          aie.use_lock(%c3_A1c, AcquireGreaterEqual, 1)
          %j1 = arith.addi %j, %c1 : index
          %ji1 = arith.index_cast %j1 : index to i32
          func.call @generic_bcast_gemv_bf16_d(%ji1, %c3_A1, %c3_silu, %c3_uni_partial, %c4, %c3_P) : (i32, memref<4608xi8>, memref<1024xbf16>, memref<128xi8>, i32, memref<2320xbf16>) -> ()
          aie.use_lock(%c3_A1p, Release, 1)
        }
        aie.use_lock(%c3_Pc, Release, 1)
      }
      aie.end
    }
    %mem_t3 = aie.mem(%t3) {
      %s0 = aie.dma_start(S2MM, 0, ^a03, ^bs3)
    ^a03:
      aie.use_lock(%c3_A0p, AcquireGreaterEqual, 1)
      aie.dma_bd(%c3_A0 : memref<4608xi8>, 0, 4608)
      aie.use_lock(%c3_A0c, Release, 1)
      aie.next_bd ^a13
    ^a13:
      aie.use_lock(%c3_A1p, AcquireGreaterEqual, 1)
      aie.dma_bd(%c3_A1 : memref<4608xi8>, 0, 4608)
      aie.use_lock(%c3_A1c, Release, 1)
      aie.next_bd ^a03
    ^bs3:
      %s1 = aie.dma_start(S2MM, 1, ^b03, ^m03)
    ^b03:
      aie.use_lock(%c3_B0p, AcquireGreaterEqual, 1)
      aie.dma_bd(%c3_B0 : memref<2320xbf16>, 0, 2320)
      aie.use_lock(%c3_B0c, Release, 1)
      aie.next_bd ^b13
    ^b13:
      aie.use_lock(%c3_B1p, AcquireGreaterEqual, 1)
      aie.dma_bd(%c3_B1 : memref<2320xbf16>, 0, 2320)
      aie.use_lock(%c3_B1c, Release, 1)
      aie.next_bd ^b23
    ^b23:
      aie.use_lock(%c3_B2p, AcquireGreaterEqual, 1)
      aie.dma_bd(%c3_B2 : memref<2320xbf16>, 0, 2320)
      aie.use_lock(%c3_B2c, Release, 1)
      aie.next_bd ^b03
    ^m03:
      %m0 = aie.dma_start(MM2S, 0, ^pf3, ^qo3)
    ^pf3:
      aie.use_lock(%c3_Pc, AcquireGreaterEqual, 1)
      aie.dma_bd(%c3_P : memref<2320xbf16>, 0, 2048)
      aie.use_lock(%c3_Pp, Release, 1)
      aie.next_bd ^pf3
    ^qo3:
      %m1 = aie.dma_start(MM2S, 1, ^qi3, ^e3)
    ^qi3:
      aie.use_lock(%c3_Qc, AcquireGreaterEqual, 1)
      aie.dma_bd(%c3_Q : memref<322xbf16>, 0, 322)
      aie.use_lock(%c3_Qp, Release, 1)
      aie.next_bd ^oh3
    ^oh3:
      aie.use_lock(%c3_Oc, AcquireGreaterEqual, 1)
      aie.dma_bd(%c3_O : memref<2048xbf16>, 0, 256)
      aie.use_lock(%c3_Op, Release, 1)
      aie.next_bd ^qi3
    ^e3:
      aie.end
    }
    %c4_A0 = aie.buffer(%t4) {sym_name = "c4_A0"} : memref<4608xi8>
    %c4_A1 = aie.buffer(%t4) {sym_name = "c4_A1"} : memref<4608xi8>
    %c4_B0 = aie.buffer(%t4) {sym_name = "c4_B0"} : memref<2320xbf16>
    %c4_B1 = aie.buffer(%t4) {sym_name = "c4_B1"} : memref<2320xbf16>
    %c4_B2 = aie.buffer(%t4) {sym_name = "c4_B2"} : memref<2320xbf16>
    %c4_Q = aie.buffer(%t4) {sym_name = "c4_Q"} : memref<322xbf16>
    %c4_O = aie.buffer(%t4) {sym_name = "c4_O"} : memref<2048xbf16>
    %c4_P = aie.buffer(%t4) {sym_name = "c4_P"} : memref<2320xbf16>
    %c4_A0p = aie.lock(%t4, 0) {init = 1 : i32, sym_name = "c4_A0p"}
    %c4_A0c = aie.lock(%t4, 1) {init = 0 : i32, sym_name = "c4_A0c"}
    %c4_B0p = aie.lock(%t4, 2) {init = 1 : i32, sym_name = "c4_B0p"}
    %c4_B0c = aie.lock(%t4, 3) {init = 0 : i32, sym_name = "c4_B0c"}
    %c4_Qp = aie.lock(%t4, 4) {init = 1 : i32, sym_name = "c4_Qp"}
    %c4_Qc = aie.lock(%t4, 5) {init = 0 : i32, sym_name = "c4_Qc"}
    %c4_Op = aie.lock(%t4, 6) {init = 1 : i32, sym_name = "c4_Op"}
    %c4_Oc = aie.lock(%t4, 7) {init = 0 : i32, sym_name = "c4_Oc"}
    %c4_Pp = aie.lock(%t4, 8) {init = 1 : i32, sym_name = "c4_Pp"}
    %c4_Pc = aie.lock(%t4, 9) {init = 0 : i32, sym_name = "c4_Pc"}
    %c4_A1p = aie.lock(%t4, 10) {init = 1 : i32, sym_name = "c4_A1p"}
    %c4_A1c = aie.lock(%t4, 11) {init = 0 : i32, sym_name = "c4_A1c"}
    %c4_B1p = aie.lock(%t4, 12) {init = 1 : i32, sym_name = "c4_B1p"}
    %c4_B1c = aie.lock(%t4, 13) {init = 0 : i32, sym_name = "c4_B1c"}
    %c4_B2p = aie.lock(%t4, 14) {init = 1 : i32, sym_name = "c4_B2p"}
    %c4_B2c = aie.lock(%t4, 15) {init = 0 : i32, sym_name = "c4_B2c"}
    %c4_gate = aie.buffer(%t4) {sym_name = "c4_gate"} : memref<1024xbf16>
    %c4_up   = aie.buffer(%t4) {sym_name = "c4_up"}   : memref<1024xbf16>
    %c4_silu = aie.buffer(%t4) {sym_name = "c4_silu"} : memref<1024xbf16>
    %c4_uni_partial = aie.buffer(%t4) {sym_name = "c4_uni_partial"} : memref<128xi8>
    %core4 = aie.core(%t4) {
      %c0 = arith.constant 0 : index
      %cN = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      %c64 = arith.constant 64 : index
      %cF = arith.constant 256 : index
      %c4 = arith.constant 4 : i32
      %c8 = arith.constant 8 : i32
      %qr = arith.constant 256 : i32
      %hc = arith.constant 1024 : i32
      scf.for %tok = %c0 to %cN step %c1 {
        func.call @_ha_noop() : () -> ()
        // ---- phase 1: Q-GEMV + rope (B0 = x_bundle, nchunk=8) ----
        aie.use_lock(%c4_B0c, AcquireGreaterEqual, 1)
        aie.use_lock(%c4_Qp, AcquireGreaterEqual, 1)
        scf.for %j = %c0 to %c64 step %c2 {
          aie.use_lock(%c4_A0c, AcquireGreaterEqual, 1)
          %ji0 = arith.index_cast %j : index to i32
          func.call @generic_bcast_gemv_bf16_q(%ji0, %c4_A0, %c4_B0, %c4_uni_partial, %c8, %c4_Q) : (i32, memref<4608xi8>, memref<2320xbf16>, memref<128xi8>, i32, memref<322xbf16>) -> ()
          aie.use_lock(%c4_A0p, Release, 1)
          aie.use_lock(%c4_A1c, AcquireGreaterEqual, 1)
          %j1 = arith.addi %j, %c1 : index
          %ji1 = arith.index_cast %j1 : index to i32
          func.call @generic_bcast_gemv_bf16_q(%ji1, %c4_A1, %c4_B0, %c4_uni_partial, %c8, %c4_Q) : (i32, memref<4608xi8>, memref<2320xbf16>, memref<128xi8>, i32, memref<322xbf16>) -> ()
          aie.use_lock(%c4_A1p, Release, 1)
        }
        func.call @rope_bundled(%c4_Q, %c4_B0, %c4_Q, %qr) : (memref<322xbf16>, memref<2320xbf16>, memref<322xbf16>, i32) -> ()
        aie.use_lock(%c4_Qc, Release, 1)
        aie.use_lock(%c4_B0p, Release, 1)
        // ---- phase 2: O-proj (B1 = attn_out, nchunk=8) ----
        aie.use_lock(%c4_B1c, AcquireGreaterEqual, 1)
        aie.use_lock(%c4_Op, AcquireGreaterEqual, 1)
        scf.for %j = %c0 to %c64 step %c2 {
          aie.use_lock(%c4_A0c, AcquireGreaterEqual, 1)
          %ji0 = arith.index_cast %j : index to i32
          func.call @generic_bcast_gemv_bf16_o(%ji0, %c4_A0, %c4_B1, %c4_uni_partial, %c8, %c4_O) : (i32, memref<4608xi8>, memref<2320xbf16>, memref<128xi8>, i32, memref<2048xbf16>) -> ()
          aie.use_lock(%c4_A0p, Release, 1)
          aie.use_lock(%c4_A1c, AcquireGreaterEqual, 1)
          %j1 = arith.addi %j, %c1 : index
          %ji1 = arith.index_cast %j1 : index to i32
          func.call @generic_bcast_gemv_bf16_o(%ji1, %c4_A1, %c4_B1, %c4_uni_partial, %c8, %c4_O) : (i32, memref<4608xi8>, memref<2320xbf16>, memref<128xi8>, i32, memref<2048xbf16>) -> ()
          aie.use_lock(%c4_A1p, Release, 1)
        }
        aie.use_lock(%c4_B1p, Release, 1)
        aie.use_lock(%c4_Oc, Release, 1)
        // ---- phase 3: FFN (B2 = ffn_in) ----
        // 3a: gate GEMV (nchunk=8, output -> gate_buf)
        aie.use_lock(%c4_B2c, AcquireGreaterEqual, 1)
        scf.for %j = %c0 to %cF step %c2 {
          aie.use_lock(%c4_A0c, AcquireGreaterEqual, 1)
          %ji0 = arith.index_cast %j : index to i32
          func.call @generic_bcast_gemv_bf16_g(%ji0, %c4_A0, %c4_B2, %c4_uni_partial, %c8, %c4_gate) : (i32, memref<4608xi8>, memref<2320xbf16>, memref<128xi8>, i32, memref<1024xbf16>) -> ()
          aie.use_lock(%c4_A0p, Release, 1)
          aie.use_lock(%c4_A1c, AcquireGreaterEqual, 1)
          %j1 = arith.addi %j, %c1 : index
          %ji1 = arith.index_cast %j1 : index to i32
          func.call @generic_bcast_gemv_bf16_g(%ji1, %c4_A1, %c4_B2, %c4_uni_partial, %c8, %c4_gate) : (i32, memref<4608xi8>, memref<2320xbf16>, memref<128xi8>, i32, memref<1024xbf16>) -> ()
          aie.use_lock(%c4_A1p, Release, 1)
        }
        // 3b: up GEMV (nchunk=8, output -> up_buf)
        scf.for %j = %c0 to %cF step %c2 {
          aie.use_lock(%c4_A0c, AcquireGreaterEqual, 1)
          %ji0 = arith.index_cast %j : index to i32
          func.call @generic_bcast_gemv_bf16_g(%ji0, %c4_A0, %c4_B2, %c4_uni_partial, %c8, %c4_up) : (i32, memref<4608xi8>, memref<2320xbf16>, memref<128xi8>, i32, memref<1024xbf16>) -> ()
          aie.use_lock(%c4_A0p, Release, 1)
          aie.use_lock(%c4_A1c, AcquireGreaterEqual, 1)
          %j1 = arith.addi %j, %c1 : index
          %ji1 = arith.index_cast %j1 : index to i32
          func.call @generic_bcast_gemv_bf16_g(%ji1, %c4_A1, %c4_B2, %c4_uni_partial, %c8, %c4_up) : (i32, memref<4608xi8>, memref<2320xbf16>, memref<128xi8>, i32, memref<1024xbf16>) -> ()
          aie.use_lock(%c4_A1p, Release, 1)
        }
        aie.use_lock(%c4_B2p, Release, 1)
        // 3c: SiLU(gate) * up -> silu_buf (explicit pointers)
        func.call @layer_fused_silu_mul_explicit_bf16(%c4_gate, %c4_up, %c4_silu, %hc) : (memref<1024xbf16>, memref<1024xbf16>, memref<1024xbf16>, i32) -> ()
        // 3d: down GEMV (nchunk=4, activation = silu_buf)
        aie.use_lock(%c4_Pp, AcquireGreaterEqual, 1)
        scf.for %j = %c0 to %cF step %c2 {
          aie.use_lock(%c4_A0c, AcquireGreaterEqual, 1)
          %ji0 = arith.index_cast %j : index to i32
          func.call @generic_bcast_gemv_bf16_d(%ji0, %c4_A0, %c4_silu, %c4_uni_partial, %c4, %c4_P) : (i32, memref<4608xi8>, memref<1024xbf16>, memref<128xi8>, i32, memref<2320xbf16>) -> ()
          aie.use_lock(%c4_A0p, Release, 1)
          aie.use_lock(%c4_A1c, AcquireGreaterEqual, 1)
          %j1 = arith.addi %j, %c1 : index
          %ji1 = arith.index_cast %j1 : index to i32
          func.call @generic_bcast_gemv_bf16_d(%ji1, %c4_A1, %c4_silu, %c4_uni_partial, %c4, %c4_P) : (i32, memref<4608xi8>, memref<1024xbf16>, memref<128xi8>, i32, memref<2320xbf16>) -> ()
          aie.use_lock(%c4_A1p, Release, 1)
        }
        aie.use_lock(%c4_Pc, Release, 1)
      }
      aie.end
    }
    %mem_t4 = aie.mem(%t4) {
      %s0 = aie.dma_start(S2MM, 0, ^a04, ^bs4)
    ^a04:
      aie.use_lock(%c4_A0p, AcquireGreaterEqual, 1)
      aie.dma_bd(%c4_A0 : memref<4608xi8>, 0, 4608)
      aie.use_lock(%c4_A0c, Release, 1)
      aie.next_bd ^a14
    ^a14:
      aie.use_lock(%c4_A1p, AcquireGreaterEqual, 1)
      aie.dma_bd(%c4_A1 : memref<4608xi8>, 0, 4608)
      aie.use_lock(%c4_A1c, Release, 1)
      aie.next_bd ^a04
    ^bs4:
      %s1 = aie.dma_start(S2MM, 1, ^b04, ^m04)
    ^b04:
      aie.use_lock(%c4_B0p, AcquireGreaterEqual, 1)
      aie.dma_bd(%c4_B0 : memref<2320xbf16>, 0, 2320)
      aie.use_lock(%c4_B0c, Release, 1)
      aie.next_bd ^b14
    ^b14:
      aie.use_lock(%c4_B1p, AcquireGreaterEqual, 1)
      aie.dma_bd(%c4_B1 : memref<2320xbf16>, 0, 2320)
      aie.use_lock(%c4_B1c, Release, 1)
      aie.next_bd ^b24
    ^b24:
      aie.use_lock(%c4_B2p, AcquireGreaterEqual, 1)
      aie.dma_bd(%c4_B2 : memref<2320xbf16>, 0, 2320)
      aie.use_lock(%c4_B2c, Release, 1)
      aie.next_bd ^b04
    ^m04:
      %m0 = aie.dma_start(MM2S, 0, ^pf4, ^qo4)
    ^pf4:
      aie.use_lock(%c4_Pc, AcquireGreaterEqual, 1)
      aie.dma_bd(%c4_P : memref<2320xbf16>, 0, 2048)
      aie.use_lock(%c4_Pp, Release, 1)
      aie.next_bd ^pf4
    ^qo4:
      %m1 = aie.dma_start(MM2S, 1, ^qi4, ^e4)
    ^qi4:
      aie.use_lock(%c4_Qc, AcquireGreaterEqual, 1)
      aie.dma_bd(%c4_Q : memref<322xbf16>, 0, 322)
      aie.use_lock(%c4_Qp, Release, 1)
      aie.next_bd ^oh4
    ^oh4:
      aie.use_lock(%c4_Oc, AcquireGreaterEqual, 1)
      aie.dma_bd(%c4_O : memref<2048xbf16>, 0, 256)
      aie.use_lock(%c4_Op, Release, 1)
      aie.next_bd ^qi4
    ^e4:
      aie.end
    }
    %c5_A0 = aie.buffer(%t5) {sym_name = "c5_A0"} : memref<4608xi8>
    %c5_A1 = aie.buffer(%t5) {sym_name = "c5_A1"} : memref<4608xi8>
    %c5_B0 = aie.buffer(%t5) {sym_name = "c5_B0"} : memref<2320xbf16>
    %c5_B1 = aie.buffer(%t5) {sym_name = "c5_B1"} : memref<2320xbf16>
    %c5_B2 = aie.buffer(%t5) {sym_name = "c5_B2"} : memref<2320xbf16>
    %c5_Q = aie.buffer(%t5) {sym_name = "c5_Q"} : memref<322xbf16>
    %c5_O = aie.buffer(%t5) {sym_name = "c5_O"} : memref<2048xbf16>
    %c5_P = aie.buffer(%t5) {sym_name = "c5_P"} : memref<2320xbf16>
    %c5_A0p = aie.lock(%t5, 0) {init = 1 : i32, sym_name = "c5_A0p"}
    %c5_A0c = aie.lock(%t5, 1) {init = 0 : i32, sym_name = "c5_A0c"}
    %c5_B0p = aie.lock(%t5, 2) {init = 1 : i32, sym_name = "c5_B0p"}
    %c5_B0c = aie.lock(%t5, 3) {init = 0 : i32, sym_name = "c5_B0c"}
    %c5_Qp = aie.lock(%t5, 4) {init = 1 : i32, sym_name = "c5_Qp"}
    %c5_Qc = aie.lock(%t5, 5) {init = 0 : i32, sym_name = "c5_Qc"}
    %c5_Op = aie.lock(%t5, 6) {init = 1 : i32, sym_name = "c5_Op"}
    %c5_Oc = aie.lock(%t5, 7) {init = 0 : i32, sym_name = "c5_Oc"}
    %c5_Pp = aie.lock(%t5, 8) {init = 1 : i32, sym_name = "c5_Pp"}
    %c5_Pc = aie.lock(%t5, 9) {init = 0 : i32, sym_name = "c5_Pc"}
    %c5_A1p = aie.lock(%t5, 10) {init = 1 : i32, sym_name = "c5_A1p"}
    %c5_A1c = aie.lock(%t5, 11) {init = 0 : i32, sym_name = "c5_A1c"}
    %c5_B1p = aie.lock(%t5, 12) {init = 1 : i32, sym_name = "c5_B1p"}
    %c5_B1c = aie.lock(%t5, 13) {init = 0 : i32, sym_name = "c5_B1c"}
    %c5_B2p = aie.lock(%t5, 14) {init = 1 : i32, sym_name = "c5_B2p"}
    %c5_B2c = aie.lock(%t5, 15) {init = 0 : i32, sym_name = "c5_B2c"}
    %c5_gate = aie.buffer(%t5) {sym_name = "c5_gate"} : memref<1024xbf16>
    %c5_up   = aie.buffer(%t5) {sym_name = "c5_up"}   : memref<1024xbf16>
    %c5_silu = aie.buffer(%t5) {sym_name = "c5_silu"} : memref<1024xbf16>
    %c5_uni_partial = aie.buffer(%t5) {sym_name = "c5_uni_partial"} : memref<128xi8>
    %core5 = aie.core(%t5) {
      %c0 = arith.constant 0 : index
      %cN = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      %c64 = arith.constant 64 : index
      %cF = arith.constant 256 : index
      %c4 = arith.constant 4 : i32
      %c8 = arith.constant 8 : i32
      %qr = arith.constant 256 : i32
      %hc = arith.constant 1024 : i32
      scf.for %tok = %c0 to %cN step %c1 {
        func.call @_ha_noop() : () -> ()
        // ---- phase 1: Q-GEMV + rope (B0 = x_bundle, nchunk=8) ----
        aie.use_lock(%c5_B0c, AcquireGreaterEqual, 1)
        aie.use_lock(%c5_Qp, AcquireGreaterEqual, 1)
        scf.for %j = %c0 to %c64 step %c2 {
          aie.use_lock(%c5_A0c, AcquireGreaterEqual, 1)
          %ji0 = arith.index_cast %j : index to i32
          func.call @generic_bcast_gemv_bf16_q(%ji0, %c5_A0, %c5_B0, %c5_uni_partial, %c8, %c5_Q) : (i32, memref<4608xi8>, memref<2320xbf16>, memref<128xi8>, i32, memref<322xbf16>) -> ()
          aie.use_lock(%c5_A0p, Release, 1)
          aie.use_lock(%c5_A1c, AcquireGreaterEqual, 1)
          %j1 = arith.addi %j, %c1 : index
          %ji1 = arith.index_cast %j1 : index to i32
          func.call @generic_bcast_gemv_bf16_q(%ji1, %c5_A1, %c5_B0, %c5_uni_partial, %c8, %c5_Q) : (i32, memref<4608xi8>, memref<2320xbf16>, memref<128xi8>, i32, memref<322xbf16>) -> ()
          aie.use_lock(%c5_A1p, Release, 1)
        }
        func.call @rope_bundled(%c5_Q, %c5_B0, %c5_Q, %qr) : (memref<322xbf16>, memref<2320xbf16>, memref<322xbf16>, i32) -> ()
        aie.use_lock(%c5_Qc, Release, 1)
        aie.use_lock(%c5_B0p, Release, 1)
        // ---- phase 2: O-proj (B1 = attn_out, nchunk=8) ----
        aie.use_lock(%c5_B1c, AcquireGreaterEqual, 1)
        aie.use_lock(%c5_Op, AcquireGreaterEqual, 1)
        scf.for %j = %c0 to %c64 step %c2 {
          aie.use_lock(%c5_A0c, AcquireGreaterEqual, 1)
          %ji0 = arith.index_cast %j : index to i32
          func.call @generic_bcast_gemv_bf16_o(%ji0, %c5_A0, %c5_B1, %c5_uni_partial, %c8, %c5_O) : (i32, memref<4608xi8>, memref<2320xbf16>, memref<128xi8>, i32, memref<2048xbf16>) -> ()
          aie.use_lock(%c5_A0p, Release, 1)
          aie.use_lock(%c5_A1c, AcquireGreaterEqual, 1)
          %j1 = arith.addi %j, %c1 : index
          %ji1 = arith.index_cast %j1 : index to i32
          func.call @generic_bcast_gemv_bf16_o(%ji1, %c5_A1, %c5_B1, %c5_uni_partial, %c8, %c5_O) : (i32, memref<4608xi8>, memref<2320xbf16>, memref<128xi8>, i32, memref<2048xbf16>) -> ()
          aie.use_lock(%c5_A1p, Release, 1)
        }
        aie.use_lock(%c5_B1p, Release, 1)
        aie.use_lock(%c5_Oc, Release, 1)
        // ---- phase 3: FFN (B2 = ffn_in) ----
        // 3a: gate GEMV (nchunk=8, output -> gate_buf)
        aie.use_lock(%c5_B2c, AcquireGreaterEqual, 1)
        scf.for %j = %c0 to %cF step %c2 {
          aie.use_lock(%c5_A0c, AcquireGreaterEqual, 1)
          %ji0 = arith.index_cast %j : index to i32
          func.call @generic_bcast_gemv_bf16_g(%ji0, %c5_A0, %c5_B2, %c5_uni_partial, %c8, %c5_gate) : (i32, memref<4608xi8>, memref<2320xbf16>, memref<128xi8>, i32, memref<1024xbf16>) -> ()
          aie.use_lock(%c5_A0p, Release, 1)
          aie.use_lock(%c5_A1c, AcquireGreaterEqual, 1)
          %j1 = arith.addi %j, %c1 : index
          %ji1 = arith.index_cast %j1 : index to i32
          func.call @generic_bcast_gemv_bf16_g(%ji1, %c5_A1, %c5_B2, %c5_uni_partial, %c8, %c5_gate) : (i32, memref<4608xi8>, memref<2320xbf16>, memref<128xi8>, i32, memref<1024xbf16>) -> ()
          aie.use_lock(%c5_A1p, Release, 1)
        }
        // 3b: up GEMV (nchunk=8, output -> up_buf)
        scf.for %j = %c0 to %cF step %c2 {
          aie.use_lock(%c5_A0c, AcquireGreaterEqual, 1)
          %ji0 = arith.index_cast %j : index to i32
          func.call @generic_bcast_gemv_bf16_g(%ji0, %c5_A0, %c5_B2, %c5_uni_partial, %c8, %c5_up) : (i32, memref<4608xi8>, memref<2320xbf16>, memref<128xi8>, i32, memref<1024xbf16>) -> ()
          aie.use_lock(%c5_A0p, Release, 1)
          aie.use_lock(%c5_A1c, AcquireGreaterEqual, 1)
          %j1 = arith.addi %j, %c1 : index
          %ji1 = arith.index_cast %j1 : index to i32
          func.call @generic_bcast_gemv_bf16_g(%ji1, %c5_A1, %c5_B2, %c5_uni_partial, %c8, %c5_up) : (i32, memref<4608xi8>, memref<2320xbf16>, memref<128xi8>, i32, memref<1024xbf16>) -> ()
          aie.use_lock(%c5_A1p, Release, 1)
        }
        aie.use_lock(%c5_B2p, Release, 1)
        // 3c: SiLU(gate) * up -> silu_buf (explicit pointers)
        func.call @layer_fused_silu_mul_explicit_bf16(%c5_gate, %c5_up, %c5_silu, %hc) : (memref<1024xbf16>, memref<1024xbf16>, memref<1024xbf16>, i32) -> ()
        // 3d: down GEMV (nchunk=4, activation = silu_buf)
        aie.use_lock(%c5_Pp, AcquireGreaterEqual, 1)
        scf.for %j = %c0 to %cF step %c2 {
          aie.use_lock(%c5_A0c, AcquireGreaterEqual, 1)
          %ji0 = arith.index_cast %j : index to i32
          func.call @generic_bcast_gemv_bf16_d(%ji0, %c5_A0, %c5_silu, %c5_uni_partial, %c4, %c5_P) : (i32, memref<4608xi8>, memref<1024xbf16>, memref<128xi8>, i32, memref<2320xbf16>) -> ()
          aie.use_lock(%c5_A0p, Release, 1)
          aie.use_lock(%c5_A1c, AcquireGreaterEqual, 1)
          %j1 = arith.addi %j, %c1 : index
          %ji1 = arith.index_cast %j1 : index to i32
          func.call @generic_bcast_gemv_bf16_d(%ji1, %c5_A1, %c5_silu, %c5_uni_partial, %c4, %c5_P) : (i32, memref<4608xi8>, memref<1024xbf16>, memref<128xi8>, i32, memref<2320xbf16>) -> ()
          aie.use_lock(%c5_A1p, Release, 1)
        }
        aie.use_lock(%c5_Pc, Release, 1)
      }
      aie.end
    }
    %mem_t5 = aie.mem(%t5) {
      %s0 = aie.dma_start(S2MM, 0, ^a05, ^bs5)
    ^a05:
      aie.use_lock(%c5_A0p, AcquireGreaterEqual, 1)
      aie.dma_bd(%c5_A0 : memref<4608xi8>, 0, 4608)
      aie.use_lock(%c5_A0c, Release, 1)
      aie.next_bd ^a15
    ^a15:
      aie.use_lock(%c5_A1p, AcquireGreaterEqual, 1)
      aie.dma_bd(%c5_A1 : memref<4608xi8>, 0, 4608)
      aie.use_lock(%c5_A1c, Release, 1)
      aie.next_bd ^a05
    ^bs5:
      %s1 = aie.dma_start(S2MM, 1, ^b05, ^m05)
    ^b05:
      aie.use_lock(%c5_B0p, AcquireGreaterEqual, 1)
      aie.dma_bd(%c5_B0 : memref<2320xbf16>, 0, 2320)
      aie.use_lock(%c5_B0c, Release, 1)
      aie.next_bd ^b15
    ^b15:
      aie.use_lock(%c5_B1p, AcquireGreaterEqual, 1)
      aie.dma_bd(%c5_B1 : memref<2320xbf16>, 0, 2320)
      aie.use_lock(%c5_B1c, Release, 1)
      aie.next_bd ^b25
    ^b25:
      aie.use_lock(%c5_B2p, AcquireGreaterEqual, 1)
      aie.dma_bd(%c5_B2 : memref<2320xbf16>, 0, 2320)
      aie.use_lock(%c5_B2c, Release, 1)
      aie.next_bd ^b05
    ^m05:
      %m0 = aie.dma_start(MM2S, 0, ^pf5, ^qo5)
    ^pf5:
      aie.use_lock(%c5_Pc, AcquireGreaterEqual, 1)
      aie.dma_bd(%c5_P : memref<2320xbf16>, 0, 2048)
      aie.use_lock(%c5_Pp, Release, 1)
      aie.next_bd ^pf5
    ^qo5:
      %m1 = aie.dma_start(MM2S, 1, ^qi5, ^e5)
    ^qi5:
      aie.use_lock(%c5_Qc, AcquireGreaterEqual, 1)
      aie.dma_bd(%c5_Q : memref<322xbf16>, 0, 322)
      aie.use_lock(%c5_Qp, Release, 1)
      aie.next_bd ^oh5
    ^oh5:
      aie.use_lock(%c5_Oc, AcquireGreaterEqual, 1)
      aie.dma_bd(%c5_O : memref<2048xbf16>, 0, 256)
      aie.use_lock(%c5_Op, Release, 1)
      aie.next_bd ^qi5
    ^e5:
      aie.end
    }
    %c6_A0 = aie.buffer(%t6) {sym_name = "c6_A0"} : memref<4608xi8>
    %c6_A1 = aie.buffer(%t6) {sym_name = "c6_A1"} : memref<4608xi8>
    %c6_B0 = aie.buffer(%t6) {sym_name = "c6_B0"} : memref<2320xbf16>
    %c6_B1 = aie.buffer(%t6) {sym_name = "c6_B1"} : memref<2320xbf16>
    %c6_B2 = aie.buffer(%t6) {sym_name = "c6_B2"} : memref<2320xbf16>
    %c6_Q = aie.buffer(%t6) {sym_name = "c6_Q"} : memref<322xbf16>
    %c6_O = aie.buffer(%t6) {sym_name = "c6_O"} : memref<2048xbf16>
    %c6_P = aie.buffer(%t6) {sym_name = "c6_P"} : memref<2320xbf16>
    %c6_A0p = aie.lock(%t6, 0) {init = 1 : i32, sym_name = "c6_A0p"}
    %c6_A0c = aie.lock(%t6, 1) {init = 0 : i32, sym_name = "c6_A0c"}
    %c6_B0p = aie.lock(%t6, 2) {init = 1 : i32, sym_name = "c6_B0p"}
    %c6_B0c = aie.lock(%t6, 3) {init = 0 : i32, sym_name = "c6_B0c"}
    %c6_Qp = aie.lock(%t6, 4) {init = 1 : i32, sym_name = "c6_Qp"}
    %c6_Qc = aie.lock(%t6, 5) {init = 0 : i32, sym_name = "c6_Qc"}
    %c6_Op = aie.lock(%t6, 6) {init = 1 : i32, sym_name = "c6_Op"}
    %c6_Oc = aie.lock(%t6, 7) {init = 0 : i32, sym_name = "c6_Oc"}
    %c6_Pp = aie.lock(%t6, 8) {init = 1 : i32, sym_name = "c6_Pp"}
    %c6_Pc = aie.lock(%t6, 9) {init = 0 : i32, sym_name = "c6_Pc"}
    %c6_A1p = aie.lock(%t6, 10) {init = 1 : i32, sym_name = "c6_A1p"}
    %c6_A1c = aie.lock(%t6, 11) {init = 0 : i32, sym_name = "c6_A1c"}
    %c6_B1p = aie.lock(%t6, 12) {init = 1 : i32, sym_name = "c6_B1p"}
    %c6_B1c = aie.lock(%t6, 13) {init = 0 : i32, sym_name = "c6_B1c"}
    %c6_B2p = aie.lock(%t6, 14) {init = 1 : i32, sym_name = "c6_B2p"}
    %c6_B2c = aie.lock(%t6, 15) {init = 0 : i32, sym_name = "c6_B2c"}
    %c6_gate = aie.buffer(%t6) {sym_name = "c6_gate"} : memref<1024xbf16>
    %c6_up   = aie.buffer(%t6) {sym_name = "c6_up"}   : memref<1024xbf16>
    %c6_silu = aie.buffer(%t6) {sym_name = "c6_silu"} : memref<1024xbf16>
    %c6_uni_partial = aie.buffer(%t6) {sym_name = "c6_uni_partial"} : memref<128xi8>
    %core6 = aie.core(%t6) {
      %c0 = arith.constant 0 : index
      %cN = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      %c64 = arith.constant 64 : index
      %cF = arith.constant 256 : index
      %c4 = arith.constant 4 : i32
      %c8 = arith.constant 8 : i32
      %qr = arith.constant 256 : i32
      %hc = arith.constant 1024 : i32
      scf.for %tok = %c0 to %cN step %c1 {
        func.call @_ha_noop() : () -> ()
        // ---- phase 1: Q-GEMV + rope (B0 = x_bundle, nchunk=8) ----
        aie.use_lock(%c6_B0c, AcquireGreaterEqual, 1)
        aie.use_lock(%c6_Qp, AcquireGreaterEqual, 1)
        scf.for %j = %c0 to %c64 step %c2 {
          aie.use_lock(%c6_A0c, AcquireGreaterEqual, 1)
          %ji0 = arith.index_cast %j : index to i32
          func.call @generic_bcast_gemv_bf16_q(%ji0, %c6_A0, %c6_B0, %c6_uni_partial, %c8, %c6_Q) : (i32, memref<4608xi8>, memref<2320xbf16>, memref<128xi8>, i32, memref<322xbf16>) -> ()
          aie.use_lock(%c6_A0p, Release, 1)
          aie.use_lock(%c6_A1c, AcquireGreaterEqual, 1)
          %j1 = arith.addi %j, %c1 : index
          %ji1 = arith.index_cast %j1 : index to i32
          func.call @generic_bcast_gemv_bf16_q(%ji1, %c6_A1, %c6_B0, %c6_uni_partial, %c8, %c6_Q) : (i32, memref<4608xi8>, memref<2320xbf16>, memref<128xi8>, i32, memref<322xbf16>) -> ()
          aie.use_lock(%c6_A1p, Release, 1)
        }
        func.call @rope_bundled(%c6_Q, %c6_B0, %c6_Q, %qr) : (memref<322xbf16>, memref<2320xbf16>, memref<322xbf16>, i32) -> ()
        aie.use_lock(%c6_Qc, Release, 1)
        aie.use_lock(%c6_B0p, Release, 1)
        // ---- phase 2: O-proj (B1 = attn_out, nchunk=8) ----
        aie.use_lock(%c6_B1c, AcquireGreaterEqual, 1)
        aie.use_lock(%c6_Op, AcquireGreaterEqual, 1)
        scf.for %j = %c0 to %c64 step %c2 {
          aie.use_lock(%c6_A0c, AcquireGreaterEqual, 1)
          %ji0 = arith.index_cast %j : index to i32
          func.call @generic_bcast_gemv_bf16_o(%ji0, %c6_A0, %c6_B1, %c6_uni_partial, %c8, %c6_O) : (i32, memref<4608xi8>, memref<2320xbf16>, memref<128xi8>, i32, memref<2048xbf16>) -> ()
          aie.use_lock(%c6_A0p, Release, 1)
          aie.use_lock(%c6_A1c, AcquireGreaterEqual, 1)
          %j1 = arith.addi %j, %c1 : index
          %ji1 = arith.index_cast %j1 : index to i32
          func.call @generic_bcast_gemv_bf16_o(%ji1, %c6_A1, %c6_B1, %c6_uni_partial, %c8, %c6_O) : (i32, memref<4608xi8>, memref<2320xbf16>, memref<128xi8>, i32, memref<2048xbf16>) -> ()
          aie.use_lock(%c6_A1p, Release, 1)
        }
        aie.use_lock(%c6_B1p, Release, 1)
        aie.use_lock(%c6_Oc, Release, 1)
        // ---- phase 3: FFN (B2 = ffn_in) ----
        // 3a: gate GEMV (nchunk=8, output -> gate_buf)
        aie.use_lock(%c6_B2c, AcquireGreaterEqual, 1)
        scf.for %j = %c0 to %cF step %c2 {
          aie.use_lock(%c6_A0c, AcquireGreaterEqual, 1)
          %ji0 = arith.index_cast %j : index to i32
          func.call @generic_bcast_gemv_bf16_g(%ji0, %c6_A0, %c6_B2, %c6_uni_partial, %c8, %c6_gate) : (i32, memref<4608xi8>, memref<2320xbf16>, memref<128xi8>, i32, memref<1024xbf16>) -> ()
          aie.use_lock(%c6_A0p, Release, 1)
          aie.use_lock(%c6_A1c, AcquireGreaterEqual, 1)
          %j1 = arith.addi %j, %c1 : index
          %ji1 = arith.index_cast %j1 : index to i32
          func.call @generic_bcast_gemv_bf16_g(%ji1, %c6_A1, %c6_B2, %c6_uni_partial, %c8, %c6_gate) : (i32, memref<4608xi8>, memref<2320xbf16>, memref<128xi8>, i32, memref<1024xbf16>) -> ()
          aie.use_lock(%c6_A1p, Release, 1)
        }
        // 3b: up GEMV (nchunk=8, output -> up_buf)
        scf.for %j = %c0 to %cF step %c2 {
          aie.use_lock(%c6_A0c, AcquireGreaterEqual, 1)
          %ji0 = arith.index_cast %j : index to i32
          func.call @generic_bcast_gemv_bf16_g(%ji0, %c6_A0, %c6_B2, %c6_uni_partial, %c8, %c6_up) : (i32, memref<4608xi8>, memref<2320xbf16>, memref<128xi8>, i32, memref<1024xbf16>) -> ()
          aie.use_lock(%c6_A0p, Release, 1)
          aie.use_lock(%c6_A1c, AcquireGreaterEqual, 1)
          %j1 = arith.addi %j, %c1 : index
          %ji1 = arith.index_cast %j1 : index to i32
          func.call @generic_bcast_gemv_bf16_g(%ji1, %c6_A1, %c6_B2, %c6_uni_partial, %c8, %c6_up) : (i32, memref<4608xi8>, memref<2320xbf16>, memref<128xi8>, i32, memref<1024xbf16>) -> ()
          aie.use_lock(%c6_A1p, Release, 1)
        }
        aie.use_lock(%c6_B2p, Release, 1)
        // 3c: SiLU(gate) * up -> silu_buf (explicit pointers)
        func.call @layer_fused_silu_mul_explicit_bf16(%c6_gate, %c6_up, %c6_silu, %hc) : (memref<1024xbf16>, memref<1024xbf16>, memref<1024xbf16>, i32) -> ()
        // 3d: down GEMV (nchunk=4, activation = silu_buf)
        aie.use_lock(%c6_Pp, AcquireGreaterEqual, 1)
        scf.for %j = %c0 to %cF step %c2 {
          aie.use_lock(%c6_A0c, AcquireGreaterEqual, 1)
          %ji0 = arith.index_cast %j : index to i32
          func.call @generic_bcast_gemv_bf16_d(%ji0, %c6_A0, %c6_silu, %c6_uni_partial, %c4, %c6_P) : (i32, memref<4608xi8>, memref<1024xbf16>, memref<128xi8>, i32, memref<2320xbf16>) -> ()
          aie.use_lock(%c6_A0p, Release, 1)
          aie.use_lock(%c6_A1c, AcquireGreaterEqual, 1)
          %j1 = arith.addi %j, %c1 : index
          %ji1 = arith.index_cast %j1 : index to i32
          func.call @generic_bcast_gemv_bf16_d(%ji1, %c6_A1, %c6_silu, %c6_uni_partial, %c4, %c6_P) : (i32, memref<4608xi8>, memref<1024xbf16>, memref<128xi8>, i32, memref<2320xbf16>) -> ()
          aie.use_lock(%c6_A1p, Release, 1)
        }
        aie.use_lock(%c6_Pc, Release, 1)
      }
      aie.end
    }
    %mem_t6 = aie.mem(%t6) {
      %s0 = aie.dma_start(S2MM, 0, ^a06, ^bs6)
    ^a06:
      aie.use_lock(%c6_A0p, AcquireGreaterEqual, 1)
      aie.dma_bd(%c6_A0 : memref<4608xi8>, 0, 4608)
      aie.use_lock(%c6_A0c, Release, 1)
      aie.next_bd ^a16
    ^a16:
      aie.use_lock(%c6_A1p, AcquireGreaterEqual, 1)
      aie.dma_bd(%c6_A1 : memref<4608xi8>, 0, 4608)
      aie.use_lock(%c6_A1c, Release, 1)
      aie.next_bd ^a06
    ^bs6:
      %s1 = aie.dma_start(S2MM, 1, ^b06, ^m06)
    ^b06:
      aie.use_lock(%c6_B0p, AcquireGreaterEqual, 1)
      aie.dma_bd(%c6_B0 : memref<2320xbf16>, 0, 2320)
      aie.use_lock(%c6_B0c, Release, 1)
      aie.next_bd ^b16
    ^b16:
      aie.use_lock(%c6_B1p, AcquireGreaterEqual, 1)
      aie.dma_bd(%c6_B1 : memref<2320xbf16>, 0, 2320)
      aie.use_lock(%c6_B1c, Release, 1)
      aie.next_bd ^b26
    ^b26:
      aie.use_lock(%c6_B2p, AcquireGreaterEqual, 1)
      aie.dma_bd(%c6_B2 : memref<2320xbf16>, 0, 2320)
      aie.use_lock(%c6_B2c, Release, 1)
      aie.next_bd ^b06
    ^m06:
      %m0 = aie.dma_start(MM2S, 0, ^pf6, ^qo6)
    ^pf6:
      aie.use_lock(%c6_Pc, AcquireGreaterEqual, 1)
      aie.dma_bd(%c6_P : memref<2320xbf16>, 0, 2048)
      aie.use_lock(%c6_Pp, Release, 1)
      aie.next_bd ^pf6
    ^qo6:
      %m1 = aie.dma_start(MM2S, 1, ^qi6, ^e6)
    ^qi6:
      aie.use_lock(%c6_Qc, AcquireGreaterEqual, 1)
      aie.dma_bd(%c6_Q : memref<322xbf16>, 0, 322)
      aie.use_lock(%c6_Qp, Release, 1)
      aie.next_bd ^oh6
    ^oh6:
      aie.use_lock(%c6_Oc, AcquireGreaterEqual, 1)
      aie.dma_bd(%c6_O : memref<2048xbf16>, 0, 256)
      aie.use_lock(%c6_Op, Release, 1)
      aie.next_bd ^qi6
    ^e6:
      aie.end
    }
    %c7_A0 = aie.buffer(%t7) {sym_name = "c7_A0"} : memref<4608xi8>
    %c7_A1 = aie.buffer(%t7) {sym_name = "c7_A1"} : memref<4608xi8>
    %c7_B0 = aie.buffer(%t7) {sym_name = "c7_B0"} : memref<2320xbf16>
    %c7_B1 = aie.buffer(%t7) {sym_name = "c7_B1"} : memref<2320xbf16>
    %c7_B2 = aie.buffer(%t7) {sym_name = "c7_B2"} : memref<2320xbf16>
    %c7_Q = aie.buffer(%t7) {sym_name = "c7_Q"} : memref<322xbf16>
    %c7_O = aie.buffer(%t7) {sym_name = "c7_O"} : memref<2048xbf16>
    %c7_P = aie.buffer(%t7) {sym_name = "c7_P"} : memref<2320xbf16>
    %c7_A0p = aie.lock(%t7, 0) {init = 1 : i32, sym_name = "c7_A0p"}
    %c7_A0c = aie.lock(%t7, 1) {init = 0 : i32, sym_name = "c7_A0c"}
    %c7_B0p = aie.lock(%t7, 2) {init = 1 : i32, sym_name = "c7_B0p"}
    %c7_B0c = aie.lock(%t7, 3) {init = 0 : i32, sym_name = "c7_B0c"}
    %c7_Qp = aie.lock(%t7, 4) {init = 1 : i32, sym_name = "c7_Qp"}
    %c7_Qc = aie.lock(%t7, 5) {init = 0 : i32, sym_name = "c7_Qc"}
    %c7_Op = aie.lock(%t7, 6) {init = 1 : i32, sym_name = "c7_Op"}
    %c7_Oc = aie.lock(%t7, 7) {init = 0 : i32, sym_name = "c7_Oc"}
    %c7_Pp = aie.lock(%t7, 8) {init = 1 : i32, sym_name = "c7_Pp"}
    %c7_Pc = aie.lock(%t7, 9) {init = 0 : i32, sym_name = "c7_Pc"}
    %c7_A1p = aie.lock(%t7, 10) {init = 1 : i32, sym_name = "c7_A1p"}
    %c7_A1c = aie.lock(%t7, 11) {init = 0 : i32, sym_name = "c7_A1c"}
    %c7_B1p = aie.lock(%t7, 12) {init = 1 : i32, sym_name = "c7_B1p"}
    %c7_B1c = aie.lock(%t7, 13) {init = 0 : i32, sym_name = "c7_B1c"}
    %c7_B2p = aie.lock(%t7, 14) {init = 1 : i32, sym_name = "c7_B2p"}
    %c7_B2c = aie.lock(%t7, 15) {init = 0 : i32, sym_name = "c7_B2c"}
    %c7_gate = aie.buffer(%t7) {sym_name = "c7_gate"} : memref<1024xbf16>
    %c7_up   = aie.buffer(%t7) {sym_name = "c7_up"}   : memref<1024xbf16>
    %c7_silu = aie.buffer(%t7) {sym_name = "c7_silu"} : memref<1024xbf16>
    %c7_uni_partial = aie.buffer(%t7) {sym_name = "c7_uni_partial"} : memref<128xi8>
    %core7 = aie.core(%t7) {
      %c0 = arith.constant 0 : index
      %cN = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      %c64 = arith.constant 64 : index
      %cF = arith.constant 256 : index
      %c4 = arith.constant 4 : i32
      %c8 = arith.constant 8 : i32
      %qr = arith.constant 256 : i32
      %hc = arith.constant 1024 : i32
      scf.for %tok = %c0 to %cN step %c1 {
        func.call @_ha_noop() : () -> ()
        // ---- phase 1: Q-GEMV + rope (B0 = x_bundle, nchunk=8) ----
        aie.use_lock(%c7_B0c, AcquireGreaterEqual, 1)
        aie.use_lock(%c7_Qp, AcquireGreaterEqual, 1)
        scf.for %j = %c0 to %c64 step %c2 {
          aie.use_lock(%c7_A0c, AcquireGreaterEqual, 1)
          %ji0 = arith.index_cast %j : index to i32
          func.call @generic_bcast_gemv_bf16_q(%ji0, %c7_A0, %c7_B0, %c7_uni_partial, %c8, %c7_Q) : (i32, memref<4608xi8>, memref<2320xbf16>, memref<128xi8>, i32, memref<322xbf16>) -> ()
          aie.use_lock(%c7_A0p, Release, 1)
          aie.use_lock(%c7_A1c, AcquireGreaterEqual, 1)
          %j1 = arith.addi %j, %c1 : index
          %ji1 = arith.index_cast %j1 : index to i32
          func.call @generic_bcast_gemv_bf16_q(%ji1, %c7_A1, %c7_B0, %c7_uni_partial, %c8, %c7_Q) : (i32, memref<4608xi8>, memref<2320xbf16>, memref<128xi8>, i32, memref<322xbf16>) -> ()
          aie.use_lock(%c7_A1p, Release, 1)
        }
        func.call @rope_bundled(%c7_Q, %c7_B0, %c7_Q, %qr) : (memref<322xbf16>, memref<2320xbf16>, memref<322xbf16>, i32) -> ()
        aie.use_lock(%c7_Qc, Release, 1)
        aie.use_lock(%c7_B0p, Release, 1)
        // ---- phase 2: O-proj (B1 = attn_out, nchunk=8) ----
        aie.use_lock(%c7_B1c, AcquireGreaterEqual, 1)
        aie.use_lock(%c7_Op, AcquireGreaterEqual, 1)
        scf.for %j = %c0 to %c64 step %c2 {
          aie.use_lock(%c7_A0c, AcquireGreaterEqual, 1)
          %ji0 = arith.index_cast %j : index to i32
          func.call @generic_bcast_gemv_bf16_o(%ji0, %c7_A0, %c7_B1, %c7_uni_partial, %c8, %c7_O) : (i32, memref<4608xi8>, memref<2320xbf16>, memref<128xi8>, i32, memref<2048xbf16>) -> ()
          aie.use_lock(%c7_A0p, Release, 1)
          aie.use_lock(%c7_A1c, AcquireGreaterEqual, 1)
          %j1 = arith.addi %j, %c1 : index
          %ji1 = arith.index_cast %j1 : index to i32
          func.call @generic_bcast_gemv_bf16_o(%ji1, %c7_A1, %c7_B1, %c7_uni_partial, %c8, %c7_O) : (i32, memref<4608xi8>, memref<2320xbf16>, memref<128xi8>, i32, memref<2048xbf16>) -> ()
          aie.use_lock(%c7_A1p, Release, 1)
        }
        aie.use_lock(%c7_B1p, Release, 1)
        aie.use_lock(%c7_Oc, Release, 1)
        // ---- phase 3: FFN (B2 = ffn_in) ----
        // 3a: gate GEMV (nchunk=8, output -> gate_buf)
        aie.use_lock(%c7_B2c, AcquireGreaterEqual, 1)
        scf.for %j = %c0 to %cF step %c2 {
          aie.use_lock(%c7_A0c, AcquireGreaterEqual, 1)
          %ji0 = arith.index_cast %j : index to i32
          func.call @generic_bcast_gemv_bf16_g(%ji0, %c7_A0, %c7_B2, %c7_uni_partial, %c8, %c7_gate) : (i32, memref<4608xi8>, memref<2320xbf16>, memref<128xi8>, i32, memref<1024xbf16>) -> ()
          aie.use_lock(%c7_A0p, Release, 1)
          aie.use_lock(%c7_A1c, AcquireGreaterEqual, 1)
          %j1 = arith.addi %j, %c1 : index
          %ji1 = arith.index_cast %j1 : index to i32
          func.call @generic_bcast_gemv_bf16_g(%ji1, %c7_A1, %c7_B2, %c7_uni_partial, %c8, %c7_gate) : (i32, memref<4608xi8>, memref<2320xbf16>, memref<128xi8>, i32, memref<1024xbf16>) -> ()
          aie.use_lock(%c7_A1p, Release, 1)
        }
        // 3b: up GEMV (nchunk=8, output -> up_buf)
        scf.for %j = %c0 to %cF step %c2 {
          aie.use_lock(%c7_A0c, AcquireGreaterEqual, 1)
          %ji0 = arith.index_cast %j : index to i32
          func.call @generic_bcast_gemv_bf16_g(%ji0, %c7_A0, %c7_B2, %c7_uni_partial, %c8, %c7_up) : (i32, memref<4608xi8>, memref<2320xbf16>, memref<128xi8>, i32, memref<1024xbf16>) -> ()
          aie.use_lock(%c7_A0p, Release, 1)
          aie.use_lock(%c7_A1c, AcquireGreaterEqual, 1)
          %j1 = arith.addi %j, %c1 : index
          %ji1 = arith.index_cast %j1 : index to i32
          func.call @generic_bcast_gemv_bf16_g(%ji1, %c7_A1, %c7_B2, %c7_uni_partial, %c8, %c7_up) : (i32, memref<4608xi8>, memref<2320xbf16>, memref<128xi8>, i32, memref<1024xbf16>) -> ()
          aie.use_lock(%c7_A1p, Release, 1)
        }
        aie.use_lock(%c7_B2p, Release, 1)
        // 3c: SiLU(gate) * up -> silu_buf (explicit pointers)
        func.call @layer_fused_silu_mul_explicit_bf16(%c7_gate, %c7_up, %c7_silu, %hc) : (memref<1024xbf16>, memref<1024xbf16>, memref<1024xbf16>, i32) -> ()
        // 3d: down GEMV (nchunk=4, activation = silu_buf)
        aie.use_lock(%c7_Pp, AcquireGreaterEqual, 1)
        scf.for %j = %c0 to %cF step %c2 {
          aie.use_lock(%c7_A0c, AcquireGreaterEqual, 1)
          %ji0 = arith.index_cast %j : index to i32
          func.call @generic_bcast_gemv_bf16_d(%ji0, %c7_A0, %c7_silu, %c7_uni_partial, %c4, %c7_P) : (i32, memref<4608xi8>, memref<1024xbf16>, memref<128xi8>, i32, memref<2320xbf16>) -> ()
          aie.use_lock(%c7_A0p, Release, 1)
          aie.use_lock(%c7_A1c, AcquireGreaterEqual, 1)
          %j1 = arith.addi %j, %c1 : index
          %ji1 = arith.index_cast %j1 : index to i32
          func.call @generic_bcast_gemv_bf16_d(%ji1, %c7_A1, %c7_silu, %c7_uni_partial, %c4, %c7_P) : (i32, memref<4608xi8>, memref<1024xbf16>, memref<128xi8>, i32, memref<2320xbf16>) -> ()
          aie.use_lock(%c7_A1p, Release, 1)
        }
        aie.use_lock(%c7_Pc, Release, 1)
      }
      aie.end
    }
    %mem_t7 = aie.mem(%t7) {
      %s0 = aie.dma_start(S2MM, 0, ^a07, ^bs7)
    ^a07:
      aie.use_lock(%c7_A0p, AcquireGreaterEqual, 1)
      aie.dma_bd(%c7_A0 : memref<4608xi8>, 0, 4608)
      aie.use_lock(%c7_A0c, Release, 1)
      aie.next_bd ^a17
    ^a17:
      aie.use_lock(%c7_A1p, AcquireGreaterEqual, 1)
      aie.dma_bd(%c7_A1 : memref<4608xi8>, 0, 4608)
      aie.use_lock(%c7_A1c, Release, 1)
      aie.next_bd ^a07
    ^bs7:
      %s1 = aie.dma_start(S2MM, 1, ^b07, ^m07)
    ^b07:
      aie.use_lock(%c7_B0p, AcquireGreaterEqual, 1)
      aie.dma_bd(%c7_B0 : memref<2320xbf16>, 0, 2320)
      aie.use_lock(%c7_B0c, Release, 1)
      aie.next_bd ^b17
    ^b17:
      aie.use_lock(%c7_B1p, AcquireGreaterEqual, 1)
      aie.dma_bd(%c7_B1 : memref<2320xbf16>, 0, 2320)
      aie.use_lock(%c7_B1c, Release, 1)
      aie.next_bd ^b27
    ^b27:
      aie.use_lock(%c7_B2p, AcquireGreaterEqual, 1)
      aie.dma_bd(%c7_B2 : memref<2320xbf16>, 0, 2320)
      aie.use_lock(%c7_B2c, Release, 1)
      aie.next_bd ^b07
    ^m07:
      %m0 = aie.dma_start(MM2S, 0, ^pf7, ^qo7)
    ^pf7:
      aie.use_lock(%c7_Pc, AcquireGreaterEqual, 1)
      aie.dma_bd(%c7_P : memref<2320xbf16>, 0, 2048)
      aie.use_lock(%c7_Pp, Release, 1)
      aie.next_bd ^pf7
    ^qo7:
      %m1 = aie.dma_start(MM2S, 1, ^qi7, ^e7)
    ^qi7:
      aie.use_lock(%c7_Qc, AcquireGreaterEqual, 1)
      aie.dma_bd(%c7_Q : memref<322xbf16>, 0, 322)
      aie.use_lock(%c7_Qp, Release, 1)
      aie.next_bd ^oh7
    ^oh7:
      aie.use_lock(%c7_Oc, AcquireGreaterEqual, 1)
      aie.dma_bd(%c7_O : memref<2048xbf16>, 0, 256)
      aie.use_lock(%c7_Op, Release, 1)
      aie.next_bd ^qi7
    ^e7:
      aie.end
    }

    %sc0_Qs = aie.buffer(%sc0) {sym_name = "sc0_Qs"} : memref<322xbf16>
    %sc0_K  = aie.buffer(%sc0) {sym_name = "sc0_K"}  : memref<8192xbf16>
    %sc0_It = aie.buffer(%sc0) {sym_name = "sc0_It"} : memref<520xbf16>
    %sc0_Oh = aie.buffer(%sc0) {sym_name = "sc0_Oh"} : memref<256xbf16>
    %sc0_Qp = aie.lock(%sc0, 0) {init = 1 : i32, sym_name = "sc0_Qp"}
    %sc0_Qc = aie.lock(%sc0, 1) {init = 0 : i32, sym_name = "sc0_Qc"}
    %sc0_Kp = aie.lock(%sc0, 2) {init = 1 : i32, sym_name = "sc0_Kp"}
    %sc0_Kc = aie.lock(%sc0, 3) {init = 0 : i32, sym_name = "sc0_Kc"}
    %sc0_Ip = aie.lock(%sc0, 4) {init = 1 : i32, sym_name = "sc0_Ip"}
    %sc0_Ic = aie.lock(%sc0, 5) {init = 0 : i32, sym_name = "sc0_Ic"}
    %sc0_Ohp = aie.lock(%sc0, 6) {init = 1 : i32, sym_name = "sc0_Ohp"}
    %sc0_Ohc = aie.lock(%sc0, 7) {init = 0 : i32, sym_name = "sc0_Ohc"}
    %sc0_Ohdp = aie.lock(%sc0, 8) {init = 1 : i32, sym_name = "sc0_Ohdp"}
    %sc0_Ohdc = aie.lock(%sc0, 9) {init = 0 : i32, sym_name = "sc0_Ohdc"}
    %core_sc0 = aie.core(%sc0) {
      %c0 = arith.constant 0 : index
      %cN = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      %a4 = arith.constant 4 : i32
      %h64 = arith.constant 64 : i32
      %cnc = arith.constant 2 : index
      %sC = arith.constant 128 : i32
      scf.for %tok = %c0 to %cN step %c1 {
        // ph1: scoring — multi-chunk online softmax (chunk=128, nchunk=2)
        func.call @flowkv_score_init_bf16(%a4) : (i32) -> ()
        aie.use_lock(%sc0_Qc, AcquireGreaterEqual, 1)
        func.call @flowkv_score_rope_q_bf16(%sc0_Qs, %a4, %h64) : (memref<322xbf16>, i32, i32) -> ()
        scf.for %ci = %c0 to %cnc step %c1 {
          aie.use_lock(%sc0_Kc, AcquireGreaterEqual, 1)
          aie.use_lock(%sc0_Ip, AcquireGreaterEqual, 1)
          func.call @flowkv_score_chunk_bf16(%sc0_Qs, %sc0_K, %sc0_It, %a4, %h64, %sC) : (memref<322xbf16>, memref<8192xbf16>, memref<520xbf16>, i32, i32, i32) -> ()
          aie.use_lock(%sc0_Kp, Release, 1)
          aie.use_lock(%sc0_Ic, Release, 1)
        }
        aie.use_lock(%sc0_Qp, Release, 1)
        // ph2: relay O_h (S2MM0-BD1 -> MM2S1, no compute, rl-style lock-dance)
        aie.use_lock(%sc0_Ohc, AcquireGreaterEqual, 1)
        aie.use_lock(%sc0_Ohdp, AcquireGreaterEqual, 1)
        aie.use_lock(%sc0_Ohp, Release, 1)
        aie.use_lock(%sc0_Ohdc, Release, 1)
      }
      aie.end
    }
    %mem_sc0 = aie.mem(%sc0) {
      %s0 = aie.dma_start(S2MM, 0, ^q0, ^ks0)
    ^q0:
      aie.use_lock(%sc0_Qp, AcquireGreaterEqual, 1)
      aie.dma_bd(%sc0_Qs : memref<322xbf16>, 0, 322)
      aie.use_lock(%sc0_Qc, Release, 1)
      aie.next_bd ^oh0
    ^oh0:
      aie.use_lock(%sc0_Ohp, AcquireGreaterEqual, 1)
      aie.dma_bd(%sc0_Oh : memref<256xbf16>, 0, 256)
      aie.use_lock(%sc0_Ohc, Release, 1)
      aie.next_bd ^q0
    ^ks0:
      %s1 = aie.dma_start(S2MM, 1, ^k0, ^im0)
    ^k0:
      aie.use_lock(%sc0_Kp, AcquireGreaterEqual, 1)
      aie.dma_bd(%sc0_K : memref<8192xbf16>, 0, 8192)
      aie.use_lock(%sc0_Kc, Release, 1)
      aie.next_bd ^k0
    ^im0:
      %m0 = aie.dma_start(MM2S, 0, ^io0, ^ohm0)
    ^io0:
      aie.use_lock(%sc0_Ic, AcquireGreaterEqual, 1)
      aie.dma_bd(%sc0_It : memref<520xbf16>, 0, 520)
      aie.use_lock(%sc0_Ip, Release, 1)
      aie.next_bd ^io0
    ^ohm0:
      %m1 = aie.dma_start(MM2S, 1, ^ohf0, ^e0)
    ^ohf0:
      aie.use_lock(%sc0_Ohdc, AcquireGreaterEqual, 1)
      aie.dma_bd(%sc0_Oh : memref<256xbf16>, 0, 256)
      aie.use_lock(%sc0_Ohdp, Release, 1)
      aie.next_bd ^ohf0
    ^e0:
      aie.end
    }
    %sc1_Qs = aie.buffer(%sc1) {sym_name = "sc1_Qs"} : memref<322xbf16>
    %sc1_K  = aie.buffer(%sc1) {sym_name = "sc1_K"}  : memref<8192xbf16>
    %sc1_It = aie.buffer(%sc1) {sym_name = "sc1_It"} : memref<520xbf16>
    %sc1_Oh = aie.buffer(%sc1) {sym_name = "sc1_Oh"} : memref<256xbf16>
    %sc1_Qp = aie.lock(%sc1, 0) {init = 1 : i32, sym_name = "sc1_Qp"}
    %sc1_Qc = aie.lock(%sc1, 1) {init = 0 : i32, sym_name = "sc1_Qc"}
    %sc1_Kp = aie.lock(%sc1, 2) {init = 1 : i32, sym_name = "sc1_Kp"}
    %sc1_Kc = aie.lock(%sc1, 3) {init = 0 : i32, sym_name = "sc1_Kc"}
    %sc1_Ip = aie.lock(%sc1, 4) {init = 1 : i32, sym_name = "sc1_Ip"}
    %sc1_Ic = aie.lock(%sc1, 5) {init = 0 : i32, sym_name = "sc1_Ic"}
    %sc1_Ohp = aie.lock(%sc1, 6) {init = 1 : i32, sym_name = "sc1_Ohp"}
    %sc1_Ohc = aie.lock(%sc1, 7) {init = 0 : i32, sym_name = "sc1_Ohc"}
    %sc1_Ohdp = aie.lock(%sc1, 8) {init = 1 : i32, sym_name = "sc1_Ohdp"}
    %sc1_Ohdc = aie.lock(%sc1, 9) {init = 0 : i32, sym_name = "sc1_Ohdc"}
    %core_sc1 = aie.core(%sc1) {
      %c0 = arith.constant 0 : index
      %cN = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      %a4 = arith.constant 4 : i32
      %h64 = arith.constant 64 : i32
      %cnc = arith.constant 2 : index
      %sC = arith.constant 128 : i32
      scf.for %tok = %c0 to %cN step %c1 {
        // ph1: scoring — multi-chunk online softmax (chunk=128, nchunk=2)
        func.call @flowkv_score_init_bf16(%a4) : (i32) -> ()
        aie.use_lock(%sc1_Qc, AcquireGreaterEqual, 1)
        func.call @flowkv_score_rope_q_bf16(%sc1_Qs, %a4, %h64) : (memref<322xbf16>, i32, i32) -> ()
        scf.for %ci = %c0 to %cnc step %c1 {
          aie.use_lock(%sc1_Kc, AcquireGreaterEqual, 1)
          aie.use_lock(%sc1_Ip, AcquireGreaterEqual, 1)
          func.call @flowkv_score_chunk_bf16(%sc1_Qs, %sc1_K, %sc1_It, %a4, %h64, %sC) : (memref<322xbf16>, memref<8192xbf16>, memref<520xbf16>, i32, i32, i32) -> ()
          aie.use_lock(%sc1_Kp, Release, 1)
          aie.use_lock(%sc1_Ic, Release, 1)
        }
        aie.use_lock(%sc1_Qp, Release, 1)
        // ph2: relay O_h (S2MM0-BD1 -> MM2S1, no compute, rl-style lock-dance)
        aie.use_lock(%sc1_Ohc, AcquireGreaterEqual, 1)
        aie.use_lock(%sc1_Ohdp, AcquireGreaterEqual, 1)
        aie.use_lock(%sc1_Ohp, Release, 1)
        aie.use_lock(%sc1_Ohdc, Release, 1)
      }
      aie.end
    }
    %mem_sc1 = aie.mem(%sc1) {
      %s0 = aie.dma_start(S2MM, 0, ^q1, ^ks1)
    ^q1:
      aie.use_lock(%sc1_Qp, AcquireGreaterEqual, 1)
      aie.dma_bd(%sc1_Qs : memref<322xbf16>, 0, 322)
      aie.use_lock(%sc1_Qc, Release, 1)
      aie.next_bd ^oh1
    ^oh1:
      aie.use_lock(%sc1_Ohp, AcquireGreaterEqual, 1)
      aie.dma_bd(%sc1_Oh : memref<256xbf16>, 0, 256)
      aie.use_lock(%sc1_Ohc, Release, 1)
      aie.next_bd ^q1
    ^ks1:
      %s1 = aie.dma_start(S2MM, 1, ^k1, ^im1)
    ^k1:
      aie.use_lock(%sc1_Kp, AcquireGreaterEqual, 1)
      aie.dma_bd(%sc1_K : memref<8192xbf16>, 0, 8192)
      aie.use_lock(%sc1_Kc, Release, 1)
      aie.next_bd ^k1
    ^im1:
      %m0 = aie.dma_start(MM2S, 0, ^io1, ^ohm1)
    ^io1:
      aie.use_lock(%sc1_Ic, AcquireGreaterEqual, 1)
      aie.dma_bd(%sc1_It : memref<520xbf16>, 0, 520)
      aie.use_lock(%sc1_Ip, Release, 1)
      aie.next_bd ^io1
    ^ohm1:
      %m1 = aie.dma_start(MM2S, 1, ^ohf1, ^e1)
    ^ohf1:
      aie.use_lock(%sc1_Ohdc, AcquireGreaterEqual, 1)
      aie.dma_bd(%sc1_Oh : memref<256xbf16>, 0, 256)
      aie.use_lock(%sc1_Ohdp, Release, 1)
      aie.next_bd ^ohf1
    ^e1:
      aie.end
    }
    %sc2_Qs = aie.buffer(%sc2) {sym_name = "sc2_Qs"} : memref<322xbf16>
    %sc2_K  = aie.buffer(%sc2) {sym_name = "sc2_K"}  : memref<8192xbf16>
    %sc2_It = aie.buffer(%sc2) {sym_name = "sc2_It"} : memref<520xbf16>
    %sc2_Oh = aie.buffer(%sc2) {sym_name = "sc2_Oh"} : memref<256xbf16>
    %sc2_Qp = aie.lock(%sc2, 0) {init = 1 : i32, sym_name = "sc2_Qp"}
    %sc2_Qc = aie.lock(%sc2, 1) {init = 0 : i32, sym_name = "sc2_Qc"}
    %sc2_Kp = aie.lock(%sc2, 2) {init = 1 : i32, sym_name = "sc2_Kp"}
    %sc2_Kc = aie.lock(%sc2, 3) {init = 0 : i32, sym_name = "sc2_Kc"}
    %sc2_Ip = aie.lock(%sc2, 4) {init = 1 : i32, sym_name = "sc2_Ip"}
    %sc2_Ic = aie.lock(%sc2, 5) {init = 0 : i32, sym_name = "sc2_Ic"}
    %sc2_Ohp = aie.lock(%sc2, 6) {init = 1 : i32, sym_name = "sc2_Ohp"}
    %sc2_Ohc = aie.lock(%sc2, 7) {init = 0 : i32, sym_name = "sc2_Ohc"}
    %sc2_Ohdp = aie.lock(%sc2, 8) {init = 1 : i32, sym_name = "sc2_Ohdp"}
    %sc2_Ohdc = aie.lock(%sc2, 9) {init = 0 : i32, sym_name = "sc2_Ohdc"}
    %core_sc2 = aie.core(%sc2) {
      %c0 = arith.constant 0 : index
      %cN = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      %a4 = arith.constant 4 : i32
      %h64 = arith.constant 64 : i32
      %cnc = arith.constant 2 : index
      %sC = arith.constant 128 : i32
      scf.for %tok = %c0 to %cN step %c1 {
        // ph1: scoring — multi-chunk online softmax (chunk=128, nchunk=2)
        func.call @flowkv_score_init_bf16(%a4) : (i32) -> ()
        aie.use_lock(%sc2_Qc, AcquireGreaterEqual, 1)
        func.call @flowkv_score_rope_q_bf16(%sc2_Qs, %a4, %h64) : (memref<322xbf16>, i32, i32) -> ()
        scf.for %ci = %c0 to %cnc step %c1 {
          aie.use_lock(%sc2_Kc, AcquireGreaterEqual, 1)
          aie.use_lock(%sc2_Ip, AcquireGreaterEqual, 1)
          func.call @flowkv_score_chunk_bf16(%sc2_Qs, %sc2_K, %sc2_It, %a4, %h64, %sC) : (memref<322xbf16>, memref<8192xbf16>, memref<520xbf16>, i32, i32, i32) -> ()
          aie.use_lock(%sc2_Kp, Release, 1)
          aie.use_lock(%sc2_Ic, Release, 1)
        }
        aie.use_lock(%sc2_Qp, Release, 1)
        // ph2: relay O_h (S2MM0-BD1 -> MM2S1, no compute, rl-style lock-dance)
        aie.use_lock(%sc2_Ohc, AcquireGreaterEqual, 1)
        aie.use_lock(%sc2_Ohdp, AcquireGreaterEqual, 1)
        aie.use_lock(%sc2_Ohp, Release, 1)
        aie.use_lock(%sc2_Ohdc, Release, 1)
      }
      aie.end
    }
    %mem_sc2 = aie.mem(%sc2) {
      %s0 = aie.dma_start(S2MM, 0, ^q2, ^ks2)
    ^q2:
      aie.use_lock(%sc2_Qp, AcquireGreaterEqual, 1)
      aie.dma_bd(%sc2_Qs : memref<322xbf16>, 0, 322)
      aie.use_lock(%sc2_Qc, Release, 1)
      aie.next_bd ^oh2
    ^oh2:
      aie.use_lock(%sc2_Ohp, AcquireGreaterEqual, 1)
      aie.dma_bd(%sc2_Oh : memref<256xbf16>, 0, 256)
      aie.use_lock(%sc2_Ohc, Release, 1)
      aie.next_bd ^q2
    ^ks2:
      %s1 = aie.dma_start(S2MM, 1, ^k2, ^im2)
    ^k2:
      aie.use_lock(%sc2_Kp, AcquireGreaterEqual, 1)
      aie.dma_bd(%sc2_K : memref<8192xbf16>, 0, 8192)
      aie.use_lock(%sc2_Kc, Release, 1)
      aie.next_bd ^k2
    ^im2:
      %m0 = aie.dma_start(MM2S, 0, ^io2, ^ohm2)
    ^io2:
      aie.use_lock(%sc2_Ic, AcquireGreaterEqual, 1)
      aie.dma_bd(%sc2_It : memref<520xbf16>, 0, 520)
      aie.use_lock(%sc2_Ip, Release, 1)
      aie.next_bd ^io2
    ^ohm2:
      %m1 = aie.dma_start(MM2S, 1, ^ohf2, ^e2)
    ^ohf2:
      aie.use_lock(%sc2_Ohdc, AcquireGreaterEqual, 1)
      aie.dma_bd(%sc2_Oh : memref<256xbf16>, 0, 256)
      aie.use_lock(%sc2_Ohdp, Release, 1)
      aie.next_bd ^ohf2
    ^e2:
      aie.end
    }
    %sc3_Qs = aie.buffer(%sc3) {sym_name = "sc3_Qs"} : memref<322xbf16>
    %sc3_K  = aie.buffer(%sc3) {sym_name = "sc3_K"}  : memref<8192xbf16>
    %sc3_It = aie.buffer(%sc3) {sym_name = "sc3_It"} : memref<520xbf16>
    %sc3_Oh = aie.buffer(%sc3) {sym_name = "sc3_Oh"} : memref<256xbf16>
    %sc3_Qp = aie.lock(%sc3, 0) {init = 1 : i32, sym_name = "sc3_Qp"}
    %sc3_Qc = aie.lock(%sc3, 1) {init = 0 : i32, sym_name = "sc3_Qc"}
    %sc3_Kp = aie.lock(%sc3, 2) {init = 1 : i32, sym_name = "sc3_Kp"}
    %sc3_Kc = aie.lock(%sc3, 3) {init = 0 : i32, sym_name = "sc3_Kc"}
    %sc3_Ip = aie.lock(%sc3, 4) {init = 1 : i32, sym_name = "sc3_Ip"}
    %sc3_Ic = aie.lock(%sc3, 5) {init = 0 : i32, sym_name = "sc3_Ic"}
    %sc3_Ohp = aie.lock(%sc3, 6) {init = 1 : i32, sym_name = "sc3_Ohp"}
    %sc3_Ohc = aie.lock(%sc3, 7) {init = 0 : i32, sym_name = "sc3_Ohc"}
    %sc3_Ohdp = aie.lock(%sc3, 8) {init = 1 : i32, sym_name = "sc3_Ohdp"}
    %sc3_Ohdc = aie.lock(%sc3, 9) {init = 0 : i32, sym_name = "sc3_Ohdc"}
    %core_sc3 = aie.core(%sc3) {
      %c0 = arith.constant 0 : index
      %cN = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      %a4 = arith.constant 4 : i32
      %h64 = arith.constant 64 : i32
      %cnc = arith.constant 2 : index
      %sC = arith.constant 128 : i32
      scf.for %tok = %c0 to %cN step %c1 {
        // ph1: scoring — multi-chunk online softmax (chunk=128, nchunk=2)
        func.call @flowkv_score_init_bf16(%a4) : (i32) -> ()
        aie.use_lock(%sc3_Qc, AcquireGreaterEqual, 1)
        func.call @flowkv_score_rope_q_bf16(%sc3_Qs, %a4, %h64) : (memref<322xbf16>, i32, i32) -> ()
        scf.for %ci = %c0 to %cnc step %c1 {
          aie.use_lock(%sc3_Kc, AcquireGreaterEqual, 1)
          aie.use_lock(%sc3_Ip, AcquireGreaterEqual, 1)
          func.call @flowkv_score_chunk_bf16(%sc3_Qs, %sc3_K, %sc3_It, %a4, %h64, %sC) : (memref<322xbf16>, memref<8192xbf16>, memref<520xbf16>, i32, i32, i32) -> ()
          aie.use_lock(%sc3_Kp, Release, 1)
          aie.use_lock(%sc3_Ic, Release, 1)
        }
        aie.use_lock(%sc3_Qp, Release, 1)
        // ph2: relay O_h (S2MM0-BD1 -> MM2S1, no compute, rl-style lock-dance)
        aie.use_lock(%sc3_Ohc, AcquireGreaterEqual, 1)
        aie.use_lock(%sc3_Ohdp, AcquireGreaterEqual, 1)
        aie.use_lock(%sc3_Ohp, Release, 1)
        aie.use_lock(%sc3_Ohdc, Release, 1)
      }
      aie.end
    }
    %mem_sc3 = aie.mem(%sc3) {
      %s0 = aie.dma_start(S2MM, 0, ^q3, ^ks3)
    ^q3:
      aie.use_lock(%sc3_Qp, AcquireGreaterEqual, 1)
      aie.dma_bd(%sc3_Qs : memref<322xbf16>, 0, 322)
      aie.use_lock(%sc3_Qc, Release, 1)
      aie.next_bd ^oh3
    ^oh3:
      aie.use_lock(%sc3_Ohp, AcquireGreaterEqual, 1)
      aie.dma_bd(%sc3_Oh : memref<256xbf16>, 0, 256)
      aie.use_lock(%sc3_Ohc, Release, 1)
      aie.next_bd ^q3
    ^ks3:
      %s1 = aie.dma_start(S2MM, 1, ^k3, ^im3)
    ^k3:
      aie.use_lock(%sc3_Kp, AcquireGreaterEqual, 1)
      aie.dma_bd(%sc3_K : memref<8192xbf16>, 0, 8192)
      aie.use_lock(%sc3_Kc, Release, 1)
      aie.next_bd ^k3
    ^im3:
      %m0 = aie.dma_start(MM2S, 0, ^io3, ^ohm3)
    ^io3:
      aie.use_lock(%sc3_Ic, AcquireGreaterEqual, 1)
      aie.dma_bd(%sc3_It : memref<520xbf16>, 0, 520)
      aie.use_lock(%sc3_Ip, Release, 1)
      aie.next_bd ^io3
    ^ohm3:
      %m1 = aie.dma_start(MM2S, 1, ^ohf3, ^e3)
    ^ohf3:
      aie.use_lock(%sc3_Ohdc, AcquireGreaterEqual, 1)
      aie.dma_bd(%sc3_Oh : memref<256xbf16>, 0, 256)
      aie.use_lock(%sc3_Ohdp, Release, 1)
      aie.next_bd ^ohf3
    ^e3:
      aie.end
    }
    %sc4_Qs = aie.buffer(%sc4) {sym_name = "sc4_Qs"} : memref<322xbf16>
    %sc4_K  = aie.buffer(%sc4) {sym_name = "sc4_K"}  : memref<8192xbf16>
    %sc4_It = aie.buffer(%sc4) {sym_name = "sc4_It"} : memref<520xbf16>
    %sc4_Oh = aie.buffer(%sc4) {sym_name = "sc4_Oh"} : memref<256xbf16>
    %sc4_Qp = aie.lock(%sc4, 0) {init = 1 : i32, sym_name = "sc4_Qp"}
    %sc4_Qc = aie.lock(%sc4, 1) {init = 0 : i32, sym_name = "sc4_Qc"}
    %sc4_Kp = aie.lock(%sc4, 2) {init = 1 : i32, sym_name = "sc4_Kp"}
    %sc4_Kc = aie.lock(%sc4, 3) {init = 0 : i32, sym_name = "sc4_Kc"}
    %sc4_Ip = aie.lock(%sc4, 4) {init = 1 : i32, sym_name = "sc4_Ip"}
    %sc4_Ic = aie.lock(%sc4, 5) {init = 0 : i32, sym_name = "sc4_Ic"}
    %sc4_Ohp = aie.lock(%sc4, 6) {init = 1 : i32, sym_name = "sc4_Ohp"}
    %sc4_Ohc = aie.lock(%sc4, 7) {init = 0 : i32, sym_name = "sc4_Ohc"}
    %sc4_Ohdp = aie.lock(%sc4, 8) {init = 1 : i32, sym_name = "sc4_Ohdp"}
    %sc4_Ohdc = aie.lock(%sc4, 9) {init = 0 : i32, sym_name = "sc4_Ohdc"}
    %core_sc4 = aie.core(%sc4) {
      %c0 = arith.constant 0 : index
      %cN = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      %a4 = arith.constant 4 : i32
      %h64 = arith.constant 64 : i32
      %cnc = arith.constant 2 : index
      %sC = arith.constant 128 : i32
      scf.for %tok = %c0 to %cN step %c1 {
        // ph1: scoring — multi-chunk online softmax (chunk=128, nchunk=2)
        func.call @flowkv_score_init_bf16(%a4) : (i32) -> ()
        aie.use_lock(%sc4_Qc, AcquireGreaterEqual, 1)
        func.call @flowkv_score_rope_q_bf16(%sc4_Qs, %a4, %h64) : (memref<322xbf16>, i32, i32) -> ()
        scf.for %ci = %c0 to %cnc step %c1 {
          aie.use_lock(%sc4_Kc, AcquireGreaterEqual, 1)
          aie.use_lock(%sc4_Ip, AcquireGreaterEqual, 1)
          func.call @flowkv_score_chunk_bf16(%sc4_Qs, %sc4_K, %sc4_It, %a4, %h64, %sC) : (memref<322xbf16>, memref<8192xbf16>, memref<520xbf16>, i32, i32, i32) -> ()
          aie.use_lock(%sc4_Kp, Release, 1)
          aie.use_lock(%sc4_Ic, Release, 1)
        }
        aie.use_lock(%sc4_Qp, Release, 1)
        // ph2: relay O_h (S2MM0-BD1 -> MM2S1, no compute, rl-style lock-dance)
        aie.use_lock(%sc4_Ohc, AcquireGreaterEqual, 1)
        aie.use_lock(%sc4_Ohdp, AcquireGreaterEqual, 1)
        aie.use_lock(%sc4_Ohp, Release, 1)
        aie.use_lock(%sc4_Ohdc, Release, 1)
      }
      aie.end
    }
    %mem_sc4 = aie.mem(%sc4) {
      %s0 = aie.dma_start(S2MM, 0, ^q4, ^ks4)
    ^q4:
      aie.use_lock(%sc4_Qp, AcquireGreaterEqual, 1)
      aie.dma_bd(%sc4_Qs : memref<322xbf16>, 0, 322)
      aie.use_lock(%sc4_Qc, Release, 1)
      aie.next_bd ^oh4
    ^oh4:
      aie.use_lock(%sc4_Ohp, AcquireGreaterEqual, 1)
      aie.dma_bd(%sc4_Oh : memref<256xbf16>, 0, 256)
      aie.use_lock(%sc4_Ohc, Release, 1)
      aie.next_bd ^q4
    ^ks4:
      %s1 = aie.dma_start(S2MM, 1, ^k4, ^im4)
    ^k4:
      aie.use_lock(%sc4_Kp, AcquireGreaterEqual, 1)
      aie.dma_bd(%sc4_K : memref<8192xbf16>, 0, 8192)
      aie.use_lock(%sc4_Kc, Release, 1)
      aie.next_bd ^k4
    ^im4:
      %m0 = aie.dma_start(MM2S, 0, ^io4, ^ohm4)
    ^io4:
      aie.use_lock(%sc4_Ic, AcquireGreaterEqual, 1)
      aie.dma_bd(%sc4_It : memref<520xbf16>, 0, 520)
      aie.use_lock(%sc4_Ip, Release, 1)
      aie.next_bd ^io4
    ^ohm4:
      %m1 = aie.dma_start(MM2S, 1, ^ohf4, ^e4)
    ^ohf4:
      aie.use_lock(%sc4_Ohdc, AcquireGreaterEqual, 1)
      aie.dma_bd(%sc4_Oh : memref<256xbf16>, 0, 256)
      aie.use_lock(%sc4_Ohdp, Release, 1)
      aie.next_bd ^ohf4
    ^e4:
      aie.end
    }
    %sc5_Qs = aie.buffer(%sc5) {sym_name = "sc5_Qs"} : memref<322xbf16>
    %sc5_K  = aie.buffer(%sc5) {sym_name = "sc5_K"}  : memref<8192xbf16>
    %sc5_It = aie.buffer(%sc5) {sym_name = "sc5_It"} : memref<520xbf16>
    %sc5_Oh = aie.buffer(%sc5) {sym_name = "sc5_Oh"} : memref<256xbf16>
    %sc5_Qp = aie.lock(%sc5, 0) {init = 1 : i32, sym_name = "sc5_Qp"}
    %sc5_Qc = aie.lock(%sc5, 1) {init = 0 : i32, sym_name = "sc5_Qc"}
    %sc5_Kp = aie.lock(%sc5, 2) {init = 1 : i32, sym_name = "sc5_Kp"}
    %sc5_Kc = aie.lock(%sc5, 3) {init = 0 : i32, sym_name = "sc5_Kc"}
    %sc5_Ip = aie.lock(%sc5, 4) {init = 1 : i32, sym_name = "sc5_Ip"}
    %sc5_Ic = aie.lock(%sc5, 5) {init = 0 : i32, sym_name = "sc5_Ic"}
    %sc5_Ohp = aie.lock(%sc5, 6) {init = 1 : i32, sym_name = "sc5_Ohp"}
    %sc5_Ohc = aie.lock(%sc5, 7) {init = 0 : i32, sym_name = "sc5_Ohc"}
    %sc5_Ohdp = aie.lock(%sc5, 8) {init = 1 : i32, sym_name = "sc5_Ohdp"}
    %sc5_Ohdc = aie.lock(%sc5, 9) {init = 0 : i32, sym_name = "sc5_Ohdc"}
    %core_sc5 = aie.core(%sc5) {
      %c0 = arith.constant 0 : index
      %cN = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      %a4 = arith.constant 4 : i32
      %h64 = arith.constant 64 : i32
      %cnc = arith.constant 2 : index
      %sC = arith.constant 128 : i32
      scf.for %tok = %c0 to %cN step %c1 {
        // ph1: scoring — multi-chunk online softmax (chunk=128, nchunk=2)
        func.call @flowkv_score_init_bf16(%a4) : (i32) -> ()
        aie.use_lock(%sc5_Qc, AcquireGreaterEqual, 1)
        func.call @flowkv_score_rope_q_bf16(%sc5_Qs, %a4, %h64) : (memref<322xbf16>, i32, i32) -> ()
        scf.for %ci = %c0 to %cnc step %c1 {
          aie.use_lock(%sc5_Kc, AcquireGreaterEqual, 1)
          aie.use_lock(%sc5_Ip, AcquireGreaterEqual, 1)
          func.call @flowkv_score_chunk_bf16(%sc5_Qs, %sc5_K, %sc5_It, %a4, %h64, %sC) : (memref<322xbf16>, memref<8192xbf16>, memref<520xbf16>, i32, i32, i32) -> ()
          aie.use_lock(%sc5_Kp, Release, 1)
          aie.use_lock(%sc5_Ic, Release, 1)
        }
        aie.use_lock(%sc5_Qp, Release, 1)
        // ph2: relay O_h (S2MM0-BD1 -> MM2S1, no compute, rl-style lock-dance)
        aie.use_lock(%sc5_Ohc, AcquireGreaterEqual, 1)
        aie.use_lock(%sc5_Ohdp, AcquireGreaterEqual, 1)
        aie.use_lock(%sc5_Ohp, Release, 1)
        aie.use_lock(%sc5_Ohdc, Release, 1)
      }
      aie.end
    }
    %mem_sc5 = aie.mem(%sc5) {
      %s0 = aie.dma_start(S2MM, 0, ^q5, ^ks5)
    ^q5:
      aie.use_lock(%sc5_Qp, AcquireGreaterEqual, 1)
      aie.dma_bd(%sc5_Qs : memref<322xbf16>, 0, 322)
      aie.use_lock(%sc5_Qc, Release, 1)
      aie.next_bd ^oh5
    ^oh5:
      aie.use_lock(%sc5_Ohp, AcquireGreaterEqual, 1)
      aie.dma_bd(%sc5_Oh : memref<256xbf16>, 0, 256)
      aie.use_lock(%sc5_Ohc, Release, 1)
      aie.next_bd ^q5
    ^ks5:
      %s1 = aie.dma_start(S2MM, 1, ^k5, ^im5)
    ^k5:
      aie.use_lock(%sc5_Kp, AcquireGreaterEqual, 1)
      aie.dma_bd(%sc5_K : memref<8192xbf16>, 0, 8192)
      aie.use_lock(%sc5_Kc, Release, 1)
      aie.next_bd ^k5
    ^im5:
      %m0 = aie.dma_start(MM2S, 0, ^io5, ^ohm5)
    ^io5:
      aie.use_lock(%sc5_Ic, AcquireGreaterEqual, 1)
      aie.dma_bd(%sc5_It : memref<520xbf16>, 0, 520)
      aie.use_lock(%sc5_Ip, Release, 1)
      aie.next_bd ^io5
    ^ohm5:
      %m1 = aie.dma_start(MM2S, 1, ^ohf5, ^e5)
    ^ohf5:
      aie.use_lock(%sc5_Ohdc, AcquireGreaterEqual, 1)
      aie.dma_bd(%sc5_Oh : memref<256xbf16>, 0, 256)
      aie.use_lock(%sc5_Ohdp, Release, 1)
      aie.next_bd ^ohf5
    ^e5:
      aie.end
    }
    %sc6_Qs = aie.buffer(%sc6) {sym_name = "sc6_Qs"} : memref<322xbf16>
    %sc6_K  = aie.buffer(%sc6) {sym_name = "sc6_K"}  : memref<8192xbf16>
    %sc6_It = aie.buffer(%sc6) {sym_name = "sc6_It"} : memref<520xbf16>
    %sc6_Oh = aie.buffer(%sc6) {sym_name = "sc6_Oh"} : memref<256xbf16>
    %sc6_Qp = aie.lock(%sc6, 0) {init = 1 : i32, sym_name = "sc6_Qp"}
    %sc6_Qc = aie.lock(%sc6, 1) {init = 0 : i32, sym_name = "sc6_Qc"}
    %sc6_Kp = aie.lock(%sc6, 2) {init = 1 : i32, sym_name = "sc6_Kp"}
    %sc6_Kc = aie.lock(%sc6, 3) {init = 0 : i32, sym_name = "sc6_Kc"}
    %sc6_Ip = aie.lock(%sc6, 4) {init = 1 : i32, sym_name = "sc6_Ip"}
    %sc6_Ic = aie.lock(%sc6, 5) {init = 0 : i32, sym_name = "sc6_Ic"}
    %sc6_Ohp = aie.lock(%sc6, 6) {init = 1 : i32, sym_name = "sc6_Ohp"}
    %sc6_Ohc = aie.lock(%sc6, 7) {init = 0 : i32, sym_name = "sc6_Ohc"}
    %sc6_Ohdp = aie.lock(%sc6, 8) {init = 1 : i32, sym_name = "sc6_Ohdp"}
    %sc6_Ohdc = aie.lock(%sc6, 9) {init = 0 : i32, sym_name = "sc6_Ohdc"}
    %core_sc6 = aie.core(%sc6) {
      %c0 = arith.constant 0 : index
      %cN = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      %a4 = arith.constant 4 : i32
      %h64 = arith.constant 64 : i32
      %cnc = arith.constant 2 : index
      %sC = arith.constant 128 : i32
      scf.for %tok = %c0 to %cN step %c1 {
        // ph1: scoring — multi-chunk online softmax (chunk=128, nchunk=2)
        func.call @flowkv_score_init_bf16(%a4) : (i32) -> ()
        aie.use_lock(%sc6_Qc, AcquireGreaterEqual, 1)
        func.call @flowkv_score_rope_q_bf16(%sc6_Qs, %a4, %h64) : (memref<322xbf16>, i32, i32) -> ()
        scf.for %ci = %c0 to %cnc step %c1 {
          aie.use_lock(%sc6_Kc, AcquireGreaterEqual, 1)
          aie.use_lock(%sc6_Ip, AcquireGreaterEqual, 1)
          func.call @flowkv_score_chunk_bf16(%sc6_Qs, %sc6_K, %sc6_It, %a4, %h64, %sC) : (memref<322xbf16>, memref<8192xbf16>, memref<520xbf16>, i32, i32, i32) -> ()
          aie.use_lock(%sc6_Kp, Release, 1)
          aie.use_lock(%sc6_Ic, Release, 1)
        }
        aie.use_lock(%sc6_Qp, Release, 1)
        // ph2: relay O_h (S2MM0-BD1 -> MM2S1, no compute, rl-style lock-dance)
        aie.use_lock(%sc6_Ohc, AcquireGreaterEqual, 1)
        aie.use_lock(%sc6_Ohdp, AcquireGreaterEqual, 1)
        aie.use_lock(%sc6_Ohp, Release, 1)
        aie.use_lock(%sc6_Ohdc, Release, 1)
      }
      aie.end
    }
    %mem_sc6 = aie.mem(%sc6) {
      %s0 = aie.dma_start(S2MM, 0, ^q6, ^ks6)
    ^q6:
      aie.use_lock(%sc6_Qp, AcquireGreaterEqual, 1)
      aie.dma_bd(%sc6_Qs : memref<322xbf16>, 0, 322)
      aie.use_lock(%sc6_Qc, Release, 1)
      aie.next_bd ^oh6
    ^oh6:
      aie.use_lock(%sc6_Ohp, AcquireGreaterEqual, 1)
      aie.dma_bd(%sc6_Oh : memref<256xbf16>, 0, 256)
      aie.use_lock(%sc6_Ohc, Release, 1)
      aie.next_bd ^q6
    ^ks6:
      %s1 = aie.dma_start(S2MM, 1, ^k6, ^im6)
    ^k6:
      aie.use_lock(%sc6_Kp, AcquireGreaterEqual, 1)
      aie.dma_bd(%sc6_K : memref<8192xbf16>, 0, 8192)
      aie.use_lock(%sc6_Kc, Release, 1)
      aie.next_bd ^k6
    ^im6:
      %m0 = aie.dma_start(MM2S, 0, ^io6, ^ohm6)
    ^io6:
      aie.use_lock(%sc6_Ic, AcquireGreaterEqual, 1)
      aie.dma_bd(%sc6_It : memref<520xbf16>, 0, 520)
      aie.use_lock(%sc6_Ip, Release, 1)
      aie.next_bd ^io6
    ^ohm6:
      %m1 = aie.dma_start(MM2S, 1, ^ohf6, ^e6)
    ^ohf6:
      aie.use_lock(%sc6_Ohdc, AcquireGreaterEqual, 1)
      aie.dma_bd(%sc6_Oh : memref<256xbf16>, 0, 256)
      aie.use_lock(%sc6_Ohdp, Release, 1)
      aie.next_bd ^ohf6
    ^e6:
      aie.end
    }
    %sc7_Qs = aie.buffer(%sc7) {sym_name = "sc7_Qs"} : memref<322xbf16>
    %sc7_K  = aie.buffer(%sc7) {sym_name = "sc7_K"}  : memref<8192xbf16>
    %sc7_It = aie.buffer(%sc7) {sym_name = "sc7_It"} : memref<520xbf16>
    %sc7_Oh = aie.buffer(%sc7) {sym_name = "sc7_Oh"} : memref<256xbf16>
    %sc7_Qp = aie.lock(%sc7, 0) {init = 1 : i32, sym_name = "sc7_Qp"}
    %sc7_Qc = aie.lock(%sc7, 1) {init = 0 : i32, sym_name = "sc7_Qc"}
    %sc7_Kp = aie.lock(%sc7, 2) {init = 1 : i32, sym_name = "sc7_Kp"}
    %sc7_Kc = aie.lock(%sc7, 3) {init = 0 : i32, sym_name = "sc7_Kc"}
    %sc7_Ip = aie.lock(%sc7, 4) {init = 1 : i32, sym_name = "sc7_Ip"}
    %sc7_Ic = aie.lock(%sc7, 5) {init = 0 : i32, sym_name = "sc7_Ic"}
    %sc7_Ohp = aie.lock(%sc7, 6) {init = 1 : i32, sym_name = "sc7_Ohp"}
    %sc7_Ohc = aie.lock(%sc7, 7) {init = 0 : i32, sym_name = "sc7_Ohc"}
    %sc7_Ohdp = aie.lock(%sc7, 8) {init = 1 : i32, sym_name = "sc7_Ohdp"}
    %sc7_Ohdc = aie.lock(%sc7, 9) {init = 0 : i32, sym_name = "sc7_Ohdc"}
    %core_sc7 = aie.core(%sc7) {
      %c0 = arith.constant 0 : index
      %cN = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      %a4 = arith.constant 4 : i32
      %h64 = arith.constant 64 : i32
      %cnc = arith.constant 2 : index
      %sC = arith.constant 128 : i32
      scf.for %tok = %c0 to %cN step %c1 {
        // ph1: scoring — multi-chunk online softmax (chunk=128, nchunk=2)
        func.call @flowkv_score_init_bf16(%a4) : (i32) -> ()
        aie.use_lock(%sc7_Qc, AcquireGreaterEqual, 1)
        func.call @flowkv_score_rope_q_bf16(%sc7_Qs, %a4, %h64) : (memref<322xbf16>, i32, i32) -> ()
        scf.for %ci = %c0 to %cnc step %c1 {
          aie.use_lock(%sc7_Kc, AcquireGreaterEqual, 1)
          aie.use_lock(%sc7_Ip, AcquireGreaterEqual, 1)
          func.call @flowkv_score_chunk_bf16(%sc7_Qs, %sc7_K, %sc7_It, %a4, %h64, %sC) : (memref<322xbf16>, memref<8192xbf16>, memref<520xbf16>, i32, i32, i32) -> ()
          aie.use_lock(%sc7_Kp, Release, 1)
          aie.use_lock(%sc7_Ic, Release, 1)
        }
        aie.use_lock(%sc7_Qp, Release, 1)
        // ph2: relay O_h (S2MM0-BD1 -> MM2S1, no compute, rl-style lock-dance)
        aie.use_lock(%sc7_Ohc, AcquireGreaterEqual, 1)
        aie.use_lock(%sc7_Ohdp, AcquireGreaterEqual, 1)
        aie.use_lock(%sc7_Ohp, Release, 1)
        aie.use_lock(%sc7_Ohdc, Release, 1)
      }
      aie.end
    }
    %mem_sc7 = aie.mem(%sc7) {
      %s0 = aie.dma_start(S2MM, 0, ^q7, ^ks7)
    ^q7:
      aie.use_lock(%sc7_Qp, AcquireGreaterEqual, 1)
      aie.dma_bd(%sc7_Qs : memref<322xbf16>, 0, 322)
      aie.use_lock(%sc7_Qc, Release, 1)
      aie.next_bd ^oh7
    ^oh7:
      aie.use_lock(%sc7_Ohp, AcquireGreaterEqual, 1)
      aie.dma_bd(%sc7_Oh : memref<256xbf16>, 0, 256)
      aie.use_lock(%sc7_Ohc, Release, 1)
      aie.next_bd ^q7
    ^ks7:
      %s1 = aie.dma_start(S2MM, 1, ^k7, ^im7)
    ^k7:
      aie.use_lock(%sc7_Kp, AcquireGreaterEqual, 1)
      aie.dma_bd(%sc7_K : memref<8192xbf16>, 0, 8192)
      aie.use_lock(%sc7_Kc, Release, 1)
      aie.next_bd ^k7
    ^im7:
      %m0 = aie.dma_start(MM2S, 0, ^io7, ^ohm7)
    ^io7:
      aie.use_lock(%sc7_Ic, AcquireGreaterEqual, 1)
      aie.dma_bd(%sc7_It : memref<520xbf16>, 0, 520)
      aie.use_lock(%sc7_Ip, Release, 1)
      aie.next_bd ^io7
    ^ohm7:
      %m1 = aie.dma_start(MM2S, 1, ^ohf7, ^e7)
    ^ohf7:
      aie.use_lock(%sc7_Ohdc, AcquireGreaterEqual, 1)
      aie.dma_bd(%sc7_Oh : memref<256xbf16>, 0, 256)
      aie.use_lock(%sc7_Ohdp, Release, 1)
      aie.next_bd ^ohf7
    ^e7:
      aie.end
    }

    %va0_Iv = aie.buffer(%va0) {sym_name = "va0_Iv"} : memref<520xbf16>
    %va0_V  = aie.buffer(%va0) {sym_name = "va0_V"}  : memref<8192xbf16>
    %va0_Of = aie.buffer(%va0) {sym_name = "va0_Of"} : memref<256xbf16>
    %va0_Ip = aie.lock(%va0, 0) {init = 1 : i32, sym_name = "va0_Ip"}
    %va0_Ic = aie.lock(%va0, 1) {init = 0 : i32, sym_name = "va0_Ic"}
    %va0_Vp = aie.lock(%va0, 2) {init = 1 : i32, sym_name = "va0_Vp"}
    %va0_Vc = aie.lock(%va0, 3) {init = 0 : i32, sym_name = "va0_Vc"}
    %va0_Op = aie.lock(%va0, 4) {init = 1 : i32, sym_name = "va0_Op"}
    %va0_Oc = aie.lock(%va0, 5) {init = 0 : i32, sym_name = "va0_Oc"}
    %core_va0 = aie.core(%va0) {
      %c0 = arith.constant 0 : index
      %cN = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      %a4 = arith.constant 4 : i32
      %h64 = arith.constant 64 : i32
      %cnc = arith.constant 2 : index
      %sC = arith.constant 128 : i32
      scf.for %tok = %c0 to %cN step %c1 {
        func.call @flowkv_value_init_bf16(%a4, %h64) : (i32, i32) -> ()
        scf.for %ci = %c0 to %cnc step %c1 {
          aie.use_lock(%va0_Ic, AcquireGreaterEqual, 1)
          aie.use_lock(%va0_Vc, AcquireGreaterEqual, 1)
          func.call @flowkv_value_accum_bf16(%va0_Iv, %va0_V, %a4, %h64, %sC) : (memref<520xbf16>, memref<8192xbf16>, i32, i32, i32) -> ()
          aie.use_lock(%va0_Ip, Release, 1)
          aie.use_lock(%va0_Vp, Release, 1)
        }
        aie.use_lock(%va0_Op, AcquireGreaterEqual, 1)
        func.call @flowkv_value_normalize_bf16(%va0_Of, %a4, %h64) : (memref<256xbf16>, i32, i32) -> ()
        aie.use_lock(%va0_Oc, Release, 1)
      }
      aie.end
    }
    %mem_va0 = aie.mem(%va0) {
      %s0 = aie.dma_start(S2MM, 0, ^iv0, ^vs0)
    ^iv0:
      aie.use_lock(%va0_Ip, AcquireGreaterEqual, 1)
      aie.dma_bd(%va0_Iv : memref<520xbf16>, 0, 520)
      aie.use_lock(%va0_Ic, Release, 1)
      aie.next_bd ^iv0
    ^vs0:
      %s1 = aie.dma_start(S2MM, 1, ^v0, ^om0)
    ^v0:
      aie.use_lock(%va0_Vp, AcquireGreaterEqual, 1)
      aie.dma_bd(%va0_V : memref<8192xbf16>, 0, 8192)
      aie.use_lock(%va0_Vc, Release, 1)
      aie.next_bd ^v0
    ^om0:
      %m0 = aie.dma_start(MM2S, 0, ^oo0, ^e0)
    ^oo0:
      aie.use_lock(%va0_Oc, AcquireGreaterEqual, 1)
      aie.dma_bd(%va0_Of : memref<256xbf16>, 0, 256)
      aie.use_lock(%va0_Op, Release, 1)
      aie.next_bd ^oo0
    ^e0:
      aie.end
    }
    %va1_Iv = aie.buffer(%va1) {sym_name = "va1_Iv"} : memref<520xbf16>
    %va1_V  = aie.buffer(%va1) {sym_name = "va1_V"}  : memref<8192xbf16>
    %va1_Of = aie.buffer(%va1) {sym_name = "va1_Of"} : memref<256xbf16>
    %va1_Ip = aie.lock(%va1, 0) {init = 1 : i32, sym_name = "va1_Ip"}
    %va1_Ic = aie.lock(%va1, 1) {init = 0 : i32, sym_name = "va1_Ic"}
    %va1_Vp = aie.lock(%va1, 2) {init = 1 : i32, sym_name = "va1_Vp"}
    %va1_Vc = aie.lock(%va1, 3) {init = 0 : i32, sym_name = "va1_Vc"}
    %va1_Op = aie.lock(%va1, 4) {init = 1 : i32, sym_name = "va1_Op"}
    %va1_Oc = aie.lock(%va1, 5) {init = 0 : i32, sym_name = "va1_Oc"}
    %core_va1 = aie.core(%va1) {
      %c0 = arith.constant 0 : index
      %cN = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      %a4 = arith.constant 4 : i32
      %h64 = arith.constant 64 : i32
      %cnc = arith.constant 2 : index
      %sC = arith.constant 128 : i32
      scf.for %tok = %c0 to %cN step %c1 {
        func.call @flowkv_value_init_bf16(%a4, %h64) : (i32, i32) -> ()
        scf.for %ci = %c0 to %cnc step %c1 {
          aie.use_lock(%va1_Ic, AcquireGreaterEqual, 1)
          aie.use_lock(%va1_Vc, AcquireGreaterEqual, 1)
          func.call @flowkv_value_accum_bf16(%va1_Iv, %va1_V, %a4, %h64, %sC) : (memref<520xbf16>, memref<8192xbf16>, i32, i32, i32) -> ()
          aie.use_lock(%va1_Ip, Release, 1)
          aie.use_lock(%va1_Vp, Release, 1)
        }
        aie.use_lock(%va1_Op, AcquireGreaterEqual, 1)
        func.call @flowkv_value_normalize_bf16(%va1_Of, %a4, %h64) : (memref<256xbf16>, i32, i32) -> ()
        aie.use_lock(%va1_Oc, Release, 1)
      }
      aie.end
    }
    %mem_va1 = aie.mem(%va1) {
      %s0 = aie.dma_start(S2MM, 0, ^iv1, ^vs1)
    ^iv1:
      aie.use_lock(%va1_Ip, AcquireGreaterEqual, 1)
      aie.dma_bd(%va1_Iv : memref<520xbf16>, 0, 520)
      aie.use_lock(%va1_Ic, Release, 1)
      aie.next_bd ^iv1
    ^vs1:
      %s1 = aie.dma_start(S2MM, 1, ^v1, ^om1)
    ^v1:
      aie.use_lock(%va1_Vp, AcquireGreaterEqual, 1)
      aie.dma_bd(%va1_V : memref<8192xbf16>, 0, 8192)
      aie.use_lock(%va1_Vc, Release, 1)
      aie.next_bd ^v1
    ^om1:
      %m0 = aie.dma_start(MM2S, 0, ^oo1, ^e1)
    ^oo1:
      aie.use_lock(%va1_Oc, AcquireGreaterEqual, 1)
      aie.dma_bd(%va1_Of : memref<256xbf16>, 0, 256)
      aie.use_lock(%va1_Op, Release, 1)
      aie.next_bd ^oo1
    ^e1:
      aie.end
    }
    %va2_Iv = aie.buffer(%va2) {sym_name = "va2_Iv"} : memref<520xbf16>
    %va2_V  = aie.buffer(%va2) {sym_name = "va2_V"}  : memref<8192xbf16>
    %va2_Of = aie.buffer(%va2) {sym_name = "va2_Of"} : memref<256xbf16>
    %va2_Ip = aie.lock(%va2, 0) {init = 1 : i32, sym_name = "va2_Ip"}
    %va2_Ic = aie.lock(%va2, 1) {init = 0 : i32, sym_name = "va2_Ic"}
    %va2_Vp = aie.lock(%va2, 2) {init = 1 : i32, sym_name = "va2_Vp"}
    %va2_Vc = aie.lock(%va2, 3) {init = 0 : i32, sym_name = "va2_Vc"}
    %va2_Op = aie.lock(%va2, 4) {init = 1 : i32, sym_name = "va2_Op"}
    %va2_Oc = aie.lock(%va2, 5) {init = 0 : i32, sym_name = "va2_Oc"}
    %core_va2 = aie.core(%va2) {
      %c0 = arith.constant 0 : index
      %cN = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      %a4 = arith.constant 4 : i32
      %h64 = arith.constant 64 : i32
      %cnc = arith.constant 2 : index
      %sC = arith.constant 128 : i32
      scf.for %tok = %c0 to %cN step %c1 {
        func.call @flowkv_value_init_bf16(%a4, %h64) : (i32, i32) -> ()
        scf.for %ci = %c0 to %cnc step %c1 {
          aie.use_lock(%va2_Ic, AcquireGreaterEqual, 1)
          aie.use_lock(%va2_Vc, AcquireGreaterEqual, 1)
          func.call @flowkv_value_accum_bf16(%va2_Iv, %va2_V, %a4, %h64, %sC) : (memref<520xbf16>, memref<8192xbf16>, i32, i32, i32) -> ()
          aie.use_lock(%va2_Ip, Release, 1)
          aie.use_lock(%va2_Vp, Release, 1)
        }
        aie.use_lock(%va2_Op, AcquireGreaterEqual, 1)
        func.call @flowkv_value_normalize_bf16(%va2_Of, %a4, %h64) : (memref<256xbf16>, i32, i32) -> ()
        aie.use_lock(%va2_Oc, Release, 1)
      }
      aie.end
    }
    %mem_va2 = aie.mem(%va2) {
      %s0 = aie.dma_start(S2MM, 0, ^iv2, ^vs2)
    ^iv2:
      aie.use_lock(%va2_Ip, AcquireGreaterEqual, 1)
      aie.dma_bd(%va2_Iv : memref<520xbf16>, 0, 520)
      aie.use_lock(%va2_Ic, Release, 1)
      aie.next_bd ^iv2
    ^vs2:
      %s1 = aie.dma_start(S2MM, 1, ^v2, ^om2)
    ^v2:
      aie.use_lock(%va2_Vp, AcquireGreaterEqual, 1)
      aie.dma_bd(%va2_V : memref<8192xbf16>, 0, 8192)
      aie.use_lock(%va2_Vc, Release, 1)
      aie.next_bd ^v2
    ^om2:
      %m0 = aie.dma_start(MM2S, 0, ^oo2, ^e2)
    ^oo2:
      aie.use_lock(%va2_Oc, AcquireGreaterEqual, 1)
      aie.dma_bd(%va2_Of : memref<256xbf16>, 0, 256)
      aie.use_lock(%va2_Op, Release, 1)
      aie.next_bd ^oo2
    ^e2:
      aie.end
    }
    %va3_Iv = aie.buffer(%va3) {sym_name = "va3_Iv"} : memref<520xbf16>
    %va3_V  = aie.buffer(%va3) {sym_name = "va3_V"}  : memref<8192xbf16>
    %va3_Of = aie.buffer(%va3) {sym_name = "va3_Of"} : memref<256xbf16>
    %va3_Ip = aie.lock(%va3, 0) {init = 1 : i32, sym_name = "va3_Ip"}
    %va3_Ic = aie.lock(%va3, 1) {init = 0 : i32, sym_name = "va3_Ic"}
    %va3_Vp = aie.lock(%va3, 2) {init = 1 : i32, sym_name = "va3_Vp"}
    %va3_Vc = aie.lock(%va3, 3) {init = 0 : i32, sym_name = "va3_Vc"}
    %va3_Op = aie.lock(%va3, 4) {init = 1 : i32, sym_name = "va3_Op"}
    %va3_Oc = aie.lock(%va3, 5) {init = 0 : i32, sym_name = "va3_Oc"}
    %core_va3 = aie.core(%va3) {
      %c0 = arith.constant 0 : index
      %cN = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      %a4 = arith.constant 4 : i32
      %h64 = arith.constant 64 : i32
      %cnc = arith.constant 2 : index
      %sC = arith.constant 128 : i32
      scf.for %tok = %c0 to %cN step %c1 {
        func.call @flowkv_value_init_bf16(%a4, %h64) : (i32, i32) -> ()
        scf.for %ci = %c0 to %cnc step %c1 {
          aie.use_lock(%va3_Ic, AcquireGreaterEqual, 1)
          aie.use_lock(%va3_Vc, AcquireGreaterEqual, 1)
          func.call @flowkv_value_accum_bf16(%va3_Iv, %va3_V, %a4, %h64, %sC) : (memref<520xbf16>, memref<8192xbf16>, i32, i32, i32) -> ()
          aie.use_lock(%va3_Ip, Release, 1)
          aie.use_lock(%va3_Vp, Release, 1)
        }
        aie.use_lock(%va3_Op, AcquireGreaterEqual, 1)
        func.call @flowkv_value_normalize_bf16(%va3_Of, %a4, %h64) : (memref<256xbf16>, i32, i32) -> ()
        aie.use_lock(%va3_Oc, Release, 1)
      }
      aie.end
    }
    %mem_va3 = aie.mem(%va3) {
      %s0 = aie.dma_start(S2MM, 0, ^iv3, ^vs3)
    ^iv3:
      aie.use_lock(%va3_Ip, AcquireGreaterEqual, 1)
      aie.dma_bd(%va3_Iv : memref<520xbf16>, 0, 520)
      aie.use_lock(%va3_Ic, Release, 1)
      aie.next_bd ^iv3
    ^vs3:
      %s1 = aie.dma_start(S2MM, 1, ^v3, ^om3)
    ^v3:
      aie.use_lock(%va3_Vp, AcquireGreaterEqual, 1)
      aie.dma_bd(%va3_V : memref<8192xbf16>, 0, 8192)
      aie.use_lock(%va3_Vc, Release, 1)
      aie.next_bd ^v3
    ^om3:
      %m0 = aie.dma_start(MM2S, 0, ^oo3, ^e3)
    ^oo3:
      aie.use_lock(%va3_Oc, AcquireGreaterEqual, 1)
      aie.dma_bd(%va3_Of : memref<256xbf16>, 0, 256)
      aie.use_lock(%va3_Op, Release, 1)
      aie.next_bd ^oo3
    ^e3:
      aie.end
    }
    %va4_Iv = aie.buffer(%va4) {sym_name = "va4_Iv"} : memref<520xbf16>
    %va4_V  = aie.buffer(%va4) {sym_name = "va4_V"}  : memref<8192xbf16>
    %va4_Of = aie.buffer(%va4) {sym_name = "va4_Of"} : memref<256xbf16>
    %va4_Ip = aie.lock(%va4, 0) {init = 1 : i32, sym_name = "va4_Ip"}
    %va4_Ic = aie.lock(%va4, 1) {init = 0 : i32, sym_name = "va4_Ic"}
    %va4_Vp = aie.lock(%va4, 2) {init = 1 : i32, sym_name = "va4_Vp"}
    %va4_Vc = aie.lock(%va4, 3) {init = 0 : i32, sym_name = "va4_Vc"}
    %va4_Op = aie.lock(%va4, 4) {init = 1 : i32, sym_name = "va4_Op"}
    %va4_Oc = aie.lock(%va4, 5) {init = 0 : i32, sym_name = "va4_Oc"}
    %core_va4 = aie.core(%va4) {
      %c0 = arith.constant 0 : index
      %cN = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      %a4 = arith.constant 4 : i32
      %h64 = arith.constant 64 : i32
      %cnc = arith.constant 2 : index
      %sC = arith.constant 128 : i32
      scf.for %tok = %c0 to %cN step %c1 {
        func.call @flowkv_value_init_bf16(%a4, %h64) : (i32, i32) -> ()
        scf.for %ci = %c0 to %cnc step %c1 {
          aie.use_lock(%va4_Ic, AcquireGreaterEqual, 1)
          aie.use_lock(%va4_Vc, AcquireGreaterEqual, 1)
          func.call @flowkv_value_accum_bf16(%va4_Iv, %va4_V, %a4, %h64, %sC) : (memref<520xbf16>, memref<8192xbf16>, i32, i32, i32) -> ()
          aie.use_lock(%va4_Ip, Release, 1)
          aie.use_lock(%va4_Vp, Release, 1)
        }
        aie.use_lock(%va4_Op, AcquireGreaterEqual, 1)
        func.call @flowkv_value_normalize_bf16(%va4_Of, %a4, %h64) : (memref<256xbf16>, i32, i32) -> ()
        aie.use_lock(%va4_Oc, Release, 1)
      }
      aie.end
    }
    %mem_va4 = aie.mem(%va4) {
      %s0 = aie.dma_start(S2MM, 0, ^iv4, ^vs4)
    ^iv4:
      aie.use_lock(%va4_Ip, AcquireGreaterEqual, 1)
      aie.dma_bd(%va4_Iv : memref<520xbf16>, 0, 520)
      aie.use_lock(%va4_Ic, Release, 1)
      aie.next_bd ^iv4
    ^vs4:
      %s1 = aie.dma_start(S2MM, 1, ^v4, ^om4)
    ^v4:
      aie.use_lock(%va4_Vp, AcquireGreaterEqual, 1)
      aie.dma_bd(%va4_V : memref<8192xbf16>, 0, 8192)
      aie.use_lock(%va4_Vc, Release, 1)
      aie.next_bd ^v4
    ^om4:
      %m0 = aie.dma_start(MM2S, 0, ^oo4, ^e4)
    ^oo4:
      aie.use_lock(%va4_Oc, AcquireGreaterEqual, 1)
      aie.dma_bd(%va4_Of : memref<256xbf16>, 0, 256)
      aie.use_lock(%va4_Op, Release, 1)
      aie.next_bd ^oo4
    ^e4:
      aie.end
    }
    %va5_Iv = aie.buffer(%va5) {sym_name = "va5_Iv"} : memref<520xbf16>
    %va5_V  = aie.buffer(%va5) {sym_name = "va5_V"}  : memref<8192xbf16>
    %va5_Of = aie.buffer(%va5) {sym_name = "va5_Of"} : memref<256xbf16>
    %va5_Ip = aie.lock(%va5, 0) {init = 1 : i32, sym_name = "va5_Ip"}
    %va5_Ic = aie.lock(%va5, 1) {init = 0 : i32, sym_name = "va5_Ic"}
    %va5_Vp = aie.lock(%va5, 2) {init = 1 : i32, sym_name = "va5_Vp"}
    %va5_Vc = aie.lock(%va5, 3) {init = 0 : i32, sym_name = "va5_Vc"}
    %va5_Op = aie.lock(%va5, 4) {init = 1 : i32, sym_name = "va5_Op"}
    %va5_Oc = aie.lock(%va5, 5) {init = 0 : i32, sym_name = "va5_Oc"}
    %core_va5 = aie.core(%va5) {
      %c0 = arith.constant 0 : index
      %cN = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      %a4 = arith.constant 4 : i32
      %h64 = arith.constant 64 : i32
      %cnc = arith.constant 2 : index
      %sC = arith.constant 128 : i32
      scf.for %tok = %c0 to %cN step %c1 {
        func.call @flowkv_value_init_bf16(%a4, %h64) : (i32, i32) -> ()
        scf.for %ci = %c0 to %cnc step %c1 {
          aie.use_lock(%va5_Ic, AcquireGreaterEqual, 1)
          aie.use_lock(%va5_Vc, AcquireGreaterEqual, 1)
          func.call @flowkv_value_accum_bf16(%va5_Iv, %va5_V, %a4, %h64, %sC) : (memref<520xbf16>, memref<8192xbf16>, i32, i32, i32) -> ()
          aie.use_lock(%va5_Ip, Release, 1)
          aie.use_lock(%va5_Vp, Release, 1)
        }
        aie.use_lock(%va5_Op, AcquireGreaterEqual, 1)
        func.call @flowkv_value_normalize_bf16(%va5_Of, %a4, %h64) : (memref<256xbf16>, i32, i32) -> ()
        aie.use_lock(%va5_Oc, Release, 1)
      }
      aie.end
    }
    %mem_va5 = aie.mem(%va5) {
      %s0 = aie.dma_start(S2MM, 0, ^iv5, ^vs5)
    ^iv5:
      aie.use_lock(%va5_Ip, AcquireGreaterEqual, 1)
      aie.dma_bd(%va5_Iv : memref<520xbf16>, 0, 520)
      aie.use_lock(%va5_Ic, Release, 1)
      aie.next_bd ^iv5
    ^vs5:
      %s1 = aie.dma_start(S2MM, 1, ^v5, ^om5)
    ^v5:
      aie.use_lock(%va5_Vp, AcquireGreaterEqual, 1)
      aie.dma_bd(%va5_V : memref<8192xbf16>, 0, 8192)
      aie.use_lock(%va5_Vc, Release, 1)
      aie.next_bd ^v5
    ^om5:
      %m0 = aie.dma_start(MM2S, 0, ^oo5, ^e5)
    ^oo5:
      aie.use_lock(%va5_Oc, AcquireGreaterEqual, 1)
      aie.dma_bd(%va5_Of : memref<256xbf16>, 0, 256)
      aie.use_lock(%va5_Op, Release, 1)
      aie.next_bd ^oo5
    ^e5:
      aie.end
    }
    %va6_Iv = aie.buffer(%va6) {sym_name = "va6_Iv"} : memref<520xbf16>
    %va6_V  = aie.buffer(%va6) {sym_name = "va6_V"}  : memref<8192xbf16>
    %va6_Of = aie.buffer(%va6) {sym_name = "va6_Of"} : memref<256xbf16>
    %va6_Ip = aie.lock(%va6, 0) {init = 1 : i32, sym_name = "va6_Ip"}
    %va6_Ic = aie.lock(%va6, 1) {init = 0 : i32, sym_name = "va6_Ic"}
    %va6_Vp = aie.lock(%va6, 2) {init = 1 : i32, sym_name = "va6_Vp"}
    %va6_Vc = aie.lock(%va6, 3) {init = 0 : i32, sym_name = "va6_Vc"}
    %va6_Op = aie.lock(%va6, 4) {init = 1 : i32, sym_name = "va6_Op"}
    %va6_Oc = aie.lock(%va6, 5) {init = 0 : i32, sym_name = "va6_Oc"}
    %core_va6 = aie.core(%va6) {
      %c0 = arith.constant 0 : index
      %cN = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      %a4 = arith.constant 4 : i32
      %h64 = arith.constant 64 : i32
      %cnc = arith.constant 2 : index
      %sC = arith.constant 128 : i32
      scf.for %tok = %c0 to %cN step %c1 {
        func.call @flowkv_value_init_bf16(%a4, %h64) : (i32, i32) -> ()
        scf.for %ci = %c0 to %cnc step %c1 {
          aie.use_lock(%va6_Ic, AcquireGreaterEqual, 1)
          aie.use_lock(%va6_Vc, AcquireGreaterEqual, 1)
          func.call @flowkv_value_accum_bf16(%va6_Iv, %va6_V, %a4, %h64, %sC) : (memref<520xbf16>, memref<8192xbf16>, i32, i32, i32) -> ()
          aie.use_lock(%va6_Ip, Release, 1)
          aie.use_lock(%va6_Vp, Release, 1)
        }
        aie.use_lock(%va6_Op, AcquireGreaterEqual, 1)
        func.call @flowkv_value_normalize_bf16(%va6_Of, %a4, %h64) : (memref<256xbf16>, i32, i32) -> ()
        aie.use_lock(%va6_Oc, Release, 1)
      }
      aie.end
    }
    %mem_va6 = aie.mem(%va6) {
      %s0 = aie.dma_start(S2MM, 0, ^iv6, ^vs6)
    ^iv6:
      aie.use_lock(%va6_Ip, AcquireGreaterEqual, 1)
      aie.dma_bd(%va6_Iv : memref<520xbf16>, 0, 520)
      aie.use_lock(%va6_Ic, Release, 1)
      aie.next_bd ^iv6
    ^vs6:
      %s1 = aie.dma_start(S2MM, 1, ^v6, ^om6)
    ^v6:
      aie.use_lock(%va6_Vp, AcquireGreaterEqual, 1)
      aie.dma_bd(%va6_V : memref<8192xbf16>, 0, 8192)
      aie.use_lock(%va6_Vc, Release, 1)
      aie.next_bd ^v6
    ^om6:
      %m0 = aie.dma_start(MM2S, 0, ^oo6, ^e6)
    ^oo6:
      aie.use_lock(%va6_Oc, AcquireGreaterEqual, 1)
      aie.dma_bd(%va6_Of : memref<256xbf16>, 0, 256)
      aie.use_lock(%va6_Op, Release, 1)
      aie.next_bd ^oo6
    ^e6:
      aie.end
    }
    %va7_Iv = aie.buffer(%va7) {sym_name = "va7_Iv"} : memref<520xbf16>
    %va7_V  = aie.buffer(%va7) {sym_name = "va7_V"}  : memref<8192xbf16>
    %va7_Of = aie.buffer(%va7) {sym_name = "va7_Of"} : memref<256xbf16>
    %va7_Ip = aie.lock(%va7, 0) {init = 1 : i32, sym_name = "va7_Ip"}
    %va7_Ic = aie.lock(%va7, 1) {init = 0 : i32, sym_name = "va7_Ic"}
    %va7_Vp = aie.lock(%va7, 2) {init = 1 : i32, sym_name = "va7_Vp"}
    %va7_Vc = aie.lock(%va7, 3) {init = 0 : i32, sym_name = "va7_Vc"}
    %va7_Op = aie.lock(%va7, 4) {init = 1 : i32, sym_name = "va7_Op"}
    %va7_Oc = aie.lock(%va7, 5) {init = 0 : i32, sym_name = "va7_Oc"}
    %core_va7 = aie.core(%va7) {
      %c0 = arith.constant 0 : index
      %cN = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      %a4 = arith.constant 4 : i32
      %h64 = arith.constant 64 : i32
      %cnc = arith.constant 2 : index
      %sC = arith.constant 128 : i32
      scf.for %tok = %c0 to %cN step %c1 {
        func.call @flowkv_value_init_bf16(%a4, %h64) : (i32, i32) -> ()
        scf.for %ci = %c0 to %cnc step %c1 {
          aie.use_lock(%va7_Ic, AcquireGreaterEqual, 1)
          aie.use_lock(%va7_Vc, AcquireGreaterEqual, 1)
          func.call @flowkv_value_accum_bf16(%va7_Iv, %va7_V, %a4, %h64, %sC) : (memref<520xbf16>, memref<8192xbf16>, i32, i32, i32) -> ()
          aie.use_lock(%va7_Ip, Release, 1)
          aie.use_lock(%va7_Vp, Release, 1)
        }
        aie.use_lock(%va7_Op, AcquireGreaterEqual, 1)
        func.call @flowkv_value_normalize_bf16(%va7_Of, %a4, %h64) : (memref<256xbf16>, i32, i32) -> ()
        aie.use_lock(%va7_Oc, Release, 1)
      }
      aie.end
    }
    %mem_va7 = aie.mem(%va7) {
      %s0 = aie.dma_start(S2MM, 0, ^iv7, ^vs7)
    ^iv7:
      aie.use_lock(%va7_Ip, AcquireGreaterEqual, 1)
      aie.dma_bd(%va7_Iv : memref<520xbf16>, 0, 520)
      aie.use_lock(%va7_Ic, Release, 1)
      aie.next_bd ^iv7
    ^vs7:
      %s1 = aie.dma_start(S2MM, 1, ^v7, ^om7)
    ^v7:
      aie.use_lock(%va7_Vp, AcquireGreaterEqual, 1)
      aie.dma_bd(%va7_V : memref<8192xbf16>, 0, 8192)
      aie.use_lock(%va7_Vc, Release, 1)
      aie.next_bd ^v7
    ^om7:
      %m0 = aie.dma_start(MM2S, 0, ^oo7, ^e7)
    ^oo7:
      aie.use_lock(%va7_Oc, AcquireGreaterEqual, 1)
      aie.dma_bd(%va7_Of : memref<256xbf16>, 0, 256)
      aie.use_lock(%va7_Op, Release, 1)
      aie.next_bd ^oo7
    ^e7:
      aie.end
    }

    %jA_buf = aie.buffer(%jA) {sym_name = "jA_buf"} : memref<2048xbf16>
    %jA_p0 = aie.lock(%jA, 0) {init = 1 : i32, sym_name = "jA_p0"}
    %jA_c0 = aie.lock(%jA, 1) {init = 0 : i32, sym_name = "jA_c0"}
    %jA_p1 = aie.lock(%jA, 2) {init = 1 : i32, sym_name = "jA_p1"}
    %jA_c1 = aie.lock(%jA, 3) {init = 0 : i32, sym_name = "jA_c1"}
    %jA_p2 = aie.lock(%jA, 4) {init = 1 : i32, sym_name = "jA_p2"}
    %jA_c2 = aie.lock(%jA, 5) {init = 0 : i32, sym_name = "jA_c2"}
    %jA_p3 = aie.lock(%jA, 6) {init = 1 : i32, sym_name = "jA_p3"}
    %jA_c3 = aie.lock(%jA, 7) {init = 0 : i32, sym_name = "jA_c3"}
    %jA_dma = aie.memtile_dma(%jA) {
      %s0 = aie.dma_start(S2MM, 0, ^jAg0, ^jAs1)
    ^jAg0:
      aie.use_lock(%jA_p0, AcquireGreaterEqual, 1)
      aie.dma_bd(%jA_buf : memref<2048xbf16>, 0, 256)
      aie.use_lock(%jA_c0, Release, 1)
      aie.next_bd ^jAg0
    ^jAs1:
      %s1 = aie.dma_start(S2MM, 1, ^jAg1, ^jAs2)
    ^jAg1:
      aie.use_lock(%jA_p1, AcquireGreaterEqual, 1)
      aie.dma_bd(%jA_buf : memref<2048xbf16>, 256, 256)
      aie.use_lock(%jA_c1, Release, 1)
      aie.next_bd ^jAg1
    ^jAs2:
      %s2 = aie.dma_start(S2MM, 2, ^jAg2, ^jAs3)
    ^jAg2:
      aie.use_lock(%jA_p2, AcquireGreaterEqual, 1)
      aie.dma_bd(%jA_buf : memref<2048xbf16>, 512, 256)
      aie.use_lock(%jA_c2, Release, 1)
      aie.next_bd ^jAg2
    ^jAs3:
      %s3 = aie.dma_start(S2MM, 3, ^jAg3, ^jAm0)
    ^jAg3:
      aie.use_lock(%jA_p3, AcquireGreaterEqual, 1)
      aie.dma_bd(%jA_buf : memref<2048xbf16>, 768, 256)
      aie.use_lock(%jA_c3, Release, 1)
      aie.next_bd ^jAg3
    ^jAm0:
      %m0 = aie.dma_start(MM2S, 0, ^jAo0, ^jAe)
    ^jAo0:
      aie.use_lock(%jA_c0, AcquireGreaterEqual, 1)
      aie.dma_bd(%jA_buf : memref<2048xbf16>, 0, 256)
      aie.use_lock(%jA_p0, Release, 1)
      aie.next_bd ^jAo1
    ^jAo1:
      aie.use_lock(%jA_c1, AcquireGreaterEqual, 1)
      aie.dma_bd(%jA_buf : memref<2048xbf16>, 256, 256)
      aie.use_lock(%jA_p1, Release, 1)
      aie.next_bd ^jAo2
    ^jAo2:
      aie.use_lock(%jA_c2, AcquireGreaterEqual, 1)
      aie.dma_bd(%jA_buf : memref<2048xbf16>, 512, 256)
      aie.use_lock(%jA_p2, Release, 1)
      aie.next_bd ^jAo3
    ^jAo3:
      aie.use_lock(%jA_c3, AcquireGreaterEqual, 1)
      aie.dma_bd(%jA_buf : memref<2048xbf16>, 768, 256)
      aie.use_lock(%jA_p3, Release, 1)
      aie.next_bd ^jAo0
    ^jAe:
      aie.end
    }
    %jB_buf = aie.buffer(%jB) {sym_name = "jB_buf"} : memref<2048xbf16>
    %jB_p0 = aie.lock(%jB, 0) {init = 1 : i32, sym_name = "jB_p0"}
    %jB_c0 = aie.lock(%jB, 1) {init = 0 : i32, sym_name = "jB_c0"}
    %jB_p1 = aie.lock(%jB, 2) {init = 1 : i32, sym_name = "jB_p1"}
    %jB_c1 = aie.lock(%jB, 3) {init = 0 : i32, sym_name = "jB_c1"}
    %jB_p2 = aie.lock(%jB, 4) {init = 1 : i32, sym_name = "jB_p2"}
    %jB_c2 = aie.lock(%jB, 5) {init = 0 : i32, sym_name = "jB_c2"}
    %jB_p3 = aie.lock(%jB, 6) {init = 1 : i32, sym_name = "jB_p3"}
    %jB_c3 = aie.lock(%jB, 7) {init = 0 : i32, sym_name = "jB_c3"}
    %jB_dma = aie.memtile_dma(%jB) {
      %s0 = aie.dma_start(S2MM, 0, ^jBg0, ^jBs1)
    ^jBg0:
      aie.use_lock(%jB_p0, AcquireGreaterEqual, 1)
      aie.dma_bd(%jB_buf : memref<2048xbf16>, 0, 256)
      aie.use_lock(%jB_c0, Release, 1)
      aie.next_bd ^jBg0
    ^jBs1:
      %s1 = aie.dma_start(S2MM, 1, ^jBg1, ^jBs2)
    ^jBg1:
      aie.use_lock(%jB_p1, AcquireGreaterEqual, 1)
      aie.dma_bd(%jB_buf : memref<2048xbf16>, 256, 256)
      aie.use_lock(%jB_c1, Release, 1)
      aie.next_bd ^jBg1
    ^jBs2:
      %s2 = aie.dma_start(S2MM, 2, ^jBg2, ^jBs3)
    ^jBg2:
      aie.use_lock(%jB_p2, AcquireGreaterEqual, 1)
      aie.dma_bd(%jB_buf : memref<2048xbf16>, 512, 256)
      aie.use_lock(%jB_c2, Release, 1)
      aie.next_bd ^jBg2
    ^jBs3:
      %s3 = aie.dma_start(S2MM, 3, ^jBg3, ^jBm0)
    ^jBg3:
      aie.use_lock(%jB_p3, AcquireGreaterEqual, 1)
      aie.dma_bd(%jB_buf : memref<2048xbf16>, 768, 256)
      aie.use_lock(%jB_c3, Release, 1)
      aie.next_bd ^jBg3
    ^jBm0:
      %m0 = aie.dma_start(MM2S, 0, ^jBo0, ^jBe)
    ^jBo0:
      aie.use_lock(%jB_c0, AcquireGreaterEqual, 1)
      aie.dma_bd(%jB_buf : memref<2048xbf16>, 0, 256)
      aie.use_lock(%jB_p0, Release, 1)
      aie.next_bd ^jBo1
    ^jBo1:
      aie.use_lock(%jB_c1, AcquireGreaterEqual, 1)
      aie.dma_bd(%jB_buf : memref<2048xbf16>, 256, 256)
      aie.use_lock(%jB_p1, Release, 1)
      aie.next_bd ^jBo2
    ^jBo2:
      aie.use_lock(%jB_c2, AcquireGreaterEqual, 1)
      aie.dma_bd(%jB_buf : memref<2048xbf16>, 512, 256)
      aie.use_lock(%jB_p2, Release, 1)
      aie.next_bd ^jBo3
    ^jBo3:
      aie.use_lock(%jB_c3, AcquireGreaterEqual, 1)
      aie.dma_bd(%jB_buf : memref<2048xbf16>, 768, 256)
      aie.use_lock(%jB_p3, Release, 1)
      aie.next_bd ^jBo0
    ^jBe:
      aie.end
    }
    %oJ_buf = aie.buffer(%oJ) {sym_name = "oJ_buf"} : memref<2048xbf16>
    %oJ_p0 = aie.lock(%oJ, 0) {init = 1 : i32, sym_name = "oJ_p0"}
    %oJ_c0 = aie.lock(%oJ, 1) {init = 0 : i32, sym_name = "oJ_c0"}
    %oJ_p1 = aie.lock(%oJ, 2) {init = 1 : i32, sym_name = "oJ_p1"}
    %oJ_c1 = aie.lock(%oJ, 3) {init = 0 : i32, sym_name = "oJ_c1"}
    %oJ_p2 = aie.lock(%oJ, 4) {init = 1 : i32, sym_name = "oJ_p2"}
    %oJ_c2 = aie.lock(%oJ, 5) {init = 0 : i32, sym_name = "oJ_c2"}
    %oJ_p3 = aie.lock(%oJ, 6) {init = 1 : i32, sym_name = "oJ_p3"}
    %oJ_c3 = aie.lock(%oJ, 7) {init = 0 : i32, sym_name = "oJ_c3"}
    %oJ_dma = aie.memtile_dma(%oJ) {
      %s0 = aie.dma_start(S2MM, 0, ^oJg0, ^oJs1)
    ^oJg0:
      aie.use_lock(%oJ_p0, AcquireGreaterEqual, 1)
      aie.dma_bd(%oJ_buf : memref<2048xbf16>, 0, 256)
      aie.use_lock(%oJ_c0, Release, 1)
      aie.next_bd ^oJg0
    ^oJs1:
      %s1 = aie.dma_start(S2MM, 1, ^oJg1, ^oJs2)
    ^oJg1:
      aie.use_lock(%oJ_p1, AcquireGreaterEqual, 1)
      aie.dma_bd(%oJ_buf : memref<2048xbf16>, 256, 256)
      aie.use_lock(%oJ_c1, Release, 1)
      aie.next_bd ^oJg1
    ^oJs2:
      %s2 = aie.dma_start(S2MM, 2, ^oJg2, ^oJs3)
    ^oJg2:
      aie.use_lock(%oJ_p2, AcquireGreaterEqual, 1)
      aie.dma_bd(%oJ_buf : memref<2048xbf16>, 512, 256)
      aie.use_lock(%oJ_c2, Release, 1)
      aie.next_bd ^oJg2
    ^oJs3:
      %s3 = aie.dma_start(S2MM, 3, ^oJg3, ^oJm0)
    ^oJg3:
      aie.use_lock(%oJ_p3, AcquireGreaterEqual, 1)
      aie.dma_bd(%oJ_buf : memref<2048xbf16>, 768, 256)
      aie.use_lock(%oJ_c3, Release, 1)
      aie.next_bd ^oJg3
    ^oJm0:
      %m0 = aie.dma_start(MM2S, 0, ^oJo0, ^oJe)
    ^oJo0:
      aie.use_lock(%oJ_c0, AcquireGreaterEqual, 1)
      aie.dma_bd(%oJ_buf : memref<2048xbf16>, 0, 256)
      aie.use_lock(%oJ_p0, Release, 1)
      aie.next_bd ^oJo1
    ^oJo1:
      aie.use_lock(%oJ_c1, AcquireGreaterEqual, 1)
      aie.dma_bd(%oJ_buf : memref<2048xbf16>, 256, 256)
      aie.use_lock(%oJ_p1, Release, 1)
      aie.next_bd ^oJo2
    ^oJo2:
      aie.use_lock(%oJ_c2, AcquireGreaterEqual, 1)
      aie.dma_bd(%oJ_buf : memref<2048xbf16>, 512, 256)
      aie.use_lock(%oJ_p2, Release, 1)
      aie.next_bd ^oJo3
    ^oJo3:
      aie.use_lock(%oJ_c3, AcquireGreaterEqual, 1)
      aie.dma_bd(%oJ_buf : memref<2048xbf16>, 768, 256)
      aie.use_lock(%oJ_p3, Release, 1)
      aie.next_bd ^oJo0
    ^oJe:
      aie.end
    }
    %oK_buf = aie.buffer(%oK) {sym_name = "oK_buf"} : memref<2048xbf16>
    %oK_p0 = aie.lock(%oK, 0) {init = 1 : i32, sym_name = "oK_p0"}
    %oK_c0 = aie.lock(%oK, 1) {init = 0 : i32, sym_name = "oK_c0"}
    %oK_p1 = aie.lock(%oK, 2) {init = 1 : i32, sym_name = "oK_p1"}
    %oK_c1 = aie.lock(%oK, 3) {init = 0 : i32, sym_name = "oK_c1"}
    %oK_p2 = aie.lock(%oK, 4) {init = 1 : i32, sym_name = "oK_p2"}
    %oK_c2 = aie.lock(%oK, 5) {init = 0 : i32, sym_name = "oK_c2"}
    %oK_p3 = aie.lock(%oK, 6) {init = 1 : i32, sym_name = "oK_p3"}
    %oK_c3 = aie.lock(%oK, 7) {init = 0 : i32, sym_name = "oK_c3"}
    %oK_dma = aie.memtile_dma(%oK) {
      %s0 = aie.dma_start(S2MM, 0, ^oKg0, ^oKs1)
    ^oKg0:
      aie.use_lock(%oK_p0, AcquireGreaterEqual, 1)
      aie.dma_bd(%oK_buf : memref<2048xbf16>, 0, 256)
      aie.use_lock(%oK_c0, Release, 1)
      aie.next_bd ^oKg0
    ^oKs1:
      %s1 = aie.dma_start(S2MM, 1, ^oKg1, ^oKs2)
    ^oKg1:
      aie.use_lock(%oK_p1, AcquireGreaterEqual, 1)
      aie.dma_bd(%oK_buf : memref<2048xbf16>, 256, 256)
      aie.use_lock(%oK_c1, Release, 1)
      aie.next_bd ^oKg1
    ^oKs2:
      %s2 = aie.dma_start(S2MM, 2, ^oKg2, ^oKs3)
    ^oKg2:
      aie.use_lock(%oK_p2, AcquireGreaterEqual, 1)
      aie.dma_bd(%oK_buf : memref<2048xbf16>, 512, 256)
      aie.use_lock(%oK_c2, Release, 1)
      aie.next_bd ^oKg2
    ^oKs3:
      %s3 = aie.dma_start(S2MM, 3, ^oKg3, ^oKm0)
    ^oKg3:
      aie.use_lock(%oK_p3, AcquireGreaterEqual, 1)
      aie.dma_bd(%oK_buf : memref<2048xbf16>, 768, 256)
      aie.use_lock(%oK_c3, Release, 1)
      aie.next_bd ^oKg3
    ^oKm0:
      %m0 = aie.dma_start(MM2S, 0, ^oKo0, ^oKe)
    ^oKo0:
      aie.use_lock(%oK_c0, AcquireGreaterEqual, 1)
      aie.dma_bd(%oK_buf : memref<2048xbf16>, 0, 256)
      aie.use_lock(%oK_p0, Release, 1)
      aie.next_bd ^oKo1
    ^oKo1:
      aie.use_lock(%oK_c1, AcquireGreaterEqual, 1)
      aie.dma_bd(%oK_buf : memref<2048xbf16>, 256, 256)
      aie.use_lock(%oK_p1, Release, 1)
      aie.next_bd ^oKo2
    ^oKo2:
      aie.use_lock(%oK_c2, AcquireGreaterEqual, 1)
      aie.dma_bd(%oK_buf : memref<2048xbf16>, 512, 256)
      aie.use_lock(%oK_p2, Release, 1)
      aie.next_bd ^oKo3
    ^oKo3:
      aie.use_lock(%oK_c3, AcquireGreaterEqual, 1)
      aie.dma_bd(%oK_buf : memref<2048xbf16>, 768, 256)
      aie.use_lock(%oK_p3, Release, 1)
      aie.next_bd ^oKo0
    ^oKe:
      aie.end
    }

    %Klo_buf = aie.buffer(%Klo) {sym_name = "Klo_buf"} : memref<65536xbf16>
    %Klo_p0 = aie.lock(%Klo, 0) {init = 1 : i32, sym_name = "Klo_p0"}
    %Klo_c0 = aie.lock(%Klo, 1) {init = 0 : i32, sym_name = "Klo_c0"}
    %Klo_p1 = aie.lock(%Klo, 2) {init = 1 : i32, sym_name = "Klo_p1"}
    %Klo_c1 = aie.lock(%Klo, 3) {init = 0 : i32, sym_name = "Klo_c1"}
    %Klo_p2 = aie.lock(%Klo, 4) {init = 1 : i32, sym_name = "Klo_p2"}
    %Klo_c2 = aie.lock(%Klo, 5) {init = 0 : i32, sym_name = "Klo_c2"}
    %Klo_p3 = aie.lock(%Klo, 6) {init = 1 : i32, sym_name = "Klo_p3"}
    %Klo_c3 = aie.lock(%Klo, 7) {init = 0 : i32, sym_name = "Klo_c3"}
    %Klo_wt = aie.buffer(%Klo) {sym_name = "Klo_wt"} : memref<9216xi8>
    %Klo_wsp0 = aie.lock(%Klo, 8) {init = 1 : i32, sym_name = "Klo_wsp0"}
    %Klo_wsc0 = aie.lock(%Klo, 9) {init = 0 : i32, sym_name = "Klo_wsc0"}
    %Klo_wsp1 = aie.lock(%Klo, 10) {init = 1 : i32, sym_name = "Klo_wsp1"}
    %Klo_wsc1 = aie.lock(%Klo, 11) {init = 0 : i32, sym_name = "Klo_wsc1"}
    %Klo_dma = aie.memtile_dma(%Klo) {
      %s0 = aie.dma_start(S2MM, 0, ^Klof0, ^Klow0)
    ^Klof0:
      aie.use_lock(%Klo_p0, AcquireGreaterEqual, 1)
      aie.dma_bd(%Klo_buf : memref<65536xbf16>, 0, 16384)
      aie.use_lock(%Klo_c0, Release, 1)
      aie.next_bd ^Klof1
    ^Klof1:
      aie.use_lock(%Klo_p1, AcquireGreaterEqual, 1)
      aie.dma_bd(%Klo_buf : memref<65536xbf16>, 16384, 16384)
      aie.use_lock(%Klo_c1, Release, 1)
      aie.next_bd ^Klof2
    ^Klof2:
      aie.use_lock(%Klo_p2, AcquireGreaterEqual, 1)
      aie.dma_bd(%Klo_buf : memref<65536xbf16>, 32768, 16384)
      aie.use_lock(%Klo_c2, Release, 1)
      aie.next_bd ^Klof3
    ^Klof3:
      aie.use_lock(%Klo_p3, AcquireGreaterEqual, 1)
      aie.dma_bd(%Klo_buf : memref<65536xbf16>, 49152, 16384)
      aie.use_lock(%Klo_c3, Release, 1)
      aie.next_bd ^Klof0
    ^Klow0:
      %w = aie.dma_start(S2MM, 4, ^Klows0, ^Klom0)
    ^Klows0:
      aie.use_lock(%Klo_wsp0, AcquireGreaterEqual, 1)
      aie.dma_bd(%Klo_wt : memref<9216xi8>, 0, 4608)
      aie.use_lock(%Klo_wsc0, Release, 1)
      aie.next_bd ^Klows1
    ^Klows1:
      aie.use_lock(%Klo_wsp1, AcquireGreaterEqual, 1)
      aie.dma_bd(%Klo_wt : memref<9216xi8>, 4608, 4608)
      aie.use_lock(%Klo_wsc1, Release, 1)
      aie.next_bd ^Klows0
    ^Klom0:
      %m0 = aie.dma_start(MM2S, 0, ^Kloo0, ^Klom1)
    ^Kloo0:
      aie.use_lock(%Klo_c0, AcquireGreaterEqual, 1)
      aie.dma_bd(%Klo_buf : memref<65536xbf16>, 0, 16384)
      aie.use_lock(%Klo_p0, Release, 1)
      aie.next_bd ^Kloo0
    ^Klom1:
      %m1 = aie.dma_start(MM2S, 1, ^Kloo1, ^Klom2)
    ^Kloo1:
      aie.use_lock(%Klo_c1, AcquireGreaterEqual, 1)
      aie.dma_bd(%Klo_buf : memref<65536xbf16>, 16384, 16384)
      aie.use_lock(%Klo_p1, Release, 1)
      aie.next_bd ^Kloo1
    ^Klom2:
      %m2 = aie.dma_start(MM2S, 2, ^Kloo2, ^Klom3)
    ^Kloo2:
      aie.use_lock(%Klo_c2, AcquireGreaterEqual, 1)
      aie.dma_bd(%Klo_buf : memref<65536xbf16>, 32768, 16384)
      aie.use_lock(%Klo_p2, Release, 1)
      aie.next_bd ^Kloo2
    ^Klom3:
      %m3 = aie.dma_start(MM2S, 3, ^Kloo3, ^Klowrel)
    ^Kloo3:
      aie.use_lock(%Klo_c3, AcquireGreaterEqual, 1)
      aie.dma_bd(%Klo_buf : memref<65536xbf16>, 49152, 16384)
      aie.use_lock(%Klo_p3, Release, 1)
      aie.next_bd ^Kloo3
    ^Klowrel:
      %x = aie.dma_start(MM2S, 4, ^Kloxs0, ^Kloe)
    ^Kloxs0:
      aie.use_lock(%Klo_wsc0, AcquireGreaterEqual, 1)
      aie.dma_bd(%Klo_wt : memref<9216xi8>, 0, 4608)
      aie.use_lock(%Klo_wsp0, Release, 1)
      aie.next_bd ^Kloxs1
    ^Kloxs1:
      aie.use_lock(%Klo_wsc1, AcquireGreaterEqual, 1)
      aie.dma_bd(%Klo_wt : memref<9216xi8>, 4608, 4608)
      aie.use_lock(%Klo_wsp1, Release, 1)
      aie.next_bd ^Kloxs0
    ^Kloe:
      aie.end
    }
    %Khi_buf = aie.buffer(%Khi) {sym_name = "Khi_buf"} : memref<65536xbf16>
    %Khi_p0 = aie.lock(%Khi, 0) {init = 1 : i32, sym_name = "Khi_p0"}
    %Khi_c0 = aie.lock(%Khi, 1) {init = 0 : i32, sym_name = "Khi_c0"}
    %Khi_p1 = aie.lock(%Khi, 2) {init = 1 : i32, sym_name = "Khi_p1"}
    %Khi_c1 = aie.lock(%Khi, 3) {init = 0 : i32, sym_name = "Khi_c1"}
    %Khi_p2 = aie.lock(%Khi, 4) {init = 1 : i32, sym_name = "Khi_p2"}
    %Khi_c2 = aie.lock(%Khi, 5) {init = 0 : i32, sym_name = "Khi_c2"}
    %Khi_p3 = aie.lock(%Khi, 6) {init = 1 : i32, sym_name = "Khi_p3"}
    %Khi_c3 = aie.lock(%Khi, 7) {init = 0 : i32, sym_name = "Khi_c3"}
    %Khi_wt = aie.buffer(%Khi) {sym_name = "Khi_wt"} : memref<9216xi8>
    %Khi_wsp0 = aie.lock(%Khi, 8) {init = 1 : i32, sym_name = "Khi_wsp0"}
    %Khi_wsc0 = aie.lock(%Khi, 9) {init = 0 : i32, sym_name = "Khi_wsc0"}
    %Khi_wsp1 = aie.lock(%Khi, 10) {init = 1 : i32, sym_name = "Khi_wsp1"}
    %Khi_wsc1 = aie.lock(%Khi, 11) {init = 0 : i32, sym_name = "Khi_wsc1"}
    %Khi_dma = aie.memtile_dma(%Khi) {
      %s0 = aie.dma_start(S2MM, 0, ^Khif0, ^Khiw0)
    ^Khif0:
      aie.use_lock(%Khi_p0, AcquireGreaterEqual, 1)
      aie.dma_bd(%Khi_buf : memref<65536xbf16>, 0, 16384)
      aie.use_lock(%Khi_c0, Release, 1)
      aie.next_bd ^Khif1
    ^Khif1:
      aie.use_lock(%Khi_p1, AcquireGreaterEqual, 1)
      aie.dma_bd(%Khi_buf : memref<65536xbf16>, 16384, 16384)
      aie.use_lock(%Khi_c1, Release, 1)
      aie.next_bd ^Khif2
    ^Khif2:
      aie.use_lock(%Khi_p2, AcquireGreaterEqual, 1)
      aie.dma_bd(%Khi_buf : memref<65536xbf16>, 32768, 16384)
      aie.use_lock(%Khi_c2, Release, 1)
      aie.next_bd ^Khif3
    ^Khif3:
      aie.use_lock(%Khi_p3, AcquireGreaterEqual, 1)
      aie.dma_bd(%Khi_buf : memref<65536xbf16>, 49152, 16384)
      aie.use_lock(%Khi_c3, Release, 1)
      aie.next_bd ^Khif0
    ^Khiw0:
      %w = aie.dma_start(S2MM, 4, ^Khiws0, ^Khim0)
    ^Khiws0:
      aie.use_lock(%Khi_wsp0, AcquireGreaterEqual, 1)
      aie.dma_bd(%Khi_wt : memref<9216xi8>, 0, 4608)
      aie.use_lock(%Khi_wsc0, Release, 1)
      aie.next_bd ^Khiws1
    ^Khiws1:
      aie.use_lock(%Khi_wsp1, AcquireGreaterEqual, 1)
      aie.dma_bd(%Khi_wt : memref<9216xi8>, 4608, 4608)
      aie.use_lock(%Khi_wsc1, Release, 1)
      aie.next_bd ^Khiws0
    ^Khim0:
      %m0 = aie.dma_start(MM2S, 0, ^Khio0, ^Khim1)
    ^Khio0:
      aie.use_lock(%Khi_c0, AcquireGreaterEqual, 1)
      aie.dma_bd(%Khi_buf : memref<65536xbf16>, 0, 16384)
      aie.use_lock(%Khi_p0, Release, 1)
      aie.next_bd ^Khio0
    ^Khim1:
      %m1 = aie.dma_start(MM2S, 1, ^Khio1, ^Khim2)
    ^Khio1:
      aie.use_lock(%Khi_c1, AcquireGreaterEqual, 1)
      aie.dma_bd(%Khi_buf : memref<65536xbf16>, 16384, 16384)
      aie.use_lock(%Khi_p1, Release, 1)
      aie.next_bd ^Khio1
    ^Khim2:
      %m2 = aie.dma_start(MM2S, 2, ^Khio2, ^Khim3)
    ^Khio2:
      aie.use_lock(%Khi_c2, AcquireGreaterEqual, 1)
      aie.dma_bd(%Khi_buf : memref<65536xbf16>, 32768, 16384)
      aie.use_lock(%Khi_p2, Release, 1)
      aie.next_bd ^Khio2
    ^Khim3:
      %m3 = aie.dma_start(MM2S, 3, ^Khio3, ^Khiwrel)
    ^Khio3:
      aie.use_lock(%Khi_c3, AcquireGreaterEqual, 1)
      aie.dma_bd(%Khi_buf : memref<65536xbf16>, 49152, 16384)
      aie.use_lock(%Khi_p3, Release, 1)
      aie.next_bd ^Khio3
    ^Khiwrel:
      %x = aie.dma_start(MM2S, 4, ^Khixs0, ^Khie)
    ^Khixs0:
      aie.use_lock(%Khi_wsc0, AcquireGreaterEqual, 1)
      aie.dma_bd(%Khi_wt : memref<9216xi8>, 0, 4608)
      aie.use_lock(%Khi_wsp0, Release, 1)
      aie.next_bd ^Khixs1
    ^Khixs1:
      aie.use_lock(%Khi_wsc1, AcquireGreaterEqual, 1)
      aie.dma_bd(%Khi_wt : memref<9216xi8>, 4608, 4608)
      aie.use_lock(%Khi_wsp1, Release, 1)
      aie.next_bd ^Khixs0
    ^Khie:
      aie.end
    }
    %Vlo_buf = aie.buffer(%Vlo) {sym_name = "Vlo_buf"} : memref<65536xbf16>
    %Vlo_p0 = aie.lock(%Vlo, 0) {init = 1 : i32, sym_name = "Vlo_p0"}
    %Vlo_c0 = aie.lock(%Vlo, 1) {init = 0 : i32, sym_name = "Vlo_c0"}
    %Vlo_p1 = aie.lock(%Vlo, 2) {init = 1 : i32, sym_name = "Vlo_p1"}
    %Vlo_c1 = aie.lock(%Vlo, 3) {init = 0 : i32, sym_name = "Vlo_c1"}
    %Vlo_p2 = aie.lock(%Vlo, 4) {init = 1 : i32, sym_name = "Vlo_p2"}
    %Vlo_c2 = aie.lock(%Vlo, 5) {init = 0 : i32, sym_name = "Vlo_c2"}
    %Vlo_p3 = aie.lock(%Vlo, 6) {init = 1 : i32, sym_name = "Vlo_p3"}
    %Vlo_c3 = aie.lock(%Vlo, 7) {init = 0 : i32, sym_name = "Vlo_c3"}
    %Vlo_wt = aie.buffer(%Vlo) {sym_name = "Vlo_wt"} : memref<9216xi8>
    %Vlo_wsp0 = aie.lock(%Vlo, 8) {init = 1 : i32, sym_name = "Vlo_wsp0"}
    %Vlo_wsc0 = aie.lock(%Vlo, 9) {init = 0 : i32, sym_name = "Vlo_wsc0"}
    %Vlo_wsp1 = aie.lock(%Vlo, 10) {init = 1 : i32, sym_name = "Vlo_wsp1"}
    %Vlo_wsc1 = aie.lock(%Vlo, 11) {init = 0 : i32, sym_name = "Vlo_wsc1"}
    %Vlo_dma = aie.memtile_dma(%Vlo) {
      %s0 = aie.dma_start(S2MM, 0, ^Vlof0, ^Vlow0)
    ^Vlof0:
      aie.use_lock(%Vlo_p0, AcquireGreaterEqual, 1)
      aie.dma_bd(%Vlo_buf : memref<65536xbf16>, 0, 16384)
      aie.use_lock(%Vlo_c0, Release, 1)
      aie.next_bd ^Vlof1
    ^Vlof1:
      aie.use_lock(%Vlo_p1, AcquireGreaterEqual, 1)
      aie.dma_bd(%Vlo_buf : memref<65536xbf16>, 16384, 16384)
      aie.use_lock(%Vlo_c1, Release, 1)
      aie.next_bd ^Vlof2
    ^Vlof2:
      aie.use_lock(%Vlo_p2, AcquireGreaterEqual, 1)
      aie.dma_bd(%Vlo_buf : memref<65536xbf16>, 32768, 16384)
      aie.use_lock(%Vlo_c2, Release, 1)
      aie.next_bd ^Vlof3
    ^Vlof3:
      aie.use_lock(%Vlo_p3, AcquireGreaterEqual, 1)
      aie.dma_bd(%Vlo_buf : memref<65536xbf16>, 49152, 16384)
      aie.use_lock(%Vlo_c3, Release, 1)
      aie.next_bd ^Vlof0
    ^Vlow0:
      %w = aie.dma_start(S2MM, 4, ^Vlows0, ^Vlom0)
    ^Vlows0:
      aie.use_lock(%Vlo_wsp0, AcquireGreaterEqual, 1)
      aie.dma_bd(%Vlo_wt : memref<9216xi8>, 0, 4608)
      aie.use_lock(%Vlo_wsc0, Release, 1)
      aie.next_bd ^Vlows1
    ^Vlows1:
      aie.use_lock(%Vlo_wsp1, AcquireGreaterEqual, 1)
      aie.dma_bd(%Vlo_wt : memref<9216xi8>, 4608, 4608)
      aie.use_lock(%Vlo_wsc1, Release, 1)
      aie.next_bd ^Vlows0
    ^Vlom0:
      %m0 = aie.dma_start(MM2S, 0, ^Vloo0, ^Vlom1)
    ^Vloo0:
      aie.use_lock(%Vlo_c0, AcquireGreaterEqual, 1)
      aie.dma_bd(%Vlo_buf : memref<65536xbf16>, 0, 16384)
      aie.use_lock(%Vlo_p0, Release, 1)
      aie.next_bd ^Vloo0
    ^Vlom1:
      %m1 = aie.dma_start(MM2S, 1, ^Vloo1, ^Vlom2)
    ^Vloo1:
      aie.use_lock(%Vlo_c1, AcquireGreaterEqual, 1)
      aie.dma_bd(%Vlo_buf : memref<65536xbf16>, 16384, 16384)
      aie.use_lock(%Vlo_p1, Release, 1)
      aie.next_bd ^Vloo1
    ^Vlom2:
      %m2 = aie.dma_start(MM2S, 2, ^Vloo2, ^Vlom3)
    ^Vloo2:
      aie.use_lock(%Vlo_c2, AcquireGreaterEqual, 1)
      aie.dma_bd(%Vlo_buf : memref<65536xbf16>, 32768, 16384)
      aie.use_lock(%Vlo_p2, Release, 1)
      aie.next_bd ^Vloo2
    ^Vlom3:
      %m3 = aie.dma_start(MM2S, 3, ^Vloo3, ^Vlowrel)
    ^Vloo3:
      aie.use_lock(%Vlo_c3, AcquireGreaterEqual, 1)
      aie.dma_bd(%Vlo_buf : memref<65536xbf16>, 49152, 16384)
      aie.use_lock(%Vlo_p3, Release, 1)
      aie.next_bd ^Vloo3
    ^Vlowrel:
      %x = aie.dma_start(MM2S, 4, ^Vloxs0, ^Vloe)
    ^Vloxs0:
      aie.use_lock(%Vlo_wsc0, AcquireGreaterEqual, 1)
      aie.dma_bd(%Vlo_wt : memref<9216xi8>, 0, 4608)
      aie.use_lock(%Vlo_wsp0, Release, 1)
      aie.next_bd ^Vloxs1
    ^Vloxs1:
      aie.use_lock(%Vlo_wsc1, AcquireGreaterEqual, 1)
      aie.dma_bd(%Vlo_wt : memref<9216xi8>, 4608, 4608)
      aie.use_lock(%Vlo_wsp1, Release, 1)
      aie.next_bd ^Vloxs0
    ^Vloe:
      aie.end
    }
    %Vhi_buf = aie.buffer(%Vhi) {sym_name = "Vhi_buf"} : memref<65536xbf16>
    %Vhi_p0 = aie.lock(%Vhi, 0) {init = 1 : i32, sym_name = "Vhi_p0"}
    %Vhi_c0 = aie.lock(%Vhi, 1) {init = 0 : i32, sym_name = "Vhi_c0"}
    %Vhi_p1 = aie.lock(%Vhi, 2) {init = 1 : i32, sym_name = "Vhi_p1"}
    %Vhi_c1 = aie.lock(%Vhi, 3) {init = 0 : i32, sym_name = "Vhi_c1"}
    %Vhi_p2 = aie.lock(%Vhi, 4) {init = 1 : i32, sym_name = "Vhi_p2"}
    %Vhi_c2 = aie.lock(%Vhi, 5) {init = 0 : i32, sym_name = "Vhi_c2"}
    %Vhi_p3 = aie.lock(%Vhi, 6) {init = 1 : i32, sym_name = "Vhi_p3"}
    %Vhi_c3 = aie.lock(%Vhi, 7) {init = 0 : i32, sym_name = "Vhi_c3"}
    %Vhi_wt = aie.buffer(%Vhi) {sym_name = "Vhi_wt"} : memref<9216xi8>
    %Vhi_wsp0 = aie.lock(%Vhi, 8) {init = 1 : i32, sym_name = "Vhi_wsp0"}
    %Vhi_wsc0 = aie.lock(%Vhi, 9) {init = 0 : i32, sym_name = "Vhi_wsc0"}
    %Vhi_wsp1 = aie.lock(%Vhi, 10) {init = 1 : i32, sym_name = "Vhi_wsp1"}
    %Vhi_wsc1 = aie.lock(%Vhi, 11) {init = 0 : i32, sym_name = "Vhi_wsc1"}
    %Vhi_dma = aie.memtile_dma(%Vhi) {
      %s0 = aie.dma_start(S2MM, 0, ^Vhif0, ^Vhiw0)
    ^Vhif0:
      aie.use_lock(%Vhi_p0, AcquireGreaterEqual, 1)
      aie.dma_bd(%Vhi_buf : memref<65536xbf16>, 0, 16384)
      aie.use_lock(%Vhi_c0, Release, 1)
      aie.next_bd ^Vhif1
    ^Vhif1:
      aie.use_lock(%Vhi_p1, AcquireGreaterEqual, 1)
      aie.dma_bd(%Vhi_buf : memref<65536xbf16>, 16384, 16384)
      aie.use_lock(%Vhi_c1, Release, 1)
      aie.next_bd ^Vhif2
    ^Vhif2:
      aie.use_lock(%Vhi_p2, AcquireGreaterEqual, 1)
      aie.dma_bd(%Vhi_buf : memref<65536xbf16>, 32768, 16384)
      aie.use_lock(%Vhi_c2, Release, 1)
      aie.next_bd ^Vhif3
    ^Vhif3:
      aie.use_lock(%Vhi_p3, AcquireGreaterEqual, 1)
      aie.dma_bd(%Vhi_buf : memref<65536xbf16>, 49152, 16384)
      aie.use_lock(%Vhi_c3, Release, 1)
      aie.next_bd ^Vhif0
    ^Vhiw0:
      %w = aie.dma_start(S2MM, 4, ^Vhiws0, ^Vhim0)
    ^Vhiws0:
      aie.use_lock(%Vhi_wsp0, AcquireGreaterEqual, 1)
      aie.dma_bd(%Vhi_wt : memref<9216xi8>, 0, 4608)
      aie.use_lock(%Vhi_wsc0, Release, 1)
      aie.next_bd ^Vhiws1
    ^Vhiws1:
      aie.use_lock(%Vhi_wsp1, AcquireGreaterEqual, 1)
      aie.dma_bd(%Vhi_wt : memref<9216xi8>, 4608, 4608)
      aie.use_lock(%Vhi_wsc1, Release, 1)
      aie.next_bd ^Vhiws0
    ^Vhim0:
      %m0 = aie.dma_start(MM2S, 0, ^Vhio0, ^Vhim1)
    ^Vhio0:
      aie.use_lock(%Vhi_c0, AcquireGreaterEqual, 1)
      aie.dma_bd(%Vhi_buf : memref<65536xbf16>, 0, 16384)
      aie.use_lock(%Vhi_p0, Release, 1)
      aie.next_bd ^Vhio0
    ^Vhim1:
      %m1 = aie.dma_start(MM2S, 1, ^Vhio1, ^Vhim2)
    ^Vhio1:
      aie.use_lock(%Vhi_c1, AcquireGreaterEqual, 1)
      aie.dma_bd(%Vhi_buf : memref<65536xbf16>, 16384, 16384)
      aie.use_lock(%Vhi_p1, Release, 1)
      aie.next_bd ^Vhio1
    ^Vhim2:
      %m2 = aie.dma_start(MM2S, 2, ^Vhio2, ^Vhim3)
    ^Vhio2:
      aie.use_lock(%Vhi_c2, AcquireGreaterEqual, 1)
      aie.dma_bd(%Vhi_buf : memref<65536xbf16>, 32768, 16384)
      aie.use_lock(%Vhi_p2, Release, 1)
      aie.next_bd ^Vhio2
    ^Vhim3:
      %m3 = aie.dma_start(MM2S, 3, ^Vhio3, ^Vhiwrel)
    ^Vhio3:
      aie.use_lock(%Vhi_c3, AcquireGreaterEqual, 1)
      aie.dma_bd(%Vhi_buf : memref<65536xbf16>, 49152, 16384)
      aie.use_lock(%Vhi_p3, Release, 1)
      aie.next_bd ^Vhio3
    ^Vhiwrel:
      %x = aie.dma_start(MM2S, 4, ^Vhixs0, ^Vhie)
    ^Vhixs0:
      aie.use_lock(%Vhi_wsc0, AcquireGreaterEqual, 1)
      aie.dma_bd(%Vhi_wt : memref<9216xi8>, 0, 4608)
      aie.use_lock(%Vhi_wsp0, Release, 1)
      aie.next_bd ^Vhixs1
    ^Vhixs1:
      aie.use_lock(%Vhi_wsc1, AcquireGreaterEqual, 1)
      aie.dma_bd(%Vhi_wt : memref<9216xi8>, 4608, 4608)
      aie.use_lock(%Vhi_wsp1, Release, 1)
      aie.next_bd ^Vhixs0
    ^Vhie:
      aie.end
    }

    %rl_A = aie.buffer(%rl) {sym_name = "rl_A"} : memref<2048xbf16>
    %rl_p0 = aie.lock(%rl, 0) {init = 1 : i32, sym_name = "rl_p0"}
    %rl_c0 = aie.lock(%rl, 1) {init = 0 : i32, sym_name = "rl_c0"}
    %rl_p1 = aie.lock(%rl, 2) {init = 1 : i32, sym_name = "rl_p1"}
    %rl_c1 = aie.lock(%rl, 3) {init = 0 : i32, sym_name = "rl_c1"}
    %rl_op = aie.lock(%rl, 4) {init = 1 : i32, sym_name = "rl_op"}
    %rl_oc = aie.lock(%rl, 5) {init = 0 : i32, sym_name = "rl_oc"}
    %core_rl = aie.core(%rl) {
      %z = arith.constant 0 : index
      %N = arith.constant 9223372036854775807 : index
      %one = arith.constant 1 : index
      scf.for %it = %z to %N step %one {
        aie.use_lock(%rl_c0, AcquireGreaterEqual, 1)
        aie.use_lock(%rl_c1, AcquireGreaterEqual, 1)
        aie.use_lock(%rl_op, AcquireGreaterEqual, 1)
        aie.use_lock(%rl_p0, Release, 1)
        aie.use_lock(%rl_p1, Release, 1)
        aie.use_lock(%rl_oc, Release, 1)
      }
      aie.end
    }
    %mem_rl = aie.mem(%rl) {
      %s0 = aie.dma_start(S2MM, 0, ^rh0, ^rs1)
    ^rh0:
      aie.use_lock(%rl_p0, AcquireGreaterEqual, 1)
      aie.dma_bd(%rl_A : memref<2048xbf16>, 0, 1024)
      aie.use_lock(%rl_c0, Release, 1)
      aie.next_bd ^rh0
    ^rs1:
      %s1 = aie.dma_start(S2MM, 1, ^rh1, ^rm0)
    ^rh1:
      aie.use_lock(%rl_p1, AcquireGreaterEqual, 1)
      aie.dma_bd(%rl_A : memref<2048xbf16>, 1024, 1024)
      aie.use_lock(%rl_c1, Release, 1)
      aie.next_bd ^rh1
    ^rm0:
      %m0 = aie.dma_start(MM2S, 0, ^ro, ^re)
    ^ro:
      aie.use_lock(%rl_oc, AcquireGreaterEqual, 1)
      aie.dma_bd_packet(0, 0)
      aie.dma_bd(%rl_A : memref<2048xbf16>, 0, 2048)
      aie.use_lock(%rl_op, Release, 1)
      aie.next_bd ^ro
    ^re:
      aie.end
    }

    %op_O  = aie.buffer(%op) {sym_name = "op_O"}  : memref<2048xbf16>
    %op_p0 = aie.lock(%op, 0) {init = 1 : i32, sym_name = "op_p0"}
    %op_c0 = aie.lock(%op, 1) {init = 0 : i32, sym_name = "op_c0"}
    %op_p1 = aie.lock(%op, 2) {init = 1 : i32, sym_name = "op_p1"}
    %op_c1 = aie.lock(%op, 3) {init = 0 : i32, sym_name = "op_c1"}
    %op_op = aie.lock(%op, 4) {init = 1 : i32, sym_name = "op_op"}
    %op_oc = aie.lock(%op, 5) {init = 0 : i32, sym_name = "op_oc"}
    %core_op = aie.core(%op) {
      %z = arith.constant 0 : index
      %N = arith.constant 9223372036854775807 : index
      %one = arith.constant 1 : index
      scf.for %it = %z to %N step %one {
        aie.use_lock(%op_c0, AcquireGreaterEqual, 1)
        aie.use_lock(%op_c1, AcquireGreaterEqual, 1)
        aie.use_lock(%op_op, AcquireGreaterEqual, 1)
        aie.use_lock(%op_p0, Release, 1)
        aie.use_lock(%op_p1, Release, 1)
        aie.use_lock(%op_oc, Release, 1)
      }
      aie.end
    }
    %mem_op = aie.mem(%op) {
      %s0 = aie.dma_start(S2MM, 0, ^oh0, ^os1)
    ^oh0:
      aie.use_lock(%op_p0, AcquireGreaterEqual, 1)
      aie.dma_bd(%op_O : memref<2048xbf16>, 0, 1024)
      aie.use_lock(%op_c0, Release, 1)
      aie.next_bd ^oh0
    ^os1:
      %s1 = aie.dma_start(S2MM, 1, ^oh1, ^om0)
    ^oh1:
      aie.use_lock(%op_p1, AcquireGreaterEqual, 1)
      aie.dma_bd(%op_O : memref<2048xbf16>, 1024, 1024)
      aie.use_lock(%op_c1, Release, 1)
      aie.next_bd ^oh1
    ^om0:
      %m0 = aie.dma_start(MM2S, 0, ^oo, ^oe)
    ^oo:
      aie.use_lock(%op_oc, AcquireGreaterEqual, 1)
      aie.dma_bd(%op_O : memref<2048xbf16>, 0, 2048)
      aie.use_lock(%op_op, Release, 1)
      aie.next_bd ^oo
    ^oe:
      aie.end
    }

    %nm_O = aie.buffer(%nm) {sym_name = "nm_O"} : memref<2048xbf16>
    %nm_R = aie.buffer(%nm) {sym_name = "nm_R"} : memref<2048xbf16>
    %nm_F = aie.buffer(%nm) {sym_name = "nm_F"} : memref<2320xbf16>
    %nm_FN = aie.buffer(%nm) {sym_name = "nm_FN"} : memref<2320xbf16>
    %nm_gain = aie.buffer(%nm) {sym_name = "nm_gain"} : memref<2048xbf16>
    %nm_Op = aie.lock(%nm, 0) {init = 1 : i32, sym_name = "nm_Op"}
    %nm_Oc = aie.lock(%nm, 1) {init = 0 : i32, sym_name = "nm_Oc"}
    %nm_Rp = aie.lock(%nm, 2) {init = 1 : i32, sym_name = "nm_Rp"}
    %nm_Rc = aie.lock(%nm, 3) {init = 0 : i32, sym_name = "nm_Rc"}
    %nm_Fp = aie.lock(%nm, 4) {init = 1 : i32, sym_name = "nm_Fp"}
    %nm_Fc = aie.lock(%nm, 5) {init = 0 : i32, sym_name = "nm_Fc"}
    %nm_Gp = aie.lock(%nm, 6) {init = 1 : i32, sym_name = "nm_Gp"}
    %nm_Gc = aie.lock(%nm, 7) {init = 0 : i32, sym_name = "nm_Gc"}
    %nm_FNp = aie.lock(%nm, 8) {init = 1 : i32, sym_name = "nm_FNp"}
    %nm_FNc = aie.lock(%nm, 9) {init = 0 : i32, sym_name = "nm_FNc"}
    %core_nm = aie.core(%nm) {
      %z = arith.constant 0 : index
      %N = arith.constant 9223372036854775807 : index
      %one = arith.constant 1 : index
      %ne = arith.constant 2048 : i32
      scf.for %it = %z to %N step %one {
        aie.use_lock(%nm_Oc, AcquireGreaterEqual, 1)
        aie.use_lock(%nm_Rc, AcquireGreaterEqual, 1)
        aie.use_lock(%nm_Gc, AcquireGreaterEqual, 1)
        aie.use_lock(%nm_Fp, AcquireGreaterEqual, 1)
        aie.use_lock(%nm_FNp, AcquireGreaterEqual, 1)
        func.call @layer_fused_add_bf16(%nm_O, %nm_R, %nm_F, %ne) : (memref<2048xbf16>, memref<2048xbf16>, memref<2320xbf16>, i32) -> ()
        func.call @layer_fused_rms_norm2_bf16(%nm_F, %nm_gain, %nm_FN, %ne) : (memref<2320xbf16>, memref<2048xbf16>, memref<2320xbf16>, i32) -> ()
        aie.use_lock(%nm_Op, Release, 1)
        aie.use_lock(%nm_Rp, Release, 1)
        aie.use_lock(%nm_Gp, Release, 1)
        aie.use_lock(%nm_Fc, Release, 1)
        aie.use_lock(%nm_FNc, Release, 1)
      }
      aie.end
    }
    %mem_nm = aie.mem(%nm) {
      %s0 = aie.dma_start(S2MM, 0, ^no, ^nrs)
    ^no:
      aie.use_lock(%nm_Op, AcquireGreaterEqual, 1)
      aie.dma_bd(%nm_O : memref<2048xbf16>, 0, 2048)
      aie.use_lock(%nm_Oc, Release, 1)
      aie.next_bd ^no
    ^nrs:
      %s1 = aie.dma_start(S2MM, 1, ^nr, ^nm0)
    ^nr:
      aie.use_lock(%nm_Rp, AcquireGreaterEqual, 1)
      aie.dma_bd(%nm_R : memref<2048xbf16>, 0, 2048)
      aie.use_lock(%nm_Rc, Release, 1)
      aie.next_bd ^ng
    ^ng:
      aie.use_lock(%nm_Gp, AcquireGreaterEqual, 1)
      aie.dma_bd(%nm_gain : memref<2048xbf16>, 0, 2048)
      aie.use_lock(%nm_Gc, Release, 1)
      aie.next_bd ^nr
    ^nm0:
      %m0 = aie.dma_start(MM2S, 0, ^nf, ^nm1)
    ^nf:
      aie.use_lock(%nm_FNc, AcquireGreaterEqual, 1)
      aie.dma_bd_packet(0, 1)
      aie.dma_bd(%nm_FN : memref<2320xbf16>, 0, 2048)
      aie.use_lock(%nm_FNp, Release, 1)
      aie.next_bd ^nf
    ^nm1:
      %m1 = aie.dma_start(MM2S, 1, ^ns, ^nme)
    ^ns:
      aie.use_lock(%nm_Fc, AcquireGreaterEqual, 1)
      aie.dma_bd(%nm_F : memref<2320xbf16>, 0, 2048)
      aie.use_lock(%nm_Fp, Release, 1)
      aie.next_bd ^ns
    ^nme:
      aie.end
    }

    %mx_x = aie.buffer(%mx) {sym_name = "mx_x"} : memref<2320xbf16>
    %mx_a = aie.buffer(%mx) {sym_name = "mx_a"} : memref<2320xbf16>
    %mx_f = aie.buffer(%mx) {sym_name = "mx_f"} : memref<2320xbf16>
    %mx_o = aie.buffer(%mx) {sym_name = "mx_o"} : memref<2320xbf16>
    %mx_xp = aie.lock(%mx, 0) {init = 1 : i32, sym_name = "mx_xp"}
    %mx_xc = aie.lock(%mx, 1) {init = 0 : i32, sym_name = "mx_xc"}
    %mx_ap = aie.lock(%mx, 2) {init = 1 : i32, sym_name = "mx_ap"}
    %mx_ac = aie.lock(%mx, 3) {init = 0 : i32, sym_name = "mx_ac"}
    %mx_fp = aie.lock(%mx, 4) {init = 1 : i32, sym_name = "mx_fp"}
    %mx_fc = aie.lock(%mx, 5) {init = 0 : i32, sym_name = "mx_fc"}
    %mx_op = aie.lock(%mx, 6) {init = 1 : i32, sym_name = "mx_op"}
    %mx_oc = aie.lock(%mx, 7) {init = 0 : i32, sym_name = "mx_oc"}
    %core_mx = aie.core(%mx) {
      %z = arith.constant 0 : index
      %N = arith.constant 9223372036854775807 : index
      %one = arith.constant 1 : index
      %nx = arith.constant 2320 : i32
      %ne = arith.constant 2048 : i32
      scf.for %it = %z to %N step %one {
        // phase1 bcast: x (2320)
        aie.use_lock(%mx_xc, AcquireGreaterEqual, 1)
        aie.use_lock(%mx_op, AcquireGreaterEqual, 1)
        func.call @attn_copy_bf16(%mx_x, %mx_o, %nx) : (memref<2320xbf16>, memref<2320xbf16>, i32) -> ()
        aie.use_lock(%mx_xp, Release, 1)
        aie.use_lock(%mx_oc, Release, 1)
        // phase2 bcast: attn_out (2048, tail keeps x LUT)
        aie.use_lock(%mx_ac, AcquireGreaterEqual, 1)
        aie.use_lock(%mx_op, AcquireGreaterEqual, 1)
        func.call @attn_copy_bf16(%mx_a, %mx_o, %ne) : (memref<2320xbf16>, memref<2320xbf16>, i32) -> ()
        aie.use_lock(%mx_ap, Release, 1)
        aie.use_lock(%mx_oc, Release, 1)
        // phase3 bcast: ffn_in (2048)
        aie.use_lock(%mx_fc, AcquireGreaterEqual, 1)
        aie.use_lock(%mx_op, AcquireGreaterEqual, 1)
        func.call @attn_copy_bf16(%mx_f, %mx_o, %ne) : (memref<2320xbf16>, memref<2320xbf16>, i32) -> ()
        aie.use_lock(%mx_fp, Release, 1)
        aie.use_lock(%mx_oc, Release, 1)
      }
      aie.end
    }
    %mem_mx = aie.mem(%mx) {
      %s0 = aie.dma_start(S2MM, 0, ^mxi, ^mxs1)
    ^mxi:
      aie.use_lock(%mx_xp, AcquireGreaterEqual, 1)
      aie.dma_bd(%mx_x : memref<2320xbf16>, 0, 2320)
      aie.use_lock(%mx_xc, Release, 1)
      aie.next_bd ^mxi
    ^mxs1:
      %s1 = aie.dma_start(S2MM, 1, ^mxa, ^mxm0)
    ^mxa:
      aie.use_lock(%mx_ap, AcquireGreaterEqual, 1)
      aie.dma_bd_packet(0, 0)
      aie.dma_bd(%mx_a : memref<2320xbf16>, 0, 2048)
      aie.use_lock(%mx_ac, Release, 1)
      aie.next_bd ^mxf
    ^mxf:
      aie.use_lock(%mx_fp, AcquireGreaterEqual, 1)
      aie.dma_bd_packet(0, 1)
      aie.dma_bd(%mx_f : memref<2320xbf16>, 0, 2048)
      aie.use_lock(%mx_fc, Release, 1)
      aie.next_bd ^mxa
    ^mxm0:
      %m0 = aie.dma_start(MM2S, 0, ^mxo, ^mxe)
    ^mxo:
      aie.use_lock(%mx_oc, AcquireGreaterEqual, 1)
      aie.dma_bd(%mx_o : memref<2320xbf16>, 0, 2320)
      aie.use_lock(%mx_op, Release, 1)
      aie.next_bd ^mxo
    ^mxe:
      aie.end
    }
    aie.shim_dma_allocation @X_alloc(%sh0, MM2S, 1)
    aie.shim_dma_allocation @R_alloc(%sh2, MM2S, 1)
    aie.shim_dma_allocation @Klo_alloc(%sh3, MM2S, 1)
    aie.shim_dma_allocation @Khi_alloc(%sh4, MM2S, 1)
    aie.shim_dma_allocation @Vlo_alloc(%sh5, MM2S, 1)
    aie.shim_dma_allocation @Vhi_alloc(%sh6, MM2S, 1)
    aie.shim_dma_allocation @S_alloc(%sh4, S2MM, 1)
    aie.shim_dma_allocation @A0(%sh0, MM2S, 0)
    aie.shim_dma_allocation @P0(%sh0, S2MM, 0)
    aie.shim_dma_allocation @A1(%sh1, MM2S, 0)
    aie.shim_dma_allocation @P1(%sh1, S2MM, 0)
    aie.shim_dma_allocation @A2(%sh2, MM2S, 0)
    aie.shim_dma_allocation @P2(%sh2, S2MM, 0)
    aie.shim_dma_allocation @A3(%sh3, MM2S, 0)
    aie.shim_dma_allocation @P3(%sh3, S2MM, 0)
    aie.shim_dma_allocation @A4(%sh4, MM2S, 0)
    aie.shim_dma_allocation @P4(%sh4, S2MM, 0)
    aie.shim_dma_allocation @A5(%sh5, MM2S, 0)
    aie.shim_dma_allocation @P5(%sh5, S2MM, 0)
    aie.shim_dma_allocation @A6(%sh6, MM2S, 0)
    aie.shim_dma_allocation @P6(%sh6, S2MM, 0)
    aie.shim_dma_allocation @A7(%sh7, MM2S, 0)
    aie.shim_dma_allocation @P7(%sh7, S2MM, 0)

    aie.runtime_sequence(%arg0: memref<18432xbf16>, %arg1: memref<6416xbf16>, %arg2: memref<33030144xi8>, %arg3: memref<2359296xi8>, %arg4: memref<262144xbf16>) {
      %tx = aiex.dma_configure_task_for @X_alloc {
        aie.dma_bd(%arg1 : memref<6416xbf16>, 0, 2320, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 2320, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%tx)
      %tr = aiex.dma_configure_task_for @R_alloc {
        aie.dma_bd(%arg1 : memref<6416xbf16>, 2320, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 4096, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%tr)
      %tklo = aiex.dma_configure_task_for @Klo_alloc {
        aie.dma_bd(%arg4 : memref<262144xbf16>, 0, 65536, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 65536, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%tklo)
      %tkhi = aiex.dma_configure_task_for @Khi_alloc {
        aie.dma_bd(%arg4 : memref<262144xbf16>, 65536, 65536, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 65536, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%tkhi)
      %tvlo = aiex.dma_configure_task_for @Vlo_alloc {
        aie.dma_bd(%arg4 : memref<262144xbf16>, 131072, 65536, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 65536, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%tvlo)
      %tvhi = aiex.dma_configure_task_for @Vhi_alloc {
        aie.dma_bd(%arg4 : memref<262144xbf16>, 196608, 65536, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 65536, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%tvhi)
      %ta0 = aiex.dma_configure_task_for @A0 {
        aie.dma_bd(%arg2 : memref<33030144xi8>, 0, 4128768, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 4128768, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%ta0)
      %tp0 = aiex.dma_configure_task_for @P0 {
        aie.dma_bd(%arg0 : memref<18432xbf16>, 0, 2048, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 2048, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%tp0)
      %ta1 = aiex.dma_configure_task_for @A1 {
        aie.dma_bd(%arg2 : memref<33030144xi8>, 4128768, 4128768, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 4128768, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%ta1)
      %tp1 = aiex.dma_configure_task_for @P1 {
        aie.dma_bd(%arg0 : memref<18432xbf16>, 2048, 2048, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 2048, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%tp1)
      %ta2 = aiex.dma_configure_task_for @A2 {
        aie.dma_bd(%arg2 : memref<33030144xi8>, 8257536, 4128768, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 4128768, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%ta2)
      %tp2 = aiex.dma_configure_task_for @P2 {
        aie.dma_bd(%arg0 : memref<18432xbf16>, 4096, 2048, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 2048, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%tp2)
      %ta3 = aiex.dma_configure_task_for @A3 {
        aie.dma_bd(%arg2 : memref<33030144xi8>, 12386304, 4128768, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 4128768, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%ta3)
      %tp3 = aiex.dma_configure_task_for @P3 {
        aie.dma_bd(%arg0 : memref<18432xbf16>, 6144, 2048, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 2048, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%tp3)
      %ta4 = aiex.dma_configure_task_for @A4 {
        aie.dma_bd(%arg2 : memref<33030144xi8>, 16515072, 4128768, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 4128768, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%ta4)
      %tp4 = aiex.dma_configure_task_for @P4 {
        aie.dma_bd(%arg0 : memref<18432xbf16>, 8192, 2048, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 2048, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%tp4)
      %ta5 = aiex.dma_configure_task_for @A5 {
        aie.dma_bd(%arg2 : memref<33030144xi8>, 20643840, 4128768, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 4128768, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%ta5)
      %tp5 = aiex.dma_configure_task_for @P5 {
        aie.dma_bd(%arg0 : memref<18432xbf16>, 10240, 2048, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 2048, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%tp5)
      %ta6 = aiex.dma_configure_task_for @A6 {
        aie.dma_bd(%arg2 : memref<33030144xi8>, 24772608, 4128768, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 4128768, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%ta6)
      %tp6 = aiex.dma_configure_task_for @P6 {
        aie.dma_bd(%arg0 : memref<18432xbf16>, 12288, 2048, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 2048, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%tp6)
      %ta7 = aiex.dma_configure_task_for @A7 {
        aie.dma_bd(%arg2 : memref<33030144xi8>, 28901376, 4128768, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 4128768, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%ta7)
      %tp7 = aiex.dma_configure_task_for @P7 {
        aie.dma_bd(%arg0 : memref<18432xbf16>, 14336, 2048, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 2048, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%tp7)
      %ts = aiex.dma_configure_task_for @S_alloc {
        aie.dma_bd(%arg0 : memref<18432xbf16>, 16384, 2048, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 2048, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%ts)
      aiex.dma_await_task(%tp0)
      aiex.dma_await_task(%tp1)
      aiex.dma_await_task(%tp2)
      aiex.dma_await_task(%tp3)
      aiex.dma_await_task(%tp4)
      aiex.dma_await_task(%tp5)
      aiex.dma_await_task(%tp6)
      aiex.dma_await_task(%tp7)
      aiex.dma_await_task(%ts)
    }
  }
}

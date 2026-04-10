module {
  aie.device(npu2) {
    %tile_0_2 = aie.tile(0, 2) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 27>}
    %tile_1_2 = aie.tile(1, 2) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 27>}
    %tile_2_2 = aie.tile(2, 2) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 27>}
    %tile_3_2 = aie.tile(3, 2) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 27>}
    %tile_0_3 = aie.tile(0, 3) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 29>}
    %tile_1_3 = aie.tile(1, 3) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 29>}
    %tile_2_3 = aie.tile(2, 3) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 29>}
    %tile_3_3 = aie.tile(3, 3) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 29>}
    %tile_0_4 = aie.tile(0, 4) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 30>}
    %tile_1_4 = aie.tile(1, 4) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 30>}
    %tile_2_4 = aie.tile(2, 4) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 30>}
    %tile_3_4 = aie.tile(3, 4) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 30>}
    %tile_0_5 = aie.tile(0, 5) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 31>}
    %tile_1_5 = aie.tile(1, 5) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 31>}
    %tile_2_5 = aie.tile(2, 5) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 31>}
    %tile_3_5 = aie.tile(3, 5) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 31>}
    %mem_tile_0_1 = aie.tile(0, 1) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 26>}
    %mem_tile_1_1 = aie.tile(1, 1) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 26>}
    %mem_tile_2_1 = aie.tile(2, 1) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 26>}
    %mem_tile_3_1 = aie.tile(3, 1) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 26>}
    %shim_noc_tile_0_0 = aie.tile(0, 0) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 15>}
    %shim_noc_tile_1_0 = aie.tile(1, 0) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 15>}
    %shim_noc_tile_2_0 = aie.tile(2, 0) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 15>}
    %shim_noc_tile_3_0 = aie.tile(3, 0) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 15>}
    %C_L2L3_3_cons_prod_lock_0 = aie.lock(%shim_noc_tile_3_0, 4) {init = 0 : i32, sym_name = "C_L2L3_3_cons_prod_lock_0"}
    %C_L2L3_3_cons_cons_lock_0 = aie.lock(%shim_noc_tile_3_0, 5) {init = 0 : i32, sym_name = "C_L2L3_3_cons_cons_lock_0"}
    %C_L2L3_3_buff_0 = aie.buffer(%mem_tile_3_1) {address = 0 : i32, mem_bank = 0 : i32, sym_name = "C_L2L3_3_buff_0"} : memref<16384xbf16> 
    %C_L2L3_3_buff_1 = aie.buffer(%mem_tile_3_1) {address = 65536 : i32, mem_bank = 1 : i32, sym_name = "C_L2L3_3_buff_1"} : memref<16384xbf16> 
    %C_L2L3_3_prod_lock_0 = aie.lock(%mem_tile_3_1, 4) {init = 2 : i32, sym_name = "C_L2L3_3_prod_lock_0"}
    %C_L2L3_3_cons_lock_0 = aie.lock(%mem_tile_3_1, 5) {init = 0 : i32, sym_name = "C_L2L3_3_cons_lock_0"}
    %C_L2L3_3_prod_lock_1 = aie.lock(%mem_tile_3_1, 6) {init = 2 : i32, sym_name = "C_L2L3_3_prod_lock_1"}
    %C_L2L3_3_cons_lock_1 = aie.lock(%mem_tile_3_1, 7) {init = 0 : i32, sym_name = "C_L2L3_3_cons_lock_1"}
    %C_L2L3_3_prod_lock_2 = aie.lock(%mem_tile_3_1, 8) {init = 2 : i32, sym_name = "C_L2L3_3_prod_lock_2"}
    %C_L2L3_3_cons_lock_2 = aie.lock(%mem_tile_3_1, 9) {init = 0 : i32, sym_name = "C_L2L3_3_cons_lock_2"}
    %C_L2L3_3_prod_lock_3 = aie.lock(%mem_tile_3_1, 10) {init = 2 : i32, sym_name = "C_L2L3_3_prod_lock_3"}
    %C_L2L3_3_cons_lock_3 = aie.lock(%mem_tile_3_1, 11) {init = 0 : i32, sym_name = "C_L2L3_3_cons_lock_3"}
    %C_L1L2_3_3_buff_0 = aie.buffer(%tile_3_5) {address = 32768 : i32, mem_bank = 2 : i32, sym_name = "C_L1L2_3_3_buff_0"} : memref<64x64xbf16> 
    %C_L1L2_3_3_prod_lock_0 = aie.lock(%tile_3_5, 5) {init = 1 : i32, sym_name = "C_L1L2_3_3_prod_lock_0"}
    %C_L1L2_3_3_cons_lock_0 = aie.lock(%tile_3_5, 6) {init = 0 : i32, sym_name = "C_L1L2_3_3_cons_lock_0"}
    %C_L1L2_3_2_buff_0 = aie.buffer(%tile_3_4) {address = 32768 : i32, mem_bank = 2 : i32, sym_name = "C_L1L2_3_2_buff_0"} : memref<64x64xbf16> 
    %C_L1L2_3_2_prod_lock_0 = aie.lock(%tile_3_4, 5) {init = 1 : i32, sym_name = "C_L1L2_3_2_prod_lock_0"}
    %C_L1L2_3_2_cons_lock_0 = aie.lock(%tile_3_4, 6) {init = 0 : i32, sym_name = "C_L1L2_3_2_cons_lock_0"}
    %C_L1L2_3_1_buff_0 = aie.buffer(%tile_3_3) {address = 32768 : i32, mem_bank = 2 : i32, sym_name = "C_L1L2_3_1_buff_0"} : memref<64x64xbf16> 
    %C_L1L2_3_1_prod_lock_0 = aie.lock(%tile_3_3, 5) {init = 1 : i32, sym_name = "C_L1L2_3_1_prod_lock_0"}
    %C_L1L2_3_1_cons_lock_0 = aie.lock(%tile_3_3, 6) {init = 0 : i32, sym_name = "C_L1L2_3_1_cons_lock_0"}
    %C_L1L2_3_0_buff_0 = aie.buffer(%tile_3_2) {address = 32768 : i32, mem_bank = 2 : i32, sym_name = "C_L1L2_3_0_buff_0"} : memref<64x64xbf16> 
    %C_L1L2_3_0_prod_lock_0 = aie.lock(%tile_3_2, 5) {init = 1 : i32, sym_name = "C_L1L2_3_0_prod_lock_0"}
    %C_L1L2_3_0_cons_lock_0 = aie.lock(%tile_3_2, 6) {init = 0 : i32, sym_name = "C_L1L2_3_0_cons_lock_0"}
    %C_L2L3_2_cons_prod_lock_0 = aie.lock(%shim_noc_tile_2_0, 4) {init = 0 : i32, sym_name = "C_L2L3_2_cons_prod_lock_0"}
    %C_L2L3_2_cons_cons_lock_0 = aie.lock(%shim_noc_tile_2_0, 5) {init = 0 : i32, sym_name = "C_L2L3_2_cons_cons_lock_0"}
    %C_L2L3_2_buff_0 = aie.buffer(%mem_tile_2_1) {address = 0 : i32, mem_bank = 0 : i32, sym_name = "C_L2L3_2_buff_0"} : memref<16384xbf16> 
    %C_L2L3_2_buff_1 = aie.buffer(%mem_tile_2_1) {address = 65536 : i32, mem_bank = 1 : i32, sym_name = "C_L2L3_2_buff_1"} : memref<16384xbf16> 
    %C_L2L3_2_prod_lock_0 = aie.lock(%mem_tile_2_1, 4) {init = 2 : i32, sym_name = "C_L2L3_2_prod_lock_0"}
    %C_L2L3_2_cons_lock_0 = aie.lock(%mem_tile_2_1, 5) {init = 0 : i32, sym_name = "C_L2L3_2_cons_lock_0"}
    %C_L2L3_2_prod_lock_1 = aie.lock(%mem_tile_2_1, 6) {init = 2 : i32, sym_name = "C_L2L3_2_prod_lock_1"}
    %C_L2L3_2_cons_lock_1 = aie.lock(%mem_tile_2_1, 7) {init = 0 : i32, sym_name = "C_L2L3_2_cons_lock_1"}
    %C_L2L3_2_prod_lock_2 = aie.lock(%mem_tile_2_1, 8) {init = 2 : i32, sym_name = "C_L2L3_2_prod_lock_2"}
    %C_L2L3_2_cons_lock_2 = aie.lock(%mem_tile_2_1, 9) {init = 0 : i32, sym_name = "C_L2L3_2_cons_lock_2"}
    %C_L2L3_2_prod_lock_3 = aie.lock(%mem_tile_2_1, 10) {init = 2 : i32, sym_name = "C_L2L3_2_prod_lock_3"}
    %C_L2L3_2_cons_lock_3 = aie.lock(%mem_tile_2_1, 11) {init = 0 : i32, sym_name = "C_L2L3_2_cons_lock_3"}
    %C_L1L2_2_3_buff_0 = aie.buffer(%tile_2_5) {address = 32768 : i32, mem_bank = 2 : i32, sym_name = "C_L1L2_2_3_buff_0"} : memref<64x64xbf16> 
    %C_L1L2_2_3_prod_lock_0 = aie.lock(%tile_2_5, 5) {init = 1 : i32, sym_name = "C_L1L2_2_3_prod_lock_0"}
    %C_L1L2_2_3_cons_lock_0 = aie.lock(%tile_2_5, 6) {init = 0 : i32, sym_name = "C_L1L2_2_3_cons_lock_0"}
    %C_L1L2_2_2_buff_0 = aie.buffer(%tile_2_4) {address = 32768 : i32, mem_bank = 2 : i32, sym_name = "C_L1L2_2_2_buff_0"} : memref<64x64xbf16> 
    %C_L1L2_2_2_prod_lock_0 = aie.lock(%tile_2_4, 5) {init = 1 : i32, sym_name = "C_L1L2_2_2_prod_lock_0"}
    %C_L1L2_2_2_cons_lock_0 = aie.lock(%tile_2_4, 6) {init = 0 : i32, sym_name = "C_L1L2_2_2_cons_lock_0"}
    %C_L1L2_2_1_buff_0 = aie.buffer(%tile_2_3) {address = 32768 : i32, mem_bank = 2 : i32, sym_name = "C_L1L2_2_1_buff_0"} : memref<64x64xbf16> 
    %C_L1L2_2_1_prod_lock_0 = aie.lock(%tile_2_3, 5) {init = 1 : i32, sym_name = "C_L1L2_2_1_prod_lock_0"}
    %C_L1L2_2_1_cons_lock_0 = aie.lock(%tile_2_3, 6) {init = 0 : i32, sym_name = "C_L1L2_2_1_cons_lock_0"}
    %C_L1L2_2_0_buff_0 = aie.buffer(%tile_2_2) {address = 32768 : i32, mem_bank = 2 : i32, sym_name = "C_L1L2_2_0_buff_0"} : memref<64x64xbf16> 
    %C_L1L2_2_0_prod_lock_0 = aie.lock(%tile_2_2, 5) {init = 1 : i32, sym_name = "C_L1L2_2_0_prod_lock_0"}
    %C_L1L2_2_0_cons_lock_0 = aie.lock(%tile_2_2, 6) {init = 0 : i32, sym_name = "C_L1L2_2_0_cons_lock_0"}
    %C_L2L3_1_cons_prod_lock_0 = aie.lock(%shim_noc_tile_1_0, 4) {init = 0 : i32, sym_name = "C_L2L3_1_cons_prod_lock_0"}
    %C_L2L3_1_cons_cons_lock_0 = aie.lock(%shim_noc_tile_1_0, 5) {init = 0 : i32, sym_name = "C_L2L3_1_cons_cons_lock_0"}
    %C_L2L3_1_buff_0 = aie.buffer(%mem_tile_1_1) {address = 0 : i32, mem_bank = 0 : i32, sym_name = "C_L2L3_1_buff_0"} : memref<16384xbf16> 
    %C_L2L3_1_buff_1 = aie.buffer(%mem_tile_1_1) {address = 65536 : i32, mem_bank = 1 : i32, sym_name = "C_L2L3_1_buff_1"} : memref<16384xbf16> 
    %C_L2L3_1_prod_lock_0 = aie.lock(%mem_tile_1_1, 4) {init = 2 : i32, sym_name = "C_L2L3_1_prod_lock_0"}
    %C_L2L3_1_cons_lock_0 = aie.lock(%mem_tile_1_1, 5) {init = 0 : i32, sym_name = "C_L2L3_1_cons_lock_0"}
    %C_L2L3_1_prod_lock_1 = aie.lock(%mem_tile_1_1, 6) {init = 2 : i32, sym_name = "C_L2L3_1_prod_lock_1"}
    %C_L2L3_1_cons_lock_1 = aie.lock(%mem_tile_1_1, 7) {init = 0 : i32, sym_name = "C_L2L3_1_cons_lock_1"}
    %C_L2L3_1_prod_lock_2 = aie.lock(%mem_tile_1_1, 8) {init = 2 : i32, sym_name = "C_L2L3_1_prod_lock_2"}
    %C_L2L3_1_cons_lock_2 = aie.lock(%mem_tile_1_1, 9) {init = 0 : i32, sym_name = "C_L2L3_1_cons_lock_2"}
    %C_L2L3_1_prod_lock_3 = aie.lock(%mem_tile_1_1, 10) {init = 2 : i32, sym_name = "C_L2L3_1_prod_lock_3"}
    %C_L2L3_1_cons_lock_3 = aie.lock(%mem_tile_1_1, 11) {init = 0 : i32, sym_name = "C_L2L3_1_cons_lock_3"}
    %C_L1L2_1_3_buff_0 = aie.buffer(%tile_1_5) {address = 32768 : i32, mem_bank = 2 : i32, sym_name = "C_L1L2_1_3_buff_0"} : memref<64x64xbf16> 
    %C_L1L2_1_3_prod_lock_0 = aie.lock(%tile_1_5, 5) {init = 1 : i32, sym_name = "C_L1L2_1_3_prod_lock_0"}
    %C_L1L2_1_3_cons_lock_0 = aie.lock(%tile_1_5, 6) {init = 0 : i32, sym_name = "C_L1L2_1_3_cons_lock_0"}
    %C_L1L2_1_2_buff_0 = aie.buffer(%tile_1_4) {address = 32768 : i32, mem_bank = 2 : i32, sym_name = "C_L1L2_1_2_buff_0"} : memref<64x64xbf16> 
    %C_L1L2_1_2_prod_lock_0 = aie.lock(%tile_1_4, 5) {init = 1 : i32, sym_name = "C_L1L2_1_2_prod_lock_0"}
    %C_L1L2_1_2_cons_lock_0 = aie.lock(%tile_1_4, 6) {init = 0 : i32, sym_name = "C_L1L2_1_2_cons_lock_0"}
    %C_L1L2_1_1_buff_0 = aie.buffer(%tile_1_3) {address = 32768 : i32, mem_bank = 2 : i32, sym_name = "C_L1L2_1_1_buff_0"} : memref<64x64xbf16> 
    %C_L1L2_1_1_prod_lock_0 = aie.lock(%tile_1_3, 5) {init = 1 : i32, sym_name = "C_L1L2_1_1_prod_lock_0"}
    %C_L1L2_1_1_cons_lock_0 = aie.lock(%tile_1_3, 6) {init = 0 : i32, sym_name = "C_L1L2_1_1_cons_lock_0"}
    %C_L1L2_1_0_buff_0 = aie.buffer(%tile_1_2) {address = 32768 : i32, mem_bank = 2 : i32, sym_name = "C_L1L2_1_0_buff_0"} : memref<64x64xbf16> 
    %C_L1L2_1_0_prod_lock_0 = aie.lock(%tile_1_2, 5) {init = 1 : i32, sym_name = "C_L1L2_1_0_prod_lock_0"}
    %C_L1L2_1_0_cons_lock_0 = aie.lock(%tile_1_2, 6) {init = 0 : i32, sym_name = "C_L1L2_1_0_cons_lock_0"}
    %C_L2L3_0_cons_prod_lock_0 = aie.lock(%shim_noc_tile_0_0, 4) {init = 0 : i32, sym_name = "C_L2L3_0_cons_prod_lock_0"}
    %C_L2L3_0_cons_cons_lock_0 = aie.lock(%shim_noc_tile_0_0, 5) {init = 0 : i32, sym_name = "C_L2L3_0_cons_cons_lock_0"}
    %C_L2L3_0_buff_0 = aie.buffer(%mem_tile_0_1) {address = 0 : i32, mem_bank = 0 : i32, sym_name = "C_L2L3_0_buff_0"} : memref<16384xbf16> 
    %C_L2L3_0_buff_1 = aie.buffer(%mem_tile_0_1) {address = 65536 : i32, mem_bank = 1 : i32, sym_name = "C_L2L3_0_buff_1"} : memref<16384xbf16> 
    %C_L2L3_0_prod_lock_0 = aie.lock(%mem_tile_0_1, 4) {init = 2 : i32, sym_name = "C_L2L3_0_prod_lock_0"}
    %C_L2L3_0_cons_lock_0 = aie.lock(%mem_tile_0_1, 5) {init = 0 : i32, sym_name = "C_L2L3_0_cons_lock_0"}
    %C_L2L3_0_prod_lock_1 = aie.lock(%mem_tile_0_1, 6) {init = 2 : i32, sym_name = "C_L2L3_0_prod_lock_1"}
    %C_L2L3_0_cons_lock_1 = aie.lock(%mem_tile_0_1, 7) {init = 0 : i32, sym_name = "C_L2L3_0_cons_lock_1"}
    %C_L2L3_0_prod_lock_2 = aie.lock(%mem_tile_0_1, 8) {init = 2 : i32, sym_name = "C_L2L3_0_prod_lock_2"}
    %C_L2L3_0_cons_lock_2 = aie.lock(%mem_tile_0_1, 9) {init = 0 : i32, sym_name = "C_L2L3_0_cons_lock_2"}
    %C_L2L3_0_prod_lock_3 = aie.lock(%mem_tile_0_1, 10) {init = 2 : i32, sym_name = "C_L2L3_0_prod_lock_3"}
    %C_L2L3_0_cons_lock_3 = aie.lock(%mem_tile_0_1, 11) {init = 0 : i32, sym_name = "C_L2L3_0_cons_lock_3"}
    %C_L1L2_0_3_buff_0 = aie.buffer(%tile_0_5) {address = 32768 : i32, mem_bank = 2 : i32, sym_name = "C_L1L2_0_3_buff_0"} : memref<64x64xbf16> 
    %C_L1L2_0_3_prod_lock_0 = aie.lock(%tile_0_5, 5) {init = 1 : i32, sym_name = "C_L1L2_0_3_prod_lock_0"}
    %C_L1L2_0_3_cons_lock_0 = aie.lock(%tile_0_5, 6) {init = 0 : i32, sym_name = "C_L1L2_0_3_cons_lock_0"}
    %C_L1L2_0_2_buff_0 = aie.buffer(%tile_0_4) {address = 32768 : i32, mem_bank = 2 : i32, sym_name = "C_L1L2_0_2_buff_0"} : memref<64x64xbf16> 
    %C_L1L2_0_2_prod_lock_0 = aie.lock(%tile_0_4, 5) {init = 1 : i32, sym_name = "C_L1L2_0_2_prod_lock_0"}
    %C_L1L2_0_2_cons_lock_0 = aie.lock(%tile_0_4, 6) {init = 0 : i32, sym_name = "C_L1L2_0_2_cons_lock_0"}
    %C_L1L2_0_1_buff_0 = aie.buffer(%tile_0_3) {address = 32768 : i32, mem_bank = 2 : i32, sym_name = "C_L1L2_0_1_buff_0"} : memref<64x64xbf16> 
    %C_L1L2_0_1_prod_lock_0 = aie.lock(%tile_0_3, 5) {init = 1 : i32, sym_name = "C_L1L2_0_1_prod_lock_0"}
    %C_L1L2_0_1_cons_lock_0 = aie.lock(%tile_0_3, 6) {init = 0 : i32, sym_name = "C_L1L2_0_1_cons_lock_0"}
    %C_L1L2_0_0_buff_0 = aie.buffer(%tile_0_2) {address = 32768 : i32, mem_bank = 2 : i32, sym_name = "C_L1L2_0_0_buff_0"} : memref<64x64xbf16> 
    %C_L1L2_0_0_prod_lock_0 = aie.lock(%tile_0_2, 5) {init = 1 : i32, sym_name = "C_L1L2_0_0_prod_lock_0"}
    %C_L1L2_0_0_cons_lock_0 = aie.lock(%tile_0_2, 6) {init = 0 : i32, sym_name = "C_L1L2_0_0_cons_lock_0"}
    %B_L3L2_3_cons_buff_0 = aie.buffer(%mem_tile_3_1) {address = 131072 : i32, mem_bank = 2 : i32, sym_name = "B_L3L2_3_cons_buff_0"} : memref<4096xbf16> 
    %B_L3L2_3_cons_buff_1 = aie.buffer(%mem_tile_3_1) {address = 196608 : i32, mem_bank = 3 : i32, sym_name = "B_L3L2_3_cons_buff_1"} : memref<4096xbf16> 
    %B_L3L2_3_cons_prod_lock_0 = aie.lock(%mem_tile_3_1, 2) {init = 2 : i32, sym_name = "B_L3L2_3_cons_prod_lock_0"}
    %B_L3L2_3_cons_cons_lock_0 = aie.lock(%mem_tile_3_1, 3) {init = 0 : i32, sym_name = "B_L3L2_3_cons_cons_lock_0"}
    %B_L3L2_3_prod_lock_0 = aie.lock(%shim_noc_tile_3_0, 2) {init = 0 : i32, sym_name = "B_L3L2_3_prod_lock_0"}
    %B_L3L2_3_cons_lock_0 = aie.lock(%shim_noc_tile_3_0, 3) {init = 0 : i32, sym_name = "B_L3L2_3_cons_lock_0"}
    %B_L2L1_3_0_cons_buff_0 = aie.buffer(%tile_3_2) {address = 49152 : i32, mem_bank = 3 : i32, sym_name = "B_L2L1_3_0_cons_buff_0"} : memref<64x64xbf16> 
    %B_L2L1_3_0_cons_buff_1 = aie.buffer(%tile_3_2) {address = 3328 : i32, mem_bank = 0 : i32, sym_name = "B_L2L1_3_0_cons_buff_1"} : memref<64x64xbf16> 
    %B_L2L1_3_0_cons_prod_lock_0 = aie.lock(%tile_3_2, 3) {init = 2 : i32, sym_name = "B_L2L1_3_0_cons_prod_lock_0"}
    %B_L2L1_3_0_cons_cons_lock_0 = aie.lock(%tile_3_2, 4) {init = 0 : i32, sym_name = "B_L2L1_3_0_cons_cons_lock_0"}
    %B_L2L1_3_1_cons_buff_0 = aie.buffer(%tile_3_3) {address = 49152 : i32, mem_bank = 3 : i32, sym_name = "B_L2L1_3_1_cons_buff_0"} : memref<64x64xbf16> 
    %B_L2L1_3_1_cons_buff_1 = aie.buffer(%tile_3_3) {address = 3328 : i32, mem_bank = 0 : i32, sym_name = "B_L2L1_3_1_cons_buff_1"} : memref<64x64xbf16> 
    %B_L2L1_3_1_cons_prod_lock_0 = aie.lock(%tile_3_3, 3) {init = 2 : i32, sym_name = "B_L2L1_3_1_cons_prod_lock_0"}
    %B_L2L1_3_1_cons_cons_lock_0 = aie.lock(%tile_3_3, 4) {init = 0 : i32, sym_name = "B_L2L1_3_1_cons_cons_lock_0"}
    %B_L2L1_3_2_cons_buff_0 = aie.buffer(%tile_3_4) {address = 49152 : i32, mem_bank = 3 : i32, sym_name = "B_L2L1_3_2_cons_buff_0"} : memref<64x64xbf16> 
    %B_L2L1_3_2_cons_buff_1 = aie.buffer(%tile_3_4) {address = 3328 : i32, mem_bank = 0 : i32, sym_name = "B_L2L1_3_2_cons_buff_1"} : memref<64x64xbf16> 
    %B_L2L1_3_2_cons_prod_lock_0 = aie.lock(%tile_3_4, 3) {init = 2 : i32, sym_name = "B_L2L1_3_2_cons_prod_lock_0"}
    %B_L2L1_3_2_cons_cons_lock_0 = aie.lock(%tile_3_4, 4) {init = 0 : i32, sym_name = "B_L2L1_3_2_cons_cons_lock_0"}
    %B_L2L1_3_3_cons_buff_0 = aie.buffer(%tile_3_5) {address = 49152 : i32, mem_bank = 3 : i32, sym_name = "B_L2L1_3_3_cons_buff_0"} : memref<64x64xbf16> 
    %B_L2L1_3_3_cons_buff_1 = aie.buffer(%tile_3_5) {address = 3328 : i32, mem_bank = 0 : i32, sym_name = "B_L2L1_3_3_cons_buff_1"} : memref<64x64xbf16> 
    %B_L2L1_3_3_cons_prod_lock_0 = aie.lock(%tile_3_5, 3) {init = 2 : i32, sym_name = "B_L2L1_3_3_cons_prod_lock_0"}
    %B_L2L1_3_3_cons_cons_lock_0 = aie.lock(%tile_3_5, 4) {init = 0 : i32, sym_name = "B_L2L1_3_3_cons_cons_lock_0"}
    %B_L3L2_2_cons_buff_0 = aie.buffer(%mem_tile_2_1) {address = 131072 : i32, mem_bank = 2 : i32, sym_name = "B_L3L2_2_cons_buff_0"} : memref<4096xbf16> 
    %B_L3L2_2_cons_buff_1 = aie.buffer(%mem_tile_2_1) {address = 196608 : i32, mem_bank = 3 : i32, sym_name = "B_L3L2_2_cons_buff_1"} : memref<4096xbf16> 
    %B_L3L2_2_cons_prod_lock_0 = aie.lock(%mem_tile_2_1, 2) {init = 2 : i32, sym_name = "B_L3L2_2_cons_prod_lock_0"}
    %B_L3L2_2_cons_cons_lock_0 = aie.lock(%mem_tile_2_1, 3) {init = 0 : i32, sym_name = "B_L3L2_2_cons_cons_lock_0"}
    %B_L3L2_2_prod_lock_0 = aie.lock(%shim_noc_tile_2_0, 2) {init = 0 : i32, sym_name = "B_L3L2_2_prod_lock_0"}
    %B_L3L2_2_cons_lock_0 = aie.lock(%shim_noc_tile_2_0, 3) {init = 0 : i32, sym_name = "B_L3L2_2_cons_lock_0"}
    %B_L2L1_2_0_cons_buff_0 = aie.buffer(%tile_2_2) {address = 49152 : i32, mem_bank = 3 : i32, sym_name = "B_L2L1_2_0_cons_buff_0"} : memref<64x64xbf16> 
    %B_L2L1_2_0_cons_buff_1 = aie.buffer(%tile_2_2) {address = 3328 : i32, mem_bank = 0 : i32, sym_name = "B_L2L1_2_0_cons_buff_1"} : memref<64x64xbf16> 
    %B_L2L1_2_0_cons_prod_lock_0 = aie.lock(%tile_2_2, 3) {init = 2 : i32, sym_name = "B_L2L1_2_0_cons_prod_lock_0"}
    %B_L2L1_2_0_cons_cons_lock_0 = aie.lock(%tile_2_2, 4) {init = 0 : i32, sym_name = "B_L2L1_2_0_cons_cons_lock_0"}
    %B_L2L1_2_1_cons_buff_0 = aie.buffer(%tile_2_3) {address = 49152 : i32, mem_bank = 3 : i32, sym_name = "B_L2L1_2_1_cons_buff_0"} : memref<64x64xbf16> 
    %B_L2L1_2_1_cons_buff_1 = aie.buffer(%tile_2_3) {address = 3328 : i32, mem_bank = 0 : i32, sym_name = "B_L2L1_2_1_cons_buff_1"} : memref<64x64xbf16> 
    %B_L2L1_2_1_cons_prod_lock_0 = aie.lock(%tile_2_3, 3) {init = 2 : i32, sym_name = "B_L2L1_2_1_cons_prod_lock_0"}
    %B_L2L1_2_1_cons_cons_lock_0 = aie.lock(%tile_2_3, 4) {init = 0 : i32, sym_name = "B_L2L1_2_1_cons_cons_lock_0"}
    %B_L2L1_2_2_cons_buff_0 = aie.buffer(%tile_2_4) {address = 49152 : i32, mem_bank = 3 : i32, sym_name = "B_L2L1_2_2_cons_buff_0"} : memref<64x64xbf16> 
    %B_L2L1_2_2_cons_buff_1 = aie.buffer(%tile_2_4) {address = 3328 : i32, mem_bank = 0 : i32, sym_name = "B_L2L1_2_2_cons_buff_1"} : memref<64x64xbf16> 
    %B_L2L1_2_2_cons_prod_lock_0 = aie.lock(%tile_2_4, 3) {init = 2 : i32, sym_name = "B_L2L1_2_2_cons_prod_lock_0"}
    %B_L2L1_2_2_cons_cons_lock_0 = aie.lock(%tile_2_4, 4) {init = 0 : i32, sym_name = "B_L2L1_2_2_cons_cons_lock_0"}
    %B_L2L1_2_3_cons_buff_0 = aie.buffer(%tile_2_5) {address = 49152 : i32, mem_bank = 3 : i32, sym_name = "B_L2L1_2_3_cons_buff_0"} : memref<64x64xbf16> 
    %B_L2L1_2_3_cons_buff_1 = aie.buffer(%tile_2_5) {address = 3328 : i32, mem_bank = 0 : i32, sym_name = "B_L2L1_2_3_cons_buff_1"} : memref<64x64xbf16> 
    %B_L2L1_2_3_cons_prod_lock_0 = aie.lock(%tile_2_5, 3) {init = 2 : i32, sym_name = "B_L2L1_2_3_cons_prod_lock_0"}
    %B_L2L1_2_3_cons_cons_lock_0 = aie.lock(%tile_2_5, 4) {init = 0 : i32, sym_name = "B_L2L1_2_3_cons_cons_lock_0"}
    %B_L3L2_1_cons_buff_0 = aie.buffer(%mem_tile_1_1) {address = 131072 : i32, mem_bank = 2 : i32, sym_name = "B_L3L2_1_cons_buff_0"} : memref<4096xbf16> 
    %B_L3L2_1_cons_buff_1 = aie.buffer(%mem_tile_1_1) {address = 196608 : i32, mem_bank = 3 : i32, sym_name = "B_L3L2_1_cons_buff_1"} : memref<4096xbf16> 
    %B_L3L2_1_cons_prod_lock_0 = aie.lock(%mem_tile_1_1, 2) {init = 2 : i32, sym_name = "B_L3L2_1_cons_prod_lock_0"}
    %B_L3L2_1_cons_cons_lock_0 = aie.lock(%mem_tile_1_1, 3) {init = 0 : i32, sym_name = "B_L3L2_1_cons_cons_lock_0"}
    %B_L3L2_1_prod_lock_0 = aie.lock(%shim_noc_tile_1_0, 2) {init = 0 : i32, sym_name = "B_L3L2_1_prod_lock_0"}
    %B_L3L2_1_cons_lock_0 = aie.lock(%shim_noc_tile_1_0, 3) {init = 0 : i32, sym_name = "B_L3L2_1_cons_lock_0"}
    %B_L2L1_1_0_cons_buff_0 = aie.buffer(%tile_1_2) {address = 49152 : i32, mem_bank = 3 : i32, sym_name = "B_L2L1_1_0_cons_buff_0"} : memref<64x64xbf16> 
    %B_L2L1_1_0_cons_buff_1 = aie.buffer(%tile_1_2) {address = 3328 : i32, mem_bank = 0 : i32, sym_name = "B_L2L1_1_0_cons_buff_1"} : memref<64x64xbf16> 
    %B_L2L1_1_0_cons_prod_lock_0 = aie.lock(%tile_1_2, 3) {init = 2 : i32, sym_name = "B_L2L1_1_0_cons_prod_lock_0"}
    %B_L2L1_1_0_cons_cons_lock_0 = aie.lock(%tile_1_2, 4) {init = 0 : i32, sym_name = "B_L2L1_1_0_cons_cons_lock_0"}
    %B_L2L1_1_1_cons_buff_0 = aie.buffer(%tile_1_3) {address = 49152 : i32, mem_bank = 3 : i32, sym_name = "B_L2L1_1_1_cons_buff_0"} : memref<64x64xbf16> 
    %B_L2L1_1_1_cons_buff_1 = aie.buffer(%tile_1_3) {address = 3328 : i32, mem_bank = 0 : i32, sym_name = "B_L2L1_1_1_cons_buff_1"} : memref<64x64xbf16> 
    %B_L2L1_1_1_cons_prod_lock_0 = aie.lock(%tile_1_3, 3) {init = 2 : i32, sym_name = "B_L2L1_1_1_cons_prod_lock_0"}
    %B_L2L1_1_1_cons_cons_lock_0 = aie.lock(%tile_1_3, 4) {init = 0 : i32, sym_name = "B_L2L1_1_1_cons_cons_lock_0"}
    %B_L2L1_1_2_cons_buff_0 = aie.buffer(%tile_1_4) {address = 49152 : i32, mem_bank = 3 : i32, sym_name = "B_L2L1_1_2_cons_buff_0"} : memref<64x64xbf16> 
    %B_L2L1_1_2_cons_buff_1 = aie.buffer(%tile_1_4) {address = 3328 : i32, mem_bank = 0 : i32, sym_name = "B_L2L1_1_2_cons_buff_1"} : memref<64x64xbf16> 
    %B_L2L1_1_2_cons_prod_lock_0 = aie.lock(%tile_1_4, 3) {init = 2 : i32, sym_name = "B_L2L1_1_2_cons_prod_lock_0"}
    %B_L2L1_1_2_cons_cons_lock_0 = aie.lock(%tile_1_4, 4) {init = 0 : i32, sym_name = "B_L2L1_1_2_cons_cons_lock_0"}
    %B_L2L1_1_3_cons_buff_0 = aie.buffer(%tile_1_5) {address = 49152 : i32, mem_bank = 3 : i32, sym_name = "B_L2L1_1_3_cons_buff_0"} : memref<64x64xbf16> 
    %B_L2L1_1_3_cons_buff_1 = aie.buffer(%tile_1_5) {address = 3328 : i32, mem_bank = 0 : i32, sym_name = "B_L2L1_1_3_cons_buff_1"} : memref<64x64xbf16> 
    %B_L2L1_1_3_cons_prod_lock_0 = aie.lock(%tile_1_5, 3) {init = 2 : i32, sym_name = "B_L2L1_1_3_cons_prod_lock_0"}
    %B_L2L1_1_3_cons_cons_lock_0 = aie.lock(%tile_1_5, 4) {init = 0 : i32, sym_name = "B_L2L1_1_3_cons_cons_lock_0"}
    %B_L3L2_0_cons_buff_0 = aie.buffer(%mem_tile_0_1) {address = 131072 : i32, mem_bank = 2 : i32, sym_name = "B_L3L2_0_cons_buff_0"} : memref<4096xbf16> 
    %B_L3L2_0_cons_buff_1 = aie.buffer(%mem_tile_0_1) {address = 196608 : i32, mem_bank = 3 : i32, sym_name = "B_L3L2_0_cons_buff_1"} : memref<4096xbf16> 
    %B_L3L2_0_cons_prod_lock_0 = aie.lock(%mem_tile_0_1, 2) {init = 2 : i32, sym_name = "B_L3L2_0_cons_prod_lock_0"}
    %B_L3L2_0_cons_cons_lock_0 = aie.lock(%mem_tile_0_1, 3) {init = 0 : i32, sym_name = "B_L3L2_0_cons_cons_lock_0"}
    %B_L3L2_0_prod_lock_0 = aie.lock(%shim_noc_tile_0_0, 2) {init = 0 : i32, sym_name = "B_L3L2_0_prod_lock_0"}
    %B_L3L2_0_cons_lock_0 = aie.lock(%shim_noc_tile_0_0, 3) {init = 0 : i32, sym_name = "B_L3L2_0_cons_lock_0"}
    %B_L2L1_0_0_cons_buff_0 = aie.buffer(%tile_0_2) {address = 49152 : i32, mem_bank = 3 : i32, sym_name = "B_L2L1_0_0_cons_buff_0"} : memref<64x64xbf16> 
    %B_L2L1_0_0_cons_buff_1 = aie.buffer(%tile_0_2) {address = 3328 : i32, mem_bank = 0 : i32, sym_name = "B_L2L1_0_0_cons_buff_1"} : memref<64x64xbf16> 
    %B_L2L1_0_0_cons_prod_lock_0 = aie.lock(%tile_0_2, 3) {init = 2 : i32, sym_name = "B_L2L1_0_0_cons_prod_lock_0"}
    %B_L2L1_0_0_cons_cons_lock_0 = aie.lock(%tile_0_2, 4) {init = 0 : i32, sym_name = "B_L2L1_0_0_cons_cons_lock_0"}
    %B_L2L1_0_1_cons_buff_0 = aie.buffer(%tile_0_3) {address = 49152 : i32, mem_bank = 3 : i32, sym_name = "B_L2L1_0_1_cons_buff_0"} : memref<64x64xbf16> 
    %B_L2L1_0_1_cons_buff_1 = aie.buffer(%tile_0_3) {address = 3328 : i32, mem_bank = 0 : i32, sym_name = "B_L2L1_0_1_cons_buff_1"} : memref<64x64xbf16> 
    %B_L2L1_0_1_cons_prod_lock_0 = aie.lock(%tile_0_3, 3) {init = 2 : i32, sym_name = "B_L2L1_0_1_cons_prod_lock_0"}
    %B_L2L1_0_1_cons_cons_lock_0 = aie.lock(%tile_0_3, 4) {init = 0 : i32, sym_name = "B_L2L1_0_1_cons_cons_lock_0"}
    %B_L2L1_0_2_cons_buff_0 = aie.buffer(%tile_0_4) {address = 49152 : i32, mem_bank = 3 : i32, sym_name = "B_L2L1_0_2_cons_buff_0"} : memref<64x64xbf16> 
    %B_L2L1_0_2_cons_buff_1 = aie.buffer(%tile_0_4) {address = 3328 : i32, mem_bank = 0 : i32, sym_name = "B_L2L1_0_2_cons_buff_1"} : memref<64x64xbf16> 
    %B_L2L1_0_2_cons_prod_lock_0 = aie.lock(%tile_0_4, 3) {init = 2 : i32, sym_name = "B_L2L1_0_2_cons_prod_lock_0"}
    %B_L2L1_0_2_cons_cons_lock_0 = aie.lock(%tile_0_4, 4) {init = 0 : i32, sym_name = "B_L2L1_0_2_cons_cons_lock_0"}
    %B_L2L1_0_3_cons_buff_0 = aie.buffer(%tile_0_5) {address = 49152 : i32, mem_bank = 3 : i32, sym_name = "B_L2L1_0_3_cons_buff_0"} : memref<64x64xbf16> 
    %B_L2L1_0_3_cons_buff_1 = aie.buffer(%tile_0_5) {address = 3328 : i32, mem_bank = 0 : i32, sym_name = "B_L2L1_0_3_cons_buff_1"} : memref<64x64xbf16> 
    %B_L2L1_0_3_cons_prod_lock_0 = aie.lock(%tile_0_5, 3) {init = 2 : i32, sym_name = "B_L2L1_0_3_cons_prod_lock_0"}
    %B_L2L1_0_3_cons_cons_lock_0 = aie.lock(%tile_0_5, 4) {init = 0 : i32, sym_name = "B_L2L1_0_3_cons_cons_lock_0"}
    %A_L3L2_3_cons_buff_0 = aie.buffer(%mem_tile_3_1) {address = 262144 : i32, mem_bank = 4 : i32, sym_name = "A_L3L2_3_cons_buff_0"} : memref<4096xbf16> 
    %A_L3L2_3_cons_buff_1 = aie.buffer(%mem_tile_3_1) {address = 327680 : i32, mem_bank = 5 : i32, sym_name = "A_L3L2_3_cons_buff_1"} : memref<4096xbf16> 
    %A_L3L2_3_cons_prod_lock_0 = aie.lock(%mem_tile_3_1, 0) {init = 2 : i32, sym_name = "A_L3L2_3_cons_prod_lock_0"}
    %A_L3L2_3_cons_cons_lock_0 = aie.lock(%mem_tile_3_1, 1) {init = 0 : i32, sym_name = "A_L3L2_3_cons_cons_lock_0"}
    %A_L3L2_3_prod_lock_0 = aie.lock(%shim_noc_tile_3_0, 0) {init = 0 : i32, sym_name = "A_L3L2_3_prod_lock_0"}
    %A_L3L2_3_cons_lock_0 = aie.lock(%shim_noc_tile_3_0, 1) {init = 0 : i32, sym_name = "A_L3L2_3_cons_lock_0"}
    %A_L2L1_3_0_cons_buff_0 = aie.buffer(%tile_0_5) {address = 40960 : i32, mem_bank = 2 : i32, sym_name = "A_L2L1_3_0_cons_buff_0"} : memref<64x64xbf16> 
    %A_L2L1_3_0_cons_buff_1 = aie.buffer(%tile_0_5) {address = 57344 : i32, mem_bank = 3 : i32, sym_name = "A_L2L1_3_0_cons_buff_1"} : memref<64x64xbf16> 
    %A_L2L1_3_0_cons_prod_lock_0 = aie.lock(%tile_0_5, 1) {init = 2 : i32, sym_name = "A_L2L1_3_0_cons_prod_lock_0"}
    %A_L2L1_3_0_cons_cons_lock_0 = aie.lock(%tile_0_5, 2) {init = 0 : i32, sym_name = "A_L2L1_3_0_cons_cons_lock_0"}
    %A_L2L1_3_1_cons_buff_0 = aie.buffer(%tile_1_5) {address = 40960 : i32, mem_bank = 2 : i32, sym_name = "A_L2L1_3_1_cons_buff_0"} : memref<64x64xbf16> 
    %A_L2L1_3_1_cons_buff_1 = aie.buffer(%tile_1_5) {address = 57344 : i32, mem_bank = 3 : i32, sym_name = "A_L2L1_3_1_cons_buff_1"} : memref<64x64xbf16> 
    %A_L2L1_3_1_cons_prod_lock_0 = aie.lock(%tile_1_5, 1) {init = 2 : i32, sym_name = "A_L2L1_3_1_cons_prod_lock_0"}
    %A_L2L1_3_1_cons_cons_lock_0 = aie.lock(%tile_1_5, 2) {init = 0 : i32, sym_name = "A_L2L1_3_1_cons_cons_lock_0"}
    %A_L2L1_3_2_cons_buff_0 = aie.buffer(%tile_2_5) {address = 40960 : i32, mem_bank = 2 : i32, sym_name = "A_L2L1_3_2_cons_buff_0"} : memref<64x64xbf16> 
    %A_L2L1_3_2_cons_buff_1 = aie.buffer(%tile_2_5) {address = 57344 : i32, mem_bank = 3 : i32, sym_name = "A_L2L1_3_2_cons_buff_1"} : memref<64x64xbf16> 
    %A_L2L1_3_2_cons_prod_lock_0 = aie.lock(%tile_2_5, 1) {init = 2 : i32, sym_name = "A_L2L1_3_2_cons_prod_lock_0"}
    %A_L2L1_3_2_cons_cons_lock_0 = aie.lock(%tile_2_5, 2) {init = 0 : i32, sym_name = "A_L2L1_3_2_cons_cons_lock_0"}
    %A_L2L1_3_3_cons_buff_0 = aie.buffer(%tile_3_5) {address = 40960 : i32, mem_bank = 2 : i32, sym_name = "A_L2L1_3_3_cons_buff_0"} : memref<64x64xbf16> 
    %A_L2L1_3_3_cons_buff_1 = aie.buffer(%tile_3_5) {address = 57344 : i32, mem_bank = 3 : i32, sym_name = "A_L2L1_3_3_cons_buff_1"} : memref<64x64xbf16> 
    %A_L2L1_3_3_cons_prod_lock_0 = aie.lock(%tile_3_5, 1) {init = 2 : i32, sym_name = "A_L2L1_3_3_cons_prod_lock_0"}
    %A_L2L1_3_3_cons_cons_lock_0 = aie.lock(%tile_3_5, 2) {init = 0 : i32, sym_name = "A_L2L1_3_3_cons_cons_lock_0"}
    %A_L3L2_2_cons_buff_0 = aie.buffer(%mem_tile_2_1) {address = 262144 : i32, mem_bank = 4 : i32, sym_name = "A_L3L2_2_cons_buff_0"} : memref<4096xbf16> 
    %A_L3L2_2_cons_buff_1 = aie.buffer(%mem_tile_2_1) {address = 327680 : i32, mem_bank = 5 : i32, sym_name = "A_L3L2_2_cons_buff_1"} : memref<4096xbf16> 
    %A_L3L2_2_cons_prod_lock_0 = aie.lock(%mem_tile_2_1, 0) {init = 2 : i32, sym_name = "A_L3L2_2_cons_prod_lock_0"}
    %A_L3L2_2_cons_cons_lock_0 = aie.lock(%mem_tile_2_1, 1) {init = 0 : i32, sym_name = "A_L3L2_2_cons_cons_lock_0"}
    %A_L3L2_2_prod_lock_0 = aie.lock(%shim_noc_tile_2_0, 0) {init = 0 : i32, sym_name = "A_L3L2_2_prod_lock_0"}
    %A_L3L2_2_cons_lock_0 = aie.lock(%shim_noc_tile_2_0, 1) {init = 0 : i32, sym_name = "A_L3L2_2_cons_lock_0"}
    %A_L2L1_2_0_cons_buff_0 = aie.buffer(%tile_0_4) {address = 40960 : i32, mem_bank = 2 : i32, sym_name = "A_L2L1_2_0_cons_buff_0"} : memref<64x64xbf16> 
    %A_L2L1_2_0_cons_buff_1 = aie.buffer(%tile_0_4) {address = 57344 : i32, mem_bank = 3 : i32, sym_name = "A_L2L1_2_0_cons_buff_1"} : memref<64x64xbf16> 
    %A_L2L1_2_0_cons_prod_lock_0 = aie.lock(%tile_0_4, 1) {init = 2 : i32, sym_name = "A_L2L1_2_0_cons_prod_lock_0"}
    %A_L2L1_2_0_cons_cons_lock_0 = aie.lock(%tile_0_4, 2) {init = 0 : i32, sym_name = "A_L2L1_2_0_cons_cons_lock_0"}
    %A_L2L1_2_1_cons_buff_0 = aie.buffer(%tile_1_4) {address = 40960 : i32, mem_bank = 2 : i32, sym_name = "A_L2L1_2_1_cons_buff_0"} : memref<64x64xbf16> 
    %A_L2L1_2_1_cons_buff_1 = aie.buffer(%tile_1_4) {address = 57344 : i32, mem_bank = 3 : i32, sym_name = "A_L2L1_2_1_cons_buff_1"} : memref<64x64xbf16> 
    %A_L2L1_2_1_cons_prod_lock_0 = aie.lock(%tile_1_4, 1) {init = 2 : i32, sym_name = "A_L2L1_2_1_cons_prod_lock_0"}
    %A_L2L1_2_1_cons_cons_lock_0 = aie.lock(%tile_1_4, 2) {init = 0 : i32, sym_name = "A_L2L1_2_1_cons_cons_lock_0"}
    %A_L2L1_2_2_cons_buff_0 = aie.buffer(%tile_2_4) {address = 40960 : i32, mem_bank = 2 : i32, sym_name = "A_L2L1_2_2_cons_buff_0"} : memref<64x64xbf16> 
    %A_L2L1_2_2_cons_buff_1 = aie.buffer(%tile_2_4) {address = 57344 : i32, mem_bank = 3 : i32, sym_name = "A_L2L1_2_2_cons_buff_1"} : memref<64x64xbf16> 
    %A_L2L1_2_2_cons_prod_lock_0 = aie.lock(%tile_2_4, 1) {init = 2 : i32, sym_name = "A_L2L1_2_2_cons_prod_lock_0"}
    %A_L2L1_2_2_cons_cons_lock_0 = aie.lock(%tile_2_4, 2) {init = 0 : i32, sym_name = "A_L2L1_2_2_cons_cons_lock_0"}
    %A_L2L1_2_3_cons_buff_0 = aie.buffer(%tile_3_4) {address = 40960 : i32, mem_bank = 2 : i32, sym_name = "A_L2L1_2_3_cons_buff_0"} : memref<64x64xbf16> 
    %A_L2L1_2_3_cons_buff_1 = aie.buffer(%tile_3_4) {address = 57344 : i32, mem_bank = 3 : i32, sym_name = "A_L2L1_2_3_cons_buff_1"} : memref<64x64xbf16> 
    %A_L2L1_2_3_cons_prod_lock_0 = aie.lock(%tile_3_4, 1) {init = 2 : i32, sym_name = "A_L2L1_2_3_cons_prod_lock_0"}
    %A_L2L1_2_3_cons_cons_lock_0 = aie.lock(%tile_3_4, 2) {init = 0 : i32, sym_name = "A_L2L1_2_3_cons_cons_lock_0"}
    %A_L3L2_1_cons_buff_0 = aie.buffer(%mem_tile_1_1) {address = 262144 : i32, mem_bank = 4 : i32, sym_name = "A_L3L2_1_cons_buff_0"} : memref<4096xbf16> 
    %A_L3L2_1_cons_buff_1 = aie.buffer(%mem_tile_1_1) {address = 327680 : i32, mem_bank = 5 : i32, sym_name = "A_L3L2_1_cons_buff_1"} : memref<4096xbf16> 
    %A_L3L2_1_cons_prod_lock_0 = aie.lock(%mem_tile_1_1, 0) {init = 2 : i32, sym_name = "A_L3L2_1_cons_prod_lock_0"}
    %A_L3L2_1_cons_cons_lock_0 = aie.lock(%mem_tile_1_1, 1) {init = 0 : i32, sym_name = "A_L3L2_1_cons_cons_lock_0"}
    %A_L3L2_1_prod_lock_0 = aie.lock(%shim_noc_tile_1_0, 0) {init = 0 : i32, sym_name = "A_L3L2_1_prod_lock_0"}
    %A_L3L2_1_cons_lock_0 = aie.lock(%shim_noc_tile_1_0, 1) {init = 0 : i32, sym_name = "A_L3L2_1_cons_lock_0"}
    %A_L2L1_1_0_cons_buff_0 = aie.buffer(%tile_0_3) {address = 40960 : i32, mem_bank = 2 : i32, sym_name = "A_L2L1_1_0_cons_buff_0"} : memref<64x64xbf16> 
    %A_L2L1_1_0_cons_buff_1 = aie.buffer(%tile_0_3) {address = 57344 : i32, mem_bank = 3 : i32, sym_name = "A_L2L1_1_0_cons_buff_1"} : memref<64x64xbf16> 
    %A_L2L1_1_0_cons_prod_lock_0 = aie.lock(%tile_0_3, 1) {init = 2 : i32, sym_name = "A_L2L1_1_0_cons_prod_lock_0"}
    %A_L2L1_1_0_cons_cons_lock_0 = aie.lock(%tile_0_3, 2) {init = 0 : i32, sym_name = "A_L2L1_1_0_cons_cons_lock_0"}
    %A_L2L1_1_1_cons_buff_0 = aie.buffer(%tile_1_3) {address = 40960 : i32, mem_bank = 2 : i32, sym_name = "A_L2L1_1_1_cons_buff_0"} : memref<64x64xbf16> 
    %A_L2L1_1_1_cons_buff_1 = aie.buffer(%tile_1_3) {address = 57344 : i32, mem_bank = 3 : i32, sym_name = "A_L2L1_1_1_cons_buff_1"} : memref<64x64xbf16> 
    %A_L2L1_1_1_cons_prod_lock_0 = aie.lock(%tile_1_3, 1) {init = 2 : i32, sym_name = "A_L2L1_1_1_cons_prod_lock_0"}
    %A_L2L1_1_1_cons_cons_lock_0 = aie.lock(%tile_1_3, 2) {init = 0 : i32, sym_name = "A_L2L1_1_1_cons_cons_lock_0"}
    %A_L2L1_1_2_cons_buff_0 = aie.buffer(%tile_2_3) {address = 40960 : i32, mem_bank = 2 : i32, sym_name = "A_L2L1_1_2_cons_buff_0"} : memref<64x64xbf16> 
    %A_L2L1_1_2_cons_buff_1 = aie.buffer(%tile_2_3) {address = 57344 : i32, mem_bank = 3 : i32, sym_name = "A_L2L1_1_2_cons_buff_1"} : memref<64x64xbf16> 
    %A_L2L1_1_2_cons_prod_lock_0 = aie.lock(%tile_2_3, 1) {init = 2 : i32, sym_name = "A_L2L1_1_2_cons_prod_lock_0"}
    %A_L2L1_1_2_cons_cons_lock_0 = aie.lock(%tile_2_3, 2) {init = 0 : i32, sym_name = "A_L2L1_1_2_cons_cons_lock_0"}
    %A_L2L1_1_3_cons_buff_0 = aie.buffer(%tile_3_3) {address = 40960 : i32, mem_bank = 2 : i32, sym_name = "A_L2L1_1_3_cons_buff_0"} : memref<64x64xbf16> 
    %A_L2L1_1_3_cons_buff_1 = aie.buffer(%tile_3_3) {address = 57344 : i32, mem_bank = 3 : i32, sym_name = "A_L2L1_1_3_cons_buff_1"} : memref<64x64xbf16> 
    %A_L2L1_1_3_cons_prod_lock_0 = aie.lock(%tile_3_3, 1) {init = 2 : i32, sym_name = "A_L2L1_1_3_cons_prod_lock_0"}
    %A_L2L1_1_3_cons_cons_lock_0 = aie.lock(%tile_3_3, 2) {init = 0 : i32, sym_name = "A_L2L1_1_3_cons_cons_lock_0"}
    %A_L3L2_0_cons_buff_0 = aie.buffer(%mem_tile_0_1) {address = 262144 : i32, mem_bank = 4 : i32, sym_name = "A_L3L2_0_cons_buff_0"} : memref<4096xbf16> 
    %A_L3L2_0_cons_buff_1 = aie.buffer(%mem_tile_0_1) {address = 327680 : i32, mem_bank = 5 : i32, sym_name = "A_L3L2_0_cons_buff_1"} : memref<4096xbf16> 
    %A_L3L2_0_cons_prod_lock_0 = aie.lock(%mem_tile_0_1, 0) {init = 2 : i32, sym_name = "A_L3L2_0_cons_prod_lock_0"}
    %A_L3L2_0_cons_cons_lock_0 = aie.lock(%mem_tile_0_1, 1) {init = 0 : i32, sym_name = "A_L3L2_0_cons_cons_lock_0"}
    %A_L3L2_0_prod_lock_0 = aie.lock(%shim_noc_tile_0_0, 0) {init = 0 : i32, sym_name = "A_L3L2_0_prod_lock_0"}
    %A_L3L2_0_cons_lock_0 = aie.lock(%shim_noc_tile_0_0, 1) {init = 0 : i32, sym_name = "A_L3L2_0_cons_lock_0"}
    %A_L2L1_0_0_cons_buff_0 = aie.buffer(%tile_0_2) {address = 40960 : i32, mem_bank = 2 : i32, sym_name = "A_L2L1_0_0_cons_buff_0"} : memref<64x64xbf16> 
    %A_L2L1_0_0_cons_buff_1 = aie.buffer(%tile_0_2) {address = 57344 : i32, mem_bank = 3 : i32, sym_name = "A_L2L1_0_0_cons_buff_1"} : memref<64x64xbf16> 
    %A_L2L1_0_0_cons_prod_lock_0 = aie.lock(%tile_0_2, 1) {init = 2 : i32, sym_name = "A_L2L1_0_0_cons_prod_lock_0"}
    %A_L2L1_0_0_cons_cons_lock_0 = aie.lock(%tile_0_2, 2) {init = 0 : i32, sym_name = "A_L2L1_0_0_cons_cons_lock_0"}
    %A_L2L1_0_1_cons_buff_0 = aie.buffer(%tile_1_2) {address = 40960 : i32, mem_bank = 2 : i32, sym_name = "A_L2L1_0_1_cons_buff_0"} : memref<64x64xbf16> 
    %A_L2L1_0_1_cons_buff_1 = aie.buffer(%tile_1_2) {address = 57344 : i32, mem_bank = 3 : i32, sym_name = "A_L2L1_0_1_cons_buff_1"} : memref<64x64xbf16> 
    %A_L2L1_0_1_cons_prod_lock_0 = aie.lock(%tile_1_2, 1) {init = 2 : i32, sym_name = "A_L2L1_0_1_cons_prod_lock_0"}
    %A_L2L1_0_1_cons_cons_lock_0 = aie.lock(%tile_1_2, 2) {init = 0 : i32, sym_name = "A_L2L1_0_1_cons_cons_lock_0"}
    %A_L2L1_0_2_cons_buff_0 = aie.buffer(%tile_2_2) {address = 40960 : i32, mem_bank = 2 : i32, sym_name = "A_L2L1_0_2_cons_buff_0"} : memref<64x64xbf16> 
    %A_L2L1_0_2_cons_buff_1 = aie.buffer(%tile_2_2) {address = 57344 : i32, mem_bank = 3 : i32, sym_name = "A_L2L1_0_2_cons_buff_1"} : memref<64x64xbf16> 
    %A_L2L1_0_2_cons_prod_lock_0 = aie.lock(%tile_2_2, 1) {init = 2 : i32, sym_name = "A_L2L1_0_2_cons_prod_lock_0"}
    %A_L2L1_0_2_cons_cons_lock_0 = aie.lock(%tile_2_2, 2) {init = 0 : i32, sym_name = "A_L2L1_0_2_cons_cons_lock_0"}
    %A_L2L1_0_3_cons_buff_0 = aie.buffer(%tile_3_2) {address = 40960 : i32, mem_bank = 2 : i32, sym_name = "A_L2L1_0_3_cons_buff_0"} : memref<64x64xbf16> 
    %A_L2L1_0_3_cons_buff_1 = aie.buffer(%tile_3_2) {address = 57344 : i32, mem_bank = 3 : i32, sym_name = "A_L2L1_0_3_cons_buff_1"} : memref<64x64xbf16> 
    %A_L2L1_0_3_cons_prod_lock_0 = aie.lock(%tile_3_2, 1) {init = 2 : i32, sym_name = "A_L2L1_0_3_cons_prod_lock_0"}
    %A_L2L1_0_3_cons_cons_lock_0 = aie.lock(%tile_3_2, 2) {init = 0 : i32, sym_name = "A_L2L1_0_3_cons_cons_lock_0"}
    aie.flow(%mem_tile_0_1, DMA : 0, %tile_3_2, DMA : 0)
    aie.flow(%mem_tile_0_1, DMA : 0, %tile_2_2, DMA : 0)
    aie.flow(%mem_tile_0_1, DMA : 0, %tile_1_2, DMA : 0)
    aie.flow(%mem_tile_0_1, DMA : 0, %tile_0_2, DMA : 0)
    aie.flow(%shim_noc_tile_0_0, DMA : 0, %mem_tile_0_1, DMA : 0)
    aie.flow(%mem_tile_1_1, DMA : 0, %tile_3_3, DMA : 0)
    aie.flow(%mem_tile_1_1, DMA : 0, %tile_2_3, DMA : 0)
    aie.flow(%mem_tile_1_1, DMA : 0, %tile_1_3, DMA : 0)
    aie.flow(%mem_tile_1_1, DMA : 0, %tile_0_3, DMA : 0)
    aie.flow(%shim_noc_tile_1_0, DMA : 0, %mem_tile_1_1, DMA : 0)
    aie.flow(%mem_tile_2_1, DMA : 0, %tile_3_4, DMA : 0)
    aie.flow(%mem_tile_2_1, DMA : 0, %tile_2_4, DMA : 0)
    aie.flow(%mem_tile_2_1, DMA : 0, %tile_1_4, DMA : 0)
    aie.flow(%mem_tile_2_1, DMA : 0, %tile_0_4, DMA : 0)
    aie.flow(%shim_noc_tile_2_0, DMA : 0, %mem_tile_2_1, DMA : 0)
    aie.flow(%mem_tile_3_1, DMA : 0, %tile_3_5, DMA : 0)
    aie.flow(%mem_tile_3_1, DMA : 0, %tile_2_5, DMA : 0)
    aie.flow(%mem_tile_3_1, DMA : 0, %tile_1_5, DMA : 0)
    aie.flow(%mem_tile_3_1, DMA : 0, %tile_0_5, DMA : 0)
    aie.flow(%shim_noc_tile_3_0, DMA : 0, %mem_tile_3_1, DMA : 0)
    aie.flow(%mem_tile_0_1, DMA : 1, %tile_0_5, DMA : 1)
    aie.flow(%mem_tile_0_1, DMA : 1, %tile_0_4, DMA : 1)
    aie.flow(%mem_tile_0_1, DMA : 1, %tile_0_3, DMA : 1)
    aie.flow(%mem_tile_0_1, DMA : 1, %tile_0_2, DMA : 1)
    aie.flow(%shim_noc_tile_0_0, DMA : 1, %mem_tile_0_1, DMA : 1)
    aie.flow(%mem_tile_1_1, DMA : 1, %tile_1_5, DMA : 1)
    aie.flow(%mem_tile_1_1, DMA : 1, %tile_1_4, DMA : 1)
    aie.flow(%mem_tile_1_1, DMA : 1, %tile_1_3, DMA : 1)
    aie.flow(%mem_tile_1_1, DMA : 1, %tile_1_2, DMA : 1)
    aie.flow(%shim_noc_tile_1_0, DMA : 1, %mem_tile_1_1, DMA : 1)
    aie.flow(%mem_tile_2_1, DMA : 1, %tile_2_5, DMA : 1)
    aie.flow(%mem_tile_2_1, DMA : 1, %tile_2_4, DMA : 1)
    aie.flow(%mem_tile_2_1, DMA : 1, %tile_2_3, DMA : 1)
    aie.flow(%mem_tile_2_1, DMA : 1, %tile_2_2, DMA : 1)
    aie.flow(%shim_noc_tile_2_0, DMA : 1, %mem_tile_2_1, DMA : 1)
    aie.flow(%mem_tile_3_1, DMA : 1, %tile_3_5, DMA : 1)
    aie.flow(%mem_tile_3_1, DMA : 1, %tile_3_4, DMA : 1)
    aie.flow(%mem_tile_3_1, DMA : 1, %tile_3_3, DMA : 1)
    aie.flow(%mem_tile_3_1, DMA : 1, %tile_3_2, DMA : 1)
    aie.flow(%shim_noc_tile_3_0, DMA : 1, %mem_tile_3_1, DMA : 1)
    aie.flow(%tile_0_2, DMA : 0, %mem_tile_0_1, DMA : 2)
    aie.flow(%tile_0_3, DMA : 0, %mem_tile_0_1, DMA : 3)
    aie.flow(%tile_0_4, DMA : 0, %mem_tile_0_1, DMA : 4)
    aie.flow(%tile_0_5, DMA : 0, %mem_tile_0_1, DMA : 5)
    aie.flow(%mem_tile_0_1, DMA : 2, %shim_noc_tile_0_0, DMA : 0)
    aie.flow(%tile_1_2, DMA : 0, %mem_tile_1_1, DMA : 2)
    aie.flow(%tile_1_3, DMA : 0, %mem_tile_1_1, DMA : 3)
    aie.flow(%tile_1_4, DMA : 0, %mem_tile_1_1, DMA : 4)
    aie.flow(%tile_1_5, DMA : 0, %mem_tile_1_1, DMA : 5)
    aie.flow(%mem_tile_1_1, DMA : 2, %shim_noc_tile_1_0, DMA : 0)
    aie.flow(%tile_2_2, DMA : 0, %mem_tile_2_1, DMA : 2)
    aie.flow(%tile_2_3, DMA : 0, %mem_tile_2_1, DMA : 3)
    aie.flow(%tile_2_4, DMA : 0, %mem_tile_2_1, DMA : 4)
    aie.flow(%tile_2_5, DMA : 0, %mem_tile_2_1, DMA : 5)
    aie.flow(%mem_tile_2_1, DMA : 2, %shim_noc_tile_2_0, DMA : 0)
    aie.flow(%tile_3_2, DMA : 0, %mem_tile_3_1, DMA : 2)
    aie.flow(%tile_3_3, DMA : 0, %mem_tile_3_1, DMA : 3)
    aie.flow(%tile_3_4, DMA : 0, %mem_tile_3_1, DMA : 4)
    aie.flow(%tile_3_5, DMA : 0, %mem_tile_3_1, DMA : 5)
    aie.flow(%mem_tile_3_1, DMA : 2, %shim_noc_tile_3_0, DMA : 0)
    func.func private @zero_f32(memref<64x64xf32>) attributes {link_with = "gemm_64x64x64_0_0_1_0_1.o"}
    func.func private @matmul_bf16_f32(memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xf32>) attributes {link_with = "gemm_64x64x64_0_0_1_0_1.o"}
    func.func private @convert_copy_f32_to_bf16(memref<64x64xf32>, memref<64x64xbf16>, i32) attributes {link_with = "convert_copy.o"}
    %rtp0_0 = aie.buffer(%tile_0_2) {address = 11532 : i32, mem_bank = 0 : i32, sym_name = "rtp0_0"} : memref<2xi32> = dense<0>
    %acc_buffer_0_0 = aie.buffer(%tile_0_2) {address = 16384 : i32, mem_bank = 1 : i32, sym_name = "acc_buffer_0_0"} : memref<64x64xf32> 
    %rtp0_1 = aie.buffer(%tile_1_2) {address = 11532 : i32, mem_bank = 0 : i32, sym_name = "rtp0_1"} : memref<2xi32> = dense<0>
    %acc_buffer_0_1 = aie.buffer(%tile_1_2) {address = 16384 : i32, mem_bank = 1 : i32, sym_name = "acc_buffer_0_1"} : memref<64x64xf32> 
    %rtp0_2 = aie.buffer(%tile_2_2) {address = 11532 : i32, mem_bank = 0 : i32, sym_name = "rtp0_2"} : memref<2xi32> = dense<0>
    %acc_buffer_0_2 = aie.buffer(%tile_2_2) {address = 16384 : i32, mem_bank = 1 : i32, sym_name = "acc_buffer_0_2"} : memref<64x64xf32> 
    %rtp0_3 = aie.buffer(%tile_3_2) {address = 11532 : i32, mem_bank = 0 : i32, sym_name = "rtp0_3"} : memref<2xi32> = dense<0>
    %acc_buffer_0_3 = aie.buffer(%tile_3_2) {address = 16384 : i32, mem_bank = 1 : i32, sym_name = "acc_buffer_0_3"} : memref<64x64xf32> 
    %rtp1_0 = aie.buffer(%tile_0_3) {address = 11532 : i32, mem_bank = 0 : i32, sym_name = "rtp1_0"} : memref<2xi32> = dense<0>
    %acc_buffer_1_0 = aie.buffer(%tile_0_3) {address = 16384 : i32, mem_bank = 1 : i32, sym_name = "acc_buffer_1_0"} : memref<64x64xf32> 
    %rtp1_1 = aie.buffer(%tile_1_3) {address = 11532 : i32, mem_bank = 0 : i32, sym_name = "rtp1_1"} : memref<2xi32> = dense<0>
    %acc_buffer_1_1 = aie.buffer(%tile_1_3) {address = 16384 : i32, mem_bank = 1 : i32, sym_name = "acc_buffer_1_1"} : memref<64x64xf32> 
    %rtp1_2 = aie.buffer(%tile_2_3) {address = 11532 : i32, mem_bank = 0 : i32, sym_name = "rtp1_2"} : memref<2xi32> = dense<0>
    %acc_buffer_1_2 = aie.buffer(%tile_2_3) {address = 16384 : i32, mem_bank = 1 : i32, sym_name = "acc_buffer_1_2"} : memref<64x64xf32> 
    %rtp1_3 = aie.buffer(%tile_3_3) {address = 11532 : i32, mem_bank = 0 : i32, sym_name = "rtp1_3"} : memref<2xi32> = dense<0>
    %acc_buffer_1_3 = aie.buffer(%tile_3_3) {address = 16384 : i32, mem_bank = 1 : i32, sym_name = "acc_buffer_1_3"} : memref<64x64xf32> 
    %rtp2_0 = aie.buffer(%tile_0_4) {address = 11532 : i32, mem_bank = 0 : i32, sym_name = "rtp2_0"} : memref<2xi32> = dense<0>
    %acc_buffer_2_0 = aie.buffer(%tile_0_4) {address = 16384 : i32, mem_bank = 1 : i32, sym_name = "acc_buffer_2_0"} : memref<64x64xf32> 
    %rtp2_1 = aie.buffer(%tile_1_4) {address = 11532 : i32, mem_bank = 0 : i32, sym_name = "rtp2_1"} : memref<2xi32> = dense<0>
    %acc_buffer_2_1 = aie.buffer(%tile_1_4) {address = 16384 : i32, mem_bank = 1 : i32, sym_name = "acc_buffer_2_1"} : memref<64x64xf32> 
    %rtp2_2 = aie.buffer(%tile_2_4) {address = 11532 : i32, mem_bank = 0 : i32, sym_name = "rtp2_2"} : memref<2xi32> = dense<0>
    %acc_buffer_2_2 = aie.buffer(%tile_2_4) {address = 16384 : i32, mem_bank = 1 : i32, sym_name = "acc_buffer_2_2"} : memref<64x64xf32> 
    %rtp2_3 = aie.buffer(%tile_3_4) {address = 11532 : i32, mem_bank = 0 : i32, sym_name = "rtp2_3"} : memref<2xi32> = dense<0>
    %acc_buffer_2_3 = aie.buffer(%tile_3_4) {address = 16384 : i32, mem_bank = 1 : i32, sym_name = "acc_buffer_2_3"} : memref<64x64xf32> 
    %rtp3_0 = aie.buffer(%tile_0_5) {address = 11532 : i32, mem_bank = 0 : i32, sym_name = "rtp3_0"} : memref<2xi32> = dense<0>
    %acc_buffer_3_0 = aie.buffer(%tile_0_5) {address = 16384 : i32, mem_bank = 1 : i32, sym_name = "acc_buffer_3_0"} : memref<64x64xf32> 
    %rtp3_1 = aie.buffer(%tile_1_5) {address = 11532 : i32, mem_bank = 0 : i32, sym_name = "rtp3_1"} : memref<2xi32> = dense<0>
    %acc_buffer_3_1 = aie.buffer(%tile_1_5) {address = 16384 : i32, mem_bank = 1 : i32, sym_name = "acc_buffer_3_1"} : memref<64x64xf32> 
    %rtp3_2 = aie.buffer(%tile_2_5) {address = 11532 : i32, mem_bank = 0 : i32, sym_name = "rtp3_2"} : memref<2xi32> = dense<0>
    %acc_buffer_3_2 = aie.buffer(%tile_2_5) {address = 16384 : i32, mem_bank = 1 : i32, sym_name = "acc_buffer_3_2"} : memref<64x64xf32> 
    %rtp3_3 = aie.buffer(%tile_3_5) {address = 11532 : i32, mem_bank = 0 : i32, sym_name = "rtp3_3"} : memref<2xi32> = dense<0>
    %acc_buffer_3_3 = aie.buffer(%tile_3_5) {address = 16384 : i32, mem_bank = 1 : i32, sym_name = "acc_buffer_3_3"} : memref<64x64xf32> 
    %lock_0_2 = aie.lock(%tile_0_2, 0)
    %_anonymous0 = aie.buffer(%tile_0_2) {address = 11520 : i32, mem_bank = 0 : i32, sym_name = "_anonymous0"} : memref<3xi32> 
    %core_0_2 = aie.core(%tile_0_2) {
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c4096_i32 = arith.constant 4096 : i32
      %c1_i32 = arith.constant 1 : i32
      %c2 = arith.constant 2 : index
      %c1 = arith.constant 1 : index
      %c0_i32 = arith.constant 0 : i32
      %c0 = arith.constant 0 : index
      %c2_i32 = arith.constant 2 : i32
      memref.store %c0_i32, %_anonymous0[%c0] : memref<3xi32>
      memref.store %c0_i32, %_anonymous0[%c1] : memref<3xi32>
      memref.store %c0_i32, %_anonymous0[%c2] : memref<3xi32>
      cf.br ^bb1(%c0 : index)
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb16
      %1 = arith.cmpi slt, %0, %c9223372036854775807 : index
      cf.cond_br %1, ^bb2, ^bb17
    ^bb2:  // pred: ^bb1
      aie.use_lock(%lock_0_2, Acquire, 1)
      %2 = memref.load %rtp0_0[%c0] : memref<2xi32>
      %3 = memref.load %rtp0_0[%c1] : memref<2xi32>
      %4 = arith.index_cast %3 : i32 to index
      %5 = arith.index_cast %2 : i32 to index
      cf.br ^bb3(%c0 : index)
    ^bb3(%6: index):  // 2 preds: ^bb2, ^bb15
      %7 = arith.cmpi slt, %6, %4 : index
      cf.cond_br %7, ^bb4, ^bb16
    ^bb4:  // pred: ^bb3
      func.call @zero_f32(%acc_buffer_0_0) : (memref<64x64xf32>) -> ()
      cf.br ^bb5(%c0 : index)
    ^bb5(%8: index):  // 2 preds: ^bb4, ^bb14
      %9 = arith.cmpi slt, %8, %5 : index
      cf.cond_br %9, ^bb6, ^bb15
    ^bb6:  // pred: ^bb5
      aie.use_lock(%A_L2L1_0_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %10 = memref.load %_anonymous0[%c0] : memref<3xi32>
      %11 = arith.index_cast %10 : i32 to index
      %12 = arith.index_cast %11 : index to i32
      cf.switch %12 : i32, [
        default: ^bb9,
        0: ^bb7,
        1: ^bb8
      ]
    ^bb7:  // pred: ^bb6
      cf.br ^bb10(%A_L2L1_0_0_cons_buff_0 : memref<64x64xbf16>)
    ^bb8:  // pred: ^bb6
      cf.br ^bb10(%A_L2L1_0_0_cons_buff_1 : memref<64x64xbf16>)
    ^bb9:  // pred: ^bb6
      cf.br ^bb10(%A_L2L1_0_0_cons_buff_0 : memref<64x64xbf16>)
    ^bb10(%13: memref<64x64xbf16>):  // 3 preds: ^bb7, ^bb8, ^bb9
      aie.use_lock(%B_L2L1_0_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %14 = memref.load %_anonymous0[%c1] : memref<3xi32>
      %15 = arith.index_cast %14 : i32 to index
      %16 = arith.index_cast %15 : index to i32
      cf.switch %16 : i32, [
        default: ^bb13,
        0: ^bb11,
        1: ^bb12
      ]
    ^bb11:  // pred: ^bb10
      cf.br ^bb14(%B_L2L1_0_0_cons_buff_0 : memref<64x64xbf16>)
    ^bb12:  // pred: ^bb10
      cf.br ^bb14(%B_L2L1_0_0_cons_buff_1 : memref<64x64xbf16>)
    ^bb13:  // pred: ^bb10
      cf.br ^bb14(%B_L2L1_0_0_cons_buff_0 : memref<64x64xbf16>)
    ^bb14(%17: memref<64x64xbf16>):  // 3 preds: ^bb11, ^bb12, ^bb13
      func.call @matmul_bf16_f32(%13, %17, %acc_buffer_0_0) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xf32>) -> ()
      aie.use_lock(%A_L2L1_0_0_cons_prod_lock_0, Release, 1)
      %18 = memref.load %_anonymous0[%c0] : memref<3xi32>
      %19 = arith.addi %18, %c1_i32 : i32
      %20 = arith.cmpi sge, %19, %c2_i32 : i32
      %21 = arith.subi %19, %c2_i32 : i32
      %22 = arith.select %20, %21, %19 : i32
      memref.store %22, %_anonymous0[%c0] : memref<3xi32>
      aie.use_lock(%B_L2L1_0_0_cons_prod_lock_0, Release, 1)
      %23 = memref.load %_anonymous0[%c1] : memref<3xi32>
      %24 = arith.addi %23, %c1_i32 : i32
      %25 = arith.cmpi sge, %24, %c2_i32 : i32
      %26 = arith.subi %24, %c2_i32 : i32
      %27 = arith.select %25, %26, %24 : i32
      memref.store %27, %_anonymous0[%c1] : memref<3xi32>
      %28 = arith.addi %8, %c1 : index
      cf.br ^bb5(%28 : index)
    ^bb15:  // pred: ^bb5
      aie.use_lock(%C_L1L2_0_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @convert_copy_f32_to_bf16(%acc_buffer_0_0, %C_L1L2_0_0_buff_0, %c4096_i32) : (memref<64x64xf32>, memref<64x64xbf16>, i32) -> ()
      aie.use_lock(%C_L1L2_0_0_cons_lock_0, Release, 1)
      %29 = memref.load %_anonymous0[%c2] : memref<3xi32>
      %30 = arith.addi %29, %c1_i32 : i32
      %31 = arith.cmpi sge, %30, %c1_i32 : i32
      %32 = arith.select %31, %29, %30 : i32
      memref.store %32, %_anonymous0[%c2] : memref<3xi32>
      %33 = arith.addi %6, %c1 : index
      cf.br ^bb3(%33 : index)
    ^bb16:  // pred: ^bb3
      %34 = arith.addi %0, %c1 : index
      cf.br ^bb1(%34 : index)
    ^bb17:  // pred: ^bb1
      aie.end
    } {link_files = ["gemm_64x64x64_0_0_1_0_1.o", "convert_copy.o"], stack_size = 3328 : i32}
    %lock_1_2 = aie.lock(%tile_1_2, 0)
    %_anonymous1 = aie.buffer(%tile_1_2) {address = 11520 : i32, mem_bank = 0 : i32, sym_name = "_anonymous1"} : memref<3xi32> 
    %core_1_2 = aie.core(%tile_1_2) {
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c4096_i32 = arith.constant 4096 : i32
      %c1_i32 = arith.constant 1 : i32
      %c2 = arith.constant 2 : index
      %c1 = arith.constant 1 : index
      %c0_i32 = arith.constant 0 : i32
      %c0 = arith.constant 0 : index
      %c2_i32 = arith.constant 2 : i32
      memref.store %c0_i32, %_anonymous1[%c0] : memref<3xi32>
      memref.store %c0_i32, %_anonymous1[%c1] : memref<3xi32>
      memref.store %c0_i32, %_anonymous1[%c2] : memref<3xi32>
      cf.br ^bb1(%c0 : index)
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb16
      %1 = arith.cmpi slt, %0, %c9223372036854775807 : index
      cf.cond_br %1, ^bb2, ^bb17
    ^bb2:  // pred: ^bb1
      aie.use_lock(%lock_1_2, Acquire, 1)
      %2 = memref.load %rtp0_1[%c0] : memref<2xi32>
      %3 = memref.load %rtp0_1[%c1] : memref<2xi32>
      %4 = arith.index_cast %3 : i32 to index
      %5 = arith.index_cast %2 : i32 to index
      cf.br ^bb3(%c0 : index)
    ^bb3(%6: index):  // 2 preds: ^bb2, ^bb15
      %7 = arith.cmpi slt, %6, %4 : index
      cf.cond_br %7, ^bb4, ^bb16
    ^bb4:  // pred: ^bb3
      func.call @zero_f32(%acc_buffer_0_1) : (memref<64x64xf32>) -> ()
      cf.br ^bb5(%c0 : index)
    ^bb5(%8: index):  // 2 preds: ^bb4, ^bb14
      %9 = arith.cmpi slt, %8, %5 : index
      cf.cond_br %9, ^bb6, ^bb15
    ^bb6:  // pred: ^bb5
      aie.use_lock(%A_L2L1_0_1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %10 = memref.load %_anonymous1[%c0] : memref<3xi32>
      %11 = arith.index_cast %10 : i32 to index
      %12 = arith.index_cast %11 : index to i32
      cf.switch %12 : i32, [
        default: ^bb9,
        0: ^bb7,
        1: ^bb8
      ]
    ^bb7:  // pred: ^bb6
      cf.br ^bb10(%A_L2L1_0_1_cons_buff_0 : memref<64x64xbf16>)
    ^bb8:  // pred: ^bb6
      cf.br ^bb10(%A_L2L1_0_1_cons_buff_1 : memref<64x64xbf16>)
    ^bb9:  // pred: ^bb6
      cf.br ^bb10(%A_L2L1_0_1_cons_buff_0 : memref<64x64xbf16>)
    ^bb10(%13: memref<64x64xbf16>):  // 3 preds: ^bb7, ^bb8, ^bb9
      aie.use_lock(%B_L2L1_1_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %14 = memref.load %_anonymous1[%c1] : memref<3xi32>
      %15 = arith.index_cast %14 : i32 to index
      %16 = arith.index_cast %15 : index to i32
      cf.switch %16 : i32, [
        default: ^bb13,
        0: ^bb11,
        1: ^bb12
      ]
    ^bb11:  // pred: ^bb10
      cf.br ^bb14(%B_L2L1_1_0_cons_buff_0 : memref<64x64xbf16>)
    ^bb12:  // pred: ^bb10
      cf.br ^bb14(%B_L2L1_1_0_cons_buff_1 : memref<64x64xbf16>)
    ^bb13:  // pred: ^bb10
      cf.br ^bb14(%B_L2L1_1_0_cons_buff_0 : memref<64x64xbf16>)
    ^bb14(%17: memref<64x64xbf16>):  // 3 preds: ^bb11, ^bb12, ^bb13
      func.call @matmul_bf16_f32(%13, %17, %acc_buffer_0_1) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xf32>) -> ()
      aie.use_lock(%A_L2L1_0_1_cons_prod_lock_0, Release, 1)
      %18 = memref.load %_anonymous1[%c0] : memref<3xi32>
      %19 = arith.addi %18, %c1_i32 : i32
      %20 = arith.cmpi sge, %19, %c2_i32 : i32
      %21 = arith.subi %19, %c2_i32 : i32
      %22 = arith.select %20, %21, %19 : i32
      memref.store %22, %_anonymous1[%c0] : memref<3xi32>
      aie.use_lock(%B_L2L1_1_0_cons_prod_lock_0, Release, 1)
      %23 = memref.load %_anonymous1[%c1] : memref<3xi32>
      %24 = arith.addi %23, %c1_i32 : i32
      %25 = arith.cmpi sge, %24, %c2_i32 : i32
      %26 = arith.subi %24, %c2_i32 : i32
      %27 = arith.select %25, %26, %24 : i32
      memref.store %27, %_anonymous1[%c1] : memref<3xi32>
      %28 = arith.addi %8, %c1 : index
      cf.br ^bb5(%28 : index)
    ^bb15:  // pred: ^bb5
      aie.use_lock(%C_L1L2_1_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @convert_copy_f32_to_bf16(%acc_buffer_0_1, %C_L1L2_1_0_buff_0, %c4096_i32) : (memref<64x64xf32>, memref<64x64xbf16>, i32) -> ()
      aie.use_lock(%C_L1L2_1_0_cons_lock_0, Release, 1)
      %29 = memref.load %_anonymous1[%c2] : memref<3xi32>
      %30 = arith.addi %29, %c1_i32 : i32
      %31 = arith.cmpi sge, %30, %c1_i32 : i32
      %32 = arith.select %31, %29, %30 : i32
      memref.store %32, %_anonymous1[%c2] : memref<3xi32>
      %33 = arith.addi %6, %c1 : index
      cf.br ^bb3(%33 : index)
    ^bb16:  // pred: ^bb3
      %34 = arith.addi %0, %c1 : index
      cf.br ^bb1(%34 : index)
    ^bb17:  // pred: ^bb1
      aie.end
    } {link_files = ["gemm_64x64x64_0_0_1_0_1.o", "convert_copy.o"], stack_size = 3328 : i32}
    %lock_2_2 = aie.lock(%tile_2_2, 0)
    %_anonymous2 = aie.buffer(%tile_2_2) {address = 11520 : i32, mem_bank = 0 : i32, sym_name = "_anonymous2"} : memref<3xi32> 
    %core_2_2 = aie.core(%tile_2_2) {
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c4096_i32 = arith.constant 4096 : i32
      %c1_i32 = arith.constant 1 : i32
      %c2 = arith.constant 2 : index
      %c1 = arith.constant 1 : index
      %c0_i32 = arith.constant 0 : i32
      %c0 = arith.constant 0 : index
      %c2_i32 = arith.constant 2 : i32
      memref.store %c0_i32, %_anonymous2[%c0] : memref<3xi32>
      memref.store %c0_i32, %_anonymous2[%c1] : memref<3xi32>
      memref.store %c0_i32, %_anonymous2[%c2] : memref<3xi32>
      cf.br ^bb1(%c0 : index)
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb16
      %1 = arith.cmpi slt, %0, %c9223372036854775807 : index
      cf.cond_br %1, ^bb2, ^bb17
    ^bb2:  // pred: ^bb1
      aie.use_lock(%lock_2_2, Acquire, 1)
      %2 = memref.load %rtp0_2[%c0] : memref<2xi32>
      %3 = memref.load %rtp0_2[%c1] : memref<2xi32>
      %4 = arith.index_cast %3 : i32 to index
      %5 = arith.index_cast %2 : i32 to index
      cf.br ^bb3(%c0 : index)
    ^bb3(%6: index):  // 2 preds: ^bb2, ^bb15
      %7 = arith.cmpi slt, %6, %4 : index
      cf.cond_br %7, ^bb4, ^bb16
    ^bb4:  // pred: ^bb3
      func.call @zero_f32(%acc_buffer_0_2) : (memref<64x64xf32>) -> ()
      cf.br ^bb5(%c0 : index)
    ^bb5(%8: index):  // 2 preds: ^bb4, ^bb14
      %9 = arith.cmpi slt, %8, %5 : index
      cf.cond_br %9, ^bb6, ^bb15
    ^bb6:  // pred: ^bb5
      aie.use_lock(%A_L2L1_0_2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %10 = memref.load %_anonymous2[%c0] : memref<3xi32>
      %11 = arith.index_cast %10 : i32 to index
      %12 = arith.index_cast %11 : index to i32
      cf.switch %12 : i32, [
        default: ^bb9,
        0: ^bb7,
        1: ^bb8
      ]
    ^bb7:  // pred: ^bb6
      cf.br ^bb10(%A_L2L1_0_2_cons_buff_0 : memref<64x64xbf16>)
    ^bb8:  // pred: ^bb6
      cf.br ^bb10(%A_L2L1_0_2_cons_buff_1 : memref<64x64xbf16>)
    ^bb9:  // pred: ^bb6
      cf.br ^bb10(%A_L2L1_0_2_cons_buff_0 : memref<64x64xbf16>)
    ^bb10(%13: memref<64x64xbf16>):  // 3 preds: ^bb7, ^bb8, ^bb9
      aie.use_lock(%B_L2L1_2_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %14 = memref.load %_anonymous2[%c1] : memref<3xi32>
      %15 = arith.index_cast %14 : i32 to index
      %16 = arith.index_cast %15 : index to i32
      cf.switch %16 : i32, [
        default: ^bb13,
        0: ^bb11,
        1: ^bb12
      ]
    ^bb11:  // pred: ^bb10
      cf.br ^bb14(%B_L2L1_2_0_cons_buff_0 : memref<64x64xbf16>)
    ^bb12:  // pred: ^bb10
      cf.br ^bb14(%B_L2L1_2_0_cons_buff_1 : memref<64x64xbf16>)
    ^bb13:  // pred: ^bb10
      cf.br ^bb14(%B_L2L1_2_0_cons_buff_0 : memref<64x64xbf16>)
    ^bb14(%17: memref<64x64xbf16>):  // 3 preds: ^bb11, ^bb12, ^bb13
      func.call @matmul_bf16_f32(%13, %17, %acc_buffer_0_2) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xf32>) -> ()
      aie.use_lock(%A_L2L1_0_2_cons_prod_lock_0, Release, 1)
      %18 = memref.load %_anonymous2[%c0] : memref<3xi32>
      %19 = arith.addi %18, %c1_i32 : i32
      %20 = arith.cmpi sge, %19, %c2_i32 : i32
      %21 = arith.subi %19, %c2_i32 : i32
      %22 = arith.select %20, %21, %19 : i32
      memref.store %22, %_anonymous2[%c0] : memref<3xi32>
      aie.use_lock(%B_L2L1_2_0_cons_prod_lock_0, Release, 1)
      %23 = memref.load %_anonymous2[%c1] : memref<3xi32>
      %24 = arith.addi %23, %c1_i32 : i32
      %25 = arith.cmpi sge, %24, %c2_i32 : i32
      %26 = arith.subi %24, %c2_i32 : i32
      %27 = arith.select %25, %26, %24 : i32
      memref.store %27, %_anonymous2[%c1] : memref<3xi32>
      %28 = arith.addi %8, %c1 : index
      cf.br ^bb5(%28 : index)
    ^bb15:  // pred: ^bb5
      aie.use_lock(%C_L1L2_2_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @convert_copy_f32_to_bf16(%acc_buffer_0_2, %C_L1L2_2_0_buff_0, %c4096_i32) : (memref<64x64xf32>, memref<64x64xbf16>, i32) -> ()
      aie.use_lock(%C_L1L2_2_0_cons_lock_0, Release, 1)
      %29 = memref.load %_anonymous2[%c2] : memref<3xi32>
      %30 = arith.addi %29, %c1_i32 : i32
      %31 = arith.cmpi sge, %30, %c1_i32 : i32
      %32 = arith.select %31, %29, %30 : i32
      memref.store %32, %_anonymous2[%c2] : memref<3xi32>
      %33 = arith.addi %6, %c1 : index
      cf.br ^bb3(%33 : index)
    ^bb16:  // pred: ^bb3
      %34 = arith.addi %0, %c1 : index
      cf.br ^bb1(%34 : index)
    ^bb17:  // pred: ^bb1
      aie.end
    } {link_files = ["gemm_64x64x64_0_0_1_0_1.o", "convert_copy.o"], stack_size = 3328 : i32}
    %lock_3_2 = aie.lock(%tile_3_2, 0)
    %_anonymous3 = aie.buffer(%tile_3_2) {address = 11520 : i32, mem_bank = 0 : i32, sym_name = "_anonymous3"} : memref<3xi32> 
    %core_3_2 = aie.core(%tile_3_2) {
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c4096_i32 = arith.constant 4096 : i32
      %c1_i32 = arith.constant 1 : i32
      %c2 = arith.constant 2 : index
      %c1 = arith.constant 1 : index
      %c0_i32 = arith.constant 0 : i32
      %c0 = arith.constant 0 : index
      %c2_i32 = arith.constant 2 : i32
      memref.store %c0_i32, %_anonymous3[%c0] : memref<3xi32>
      memref.store %c0_i32, %_anonymous3[%c1] : memref<3xi32>
      memref.store %c0_i32, %_anonymous3[%c2] : memref<3xi32>
      cf.br ^bb1(%c0 : index)
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb16
      %1 = arith.cmpi slt, %0, %c9223372036854775807 : index
      cf.cond_br %1, ^bb2, ^bb17
    ^bb2:  // pred: ^bb1
      aie.use_lock(%lock_3_2, Acquire, 1)
      %2 = memref.load %rtp0_3[%c0] : memref<2xi32>
      %3 = memref.load %rtp0_3[%c1] : memref<2xi32>
      %4 = arith.index_cast %3 : i32 to index
      %5 = arith.index_cast %2 : i32 to index
      cf.br ^bb3(%c0 : index)
    ^bb3(%6: index):  // 2 preds: ^bb2, ^bb15
      %7 = arith.cmpi slt, %6, %4 : index
      cf.cond_br %7, ^bb4, ^bb16
    ^bb4:  // pred: ^bb3
      func.call @zero_f32(%acc_buffer_0_3) : (memref<64x64xf32>) -> ()
      cf.br ^bb5(%c0 : index)
    ^bb5(%8: index):  // 2 preds: ^bb4, ^bb14
      %9 = arith.cmpi slt, %8, %5 : index
      cf.cond_br %9, ^bb6, ^bb15
    ^bb6:  // pred: ^bb5
      aie.use_lock(%A_L2L1_0_3_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %10 = memref.load %_anonymous3[%c0] : memref<3xi32>
      %11 = arith.index_cast %10 : i32 to index
      %12 = arith.index_cast %11 : index to i32
      cf.switch %12 : i32, [
        default: ^bb9,
        0: ^bb7,
        1: ^bb8
      ]
    ^bb7:  // pred: ^bb6
      cf.br ^bb10(%A_L2L1_0_3_cons_buff_0 : memref<64x64xbf16>)
    ^bb8:  // pred: ^bb6
      cf.br ^bb10(%A_L2L1_0_3_cons_buff_1 : memref<64x64xbf16>)
    ^bb9:  // pred: ^bb6
      cf.br ^bb10(%A_L2L1_0_3_cons_buff_0 : memref<64x64xbf16>)
    ^bb10(%13: memref<64x64xbf16>):  // 3 preds: ^bb7, ^bb8, ^bb9
      aie.use_lock(%B_L2L1_3_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %14 = memref.load %_anonymous3[%c1] : memref<3xi32>
      %15 = arith.index_cast %14 : i32 to index
      %16 = arith.index_cast %15 : index to i32
      cf.switch %16 : i32, [
        default: ^bb13,
        0: ^bb11,
        1: ^bb12
      ]
    ^bb11:  // pred: ^bb10
      cf.br ^bb14(%B_L2L1_3_0_cons_buff_0 : memref<64x64xbf16>)
    ^bb12:  // pred: ^bb10
      cf.br ^bb14(%B_L2L1_3_0_cons_buff_1 : memref<64x64xbf16>)
    ^bb13:  // pred: ^bb10
      cf.br ^bb14(%B_L2L1_3_0_cons_buff_0 : memref<64x64xbf16>)
    ^bb14(%17: memref<64x64xbf16>):  // 3 preds: ^bb11, ^bb12, ^bb13
      func.call @matmul_bf16_f32(%13, %17, %acc_buffer_0_3) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xf32>) -> ()
      aie.use_lock(%A_L2L1_0_3_cons_prod_lock_0, Release, 1)
      %18 = memref.load %_anonymous3[%c0] : memref<3xi32>
      %19 = arith.addi %18, %c1_i32 : i32
      %20 = arith.cmpi sge, %19, %c2_i32 : i32
      %21 = arith.subi %19, %c2_i32 : i32
      %22 = arith.select %20, %21, %19 : i32
      memref.store %22, %_anonymous3[%c0] : memref<3xi32>
      aie.use_lock(%B_L2L1_3_0_cons_prod_lock_0, Release, 1)
      %23 = memref.load %_anonymous3[%c1] : memref<3xi32>
      %24 = arith.addi %23, %c1_i32 : i32
      %25 = arith.cmpi sge, %24, %c2_i32 : i32
      %26 = arith.subi %24, %c2_i32 : i32
      %27 = arith.select %25, %26, %24 : i32
      memref.store %27, %_anonymous3[%c1] : memref<3xi32>
      %28 = arith.addi %8, %c1 : index
      cf.br ^bb5(%28 : index)
    ^bb15:  // pred: ^bb5
      aie.use_lock(%C_L1L2_3_0_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @convert_copy_f32_to_bf16(%acc_buffer_0_3, %C_L1L2_3_0_buff_0, %c4096_i32) : (memref<64x64xf32>, memref<64x64xbf16>, i32) -> ()
      aie.use_lock(%C_L1L2_3_0_cons_lock_0, Release, 1)
      %29 = memref.load %_anonymous3[%c2] : memref<3xi32>
      %30 = arith.addi %29, %c1_i32 : i32
      %31 = arith.cmpi sge, %30, %c1_i32 : i32
      %32 = arith.select %31, %29, %30 : i32
      memref.store %32, %_anonymous3[%c2] : memref<3xi32>
      %33 = arith.addi %6, %c1 : index
      cf.br ^bb3(%33 : index)
    ^bb16:  // pred: ^bb3
      %34 = arith.addi %0, %c1 : index
      cf.br ^bb1(%34 : index)
    ^bb17:  // pred: ^bb1
      aie.end
    } {link_files = ["gemm_64x64x64_0_0_1_0_1.o", "convert_copy.o"], stack_size = 3328 : i32}
    %lock_0_3 = aie.lock(%tile_0_3, 0)
    %_anonymous4 = aie.buffer(%tile_0_3) {address = 11520 : i32, mem_bank = 0 : i32, sym_name = "_anonymous4"} : memref<3xi32> 
    %core_0_3 = aie.core(%tile_0_3) {
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c4096_i32 = arith.constant 4096 : i32
      %c1_i32 = arith.constant 1 : i32
      %c2 = arith.constant 2 : index
      %c1 = arith.constant 1 : index
      %c0_i32 = arith.constant 0 : i32
      %c0 = arith.constant 0 : index
      %c2_i32 = arith.constant 2 : i32
      memref.store %c0_i32, %_anonymous4[%c0] : memref<3xi32>
      memref.store %c0_i32, %_anonymous4[%c1] : memref<3xi32>
      memref.store %c0_i32, %_anonymous4[%c2] : memref<3xi32>
      cf.br ^bb1(%c0 : index)
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb16
      %1 = arith.cmpi slt, %0, %c9223372036854775807 : index
      cf.cond_br %1, ^bb2, ^bb17
    ^bb2:  // pred: ^bb1
      aie.use_lock(%lock_0_3, Acquire, 1)
      %2 = memref.load %rtp1_0[%c0] : memref<2xi32>
      %3 = memref.load %rtp1_0[%c1] : memref<2xi32>
      %4 = arith.index_cast %3 : i32 to index
      %5 = arith.index_cast %2 : i32 to index
      cf.br ^bb3(%c0 : index)
    ^bb3(%6: index):  // 2 preds: ^bb2, ^bb15
      %7 = arith.cmpi slt, %6, %4 : index
      cf.cond_br %7, ^bb4, ^bb16
    ^bb4:  // pred: ^bb3
      func.call @zero_f32(%acc_buffer_1_0) : (memref<64x64xf32>) -> ()
      cf.br ^bb5(%c0 : index)
    ^bb5(%8: index):  // 2 preds: ^bb4, ^bb14
      %9 = arith.cmpi slt, %8, %5 : index
      cf.cond_br %9, ^bb6, ^bb15
    ^bb6:  // pred: ^bb5
      aie.use_lock(%A_L2L1_1_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %10 = memref.load %_anonymous4[%c0] : memref<3xi32>
      %11 = arith.index_cast %10 : i32 to index
      %12 = arith.index_cast %11 : index to i32
      cf.switch %12 : i32, [
        default: ^bb9,
        0: ^bb7,
        1: ^bb8
      ]
    ^bb7:  // pred: ^bb6
      cf.br ^bb10(%A_L2L1_1_0_cons_buff_0 : memref<64x64xbf16>)
    ^bb8:  // pred: ^bb6
      cf.br ^bb10(%A_L2L1_1_0_cons_buff_1 : memref<64x64xbf16>)
    ^bb9:  // pred: ^bb6
      cf.br ^bb10(%A_L2L1_1_0_cons_buff_0 : memref<64x64xbf16>)
    ^bb10(%13: memref<64x64xbf16>):  // 3 preds: ^bb7, ^bb8, ^bb9
      aie.use_lock(%B_L2L1_0_1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %14 = memref.load %_anonymous4[%c1] : memref<3xi32>
      %15 = arith.index_cast %14 : i32 to index
      %16 = arith.index_cast %15 : index to i32
      cf.switch %16 : i32, [
        default: ^bb13,
        0: ^bb11,
        1: ^bb12
      ]
    ^bb11:  // pred: ^bb10
      cf.br ^bb14(%B_L2L1_0_1_cons_buff_0 : memref<64x64xbf16>)
    ^bb12:  // pred: ^bb10
      cf.br ^bb14(%B_L2L1_0_1_cons_buff_1 : memref<64x64xbf16>)
    ^bb13:  // pred: ^bb10
      cf.br ^bb14(%B_L2L1_0_1_cons_buff_0 : memref<64x64xbf16>)
    ^bb14(%17: memref<64x64xbf16>):  // 3 preds: ^bb11, ^bb12, ^bb13
      func.call @matmul_bf16_f32(%13, %17, %acc_buffer_1_0) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xf32>) -> ()
      aie.use_lock(%A_L2L1_1_0_cons_prod_lock_0, Release, 1)
      %18 = memref.load %_anonymous4[%c0] : memref<3xi32>
      %19 = arith.addi %18, %c1_i32 : i32
      %20 = arith.cmpi sge, %19, %c2_i32 : i32
      %21 = arith.subi %19, %c2_i32 : i32
      %22 = arith.select %20, %21, %19 : i32
      memref.store %22, %_anonymous4[%c0] : memref<3xi32>
      aie.use_lock(%B_L2L1_0_1_cons_prod_lock_0, Release, 1)
      %23 = memref.load %_anonymous4[%c1] : memref<3xi32>
      %24 = arith.addi %23, %c1_i32 : i32
      %25 = arith.cmpi sge, %24, %c2_i32 : i32
      %26 = arith.subi %24, %c2_i32 : i32
      %27 = arith.select %25, %26, %24 : i32
      memref.store %27, %_anonymous4[%c1] : memref<3xi32>
      %28 = arith.addi %8, %c1 : index
      cf.br ^bb5(%28 : index)
    ^bb15:  // pred: ^bb5
      aie.use_lock(%C_L1L2_0_1_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @convert_copy_f32_to_bf16(%acc_buffer_1_0, %C_L1L2_0_1_buff_0, %c4096_i32) : (memref<64x64xf32>, memref<64x64xbf16>, i32) -> ()
      aie.use_lock(%C_L1L2_0_1_cons_lock_0, Release, 1)
      %29 = memref.load %_anonymous4[%c2] : memref<3xi32>
      %30 = arith.addi %29, %c1_i32 : i32
      %31 = arith.cmpi sge, %30, %c1_i32 : i32
      %32 = arith.select %31, %29, %30 : i32
      memref.store %32, %_anonymous4[%c2] : memref<3xi32>
      %33 = arith.addi %6, %c1 : index
      cf.br ^bb3(%33 : index)
    ^bb16:  // pred: ^bb3
      %34 = arith.addi %0, %c1 : index
      cf.br ^bb1(%34 : index)
    ^bb17:  // pred: ^bb1
      aie.end
    } {link_files = ["gemm_64x64x64_0_0_1_0_1.o", "convert_copy.o"], stack_size = 3328 : i32}
    %lock_1_3 = aie.lock(%tile_1_3, 0)
    %_anonymous5 = aie.buffer(%tile_1_3) {address = 11520 : i32, mem_bank = 0 : i32, sym_name = "_anonymous5"} : memref<3xi32> 
    %core_1_3 = aie.core(%tile_1_3) {
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c4096_i32 = arith.constant 4096 : i32
      %c1_i32 = arith.constant 1 : i32
      %c2 = arith.constant 2 : index
      %c1 = arith.constant 1 : index
      %c0_i32 = arith.constant 0 : i32
      %c0 = arith.constant 0 : index
      %c2_i32 = arith.constant 2 : i32
      memref.store %c0_i32, %_anonymous5[%c0] : memref<3xi32>
      memref.store %c0_i32, %_anonymous5[%c1] : memref<3xi32>
      memref.store %c0_i32, %_anonymous5[%c2] : memref<3xi32>
      cf.br ^bb1(%c0 : index)
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb16
      %1 = arith.cmpi slt, %0, %c9223372036854775807 : index
      cf.cond_br %1, ^bb2, ^bb17
    ^bb2:  // pred: ^bb1
      aie.use_lock(%lock_1_3, Acquire, 1)
      %2 = memref.load %rtp1_1[%c0] : memref<2xi32>
      %3 = memref.load %rtp1_1[%c1] : memref<2xi32>
      %4 = arith.index_cast %3 : i32 to index
      %5 = arith.index_cast %2 : i32 to index
      cf.br ^bb3(%c0 : index)
    ^bb3(%6: index):  // 2 preds: ^bb2, ^bb15
      %7 = arith.cmpi slt, %6, %4 : index
      cf.cond_br %7, ^bb4, ^bb16
    ^bb4:  // pred: ^bb3
      func.call @zero_f32(%acc_buffer_1_1) : (memref<64x64xf32>) -> ()
      cf.br ^bb5(%c0 : index)
    ^bb5(%8: index):  // 2 preds: ^bb4, ^bb14
      %9 = arith.cmpi slt, %8, %5 : index
      cf.cond_br %9, ^bb6, ^bb15
    ^bb6:  // pred: ^bb5
      aie.use_lock(%A_L2L1_1_1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %10 = memref.load %_anonymous5[%c0] : memref<3xi32>
      %11 = arith.index_cast %10 : i32 to index
      %12 = arith.index_cast %11 : index to i32
      cf.switch %12 : i32, [
        default: ^bb9,
        0: ^bb7,
        1: ^bb8
      ]
    ^bb7:  // pred: ^bb6
      cf.br ^bb10(%A_L2L1_1_1_cons_buff_0 : memref<64x64xbf16>)
    ^bb8:  // pred: ^bb6
      cf.br ^bb10(%A_L2L1_1_1_cons_buff_1 : memref<64x64xbf16>)
    ^bb9:  // pred: ^bb6
      cf.br ^bb10(%A_L2L1_1_1_cons_buff_0 : memref<64x64xbf16>)
    ^bb10(%13: memref<64x64xbf16>):  // 3 preds: ^bb7, ^bb8, ^bb9
      aie.use_lock(%B_L2L1_1_1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %14 = memref.load %_anonymous5[%c1] : memref<3xi32>
      %15 = arith.index_cast %14 : i32 to index
      %16 = arith.index_cast %15 : index to i32
      cf.switch %16 : i32, [
        default: ^bb13,
        0: ^bb11,
        1: ^bb12
      ]
    ^bb11:  // pred: ^bb10
      cf.br ^bb14(%B_L2L1_1_1_cons_buff_0 : memref<64x64xbf16>)
    ^bb12:  // pred: ^bb10
      cf.br ^bb14(%B_L2L1_1_1_cons_buff_1 : memref<64x64xbf16>)
    ^bb13:  // pred: ^bb10
      cf.br ^bb14(%B_L2L1_1_1_cons_buff_0 : memref<64x64xbf16>)
    ^bb14(%17: memref<64x64xbf16>):  // 3 preds: ^bb11, ^bb12, ^bb13
      func.call @matmul_bf16_f32(%13, %17, %acc_buffer_1_1) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xf32>) -> ()
      aie.use_lock(%A_L2L1_1_1_cons_prod_lock_0, Release, 1)
      %18 = memref.load %_anonymous5[%c0] : memref<3xi32>
      %19 = arith.addi %18, %c1_i32 : i32
      %20 = arith.cmpi sge, %19, %c2_i32 : i32
      %21 = arith.subi %19, %c2_i32 : i32
      %22 = arith.select %20, %21, %19 : i32
      memref.store %22, %_anonymous5[%c0] : memref<3xi32>
      aie.use_lock(%B_L2L1_1_1_cons_prod_lock_0, Release, 1)
      %23 = memref.load %_anonymous5[%c1] : memref<3xi32>
      %24 = arith.addi %23, %c1_i32 : i32
      %25 = arith.cmpi sge, %24, %c2_i32 : i32
      %26 = arith.subi %24, %c2_i32 : i32
      %27 = arith.select %25, %26, %24 : i32
      memref.store %27, %_anonymous5[%c1] : memref<3xi32>
      %28 = arith.addi %8, %c1 : index
      cf.br ^bb5(%28 : index)
    ^bb15:  // pred: ^bb5
      aie.use_lock(%C_L1L2_1_1_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @convert_copy_f32_to_bf16(%acc_buffer_1_1, %C_L1L2_1_1_buff_0, %c4096_i32) : (memref<64x64xf32>, memref<64x64xbf16>, i32) -> ()
      aie.use_lock(%C_L1L2_1_1_cons_lock_0, Release, 1)
      %29 = memref.load %_anonymous5[%c2] : memref<3xi32>
      %30 = arith.addi %29, %c1_i32 : i32
      %31 = arith.cmpi sge, %30, %c1_i32 : i32
      %32 = arith.select %31, %29, %30 : i32
      memref.store %32, %_anonymous5[%c2] : memref<3xi32>
      %33 = arith.addi %6, %c1 : index
      cf.br ^bb3(%33 : index)
    ^bb16:  // pred: ^bb3
      %34 = arith.addi %0, %c1 : index
      cf.br ^bb1(%34 : index)
    ^bb17:  // pred: ^bb1
      aie.end
    } {link_files = ["gemm_64x64x64_0_0_1_0_1.o", "convert_copy.o"], stack_size = 3328 : i32}
    %lock_2_3 = aie.lock(%tile_2_3, 0)
    %_anonymous6 = aie.buffer(%tile_2_3) {address = 11520 : i32, mem_bank = 0 : i32, sym_name = "_anonymous6"} : memref<3xi32> 
    %core_2_3 = aie.core(%tile_2_3) {
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c4096_i32 = arith.constant 4096 : i32
      %c1_i32 = arith.constant 1 : i32
      %c2 = arith.constant 2 : index
      %c1 = arith.constant 1 : index
      %c0_i32 = arith.constant 0 : i32
      %c0 = arith.constant 0 : index
      %c2_i32 = arith.constant 2 : i32
      memref.store %c0_i32, %_anonymous6[%c0] : memref<3xi32>
      memref.store %c0_i32, %_anonymous6[%c1] : memref<3xi32>
      memref.store %c0_i32, %_anonymous6[%c2] : memref<3xi32>
      cf.br ^bb1(%c0 : index)
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb16
      %1 = arith.cmpi slt, %0, %c9223372036854775807 : index
      cf.cond_br %1, ^bb2, ^bb17
    ^bb2:  // pred: ^bb1
      aie.use_lock(%lock_2_3, Acquire, 1)
      %2 = memref.load %rtp1_2[%c0] : memref<2xi32>
      %3 = memref.load %rtp1_2[%c1] : memref<2xi32>
      %4 = arith.index_cast %3 : i32 to index
      %5 = arith.index_cast %2 : i32 to index
      cf.br ^bb3(%c0 : index)
    ^bb3(%6: index):  // 2 preds: ^bb2, ^bb15
      %7 = arith.cmpi slt, %6, %4 : index
      cf.cond_br %7, ^bb4, ^bb16
    ^bb4:  // pred: ^bb3
      func.call @zero_f32(%acc_buffer_1_2) : (memref<64x64xf32>) -> ()
      cf.br ^bb5(%c0 : index)
    ^bb5(%8: index):  // 2 preds: ^bb4, ^bb14
      %9 = arith.cmpi slt, %8, %5 : index
      cf.cond_br %9, ^bb6, ^bb15
    ^bb6:  // pred: ^bb5
      aie.use_lock(%A_L2L1_1_2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %10 = memref.load %_anonymous6[%c0] : memref<3xi32>
      %11 = arith.index_cast %10 : i32 to index
      %12 = arith.index_cast %11 : index to i32
      cf.switch %12 : i32, [
        default: ^bb9,
        0: ^bb7,
        1: ^bb8
      ]
    ^bb7:  // pred: ^bb6
      cf.br ^bb10(%A_L2L1_1_2_cons_buff_0 : memref<64x64xbf16>)
    ^bb8:  // pred: ^bb6
      cf.br ^bb10(%A_L2L1_1_2_cons_buff_1 : memref<64x64xbf16>)
    ^bb9:  // pred: ^bb6
      cf.br ^bb10(%A_L2L1_1_2_cons_buff_0 : memref<64x64xbf16>)
    ^bb10(%13: memref<64x64xbf16>):  // 3 preds: ^bb7, ^bb8, ^bb9
      aie.use_lock(%B_L2L1_2_1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %14 = memref.load %_anonymous6[%c1] : memref<3xi32>
      %15 = arith.index_cast %14 : i32 to index
      %16 = arith.index_cast %15 : index to i32
      cf.switch %16 : i32, [
        default: ^bb13,
        0: ^bb11,
        1: ^bb12
      ]
    ^bb11:  // pred: ^bb10
      cf.br ^bb14(%B_L2L1_2_1_cons_buff_0 : memref<64x64xbf16>)
    ^bb12:  // pred: ^bb10
      cf.br ^bb14(%B_L2L1_2_1_cons_buff_1 : memref<64x64xbf16>)
    ^bb13:  // pred: ^bb10
      cf.br ^bb14(%B_L2L1_2_1_cons_buff_0 : memref<64x64xbf16>)
    ^bb14(%17: memref<64x64xbf16>):  // 3 preds: ^bb11, ^bb12, ^bb13
      func.call @matmul_bf16_f32(%13, %17, %acc_buffer_1_2) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xf32>) -> ()
      aie.use_lock(%A_L2L1_1_2_cons_prod_lock_0, Release, 1)
      %18 = memref.load %_anonymous6[%c0] : memref<3xi32>
      %19 = arith.addi %18, %c1_i32 : i32
      %20 = arith.cmpi sge, %19, %c2_i32 : i32
      %21 = arith.subi %19, %c2_i32 : i32
      %22 = arith.select %20, %21, %19 : i32
      memref.store %22, %_anonymous6[%c0] : memref<3xi32>
      aie.use_lock(%B_L2L1_2_1_cons_prod_lock_0, Release, 1)
      %23 = memref.load %_anonymous6[%c1] : memref<3xi32>
      %24 = arith.addi %23, %c1_i32 : i32
      %25 = arith.cmpi sge, %24, %c2_i32 : i32
      %26 = arith.subi %24, %c2_i32 : i32
      %27 = arith.select %25, %26, %24 : i32
      memref.store %27, %_anonymous6[%c1] : memref<3xi32>
      %28 = arith.addi %8, %c1 : index
      cf.br ^bb5(%28 : index)
    ^bb15:  // pred: ^bb5
      aie.use_lock(%C_L1L2_2_1_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @convert_copy_f32_to_bf16(%acc_buffer_1_2, %C_L1L2_2_1_buff_0, %c4096_i32) : (memref<64x64xf32>, memref<64x64xbf16>, i32) -> ()
      aie.use_lock(%C_L1L2_2_1_cons_lock_0, Release, 1)
      %29 = memref.load %_anonymous6[%c2] : memref<3xi32>
      %30 = arith.addi %29, %c1_i32 : i32
      %31 = arith.cmpi sge, %30, %c1_i32 : i32
      %32 = arith.select %31, %29, %30 : i32
      memref.store %32, %_anonymous6[%c2] : memref<3xi32>
      %33 = arith.addi %6, %c1 : index
      cf.br ^bb3(%33 : index)
    ^bb16:  // pred: ^bb3
      %34 = arith.addi %0, %c1 : index
      cf.br ^bb1(%34 : index)
    ^bb17:  // pred: ^bb1
      aie.end
    } {link_files = ["gemm_64x64x64_0_0_1_0_1.o", "convert_copy.o"], stack_size = 3328 : i32}
    %lock_3_3 = aie.lock(%tile_3_3, 0)
    %_anonymous7 = aie.buffer(%tile_3_3) {address = 11520 : i32, mem_bank = 0 : i32, sym_name = "_anonymous7"} : memref<3xi32> 
    %core_3_3 = aie.core(%tile_3_3) {
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c4096_i32 = arith.constant 4096 : i32
      %c1_i32 = arith.constant 1 : i32
      %c2 = arith.constant 2 : index
      %c1 = arith.constant 1 : index
      %c0_i32 = arith.constant 0 : i32
      %c0 = arith.constant 0 : index
      %c2_i32 = arith.constant 2 : i32
      memref.store %c0_i32, %_anonymous7[%c0] : memref<3xi32>
      memref.store %c0_i32, %_anonymous7[%c1] : memref<3xi32>
      memref.store %c0_i32, %_anonymous7[%c2] : memref<3xi32>
      cf.br ^bb1(%c0 : index)
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb16
      %1 = arith.cmpi slt, %0, %c9223372036854775807 : index
      cf.cond_br %1, ^bb2, ^bb17
    ^bb2:  // pred: ^bb1
      aie.use_lock(%lock_3_3, Acquire, 1)
      %2 = memref.load %rtp1_3[%c0] : memref<2xi32>
      %3 = memref.load %rtp1_3[%c1] : memref<2xi32>
      %4 = arith.index_cast %3 : i32 to index
      %5 = arith.index_cast %2 : i32 to index
      cf.br ^bb3(%c0 : index)
    ^bb3(%6: index):  // 2 preds: ^bb2, ^bb15
      %7 = arith.cmpi slt, %6, %4 : index
      cf.cond_br %7, ^bb4, ^bb16
    ^bb4:  // pred: ^bb3
      func.call @zero_f32(%acc_buffer_1_3) : (memref<64x64xf32>) -> ()
      cf.br ^bb5(%c0 : index)
    ^bb5(%8: index):  // 2 preds: ^bb4, ^bb14
      %9 = arith.cmpi slt, %8, %5 : index
      cf.cond_br %9, ^bb6, ^bb15
    ^bb6:  // pred: ^bb5
      aie.use_lock(%A_L2L1_1_3_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %10 = memref.load %_anonymous7[%c0] : memref<3xi32>
      %11 = arith.index_cast %10 : i32 to index
      %12 = arith.index_cast %11 : index to i32
      cf.switch %12 : i32, [
        default: ^bb9,
        0: ^bb7,
        1: ^bb8
      ]
    ^bb7:  // pred: ^bb6
      cf.br ^bb10(%A_L2L1_1_3_cons_buff_0 : memref<64x64xbf16>)
    ^bb8:  // pred: ^bb6
      cf.br ^bb10(%A_L2L1_1_3_cons_buff_1 : memref<64x64xbf16>)
    ^bb9:  // pred: ^bb6
      cf.br ^bb10(%A_L2L1_1_3_cons_buff_0 : memref<64x64xbf16>)
    ^bb10(%13: memref<64x64xbf16>):  // 3 preds: ^bb7, ^bb8, ^bb9
      aie.use_lock(%B_L2L1_3_1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %14 = memref.load %_anonymous7[%c1] : memref<3xi32>
      %15 = arith.index_cast %14 : i32 to index
      %16 = arith.index_cast %15 : index to i32
      cf.switch %16 : i32, [
        default: ^bb13,
        0: ^bb11,
        1: ^bb12
      ]
    ^bb11:  // pred: ^bb10
      cf.br ^bb14(%B_L2L1_3_1_cons_buff_0 : memref<64x64xbf16>)
    ^bb12:  // pred: ^bb10
      cf.br ^bb14(%B_L2L1_3_1_cons_buff_1 : memref<64x64xbf16>)
    ^bb13:  // pred: ^bb10
      cf.br ^bb14(%B_L2L1_3_1_cons_buff_0 : memref<64x64xbf16>)
    ^bb14(%17: memref<64x64xbf16>):  // 3 preds: ^bb11, ^bb12, ^bb13
      func.call @matmul_bf16_f32(%13, %17, %acc_buffer_1_3) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xf32>) -> ()
      aie.use_lock(%A_L2L1_1_3_cons_prod_lock_0, Release, 1)
      %18 = memref.load %_anonymous7[%c0] : memref<3xi32>
      %19 = arith.addi %18, %c1_i32 : i32
      %20 = arith.cmpi sge, %19, %c2_i32 : i32
      %21 = arith.subi %19, %c2_i32 : i32
      %22 = arith.select %20, %21, %19 : i32
      memref.store %22, %_anonymous7[%c0] : memref<3xi32>
      aie.use_lock(%B_L2L1_3_1_cons_prod_lock_0, Release, 1)
      %23 = memref.load %_anonymous7[%c1] : memref<3xi32>
      %24 = arith.addi %23, %c1_i32 : i32
      %25 = arith.cmpi sge, %24, %c2_i32 : i32
      %26 = arith.subi %24, %c2_i32 : i32
      %27 = arith.select %25, %26, %24 : i32
      memref.store %27, %_anonymous7[%c1] : memref<3xi32>
      %28 = arith.addi %8, %c1 : index
      cf.br ^bb5(%28 : index)
    ^bb15:  // pred: ^bb5
      aie.use_lock(%C_L1L2_3_1_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @convert_copy_f32_to_bf16(%acc_buffer_1_3, %C_L1L2_3_1_buff_0, %c4096_i32) : (memref<64x64xf32>, memref<64x64xbf16>, i32) -> ()
      aie.use_lock(%C_L1L2_3_1_cons_lock_0, Release, 1)
      %29 = memref.load %_anonymous7[%c2] : memref<3xi32>
      %30 = arith.addi %29, %c1_i32 : i32
      %31 = arith.cmpi sge, %30, %c1_i32 : i32
      %32 = arith.select %31, %29, %30 : i32
      memref.store %32, %_anonymous7[%c2] : memref<3xi32>
      %33 = arith.addi %6, %c1 : index
      cf.br ^bb3(%33 : index)
    ^bb16:  // pred: ^bb3
      %34 = arith.addi %0, %c1 : index
      cf.br ^bb1(%34 : index)
    ^bb17:  // pred: ^bb1
      aie.end
    } {link_files = ["gemm_64x64x64_0_0_1_0_1.o", "convert_copy.o"], stack_size = 3328 : i32}
    %lock_0_4 = aie.lock(%tile_0_4, 0)
    %_anonymous8 = aie.buffer(%tile_0_4) {address = 11520 : i32, mem_bank = 0 : i32, sym_name = "_anonymous8"} : memref<3xi32> 
    %core_0_4 = aie.core(%tile_0_4) {
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c4096_i32 = arith.constant 4096 : i32
      %c1_i32 = arith.constant 1 : i32
      %c2 = arith.constant 2 : index
      %c1 = arith.constant 1 : index
      %c0_i32 = arith.constant 0 : i32
      %c0 = arith.constant 0 : index
      %c2_i32 = arith.constant 2 : i32
      memref.store %c0_i32, %_anonymous8[%c0] : memref<3xi32>
      memref.store %c0_i32, %_anonymous8[%c1] : memref<3xi32>
      memref.store %c0_i32, %_anonymous8[%c2] : memref<3xi32>
      cf.br ^bb1(%c0 : index)
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb16
      %1 = arith.cmpi slt, %0, %c9223372036854775807 : index
      cf.cond_br %1, ^bb2, ^bb17
    ^bb2:  // pred: ^bb1
      aie.use_lock(%lock_0_4, Acquire, 1)
      %2 = memref.load %rtp2_0[%c0] : memref<2xi32>
      %3 = memref.load %rtp2_0[%c1] : memref<2xi32>
      %4 = arith.index_cast %3 : i32 to index
      %5 = arith.index_cast %2 : i32 to index
      cf.br ^bb3(%c0 : index)
    ^bb3(%6: index):  // 2 preds: ^bb2, ^bb15
      %7 = arith.cmpi slt, %6, %4 : index
      cf.cond_br %7, ^bb4, ^bb16
    ^bb4:  // pred: ^bb3
      func.call @zero_f32(%acc_buffer_2_0) : (memref<64x64xf32>) -> ()
      cf.br ^bb5(%c0 : index)
    ^bb5(%8: index):  // 2 preds: ^bb4, ^bb14
      %9 = arith.cmpi slt, %8, %5 : index
      cf.cond_br %9, ^bb6, ^bb15
    ^bb6:  // pred: ^bb5
      aie.use_lock(%A_L2L1_2_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %10 = memref.load %_anonymous8[%c0] : memref<3xi32>
      %11 = arith.index_cast %10 : i32 to index
      %12 = arith.index_cast %11 : index to i32
      cf.switch %12 : i32, [
        default: ^bb9,
        0: ^bb7,
        1: ^bb8
      ]
    ^bb7:  // pred: ^bb6
      cf.br ^bb10(%A_L2L1_2_0_cons_buff_0 : memref<64x64xbf16>)
    ^bb8:  // pred: ^bb6
      cf.br ^bb10(%A_L2L1_2_0_cons_buff_1 : memref<64x64xbf16>)
    ^bb9:  // pred: ^bb6
      cf.br ^bb10(%A_L2L1_2_0_cons_buff_0 : memref<64x64xbf16>)
    ^bb10(%13: memref<64x64xbf16>):  // 3 preds: ^bb7, ^bb8, ^bb9
      aie.use_lock(%B_L2L1_0_2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %14 = memref.load %_anonymous8[%c1] : memref<3xi32>
      %15 = arith.index_cast %14 : i32 to index
      %16 = arith.index_cast %15 : index to i32
      cf.switch %16 : i32, [
        default: ^bb13,
        0: ^bb11,
        1: ^bb12
      ]
    ^bb11:  // pred: ^bb10
      cf.br ^bb14(%B_L2L1_0_2_cons_buff_0 : memref<64x64xbf16>)
    ^bb12:  // pred: ^bb10
      cf.br ^bb14(%B_L2L1_0_2_cons_buff_1 : memref<64x64xbf16>)
    ^bb13:  // pred: ^bb10
      cf.br ^bb14(%B_L2L1_0_2_cons_buff_0 : memref<64x64xbf16>)
    ^bb14(%17: memref<64x64xbf16>):  // 3 preds: ^bb11, ^bb12, ^bb13
      func.call @matmul_bf16_f32(%13, %17, %acc_buffer_2_0) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xf32>) -> ()
      aie.use_lock(%A_L2L1_2_0_cons_prod_lock_0, Release, 1)
      %18 = memref.load %_anonymous8[%c0] : memref<3xi32>
      %19 = arith.addi %18, %c1_i32 : i32
      %20 = arith.cmpi sge, %19, %c2_i32 : i32
      %21 = arith.subi %19, %c2_i32 : i32
      %22 = arith.select %20, %21, %19 : i32
      memref.store %22, %_anonymous8[%c0] : memref<3xi32>
      aie.use_lock(%B_L2L1_0_2_cons_prod_lock_0, Release, 1)
      %23 = memref.load %_anonymous8[%c1] : memref<3xi32>
      %24 = arith.addi %23, %c1_i32 : i32
      %25 = arith.cmpi sge, %24, %c2_i32 : i32
      %26 = arith.subi %24, %c2_i32 : i32
      %27 = arith.select %25, %26, %24 : i32
      memref.store %27, %_anonymous8[%c1] : memref<3xi32>
      %28 = arith.addi %8, %c1 : index
      cf.br ^bb5(%28 : index)
    ^bb15:  // pred: ^bb5
      aie.use_lock(%C_L1L2_0_2_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @convert_copy_f32_to_bf16(%acc_buffer_2_0, %C_L1L2_0_2_buff_0, %c4096_i32) : (memref<64x64xf32>, memref<64x64xbf16>, i32) -> ()
      aie.use_lock(%C_L1L2_0_2_cons_lock_0, Release, 1)
      %29 = memref.load %_anonymous8[%c2] : memref<3xi32>
      %30 = arith.addi %29, %c1_i32 : i32
      %31 = arith.cmpi sge, %30, %c1_i32 : i32
      %32 = arith.select %31, %29, %30 : i32
      memref.store %32, %_anonymous8[%c2] : memref<3xi32>
      %33 = arith.addi %6, %c1 : index
      cf.br ^bb3(%33 : index)
    ^bb16:  // pred: ^bb3
      %34 = arith.addi %0, %c1 : index
      cf.br ^bb1(%34 : index)
    ^bb17:  // pred: ^bb1
      aie.end
    } {link_files = ["gemm_64x64x64_0_0_1_0_1.o", "convert_copy.o"], stack_size = 3328 : i32}
    %lock_1_4 = aie.lock(%tile_1_4, 0)
    %_anonymous9 = aie.buffer(%tile_1_4) {address = 11520 : i32, mem_bank = 0 : i32, sym_name = "_anonymous9"} : memref<3xi32> 
    %core_1_4 = aie.core(%tile_1_4) {
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c4096_i32 = arith.constant 4096 : i32
      %c1_i32 = arith.constant 1 : i32
      %c2 = arith.constant 2 : index
      %c1 = arith.constant 1 : index
      %c0_i32 = arith.constant 0 : i32
      %c0 = arith.constant 0 : index
      %c2_i32 = arith.constant 2 : i32
      memref.store %c0_i32, %_anonymous9[%c0] : memref<3xi32>
      memref.store %c0_i32, %_anonymous9[%c1] : memref<3xi32>
      memref.store %c0_i32, %_anonymous9[%c2] : memref<3xi32>
      cf.br ^bb1(%c0 : index)
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb16
      %1 = arith.cmpi slt, %0, %c9223372036854775807 : index
      cf.cond_br %1, ^bb2, ^bb17
    ^bb2:  // pred: ^bb1
      aie.use_lock(%lock_1_4, Acquire, 1)
      %2 = memref.load %rtp2_1[%c0] : memref<2xi32>
      %3 = memref.load %rtp2_1[%c1] : memref<2xi32>
      %4 = arith.index_cast %3 : i32 to index
      %5 = arith.index_cast %2 : i32 to index
      cf.br ^bb3(%c0 : index)
    ^bb3(%6: index):  // 2 preds: ^bb2, ^bb15
      %7 = arith.cmpi slt, %6, %4 : index
      cf.cond_br %7, ^bb4, ^bb16
    ^bb4:  // pred: ^bb3
      func.call @zero_f32(%acc_buffer_2_1) : (memref<64x64xf32>) -> ()
      cf.br ^bb5(%c0 : index)
    ^bb5(%8: index):  // 2 preds: ^bb4, ^bb14
      %9 = arith.cmpi slt, %8, %5 : index
      cf.cond_br %9, ^bb6, ^bb15
    ^bb6:  // pred: ^bb5
      aie.use_lock(%A_L2L1_2_1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %10 = memref.load %_anonymous9[%c0] : memref<3xi32>
      %11 = arith.index_cast %10 : i32 to index
      %12 = arith.index_cast %11 : index to i32
      cf.switch %12 : i32, [
        default: ^bb9,
        0: ^bb7,
        1: ^bb8
      ]
    ^bb7:  // pred: ^bb6
      cf.br ^bb10(%A_L2L1_2_1_cons_buff_0 : memref<64x64xbf16>)
    ^bb8:  // pred: ^bb6
      cf.br ^bb10(%A_L2L1_2_1_cons_buff_1 : memref<64x64xbf16>)
    ^bb9:  // pred: ^bb6
      cf.br ^bb10(%A_L2L1_2_1_cons_buff_0 : memref<64x64xbf16>)
    ^bb10(%13: memref<64x64xbf16>):  // 3 preds: ^bb7, ^bb8, ^bb9
      aie.use_lock(%B_L2L1_1_2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %14 = memref.load %_anonymous9[%c1] : memref<3xi32>
      %15 = arith.index_cast %14 : i32 to index
      %16 = arith.index_cast %15 : index to i32
      cf.switch %16 : i32, [
        default: ^bb13,
        0: ^bb11,
        1: ^bb12
      ]
    ^bb11:  // pred: ^bb10
      cf.br ^bb14(%B_L2L1_1_2_cons_buff_0 : memref<64x64xbf16>)
    ^bb12:  // pred: ^bb10
      cf.br ^bb14(%B_L2L1_1_2_cons_buff_1 : memref<64x64xbf16>)
    ^bb13:  // pred: ^bb10
      cf.br ^bb14(%B_L2L1_1_2_cons_buff_0 : memref<64x64xbf16>)
    ^bb14(%17: memref<64x64xbf16>):  // 3 preds: ^bb11, ^bb12, ^bb13
      func.call @matmul_bf16_f32(%13, %17, %acc_buffer_2_1) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xf32>) -> ()
      aie.use_lock(%A_L2L1_2_1_cons_prod_lock_0, Release, 1)
      %18 = memref.load %_anonymous9[%c0] : memref<3xi32>
      %19 = arith.addi %18, %c1_i32 : i32
      %20 = arith.cmpi sge, %19, %c2_i32 : i32
      %21 = arith.subi %19, %c2_i32 : i32
      %22 = arith.select %20, %21, %19 : i32
      memref.store %22, %_anonymous9[%c0] : memref<3xi32>
      aie.use_lock(%B_L2L1_1_2_cons_prod_lock_0, Release, 1)
      %23 = memref.load %_anonymous9[%c1] : memref<3xi32>
      %24 = arith.addi %23, %c1_i32 : i32
      %25 = arith.cmpi sge, %24, %c2_i32 : i32
      %26 = arith.subi %24, %c2_i32 : i32
      %27 = arith.select %25, %26, %24 : i32
      memref.store %27, %_anonymous9[%c1] : memref<3xi32>
      %28 = arith.addi %8, %c1 : index
      cf.br ^bb5(%28 : index)
    ^bb15:  // pred: ^bb5
      aie.use_lock(%C_L1L2_1_2_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @convert_copy_f32_to_bf16(%acc_buffer_2_1, %C_L1L2_1_2_buff_0, %c4096_i32) : (memref<64x64xf32>, memref<64x64xbf16>, i32) -> ()
      aie.use_lock(%C_L1L2_1_2_cons_lock_0, Release, 1)
      %29 = memref.load %_anonymous9[%c2] : memref<3xi32>
      %30 = arith.addi %29, %c1_i32 : i32
      %31 = arith.cmpi sge, %30, %c1_i32 : i32
      %32 = arith.select %31, %29, %30 : i32
      memref.store %32, %_anonymous9[%c2] : memref<3xi32>
      %33 = arith.addi %6, %c1 : index
      cf.br ^bb3(%33 : index)
    ^bb16:  // pred: ^bb3
      %34 = arith.addi %0, %c1 : index
      cf.br ^bb1(%34 : index)
    ^bb17:  // pred: ^bb1
      aie.end
    } {link_files = ["gemm_64x64x64_0_0_1_0_1.o", "convert_copy.o"], stack_size = 3328 : i32}
    %lock_2_4 = aie.lock(%tile_2_4, 0)
    %_anonymous10 = aie.buffer(%tile_2_4) {address = 11520 : i32, mem_bank = 0 : i32, sym_name = "_anonymous10"} : memref<3xi32> 
    %core_2_4 = aie.core(%tile_2_4) {
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c4096_i32 = arith.constant 4096 : i32
      %c1_i32 = arith.constant 1 : i32
      %c2 = arith.constant 2 : index
      %c1 = arith.constant 1 : index
      %c0_i32 = arith.constant 0 : i32
      %c0 = arith.constant 0 : index
      %c2_i32 = arith.constant 2 : i32
      memref.store %c0_i32, %_anonymous10[%c0] : memref<3xi32>
      memref.store %c0_i32, %_anonymous10[%c1] : memref<3xi32>
      memref.store %c0_i32, %_anonymous10[%c2] : memref<3xi32>
      cf.br ^bb1(%c0 : index)
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb16
      %1 = arith.cmpi slt, %0, %c9223372036854775807 : index
      cf.cond_br %1, ^bb2, ^bb17
    ^bb2:  // pred: ^bb1
      aie.use_lock(%lock_2_4, Acquire, 1)
      %2 = memref.load %rtp2_2[%c0] : memref<2xi32>
      %3 = memref.load %rtp2_2[%c1] : memref<2xi32>
      %4 = arith.index_cast %3 : i32 to index
      %5 = arith.index_cast %2 : i32 to index
      cf.br ^bb3(%c0 : index)
    ^bb3(%6: index):  // 2 preds: ^bb2, ^bb15
      %7 = arith.cmpi slt, %6, %4 : index
      cf.cond_br %7, ^bb4, ^bb16
    ^bb4:  // pred: ^bb3
      func.call @zero_f32(%acc_buffer_2_2) : (memref<64x64xf32>) -> ()
      cf.br ^bb5(%c0 : index)
    ^bb5(%8: index):  // 2 preds: ^bb4, ^bb14
      %9 = arith.cmpi slt, %8, %5 : index
      cf.cond_br %9, ^bb6, ^bb15
    ^bb6:  // pred: ^bb5
      aie.use_lock(%A_L2L1_2_2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %10 = memref.load %_anonymous10[%c0] : memref<3xi32>
      %11 = arith.index_cast %10 : i32 to index
      %12 = arith.index_cast %11 : index to i32
      cf.switch %12 : i32, [
        default: ^bb9,
        0: ^bb7,
        1: ^bb8
      ]
    ^bb7:  // pred: ^bb6
      cf.br ^bb10(%A_L2L1_2_2_cons_buff_0 : memref<64x64xbf16>)
    ^bb8:  // pred: ^bb6
      cf.br ^bb10(%A_L2L1_2_2_cons_buff_1 : memref<64x64xbf16>)
    ^bb9:  // pred: ^bb6
      cf.br ^bb10(%A_L2L1_2_2_cons_buff_0 : memref<64x64xbf16>)
    ^bb10(%13: memref<64x64xbf16>):  // 3 preds: ^bb7, ^bb8, ^bb9
      aie.use_lock(%B_L2L1_2_2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %14 = memref.load %_anonymous10[%c1] : memref<3xi32>
      %15 = arith.index_cast %14 : i32 to index
      %16 = arith.index_cast %15 : index to i32
      cf.switch %16 : i32, [
        default: ^bb13,
        0: ^bb11,
        1: ^bb12
      ]
    ^bb11:  // pred: ^bb10
      cf.br ^bb14(%B_L2L1_2_2_cons_buff_0 : memref<64x64xbf16>)
    ^bb12:  // pred: ^bb10
      cf.br ^bb14(%B_L2L1_2_2_cons_buff_1 : memref<64x64xbf16>)
    ^bb13:  // pred: ^bb10
      cf.br ^bb14(%B_L2L1_2_2_cons_buff_0 : memref<64x64xbf16>)
    ^bb14(%17: memref<64x64xbf16>):  // 3 preds: ^bb11, ^bb12, ^bb13
      func.call @matmul_bf16_f32(%13, %17, %acc_buffer_2_2) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xf32>) -> ()
      aie.use_lock(%A_L2L1_2_2_cons_prod_lock_0, Release, 1)
      %18 = memref.load %_anonymous10[%c0] : memref<3xi32>
      %19 = arith.addi %18, %c1_i32 : i32
      %20 = arith.cmpi sge, %19, %c2_i32 : i32
      %21 = arith.subi %19, %c2_i32 : i32
      %22 = arith.select %20, %21, %19 : i32
      memref.store %22, %_anonymous10[%c0] : memref<3xi32>
      aie.use_lock(%B_L2L1_2_2_cons_prod_lock_0, Release, 1)
      %23 = memref.load %_anonymous10[%c1] : memref<3xi32>
      %24 = arith.addi %23, %c1_i32 : i32
      %25 = arith.cmpi sge, %24, %c2_i32 : i32
      %26 = arith.subi %24, %c2_i32 : i32
      %27 = arith.select %25, %26, %24 : i32
      memref.store %27, %_anonymous10[%c1] : memref<3xi32>
      %28 = arith.addi %8, %c1 : index
      cf.br ^bb5(%28 : index)
    ^bb15:  // pred: ^bb5
      aie.use_lock(%C_L1L2_2_2_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @convert_copy_f32_to_bf16(%acc_buffer_2_2, %C_L1L2_2_2_buff_0, %c4096_i32) : (memref<64x64xf32>, memref<64x64xbf16>, i32) -> ()
      aie.use_lock(%C_L1L2_2_2_cons_lock_0, Release, 1)
      %29 = memref.load %_anonymous10[%c2] : memref<3xi32>
      %30 = arith.addi %29, %c1_i32 : i32
      %31 = arith.cmpi sge, %30, %c1_i32 : i32
      %32 = arith.select %31, %29, %30 : i32
      memref.store %32, %_anonymous10[%c2] : memref<3xi32>
      %33 = arith.addi %6, %c1 : index
      cf.br ^bb3(%33 : index)
    ^bb16:  // pred: ^bb3
      %34 = arith.addi %0, %c1 : index
      cf.br ^bb1(%34 : index)
    ^bb17:  // pred: ^bb1
      aie.end
    } {link_files = ["gemm_64x64x64_0_0_1_0_1.o", "convert_copy.o"], stack_size = 3328 : i32}
    %lock_3_4 = aie.lock(%tile_3_4, 0)
    %_anonymous11 = aie.buffer(%tile_3_4) {address = 11520 : i32, mem_bank = 0 : i32, sym_name = "_anonymous11"} : memref<3xi32> 
    %core_3_4 = aie.core(%tile_3_4) {
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c4096_i32 = arith.constant 4096 : i32
      %c1_i32 = arith.constant 1 : i32
      %c2 = arith.constant 2 : index
      %c1 = arith.constant 1 : index
      %c0_i32 = arith.constant 0 : i32
      %c0 = arith.constant 0 : index
      %c2_i32 = arith.constant 2 : i32
      memref.store %c0_i32, %_anonymous11[%c0] : memref<3xi32>
      memref.store %c0_i32, %_anonymous11[%c1] : memref<3xi32>
      memref.store %c0_i32, %_anonymous11[%c2] : memref<3xi32>
      cf.br ^bb1(%c0 : index)
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb16
      %1 = arith.cmpi slt, %0, %c9223372036854775807 : index
      cf.cond_br %1, ^bb2, ^bb17
    ^bb2:  // pred: ^bb1
      aie.use_lock(%lock_3_4, Acquire, 1)
      %2 = memref.load %rtp2_3[%c0] : memref<2xi32>
      %3 = memref.load %rtp2_3[%c1] : memref<2xi32>
      %4 = arith.index_cast %3 : i32 to index
      %5 = arith.index_cast %2 : i32 to index
      cf.br ^bb3(%c0 : index)
    ^bb3(%6: index):  // 2 preds: ^bb2, ^bb15
      %7 = arith.cmpi slt, %6, %4 : index
      cf.cond_br %7, ^bb4, ^bb16
    ^bb4:  // pred: ^bb3
      func.call @zero_f32(%acc_buffer_2_3) : (memref<64x64xf32>) -> ()
      cf.br ^bb5(%c0 : index)
    ^bb5(%8: index):  // 2 preds: ^bb4, ^bb14
      %9 = arith.cmpi slt, %8, %5 : index
      cf.cond_br %9, ^bb6, ^bb15
    ^bb6:  // pred: ^bb5
      aie.use_lock(%A_L2L1_2_3_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %10 = memref.load %_anonymous11[%c0] : memref<3xi32>
      %11 = arith.index_cast %10 : i32 to index
      %12 = arith.index_cast %11 : index to i32
      cf.switch %12 : i32, [
        default: ^bb9,
        0: ^bb7,
        1: ^bb8
      ]
    ^bb7:  // pred: ^bb6
      cf.br ^bb10(%A_L2L1_2_3_cons_buff_0 : memref<64x64xbf16>)
    ^bb8:  // pred: ^bb6
      cf.br ^bb10(%A_L2L1_2_3_cons_buff_1 : memref<64x64xbf16>)
    ^bb9:  // pred: ^bb6
      cf.br ^bb10(%A_L2L1_2_3_cons_buff_0 : memref<64x64xbf16>)
    ^bb10(%13: memref<64x64xbf16>):  // 3 preds: ^bb7, ^bb8, ^bb9
      aie.use_lock(%B_L2L1_3_2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %14 = memref.load %_anonymous11[%c1] : memref<3xi32>
      %15 = arith.index_cast %14 : i32 to index
      %16 = arith.index_cast %15 : index to i32
      cf.switch %16 : i32, [
        default: ^bb13,
        0: ^bb11,
        1: ^bb12
      ]
    ^bb11:  // pred: ^bb10
      cf.br ^bb14(%B_L2L1_3_2_cons_buff_0 : memref<64x64xbf16>)
    ^bb12:  // pred: ^bb10
      cf.br ^bb14(%B_L2L1_3_2_cons_buff_1 : memref<64x64xbf16>)
    ^bb13:  // pred: ^bb10
      cf.br ^bb14(%B_L2L1_3_2_cons_buff_0 : memref<64x64xbf16>)
    ^bb14(%17: memref<64x64xbf16>):  // 3 preds: ^bb11, ^bb12, ^bb13
      func.call @matmul_bf16_f32(%13, %17, %acc_buffer_2_3) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xf32>) -> ()
      aie.use_lock(%A_L2L1_2_3_cons_prod_lock_0, Release, 1)
      %18 = memref.load %_anonymous11[%c0] : memref<3xi32>
      %19 = arith.addi %18, %c1_i32 : i32
      %20 = arith.cmpi sge, %19, %c2_i32 : i32
      %21 = arith.subi %19, %c2_i32 : i32
      %22 = arith.select %20, %21, %19 : i32
      memref.store %22, %_anonymous11[%c0] : memref<3xi32>
      aie.use_lock(%B_L2L1_3_2_cons_prod_lock_0, Release, 1)
      %23 = memref.load %_anonymous11[%c1] : memref<3xi32>
      %24 = arith.addi %23, %c1_i32 : i32
      %25 = arith.cmpi sge, %24, %c2_i32 : i32
      %26 = arith.subi %24, %c2_i32 : i32
      %27 = arith.select %25, %26, %24 : i32
      memref.store %27, %_anonymous11[%c1] : memref<3xi32>
      %28 = arith.addi %8, %c1 : index
      cf.br ^bb5(%28 : index)
    ^bb15:  // pred: ^bb5
      aie.use_lock(%C_L1L2_3_2_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @convert_copy_f32_to_bf16(%acc_buffer_2_3, %C_L1L2_3_2_buff_0, %c4096_i32) : (memref<64x64xf32>, memref<64x64xbf16>, i32) -> ()
      aie.use_lock(%C_L1L2_3_2_cons_lock_0, Release, 1)
      %29 = memref.load %_anonymous11[%c2] : memref<3xi32>
      %30 = arith.addi %29, %c1_i32 : i32
      %31 = arith.cmpi sge, %30, %c1_i32 : i32
      %32 = arith.select %31, %29, %30 : i32
      memref.store %32, %_anonymous11[%c2] : memref<3xi32>
      %33 = arith.addi %6, %c1 : index
      cf.br ^bb3(%33 : index)
    ^bb16:  // pred: ^bb3
      %34 = arith.addi %0, %c1 : index
      cf.br ^bb1(%34 : index)
    ^bb17:  // pred: ^bb1
      aie.end
    } {link_files = ["gemm_64x64x64_0_0_1_0_1.o", "convert_copy.o"], stack_size = 3328 : i32}
    %lock_0_5 = aie.lock(%tile_0_5, 0)
    %_anonymous12 = aie.buffer(%tile_0_5) {address = 11520 : i32, mem_bank = 0 : i32, sym_name = "_anonymous12"} : memref<3xi32> 
    %core_0_5 = aie.core(%tile_0_5) {
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c4096_i32 = arith.constant 4096 : i32
      %c1_i32 = arith.constant 1 : i32
      %c2 = arith.constant 2 : index
      %c1 = arith.constant 1 : index
      %c0_i32 = arith.constant 0 : i32
      %c0 = arith.constant 0 : index
      %c2_i32 = arith.constant 2 : i32
      memref.store %c0_i32, %_anonymous12[%c0] : memref<3xi32>
      memref.store %c0_i32, %_anonymous12[%c1] : memref<3xi32>
      memref.store %c0_i32, %_anonymous12[%c2] : memref<3xi32>
      cf.br ^bb1(%c0 : index)
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb16
      %1 = arith.cmpi slt, %0, %c9223372036854775807 : index
      cf.cond_br %1, ^bb2, ^bb17
    ^bb2:  // pred: ^bb1
      aie.use_lock(%lock_0_5, Acquire, 1)
      %2 = memref.load %rtp3_0[%c0] : memref<2xi32>
      %3 = memref.load %rtp3_0[%c1] : memref<2xi32>
      %4 = arith.index_cast %3 : i32 to index
      %5 = arith.index_cast %2 : i32 to index
      cf.br ^bb3(%c0 : index)
    ^bb3(%6: index):  // 2 preds: ^bb2, ^bb15
      %7 = arith.cmpi slt, %6, %4 : index
      cf.cond_br %7, ^bb4, ^bb16
    ^bb4:  // pred: ^bb3
      func.call @zero_f32(%acc_buffer_3_0) : (memref<64x64xf32>) -> ()
      cf.br ^bb5(%c0 : index)
    ^bb5(%8: index):  // 2 preds: ^bb4, ^bb14
      %9 = arith.cmpi slt, %8, %5 : index
      cf.cond_br %9, ^bb6, ^bb15
    ^bb6:  // pred: ^bb5
      aie.use_lock(%A_L2L1_3_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %10 = memref.load %_anonymous12[%c0] : memref<3xi32>
      %11 = arith.index_cast %10 : i32 to index
      %12 = arith.index_cast %11 : index to i32
      cf.switch %12 : i32, [
        default: ^bb9,
        0: ^bb7,
        1: ^bb8
      ]
    ^bb7:  // pred: ^bb6
      cf.br ^bb10(%A_L2L1_3_0_cons_buff_0 : memref<64x64xbf16>)
    ^bb8:  // pred: ^bb6
      cf.br ^bb10(%A_L2L1_3_0_cons_buff_1 : memref<64x64xbf16>)
    ^bb9:  // pred: ^bb6
      cf.br ^bb10(%A_L2L1_3_0_cons_buff_0 : memref<64x64xbf16>)
    ^bb10(%13: memref<64x64xbf16>):  // 3 preds: ^bb7, ^bb8, ^bb9
      aie.use_lock(%B_L2L1_0_3_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %14 = memref.load %_anonymous12[%c1] : memref<3xi32>
      %15 = arith.index_cast %14 : i32 to index
      %16 = arith.index_cast %15 : index to i32
      cf.switch %16 : i32, [
        default: ^bb13,
        0: ^bb11,
        1: ^bb12
      ]
    ^bb11:  // pred: ^bb10
      cf.br ^bb14(%B_L2L1_0_3_cons_buff_0 : memref<64x64xbf16>)
    ^bb12:  // pred: ^bb10
      cf.br ^bb14(%B_L2L1_0_3_cons_buff_1 : memref<64x64xbf16>)
    ^bb13:  // pred: ^bb10
      cf.br ^bb14(%B_L2L1_0_3_cons_buff_0 : memref<64x64xbf16>)
    ^bb14(%17: memref<64x64xbf16>):  // 3 preds: ^bb11, ^bb12, ^bb13
      func.call @matmul_bf16_f32(%13, %17, %acc_buffer_3_0) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xf32>) -> ()
      aie.use_lock(%A_L2L1_3_0_cons_prod_lock_0, Release, 1)
      %18 = memref.load %_anonymous12[%c0] : memref<3xi32>
      %19 = arith.addi %18, %c1_i32 : i32
      %20 = arith.cmpi sge, %19, %c2_i32 : i32
      %21 = arith.subi %19, %c2_i32 : i32
      %22 = arith.select %20, %21, %19 : i32
      memref.store %22, %_anonymous12[%c0] : memref<3xi32>
      aie.use_lock(%B_L2L1_0_3_cons_prod_lock_0, Release, 1)
      %23 = memref.load %_anonymous12[%c1] : memref<3xi32>
      %24 = arith.addi %23, %c1_i32 : i32
      %25 = arith.cmpi sge, %24, %c2_i32 : i32
      %26 = arith.subi %24, %c2_i32 : i32
      %27 = arith.select %25, %26, %24 : i32
      memref.store %27, %_anonymous12[%c1] : memref<3xi32>
      %28 = arith.addi %8, %c1 : index
      cf.br ^bb5(%28 : index)
    ^bb15:  // pred: ^bb5
      aie.use_lock(%C_L1L2_0_3_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @convert_copy_f32_to_bf16(%acc_buffer_3_0, %C_L1L2_0_3_buff_0, %c4096_i32) : (memref<64x64xf32>, memref<64x64xbf16>, i32) -> ()
      aie.use_lock(%C_L1L2_0_3_cons_lock_0, Release, 1)
      %29 = memref.load %_anonymous12[%c2] : memref<3xi32>
      %30 = arith.addi %29, %c1_i32 : i32
      %31 = arith.cmpi sge, %30, %c1_i32 : i32
      %32 = arith.select %31, %29, %30 : i32
      memref.store %32, %_anonymous12[%c2] : memref<3xi32>
      %33 = arith.addi %6, %c1 : index
      cf.br ^bb3(%33 : index)
    ^bb16:  // pred: ^bb3
      %34 = arith.addi %0, %c1 : index
      cf.br ^bb1(%34 : index)
    ^bb17:  // pred: ^bb1
      aie.end
    } {link_files = ["gemm_64x64x64_0_0_1_0_1.o", "convert_copy.o"], stack_size = 3328 : i32}
    %lock_1_5 = aie.lock(%tile_1_5, 0)
    %_anonymous13 = aie.buffer(%tile_1_5) {address = 11520 : i32, mem_bank = 0 : i32, sym_name = "_anonymous13"} : memref<3xi32> 
    %core_1_5 = aie.core(%tile_1_5) {
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c4096_i32 = arith.constant 4096 : i32
      %c1_i32 = arith.constant 1 : i32
      %c2 = arith.constant 2 : index
      %c1 = arith.constant 1 : index
      %c0_i32 = arith.constant 0 : i32
      %c0 = arith.constant 0 : index
      %c2_i32 = arith.constant 2 : i32
      memref.store %c0_i32, %_anonymous13[%c0] : memref<3xi32>
      memref.store %c0_i32, %_anonymous13[%c1] : memref<3xi32>
      memref.store %c0_i32, %_anonymous13[%c2] : memref<3xi32>
      cf.br ^bb1(%c0 : index)
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb16
      %1 = arith.cmpi slt, %0, %c9223372036854775807 : index
      cf.cond_br %1, ^bb2, ^bb17
    ^bb2:  // pred: ^bb1
      aie.use_lock(%lock_1_5, Acquire, 1)
      %2 = memref.load %rtp3_1[%c0] : memref<2xi32>
      %3 = memref.load %rtp3_1[%c1] : memref<2xi32>
      %4 = arith.index_cast %3 : i32 to index
      %5 = arith.index_cast %2 : i32 to index
      cf.br ^bb3(%c0 : index)
    ^bb3(%6: index):  // 2 preds: ^bb2, ^bb15
      %7 = arith.cmpi slt, %6, %4 : index
      cf.cond_br %7, ^bb4, ^bb16
    ^bb4:  // pred: ^bb3
      func.call @zero_f32(%acc_buffer_3_1) : (memref<64x64xf32>) -> ()
      cf.br ^bb5(%c0 : index)
    ^bb5(%8: index):  // 2 preds: ^bb4, ^bb14
      %9 = arith.cmpi slt, %8, %5 : index
      cf.cond_br %9, ^bb6, ^bb15
    ^bb6:  // pred: ^bb5
      aie.use_lock(%A_L2L1_3_1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %10 = memref.load %_anonymous13[%c0] : memref<3xi32>
      %11 = arith.index_cast %10 : i32 to index
      %12 = arith.index_cast %11 : index to i32
      cf.switch %12 : i32, [
        default: ^bb9,
        0: ^bb7,
        1: ^bb8
      ]
    ^bb7:  // pred: ^bb6
      cf.br ^bb10(%A_L2L1_3_1_cons_buff_0 : memref<64x64xbf16>)
    ^bb8:  // pred: ^bb6
      cf.br ^bb10(%A_L2L1_3_1_cons_buff_1 : memref<64x64xbf16>)
    ^bb9:  // pred: ^bb6
      cf.br ^bb10(%A_L2L1_3_1_cons_buff_0 : memref<64x64xbf16>)
    ^bb10(%13: memref<64x64xbf16>):  // 3 preds: ^bb7, ^bb8, ^bb9
      aie.use_lock(%B_L2L1_1_3_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %14 = memref.load %_anonymous13[%c1] : memref<3xi32>
      %15 = arith.index_cast %14 : i32 to index
      %16 = arith.index_cast %15 : index to i32
      cf.switch %16 : i32, [
        default: ^bb13,
        0: ^bb11,
        1: ^bb12
      ]
    ^bb11:  // pred: ^bb10
      cf.br ^bb14(%B_L2L1_1_3_cons_buff_0 : memref<64x64xbf16>)
    ^bb12:  // pred: ^bb10
      cf.br ^bb14(%B_L2L1_1_3_cons_buff_1 : memref<64x64xbf16>)
    ^bb13:  // pred: ^bb10
      cf.br ^bb14(%B_L2L1_1_3_cons_buff_0 : memref<64x64xbf16>)
    ^bb14(%17: memref<64x64xbf16>):  // 3 preds: ^bb11, ^bb12, ^bb13
      func.call @matmul_bf16_f32(%13, %17, %acc_buffer_3_1) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xf32>) -> ()
      aie.use_lock(%A_L2L1_3_1_cons_prod_lock_0, Release, 1)
      %18 = memref.load %_anonymous13[%c0] : memref<3xi32>
      %19 = arith.addi %18, %c1_i32 : i32
      %20 = arith.cmpi sge, %19, %c2_i32 : i32
      %21 = arith.subi %19, %c2_i32 : i32
      %22 = arith.select %20, %21, %19 : i32
      memref.store %22, %_anonymous13[%c0] : memref<3xi32>
      aie.use_lock(%B_L2L1_1_3_cons_prod_lock_0, Release, 1)
      %23 = memref.load %_anonymous13[%c1] : memref<3xi32>
      %24 = arith.addi %23, %c1_i32 : i32
      %25 = arith.cmpi sge, %24, %c2_i32 : i32
      %26 = arith.subi %24, %c2_i32 : i32
      %27 = arith.select %25, %26, %24 : i32
      memref.store %27, %_anonymous13[%c1] : memref<3xi32>
      %28 = arith.addi %8, %c1 : index
      cf.br ^bb5(%28 : index)
    ^bb15:  // pred: ^bb5
      aie.use_lock(%C_L1L2_1_3_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @convert_copy_f32_to_bf16(%acc_buffer_3_1, %C_L1L2_1_3_buff_0, %c4096_i32) : (memref<64x64xf32>, memref<64x64xbf16>, i32) -> ()
      aie.use_lock(%C_L1L2_1_3_cons_lock_0, Release, 1)
      %29 = memref.load %_anonymous13[%c2] : memref<3xi32>
      %30 = arith.addi %29, %c1_i32 : i32
      %31 = arith.cmpi sge, %30, %c1_i32 : i32
      %32 = arith.select %31, %29, %30 : i32
      memref.store %32, %_anonymous13[%c2] : memref<3xi32>
      %33 = arith.addi %6, %c1 : index
      cf.br ^bb3(%33 : index)
    ^bb16:  // pred: ^bb3
      %34 = arith.addi %0, %c1 : index
      cf.br ^bb1(%34 : index)
    ^bb17:  // pred: ^bb1
      aie.end
    } {link_files = ["gemm_64x64x64_0_0_1_0_1.o", "convert_copy.o"], stack_size = 3328 : i32}
    %lock_2_5 = aie.lock(%tile_2_5, 0)
    %_anonymous14 = aie.buffer(%tile_2_5) {address = 11520 : i32, mem_bank = 0 : i32, sym_name = "_anonymous14"} : memref<3xi32> 
    %core_2_5 = aie.core(%tile_2_5) {
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c4096_i32 = arith.constant 4096 : i32
      %c1_i32 = arith.constant 1 : i32
      %c2 = arith.constant 2 : index
      %c1 = arith.constant 1 : index
      %c0_i32 = arith.constant 0 : i32
      %c0 = arith.constant 0 : index
      %c2_i32 = arith.constant 2 : i32
      memref.store %c0_i32, %_anonymous14[%c0] : memref<3xi32>
      memref.store %c0_i32, %_anonymous14[%c1] : memref<3xi32>
      memref.store %c0_i32, %_anonymous14[%c2] : memref<3xi32>
      cf.br ^bb1(%c0 : index)
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb16
      %1 = arith.cmpi slt, %0, %c9223372036854775807 : index
      cf.cond_br %1, ^bb2, ^bb17
    ^bb2:  // pred: ^bb1
      aie.use_lock(%lock_2_5, Acquire, 1)
      %2 = memref.load %rtp3_2[%c0] : memref<2xi32>
      %3 = memref.load %rtp3_2[%c1] : memref<2xi32>
      %4 = arith.index_cast %3 : i32 to index
      %5 = arith.index_cast %2 : i32 to index
      cf.br ^bb3(%c0 : index)
    ^bb3(%6: index):  // 2 preds: ^bb2, ^bb15
      %7 = arith.cmpi slt, %6, %4 : index
      cf.cond_br %7, ^bb4, ^bb16
    ^bb4:  // pred: ^bb3
      func.call @zero_f32(%acc_buffer_3_2) : (memref<64x64xf32>) -> ()
      cf.br ^bb5(%c0 : index)
    ^bb5(%8: index):  // 2 preds: ^bb4, ^bb14
      %9 = arith.cmpi slt, %8, %5 : index
      cf.cond_br %9, ^bb6, ^bb15
    ^bb6:  // pred: ^bb5
      aie.use_lock(%A_L2L1_3_2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %10 = memref.load %_anonymous14[%c0] : memref<3xi32>
      %11 = arith.index_cast %10 : i32 to index
      %12 = arith.index_cast %11 : index to i32
      cf.switch %12 : i32, [
        default: ^bb9,
        0: ^bb7,
        1: ^bb8
      ]
    ^bb7:  // pred: ^bb6
      cf.br ^bb10(%A_L2L1_3_2_cons_buff_0 : memref<64x64xbf16>)
    ^bb8:  // pred: ^bb6
      cf.br ^bb10(%A_L2L1_3_2_cons_buff_1 : memref<64x64xbf16>)
    ^bb9:  // pred: ^bb6
      cf.br ^bb10(%A_L2L1_3_2_cons_buff_0 : memref<64x64xbf16>)
    ^bb10(%13: memref<64x64xbf16>):  // 3 preds: ^bb7, ^bb8, ^bb9
      aie.use_lock(%B_L2L1_2_3_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %14 = memref.load %_anonymous14[%c1] : memref<3xi32>
      %15 = arith.index_cast %14 : i32 to index
      %16 = arith.index_cast %15 : index to i32
      cf.switch %16 : i32, [
        default: ^bb13,
        0: ^bb11,
        1: ^bb12
      ]
    ^bb11:  // pred: ^bb10
      cf.br ^bb14(%B_L2L1_2_3_cons_buff_0 : memref<64x64xbf16>)
    ^bb12:  // pred: ^bb10
      cf.br ^bb14(%B_L2L1_2_3_cons_buff_1 : memref<64x64xbf16>)
    ^bb13:  // pred: ^bb10
      cf.br ^bb14(%B_L2L1_2_3_cons_buff_0 : memref<64x64xbf16>)
    ^bb14(%17: memref<64x64xbf16>):  // 3 preds: ^bb11, ^bb12, ^bb13
      func.call @matmul_bf16_f32(%13, %17, %acc_buffer_3_2) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xf32>) -> ()
      aie.use_lock(%A_L2L1_3_2_cons_prod_lock_0, Release, 1)
      %18 = memref.load %_anonymous14[%c0] : memref<3xi32>
      %19 = arith.addi %18, %c1_i32 : i32
      %20 = arith.cmpi sge, %19, %c2_i32 : i32
      %21 = arith.subi %19, %c2_i32 : i32
      %22 = arith.select %20, %21, %19 : i32
      memref.store %22, %_anonymous14[%c0] : memref<3xi32>
      aie.use_lock(%B_L2L1_2_3_cons_prod_lock_0, Release, 1)
      %23 = memref.load %_anonymous14[%c1] : memref<3xi32>
      %24 = arith.addi %23, %c1_i32 : i32
      %25 = arith.cmpi sge, %24, %c2_i32 : i32
      %26 = arith.subi %24, %c2_i32 : i32
      %27 = arith.select %25, %26, %24 : i32
      memref.store %27, %_anonymous14[%c1] : memref<3xi32>
      %28 = arith.addi %8, %c1 : index
      cf.br ^bb5(%28 : index)
    ^bb15:  // pred: ^bb5
      aie.use_lock(%C_L1L2_2_3_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @convert_copy_f32_to_bf16(%acc_buffer_3_2, %C_L1L2_2_3_buff_0, %c4096_i32) : (memref<64x64xf32>, memref<64x64xbf16>, i32) -> ()
      aie.use_lock(%C_L1L2_2_3_cons_lock_0, Release, 1)
      %29 = memref.load %_anonymous14[%c2] : memref<3xi32>
      %30 = arith.addi %29, %c1_i32 : i32
      %31 = arith.cmpi sge, %30, %c1_i32 : i32
      %32 = arith.select %31, %29, %30 : i32
      memref.store %32, %_anonymous14[%c2] : memref<3xi32>
      %33 = arith.addi %6, %c1 : index
      cf.br ^bb3(%33 : index)
    ^bb16:  // pred: ^bb3
      %34 = arith.addi %0, %c1 : index
      cf.br ^bb1(%34 : index)
    ^bb17:  // pred: ^bb1
      aie.end
    } {link_files = ["gemm_64x64x64_0_0_1_0_1.o", "convert_copy.o"], stack_size = 3328 : i32}
    %lock_3_5 = aie.lock(%tile_3_5, 0)
    %_anonymous15 = aie.buffer(%tile_3_5) {address = 11520 : i32, mem_bank = 0 : i32, sym_name = "_anonymous15"} : memref<3xi32> 
    %core_3_5 = aie.core(%tile_3_5) {
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c4096_i32 = arith.constant 4096 : i32
      %c1_i32 = arith.constant 1 : i32
      %c2 = arith.constant 2 : index
      %c1 = arith.constant 1 : index
      %c0_i32 = arith.constant 0 : i32
      %c0 = arith.constant 0 : index
      %c2_i32 = arith.constant 2 : i32
      memref.store %c0_i32, %_anonymous15[%c0] : memref<3xi32>
      memref.store %c0_i32, %_anonymous15[%c1] : memref<3xi32>
      memref.store %c0_i32, %_anonymous15[%c2] : memref<3xi32>
      cf.br ^bb1(%c0 : index)
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb16
      %1 = arith.cmpi slt, %0, %c9223372036854775807 : index
      cf.cond_br %1, ^bb2, ^bb17
    ^bb2:  // pred: ^bb1
      aie.use_lock(%lock_3_5, Acquire, 1)
      %2 = memref.load %rtp3_3[%c0] : memref<2xi32>
      %3 = memref.load %rtp3_3[%c1] : memref<2xi32>
      %4 = arith.index_cast %3 : i32 to index
      %5 = arith.index_cast %2 : i32 to index
      cf.br ^bb3(%c0 : index)
    ^bb3(%6: index):  // 2 preds: ^bb2, ^bb15
      %7 = arith.cmpi slt, %6, %4 : index
      cf.cond_br %7, ^bb4, ^bb16
    ^bb4:  // pred: ^bb3
      func.call @zero_f32(%acc_buffer_3_3) : (memref<64x64xf32>) -> ()
      cf.br ^bb5(%c0 : index)
    ^bb5(%8: index):  // 2 preds: ^bb4, ^bb14
      %9 = arith.cmpi slt, %8, %5 : index
      cf.cond_br %9, ^bb6, ^bb15
    ^bb6:  // pred: ^bb5
      aie.use_lock(%A_L2L1_3_3_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %10 = memref.load %_anonymous15[%c0] : memref<3xi32>
      %11 = arith.index_cast %10 : i32 to index
      %12 = arith.index_cast %11 : index to i32
      cf.switch %12 : i32, [
        default: ^bb9,
        0: ^bb7,
        1: ^bb8
      ]
    ^bb7:  // pred: ^bb6
      cf.br ^bb10(%A_L2L1_3_3_cons_buff_0 : memref<64x64xbf16>)
    ^bb8:  // pred: ^bb6
      cf.br ^bb10(%A_L2L1_3_3_cons_buff_1 : memref<64x64xbf16>)
    ^bb9:  // pred: ^bb6
      cf.br ^bb10(%A_L2L1_3_3_cons_buff_0 : memref<64x64xbf16>)
    ^bb10(%13: memref<64x64xbf16>):  // 3 preds: ^bb7, ^bb8, ^bb9
      aie.use_lock(%B_L2L1_3_3_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %14 = memref.load %_anonymous15[%c1] : memref<3xi32>
      %15 = arith.index_cast %14 : i32 to index
      %16 = arith.index_cast %15 : index to i32
      cf.switch %16 : i32, [
        default: ^bb13,
        0: ^bb11,
        1: ^bb12
      ]
    ^bb11:  // pred: ^bb10
      cf.br ^bb14(%B_L2L1_3_3_cons_buff_0 : memref<64x64xbf16>)
    ^bb12:  // pred: ^bb10
      cf.br ^bb14(%B_L2L1_3_3_cons_buff_1 : memref<64x64xbf16>)
    ^bb13:  // pred: ^bb10
      cf.br ^bb14(%B_L2L1_3_3_cons_buff_0 : memref<64x64xbf16>)
    ^bb14(%17: memref<64x64xbf16>):  // 3 preds: ^bb11, ^bb12, ^bb13
      func.call @matmul_bf16_f32(%13, %17, %acc_buffer_3_3) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xf32>) -> ()
      aie.use_lock(%A_L2L1_3_3_cons_prod_lock_0, Release, 1)
      %18 = memref.load %_anonymous15[%c0] : memref<3xi32>
      %19 = arith.addi %18, %c1_i32 : i32
      %20 = arith.cmpi sge, %19, %c2_i32 : i32
      %21 = arith.subi %19, %c2_i32 : i32
      %22 = arith.select %20, %21, %19 : i32
      memref.store %22, %_anonymous15[%c0] : memref<3xi32>
      aie.use_lock(%B_L2L1_3_3_cons_prod_lock_0, Release, 1)
      %23 = memref.load %_anonymous15[%c1] : memref<3xi32>
      %24 = arith.addi %23, %c1_i32 : i32
      %25 = arith.cmpi sge, %24, %c2_i32 : i32
      %26 = arith.subi %24, %c2_i32 : i32
      %27 = arith.select %25, %26, %24 : i32
      memref.store %27, %_anonymous15[%c1] : memref<3xi32>
      %28 = arith.addi %8, %c1 : index
      cf.br ^bb5(%28 : index)
    ^bb15:  // pred: ^bb5
      aie.use_lock(%C_L1L2_3_3_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @convert_copy_f32_to_bf16(%acc_buffer_3_3, %C_L1L2_3_3_buff_0, %c4096_i32) : (memref<64x64xf32>, memref<64x64xbf16>, i32) -> ()
      aie.use_lock(%C_L1L2_3_3_cons_lock_0, Release, 1)
      %29 = memref.load %_anonymous15[%c2] : memref<3xi32>
      %30 = arith.addi %29, %c1_i32 : i32
      %31 = arith.cmpi sge, %30, %c1_i32 : i32
      %32 = arith.select %31, %29, %30 : i32
      memref.store %32, %_anonymous15[%c2] : memref<3xi32>
      %33 = arith.addi %6, %c1 : index
      cf.br ^bb3(%33 : index)
    ^bb16:  // pred: ^bb3
      %34 = arith.addi %0, %c1 : index
      cf.br ^bb1(%34 : index)
    ^bb17:  // pred: ^bb1
      aie.end
    } {link_files = ["gemm_64x64x64_0_0_1_0_1.o", "convert_copy.o"], stack_size = 3328 : i32}
    aie.runtime_sequence(%arg0: memref<524288xbf16>, %arg1: memref<4194304xbf16>, %arg2: memref<524288xbf16>) {
      aiex.npu.rtp_write(@rtp0_0, 0, 32)
      aiex.npu.rtp_write(@rtp0_0, 1, 8)
      aiex.npu.rtp_write(@rtp0_1, 0, 32)
      aiex.npu.rtp_write(@rtp0_1, 1, 8)
      aiex.npu.rtp_write(@rtp0_2, 0, 32)
      aiex.npu.rtp_write(@rtp0_2, 1, 8)
      aiex.npu.rtp_write(@rtp0_3, 0, 32)
      aiex.npu.rtp_write(@rtp0_3, 1, 8)
      aiex.npu.rtp_write(@rtp1_0, 0, 32)
      aiex.npu.rtp_write(@rtp1_0, 1, 8)
      aiex.npu.rtp_write(@rtp1_1, 0, 32)
      aiex.npu.rtp_write(@rtp1_1, 1, 8)
      aiex.npu.rtp_write(@rtp1_2, 0, 32)
      aiex.npu.rtp_write(@rtp1_2, 1, 8)
      aiex.npu.rtp_write(@rtp1_3, 0, 32)
      aiex.npu.rtp_write(@rtp1_3, 1, 8)
      aiex.npu.rtp_write(@rtp2_0, 0, 32)
      aiex.npu.rtp_write(@rtp2_0, 1, 8)
      aiex.npu.rtp_write(@rtp2_1, 0, 32)
      aiex.npu.rtp_write(@rtp2_1, 1, 8)
      aiex.npu.rtp_write(@rtp2_2, 0, 32)
      aiex.npu.rtp_write(@rtp2_2, 1, 8)
      aiex.npu.rtp_write(@rtp2_3, 0, 32)
      aiex.npu.rtp_write(@rtp2_3, 1, 8)
      aiex.npu.rtp_write(@rtp3_0, 0, 32)
      aiex.npu.rtp_write(@rtp3_0, 1, 8)
      aiex.npu.rtp_write(@rtp3_1, 0, 32)
      aiex.npu.rtp_write(@rtp3_1, 1, 8)
      aiex.npu.rtp_write(@rtp3_2, 0, 32)
      aiex.npu.rtp_write(@rtp3_2, 1, 8)
      aiex.npu.rtp_write(@rtp3_3, 0, 32)
      aiex.npu.rtp_write(@rtp3_3, 1, 8)
      aiex.set_lock(%lock_0_2, 1)
      aiex.set_lock(%lock_1_2, 1)
      aiex.set_lock(%lock_2_2, 1)
      aiex.set_lock(%lock_3_2, 1)
      aiex.set_lock(%lock_0_3, 1)
      aiex.set_lock(%lock_1_3, 1)
      aiex.set_lock(%lock_2_3, 1)
      aiex.set_lock(%lock_3_3, 1)
      aiex.set_lock(%lock_0_4, 1)
      aiex.set_lock(%lock_1_4, 1)
      aiex.set_lock(%lock_2_4, 1)
      aiex.set_lock(%lock_3_4, 1)
      aiex.set_lock(%lock_0_5, 1)
      aiex.set_lock(%lock_1_5, 1)
      aiex.set_lock(%lock_2_5, 1)
      aiex.set_lock(%lock_3_5, 1)
      %0 = aiex.dma_configure_task_for @C_L2L3_0_shim_alloc {
        aie.dma_bd(%arg2 : memref<524288xbf16>, 0, 131072, [<size = 1, stride = 0>, <size = 8, stride = 256>, <size = 256, stride = 2048>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%0)
      %1 = aiex.dma_configure_task_for @A_L3L2_0_shim_alloc {
        aie.dma_bd(%arg0 : memref<524288xbf16>, 0, 131072, [<size = 8, stride = 0>, <size = 32, stride = 64>, <size = 64, stride = 2048>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%1)
      %2 = aiex.dma_configure_task_for @B_L3L2_0_shim_alloc {
        aie.dma_bd(%arg1 : memref<4194304xbf16>, 0, 131072, [<size = 8, stride = 256>, <size = 32, stride = 131072>, <size = 64, stride = 2048>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%2)
      %3 = aiex.dma_configure_task_for @C_L2L3_1_shim_alloc {
        aie.dma_bd(%arg2 : memref<524288xbf16>, 64, 131072, [<size = 1, stride = 0>, <size = 8, stride = 256>, <size = 256, stride = 2048>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%3)
      %4 = aiex.dma_configure_task_for @A_L3L2_1_shim_alloc {
        aie.dma_bd(%arg0 : memref<524288xbf16>, 131072, 131072, [<size = 8, stride = 0>, <size = 32, stride = 64>, <size = 64, stride = 2048>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%4)
      %5 = aiex.dma_configure_task_for @B_L3L2_1_shim_alloc {
        aie.dma_bd(%arg1 : memref<4194304xbf16>, 64, 131072, [<size = 8, stride = 256>, <size = 32, stride = 131072>, <size = 64, stride = 2048>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%5)
      %6 = aiex.dma_configure_task_for @C_L2L3_2_shim_alloc {
        aie.dma_bd(%arg2 : memref<524288xbf16>, 128, 131072, [<size = 1, stride = 0>, <size = 8, stride = 256>, <size = 256, stride = 2048>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%6)
      %7 = aiex.dma_configure_task_for @A_L3L2_2_shim_alloc {
        aie.dma_bd(%arg0 : memref<524288xbf16>, 262144, 131072, [<size = 8, stride = 0>, <size = 32, stride = 64>, <size = 64, stride = 2048>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%7)
      %8 = aiex.dma_configure_task_for @B_L3L2_2_shim_alloc {
        aie.dma_bd(%arg1 : memref<4194304xbf16>, 128, 131072, [<size = 8, stride = 256>, <size = 32, stride = 131072>, <size = 64, stride = 2048>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%8)
      %9 = aiex.dma_configure_task_for @C_L2L3_3_shim_alloc {
        aie.dma_bd(%arg2 : memref<524288xbf16>, 192, 131072, [<size = 1, stride = 0>, <size = 8, stride = 256>, <size = 256, stride = 2048>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%9)
      %10 = aiex.dma_configure_task_for @A_L3L2_3_shim_alloc {
        aie.dma_bd(%arg0 : memref<524288xbf16>, 393216, 131072, [<size = 8, stride = 0>, <size = 32, stride = 64>, <size = 64, stride = 2048>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%10)
      %11 = aiex.dma_configure_task_for @B_L3L2_3_shim_alloc {
        aie.dma_bd(%arg1 : memref<4194304xbf16>, 192, 131072, [<size = 8, stride = 256>, <size = 32, stride = 131072>, <size = 64, stride = 2048>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%11)
      aiex.dma_await_task(%0)
      aiex.dma_await_task(%3)
      aiex.dma_await_task(%6)
      aiex.dma_await_task(%9)
      aiex.dma_free_task(%1)
      aiex.dma_free_task(%2)
      aiex.dma_free_task(%4)
      aiex.dma_free_task(%5)
      aiex.dma_free_task(%7)
      aiex.dma_free_task(%8)
      aiex.dma_free_task(%10)
      aiex.dma_free_task(%11)
    }
    %memtile_dma_0_1 = aie.memtile_dma(%mem_tile_0_1) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%A_L3L2_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L3L2_0_cons_buff_0 : memref<4096xbf16>, 0, 4096, [<size = 16, stride = 256>, <size = 8, stride = 8>, <size = 4, stride = 64>, <size = 8, stride = 1>]) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%A_L3L2_0_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%A_L3L2_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L3L2_0_cons_buff_1 : memref<4096xbf16>, 0, 4096, [<size = 16, stride = 256>, <size = 8, stride = 8>, <size = 4, stride = 64>, <size = 8, stride = 1>]) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%A_L3L2_0_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 0, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%A_L3L2_0_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L3L2_0_cons_buff_0 : memref<4096xbf16>, 0, 4096) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%A_L3L2_0_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%A_L3L2_0_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L3L2_0_cons_buff_1 : memref<4096xbf16>, 0, 4096) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%A_L3L2_0_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      %2 = aie.dma_start(MM2S, 1, ^bb7, ^bb9)
    ^bb7:  // 2 preds: ^bb6, ^bb8
      aie.use_lock(%B_L3L2_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L3L2_0_cons_buff_0 : memref<4096xbf16>, 0, 4096, [<size = 8, stride = 512>, <size = 8, stride = 8>, <size = 8, stride = 64>, <size = 8, stride = 1>]) {bd_id = 24 : i32, next_bd_id = 25 : i32}
      aie.use_lock(%B_L3L2_0_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb8
    ^bb8:  // pred: ^bb7
      aie.use_lock(%B_L3L2_0_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L3L2_0_cons_buff_1 : memref<4096xbf16>, 0, 4096, [<size = 8, stride = 512>, <size = 8, stride = 8>, <size = 8, stride = 64>, <size = 8, stride = 1>]) {bd_id = 25 : i32, next_bd_id = 24 : i32}
      aie.use_lock(%B_L3L2_0_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb7
    ^bb9:  // pred: ^bb6
      %3 = aie.dma_start(S2MM, 1, ^bb10, ^bb12)
    ^bb10:  // 2 preds: ^bb9, ^bb11
      aie.use_lock(%B_L3L2_0_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L3L2_0_cons_buff_0 : memref<4096xbf16>, 0, 4096) {bd_id = 26 : i32, next_bd_id = 27 : i32}
      aie.use_lock(%B_L3L2_0_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb11
    ^bb11:  // pred: ^bb10
      aie.use_lock(%B_L3L2_0_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L3L2_0_cons_buff_1 : memref<4096xbf16>, 0, 4096) {bd_id = 27 : i32, next_bd_id = 26 : i32}
      aie.use_lock(%B_L3L2_0_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb10
    ^bb12:  // pred: ^bb9
      %4 = aie.dma_start(S2MM, 2, ^bb13, ^bb15)
    ^bb13:  // 2 preds: ^bb12, ^bb14
      aie.use_lock(%C_L2L3_0_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_0_buff_0 : memref<16384xbf16>, 0, 4096) {bd_id = 4 : i32, next_bd_id = 5 : i32}
      aie.use_lock(%C_L2L3_0_cons_lock_0, Release, 1)
      aie.next_bd ^bb14
    ^bb14:  // pred: ^bb13
      aie.use_lock(%C_L2L3_0_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_0_buff_1 : memref<16384xbf16>, 0, 4096) {bd_id = 5 : i32, next_bd_id = 4 : i32}
      aie.use_lock(%C_L2L3_0_cons_lock_0, Release, 1)
      aie.next_bd ^bb13
    ^bb15:  // pred: ^bb12
      %5 = aie.dma_start(S2MM, 3, ^bb16, ^bb18)
    ^bb16:  // 2 preds: ^bb15, ^bb17
      aie.use_lock(%C_L2L3_0_prod_lock_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_0_buff_0 : memref<16384xbf16>, 4096, 4096) {bd_id = 28 : i32, next_bd_id = 29 : i32}
      aie.use_lock(%C_L2L3_0_cons_lock_1, Release, 1)
      aie.next_bd ^bb17
    ^bb17:  // pred: ^bb16
      aie.use_lock(%C_L2L3_0_prod_lock_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_0_buff_1 : memref<16384xbf16>, 4096, 4096) {bd_id = 29 : i32, next_bd_id = 28 : i32}
      aie.use_lock(%C_L2L3_0_cons_lock_1, Release, 1)
      aie.next_bd ^bb16
    ^bb18:  // pred: ^bb15
      %6 = aie.dma_start(S2MM, 4, ^bb19, ^bb21)
    ^bb19:  // 2 preds: ^bb18, ^bb20
      aie.use_lock(%C_L2L3_0_prod_lock_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_0_buff_0 : memref<16384xbf16>, 8192, 4096) {bd_id = 6 : i32, next_bd_id = 7 : i32}
      aie.use_lock(%C_L2L3_0_cons_lock_2, Release, 1)
      aie.next_bd ^bb20
    ^bb20:  // pred: ^bb19
      aie.use_lock(%C_L2L3_0_prod_lock_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_0_buff_1 : memref<16384xbf16>, 8192, 4096) {bd_id = 7 : i32, next_bd_id = 6 : i32}
      aie.use_lock(%C_L2L3_0_cons_lock_2, Release, 1)
      aie.next_bd ^bb19
    ^bb21:  // pred: ^bb18
      %7 = aie.dma_start(S2MM, 5, ^bb22, ^bb24)
    ^bb22:  // 2 preds: ^bb21, ^bb23
      aie.use_lock(%C_L2L3_0_prod_lock_3, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_0_buff_0 : memref<16384xbf16>, 12288, 4096) {bd_id = 30 : i32, next_bd_id = 31 : i32}
      aie.use_lock(%C_L2L3_0_cons_lock_3, Release, 1)
      aie.next_bd ^bb23
    ^bb23:  // pred: ^bb22
      aie.use_lock(%C_L2L3_0_prod_lock_3, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_0_buff_1 : memref<16384xbf16>, 12288, 4096) {bd_id = 31 : i32, next_bd_id = 30 : i32}
      aie.use_lock(%C_L2L3_0_cons_lock_3, Release, 1)
      aie.next_bd ^bb22
    ^bb24:  // pred: ^bb21
      %8 = aie.dma_start(MM2S, 2, ^bb25, ^bb33)
    ^bb25:  // 2 preds: ^bb24, ^bb32
      aie.use_lock(%C_L2L3_0_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_0_buff_0 : memref<16384xbf16>, 0, 4096, [<size = 16, stride = 256>, <size = 4, stride = 8>, <size = 8, stride = 32>, <size = 8, stride = 1>]) {bd_id = 8 : i32, next_bd_id = 9 : i32}
      aie.use_lock(%C_L2L3_0_prod_lock_0, Release, 1)
      aie.next_bd ^bb26
    ^bb26:  // pred: ^bb25
      aie.use_lock(%C_L2L3_0_cons_lock_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_0_buff_0 : memref<16384xbf16>, 4096, 4096, [<size = 16, stride = 256>, <size = 4, stride = 8>, <size = 8, stride = 32>, <size = 8, stride = 1>]) {bd_id = 9 : i32, next_bd_id = 10 : i32}
      aie.use_lock(%C_L2L3_0_prod_lock_1, Release, 1)
      aie.next_bd ^bb27
    ^bb27:  // pred: ^bb26
      aie.use_lock(%C_L2L3_0_cons_lock_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_0_buff_0 : memref<16384xbf16>, 8192, 4096, [<size = 16, stride = 256>, <size = 4, stride = 8>, <size = 8, stride = 32>, <size = 8, stride = 1>]) {bd_id = 10 : i32, next_bd_id = 11 : i32}
      aie.use_lock(%C_L2L3_0_prod_lock_2, Release, 1)
      aie.next_bd ^bb28
    ^bb28:  // pred: ^bb27
      aie.use_lock(%C_L2L3_0_cons_lock_3, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_0_buff_0 : memref<16384xbf16>, 12288, 4096, [<size = 16, stride = 256>, <size = 4, stride = 8>, <size = 8, stride = 32>, <size = 8, stride = 1>]) {bd_id = 11 : i32, next_bd_id = 12 : i32}
      aie.use_lock(%C_L2L3_0_prod_lock_3, Release, 1)
      aie.next_bd ^bb29
    ^bb29:  // pred: ^bb28
      aie.use_lock(%C_L2L3_0_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_0_buff_1 : memref<16384xbf16>, 0, 4096, [<size = 16, stride = 256>, <size = 4, stride = 8>, <size = 8, stride = 32>, <size = 8, stride = 1>]) {bd_id = 12 : i32, next_bd_id = 13 : i32}
      aie.use_lock(%C_L2L3_0_prod_lock_0, Release, 1)
      aie.next_bd ^bb30
    ^bb30:  // pred: ^bb29
      aie.use_lock(%C_L2L3_0_cons_lock_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_0_buff_1 : memref<16384xbf16>, 4096, 4096, [<size = 16, stride = 256>, <size = 4, stride = 8>, <size = 8, stride = 32>, <size = 8, stride = 1>]) {bd_id = 13 : i32, next_bd_id = 14 : i32}
      aie.use_lock(%C_L2L3_0_prod_lock_1, Release, 1)
      aie.next_bd ^bb31
    ^bb31:  // pred: ^bb30
      aie.use_lock(%C_L2L3_0_cons_lock_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_0_buff_1 : memref<16384xbf16>, 8192, 4096, [<size = 16, stride = 256>, <size = 4, stride = 8>, <size = 8, stride = 32>, <size = 8, stride = 1>]) {bd_id = 14 : i32, next_bd_id = 15 : i32}
      aie.use_lock(%C_L2L3_0_prod_lock_2, Release, 1)
      aie.next_bd ^bb32
    ^bb32:  // pred: ^bb31
      aie.use_lock(%C_L2L3_0_cons_lock_3, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_0_buff_1 : memref<16384xbf16>, 12288, 4096, [<size = 16, stride = 256>, <size = 4, stride = 8>, <size = 8, stride = 32>, <size = 8, stride = 1>]) {bd_id = 15 : i32, next_bd_id = 8 : i32}
      aie.use_lock(%C_L2L3_0_prod_lock_3, Release, 1)
      aie.next_bd ^bb25
    ^bb33:  // pred: ^bb24
      aie.end
    }
    %mem_0_2 = aie.mem(%tile_0_2) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%A_L2L1_0_0_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L2L1_0_0_cons_buff_0 : memref<64x64xbf16>, 0, 4096) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%A_L2L1_0_0_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%A_L2L1_0_0_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L2L1_0_0_cons_buff_1 : memref<64x64xbf16>, 0, 4096) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%A_L2L1_0_0_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%B_L2L1_0_0_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L2L1_0_0_cons_buff_0 : memref<64x64xbf16>, 0, 4096) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%B_L2L1_0_0_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%B_L2L1_0_0_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L2L1_0_0_cons_buff_1 : memref<64x64xbf16>, 0, 4096) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%B_L2L1_0_0_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      %2 = aie.dma_start(MM2S, 0, ^bb7, ^bb8)
    ^bb7:  // 2 preds: ^bb6, ^bb7
      aie.use_lock(%C_L1L2_0_0_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L1L2_0_0_buff_0 : memref<64x64xbf16>, 0, 4096) {bd_id = 4 : i32, next_bd_id = 4 : i32}
      aie.use_lock(%C_L1L2_0_0_prod_lock_0, Release, 1)
      aie.next_bd ^bb7
    ^bb8:  // pred: ^bb6
      aie.end
    }
    %mem_1_2 = aie.mem(%tile_1_2) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%A_L2L1_0_1_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L2L1_0_1_cons_buff_0 : memref<64x64xbf16>, 0, 4096) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%A_L2L1_0_1_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%A_L2L1_0_1_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L2L1_0_1_cons_buff_1 : memref<64x64xbf16>, 0, 4096) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%A_L2L1_0_1_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%B_L2L1_1_0_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L2L1_1_0_cons_buff_0 : memref<64x64xbf16>, 0, 4096) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%B_L2L1_1_0_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%B_L2L1_1_0_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L2L1_1_0_cons_buff_1 : memref<64x64xbf16>, 0, 4096) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%B_L2L1_1_0_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      %2 = aie.dma_start(MM2S, 0, ^bb7, ^bb8)
    ^bb7:  // 2 preds: ^bb6, ^bb7
      aie.use_lock(%C_L1L2_1_0_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L1L2_1_0_buff_0 : memref<64x64xbf16>, 0, 4096) {bd_id = 4 : i32, next_bd_id = 4 : i32}
      aie.use_lock(%C_L1L2_1_0_prod_lock_0, Release, 1)
      aie.next_bd ^bb7
    ^bb8:  // pred: ^bb6
      aie.end
    }
    %mem_2_2 = aie.mem(%tile_2_2) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%A_L2L1_0_2_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L2L1_0_2_cons_buff_0 : memref<64x64xbf16>, 0, 4096) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%A_L2L1_0_2_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%A_L2L1_0_2_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L2L1_0_2_cons_buff_1 : memref<64x64xbf16>, 0, 4096) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%A_L2L1_0_2_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%B_L2L1_2_0_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L2L1_2_0_cons_buff_0 : memref<64x64xbf16>, 0, 4096) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%B_L2L1_2_0_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%B_L2L1_2_0_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L2L1_2_0_cons_buff_1 : memref<64x64xbf16>, 0, 4096) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%B_L2L1_2_0_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      %2 = aie.dma_start(MM2S, 0, ^bb7, ^bb8)
    ^bb7:  // 2 preds: ^bb6, ^bb7
      aie.use_lock(%C_L1L2_2_0_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L1L2_2_0_buff_0 : memref<64x64xbf16>, 0, 4096) {bd_id = 4 : i32, next_bd_id = 4 : i32}
      aie.use_lock(%C_L1L2_2_0_prod_lock_0, Release, 1)
      aie.next_bd ^bb7
    ^bb8:  // pred: ^bb6
      aie.end
    }
    %mem_3_2 = aie.mem(%tile_3_2) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%A_L2L1_0_3_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L2L1_0_3_cons_buff_0 : memref<64x64xbf16>, 0, 4096) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%A_L2L1_0_3_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%A_L2L1_0_3_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L2L1_0_3_cons_buff_1 : memref<64x64xbf16>, 0, 4096) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%A_L2L1_0_3_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%B_L2L1_3_0_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L2L1_3_0_cons_buff_0 : memref<64x64xbf16>, 0, 4096) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%B_L2L1_3_0_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%B_L2L1_3_0_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L2L1_3_0_cons_buff_1 : memref<64x64xbf16>, 0, 4096) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%B_L2L1_3_0_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      %2 = aie.dma_start(MM2S, 0, ^bb7, ^bb8)
    ^bb7:  // 2 preds: ^bb6, ^bb7
      aie.use_lock(%C_L1L2_3_0_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L1L2_3_0_buff_0 : memref<64x64xbf16>, 0, 4096) {bd_id = 4 : i32, next_bd_id = 4 : i32}
      aie.use_lock(%C_L1L2_3_0_prod_lock_0, Release, 1)
      aie.next_bd ^bb7
    ^bb8:  // pred: ^bb6
      aie.end
    }
    aie.shim_dma_allocation @A_L3L2_0_shim_alloc(%shim_noc_tile_0_0, MM2S, 0)
    %memtile_dma_1_1 = aie.memtile_dma(%mem_tile_1_1) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%A_L3L2_1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L3L2_1_cons_buff_0 : memref<4096xbf16>, 0, 4096, [<size = 16, stride = 256>, <size = 8, stride = 8>, <size = 4, stride = 64>, <size = 8, stride = 1>]) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%A_L3L2_1_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%A_L3L2_1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L3L2_1_cons_buff_1 : memref<4096xbf16>, 0, 4096, [<size = 16, stride = 256>, <size = 8, stride = 8>, <size = 4, stride = 64>, <size = 8, stride = 1>]) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%A_L3L2_1_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 0, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%A_L3L2_1_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L3L2_1_cons_buff_0 : memref<4096xbf16>, 0, 4096) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%A_L3L2_1_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%A_L3L2_1_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L3L2_1_cons_buff_1 : memref<4096xbf16>, 0, 4096) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%A_L3L2_1_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      %2 = aie.dma_start(MM2S, 1, ^bb7, ^bb9)
    ^bb7:  // 2 preds: ^bb6, ^bb8
      aie.use_lock(%B_L3L2_1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L3L2_1_cons_buff_0 : memref<4096xbf16>, 0, 4096, [<size = 8, stride = 512>, <size = 8, stride = 8>, <size = 8, stride = 64>, <size = 8, stride = 1>]) {bd_id = 24 : i32, next_bd_id = 25 : i32}
      aie.use_lock(%B_L3L2_1_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb8
    ^bb8:  // pred: ^bb7
      aie.use_lock(%B_L3L2_1_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L3L2_1_cons_buff_1 : memref<4096xbf16>, 0, 4096, [<size = 8, stride = 512>, <size = 8, stride = 8>, <size = 8, stride = 64>, <size = 8, stride = 1>]) {bd_id = 25 : i32, next_bd_id = 24 : i32}
      aie.use_lock(%B_L3L2_1_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb7
    ^bb9:  // pred: ^bb6
      %3 = aie.dma_start(S2MM, 1, ^bb10, ^bb12)
    ^bb10:  // 2 preds: ^bb9, ^bb11
      aie.use_lock(%B_L3L2_1_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L3L2_1_cons_buff_0 : memref<4096xbf16>, 0, 4096) {bd_id = 26 : i32, next_bd_id = 27 : i32}
      aie.use_lock(%B_L3L2_1_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb11
    ^bb11:  // pred: ^bb10
      aie.use_lock(%B_L3L2_1_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L3L2_1_cons_buff_1 : memref<4096xbf16>, 0, 4096) {bd_id = 27 : i32, next_bd_id = 26 : i32}
      aie.use_lock(%B_L3L2_1_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb10
    ^bb12:  // pred: ^bb9
      %4 = aie.dma_start(S2MM, 2, ^bb13, ^bb15)
    ^bb13:  // 2 preds: ^bb12, ^bb14
      aie.use_lock(%C_L2L3_1_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_1_buff_0 : memref<16384xbf16>, 0, 4096) {bd_id = 4 : i32, next_bd_id = 5 : i32}
      aie.use_lock(%C_L2L3_1_cons_lock_0, Release, 1)
      aie.next_bd ^bb14
    ^bb14:  // pred: ^bb13
      aie.use_lock(%C_L2L3_1_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_1_buff_1 : memref<16384xbf16>, 0, 4096) {bd_id = 5 : i32, next_bd_id = 4 : i32}
      aie.use_lock(%C_L2L3_1_cons_lock_0, Release, 1)
      aie.next_bd ^bb13
    ^bb15:  // pred: ^bb12
      %5 = aie.dma_start(S2MM, 3, ^bb16, ^bb18)
    ^bb16:  // 2 preds: ^bb15, ^bb17
      aie.use_lock(%C_L2L3_1_prod_lock_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_1_buff_0 : memref<16384xbf16>, 4096, 4096) {bd_id = 28 : i32, next_bd_id = 29 : i32}
      aie.use_lock(%C_L2L3_1_cons_lock_1, Release, 1)
      aie.next_bd ^bb17
    ^bb17:  // pred: ^bb16
      aie.use_lock(%C_L2L3_1_prod_lock_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_1_buff_1 : memref<16384xbf16>, 4096, 4096) {bd_id = 29 : i32, next_bd_id = 28 : i32}
      aie.use_lock(%C_L2L3_1_cons_lock_1, Release, 1)
      aie.next_bd ^bb16
    ^bb18:  // pred: ^bb15
      %6 = aie.dma_start(S2MM, 4, ^bb19, ^bb21)
    ^bb19:  // 2 preds: ^bb18, ^bb20
      aie.use_lock(%C_L2L3_1_prod_lock_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_1_buff_0 : memref<16384xbf16>, 8192, 4096) {bd_id = 6 : i32, next_bd_id = 7 : i32}
      aie.use_lock(%C_L2L3_1_cons_lock_2, Release, 1)
      aie.next_bd ^bb20
    ^bb20:  // pred: ^bb19
      aie.use_lock(%C_L2L3_1_prod_lock_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_1_buff_1 : memref<16384xbf16>, 8192, 4096) {bd_id = 7 : i32, next_bd_id = 6 : i32}
      aie.use_lock(%C_L2L3_1_cons_lock_2, Release, 1)
      aie.next_bd ^bb19
    ^bb21:  // pred: ^bb18
      %7 = aie.dma_start(S2MM, 5, ^bb22, ^bb24)
    ^bb22:  // 2 preds: ^bb21, ^bb23
      aie.use_lock(%C_L2L3_1_prod_lock_3, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_1_buff_0 : memref<16384xbf16>, 12288, 4096) {bd_id = 30 : i32, next_bd_id = 31 : i32}
      aie.use_lock(%C_L2L3_1_cons_lock_3, Release, 1)
      aie.next_bd ^bb23
    ^bb23:  // pred: ^bb22
      aie.use_lock(%C_L2L3_1_prod_lock_3, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_1_buff_1 : memref<16384xbf16>, 12288, 4096) {bd_id = 31 : i32, next_bd_id = 30 : i32}
      aie.use_lock(%C_L2L3_1_cons_lock_3, Release, 1)
      aie.next_bd ^bb22
    ^bb24:  // pred: ^bb21
      %8 = aie.dma_start(MM2S, 2, ^bb25, ^bb33)
    ^bb25:  // 2 preds: ^bb24, ^bb32
      aie.use_lock(%C_L2L3_1_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_1_buff_0 : memref<16384xbf16>, 0, 4096, [<size = 16, stride = 256>, <size = 4, stride = 8>, <size = 8, stride = 32>, <size = 8, stride = 1>]) {bd_id = 8 : i32, next_bd_id = 9 : i32}
      aie.use_lock(%C_L2L3_1_prod_lock_0, Release, 1)
      aie.next_bd ^bb26
    ^bb26:  // pred: ^bb25
      aie.use_lock(%C_L2L3_1_cons_lock_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_1_buff_0 : memref<16384xbf16>, 4096, 4096, [<size = 16, stride = 256>, <size = 4, stride = 8>, <size = 8, stride = 32>, <size = 8, stride = 1>]) {bd_id = 9 : i32, next_bd_id = 10 : i32}
      aie.use_lock(%C_L2L3_1_prod_lock_1, Release, 1)
      aie.next_bd ^bb27
    ^bb27:  // pred: ^bb26
      aie.use_lock(%C_L2L3_1_cons_lock_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_1_buff_0 : memref<16384xbf16>, 8192, 4096, [<size = 16, stride = 256>, <size = 4, stride = 8>, <size = 8, stride = 32>, <size = 8, stride = 1>]) {bd_id = 10 : i32, next_bd_id = 11 : i32}
      aie.use_lock(%C_L2L3_1_prod_lock_2, Release, 1)
      aie.next_bd ^bb28
    ^bb28:  // pred: ^bb27
      aie.use_lock(%C_L2L3_1_cons_lock_3, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_1_buff_0 : memref<16384xbf16>, 12288, 4096, [<size = 16, stride = 256>, <size = 4, stride = 8>, <size = 8, stride = 32>, <size = 8, stride = 1>]) {bd_id = 11 : i32, next_bd_id = 12 : i32}
      aie.use_lock(%C_L2L3_1_prod_lock_3, Release, 1)
      aie.next_bd ^bb29
    ^bb29:  // pred: ^bb28
      aie.use_lock(%C_L2L3_1_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_1_buff_1 : memref<16384xbf16>, 0, 4096, [<size = 16, stride = 256>, <size = 4, stride = 8>, <size = 8, stride = 32>, <size = 8, stride = 1>]) {bd_id = 12 : i32, next_bd_id = 13 : i32}
      aie.use_lock(%C_L2L3_1_prod_lock_0, Release, 1)
      aie.next_bd ^bb30
    ^bb30:  // pred: ^bb29
      aie.use_lock(%C_L2L3_1_cons_lock_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_1_buff_1 : memref<16384xbf16>, 4096, 4096, [<size = 16, stride = 256>, <size = 4, stride = 8>, <size = 8, stride = 32>, <size = 8, stride = 1>]) {bd_id = 13 : i32, next_bd_id = 14 : i32}
      aie.use_lock(%C_L2L3_1_prod_lock_1, Release, 1)
      aie.next_bd ^bb31
    ^bb31:  // pred: ^bb30
      aie.use_lock(%C_L2L3_1_cons_lock_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_1_buff_1 : memref<16384xbf16>, 8192, 4096, [<size = 16, stride = 256>, <size = 4, stride = 8>, <size = 8, stride = 32>, <size = 8, stride = 1>]) {bd_id = 14 : i32, next_bd_id = 15 : i32}
      aie.use_lock(%C_L2L3_1_prod_lock_2, Release, 1)
      aie.next_bd ^bb32
    ^bb32:  // pred: ^bb31
      aie.use_lock(%C_L2L3_1_cons_lock_3, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_1_buff_1 : memref<16384xbf16>, 12288, 4096, [<size = 16, stride = 256>, <size = 4, stride = 8>, <size = 8, stride = 32>, <size = 8, stride = 1>]) {bd_id = 15 : i32, next_bd_id = 8 : i32}
      aie.use_lock(%C_L2L3_1_prod_lock_3, Release, 1)
      aie.next_bd ^bb25
    ^bb33:  // pred: ^bb24
      aie.end
    }
    %mem_0_3 = aie.mem(%tile_0_3) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%A_L2L1_1_0_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L2L1_1_0_cons_buff_0 : memref<64x64xbf16>, 0, 4096) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%A_L2L1_1_0_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%A_L2L1_1_0_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L2L1_1_0_cons_buff_1 : memref<64x64xbf16>, 0, 4096) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%A_L2L1_1_0_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%B_L2L1_0_1_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L2L1_0_1_cons_buff_0 : memref<64x64xbf16>, 0, 4096) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%B_L2L1_0_1_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%B_L2L1_0_1_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L2L1_0_1_cons_buff_1 : memref<64x64xbf16>, 0, 4096) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%B_L2L1_0_1_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      %2 = aie.dma_start(MM2S, 0, ^bb7, ^bb8)
    ^bb7:  // 2 preds: ^bb6, ^bb7
      aie.use_lock(%C_L1L2_0_1_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L1L2_0_1_buff_0 : memref<64x64xbf16>, 0, 4096) {bd_id = 4 : i32, next_bd_id = 4 : i32}
      aie.use_lock(%C_L1L2_0_1_prod_lock_0, Release, 1)
      aie.next_bd ^bb7
    ^bb8:  // pred: ^bb6
      aie.end
    }
    %mem_1_3 = aie.mem(%tile_1_3) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%A_L2L1_1_1_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L2L1_1_1_cons_buff_0 : memref<64x64xbf16>, 0, 4096) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%A_L2L1_1_1_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%A_L2L1_1_1_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L2L1_1_1_cons_buff_1 : memref<64x64xbf16>, 0, 4096) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%A_L2L1_1_1_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%B_L2L1_1_1_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L2L1_1_1_cons_buff_0 : memref<64x64xbf16>, 0, 4096) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%B_L2L1_1_1_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%B_L2L1_1_1_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L2L1_1_1_cons_buff_1 : memref<64x64xbf16>, 0, 4096) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%B_L2L1_1_1_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      %2 = aie.dma_start(MM2S, 0, ^bb7, ^bb8)
    ^bb7:  // 2 preds: ^bb6, ^bb7
      aie.use_lock(%C_L1L2_1_1_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L1L2_1_1_buff_0 : memref<64x64xbf16>, 0, 4096) {bd_id = 4 : i32, next_bd_id = 4 : i32}
      aie.use_lock(%C_L1L2_1_1_prod_lock_0, Release, 1)
      aie.next_bd ^bb7
    ^bb8:  // pred: ^bb6
      aie.end
    }
    %mem_2_3 = aie.mem(%tile_2_3) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%A_L2L1_1_2_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L2L1_1_2_cons_buff_0 : memref<64x64xbf16>, 0, 4096) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%A_L2L1_1_2_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%A_L2L1_1_2_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L2L1_1_2_cons_buff_1 : memref<64x64xbf16>, 0, 4096) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%A_L2L1_1_2_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%B_L2L1_2_1_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L2L1_2_1_cons_buff_0 : memref<64x64xbf16>, 0, 4096) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%B_L2L1_2_1_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%B_L2L1_2_1_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L2L1_2_1_cons_buff_1 : memref<64x64xbf16>, 0, 4096) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%B_L2L1_2_1_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      %2 = aie.dma_start(MM2S, 0, ^bb7, ^bb8)
    ^bb7:  // 2 preds: ^bb6, ^bb7
      aie.use_lock(%C_L1L2_2_1_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L1L2_2_1_buff_0 : memref<64x64xbf16>, 0, 4096) {bd_id = 4 : i32, next_bd_id = 4 : i32}
      aie.use_lock(%C_L1L2_2_1_prod_lock_0, Release, 1)
      aie.next_bd ^bb7
    ^bb8:  // pred: ^bb6
      aie.end
    }
    %mem_3_3 = aie.mem(%tile_3_3) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%A_L2L1_1_3_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L2L1_1_3_cons_buff_0 : memref<64x64xbf16>, 0, 4096) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%A_L2L1_1_3_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%A_L2L1_1_3_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L2L1_1_3_cons_buff_1 : memref<64x64xbf16>, 0, 4096) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%A_L2L1_1_3_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%B_L2L1_3_1_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L2L1_3_1_cons_buff_0 : memref<64x64xbf16>, 0, 4096) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%B_L2L1_3_1_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%B_L2L1_3_1_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L2L1_3_1_cons_buff_1 : memref<64x64xbf16>, 0, 4096) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%B_L2L1_3_1_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      %2 = aie.dma_start(MM2S, 0, ^bb7, ^bb8)
    ^bb7:  // 2 preds: ^bb6, ^bb7
      aie.use_lock(%C_L1L2_3_1_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L1L2_3_1_buff_0 : memref<64x64xbf16>, 0, 4096) {bd_id = 4 : i32, next_bd_id = 4 : i32}
      aie.use_lock(%C_L1L2_3_1_prod_lock_0, Release, 1)
      aie.next_bd ^bb7
    ^bb8:  // pred: ^bb6
      aie.end
    }
    aie.shim_dma_allocation @A_L3L2_1_shim_alloc(%shim_noc_tile_1_0, MM2S, 0)
    %memtile_dma_2_1 = aie.memtile_dma(%mem_tile_2_1) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%A_L3L2_2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L3L2_2_cons_buff_0 : memref<4096xbf16>, 0, 4096, [<size = 16, stride = 256>, <size = 8, stride = 8>, <size = 4, stride = 64>, <size = 8, stride = 1>]) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%A_L3L2_2_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%A_L3L2_2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L3L2_2_cons_buff_1 : memref<4096xbf16>, 0, 4096, [<size = 16, stride = 256>, <size = 8, stride = 8>, <size = 4, stride = 64>, <size = 8, stride = 1>]) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%A_L3L2_2_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 0, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%A_L3L2_2_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L3L2_2_cons_buff_0 : memref<4096xbf16>, 0, 4096) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%A_L3L2_2_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%A_L3L2_2_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L3L2_2_cons_buff_1 : memref<4096xbf16>, 0, 4096) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%A_L3L2_2_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      %2 = aie.dma_start(MM2S, 1, ^bb7, ^bb9)
    ^bb7:  // 2 preds: ^bb6, ^bb8
      aie.use_lock(%B_L3L2_2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L3L2_2_cons_buff_0 : memref<4096xbf16>, 0, 4096, [<size = 8, stride = 512>, <size = 8, stride = 8>, <size = 8, stride = 64>, <size = 8, stride = 1>]) {bd_id = 24 : i32, next_bd_id = 25 : i32}
      aie.use_lock(%B_L3L2_2_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb8
    ^bb8:  // pred: ^bb7
      aie.use_lock(%B_L3L2_2_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L3L2_2_cons_buff_1 : memref<4096xbf16>, 0, 4096, [<size = 8, stride = 512>, <size = 8, stride = 8>, <size = 8, stride = 64>, <size = 8, stride = 1>]) {bd_id = 25 : i32, next_bd_id = 24 : i32}
      aie.use_lock(%B_L3L2_2_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb7
    ^bb9:  // pred: ^bb6
      %3 = aie.dma_start(S2MM, 1, ^bb10, ^bb12)
    ^bb10:  // 2 preds: ^bb9, ^bb11
      aie.use_lock(%B_L3L2_2_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L3L2_2_cons_buff_0 : memref<4096xbf16>, 0, 4096) {bd_id = 26 : i32, next_bd_id = 27 : i32}
      aie.use_lock(%B_L3L2_2_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb11
    ^bb11:  // pred: ^bb10
      aie.use_lock(%B_L3L2_2_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L3L2_2_cons_buff_1 : memref<4096xbf16>, 0, 4096) {bd_id = 27 : i32, next_bd_id = 26 : i32}
      aie.use_lock(%B_L3L2_2_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb10
    ^bb12:  // pred: ^bb9
      %4 = aie.dma_start(S2MM, 2, ^bb13, ^bb15)
    ^bb13:  // 2 preds: ^bb12, ^bb14
      aie.use_lock(%C_L2L3_2_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_2_buff_0 : memref<16384xbf16>, 0, 4096) {bd_id = 4 : i32, next_bd_id = 5 : i32}
      aie.use_lock(%C_L2L3_2_cons_lock_0, Release, 1)
      aie.next_bd ^bb14
    ^bb14:  // pred: ^bb13
      aie.use_lock(%C_L2L3_2_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_2_buff_1 : memref<16384xbf16>, 0, 4096) {bd_id = 5 : i32, next_bd_id = 4 : i32}
      aie.use_lock(%C_L2L3_2_cons_lock_0, Release, 1)
      aie.next_bd ^bb13
    ^bb15:  // pred: ^bb12
      %5 = aie.dma_start(S2MM, 3, ^bb16, ^bb18)
    ^bb16:  // 2 preds: ^bb15, ^bb17
      aie.use_lock(%C_L2L3_2_prod_lock_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_2_buff_0 : memref<16384xbf16>, 4096, 4096) {bd_id = 28 : i32, next_bd_id = 29 : i32}
      aie.use_lock(%C_L2L3_2_cons_lock_1, Release, 1)
      aie.next_bd ^bb17
    ^bb17:  // pred: ^bb16
      aie.use_lock(%C_L2L3_2_prod_lock_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_2_buff_1 : memref<16384xbf16>, 4096, 4096) {bd_id = 29 : i32, next_bd_id = 28 : i32}
      aie.use_lock(%C_L2L3_2_cons_lock_1, Release, 1)
      aie.next_bd ^bb16
    ^bb18:  // pred: ^bb15
      %6 = aie.dma_start(S2MM, 4, ^bb19, ^bb21)
    ^bb19:  // 2 preds: ^bb18, ^bb20
      aie.use_lock(%C_L2L3_2_prod_lock_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_2_buff_0 : memref<16384xbf16>, 8192, 4096) {bd_id = 6 : i32, next_bd_id = 7 : i32}
      aie.use_lock(%C_L2L3_2_cons_lock_2, Release, 1)
      aie.next_bd ^bb20
    ^bb20:  // pred: ^bb19
      aie.use_lock(%C_L2L3_2_prod_lock_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_2_buff_1 : memref<16384xbf16>, 8192, 4096) {bd_id = 7 : i32, next_bd_id = 6 : i32}
      aie.use_lock(%C_L2L3_2_cons_lock_2, Release, 1)
      aie.next_bd ^bb19
    ^bb21:  // pred: ^bb18
      %7 = aie.dma_start(S2MM, 5, ^bb22, ^bb24)
    ^bb22:  // 2 preds: ^bb21, ^bb23
      aie.use_lock(%C_L2L3_2_prod_lock_3, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_2_buff_0 : memref<16384xbf16>, 12288, 4096) {bd_id = 30 : i32, next_bd_id = 31 : i32}
      aie.use_lock(%C_L2L3_2_cons_lock_3, Release, 1)
      aie.next_bd ^bb23
    ^bb23:  // pred: ^bb22
      aie.use_lock(%C_L2L3_2_prod_lock_3, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_2_buff_1 : memref<16384xbf16>, 12288, 4096) {bd_id = 31 : i32, next_bd_id = 30 : i32}
      aie.use_lock(%C_L2L3_2_cons_lock_3, Release, 1)
      aie.next_bd ^bb22
    ^bb24:  // pred: ^bb21
      %8 = aie.dma_start(MM2S, 2, ^bb25, ^bb33)
    ^bb25:  // 2 preds: ^bb24, ^bb32
      aie.use_lock(%C_L2L3_2_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_2_buff_0 : memref<16384xbf16>, 0, 4096, [<size = 16, stride = 256>, <size = 4, stride = 8>, <size = 8, stride = 32>, <size = 8, stride = 1>]) {bd_id = 8 : i32, next_bd_id = 9 : i32}
      aie.use_lock(%C_L2L3_2_prod_lock_0, Release, 1)
      aie.next_bd ^bb26
    ^bb26:  // pred: ^bb25
      aie.use_lock(%C_L2L3_2_cons_lock_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_2_buff_0 : memref<16384xbf16>, 4096, 4096, [<size = 16, stride = 256>, <size = 4, stride = 8>, <size = 8, stride = 32>, <size = 8, stride = 1>]) {bd_id = 9 : i32, next_bd_id = 10 : i32}
      aie.use_lock(%C_L2L3_2_prod_lock_1, Release, 1)
      aie.next_bd ^bb27
    ^bb27:  // pred: ^bb26
      aie.use_lock(%C_L2L3_2_cons_lock_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_2_buff_0 : memref<16384xbf16>, 8192, 4096, [<size = 16, stride = 256>, <size = 4, stride = 8>, <size = 8, stride = 32>, <size = 8, stride = 1>]) {bd_id = 10 : i32, next_bd_id = 11 : i32}
      aie.use_lock(%C_L2L3_2_prod_lock_2, Release, 1)
      aie.next_bd ^bb28
    ^bb28:  // pred: ^bb27
      aie.use_lock(%C_L2L3_2_cons_lock_3, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_2_buff_0 : memref<16384xbf16>, 12288, 4096, [<size = 16, stride = 256>, <size = 4, stride = 8>, <size = 8, stride = 32>, <size = 8, stride = 1>]) {bd_id = 11 : i32, next_bd_id = 12 : i32}
      aie.use_lock(%C_L2L3_2_prod_lock_3, Release, 1)
      aie.next_bd ^bb29
    ^bb29:  // pred: ^bb28
      aie.use_lock(%C_L2L3_2_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_2_buff_1 : memref<16384xbf16>, 0, 4096, [<size = 16, stride = 256>, <size = 4, stride = 8>, <size = 8, stride = 32>, <size = 8, stride = 1>]) {bd_id = 12 : i32, next_bd_id = 13 : i32}
      aie.use_lock(%C_L2L3_2_prod_lock_0, Release, 1)
      aie.next_bd ^bb30
    ^bb30:  // pred: ^bb29
      aie.use_lock(%C_L2L3_2_cons_lock_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_2_buff_1 : memref<16384xbf16>, 4096, 4096, [<size = 16, stride = 256>, <size = 4, stride = 8>, <size = 8, stride = 32>, <size = 8, stride = 1>]) {bd_id = 13 : i32, next_bd_id = 14 : i32}
      aie.use_lock(%C_L2L3_2_prod_lock_1, Release, 1)
      aie.next_bd ^bb31
    ^bb31:  // pred: ^bb30
      aie.use_lock(%C_L2L3_2_cons_lock_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_2_buff_1 : memref<16384xbf16>, 8192, 4096, [<size = 16, stride = 256>, <size = 4, stride = 8>, <size = 8, stride = 32>, <size = 8, stride = 1>]) {bd_id = 14 : i32, next_bd_id = 15 : i32}
      aie.use_lock(%C_L2L3_2_prod_lock_2, Release, 1)
      aie.next_bd ^bb32
    ^bb32:  // pred: ^bb31
      aie.use_lock(%C_L2L3_2_cons_lock_3, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_2_buff_1 : memref<16384xbf16>, 12288, 4096, [<size = 16, stride = 256>, <size = 4, stride = 8>, <size = 8, stride = 32>, <size = 8, stride = 1>]) {bd_id = 15 : i32, next_bd_id = 8 : i32}
      aie.use_lock(%C_L2L3_2_prod_lock_3, Release, 1)
      aie.next_bd ^bb25
    ^bb33:  // pred: ^bb24
      aie.end
    }
    %mem_0_4 = aie.mem(%tile_0_4) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%A_L2L1_2_0_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L2L1_2_0_cons_buff_0 : memref<64x64xbf16>, 0, 4096) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%A_L2L1_2_0_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%A_L2L1_2_0_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L2L1_2_0_cons_buff_1 : memref<64x64xbf16>, 0, 4096) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%A_L2L1_2_0_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%B_L2L1_0_2_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L2L1_0_2_cons_buff_0 : memref<64x64xbf16>, 0, 4096) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%B_L2L1_0_2_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%B_L2L1_0_2_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L2L1_0_2_cons_buff_1 : memref<64x64xbf16>, 0, 4096) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%B_L2L1_0_2_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      %2 = aie.dma_start(MM2S, 0, ^bb7, ^bb8)
    ^bb7:  // 2 preds: ^bb6, ^bb7
      aie.use_lock(%C_L1L2_0_2_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L1L2_0_2_buff_0 : memref<64x64xbf16>, 0, 4096) {bd_id = 4 : i32, next_bd_id = 4 : i32}
      aie.use_lock(%C_L1L2_0_2_prod_lock_0, Release, 1)
      aie.next_bd ^bb7
    ^bb8:  // pred: ^bb6
      aie.end
    }
    %mem_1_4 = aie.mem(%tile_1_4) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%A_L2L1_2_1_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L2L1_2_1_cons_buff_0 : memref<64x64xbf16>, 0, 4096) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%A_L2L1_2_1_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%A_L2L1_2_1_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L2L1_2_1_cons_buff_1 : memref<64x64xbf16>, 0, 4096) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%A_L2L1_2_1_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%B_L2L1_1_2_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L2L1_1_2_cons_buff_0 : memref<64x64xbf16>, 0, 4096) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%B_L2L1_1_2_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%B_L2L1_1_2_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L2L1_1_2_cons_buff_1 : memref<64x64xbf16>, 0, 4096) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%B_L2L1_1_2_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      %2 = aie.dma_start(MM2S, 0, ^bb7, ^bb8)
    ^bb7:  // 2 preds: ^bb6, ^bb7
      aie.use_lock(%C_L1L2_1_2_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L1L2_1_2_buff_0 : memref<64x64xbf16>, 0, 4096) {bd_id = 4 : i32, next_bd_id = 4 : i32}
      aie.use_lock(%C_L1L2_1_2_prod_lock_0, Release, 1)
      aie.next_bd ^bb7
    ^bb8:  // pred: ^bb6
      aie.end
    }
    %mem_2_4 = aie.mem(%tile_2_4) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%A_L2L1_2_2_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L2L1_2_2_cons_buff_0 : memref<64x64xbf16>, 0, 4096) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%A_L2L1_2_2_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%A_L2L1_2_2_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L2L1_2_2_cons_buff_1 : memref<64x64xbf16>, 0, 4096) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%A_L2L1_2_2_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%B_L2L1_2_2_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L2L1_2_2_cons_buff_0 : memref<64x64xbf16>, 0, 4096) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%B_L2L1_2_2_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%B_L2L1_2_2_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L2L1_2_2_cons_buff_1 : memref<64x64xbf16>, 0, 4096) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%B_L2L1_2_2_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      %2 = aie.dma_start(MM2S, 0, ^bb7, ^bb8)
    ^bb7:  // 2 preds: ^bb6, ^bb7
      aie.use_lock(%C_L1L2_2_2_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L1L2_2_2_buff_0 : memref<64x64xbf16>, 0, 4096) {bd_id = 4 : i32, next_bd_id = 4 : i32}
      aie.use_lock(%C_L1L2_2_2_prod_lock_0, Release, 1)
      aie.next_bd ^bb7
    ^bb8:  // pred: ^bb6
      aie.end
    }
    %mem_3_4 = aie.mem(%tile_3_4) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%A_L2L1_2_3_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L2L1_2_3_cons_buff_0 : memref<64x64xbf16>, 0, 4096) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%A_L2L1_2_3_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%A_L2L1_2_3_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L2L1_2_3_cons_buff_1 : memref<64x64xbf16>, 0, 4096) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%A_L2L1_2_3_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%B_L2L1_3_2_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L2L1_3_2_cons_buff_0 : memref<64x64xbf16>, 0, 4096) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%B_L2L1_3_2_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%B_L2L1_3_2_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L2L1_3_2_cons_buff_1 : memref<64x64xbf16>, 0, 4096) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%B_L2L1_3_2_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      %2 = aie.dma_start(MM2S, 0, ^bb7, ^bb8)
    ^bb7:  // 2 preds: ^bb6, ^bb7
      aie.use_lock(%C_L1L2_3_2_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L1L2_3_2_buff_0 : memref<64x64xbf16>, 0, 4096) {bd_id = 4 : i32, next_bd_id = 4 : i32}
      aie.use_lock(%C_L1L2_3_2_prod_lock_0, Release, 1)
      aie.next_bd ^bb7
    ^bb8:  // pred: ^bb6
      aie.end
    }
    aie.shim_dma_allocation @A_L3L2_2_shim_alloc(%shim_noc_tile_2_0, MM2S, 0)
    %memtile_dma_3_1 = aie.memtile_dma(%mem_tile_3_1) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%A_L3L2_3_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L3L2_3_cons_buff_0 : memref<4096xbf16>, 0, 4096, [<size = 16, stride = 256>, <size = 8, stride = 8>, <size = 4, stride = 64>, <size = 8, stride = 1>]) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%A_L3L2_3_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%A_L3L2_3_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L3L2_3_cons_buff_1 : memref<4096xbf16>, 0, 4096, [<size = 16, stride = 256>, <size = 8, stride = 8>, <size = 4, stride = 64>, <size = 8, stride = 1>]) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%A_L3L2_3_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 0, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%A_L3L2_3_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L3L2_3_cons_buff_0 : memref<4096xbf16>, 0, 4096) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%A_L3L2_3_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%A_L3L2_3_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L3L2_3_cons_buff_1 : memref<4096xbf16>, 0, 4096) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%A_L3L2_3_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      %2 = aie.dma_start(MM2S, 1, ^bb7, ^bb9)
    ^bb7:  // 2 preds: ^bb6, ^bb8
      aie.use_lock(%B_L3L2_3_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L3L2_3_cons_buff_0 : memref<4096xbf16>, 0, 4096, [<size = 8, stride = 512>, <size = 8, stride = 8>, <size = 8, stride = 64>, <size = 8, stride = 1>]) {bd_id = 24 : i32, next_bd_id = 25 : i32}
      aie.use_lock(%B_L3L2_3_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb8
    ^bb8:  // pred: ^bb7
      aie.use_lock(%B_L3L2_3_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L3L2_3_cons_buff_1 : memref<4096xbf16>, 0, 4096, [<size = 8, stride = 512>, <size = 8, stride = 8>, <size = 8, stride = 64>, <size = 8, stride = 1>]) {bd_id = 25 : i32, next_bd_id = 24 : i32}
      aie.use_lock(%B_L3L2_3_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb7
    ^bb9:  // pred: ^bb6
      %3 = aie.dma_start(S2MM, 1, ^bb10, ^bb12)
    ^bb10:  // 2 preds: ^bb9, ^bb11
      aie.use_lock(%B_L3L2_3_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L3L2_3_cons_buff_0 : memref<4096xbf16>, 0, 4096) {bd_id = 26 : i32, next_bd_id = 27 : i32}
      aie.use_lock(%B_L3L2_3_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb11
    ^bb11:  // pred: ^bb10
      aie.use_lock(%B_L3L2_3_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L3L2_3_cons_buff_1 : memref<4096xbf16>, 0, 4096) {bd_id = 27 : i32, next_bd_id = 26 : i32}
      aie.use_lock(%B_L3L2_3_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb10
    ^bb12:  // pred: ^bb9
      %4 = aie.dma_start(S2MM, 2, ^bb13, ^bb15)
    ^bb13:  // 2 preds: ^bb12, ^bb14
      aie.use_lock(%C_L2L3_3_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_3_buff_0 : memref<16384xbf16>, 0, 4096) {bd_id = 4 : i32, next_bd_id = 5 : i32}
      aie.use_lock(%C_L2L3_3_cons_lock_0, Release, 1)
      aie.next_bd ^bb14
    ^bb14:  // pred: ^bb13
      aie.use_lock(%C_L2L3_3_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_3_buff_1 : memref<16384xbf16>, 0, 4096) {bd_id = 5 : i32, next_bd_id = 4 : i32}
      aie.use_lock(%C_L2L3_3_cons_lock_0, Release, 1)
      aie.next_bd ^bb13
    ^bb15:  // pred: ^bb12
      %5 = aie.dma_start(S2MM, 3, ^bb16, ^bb18)
    ^bb16:  // 2 preds: ^bb15, ^bb17
      aie.use_lock(%C_L2L3_3_prod_lock_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_3_buff_0 : memref<16384xbf16>, 4096, 4096) {bd_id = 28 : i32, next_bd_id = 29 : i32}
      aie.use_lock(%C_L2L3_3_cons_lock_1, Release, 1)
      aie.next_bd ^bb17
    ^bb17:  // pred: ^bb16
      aie.use_lock(%C_L2L3_3_prod_lock_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_3_buff_1 : memref<16384xbf16>, 4096, 4096) {bd_id = 29 : i32, next_bd_id = 28 : i32}
      aie.use_lock(%C_L2L3_3_cons_lock_1, Release, 1)
      aie.next_bd ^bb16
    ^bb18:  // pred: ^bb15
      %6 = aie.dma_start(S2MM, 4, ^bb19, ^bb21)
    ^bb19:  // 2 preds: ^bb18, ^bb20
      aie.use_lock(%C_L2L3_3_prod_lock_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_3_buff_0 : memref<16384xbf16>, 8192, 4096) {bd_id = 6 : i32, next_bd_id = 7 : i32}
      aie.use_lock(%C_L2L3_3_cons_lock_2, Release, 1)
      aie.next_bd ^bb20
    ^bb20:  // pred: ^bb19
      aie.use_lock(%C_L2L3_3_prod_lock_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_3_buff_1 : memref<16384xbf16>, 8192, 4096) {bd_id = 7 : i32, next_bd_id = 6 : i32}
      aie.use_lock(%C_L2L3_3_cons_lock_2, Release, 1)
      aie.next_bd ^bb19
    ^bb21:  // pred: ^bb18
      %7 = aie.dma_start(S2MM, 5, ^bb22, ^bb24)
    ^bb22:  // 2 preds: ^bb21, ^bb23
      aie.use_lock(%C_L2L3_3_prod_lock_3, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_3_buff_0 : memref<16384xbf16>, 12288, 4096) {bd_id = 30 : i32, next_bd_id = 31 : i32}
      aie.use_lock(%C_L2L3_3_cons_lock_3, Release, 1)
      aie.next_bd ^bb23
    ^bb23:  // pred: ^bb22
      aie.use_lock(%C_L2L3_3_prod_lock_3, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_3_buff_1 : memref<16384xbf16>, 12288, 4096) {bd_id = 31 : i32, next_bd_id = 30 : i32}
      aie.use_lock(%C_L2L3_3_cons_lock_3, Release, 1)
      aie.next_bd ^bb22
    ^bb24:  // pred: ^bb21
      %8 = aie.dma_start(MM2S, 2, ^bb25, ^bb33)
    ^bb25:  // 2 preds: ^bb24, ^bb32
      aie.use_lock(%C_L2L3_3_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_3_buff_0 : memref<16384xbf16>, 0, 4096, [<size = 16, stride = 256>, <size = 4, stride = 8>, <size = 8, stride = 32>, <size = 8, stride = 1>]) {bd_id = 8 : i32, next_bd_id = 9 : i32}
      aie.use_lock(%C_L2L3_3_prod_lock_0, Release, 1)
      aie.next_bd ^bb26
    ^bb26:  // pred: ^bb25
      aie.use_lock(%C_L2L3_3_cons_lock_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_3_buff_0 : memref<16384xbf16>, 4096, 4096, [<size = 16, stride = 256>, <size = 4, stride = 8>, <size = 8, stride = 32>, <size = 8, stride = 1>]) {bd_id = 9 : i32, next_bd_id = 10 : i32}
      aie.use_lock(%C_L2L3_3_prod_lock_1, Release, 1)
      aie.next_bd ^bb27
    ^bb27:  // pred: ^bb26
      aie.use_lock(%C_L2L3_3_cons_lock_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_3_buff_0 : memref<16384xbf16>, 8192, 4096, [<size = 16, stride = 256>, <size = 4, stride = 8>, <size = 8, stride = 32>, <size = 8, stride = 1>]) {bd_id = 10 : i32, next_bd_id = 11 : i32}
      aie.use_lock(%C_L2L3_3_prod_lock_2, Release, 1)
      aie.next_bd ^bb28
    ^bb28:  // pred: ^bb27
      aie.use_lock(%C_L2L3_3_cons_lock_3, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_3_buff_0 : memref<16384xbf16>, 12288, 4096, [<size = 16, stride = 256>, <size = 4, stride = 8>, <size = 8, stride = 32>, <size = 8, stride = 1>]) {bd_id = 11 : i32, next_bd_id = 12 : i32}
      aie.use_lock(%C_L2L3_3_prod_lock_3, Release, 1)
      aie.next_bd ^bb29
    ^bb29:  // pred: ^bb28
      aie.use_lock(%C_L2L3_3_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_3_buff_1 : memref<16384xbf16>, 0, 4096, [<size = 16, stride = 256>, <size = 4, stride = 8>, <size = 8, stride = 32>, <size = 8, stride = 1>]) {bd_id = 12 : i32, next_bd_id = 13 : i32}
      aie.use_lock(%C_L2L3_3_prod_lock_0, Release, 1)
      aie.next_bd ^bb30
    ^bb30:  // pred: ^bb29
      aie.use_lock(%C_L2L3_3_cons_lock_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_3_buff_1 : memref<16384xbf16>, 4096, 4096, [<size = 16, stride = 256>, <size = 4, stride = 8>, <size = 8, stride = 32>, <size = 8, stride = 1>]) {bd_id = 13 : i32, next_bd_id = 14 : i32}
      aie.use_lock(%C_L2L3_3_prod_lock_1, Release, 1)
      aie.next_bd ^bb31
    ^bb31:  // pred: ^bb30
      aie.use_lock(%C_L2L3_3_cons_lock_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_3_buff_1 : memref<16384xbf16>, 8192, 4096, [<size = 16, stride = 256>, <size = 4, stride = 8>, <size = 8, stride = 32>, <size = 8, stride = 1>]) {bd_id = 14 : i32, next_bd_id = 15 : i32}
      aie.use_lock(%C_L2L3_3_prod_lock_2, Release, 1)
      aie.next_bd ^bb32
    ^bb32:  // pred: ^bb31
      aie.use_lock(%C_L2L3_3_cons_lock_3, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L2L3_3_buff_1 : memref<16384xbf16>, 12288, 4096, [<size = 16, stride = 256>, <size = 4, stride = 8>, <size = 8, stride = 32>, <size = 8, stride = 1>]) {bd_id = 15 : i32, next_bd_id = 8 : i32}
      aie.use_lock(%C_L2L3_3_prod_lock_3, Release, 1)
      aie.next_bd ^bb25
    ^bb33:  // pred: ^bb24
      aie.end
    }
    %mem_0_5 = aie.mem(%tile_0_5) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%A_L2L1_3_0_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L2L1_3_0_cons_buff_0 : memref<64x64xbf16>, 0, 4096) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%A_L2L1_3_0_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%A_L2L1_3_0_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L2L1_3_0_cons_buff_1 : memref<64x64xbf16>, 0, 4096) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%A_L2L1_3_0_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%B_L2L1_0_3_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L2L1_0_3_cons_buff_0 : memref<64x64xbf16>, 0, 4096) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%B_L2L1_0_3_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%B_L2L1_0_3_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L2L1_0_3_cons_buff_1 : memref<64x64xbf16>, 0, 4096) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%B_L2L1_0_3_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      %2 = aie.dma_start(MM2S, 0, ^bb7, ^bb8)
    ^bb7:  // 2 preds: ^bb6, ^bb7
      aie.use_lock(%C_L1L2_0_3_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L1L2_0_3_buff_0 : memref<64x64xbf16>, 0, 4096) {bd_id = 4 : i32, next_bd_id = 4 : i32}
      aie.use_lock(%C_L1L2_0_3_prod_lock_0, Release, 1)
      aie.next_bd ^bb7
    ^bb8:  // pred: ^bb6
      aie.end
    }
    %mem_1_5 = aie.mem(%tile_1_5) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%A_L2L1_3_1_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L2L1_3_1_cons_buff_0 : memref<64x64xbf16>, 0, 4096) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%A_L2L1_3_1_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%A_L2L1_3_1_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L2L1_3_1_cons_buff_1 : memref<64x64xbf16>, 0, 4096) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%A_L2L1_3_1_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%B_L2L1_1_3_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L2L1_1_3_cons_buff_0 : memref<64x64xbf16>, 0, 4096) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%B_L2L1_1_3_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%B_L2L1_1_3_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L2L1_1_3_cons_buff_1 : memref<64x64xbf16>, 0, 4096) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%B_L2L1_1_3_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      %2 = aie.dma_start(MM2S, 0, ^bb7, ^bb8)
    ^bb7:  // 2 preds: ^bb6, ^bb7
      aie.use_lock(%C_L1L2_1_3_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L1L2_1_3_buff_0 : memref<64x64xbf16>, 0, 4096) {bd_id = 4 : i32, next_bd_id = 4 : i32}
      aie.use_lock(%C_L1L2_1_3_prod_lock_0, Release, 1)
      aie.next_bd ^bb7
    ^bb8:  // pred: ^bb6
      aie.end
    }
    %mem_2_5 = aie.mem(%tile_2_5) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%A_L2L1_3_2_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L2L1_3_2_cons_buff_0 : memref<64x64xbf16>, 0, 4096) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%A_L2L1_3_2_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%A_L2L1_3_2_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L2L1_3_2_cons_buff_1 : memref<64x64xbf16>, 0, 4096) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%A_L2L1_3_2_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%B_L2L1_2_3_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L2L1_2_3_cons_buff_0 : memref<64x64xbf16>, 0, 4096) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%B_L2L1_2_3_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%B_L2L1_2_3_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L2L1_2_3_cons_buff_1 : memref<64x64xbf16>, 0, 4096) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%B_L2L1_2_3_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      %2 = aie.dma_start(MM2S, 0, ^bb7, ^bb8)
    ^bb7:  // 2 preds: ^bb6, ^bb7
      aie.use_lock(%C_L1L2_2_3_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L1L2_2_3_buff_0 : memref<64x64xbf16>, 0, 4096) {bd_id = 4 : i32, next_bd_id = 4 : i32}
      aie.use_lock(%C_L1L2_2_3_prod_lock_0, Release, 1)
      aie.next_bd ^bb7
    ^bb8:  // pred: ^bb6
      aie.end
    }
    %mem_3_5 = aie.mem(%tile_3_5) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%A_L2L1_3_3_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L2L1_3_3_cons_buff_0 : memref<64x64xbf16>, 0, 4096) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%A_L2L1_3_3_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%A_L2L1_3_3_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%A_L2L1_3_3_cons_buff_1 : memref<64x64xbf16>, 0, 4096) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%A_L2L1_3_3_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%B_L2L1_3_3_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L2L1_3_3_cons_buff_0 : memref<64x64xbf16>, 0, 4096) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%B_L2L1_3_3_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%B_L2L1_3_3_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%B_L2L1_3_3_cons_buff_1 : memref<64x64xbf16>, 0, 4096) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%B_L2L1_3_3_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      %2 = aie.dma_start(MM2S, 0, ^bb7, ^bb8)
    ^bb7:  // 2 preds: ^bb6, ^bb7
      aie.use_lock(%C_L1L2_3_3_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%C_L1L2_3_3_buff_0 : memref<64x64xbf16>, 0, 4096) {bd_id = 4 : i32, next_bd_id = 4 : i32}
      aie.use_lock(%C_L1L2_3_3_prod_lock_0, Release, 1)
      aie.next_bd ^bb7
    ^bb8:  // pred: ^bb6
      aie.end
    }
    aie.shim_dma_allocation @A_L3L2_3_shim_alloc(%shim_noc_tile_3_0, MM2S, 0)
    aie.shim_dma_allocation @B_L3L2_0_shim_alloc(%shim_noc_tile_0_0, MM2S, 1)
    aie.shim_dma_allocation @B_L3L2_1_shim_alloc(%shim_noc_tile_1_0, MM2S, 1)
    aie.shim_dma_allocation @B_L3L2_2_shim_alloc(%shim_noc_tile_2_0, MM2S, 1)
    aie.shim_dma_allocation @B_L3L2_3_shim_alloc(%shim_noc_tile_3_0, MM2S, 1)
    aie.shim_dma_allocation @C_L2L3_0_shim_alloc(%shim_noc_tile_0_0, S2MM, 0)
    aie.shim_dma_allocation @C_L2L3_1_shim_alloc(%shim_noc_tile_1_0, S2MM, 0)
    aie.shim_dma_allocation @C_L2L3_2_shim_alloc(%shim_noc_tile_2_0, S2MM, 0)
    aie.shim_dma_allocation @C_L2L3_3_shim_alloc(%shim_noc_tile_3_0, S2MM, 0)
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
  }
}

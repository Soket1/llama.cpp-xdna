module {
  aie.device(npu2) {
    %tile_0_2 = aie.tile(0, 2)
    %tile_1_2 = aie.tile(1, 2)
    %tile_2_2 = aie.tile(2, 2)
    %tile_3_2 = aie.tile(3, 2)
    %tile_0_3 = aie.tile(0, 3)
    %tile_1_3 = aie.tile(1, 3)
    %tile_2_3 = aie.tile(2, 3)
    %tile_3_3 = aie.tile(3, 3)
    %tile_0_4 = aie.tile(0, 4)
    %tile_1_4 = aie.tile(1, 4)
    %tile_2_4 = aie.tile(2, 4)
    %tile_3_4 = aie.tile(3, 4)
    %tile_0_5 = aie.tile(0, 5)
    %tile_1_5 = aie.tile(1, 5)
    %tile_2_5 = aie.tile(2, 5)
    %tile_3_5 = aie.tile(3, 5)
    %mem_tile_0_1 = aie.tile(0, 1)
    %mem_tile_1_1 = aie.tile(1, 1)
    %mem_tile_2_1 = aie.tile(2, 1)
    %mem_tile_3_1 = aie.tile(3, 1)
    %shim_noc_tile_0_0 = aie.tile(0, 0)
    %shim_noc_tile_1_0 = aie.tile(1, 0)
    %shim_noc_tile_2_0 = aie.tile(2, 0)
    %shim_noc_tile_3_0 = aie.tile(3, 0)
    aie.objectfifo @A_L2L1_0(%mem_tile_0_1 dimensionsToStream [<size = 8, stride = 512>, <size = 8, stride = 8>, <size = 8, stride = 64>, <size = 8, stride = 1>], {%tile_0_2, %tile_1_2, %tile_2_2, %tile_3_2}, 2 : i32) : !aie.objectfifo<memref<64x64xbf16>> 
    aie.objectfifo @A_L3L2_0(%shim_noc_tile_0_0, {%mem_tile_0_1}, 2 : i32) : !aie.objectfifo<memref<4096xbf16>> 
    aie.objectfifo.link [@A_L3L2_0] -> [@A_L2L1_0]([] [0])
    aie.objectfifo @A_L2L1_1(%mem_tile_1_1 dimensionsToStream [<size = 8, stride = 512>, <size = 8, stride = 8>, <size = 8, stride = 64>, <size = 8, stride = 1>], {%tile_0_3, %tile_1_3, %tile_2_3, %tile_3_3}, 2 : i32) : !aie.objectfifo<memref<64x64xbf16>> 
    aie.objectfifo @A_L3L2_1(%shim_noc_tile_1_0, {%mem_tile_1_1}, 2 : i32) : !aie.objectfifo<memref<4096xbf16>> 
    aie.objectfifo.link [@A_L3L2_1] -> [@A_L2L1_1]([] [0])
    aie.objectfifo @A_L2L1_2(%mem_tile_2_1 dimensionsToStream [<size = 8, stride = 512>, <size = 8, stride = 8>, <size = 8, stride = 64>, <size = 8, stride = 1>], {%tile_0_4, %tile_1_4, %tile_2_4, %tile_3_4}, 2 : i32) : !aie.objectfifo<memref<64x64xbf16>> 
    aie.objectfifo @A_L3L2_2(%shim_noc_tile_2_0, {%mem_tile_2_1}, 2 : i32) : !aie.objectfifo<memref<4096xbf16>> 
    aie.objectfifo.link [@A_L3L2_2] -> [@A_L2L1_2]([] [0])
    aie.objectfifo @A_L2L1_3(%mem_tile_3_1 dimensionsToStream [<size = 8, stride = 512>, <size = 8, stride = 8>, <size = 8, stride = 64>, <size = 8, stride = 1>], {%tile_0_5, %tile_1_5, %tile_2_5, %tile_3_5}, 2 : i32) : !aie.objectfifo<memref<64x64xbf16>> 
    aie.objectfifo @A_L3L2_3(%shim_noc_tile_3_0, {%mem_tile_3_1}, 2 : i32) : !aie.objectfifo<memref<4096xbf16>> 
    aie.objectfifo.link [@A_L3L2_3] -> [@A_L2L1_3]([] [0])
    aie.objectfifo @B_L2L1_0(%mem_tile_0_1 dimensionsToStream [<size = 8, stride = 512>, <size = 8, stride = 8>, <size = 8, stride = 64>, <size = 8, stride = 1>], {%tile_0_2, %tile_0_3, %tile_0_4, %tile_0_5}, 2 : i32) : !aie.objectfifo<memref<64x64xbf16>> 
    aie.objectfifo @B_L3L2_0(%shim_noc_tile_0_0, {%mem_tile_0_1}, 2 : i32) : !aie.objectfifo<memref<4096xbf16>> 
    aie.objectfifo.link [@B_L3L2_0] -> [@B_L2L1_0]([] [0])
    aie.objectfifo @B_L2L1_1(%mem_tile_1_1 dimensionsToStream [<size = 8, stride = 512>, <size = 8, stride = 8>, <size = 8, stride = 64>, <size = 8, stride = 1>], {%tile_1_2, %tile_1_3, %tile_1_4, %tile_1_5}, 2 : i32) : !aie.objectfifo<memref<64x64xbf16>> 
    aie.objectfifo @B_L3L2_1(%shim_noc_tile_1_0, {%mem_tile_1_1}, 2 : i32) : !aie.objectfifo<memref<4096xbf16>> 
    aie.objectfifo.link [@B_L3L2_1] -> [@B_L2L1_1]([] [0])
    aie.objectfifo @B_L2L1_2(%mem_tile_2_1 dimensionsToStream [<size = 8, stride = 512>, <size = 8, stride = 8>, <size = 8, stride = 64>, <size = 8, stride = 1>], {%tile_2_2, %tile_2_3, %tile_2_4, %tile_2_5}, 2 : i32) : !aie.objectfifo<memref<64x64xbf16>> 
    aie.objectfifo @B_L3L2_2(%shim_noc_tile_2_0, {%mem_tile_2_1}, 2 : i32) : !aie.objectfifo<memref<4096xbf16>> 
    aie.objectfifo.link [@B_L3L2_2] -> [@B_L2L1_2]([] [0])
    aie.objectfifo @B_L2L1_3(%mem_tile_3_1 dimensionsToStream [<size = 8, stride = 512>, <size = 8, stride = 8>, <size = 8, stride = 64>, <size = 8, stride = 1>], {%tile_3_2, %tile_3_3, %tile_3_4, %tile_3_5}, 2 : i32) : !aie.objectfifo<memref<64x64xbf16>> 
    aie.objectfifo @B_L3L2_3(%shim_noc_tile_3_0, {%mem_tile_3_1}, 2 : i32) : !aie.objectfifo<memref<4096xbf16>> 
    aie.objectfifo.link [@B_L3L2_3] -> [@B_L2L1_3]([] [0])
    aie.objectfifo @C_L1L2_0_0(%tile_0_2, {%mem_tile_0_1}, 2 : i32) : !aie.objectfifo<memref<64x64xbf16>> 
    aie.objectfifo @C_L1L2_0_1(%tile_0_3, {%mem_tile_0_1}, 2 : i32) : !aie.objectfifo<memref<64x64xbf16>> 
    aie.objectfifo @C_L1L2_0_2(%tile_0_4, {%mem_tile_0_1}, 2 : i32) : !aie.objectfifo<memref<64x64xbf16>> 
    aie.objectfifo @C_L1L2_0_3(%tile_0_5, {%mem_tile_0_1}, 2 : i32) : !aie.objectfifo<memref<64x64xbf16>> 
    aie.objectfifo @C_L2L3_0(%mem_tile_0_1 dimensionsToStream [<size = 8, stride = 512>, <size = 8, stride = 8>, <size = 8, stride = 64>, <size = 8, stride = 1>], {%shim_noc_tile_0_0}, 2 : i32) : !aie.objectfifo<memref<16384xbf16>> 
    aie.objectfifo.link [@C_L1L2_0_0, @C_L1L2_0_1, @C_L1L2_0_2, @C_L1L2_0_3] -> [@C_L2L3_0]([0, 4096, 8192, 12288] [])
    aie.objectfifo @C_L1L2_1_0(%tile_1_2, {%mem_tile_1_1}, 2 : i32) : !aie.objectfifo<memref<64x64xbf16>> 
    aie.objectfifo @C_L1L2_1_1(%tile_1_3, {%mem_tile_1_1}, 2 : i32) : !aie.objectfifo<memref<64x64xbf16>> 
    aie.objectfifo @C_L1L2_1_2(%tile_1_4, {%mem_tile_1_1}, 2 : i32) : !aie.objectfifo<memref<64x64xbf16>> 
    aie.objectfifo @C_L1L2_1_3(%tile_1_5, {%mem_tile_1_1}, 2 : i32) : !aie.objectfifo<memref<64x64xbf16>> 
    aie.objectfifo @C_L2L3_1(%mem_tile_1_1 dimensionsToStream [<size = 8, stride = 512>, <size = 8, stride = 8>, <size = 8, stride = 64>, <size = 8, stride = 1>], {%shim_noc_tile_1_0}, 2 : i32) : !aie.objectfifo<memref<16384xbf16>> 
    aie.objectfifo.link [@C_L1L2_1_0, @C_L1L2_1_1, @C_L1L2_1_2, @C_L1L2_1_3] -> [@C_L2L3_1]([0, 4096, 8192, 12288] [])
    aie.objectfifo @C_L1L2_2_0(%tile_2_2, {%mem_tile_2_1}, 2 : i32) : !aie.objectfifo<memref<64x64xbf16>> 
    aie.objectfifo @C_L1L2_2_1(%tile_2_3, {%mem_tile_2_1}, 2 : i32) : !aie.objectfifo<memref<64x64xbf16>> 
    aie.objectfifo @C_L1L2_2_2(%tile_2_4, {%mem_tile_2_1}, 2 : i32) : !aie.objectfifo<memref<64x64xbf16>> 
    aie.objectfifo @C_L1L2_2_3(%tile_2_5, {%mem_tile_2_1}, 2 : i32) : !aie.objectfifo<memref<64x64xbf16>> 
    aie.objectfifo @C_L2L3_2(%mem_tile_2_1 dimensionsToStream [<size = 8, stride = 512>, <size = 8, stride = 8>, <size = 8, stride = 64>, <size = 8, stride = 1>], {%shim_noc_tile_2_0}, 2 : i32) : !aie.objectfifo<memref<16384xbf16>> 
    aie.objectfifo.link [@C_L1L2_2_0, @C_L1L2_2_1, @C_L1L2_2_2, @C_L1L2_2_3] -> [@C_L2L3_2]([0, 4096, 8192, 12288] [])
    aie.objectfifo @C_L1L2_3_0(%tile_3_2, {%mem_tile_3_1}, 2 : i32) : !aie.objectfifo<memref<64x64xbf16>> 
    aie.objectfifo @C_L1L2_3_1(%tile_3_3, {%mem_tile_3_1}, 2 : i32) : !aie.objectfifo<memref<64x64xbf16>> 
    aie.objectfifo @C_L1L2_3_2(%tile_3_4, {%mem_tile_3_1}, 2 : i32) : !aie.objectfifo<memref<64x64xbf16>> 
    aie.objectfifo @C_L1L2_3_3(%tile_3_5, {%mem_tile_3_1}, 2 : i32) : !aie.objectfifo<memref<64x64xbf16>> 
    aie.objectfifo @C_L2L3_3(%mem_tile_3_1 dimensionsToStream [<size = 8, stride = 512>, <size = 8, stride = 8>, <size = 8, stride = 64>, <size = 8, stride = 1>], {%shim_noc_tile_3_0}, 2 : i32) : !aie.objectfifo<memref<16384xbf16>> 
    aie.objectfifo.link [@C_L1L2_3_0, @C_L1L2_3_1, @C_L1L2_3_2, @C_L1L2_3_3] -> [@C_L2L3_3]([0, 4096, 8192, 12288] [])
    func.func private @zero_bf16(memref<64x64xbf16>) attributes {link_with = "gemm_64x64x64_0_0_0_1_1.o"}
    func.func private @matmul_bf16_bf16(memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) attributes {link_with = "gemm_64x64x64_0_0_0_1_1.o"}
    %rtp0_0 = aie.buffer(%tile_0_2) {sym_name = "rtp0_0"} : memref<2xi32> = dense<0>
    %rtp0_1 = aie.buffer(%tile_1_2) {sym_name = "rtp0_1"} : memref<2xi32> = dense<0>
    %rtp0_2 = aie.buffer(%tile_2_2) {sym_name = "rtp0_2"} : memref<2xi32> = dense<0>
    %rtp0_3 = aie.buffer(%tile_3_2) {sym_name = "rtp0_3"} : memref<2xi32> = dense<0>
    %rtp1_0 = aie.buffer(%tile_0_3) {sym_name = "rtp1_0"} : memref<2xi32> = dense<0>
    %rtp1_1 = aie.buffer(%tile_1_3) {sym_name = "rtp1_1"} : memref<2xi32> = dense<0>
    %rtp1_2 = aie.buffer(%tile_2_3) {sym_name = "rtp1_2"} : memref<2xi32> = dense<0>
    %rtp1_3 = aie.buffer(%tile_3_3) {sym_name = "rtp1_3"} : memref<2xi32> = dense<0>
    %rtp2_0 = aie.buffer(%tile_0_4) {sym_name = "rtp2_0"} : memref<2xi32> = dense<0>
    %rtp2_1 = aie.buffer(%tile_1_4) {sym_name = "rtp2_1"} : memref<2xi32> = dense<0>
    %rtp2_2 = aie.buffer(%tile_2_4) {sym_name = "rtp2_2"} : memref<2xi32> = dense<0>
    %rtp2_3 = aie.buffer(%tile_3_4) {sym_name = "rtp2_3"} : memref<2xi32> = dense<0>
    %rtp3_0 = aie.buffer(%tile_0_5) {sym_name = "rtp3_0"} : memref<2xi32> = dense<0>
    %rtp3_1 = aie.buffer(%tile_1_5) {sym_name = "rtp3_1"} : memref<2xi32> = dense<0>
    %rtp3_2 = aie.buffer(%tile_2_5) {sym_name = "rtp3_2"} : memref<2xi32> = dense<0>
    %rtp3_3 = aie.buffer(%tile_3_5) {sym_name = "rtp3_3"} : memref<2xi32> = dense<0>
    %lock_0_2 = aie.lock(%tile_0_2)
    %core_0_2 = aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        aie.use_lock(%lock_0_2, Acquire, 1)
        %c0_0 = arith.constant 0 : index
        %0 = memref.load %rtp0_0[%c0_0] : memref<2xi32>
        %c1_1 = arith.constant 1 : index
        %1 = memref.load %rtp0_0[%c1_1] : memref<2xi32>
        %c1_i32 = arith.constant 1 : i32
        %2 = arith.cmpi sgt, %1, %c1_i32 : i32
        %c0_2 = arith.constant 0 : index
        %3 = arith.index_cast %1 : i32 to index
        %c1_3 = arith.constant 1 : index
        scf.for %arg1 = %c0_2 to %3 step %c1_3 {
          %4 = aie.objectfifo.acquire @C_L1L2_0_0(Produce, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
          %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
          func.call @zero_bf16(%5) : (memref<64x64xbf16>) -> ()
          %c0_4 = arith.constant 0 : index
          %6 = arith.index_cast %0 : i32 to index
          %c1_5 = arith.constant 1 : index
          scf.for %arg2 = %c0_4 to %6 step %c1_5 {
            %7 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %8 = aie.objectfifo.subview.access %7[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            %9 = aie.objectfifo.acquire @B_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %10 = aie.objectfifo.subview.access %9[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            func.call @matmul_bf16_bf16(%8, %10, %5) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
            aie.objectfifo.release @A_L2L1_0(Consume, 1)
            aie.objectfifo.release @B_L2L1_0(Consume, 1)
          }
          aie.objectfifo.release @C_L1L2_0_0(Produce, 1)
        }
      }
      aie.end
    } {stack_size = 3328 : i32}
    %lock_1_2 = aie.lock(%tile_1_2)
    %core_1_2 = aie.core(%tile_1_2) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        aie.use_lock(%lock_1_2, Acquire, 1)
        %c0_0 = arith.constant 0 : index
        %0 = memref.load %rtp0_1[%c0_0] : memref<2xi32>
        %c1_1 = arith.constant 1 : index
        %1 = memref.load %rtp0_1[%c1_1] : memref<2xi32>
        %c1_i32 = arith.constant 1 : i32
        %2 = arith.cmpi sgt, %1, %c1_i32 : i32
        %c0_2 = arith.constant 0 : index
        %3 = arith.index_cast %1 : i32 to index
        %c1_3 = arith.constant 1 : index
        scf.for %arg1 = %c0_2 to %3 step %c1_3 {
          %4 = aie.objectfifo.acquire @C_L1L2_1_0(Produce, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
          %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
          func.call @zero_bf16(%5) : (memref<64x64xbf16>) -> ()
          %c0_4 = arith.constant 0 : index
          %6 = arith.index_cast %0 : i32 to index
          %c1_5 = arith.constant 1 : index
          scf.for %arg2 = %c0_4 to %6 step %c1_5 {
            %7 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %8 = aie.objectfifo.subview.access %7[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            %9 = aie.objectfifo.acquire @B_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %10 = aie.objectfifo.subview.access %9[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            func.call @matmul_bf16_bf16(%8, %10, %5) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
            aie.objectfifo.release @A_L2L1_0(Consume, 1)
            aie.objectfifo.release @B_L2L1_1(Consume, 1)
          }
          aie.objectfifo.release @C_L1L2_1_0(Produce, 1)
        }
      }
      aie.end
    } {stack_size = 3328 : i32}
    %lock_2_2 = aie.lock(%tile_2_2)
    %core_2_2 = aie.core(%tile_2_2) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        aie.use_lock(%lock_2_2, Acquire, 1)
        %c0_0 = arith.constant 0 : index
        %0 = memref.load %rtp0_2[%c0_0] : memref<2xi32>
        %c1_1 = arith.constant 1 : index
        %1 = memref.load %rtp0_2[%c1_1] : memref<2xi32>
        %c1_i32 = arith.constant 1 : i32
        %2 = arith.cmpi sgt, %1, %c1_i32 : i32
        %c0_2 = arith.constant 0 : index
        %3 = arith.index_cast %1 : i32 to index
        %c1_3 = arith.constant 1 : index
        scf.for %arg1 = %c0_2 to %3 step %c1_3 {
          %4 = aie.objectfifo.acquire @C_L1L2_2_0(Produce, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
          %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
          func.call @zero_bf16(%5) : (memref<64x64xbf16>) -> ()
          %c0_4 = arith.constant 0 : index
          %6 = arith.index_cast %0 : i32 to index
          %c1_5 = arith.constant 1 : index
          scf.for %arg2 = %c0_4 to %6 step %c1_5 {
            %7 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %8 = aie.objectfifo.subview.access %7[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            %9 = aie.objectfifo.acquire @B_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %10 = aie.objectfifo.subview.access %9[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            func.call @matmul_bf16_bf16(%8, %10, %5) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
            aie.objectfifo.release @A_L2L1_0(Consume, 1)
            aie.objectfifo.release @B_L2L1_2(Consume, 1)
          }
          aie.objectfifo.release @C_L1L2_2_0(Produce, 1)
        }
      }
      aie.end
    } {stack_size = 3328 : i32}
    %lock_3_2 = aie.lock(%tile_3_2)
    %core_3_2 = aie.core(%tile_3_2) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        aie.use_lock(%lock_3_2, Acquire, 1)
        %c0_0 = arith.constant 0 : index
        %0 = memref.load %rtp0_3[%c0_0] : memref<2xi32>
        %c1_1 = arith.constant 1 : index
        %1 = memref.load %rtp0_3[%c1_1] : memref<2xi32>
        %c1_i32 = arith.constant 1 : i32
        %2 = arith.cmpi sgt, %1, %c1_i32 : i32
        %c0_2 = arith.constant 0 : index
        %3 = arith.index_cast %1 : i32 to index
        %c1_3 = arith.constant 1 : index
        scf.for %arg1 = %c0_2 to %3 step %c1_3 {
          %4 = aie.objectfifo.acquire @C_L1L2_3_0(Produce, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
          %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
          func.call @zero_bf16(%5) : (memref<64x64xbf16>) -> ()
          %c0_4 = arith.constant 0 : index
          %6 = arith.index_cast %0 : i32 to index
          %c1_5 = arith.constant 1 : index
          scf.for %arg2 = %c0_4 to %6 step %c1_5 {
            %7 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %8 = aie.objectfifo.subview.access %7[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            %9 = aie.objectfifo.acquire @B_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %10 = aie.objectfifo.subview.access %9[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            func.call @matmul_bf16_bf16(%8, %10, %5) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
            aie.objectfifo.release @A_L2L1_0(Consume, 1)
            aie.objectfifo.release @B_L2L1_3(Consume, 1)
          }
          aie.objectfifo.release @C_L1L2_3_0(Produce, 1)
        }
      }
      aie.end
    } {stack_size = 3328 : i32}
    %lock_0_3 = aie.lock(%tile_0_3)
    %core_0_3 = aie.core(%tile_0_3) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        aie.use_lock(%lock_0_3, Acquire, 1)
        %c0_0 = arith.constant 0 : index
        %0 = memref.load %rtp1_0[%c0_0] : memref<2xi32>
        %c1_1 = arith.constant 1 : index
        %1 = memref.load %rtp1_0[%c1_1] : memref<2xi32>
        %c1_i32 = arith.constant 1 : i32
        %2 = arith.cmpi sgt, %1, %c1_i32 : i32
        %c0_2 = arith.constant 0 : index
        %3 = arith.index_cast %1 : i32 to index
        %c1_3 = arith.constant 1 : index
        scf.for %arg1 = %c0_2 to %3 step %c1_3 {
          %4 = aie.objectfifo.acquire @C_L1L2_0_1(Produce, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
          %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
          func.call @zero_bf16(%5) : (memref<64x64xbf16>) -> ()
          %c0_4 = arith.constant 0 : index
          %6 = arith.index_cast %0 : i32 to index
          %c1_5 = arith.constant 1 : index
          scf.for %arg2 = %c0_4 to %6 step %c1_5 {
            %7 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %8 = aie.objectfifo.subview.access %7[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            %9 = aie.objectfifo.acquire @B_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %10 = aie.objectfifo.subview.access %9[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            func.call @matmul_bf16_bf16(%8, %10, %5) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
            aie.objectfifo.release @A_L2L1_1(Consume, 1)
            aie.objectfifo.release @B_L2L1_0(Consume, 1)
          }
          aie.objectfifo.release @C_L1L2_0_1(Produce, 1)
        }
      }
      aie.end
    } {stack_size = 3328 : i32}
    %lock_1_3 = aie.lock(%tile_1_3)
    %core_1_3 = aie.core(%tile_1_3) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        aie.use_lock(%lock_1_3, Acquire, 1)
        %c0_0 = arith.constant 0 : index
        %0 = memref.load %rtp1_1[%c0_0] : memref<2xi32>
        %c1_1 = arith.constant 1 : index
        %1 = memref.load %rtp1_1[%c1_1] : memref<2xi32>
        %c1_i32 = arith.constant 1 : i32
        %2 = arith.cmpi sgt, %1, %c1_i32 : i32
        %c0_2 = arith.constant 0 : index
        %3 = arith.index_cast %1 : i32 to index
        %c1_3 = arith.constant 1 : index
        scf.for %arg1 = %c0_2 to %3 step %c1_3 {
          %4 = aie.objectfifo.acquire @C_L1L2_1_1(Produce, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
          %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
          func.call @zero_bf16(%5) : (memref<64x64xbf16>) -> ()
          %c0_4 = arith.constant 0 : index
          %6 = arith.index_cast %0 : i32 to index
          %c1_5 = arith.constant 1 : index
          scf.for %arg2 = %c0_4 to %6 step %c1_5 {
            %7 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %8 = aie.objectfifo.subview.access %7[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            %9 = aie.objectfifo.acquire @B_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %10 = aie.objectfifo.subview.access %9[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            func.call @matmul_bf16_bf16(%8, %10, %5) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
            aie.objectfifo.release @A_L2L1_1(Consume, 1)
            aie.objectfifo.release @B_L2L1_1(Consume, 1)
          }
          aie.objectfifo.release @C_L1L2_1_1(Produce, 1)
        }
      }
      aie.end
    } {stack_size = 3328 : i32}
    %lock_2_3 = aie.lock(%tile_2_3)
    %core_2_3 = aie.core(%tile_2_3) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        aie.use_lock(%lock_2_3, Acquire, 1)
        %c0_0 = arith.constant 0 : index
        %0 = memref.load %rtp1_2[%c0_0] : memref<2xi32>
        %c1_1 = arith.constant 1 : index
        %1 = memref.load %rtp1_2[%c1_1] : memref<2xi32>
        %c1_i32 = arith.constant 1 : i32
        %2 = arith.cmpi sgt, %1, %c1_i32 : i32
        %c0_2 = arith.constant 0 : index
        %3 = arith.index_cast %1 : i32 to index
        %c1_3 = arith.constant 1 : index
        scf.for %arg1 = %c0_2 to %3 step %c1_3 {
          %4 = aie.objectfifo.acquire @C_L1L2_2_1(Produce, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
          %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
          func.call @zero_bf16(%5) : (memref<64x64xbf16>) -> ()
          %c0_4 = arith.constant 0 : index
          %6 = arith.index_cast %0 : i32 to index
          %c1_5 = arith.constant 1 : index
          scf.for %arg2 = %c0_4 to %6 step %c1_5 {
            %7 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %8 = aie.objectfifo.subview.access %7[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            %9 = aie.objectfifo.acquire @B_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %10 = aie.objectfifo.subview.access %9[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            func.call @matmul_bf16_bf16(%8, %10, %5) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
            aie.objectfifo.release @A_L2L1_1(Consume, 1)
            aie.objectfifo.release @B_L2L1_2(Consume, 1)
          }
          aie.objectfifo.release @C_L1L2_2_1(Produce, 1)
        }
      }
      aie.end
    } {stack_size = 3328 : i32}
    %lock_3_3 = aie.lock(%tile_3_3)
    %core_3_3 = aie.core(%tile_3_3) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        aie.use_lock(%lock_3_3, Acquire, 1)
        %c0_0 = arith.constant 0 : index
        %0 = memref.load %rtp1_3[%c0_0] : memref<2xi32>
        %c1_1 = arith.constant 1 : index
        %1 = memref.load %rtp1_3[%c1_1] : memref<2xi32>
        %c1_i32 = arith.constant 1 : i32
        %2 = arith.cmpi sgt, %1, %c1_i32 : i32
        %c0_2 = arith.constant 0 : index
        %3 = arith.index_cast %1 : i32 to index
        %c1_3 = arith.constant 1 : index
        scf.for %arg1 = %c0_2 to %3 step %c1_3 {
          %4 = aie.objectfifo.acquire @C_L1L2_3_1(Produce, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
          %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
          func.call @zero_bf16(%5) : (memref<64x64xbf16>) -> ()
          %c0_4 = arith.constant 0 : index
          %6 = arith.index_cast %0 : i32 to index
          %c1_5 = arith.constant 1 : index
          scf.for %arg2 = %c0_4 to %6 step %c1_5 {
            %7 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %8 = aie.objectfifo.subview.access %7[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            %9 = aie.objectfifo.acquire @B_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %10 = aie.objectfifo.subview.access %9[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            func.call @matmul_bf16_bf16(%8, %10, %5) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
            aie.objectfifo.release @A_L2L1_1(Consume, 1)
            aie.objectfifo.release @B_L2L1_3(Consume, 1)
          }
          aie.objectfifo.release @C_L1L2_3_1(Produce, 1)
        }
      }
      aie.end
    } {stack_size = 3328 : i32}
    %lock_0_4 = aie.lock(%tile_0_4)
    %core_0_4 = aie.core(%tile_0_4) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        aie.use_lock(%lock_0_4, Acquire, 1)
        %c0_0 = arith.constant 0 : index
        %0 = memref.load %rtp2_0[%c0_0] : memref<2xi32>
        %c1_1 = arith.constant 1 : index
        %1 = memref.load %rtp2_0[%c1_1] : memref<2xi32>
        %c1_i32 = arith.constant 1 : i32
        %2 = arith.cmpi sgt, %1, %c1_i32 : i32
        %c0_2 = arith.constant 0 : index
        %3 = arith.index_cast %1 : i32 to index
        %c1_3 = arith.constant 1 : index
        scf.for %arg1 = %c0_2 to %3 step %c1_3 {
          %4 = aie.objectfifo.acquire @C_L1L2_0_2(Produce, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
          %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
          func.call @zero_bf16(%5) : (memref<64x64xbf16>) -> ()
          %c0_4 = arith.constant 0 : index
          %6 = arith.index_cast %0 : i32 to index
          %c1_5 = arith.constant 1 : index
          scf.for %arg2 = %c0_4 to %6 step %c1_5 {
            %7 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %8 = aie.objectfifo.subview.access %7[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            %9 = aie.objectfifo.acquire @B_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %10 = aie.objectfifo.subview.access %9[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            func.call @matmul_bf16_bf16(%8, %10, %5) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
            aie.objectfifo.release @A_L2L1_2(Consume, 1)
            aie.objectfifo.release @B_L2L1_0(Consume, 1)
          }
          aie.objectfifo.release @C_L1L2_0_2(Produce, 1)
        }
      }
      aie.end
    } {stack_size = 3328 : i32}
    %lock_1_4 = aie.lock(%tile_1_4)
    %core_1_4 = aie.core(%tile_1_4) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        aie.use_lock(%lock_1_4, Acquire, 1)
        %c0_0 = arith.constant 0 : index
        %0 = memref.load %rtp2_1[%c0_0] : memref<2xi32>
        %c1_1 = arith.constant 1 : index
        %1 = memref.load %rtp2_1[%c1_1] : memref<2xi32>
        %c1_i32 = arith.constant 1 : i32
        %2 = arith.cmpi sgt, %1, %c1_i32 : i32
        %c0_2 = arith.constant 0 : index
        %3 = arith.index_cast %1 : i32 to index
        %c1_3 = arith.constant 1 : index
        scf.for %arg1 = %c0_2 to %3 step %c1_3 {
          %4 = aie.objectfifo.acquire @C_L1L2_1_2(Produce, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
          %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
          func.call @zero_bf16(%5) : (memref<64x64xbf16>) -> ()
          %c0_4 = arith.constant 0 : index
          %6 = arith.index_cast %0 : i32 to index
          %c1_5 = arith.constant 1 : index
          scf.for %arg2 = %c0_4 to %6 step %c1_5 {
            %7 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %8 = aie.objectfifo.subview.access %7[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            %9 = aie.objectfifo.acquire @B_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %10 = aie.objectfifo.subview.access %9[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            func.call @matmul_bf16_bf16(%8, %10, %5) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
            aie.objectfifo.release @A_L2L1_2(Consume, 1)
            aie.objectfifo.release @B_L2L1_1(Consume, 1)
          }
          aie.objectfifo.release @C_L1L2_1_2(Produce, 1)
        }
      }
      aie.end
    } {stack_size = 3328 : i32}
    %lock_2_4 = aie.lock(%tile_2_4)
    %core_2_4 = aie.core(%tile_2_4) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        aie.use_lock(%lock_2_4, Acquire, 1)
        %c0_0 = arith.constant 0 : index
        %0 = memref.load %rtp2_2[%c0_0] : memref<2xi32>
        %c1_1 = arith.constant 1 : index
        %1 = memref.load %rtp2_2[%c1_1] : memref<2xi32>
        %c1_i32 = arith.constant 1 : i32
        %2 = arith.cmpi sgt, %1, %c1_i32 : i32
        %c0_2 = arith.constant 0 : index
        %3 = arith.index_cast %1 : i32 to index
        %c1_3 = arith.constant 1 : index
        scf.for %arg1 = %c0_2 to %3 step %c1_3 {
          %4 = aie.objectfifo.acquire @C_L1L2_2_2(Produce, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
          %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
          func.call @zero_bf16(%5) : (memref<64x64xbf16>) -> ()
          %c0_4 = arith.constant 0 : index
          %6 = arith.index_cast %0 : i32 to index
          %c1_5 = arith.constant 1 : index
          scf.for %arg2 = %c0_4 to %6 step %c1_5 {
            %7 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %8 = aie.objectfifo.subview.access %7[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            %9 = aie.objectfifo.acquire @B_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %10 = aie.objectfifo.subview.access %9[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            func.call @matmul_bf16_bf16(%8, %10, %5) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
            aie.objectfifo.release @A_L2L1_2(Consume, 1)
            aie.objectfifo.release @B_L2L1_2(Consume, 1)
          }
          aie.objectfifo.release @C_L1L2_2_2(Produce, 1)
        }
      }
      aie.end
    } {stack_size = 3328 : i32}
    %lock_3_4 = aie.lock(%tile_3_4)
    %core_3_4 = aie.core(%tile_3_4) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        aie.use_lock(%lock_3_4, Acquire, 1)
        %c0_0 = arith.constant 0 : index
        %0 = memref.load %rtp2_3[%c0_0] : memref<2xi32>
        %c1_1 = arith.constant 1 : index
        %1 = memref.load %rtp2_3[%c1_1] : memref<2xi32>
        %c1_i32 = arith.constant 1 : i32
        %2 = arith.cmpi sgt, %1, %c1_i32 : i32
        %c0_2 = arith.constant 0 : index
        %3 = arith.index_cast %1 : i32 to index
        %c1_3 = arith.constant 1 : index
        scf.for %arg1 = %c0_2 to %3 step %c1_3 {
          %4 = aie.objectfifo.acquire @C_L1L2_3_2(Produce, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
          %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
          func.call @zero_bf16(%5) : (memref<64x64xbf16>) -> ()
          %c0_4 = arith.constant 0 : index
          %6 = arith.index_cast %0 : i32 to index
          %c1_5 = arith.constant 1 : index
          scf.for %arg2 = %c0_4 to %6 step %c1_5 {
            %7 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %8 = aie.objectfifo.subview.access %7[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            %9 = aie.objectfifo.acquire @B_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %10 = aie.objectfifo.subview.access %9[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            func.call @matmul_bf16_bf16(%8, %10, %5) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
            aie.objectfifo.release @A_L2L1_2(Consume, 1)
            aie.objectfifo.release @B_L2L1_3(Consume, 1)
          }
          aie.objectfifo.release @C_L1L2_3_2(Produce, 1)
        }
      }
      aie.end
    } {stack_size = 3328 : i32}
    %lock_0_5 = aie.lock(%tile_0_5)
    %core_0_5 = aie.core(%tile_0_5) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        aie.use_lock(%lock_0_5, Acquire, 1)
        %c0_0 = arith.constant 0 : index
        %0 = memref.load %rtp3_0[%c0_0] : memref<2xi32>
        %c1_1 = arith.constant 1 : index
        %1 = memref.load %rtp3_0[%c1_1] : memref<2xi32>
        %c1_i32 = arith.constant 1 : i32
        %2 = arith.cmpi sgt, %1, %c1_i32 : i32
        %c0_2 = arith.constant 0 : index
        %3 = arith.index_cast %1 : i32 to index
        %c1_3 = arith.constant 1 : index
        scf.for %arg1 = %c0_2 to %3 step %c1_3 {
          %4 = aie.objectfifo.acquire @C_L1L2_0_3(Produce, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
          %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
          func.call @zero_bf16(%5) : (memref<64x64xbf16>) -> ()
          %c0_4 = arith.constant 0 : index
          %6 = arith.index_cast %0 : i32 to index
          %c1_5 = arith.constant 1 : index
          scf.for %arg2 = %c0_4 to %6 step %c1_5 {
            %7 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %8 = aie.objectfifo.subview.access %7[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            %9 = aie.objectfifo.acquire @B_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %10 = aie.objectfifo.subview.access %9[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            func.call @matmul_bf16_bf16(%8, %10, %5) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
            aie.objectfifo.release @A_L2L1_3(Consume, 1)
            aie.objectfifo.release @B_L2L1_0(Consume, 1)
          }
          aie.objectfifo.release @C_L1L2_0_3(Produce, 1)
        }
      }
      aie.end
    } {stack_size = 3328 : i32}
    %lock_1_5 = aie.lock(%tile_1_5)
    %core_1_5 = aie.core(%tile_1_5) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        aie.use_lock(%lock_1_5, Acquire, 1)
        %c0_0 = arith.constant 0 : index
        %0 = memref.load %rtp3_1[%c0_0] : memref<2xi32>
        %c1_1 = arith.constant 1 : index
        %1 = memref.load %rtp3_1[%c1_1] : memref<2xi32>
        %c1_i32 = arith.constant 1 : i32
        %2 = arith.cmpi sgt, %1, %c1_i32 : i32
        %c0_2 = arith.constant 0 : index
        %3 = arith.index_cast %1 : i32 to index
        %c1_3 = arith.constant 1 : index
        scf.for %arg1 = %c0_2 to %3 step %c1_3 {
          %4 = aie.objectfifo.acquire @C_L1L2_1_3(Produce, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
          %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
          func.call @zero_bf16(%5) : (memref<64x64xbf16>) -> ()
          %c0_4 = arith.constant 0 : index
          %6 = arith.index_cast %0 : i32 to index
          %c1_5 = arith.constant 1 : index
          scf.for %arg2 = %c0_4 to %6 step %c1_5 {
            %7 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %8 = aie.objectfifo.subview.access %7[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            %9 = aie.objectfifo.acquire @B_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %10 = aie.objectfifo.subview.access %9[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            func.call @matmul_bf16_bf16(%8, %10, %5) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
            aie.objectfifo.release @A_L2L1_3(Consume, 1)
            aie.objectfifo.release @B_L2L1_1(Consume, 1)
          }
          aie.objectfifo.release @C_L1L2_1_3(Produce, 1)
        }
      }
      aie.end
    } {stack_size = 3328 : i32}
    %lock_2_5 = aie.lock(%tile_2_5)
    %core_2_5 = aie.core(%tile_2_5) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        aie.use_lock(%lock_2_5, Acquire, 1)
        %c0_0 = arith.constant 0 : index
        %0 = memref.load %rtp3_2[%c0_0] : memref<2xi32>
        %c1_1 = arith.constant 1 : index
        %1 = memref.load %rtp3_2[%c1_1] : memref<2xi32>
        %c1_i32 = arith.constant 1 : i32
        %2 = arith.cmpi sgt, %1, %c1_i32 : i32
        %c0_2 = arith.constant 0 : index
        %3 = arith.index_cast %1 : i32 to index
        %c1_3 = arith.constant 1 : index
        scf.for %arg1 = %c0_2 to %3 step %c1_3 {
          %4 = aie.objectfifo.acquire @C_L1L2_2_3(Produce, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
          %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
          func.call @zero_bf16(%5) : (memref<64x64xbf16>) -> ()
          %c0_4 = arith.constant 0 : index
          %6 = arith.index_cast %0 : i32 to index
          %c1_5 = arith.constant 1 : index
          scf.for %arg2 = %c0_4 to %6 step %c1_5 {
            %7 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %8 = aie.objectfifo.subview.access %7[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            %9 = aie.objectfifo.acquire @B_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %10 = aie.objectfifo.subview.access %9[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            func.call @matmul_bf16_bf16(%8, %10, %5) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
            aie.objectfifo.release @A_L2L1_3(Consume, 1)
            aie.objectfifo.release @B_L2L1_2(Consume, 1)
          }
          aie.objectfifo.release @C_L1L2_2_3(Produce, 1)
        }
      }
      aie.end
    } {stack_size = 3328 : i32}
    %lock_3_5 = aie.lock(%tile_3_5)
    %core_3_5 = aie.core(%tile_3_5) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        aie.use_lock(%lock_3_5, Acquire, 1)
        %c0_0 = arith.constant 0 : index
        %0 = memref.load %rtp3_3[%c0_0] : memref<2xi32>
        %c1_1 = arith.constant 1 : index
        %1 = memref.load %rtp3_3[%c1_1] : memref<2xi32>
        %c1_i32 = arith.constant 1 : i32
        %2 = arith.cmpi sgt, %1, %c1_i32 : i32
        %c0_2 = arith.constant 0 : index
        %3 = arith.index_cast %1 : i32 to index
        %c1_3 = arith.constant 1 : index
        scf.for %arg1 = %c0_2 to %3 step %c1_3 {
          %4 = aie.objectfifo.acquire @C_L1L2_3_3(Produce, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
          %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
          func.call @zero_bf16(%5) : (memref<64x64xbf16>) -> ()
          %c0_4 = arith.constant 0 : index
          %6 = arith.index_cast %0 : i32 to index
          %c1_5 = arith.constant 1 : index
          scf.for %arg2 = %c0_4 to %6 step %c1_5 {
            %7 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %8 = aie.objectfifo.subview.access %7[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            %9 = aie.objectfifo.acquire @B_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %10 = aie.objectfifo.subview.access %9[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            func.call @matmul_bf16_bf16(%8, %10, %5) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
            aie.objectfifo.release @A_L2L1_3(Consume, 1)
            aie.objectfifo.release @B_L2L1_3(Consume, 1)
          }
          aie.objectfifo.release @C_L1L2_3_3(Produce, 1)
        }
      }
      aie.end
    } {stack_size = 3328 : i32}
    aie.runtime_sequence(%arg0: memref<65536xbf16>, %arg1: memref<131072xbf16>, %arg2: memref<131072xbf16>) {
      aiex.npu.rtp_write(@rtp0_0, 0, 4)
      aiex.npu.rtp_write(@rtp0_0, 1, 2)
      aiex.npu.rtp_write(@rtp0_1, 0, 4)
      aiex.npu.rtp_write(@rtp0_1, 1, 2)
      aiex.npu.rtp_write(@rtp0_2, 0, 4)
      aiex.npu.rtp_write(@rtp0_2, 1, 2)
      aiex.npu.rtp_write(@rtp0_3, 0, 4)
      aiex.npu.rtp_write(@rtp0_3, 1, 2)
      aiex.npu.rtp_write(@rtp1_0, 0, 4)
      aiex.npu.rtp_write(@rtp1_0, 1, 2)
      aiex.npu.rtp_write(@rtp1_1, 0, 4)
      aiex.npu.rtp_write(@rtp1_1, 1, 2)
      aiex.npu.rtp_write(@rtp1_2, 0, 4)
      aiex.npu.rtp_write(@rtp1_2, 1, 2)
      aiex.npu.rtp_write(@rtp1_3, 0, 4)
      aiex.npu.rtp_write(@rtp1_3, 1, 2)
      aiex.npu.rtp_write(@rtp2_0, 0, 4)
      aiex.npu.rtp_write(@rtp2_0, 1, 2)
      aiex.npu.rtp_write(@rtp2_1, 0, 4)
      aiex.npu.rtp_write(@rtp2_1, 1, 2)
      aiex.npu.rtp_write(@rtp2_2, 0, 4)
      aiex.npu.rtp_write(@rtp2_2, 1, 2)
      aiex.npu.rtp_write(@rtp2_3, 0, 4)
      aiex.npu.rtp_write(@rtp2_3, 1, 2)
      aiex.npu.rtp_write(@rtp3_0, 0, 4)
      aiex.npu.rtp_write(@rtp3_0, 1, 2)
      aiex.npu.rtp_write(@rtp3_1, 0, 4)
      aiex.npu.rtp_write(@rtp3_1, 1, 2)
      aiex.npu.rtp_write(@rtp3_2, 0, 4)
      aiex.npu.rtp_write(@rtp3_2, 1, 2)
      aiex.npu.rtp_write(@rtp3_3, 0, 4)
      aiex.npu.rtp_write(@rtp3_3, 1, 2)
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
      %0 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%arg2 : memref<131072xbf16>, 0, 32768, [<size = 1, stride = 0>, <size = 2, stride = 256>, <size = 256, stride = 512>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%0)
      %1 = aiex.dma_configure_task_for @A_L3L2_0 {
        aie.dma_bd(%arg0 : memref<65536xbf16>, 0, 16384, [<size = 2, stride = 0>, <size = 4, stride = 64>, <size = 64, stride = 256>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {repeat_count = 1 : i32}
      aiex.dma_start_task(%1)
      %2 = aiex.dma_configure_task_for @B_L3L2_0 {
        aie.dma_bd(%arg1 : memref<131072xbf16>, 0, 16384, [<size = 2, stride = 256>, <size = 4, stride = 32768>, <size = 64, stride = 512>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {repeat_count = 1 : i32}
      aiex.dma_start_task(%2)
      %3 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%arg2 : memref<131072xbf16>, 64, 32768, [<size = 1, stride = 0>, <size = 2, stride = 256>, <size = 256, stride = 512>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%3)
      %4 = aiex.dma_configure_task_for @A_L3L2_1 {
        aie.dma_bd(%arg0 : memref<65536xbf16>, 16384, 16384, [<size = 2, stride = 0>, <size = 4, stride = 64>, <size = 64, stride = 256>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {repeat_count = 1 : i32}
      aiex.dma_start_task(%4)
      %5 = aiex.dma_configure_task_for @B_L3L2_1 {
        aie.dma_bd(%arg1 : memref<131072xbf16>, 64, 16384, [<size = 2, stride = 256>, <size = 4, stride = 32768>, <size = 64, stride = 512>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {repeat_count = 1 : i32}
      aiex.dma_start_task(%5)
      %6 = aiex.dma_configure_task_for @C_L2L3_2 {
        aie.dma_bd(%arg2 : memref<131072xbf16>, 128, 32768, [<size = 1, stride = 0>, <size = 2, stride = 256>, <size = 256, stride = 512>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%6)
      %7 = aiex.dma_configure_task_for @A_L3L2_2 {
        aie.dma_bd(%arg0 : memref<65536xbf16>, 32768, 16384, [<size = 2, stride = 0>, <size = 4, stride = 64>, <size = 64, stride = 256>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {repeat_count = 1 : i32}
      aiex.dma_start_task(%7)
      %8 = aiex.dma_configure_task_for @B_L3L2_2 {
        aie.dma_bd(%arg1 : memref<131072xbf16>, 128, 16384, [<size = 2, stride = 256>, <size = 4, stride = 32768>, <size = 64, stride = 512>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {repeat_count = 1 : i32}
      aiex.dma_start_task(%8)
      %9 = aiex.dma_configure_task_for @C_L2L3_3 {
        aie.dma_bd(%arg2 : memref<131072xbf16>, 192, 32768, [<size = 1, stride = 0>, <size = 2, stride = 256>, <size = 256, stride = 512>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%9)
      %10 = aiex.dma_configure_task_for @A_L3L2_3 {
        aie.dma_bd(%arg0 : memref<65536xbf16>, 49152, 16384, [<size = 2, stride = 0>, <size = 4, stride = 64>, <size = 64, stride = 256>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {repeat_count = 1 : i32}
      aiex.dma_start_task(%10)
      %11 = aiex.dma_configure_task_for @B_L3L2_3 {
        aie.dma_bd(%arg1 : memref<131072xbf16>, 192, 16384, [<size = 2, stride = 256>, <size = 4, stride = 32768>, <size = 64, stride = 512>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {repeat_count = 1 : i32}
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
  }
}

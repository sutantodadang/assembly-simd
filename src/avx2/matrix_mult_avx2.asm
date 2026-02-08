;==============================================================================
;                    SIMD MATRIX MULTIPLICATION - AVX2
;==============================================================================
;
; High-performance matrix multiplication using AVX2 (256-bit) SIMD instructions.
; Computes C = A × B for single-precision floating-point matrices.
;
; This implementation uses:
;   - FMA (Fused Multiply-Add) instructions for maximum throughput
;   - Register blocking to minimize memory access
;   - Row-wise iteration for optimal cache performance
;   - Edge handling for non-multiple-of-8 dimensions
;
; Supported calling conventions:
;   - System V AMD64 (Linux, macOS, BSD)
;   - Microsoft x64 (Windows) - controlled by WIN64 define
;
; Author: SIMD Matrix Multiplication Project
; License: MIT
;
;==============================================================================

; Enable for Windows calling convention (auto-detected)
%define WIN64

;------------------------------------------------------------------------------
; Section: Read-only data
;------------------------------------------------------------------------------
section .rodata
    align 32
    ; Mask for handling edge cases (last 1-7 floats)
    edge_mask:
        dd 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF
        dd 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF
        dd 0x00000000, 0x00000000, 0x00000000, 0x00000000
        dd 0x00000000, 0x00000000, 0x00000000, 0x00000000

;------------------------------------------------------------------------------
; Section: Code
;------------------------------------------------------------------------------
section .text

;==============================================================================
; Function: matrix_mult_avx2
;==============================================================================
;
; Computes C = A × B for float matrices.
;
; Parameters (System V AMD64 ABI):
;   rdi = A    - Pointer to matrix A (M × K), row-major order
;   rsi = B    - Pointer to matrix B (K × N), row-major order
;   rdx = C    - Pointer to result matrix C (M × N), row-major order
;   rcx = M    - Number of rows in A and C
;   r8  = N    - Number of columns in B and C
;   r9  = K    - Number of columns in A / rows in B
;
; Parameters (Microsoft x64 ABI):
;   rcx = A    - Pointer to matrix A
;   rdx = B    - Pointer to matrix B
;   r8  = C    - Pointer to result matrix C
;   r9  = M    - Number of rows in A and C
;   [rsp+40] = N   - Number of columns in B and C
;   [rsp+48] = K   - Number of columns in A / rows in B
;
; Returns: Nothing (C is modified in place)
;
; Memory requirements:
;   - For best performance, all matrices should be 32-byte aligned
;   - K should be a multiple of 8 for best performance
;
;==============================================================================

global matrix_mult_avx2
matrix_mult_avx2:
    ;--------------------------------------------------------------------------
    ; PROLOGUE: Save callee-saved registers and set up stack frame
    ;--------------------------------------------------------------------------
    ; Per System V AMD64 ABI, we must preserve: rbx, rbp, r12-r15
    ; YMM6-YMM15 are caller-saved (no need to preserve on Linux/macOS)
    ; For Windows, XMM6-XMM15 are callee-saved (need to preserve)
    ;--------------------------------------------------------------------------
    
    push    rbp                         ; Save base pointer
    mov     rbp, rsp                    ; Set up stack frame
    push    rbx                         ; Save callee-saved registers
    push    r12
    push    r13
    push    r14
    push    r15
    
%ifdef WIN64
    ; Windows: Allocate space and preserve RDI/RSI (callee-saved)
    push    rdi
    push    rsi
    
    ; Stack alignment check:
    ; 5 previous pushes = 40 bytes
    ; 2 new pushes = 16 bytes
    ; Total pushes = 56 bytes (8 mod 16)
    ; Need sub rsp to be 8 mod 16 to align to 16 bytes.
    ; 248 = 16*15 + 8. Correct.
    
    sub     rsp, 248                    ; Space for locals + 10 XMM registers + alignment
    
    ; Save XMM6-XMM15 at bottom of stack (below local variables)
    vmovaps [rsp],      xmm6
    vmovaps [rsp + 16], xmm7
    vmovaps [rsp + 32], xmm8
    vmovaps [rsp + 48], xmm9
    vmovaps [rsp + 64], xmm10
    vmovaps [rsp + 80], xmm11
    vmovaps [rsp + 96], xmm12
    vmovaps [rsp + 112], xmm13
    vmovaps [rsp + 128], xmm14
    vmovaps [rsp + 144], xmm15
    
    ; Remap Windows x64 parameters to match System V layout
    ; Windows: rcx=A, rdx=B, r8=C, r9=M, [rbp+48]=N, [rbp+56]=K
    mov     rdi, rcx                    ; A
    mov     rsi, rdx                    ; B
    mov     rdx, r8                     ; C
    mov     rcx, r9                     ; M
    mov     r8,  [rbp + 48]             ; N (5th param, after shadow space)
    mov     r9,  [rbp + 56]             ; K (6th param)
%else
    ; Linux/macOS: Just allocate space for local variables
    sub     rsp, 80
%endif

    ;--------------------------------------------------------------------------
    ; PARAMETER SETUP: Store parameters in registers we'll use
    ;--------------------------------------------------------------------------
    ; Register allocation for the main computation:
    ;   rdi = A base pointer (preserved)
    ;   rsi = B base pointer (preserved)
    ;   rdx = C base pointer (preserved)
    ;   rcx = M (row count)
    ;   r8  = N (column count)
    ;   r9  = K (inner dimension)
    ;   r10 = A row stride (K * 4 bytes)
    ;   r11 = B row stride (N * 4 bytes)
    ;   r12 = C row stride (N * 4 bytes) 
    ;   r13 = Current row index (i)
    ;   r14 = Current column index (j)
    ;   r15 = Current inner index (k)
    ;   rbx = Temporary for address calculations
    ;--------------------------------------------------------------------------
    
    ; Store original pointers and dimensions
    mov     [rbp - 64], rdi             ; Save A base
    mov     [rbp - 72], rsi             ; Save B base
    mov     [rbp - 80], rdx             ; Save C base
    mov     [rbp - 88], rcx             ; Save M
    mov     [rbp - 96], r8              ; Save N
    mov     [rbp - 104], r9              ; Save K
    
    ; Calculate row strides (in bytes)
    mov     r10, r9                      
    shl     r10, 2                       ; r10 = K * 4 = A row stride
    mov     r11, r8
    shl     r11, 2                       ; r11 = N * 4 = B row stride
    mov     r12, r11                     ; r12 = N * 4 = C row stride
    
    ; Calculate how many full 8-column blocks (for vectorized inner loop)
    mov     rax, r8                      ; N
    shr     rax, 3                       ; N / 8 = number of full YMM blocks
    mov     [rbp - 112], rax              ; Save block count
    
    ; Calculate remainder columns (N mod 8)
    mov     rax, r8
    and     rax, 7                       ; N mod 8
    mov     [rbp - 120], rax             ; Save remainder
    
    ;--------------------------------------------------------------------------
    ; MAIN COMPUTATION: Register-blocked tiled matrix multiplication
    ;--------------------------------------------------------------------------
    ; OPTIMIZED LOOP ORDER: i → j_tile → k → j_inner
    ; 
    ; Algorithm (register blocking with 4×8 = 32 column tiles):
    ;   for i = 0 to M-1:
    ;       for j_tile = 0 to N (step 32):         ; Process 32 columns
    ;           acc0..acc3 = 0                      ; 4 YMM accumulators
    ;           for k = 0 to K-1:
    ;               a_ik = broadcast A[i][k]
    ;               for lane = 0 to 3:              ; 4 × 8 = 32 columns
    ;                   acc[lane] += a_ik * B[k][j_tile + lane*8 : +8]
    ;           store acc0..acc3 to C[i][j_tile:j_tile+32]
    ;
    ; Benefits:
    ; 1. C stays in YMM registers across entire k-loop (no load-store per k)
    ; 2. B is still accessed sequentially within each k iteration
    ; 3. High FMA throughput (4 FMAs per k iteration)
    ;--------------------------------------------------------------------------
    
    xor     r13, r13                     ; i = 0 (row counter)
    
.row_loop:
    ;==========================================================================
    ; ROW LOOP: Process one row of C at a time
    ;==========================================================================
    cmp     r13, [rbp - 88]              ; Compare i with M
    jge     .done                        ; If i >= M, exit loop
    
    ;--------------------------------------------------------------------------
    ; Calculate A row pointer: A_row = A + i * K * 4
    ;--------------------------------------------------------------------------
    mov     rax, r13                     ; i
    imul    rax, r10                     ; i * A_stride (K * 4)
    mov     rbx, [rbp - 64]              ; A base
    add     rbx, rax                     ; A_row = A + i * K * 4
    mov     [rbp - 128], rbx             ; Save A row pointer
    
    ;--------------------------------------------------------------------------
    ; Calculate C row pointer: C_row = C + i * N * 4
    ;--------------------------------------------------------------------------
    mov     rax, r13                     ; i
    imul    rax, r12                     ; i * C_stride (N * 4)
    mov     r14, [rbp - 80]              ; C base
    add     r14, rax                     ; C_row = C + i * N * 4
    
    ;--------------------------------------------------------------------------
    ; Calculate number of 32-column tiles
    ;--------------------------------------------------------------------------
    mov     rax, [rbp - 96]              ; N
    shr     rax, 5                       ; N / 32 = number of full tiles
    mov     [rbp - 136], rax             ; Save tile count
    
    xor     r15, r15                     ; j_tile = 0 (column tile counter)
    
.tile_loop:
    ;==========================================================================
    ; TILE LOOP: Process 32 columns (4 YMM registers) at a time
    ;==========================================================================
    mov     rax, r15
    shr     rax, 5                       ; j_tile / 32 = current tile
    cmp     rax, [rbp - 136]             ; Compare with tile count
    jge     .tile_remainder              ; Handle remaining columns
    
    ;--------------------------------------------------------------------------
    ; Initialize 4 YMM accumulators to zero
    ;--------------------------------------------------------------------------
    vxorps  ymm0, ymm0, ymm0             ; C[i][j:j+8]
    vxorps  ymm1, ymm1, ymm1             ; C[i][j+8:j+16]
    vxorps  ymm2, ymm2, ymm2             ; C[i][j+16:j+24]
    vxorps  ymm3, ymm3, ymm3             ; C[i][j+24:j+32]
    
    ;--------------------------------------------------------------------------
    ; K-loop: Iterate over entire inner dimension with accumulators in regs
    ;--------------------------------------------------------------------------
    mov     rbx, [rbp - 128]             ; A_row pointer
    mov     r9, [rbp - 104]              ; K
    
    ; Calculate B column pointer: B[0][j]
    mov     rcx, [rbp - 72]              ; B base
    mov     rax, r15
    shl     rax, 2                       ; j * 4
    add     rcx, rax                     ; B[0][j]
    
.k_loop_tiled:
    test    r9, r9
    jz      .k_done_tiled
    
    ;--------------------------------------------------------------------------
    ; Broadcast A[i][k] to ymm4
    ;--------------------------------------------------------------------------
    vbroadcastss ymm4, [rbx]             ; ymm4 = A[i][k] broadcast
    
    ;--------------------------------------------------------------------------
    ; FMA for 4 column blocks (sequential B access!)
    ;--------------------------------------------------------------------------
    vfmadd231ps ymm0, ymm4, [rcx]        ; C[i][j:j+8] += A * B
    vfmadd231ps ymm1, ymm4, [rcx + 32]   ; C[i][j+8:j+16] += A * B
    vfmadd231ps ymm2, ymm4, [rcx + 64]   ; C[i][j+16:j+24] += A * B
    vfmadd231ps ymm3, ymm4, [rcx + 96]   ; C[i][j+24:j+32] += A * B
    
    ;--------------------------------------------------------------------------
    ; Advance pointers
    ;--------------------------------------------------------------------------
    add     rbx, 4                       ; A pointer += 1 element
    add     rcx, r11                     ; B pointer += N elements (next row)
    dec     r9
    jnz     .k_loop_tiled
    
.k_done_tiled:
    ;--------------------------------------------------------------------------
    ; Store 4 YMM accumulators to C
    ;--------------------------------------------------------------------------
    vmovups [r14], ymm0
    vmovups [r14 + 32], ymm1
    vmovups [r14 + 64], ymm2
    vmovups [r14 + 96], ymm3
    
    add     r14, 128                     ; C pointer += 32 floats
    add     r15, 32                      ; j += 32
    jmp     .tile_loop
    
.tile_remainder:
    ;==========================================================================
    ; REMAINDER: Handle remaining columns (N mod 32)
    ; Process one 8-column block at a time
    ;==========================================================================
    mov     rax, [rbp - 96]              ; N
    and     rax, 31                      ; N mod 32
    test    rax, rax
    jz      .row_next                    ; No remainder
    
    ; Calculate number of remaining 8-column blocks
    mov     rdx, rax
    shr     rdx, 3                       ; remainder / 8
    mov     r8, rax
    and     r8, 7                        ; remainder mod 8 (final edge)
    
    ; r15 = current column position
    ; Process remaining full 8-column blocks
    test    rdx, rdx
    jz      .final_edge
    
.rem_8_loop:
    ;--------------------------------------------------------------------------
    ; Process one 8-column block
    ;--------------------------------------------------------------------------
    vxorps  ymm0, ymm0, ymm0             ; Accumulator
    mov     rbx, [rbp - 128]             ; A_row pointer
    mov     r9, [rbp - 104]              ; K
    
    ; B pointer for this column
    mov     rcx, [rbp - 72]
    mov     rax, r15
    shl     rax, 2
    add     rcx, rax
    
.rem_8_k_loop:
    test    r9, r9
    jz      .rem_8_k_done
    
    vbroadcastss ymm4, [rbx]
    vfmadd231ps ymm0, ymm4, [rcx]
    
    add     rbx, 4
    add     rcx, r11
    dec     r9
    jnz     .rem_8_k_loop
    
.rem_8_k_done:
    vmovups [r14], ymm0
    add     r14, 32
    add     r15, 8
    dec     rdx
    jnz     .rem_8_loop
    
.final_edge:
    ;--------------------------------------------------------------------------
    ; Handle final 1-7 columns
    ;--------------------------------------------------------------------------
    test    r8, r8
    jz      .row_next
    
    ; Create mask
    lea     rsi, [rel edge_mask]
    mov     rax, 8
    sub     rax, r8
    shl     rax, 2
    vmovups ymm15, [rsi + rax]
    
    ; Compute with mask
    vxorps  ymm0, ymm0, ymm0
    mov     rbx, [rbp - 128]
    mov     r9, [rbp - 104]
    mov     rcx, [rbp - 72]
    mov     rax, r15
    shl     rax, 2
    add     rcx, rax
    
.final_k_loop:
    test    r9, r9
    jz      .final_k_done
    
    vbroadcastss ymm4, [rbx]
    vmaskmovps ymm5, ymm15, [rcx]
    vfmadd231ps ymm0, ymm4, ymm5
    
    add     rbx, 4
    add     rcx, r11
    dec     r9
    jnz     .final_k_loop
    
.final_k_done:
    vmaskmovps [r14], ymm15, ymm0
    
.row_next:
    ;--------------------------------------------------------------------------
    ; Advance to next row
    ;--------------------------------------------------------------------------
    inc     r13                          ; i++
    jmp     .row_loop                    ; Next row
    
.done:
    ;--------------------------------------------------------------------------
    ; EPILOGUE: Restore callee-saved registers and return
    ;--------------------------------------------------------------------------
    
    vzeroupper                           ; Clear upper YMM bits (AVX-SSE transition)
    
%ifdef WIN64
    ; Windows: Restore XMM registers
    vmovaps xmm15, [rsp + 144]
    vmovaps xmm14, [rsp + 128]
    vmovaps xmm13, [rsp + 112]
    vmovaps xmm12, [rsp + 96]
    vmovaps xmm11, [rsp + 80]
    vmovaps xmm10, [rsp + 64]
    vmovaps xmm9,  [rsp + 48]
    vmovaps xmm8,  [rsp + 32]
    vmovaps xmm7,  [rsp + 16]
    vmovaps xmm6,  [rsp]
    add     rsp, 248
    pop     rsi
    pop     rdi
%else
    add     rsp, 80
%endif

    pop     r15
    pop     r14
    pop     r13
    pop     r12
    pop     rbx
    pop     rbp
    ret


;==============================================================================
; Function: matrix_mult_avx2_aligned
;==============================================================================
;
; Optimized version for 32-byte aligned matrices with dimensions that are
; multiples of 8. Skips edge handling for maximum performance.
;
; Same parameters as matrix_mult_avx2.
;
; Requirements:
;   - All matrices must be 32-byte aligned
;   - N must be a multiple of 8
;   - K should be a multiple of 4 for best performance
;
;==============================================================================

global matrix_mult_avx2_aligned
matrix_mult_avx2_aligned:
    ;--------------------------------------------------------------------------
    ; PROLOGUE
    ;--------------------------------------------------------------------------
    push    rbp
    mov     rbp, rsp
    push    rbx
    push    r12
    push    r13
    push    r14
    push    r15
    
%ifdef WIN64
    push    rdi
    push    rsi
    sub     rsp, 248                    ; Space for locals + 10 XMM registers + alignment
    vmovaps [rsp],      xmm6
    vmovaps [rsp + 16], xmm7
    vmovaps [rsp + 32], xmm8
    vmovaps [rsp + 48], xmm9
    vmovaps [rsp + 64], xmm10
    vmovaps [rsp + 80], xmm11
    vmovaps [rsp + 96], xmm12
    vmovaps [rsp + 112], xmm13
    vmovaps [rsp + 128], xmm14
    vmovaps [rsp + 144], xmm15
    
    mov     rdi, rcx
    mov     rsi, rdx
    mov     rdx, r8
    mov     rcx, r9
    mov     r8,  [rbp + 48]
    mov     r9,  [rbp + 56]
%else
    sub     rsp, 80
%endif

    ;--------------------------------------------------------------------------
    ; SETUP: Calculate strides
    ;--------------------------------------------------------------------------
    ; rdi = A, rsi = B, rdx = C, rcx = M, r8 = N, r9 = K
    
    mov     r10, r9
    shl     r10, 2                       ; A row stride = K * 4
    mov     r11, r8
    shl     r11, 2                       ; B row stride = N * 4
    mov     r12, r11                     ; C row stride = N * 4
    
    ; Store parameters
    mov     [rbp - 64], rdi              ; A
    mov     [rbp - 72], rsi              ; B
    mov     [rbp - 80], rdx              ; C
    mov     [rbp - 88], rcx              ; M
    mov     [rbp - 96], r8               ; N
    mov     [rbp - 104], r9               ; K
    
    ; Calculate N / 8 (number of column blocks)
    mov     rax, r8
    shr     rax, 3
    mov     [rbp - 112], rax              ; N_blocks
    
    ;--------------------------------------------------------------------------
    ; MAIN LOOP: Optimized with 4-row blocking
    ;--------------------------------------------------------------------------
    xor     r13, r13                     ; i = 0
    
.aligned_row_loop:
    cmp     r13, [rbp - 88]
    jge     .aligned_done
    
    ; Calculate A row pointer
    mov     rax, r13
    imul    rax, r10
    mov     rbx, [rbp - 64]
    add     rbx, rax                     ; A_row = A + i * K * 4
    mov     [rbp - 128], rbx
    
    ; Calculate C row pointer
    mov     rax, r13
    imul    rax, r12
    mov     r14, [rbp - 80]
    add     r14, rax                     ; C_row = C + i * N * 4
    
    xor     r15, r15                     ; j = 0 (block counter)
    
.aligned_col_loop:
    cmp     r15, [rbp - 112]              ; Compare with N_blocks
    jge     .aligned_row_next
    
    ; Initialize accumulator
    vxorps  ymm0, ymm0, ymm0
    
    ; Set up inner loop
    mov     rbx, [rbp - 128]             ; A_row
    mov     rcx, [rbp - 72]              ; B base
    
    ; B column offset = j * 8 * 4 = j * 32
    mov     rax, r15
    shl     rax, 5                       ; j * 32
    add     rcx, rax                     ; B[0][j*8]
    
    mov     r9, [rbp - 104]               ; K
    
    ;--- Unrolled inner loop with 4 k-values per iteration ---
.aligned_inner_loop:
    cmp     r9, 4
    jl      .aligned_inner_remainder
    
    ; k+0
    vbroadcastss ymm1, [rbx]
    vmovaps      ymm2, [rcx]             ; Aligned load
    vfmadd231ps  ymm0, ymm1, ymm2
    
    ; k+1
    vbroadcastss ymm3, [rbx + 4]
    vmovaps      ymm4, [rcx + r11]
    vfmadd231ps  ymm0, ymm3, ymm4
    
    ; k+2
    lea         rax, [r11 + r11]
    vbroadcastss ymm5, [rbx + 8]
    vmovaps      ymm6, [rcx + rax]
    vfmadd231ps  ymm0, ymm5, ymm6
    
    ; k+3
    add         rax, r11
    vbroadcastss ymm7, [rbx + 12]
    vmovaps      ymm8, [rcx + rax]
    vfmadd231ps  ymm0, ymm7, ymm8
    
    add     rbx, 16
    lea     rcx, [rcx + r11*4]
    sub     r9, 4
    jmp     .aligned_inner_loop
    
.aligned_inner_remainder:
    test    r9, r9
    jz      .aligned_inner_done
    
    vbroadcastss ymm1, [rbx]
    vmovaps      ymm2, [rcx]
    vfmadd231ps  ymm0, ymm1, ymm2
    
    add     rbx, 4
    add     rcx, r11
    dec     r9
    jnz     .aligned_inner_remainder
    
.aligned_inner_done:
    ; Store result (aligned)
    vmovaps [r14], ymm0
    
    add     r14, 32                      ; Next column block
    inc     r15                          ; j++
    jmp     .aligned_col_loop
    
.aligned_row_next:
    inc     r13
    jmp     .aligned_row_loop
    
.aligned_done:
    vzeroupper
    
%ifdef WIN64
    vmovaps xmm15, [rsp + 144]
    vmovaps xmm14, [rsp + 128]
    vmovaps xmm13, [rsp + 112]
    vmovaps xmm12, [rsp + 96]
    vmovaps xmm11, [rsp + 80]
    vmovaps xmm10, [rsp + 64]
    vmovaps xmm9,  [rsp + 48]
    vmovaps xmm8,  [rsp + 32]
    vmovaps xmm7,  [rsp + 16]
    vmovaps xmm6,  [rsp]
    add     rsp, 248
    pop     rsi
    pop     rdi
%else
    add     rsp, 80
%endif

    pop     r15
    pop     r14
    pop     r13
    pop     r12
    pop     rbx
    pop     rbp
    ret


;==============================================================================
; Function: check_avx2_support
;==============================================================================
;
; Check if the CPU supports AVX2 and FMA instructions.
;
; Parameters: None
;
; Returns:
;   eax = 1 if AVX2+FMA supported, 0 otherwise
;
;==============================================================================

global check_avx2_support
check_avx2_support:
    push    rbx                          ; CPUID clobbers rbx
    
    ; Check CPUID is supported (always true on x86_64)
    ; Check for AVX2: CPUID(7, 0).EBX bit 5
    mov     eax, 7
    xor     ecx, ecx
    cpuid
    
    ; Check AVX2 bit (bit 5 of EBX)
    bt      ebx, 5
    jnc     .not_supported               ; If carry not set, AVX2 not supported
    
    ; Check for FMA: CPUID(1).ECX bit 12
    mov     eax, 1
    cpuid
    
    ; Check FMA bit (bit 12 of ECX)
    bt      ecx, 12
    jnc     .not_supported
    
    ; Check that OS has enabled AVX (XCR0 bits 1 and 2)
    xor     ecx, ecx
    xgetbv                               ; Get XCR0 into EDX:EAX
    
    and     eax, 6                       ; Check bits 1 and 2
    cmp     eax, 6
    jne     .not_supported
    
    mov     eax, 1                       ; Return 1 (supported)
    pop     rbx
    ret
    
.not_supported:
    xor     eax, eax                     ; Return 0 (not supported)
    pop     rbx
    ret

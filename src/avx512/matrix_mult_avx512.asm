;==============================================================================
;                    SIMD MATRIX MULTIPLICATION - AVX-512
;==============================================================================
;
; High-performance matrix multiplication using AVX-512 (512-bit) SIMD instructions.
; Computes C = A × B for single-precision floating-point matrices.
;
; This implementation uses:
;   - AVX-512F (Foundation) instructions
;   - FMA (Fused Multiply-Add) with 512-bit registers
;   - Mask registers for elegant edge handling
;   - Embedded broadcast for efficient scalar multiplication
;   - 32 ZMM registers for extensive register blocking
;
; Advantages over AVX2:
;   - 16 floats per operation instead of 8 (2× throughput)
;   - 32 vector registers instead of 16 (more accumulator space)
;   - Native masking eliminates most branch-based edge handling
;   - Embedded broadcast reduces instruction count
;
; Supported calling conventions:
;   - System V AMD64 (Linux, macOS, BSD)
;   - Microsoft x64 (Windows) - controlled by WIN64 define
;
; CPU Requirements:
;   - Intel: Skylake-X (2017) or later
;   - AMD: Zen 4 (2022) or later
;
; Author: SIMD Matrix Multiplication Project
; License: MIT
;
;==============================================================================

; Enable for Windows calling convention (auto-detected)
%define WIN64

;------------------------------------------------------------------------------
; Section: Code
;------------------------------------------------------------------------------
section .text

;==============================================================================
; Function: matrix_mult_avx512
;==============================================================================
;
; Computes C = A × B for float matrices using AVX-512 SIMD.
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
; Memory recommendations:
;   - For best performance, matrices should be 64-byte aligned
;   - N should be a multiple of 16 for optimal performance
;
; Register usage:
;   ZMM0-ZMM7:   C row accumulators (can process up to 8 rows simultaneously)
;   ZMM8-ZMM15:  Additional accumulators for multi-row blocking
;   ZMM16-ZMM23: Temporary for A broadcasts
;   ZMM24-ZMM31: Temporary for B row loads
;   k1-k7:       Mask registers for edge handling
;
;==============================================================================

global matrix_mult_avx512
matrix_mult_avx512:
    ;--------------------------------------------------------------------------
    ; PROLOGUE: Save callee-saved registers
    ;--------------------------------------------------------------------------
    ; Per System V AMD64 ABI: preserve rbx, rbp, r12-r15
    ; ZMM registers are all caller-saved (no need to preserve)
    ; Mask registers k0-k7 are caller-saved (no need to preserve)
    ;--------------------------------------------------------------------------
    
    push    rbp
    mov     rbp, rsp
    push    rbx
    push    r12
    push    r13
    push    r14
    push    r15
    
    ; Allocate stack space for local variables
    sub     rsp, 128
    
%ifdef WIN64
    ; Windows x64: XMM6-XMM15 are callee-saved
    ; But we're not using them (only ZMM), so just remap parameters
    
    ; Save additional registers for Windows
    mov     [rbp - 48], rdi
    mov     [rbp - 56], rsi
    
    ; Remap Windows parameters to System V layout
    mov     rdi, rcx                    ; A
    mov     rsi, rdx                    ; B
    mov     rdx, r8                     ; C
    mov     rcx, r9                     ; M
    mov     r8,  [rbp + 48]             ; N (stack parameter)
    mov     r9,  [rbp + 56]             ; K (stack parameter)
%endif

    ;--------------------------------------------------------------------------
    ; PARAMETER SETUP
    ;--------------------------------------------------------------------------
    ; Store parameters in stack-local variables for easy access
    ;
    ; Stack layout (relative to rbp):
    ;   [rbp - 64]  = A pointer
    ;   [rbp - 72]  = B pointer
    ;   [rbp - 80]  = C pointer
    ;   [rbp - 88]  = M (rows of C)
    ;   [rbp - 96]  = N (columns of C)
    ;   [rbp - 104] = K (inner dimension)
    ;   [rbp - 112] = A row stride (K * 4)
    ;   [rbp - 120] = B row stride (N * 4)
    ;   [rbp - 128] = C row stride (N * 4)
    ;   [rbp - 136] = N / 16 (full ZMM blocks)
    ;   [rbp - 144] = N mod 16 (remainder columns)
    ;   [rbp - 152] = Current A row pointer
    ;--------------------------------------------------------------------------
    
    mov     [rbp - 64], rdi              ; A
    mov     [rbp - 72], rsi              ; B
    mov     [rbp - 80], rdx              ; C
    mov     [rbp - 88], rcx              ; M
    mov     [rbp - 96], r8               ; N
    mov     [rbp - 104], r9              ; K
    
    ; Calculate strides (in bytes)
    mov     r10, r9                      ; K
    shl     r10, 2                       ; K * 4 = A row stride
    mov     [rbp - 112], r10
    
    mov     r11, r8                      ; N
    shl     r11, 2                       ; N * 4 = B/C row stride
    mov     [rbp - 120], r11
    mov     [rbp - 128], r11
    
    ; Calculate number of full 16-column blocks
    mov     rax, r8                      ; N
    shr     rax, 4                       ; N / 16
    mov     [rbp - 136], rax
    
    ; Calculate remainder columns (N mod 16)
    mov     rax, r8
    and     rax, 15                      ; N mod 16
    mov     [rbp - 144], rax
    
    ;--------------------------------------------------------------------------
    ; CREATE EDGE MASK
    ;--------------------------------------------------------------------------
    ; AVX-512 uses mask registers (k1-k7) for predicated execution.
    ; For N not divisible by 16, we create a mask for the remaining columns.
    ;
    ; Example: If N mod 16 = 5, we create mask k1 = 0b0000000000011111
    ;          This mask enables operations only on the first 5 elements.
    ;
    ; The mask is created using:
    ;   mask = (1 << remainder) - 1
    ;   e.g., (1 << 5) - 1 = 32 - 1 = 31 = 0b11111
    ;--------------------------------------------------------------------------
    
    mov     rcx, [rbp - 144]             ; Remainder
    test    rcx, rcx                     ; Check if remainder > 0
    jz      .skip_mask_creation          ; No remainder, skip mask setup
    
    mov     rax, 1
    shl     rax, cl                      ; 1 << remainder
    dec     rax                          ; (1 << remainder) - 1
    kmovw   k1, eax                      ; Store in mask register k1
    jmp     .mask_done
    
.skip_mask_creation:
    ; Set k1 to all zeros (won't be used)
    xor     eax, eax
    kmovw   k1, eax
    
.mask_done:

    ;--------------------------------------------------------------------------
    ; MAIN COMPUTATION: Triple-nested loop with AVX-512 vectorization
    ;--------------------------------------------------------------------------
    ; Strategy: Process 4 rows of C simultaneously for better register utilization
    ;
    ; Outer loop:  i over rows of A (and C)        0..M-1 (step 4)
    ; Middle loop: j over columns of B (and C)     0..N-1 (step 16)
    ; Inner loop:  k over columns of A / rows of B 0..K-1
    ;
    ; Register allocation for 4-row blocking:
    ;   ZMM0: C[i+0][j:j+16] accumulator
    ;   ZMM1: C[i+1][j:j+16] accumulator
    ;   ZMM2: C[i+2][j:j+16] accumulator
    ;   ZMM3: C[i+3][j:j+16] accumulator
    ;   ZMM24-ZMM27: B[k][j:j+16] (can be reused)
    ;   ZMM28-ZMM31: A[i+x][k] broadcast values
    ;--------------------------------------------------------------------------
    
    xor     r13, r13                     ; i = 0 (row counter)
    
.row_loop_4:
    ;==========================================================================
    ; OUTER LOOP: Process 4 rows at a time for register blocking efficiency
    ;==========================================================================
    
    ; Check if we can process 4 rows
    lea     rax, [r13 + 4]
    cmp     rax, [rbp - 88]              ; i + 4 vs M
    jg      .row_loop_1                  ; Less than 4 rows left, switch to 1-at-a-time
    
    ;--------------------------------------------------------------------------
    ; Calculate row pointers for 4 rows
    ;--------------------------------------------------------------------------
    mov     rax, r13                     ; i
    mov     r10, [rbp - 112]             ; A row stride
    imul    rax, r10                     ; i * A_stride
    mov     rbx, [rbp - 64]              ; A base
    add     rbx, rax                     ; A[i]
    mov     [rbp - 152], rbx             ; Save for inner loop reset
    
    mov     rax, r13                     ; i
    mov     r12, [rbp - 128]             ; C row stride
    imul    rax, r12                     ; i * C_stride
    mov     r14, [rbp - 80]              ; C base
    add     r14, rax                     ; C[i] base pointer
    
    xor     r15, r15                     ; j = 0 (column block counter)
    
.col_loop_4:
    ;--------------------------------------------------------------------------
    ; Check if we have full 16-column block
    ;--------------------------------------------------------------------------
    cmp     r15, [rbp - 136]             ; Compare with N/16
    jge     .col_remainder_4             ; Handle remaining columns
    
    ;--------------------------------------------------------------------------
    ; Initialize 4 accumulator registers to zero
    ; These will hold C[i][j:j+16], C[i+1][j:j+16], C[i+2][j:j+16], C[i+3][j:j+16]
    ;--------------------------------------------------------------------------
    vxorps  zmm0, zmm0, zmm0             ; C[i+0][j:j+16] = 0
    vxorps  zmm1, zmm1, zmm1             ; C[i+1][j:j+16] = 0
    vxorps  zmm2, zmm2, zmm2             ; C[i+2][j:j+16] = 0
    vxorps  zmm3, zmm3, zmm3             ; C[i+3][j:j+16] = 0
    
    ;--------------------------------------------------------------------------
    ; Set up inner loop pointers
    ;--------------------------------------------------------------------------
    mov     rbx, [rbp - 152]             ; A[i][0]
    mov     rcx, [rbp - 72]              ; B base
    
    ; Calculate B column offset: j_block * 16 * 4 = j_block * 64
    mov     rax, r15
    shl     rax, 6                       ; j * 64
    add     rcx, rax                     ; B[0][j*16]
    
    mov     r9, [rbp - 104]              ; K
    mov     r10, [rbp - 112]             ; A row stride
    mov     r11, [rbp - 120]             ; B row stride
    
.inner_loop_4:
    ;==========================================================================
    ; INNER LOOP: The heart of AVX-512 matrix multiplication
    ;==========================================================================
    ; Process one k-value, updating all 4 rows of C accumulators
    ;
    ; For each row C[i+r]:
    ;   1. Broadcast A[i+r][k] to all 16 lanes of a ZMM register
    ;   2. Load B[k][j:j+16] (16 consecutive floats)
    ;   3. FMA: C[i+r] += A[i+r][k] × B[k]
    ;
    ; AVX-512 advantage: Can use embedded broadcast {1to16}!
    ; Instead of separate broadcast instruction, we can broadcast directly
    ; in the FMA instruction:
    ;   vfmadd231ps zmm0, zmm_B, [A_ptr]{1to16}
    ;
    ; This eliminates the broadcast instruction entirely!
    ;==========================================================================
    
    test    r9, r9
    jz      .inner_done_4
    
    ; Check if we can unroll by 4
    cmp     r9, 4
    jl      .inner_single_4
    
    ;--- Unrolled iteration: 4 k-values per loop iteration ---
    ; This improves instruction throughput by:
    ; 1. Hiding load latencies
    ; 2. Allowing better out-of-order execution
    ; 3. Reducing loop overhead
    
.inner_unrolled_4:
    ;--- k+0 ---
    vmovups zmm24, [rcx]                 ; B[k+0][j:j+16]
    
    ; USE EMBEDDED BROADCAST: {1to16} broadcasts the scalar from memory
    ; to all 16 lanes of the ZMM register during the FMA operation!
    ; This is a unique AVX-512 feature not available in AVX2.
    
    vfmadd231ps zmm0, zmm24, [rbx]{1to16}           ; C[i+0] += A[i+0][k+0] × B
    vfmadd231ps zmm1, zmm24, [rbx + r10]{1to16}     ; C[i+1] += A[i+1][k+0] × B
    lea     rax, [r10 + r10]                         ; 2 * A_stride
    vfmadd231ps zmm2, zmm24, [rbx + rax]{1to16}     ; C[i+2] += A[i+2][k+0] × B
    add     rax, r10                                 ; 3 * A_stride
    vfmadd231ps zmm3, zmm24, [rbx + rax]{1to16}     ; C[i+3] += A[i+3][k+0] × B
    
    ;--- k+1 ---
    vmovups zmm25, [rcx + r11]           ; B[k+1][j:j+16]
    
    vfmadd231ps zmm0, zmm25, [rbx + 4]{1to16}       ; k+1 is at offset 4 bytes
    vfmadd231ps zmm1, zmm25, [rbx + r10 + 4]{1to16}
    lea     rax, [r10 + r10]
    vfmadd231ps zmm2, zmm25, [rbx + rax + 4]{1to16}
    add     rax, r10
    vfmadd231ps zmm3, zmm25, [rbx + rax + 4]{1to16}
    
    ;--- k+2 ---
    lea     rax, [r11 + r11]             ; 2 * B_stride
    vmovups zmm26, [rcx + rax]           ; B[k+2][j:j+16]
    
    vfmadd231ps zmm0, zmm26, [rbx + 8]{1to16}
    vfmadd231ps zmm1, zmm26, [rbx + r10 + 8]{1to16}
    lea     rax, [r10 + r10]
    vfmadd231ps zmm2, zmm26, [rbx + rax + 8]{1to16}
    add     rax, r10
    vfmadd231ps zmm3, zmm26, [rbx + rax + 8]{1to16}
    
    ;--- k+3 ---
    lea     rax, [r11 + r11]
    add     rax, r11                     ; 3 * B_stride
    vmovups zmm27, [rcx + rax]           ; B[k+3][j:j+16]
    
    vfmadd231ps zmm0, zmm27, [rbx + 12]{1to16}
    vfmadd231ps zmm1, zmm27, [rbx + r10 + 12]{1to16}
    lea     rax, [r10 + r10]
    vfmadd231ps zmm2, zmm27, [rbx + rax + 12]{1to16}
    add     rax, r10
    vfmadd231ps zmm3, zmm27, [rbx + rax + 12]{1to16}
    
    ; Advance pointers
    add     rbx, 16                      ; A += 4 elements
    lea     rcx, [rcx + r11*4]           ; B += 4 rows
    sub     r9, 4
    
    cmp     r9, 4
    jge     .inner_unrolled_4
    
    test    r9, r9
    jz      .inner_done_4
    
.inner_single_4:
    ;--- Single k iteration (for remaining 1-3 k values) ---
    vmovups zmm24, [rcx]                 ; B[k][j:j+16]
    
    vfmadd231ps zmm0, zmm24, [rbx]{1to16}
    vfmadd231ps zmm1, zmm24, [rbx + r10]{1to16}
    lea     rax, [r10 + r10]
    vfmadd231ps zmm2, zmm24, [rbx + rax]{1to16}
    add     rax, r10
    vfmadd231ps zmm3, zmm24, [rbx + rax]{1to16}
    
    add     rbx, 4
    add     rcx, r11
    dec     r9
    jnz     .inner_single_4
    
.inner_done_4:
    ;--------------------------------------------------------------------------
    ; Store results for 4 rows
    ;--------------------------------------------------------------------------
    vmovups [r14],              zmm0     ; Store C[i+0][j:j+16]
    vmovups [r14 + r12],        zmm1     ; Store C[i+1][j:j+16]
    lea     rax, [r12 + r12]             ; 2 * C_stride
    vmovups [r14 + rax],        zmm2     ; Store C[i+2][j:j+16]
    add     rax, r12                     ; 3 * C_stride
    vmovups [r14 + rax],        zmm3     ; Store C[i+3][j:j+16]
    
    ; Advance to next column block
    add     r14, 64                      ; C += 16 floats
    inc     r15                          ; j++
    jmp     .col_loop_4
    
.col_remainder_4:
    ;==========================================================================
    ; EDGE HANDLING: Remaining columns (N mod 16) using mask register k1
    ;==========================================================================
    ; AVX-512's mask registers make edge handling elegant:
    ; - Masked loads only read valid memory locations
    ; - Masked stores only write to valid locations
    ; - No need for separate cleanup code or memcpy
    ;==========================================================================
    
    mov     rax, [rbp - 144]             ; Remainder
    test    rax, rax
    jz      .row_next_4                  ; No remainder, next row batch
    
    ; Initialize accumulators
    vxorps  zmm0, zmm0, zmm0
    vxorps  zmm1, zmm1, zmm1
    vxorps  zmm2, zmm2, zmm2
    vxorps  zmm3, zmm3, zmm3
    
    ; Set up pointers
    mov     rbx, [rbp - 152]             ; A[i][0]
    mov     rcx, [rbp - 72]              ; B base
    
    ; Calculate B column offset for remainder
    mov     rax, r15
    shl     rax, 6                       ; j * 64
    add     rcx, rax
    
    mov     r9, [rbp - 104]              ; K
    mov     r10, [rbp - 112]             ; A stride
    mov     r11, [rbp - 120]             ; B stride
    
.rem_inner_4:
    test    r9, r9
    jz      .rem_inner_done_4
    
    ;--------------------------------------------------------------------------
    ; MASKED LOAD: Only load valid elements
    ; Using {k1}{z}: load where k1 is 1, zero where k1 is 0
    ;--------------------------------------------------------------------------
    vmovups zmm24{k1}{z}, [rcx]          ; Masked load of B[k][j:j+rem]
    
    ; FMA with embedded broadcast
    vfmadd231ps zmm0, zmm24, [rbx]{1to16}
    vfmadd231ps zmm1, zmm24, [rbx + r10]{1to16}
    lea     rax, [r10 + r10]
    vfmadd231ps zmm2, zmm24, [rbx + rax]{1to16}
    add     rax, r10
    vfmadd231ps zmm3, zmm24, [rbx + rax]{1to16}
    
    add     rbx, 4
    add     rcx, r11
    dec     r9
    jnz     .rem_inner_4
    
.rem_inner_done_4:
    ;--------------------------------------------------------------------------
    ; MASKED STORE: Only store valid elements
    ; Using {k1}: store only where k1 is 1
    ;--------------------------------------------------------------------------
    vmovups [r14]{k1},              zmm0
    vmovups [r14 + r12]{k1},        zmm1
    lea     rax, [r12 + r12]
    vmovups [r14 + rax]{k1},        zmm2
    add     rax, r12
    vmovups [r14 + rax]{k1},        zmm3
    
.row_next_4:
    ; Advance to next row batch (4 rows)
    add     r13, 4
    jmp     .row_loop_4
    
.row_loop_1:
    ;==========================================================================
    ; SINGLE ROW PROCESSING: Handle remaining 1-3 rows
    ;==========================================================================
    ; When M is not divisible by 4, process remaining rows one at a time
    ;==========================================================================
    
    cmp     r13, [rbp - 88]              ; i vs M
    jge     .done                        ; All rows processed
    
    ; Calculate row pointers
    mov     rax, r13
    mov     r10, [rbp - 112]
    imul    rax, r10
    mov     rbx, [rbp - 64]
    add     rbx, rax                     ; A[i]
    mov     [rbp - 152], rbx
    
    mov     rax, r13
    mov     r12, [rbp - 128]
    imul    rax, r12
    mov     r14, [rbp - 80]
    add     r14, rax                     ; C[i]
    
    xor     r15, r15                     ; j = 0
    
.col_loop_1:
    cmp     r15, [rbp - 136]
    jge     .col_remainder_1
    
    ; Initialize single accumulator
    vxorps  zmm0, zmm0, zmm0
    
    ; Set up pointers
    mov     rbx, [rbp - 152]
    mov     rcx, [rbp - 72]
    mov     rax, r15
    shl     rax, 6
    add     rcx, rax
    
    mov     r9, [rbp - 104]
    mov     r11, [rbp - 120]
    
.inner_loop_1:
    test    r9, r9
    jz      .inner_done_1
    
    ; Single-row inner loop with unrolling
    cmp     r9, 4
    jl      .inner_single_1
    
    vmovups zmm24, [rcx]
    vfmadd231ps zmm0, zmm24, [rbx]{1to16}
    
    vmovups zmm25, [rcx + r11]
    vfmadd231ps zmm0, zmm25, [rbx + 4]{1to16}
    
    lea     rax, [r11 + r11]
    vmovups zmm26, [rcx + rax]
    vfmadd231ps zmm0, zmm26, [rbx + 8]{1to16}
    
    add     rax, r11
    vmovups zmm27, [rcx + rax]
    vfmadd231ps zmm0, zmm27, [rbx + 12]{1to16}
    
    add     rbx, 16
    lea     rcx, [rcx + r11*4]
    sub     r9, 4
    jmp     .inner_loop_1
    
.inner_single_1:
    vmovups zmm24, [rcx]
    vfmadd231ps zmm0, zmm24, [rbx]{1to16}
    
    add     rbx, 4
    add     rcx, r11
    dec     r9
    jnz     .inner_single_1
    
.inner_done_1:
    vmovups [r14], zmm0
    add     r14, 64
    inc     r15
    jmp     .col_loop_1
    
.col_remainder_1:
    ; Handle remaining columns for single row
    mov     rax, [rbp - 144]
    test    rax, rax
    jz      .row_next_1
    
    vxorps  zmm0, zmm0, zmm0
    
    mov     rbx, [rbp - 152]
    mov     rcx, [rbp - 72]
    mov     rax, r15
    shl     rax, 6
    add     rcx, rax
    
    mov     r9, [rbp - 104]
    mov     r11, [rbp - 120]
    
.rem_inner_1:
    test    r9, r9
    jz      .rem_inner_done_1
    
    vmovups zmm24{k1}{z}, [rcx]
    vfmadd231ps zmm0, zmm24, [rbx]{1to16}
    
    add     rbx, 4
    add     rcx, r11
    dec     r9
    jnz     .rem_inner_1
    
.rem_inner_done_1:
    vmovups [r14]{k1}, zmm0
    
.row_next_1:
    inc     r13
    jmp     .row_loop_1
    
.done:
    ;--------------------------------------------------------------------------
    ; EPILOGUE: Clean up and return
    ;--------------------------------------------------------------------------
    
    ; Note: vzeroupper is NOT needed for AVX-512
    ; AVX-512 does not cause the same SSE/AVX transition penalties as AVX/AVX2
    ; However, for compatibility with code that may call SSE functions,
    ; we still clear the upper bits
    vzeroupper
    
%ifdef WIN64
    mov     rdi, [rbp - 48]
    mov     rsi, [rbp - 56]
%endif

    add     rsp, 128                     ; Deallocate local variables
    pop     r15
    pop     r14
    pop     r13
    pop     r12
    pop     rbx
    pop     rbp
    ret


;==============================================================================
; Function: matrix_mult_avx512_aligned
;==============================================================================
;
; Optimized version for 64-byte aligned matrices with N divisible by 16.
; Omits edge handling for maximum performance.
;
; Same parameters as matrix_mult_avx512.
;
; Requirements:
;   - All matrices must be 64-byte aligned
;   - N must be a multiple of 16
;   - M should be a multiple of 4 for best performance
;
;==============================================================================

global matrix_mult_avx512_aligned
matrix_mult_avx512_aligned:
    push    rbp
    mov     rbp, rsp
    push    rbx
    push    r12
    push    r13
    push    r14
    push    r15
    sub     rsp, 128
    
%ifdef WIN64
    mov     [rbp - 48], rdi
    mov     [rbp - 56], rsi
    mov     rdi, rcx
    mov     rsi, rdx
    mov     rdx, r8
    mov     rcx, r9
    mov     r8,  [rbp + 48]
    mov     r9,  [rbp + 56]
%endif

    ; Store parameters
    mov     [rbp - 64], rdi              ; A
    mov     [rbp - 72], rsi              ; B
    mov     [rbp - 80], rdx              ; C
    mov     [rbp - 88], rcx              ; M
    mov     [rbp - 96], r8               ; N
    mov     [rbp - 104], r9              ; K
    
    ; Calculate strides
    mov     r10, r9
    shl     r10, 2
    mov     [rbp - 112], r10             ; A stride
    
    mov     r11, r8
    shl     r11, 2
    mov     [rbp - 120], r11             ; B stride
    mov     [rbp - 128], r11             ; C stride
    
    mov     rax, r8
    shr     rax, 4
    mov     [rbp - 136], rax             ; N / 16
    
    xor     r13, r13                     ; i = 0
    
.aligned_row_loop:
    lea     rax, [r13 + 4]
    cmp     rax, [rbp - 88]
    jg      .aligned_row_single
    
    ; Calculate A row pointer
    mov     rax, r13
    mov     r10, [rbp - 112]
    imul    rax, r10
    mov     rbx, [rbp - 64]
    add     rbx, rax
    mov     [rbp - 152], rbx
    
    ; Calculate C row pointer
    mov     rax, r13
    mov     r12, [rbp - 128]
    imul    rax, r12
    mov     r14, [rbp - 80]
    add     r14, rax
    
    xor     r15, r15
    
.aligned_col_loop:
    cmp     r15, [rbp - 136]
    jge     .aligned_row_next
    
    vxorps  zmm0, zmm0, zmm0
    vxorps  zmm1, zmm1, zmm1
    vxorps  zmm2, zmm2, zmm2
    vxorps  zmm3, zmm3, zmm3
    
    mov     rbx, [rbp - 152]
    mov     rcx, [rbp - 72]
    mov     rax, r15
    shl     rax, 6
    add     rcx, rax
    
    mov     r9, [rbp - 104]
    mov     r10, [rbp - 112]
    mov     r11, [rbp - 120]
    
.aligned_inner:
    cmp     r9, 4
    jl      .aligned_inner_single
    
    vmovaps zmm24, [rcx]
    vfmadd231ps zmm0, zmm24, [rbx]{1to16}
    vfmadd231ps zmm1, zmm24, [rbx + r10]{1to16}
    lea     rax, [r10 + r10]
    vfmadd231ps zmm2, zmm24, [rbx + rax]{1to16}
    add     rax, r10
    vfmadd231ps zmm3, zmm24, [rbx + rax]{1to16}
    
    vmovaps zmm25, [rcx + r11]
    vfmadd231ps zmm0, zmm25, [rbx + 4]{1to16}
    vfmadd231ps zmm1, zmm25, [rbx + r10 + 4]{1to16}
    lea     rax, [r10 + r10]
    vfmadd231ps zmm2, zmm25, [rbx + rax + 4]{1to16}
    add     rax, r10
    vfmadd231ps zmm3, zmm25, [rbx + rax + 4]{1to16}
    
    lea     rax, [r11 + r11]
    vmovaps zmm26, [rcx + rax]
    vfmadd231ps zmm0, zmm26, [rbx + 8]{1to16}
    vfmadd231ps zmm1, zmm26, [rbx + r10 + 8]{1to16}
    lea     rax, [r10 + r10]
    vfmadd231ps zmm2, zmm26, [rbx + rax + 8]{1to16}
    add     rax, r10
    vfmadd231ps zmm3, zmm26, [rbx + rax + 8]{1to16}
    
    lea     rax, [r11 + r11]
    add     rax, r11
    vmovaps zmm27, [rcx + rax]
    vfmadd231ps zmm0, zmm27, [rbx + 12]{1to16}
    vfmadd231ps zmm1, zmm27, [rbx + r10 + 12]{1to16}
    lea     rax, [r10 + r10]
    vfmadd231ps zmm2, zmm27, [rbx + rax + 12]{1to16}
    add     rax, r10
    vfmadd231ps zmm3, zmm27, [rbx + rax + 12]{1to16}
    
    add     rbx, 16
    lea     rcx, [rcx + r11*4]
    sub     r9, 4
    jmp     .aligned_inner
    
.aligned_inner_single:
    test    r9, r9
    jz      .aligned_inner_done
    
    vmovaps zmm24, [rcx]
    vfmadd231ps zmm0, zmm24, [rbx]{1to16}
    vfmadd231ps zmm1, zmm24, [rbx + r10]{1to16}
    lea     rax, [r10 + r10]
    vfmadd231ps zmm2, zmm24, [rbx + rax]{1to16}
    add     rax, r10
    vfmadd231ps zmm3, zmm24, [rbx + rax]{1to16}
    
    add     rbx, 4
    add     rcx, r11
    dec     r9
    jnz     .aligned_inner_single
    
.aligned_inner_done:
    vmovaps [r14],              zmm0
    vmovaps [r14 + r12],        zmm1
    lea     rax, [r12 + r12]
    vmovaps [r14 + rax],        zmm2
    add     rax, r12
    vmovaps [r14 + rax],        zmm3
    
    add     r14, 64
    inc     r15
    jmp     .aligned_col_loop
    
.aligned_row_next:
    add     r13, 4
    jmp     .aligned_row_loop
    
.aligned_row_single:
    cmp     r13, [rbp - 88]
    jge     .aligned_done
    
    mov     rax, r13
    mov     r10, [rbp - 112]
    imul    rax, r10
    mov     rbx, [rbp - 64]
    add     rbx, rax
    
    mov     rax, r13
    mov     r12, [rbp - 128]
    imul    rax, r12
    mov     r14, [rbp - 80]
    add     r14, rax
    
    xor     r15, r15
    
.aligned_col_single:
    cmp     r15, [rbp - 136]
    jge     .aligned_row_single_next
    
    vxorps  zmm0, zmm0, zmm0
    
    mov     rcx, [rbp - 72]
    mov     rax, r15
    shl     rax, 6
    add     rcx, rax
    
    push    rbx
    mov     r9, [rbp - 104]
    mov     r11, [rbp - 120]
    
.aligned_inner_single_row:
    test    r9, r9
    jz      .aligned_inner_single_row_done
    
    vmovaps zmm24, [rcx]
    vfmadd231ps zmm0, zmm24, [rbx]{1to16}
    
    add     rbx, 4
    add     rcx, r11
    dec     r9
    jnz     .aligned_inner_single_row
    
.aligned_inner_single_row_done:
    pop     rbx
    vmovaps [r14], zmm0
    add     r14, 64
    inc     r15
    jmp     .aligned_col_single
    
.aligned_row_single_next:
    inc     r13
    jmp     .aligned_row_single
    
.aligned_done:
    vzeroupper
    
%ifdef WIN64
    mov     rdi, [rbp - 48]
    mov     rsi, [rbp - 56]
%endif

    add     rsp, 128
    pop     r15
    pop     r14
    pop     r13
    pop     r12
    pop     rbx
    pop     rbp
    ret


;==============================================================================
; Function: check_avx512_support
;==============================================================================
;
; Check if the CPU supports AVX-512F (Foundation) instructions.
;
; Parameters: None
;
; Returns:
;   eax = 1 if AVX-512F supported, 0 otherwise
;
;==============================================================================

global check_avx512_support
check_avx512_support:
    push    rbx
    
    ; Check CPUID leaf 7 for AVX-512F support
    mov     eax, 7
    xor     ecx, ecx
    cpuid
    
    ; AVX-512F is bit 16 of EBX
    bt      ebx, 16
    jnc     .avx512_not_supported
    
    ; Check that OS has enabled AVX-512 (XCR0 bits 5, 6, 7)
    xor     ecx, ecx
    xgetbv                               ; Get XCR0 into EDX:EAX
    
    ; Check bits 1, 2 (AVX state) + bits 5, 6, 7 (AVX-512 state)
    ; Mask: 0b11100110 = 0xE6
    and     eax, 0xE6
    cmp     eax, 0xE6
    jne     .avx512_not_supported
    
    mov     eax, 1
    pop     rbx
    ret
    
.avx512_not_supported:
    xor     eax, eax
    pop     rbx
    ret

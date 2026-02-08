# AVX2 Instruction Reference for Matrix Multiplication

## Table of Contents
1. [Overview](#overview)
2. [Register Layout](#register-layout)
3. [Essential Instructions](#essential-instructions)
4. [Memory Operations](#memory-operations)
5. [Arithmetic Operations](#arithmetic-operations)
6. [Data Movement](#data-movement)
7. [Comparison Operations](#comparison-operations)
8. [Practical Examples](#practical-examples)

---

## Overview

**AVX2** (Advanced Vector Extensions 2) was introduced with Intel Haswell processors in 2013. It extends AVX with:
- Full 256-bit integer operations
- Gather instructions
- New permute instructions
- FMA3 (Fused Multiply-Add) - technically separate but always paired with AVX2

### Key Characteristics

| Feature | Specification |
|---------|---------------|
| Register Width | 256 bits |
| Registers | YMM0-YMM15 (16 registers) |
| Floats per Register | 8 × 32-bit single precision |
| Doubles per Register | 4 × 64-bit double precision |
| Integers | 32×8-bit, 16×16-bit, 8×32-bit, 4×64-bit |
| FMA Support | Yes (FMA3) |
| Mask Registers | No (use comparison + blend) |

---

## Register Layout

### YMM Registers (256-bit)

```
YMM Register (256 bits = 32 bytes):
┌─────────────────────────────────────────────────────────────────────────────┐
│                              YMM0 (256 bits)                                │
├─────────────────────────────────┬───────────────────────────────────────────┤
│    High 128 bits (NEW in AVX)   │              XMM0 (lower 128 bits)        │
└─────────────────────────────────┴───────────────────────────────────────────┘
  255                           128  127                                     0

As 8 single-precision floats (PS = Packed Single):
┌─────────┬─────────┬─────────┬─────────┬─────────┬─────────┬─────────┬─────────┐
│ float[7]│ float[6]│ float[5]│ float[4]│ float[3]│ float[2]│ float[1]│ float[0]│
│ 255:224 │ 223:192 │ 191:160 │ 159:128 │ 127:96  │  95:64  │  63:32  │  31:0   │
└─────────┴─────────┴─────────┴─────────┴─────────┴─────────┴─────────┴─────────┘
   High 128-bit lane (lane 1)    │         Low 128-bit lane (lane 0)

As 4 double-precision floats (PD = Packed Double):
┌───────────────────┬───────────────────┬───────────────────┬───────────────────┐
│    double[3]      │    double[2]      │    double[1]      │    double[0]      │
│    255:192        │    191:128        │    127:64         │     63:0          │
└───────────────────┴───────────────────┴───────────────────┴───────────────────┘
```

### Lane Concept

AVX2 processes data in two 128-bit "lanes". Some operations work within lanes:

```
Lane-based operation (e.g., vpunpcklwd):

            Lane 1 (High 128)        Lane 0 (Low 128)
          ┌─────────────────────┬─────────────────────┐
Source A: │ A7 A6 A5 A4 A3 A2 A1 A0 │ a7 a6 a5 a4 a3 a2 a1 a0 │
          └─────────────────────┴─────────────────────┘
          ┌─────────────────────┬─────────────────────┐
Source B: │ B7 B6 B5 B4 B3 B2 B1 B0 │ b7 b6 b5 b4 b3 b2 b1 b0 │
          └─────────────────────┴─────────────────────┘
                      ↓ Interleave within lanes ↓
          ┌─────────────────────┬─────────────────────┐
Result:   │ B3 A3 B2 A2 B1 A1 B0 A0 │ b3 a3 b2 a2 b1 a1 b0 a0 │
          └─────────────────────┴─────────────────────┘
```

---

## Essential Instructions

### Instruction Naming Convention

```
v     fmadd   231     p       s
│       │      │      │       │
│       │      │      │       └── s = Single precision (float)
│       │      │      │           d = Double precision
│       │      │      │
│       │      │      └── p = Packed (all lanes)
│       │      │          s = Scalar (single element)
│       │      │
│       │      └── Operand order: 231 means dst = src2 * src3 + src1
│       │                        (where dst = src1)
│       │
│       └── Operation: add, mul, sub, fmadd, etc.
│
└── VEX prefix (required for AVX)
```

---

## Memory Operations

### Load Operations

| Instruction | Description | Alignment |
|-------------|-------------|-----------|
| `vmovaps ymm, mem` | Load aligned packed singles | 32-byte |
| `vmovups ymm, mem` | Load unaligned packed singles | Any |
| `vmovapd ymm, mem` | Load aligned packed doubles | 32-byte |
| `vmovupd ymm, mem` | Load unaligned packed doubles | Any |

```nasm
; Aligned load (fastest, requires 32-byte alignment)
vmovaps ymm0, [rdi]       ; Load 8 floats (32 bytes) from address in rdi

; Unaligned load (works with any alignment)
vmovups ymm1, [rsi + 4]   ; Load 8 floats from potentially unaligned address

; Load with offset
vmovaps ymm2, [rdi + 32]  ; Load next 8 floats (32 bytes later)

; Scaled index (for arrays)
vmovaps ymm3, [rdi + rax*4]  ; Load from rdi + rax*4
```

### Store Operations

| Instruction | Description | Alignment |
|-------------|-------------|-----------|
| `vmovaps mem, ymm` | Store aligned packed singles | 32-byte |
| `vmovups mem, ymm` | Store unaligned packed singles | Any |
| `vmovapd mem, ymm` | Store aligned packed doubles | 32-byte |
| `vmovupd mem, ymm` | Store unaligned packed doubles | Any |
| `vmovntps mem, ymm` | Non-temporal store (bypasses cache) | 32-byte |

```nasm
; Aligned store
vmovaps [rdi], ymm0       ; Store 8 floats

; Non-temporal store (use for write-only large arrays)
vmovntps [rdi], ymm0      ; Store without polluting cache
```

### Broadcast Operations

| Instruction | Description |
|-------------|-------------|
| `vbroadcastss ymm, mem/xmm` | Broadcast single float to all 8 lanes |
| `vbroadcastsd ymm, mem/xmm` | Broadcast double to all 4 lanes |
| `vbroadcastf128 ymm, mem` | Broadcast 128 bits to both lanes |

```nasm
; Broadcast single float from memory
vbroadcastss ymm0, [rdi]
; Result: ymm0 = [x, x, x, x, x, x, x, x] where x = [rdi]

; Broadcast from XMM to YMM
vmovss  xmm1, [rdi]           ; Load single float to xmm1[0]
vbroadcastss ymm0, xmm1       ; Broadcast to all 8 lanes

; Broadcast 128 bits
vbroadcastf128 ymm2, [rdi]    ; Copy [rdi..rdi+15] to both 128-bit lanes
```

### Gather Operations (AVX2)

Load non-contiguous elements using index registers:

```nasm
; Gather 8 floats using 32-bit indices
; ymm2 must be all 1s (mask), gets zeroed after gather
vgatherdps ymm0, [rdi + ymm1*4], ymm2

; Example:
; rdi = base address of array
; ymm1 = [0, 2, 4, 6, 1, 3, 5, 7]  (indices)
; Result: ymm0 = [array[0], array[2], array[4], array[6], 
;                 array[1], array[3], array[5], array[7]]
```

---

## Arithmetic Operations

### Addition and Subtraction

| Instruction | Operation |
|-------------|-----------|
| `vaddps ymm, ymm, ymm/mem` | Add packed singles |
| `vsubps ymm, ymm, ymm/mem` | Subtract packed singles |
| `vaddpd ymm, ymm, ymm/mem` | Add packed doubles |
| `vsubpd ymm, ymm, ymm/mem` | Subtract packed doubles |
| `vhaddps ymm, ymm, ymm/mem` | Horizontal add (within lanes) |

```nasm
; Add element-wise
vaddps ymm0, ymm1, ymm2    ; ymm0[i] = ymm1[i] + ymm2[i] for i=0..7

; Add with memory operand
vaddps ymm0, ymm1, [rdi]   ; ymm0 = ymm1 + memory

; Subtract
vsubps ymm0, ymm1, ymm2    ; ymm0[i] = ymm1[i] - ymm2[i]
```

### Multiplication and Division

| Instruction | Operation |
|-------------|-----------|
| `vmulps ymm, ymm, ymm/mem` | Multiply packed singles |
| `vdivps ymm, ymm, ymm/mem` | Divide packed singles |
| `vrcpps ymm, ymm/mem` | Approximate reciprocal (fast) |
| `vrsqrtps ymm, ymm/mem` | Approximate inverse square root |
| `vsqrtps ymm, ymm/mem` | Square root |

```nasm
; Multiply
vmulps ymm0, ymm1, ymm2    ; ymm0[i] = ymm1[i] * ymm2[i]

; Division (slower, 11-13 cycles latency)
vdivps ymm0, ymm1, ymm2    ; ymm0[i] = ymm1[i] / ymm2[i]

; Fast reciprocal (12-bit precision, ~4 cycles)
vrcpps ymm0, ymm1          ; ymm0[i] ≈ 1.0 / ymm1[i]
```

### Fused Multiply-Add (FMA3)

**Key Operations for Matrix Multiplication:**

| Instruction | Operation | Notes |
|-------------|-----------|-------|
| `vfmadd132ps` | d = d * c + b | Destination × Operand3 + Operand2 |
| `vfmadd213ps` | d = b * d + c | Operand2 × Destination + Operand3 |
| `vfmadd231ps` | d = b * c + d | **Operand2 × Operand3 + Destination** ★ |
| `vfnmadd231ps` | d = -(b * c) + d | Negated multiply-add |
| `vfmsub231ps` | d = b * c - d | Multiply-subtract |

**Most Important: `vfmadd231ps`**

```nasm
; vfmadd231ps: dst = src2 * src3 + dst
;
; Perfect for accumulation: accumulator += A * B
;
vfmadd231ps ymm0, ymm1, ymm2
; ymm0 = ymm1 * ymm2 + ymm0
; ───────────────────────────
; ymm0[0] = ymm1[0] * ymm2[0] + ymm0[0]
; ymm0[1] = ymm1[1] * ymm2[1] + ymm0[1]
; ...
; ymm0[7] = ymm1[7] * ymm2[7] + ymm0[7]
```

**Visual Example:**

```
vfmadd231ps ymm0, ymm1, ymm2

BEFORE:
ymm0 (accumulator): [ 1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  8.0 ]
ymm1 (factor A):    [ 2.0,  2.0,  2.0,  2.0,  2.0,  2.0,  2.0,  2.0 ]
ymm2 (factor B):    [ 3.0,  3.0,  3.0,  3.0,  3.0,  3.0,  3.0,  3.0 ]

AFTER:
ymm0 = ymm1 * ymm2 + ymm0
ymm0: [ 7.0,  8.0,  9.0, 10.0, 11.0, 12.0, 13.0, 14.0 ]
       ↑
       2*3+1=7
```

### Min/Max Operations

```nasm
vminps ymm0, ymm1, ymm2    ; ymm0[i] = min(ymm1[i], ymm2[i])
vmaxps ymm0, ymm1, ymm2    ; ymm0[i] = max(ymm1[i], ymm2[i])
```

---

## Data Movement

### Register-to-Register Moves

```nasm
vmovaps ymm0, ymm1         ; Copy ymm1 to ymm0

; Move between XMM and YMM (preserves upper 128 if using XMM)
vmovaps xmm0, xmm1         ; Copy lower 128 bits, zeros upper 128

; Clear register (fastest way)
vxorps  ymm0, ymm0, ymm0   ; ymm0 = 0
```

### Shuffle and Permute

| Instruction | Description |
|-------------|-------------|
| `vshufps ymm, ymm, ymm, imm8` | Shuffle within lanes |
| `vpermilps ymm, ymm, imm8` | Permute in-lane |
| `vperm2f128 ymm, ymm, ymm, imm8` | Permute 128-bit lanes |
| `vpermpd ymm, ymm, imm8` | Permute 64-bit elements (AVX2) |
| `vpermps ymm, ymm, ymm` | Permute 32-bit using indices (AVX2) |

```nasm
; Permute 128-bit lanes
vperm2f128 ymm0, ymm1, ymm2, 0x31
; 0x31 = 0011 0001
;        ──── ────
;         │    └── Low 128: ymm1[high128]
;         └────── High 128: ymm2[high128]

; Permute floats using indices in ymm1
vpermps ymm0, ymm1, ymm2    ; ymm0[i] = ymm2[ymm1[i]]
```

### Insert and Extract

```nasm
; Extract upper 128 bits to XMM
vextractf128 xmm0, ymm1, 1   ; xmm0 = ymm1[255:128]

; Extract lower 128 bits 
vextractf128 xmm0, ymm1, 0   ; xmm0 = ymm1[127:0]

; Insert XMM into YMM
vinsertf128  ymm0, ymm1, xmm2, 1  ; ymm0 = [xmm2, ymm1[127:0]]
```

### Blend Operations

```nasm
; Blend using immediate mask
vblendps ymm0, ymm1, ymm2, 0b10101010
; For each bit in mask:
;   0 = take from ymm1
;   1 = take from ymm2
; Result: ymm0 = [ymm2[7], ymm1[6], ymm2[5], ymm1[4], ...]

; Blend using variable mask (comparison result)
vblendvps ymm0, ymm1, ymm2, ymm3
; For each lane:
;   If ymm3[i] sign bit is 0: take ymm1[i]
;   If ymm3[i] sign bit is 1: take ymm2[i]
```

---

## Comparison Operations

### Compare and Generate Mask

```nasm
; Compare and set mask in destination
vcmpps ymm0, ymm1, ymm2, 0   ; 0 = Equal
vcmpps ymm0, ymm1, ymm2, 1   ; 1 = Less than
vcmpps ymm0, ymm1, ymm2, 2   ; 2 = Less than or equal
vcmpps ymm0, ymm1, ymm2, 4   ; 4 = Not equal
; Result: 0xFFFFFFFF where true, 0x00000000 where false

; Common comparison predicates
; 0  = CMP_EQ_OQ    (equal, ordered, quiet)
; 1  = CMP_LT_OS    (less than)
; 2  = CMP_LE_OS    (less or equal)
; 4  = CMP_NEQ_UQ   (not equal)
; 5  = CMP_NLT_US   (not less than = >=)
; 6  = CMP_NLE_US   (not less or equal = >)
```

### Test Operations

```nasm
; Test bits and set flags
vtestps ymm0, ymm1
; Sets ZF if (ymm0 AND ymm1) == 0
; Sets CF if (ymm0 AND NOT ymm1) == 0
; Can then use jz, jnz, jc, jnc for branching
```

---

## Practical Examples

### Example 1: Vector Dot Product

```nasm
; Calculate dot product of two 16-float vectors
; Input: rdi = pointer to vector A
;        rsi = pointer to vector B
; Output: xmm0[0] = dot product result

vector_dot_avx2:
    ; Load 16 floats (two YMM registers worth)
    vmovaps      ymm0, [rdi]          ; A[0..7]
    vmovaps      ymm1, [rdi + 32]     ; A[8..15]
    
    ; Multiply element-wise
    vmulps       ymm0, ymm0, [rsi]      ; A[0..7] * B[0..7]
    vmulps       ymm1, ymm1, [rsi + 32] ; A[8..15] * B[8..15]
    
    ; Add ymm0 and ymm1
    vaddps       ymm0, ymm0, ymm1      ; Sum both vectors
    
    ; Horizontal reduction: sum all 8 elements
    ; Step 1: Add high 128 bits to low 128 bits
    vextractf128 xmm1, ymm0, 1         ; xmm1 = ymm0[255:128]
    vaddps       xmm0, xmm0, xmm1      ; xmm0 += xmm1 (4 elements)
    
    ; Step 2: Horizontal add to get final sum
    vhaddps      xmm0, xmm0, xmm0      ; [a+b, c+d, a+b, c+d]
    vhaddps      xmm0, xmm0, xmm0      ; [total, total, total, total]
    
    ; Result in xmm0[0]
    vzeroupper
    ret
```

### Example 2: Element-wise Operations

```nasm
; Compute C[i] = A[i]^2 + 2*A[i]*B[i] + B[i]^2  (which is (A+B)^2)
; Input: rdi = A, rsi = B, rdx = C, rcx = count (multiple of 8)

add_squares_avx2:
    xor     rax, rax                  ; index = 0
    
.loop:
    vmovaps      ymm0, [rdi + rax]    ; Load A
    vmovaps      ymm1, [rsi + rax]    ; Load B
    vaddps       ymm2, ymm0, ymm1     ; A + B
    vmulps       ymm2, ymm2, ymm2     ; (A + B)^2
    vmovaps      [rdx + rax], ymm2    ; Store result
    
    add     rax, 32                   ; Move to next 8 floats
    cmp     rax, rcx
    jl      .loop
    
    vzeroupper
    ret
```

### Example 3: Matrix Row Sum

```nasm
; Sum each row of an M×N matrix
; Input: rdi = matrix (row-major), rsi = output, rdx = M, rcx = N

row_sums_avx2:
    push    rbx
    xor     r8, r8                    ; row index
    
.row_loop:
    vxorps       ymm0, ymm0, ymm0     ; accumulator = 0
    xor     r9, r9                    ; column index
    
    ; Process 8 columns at a time
.col_loop:
    vaddps       ymm0, ymm0, [rdi + r9]
    add     r9, 32
    cmp     r9, rcx                   ; Compare with N*4 bytes
    jl      .col_loop
    
    ; Horizontal sum
    vextractf128 xmm1, ymm0, 1
    vaddps       xmm0, xmm0, xmm1
    vhaddps      xmm0, xmm0, xmm0
    vhaddps      xmm0, xmm0, xmm0
    
    vmovss       [rsi + r8*4], xmm0   ; Store row sum
    
    add     rdi, rcx                  ; Move to next row
    inc     r8
    cmp     r8, rdx                   ; Compare with M
    jl      .row_loop
    
    vzeroupper
    pop     rbx
    ret
```

---

## Best Practices

### Do's

1. **Use `vzeroupper` before returning** from AVX code to avoid SSE-AVX transition penalties
2. **Align data to 32 bytes** when possible for best performance
3. **Use `vmovups` for unaligned data** - modern CPUs have minimal penalty
4. **Prefer `vfmadd231ps`** for accumulation patterns
5. **Unroll loops** to hide latency and maximize throughput

### Don'ts

1. **Don't mix legacy SSE and AVX** without `vzeroupper`
2. **Don't assume AVX2 is available** - check CPUID first
3. **Don't use `vdivps` in tight loops** - use reciprocal approximation instead
4. **Don't ignore cache locality** - SIMD doesn't help if memory is the bottleneck
5. **Don't load/store C in inner k-loop** - use register blocking to keep accumulators in YMM registers

---

## Register-Blocked Tiling Pattern

For matrix multiplication, use this pattern to maximize performance:

```nasm
; Process 32 columns at a time with 4 YMM accumulators
vxorps  ymm0, ymm0, ymm0    ; C[i][j+0:8]
vxorps  ymm1, ymm1, ymm1    ; C[i][j+8:16]
vxorps  ymm2, ymm2, ymm2    ; C[i][j+16:24]
vxorps  ymm3, ymm3, ymm3    ; C[i][j+24:32]

.k_loop:
    vbroadcastss ymm4, [A + k*4]      ; Broadcast A[i][k]
    vfmadd231ps ymm0, ymm4, [B + 0]   ; 4 FMAs per k iteration
    vfmadd231ps ymm1, ymm4, [B + 32]
    vfmadd231ps ymm2, ymm4, [B + 64]
    vfmadd231ps ymm3, ymm4, [B + 96]
    add B, row_stride
    dec k
    jnz .k_loop

vmovups [C], ymm0                     ; Store only ONCE per tile!
vmovups [C + 32], ymm1
vmovups [C + 64], ymm2
vmovups [C + 96], ymm3
```

**Key insight**: Accumulators stay in registers across entire k-loop, eliminating load/store overhead.

---

## Next Steps

- [AVX512_REFERENCE.md](AVX512_REFERENCE.md) - Learn about AVX-512's expanded capabilities
- [MATRIX_MULT_ALGORITHM.md](MATRIX_MULT_ALGORITHM.md) - See these instructions applied to matrix multiplication

# AVX-512 Instruction Reference for Matrix Multiplication

## Table of Contents
1. [Overview](#overview)
2. [Key Differences from AVX2](#key-differences-from-avx2)
3. [Register Layout](#register-layout)
4. [Mask Registers](#mask-registers)
5. [Essential Instructions](#essential-instructions)
6. [Memory Operations](#memory-operations)
7. [Arithmetic Operations](#arithmetic-operations)
8. [Masking Operations](#masking-operations)
9. [Broadcast and Embedding](#broadcast-and-embedding)
10. [Practical Examples](#practical-examples)

---

## Overview

**AVX-512** (Advanced Vector Extensions 512) is a major extension introduced with Intel Skylake-X (2017) and AMD Zen 4 (2022). It provides:

- **512-bit** vector registers (double the width of AVX2)
- **32 vector registers** (double the count of AVX2)
- **8 dedicated mask registers** for predicated operations
- **Embedded broadcast** for memory operands
- **Rounding control** embedded in instructions

### AVX-512 Subsets

AVX-512 is modular, with different CPUs supporting different subsets:

| Subset | Description | Required For |
|--------|-------------|--------------|
| **AVX-512F** | Foundation | All AVX-512 CPUs |
| **AVX-512CD** | Conflict Detection | All AVX-512 CPUs |
| **AVX-512VL** | Vector Length (128/256-bit with masks) | Most CPUs |
| **AVX-512BW** | Byte and Word operations | Most CPUs |
| **AVX-512DQ** | Doubleword and Quadword | Most CPUs |
| AVX-512IFMA | Integer FMA | Cannon Lake+ |
| AVX-512VNNI | Neural Network Instructions | Ice Lake+ |
| AVX-512FP16 | Half-precision float | Sapphire Rapids+ |

### Key Characteristics

| Feature | AVX2 | AVX-512 |
|---------|------|---------|
| Register Width | 256 bits | **512 bits** |
| Register Count | 16 (YMM0-15) | **32 (ZMM0-31)** |
| Floats per Register | 8 | **16** |
| Doubles per Register | 4 | **8** |
| Mask Registers | None | **8 (k0-k7)** |
| Embedded Broadcast | No | **Yes** |

---

## Key Differences from AVX2

### 1. Doubled Width = Doubled Throughput

```
AVX2 (256-bit YMM):
┌─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┐
│ f[7]│ f[6]│ f[5]│ f[4]│ f[3]│ f[2]│ f[1]│ f[0]│  8 floats
└─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┘

AVX-512 (512-bit ZMM):
┌─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┐
│f[15]│f[14]│f[13]│f[12]│f[11]│f[10]│f[9] │f[8] │f[7] │f[6] │f[5] │f[4] │f[3] │f[2] │f[1] │f[0] │
└─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┘
                                                                                    16 floats
```

### 2. More Registers = Less Spilling

```
AVX2:  16 registers (YMM0-YMM15)   → May need to spill to memory
AVX-512: 32 registers (ZMM0-ZMM31) → More room for accumulator blocking
```

### 3. Native Masking = No More Blend

```nasm
; AVX2: Conditional operation requires blend
vcmpps  ymm3, ymm1, ymm2, 1      ; Compare, generate mask
vblendvps ymm0, ymm0, ymm4, ymm3 ; Blend based on mask

; AVX-512: Mask register directly
vcmpps  k1, zmm1, zmm2, 1        ; Compare, result in k1
vaddps  zmm0{k1}, zmm0, zmm4     ; Add only where k1 is set
```

### 4. Embedded Broadcast

```nasm
; AVX2: Broadcast requires separate instruction
vbroadcastss ymm1, [rdi]
vmulps       ymm0, ymm0, ymm1

; AVX-512: Broadcast embedded in instruction
vmulps       zmm0, zmm0, [rdi]{1to16}   ; Broadcast and multiply in one!
```

---

## Register Layout

### ZMM Registers (512-bit)

```
ZMM Register hierarchy:
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│                                    ZMM0 (512 bits)                                          │
├─────────────────────────────────────────────────────┬───────────────────────────────────────┤
│              High 256 bits                          │            YMM0 (low 256 bits)        │
│                                                     ├─────────────────┬─────────────────────┤
│                                                     │   High 128      │    XMM0 (low 128)   │
└─────────────────────────────────────────────────────┴─────────────────┴─────────────────────┘
511                                                256  255             128  127              0

As 16 single-precision floats:
┌────┬────┬────┬────┬────┬────┬────┬────┬────┬────┬────┬────┬────┬────┬────┬────┐
│[15]│[14]│[13]│[12]│[11]│[10]│ [9]│ [8]│ [7]│ [6]│ [5]│ [4]│ [3]│ [2]│ [1]│ [0]│
└────┴────┴────┴────┴────┴────┴────┴────┴────┴────┴────┴────┴────┴────┴────┴────┘
 511:480  ...                                                              31:0

As 8 double-precision floats:
┌──────────┬──────────┬──────────┬──────────┬──────────┬──────────┬──────────┬──────────┐
│  d[7]    │   d[6]   │   d[5]   │   d[4]   │   d[3]   │   d[2]   │   d[1]   │   d[0]   │
└──────────┴──────────┴──────────┴──────────┴──────────┴──────────┴──────────┴──────────┘
```

### All 32 Registers

```
General Purpose (often used for temporary/accumulation):
ZMM0-ZMM15   (also accessible as YMM0-15, XMM0-15)

Extended (AVX-512 only):
ZMM16-ZMM31  (no YMM/XMM aliases, only accessible with EVEX encoding)
```

### Caller/Callee Saved (Windows x64)

```
Volatile (caller-saved):     ZMM0-ZMM5, ZMM16-ZMM31 (plus k0-k7)
Non-volatile (callee-saved): ZMM6-ZMM15
```

### Caller/Callee Saved (Linux/macOS SysV)

```
Volatile (caller-saved):     All ZMM registers (ZMM0-ZMM31, k0-k7)
Non-volatile (callee-saved): None (vector registers not preserved)
```

---

## Mask Registers

AVX-512 introduces **8 mask registers (k0-k7)** for predicated execution.

### Mask Register Layout

```
k0-k7: 64-bit mask registers

For 16 single floats in ZMM:
k1 = 0b1111111111111111  (16 bits used)
      ↑              ↑
    bit 15         bit 0
    
For 8 doubles in ZMM:
k1 = 0b11111111  (8 bits used)
```

### Special Role of k0

**k0 is ALWAYS treated as "all ones"** - it cannot be used for masking:

```nasm
; k0 means "no mask" (all elements)
vaddps  zmm0{k0}, zmm1, zmm2   ; Same as: vaddps zmm0, zmm1, zmm2

; k1-k7 can be used for actual masking
vaddps  zmm0{k1}, zmm1, zmm2   ; Only updates where k1 bits are set
```

### Mask Operations

| Instruction | Description |
|-------------|-------------|
| `kmovw k1, eax` | Move 16 bits from GPR to mask |
| `kmovw eax, k1` | Move mask to GPR |
| `kandw k1, k2, k3` | AND two masks |
| `korw k1, k2, k3` | OR two masks |
| `knotw k1, k2` | NOT mask |
| `kshiftlw k1, k2, imm` | Shift mask left |
| `kunpckbw k1, k2, k3` | Unpack byte masks to word |

```nasm
; Create mask for first 5 elements
mov     eax, 0b0000000000011111
kmovw   k1, eax

; Create mask from comparison
vcmpps  k1, zmm0, zmm1, 1      ; k1[i] = 1 if zmm0[i] < zmm1[i]
```

### Masking Modes

Two masking modes available:

```nasm
; Zero-masking {k1}{z} - unmasked elements become zero
vaddps  zmm0{k1}{z}, zmm1, zmm2
; zmm0[i] = k1[i] ? (zmm1[i] + zmm2[i]) : 0.0

; Merge-masking {k1} - unmasked elements keep original value
vaddps  zmm0{k1}, zmm1, zmm2
; zmm0[i] = k1[i] ? (zmm1[i] + zmm2[i]) : zmm0[i]
```

---

## Essential Instructions

### EVEX Prefix

AVX-512 uses the **EVEX** (Extended VEX) prefix instead of VEX:

```
EVEX provides:
- 512-bit operations
- 32 registers
- Mask register encoding
- Embedded broadcast
- Rounding control
- Suppress all exceptions
```

### Instruction Encoding

```
v     fmadd   231     ps           {k1}{z}
│       │      │       │              │
│       │      │       │              └── Masking: k1 mask, zero-masked
│       │      │       │
│       │      │       └── ps = packed single (16 floats)
│       │      │           pd = packed double (8 doubles)
│       │      │
│       │      └── Operand order (same as AVX2)
│       │
│       └── Operation
│
└── VEX/EVEX prefix
```

---

## Memory Operations

### Aligned and Unaligned Loads

```nasm
; Aligned load (64-byte boundary)
vmovaps  zmm0, [rdi]           ; Load 16 floats (64 bytes)

; Unaligned load
vmovups  zmm0, [rdi]           ; Works with any alignment

; Masked load
vmovups  zmm0{k1}{z}, [rdi]    ; Load only where k1 is set, zero rest
```

### Stores

```nasm
; Aligned store
vmovaps  [rdi], zmm0           ; Store 16 floats

; Masked store (very useful for edge cases!)
vmovups  [rdi]{k1}, zmm0       ; Only store elements where k1 is set

; Non-temporal store (write-combining)
vmovntps [rdi], zmm0           ; Bypass cache for streaming writes
```

### Broadcast from Memory

**Embedded broadcast** - load one element and replicate:

```nasm
; Load single float and broadcast to all 16 lanes
vmulps   zmm0, zmm1, [rdi]{1to16}
; Equivalent to:
;   vbroadcastss zmm2, [rdi]
;   vmulps zmm0, zmm1, zmm2
; But in ONE instruction!

; Broadcast patterns:
; {1to16} - Broadcast 1 float to 16 lanes
; {1to8}  - Broadcast 1 double to 8 lanes
; {4to16} - Broadcast 4 floats (128-bit) to 16 lanes
```

### Gather and Scatter

```nasm
; ============ GATHER ============
; Load non-contiguous elements using index vector
; zmm1 contains 16 32-bit indices
vgatherdps zmm0{k1}, [rdi + zmm1*4]
; k1 specifies which elements to gather
; k1 is zeroed after operation (to prevent double-loading)

; Example:
; rdi = base address
; zmm1 = [0, 5, 2, 7, 1, 6, 3, 8, 4, 9, ...] (indices)
; Result: zmm0 = [array[0], array[5], array[2], ...]

; ============ SCATTER ============
; Store to non-contiguous locations (AVX-512 only!)
vscatterdps [rdi + zmm1*4]{k1}, zmm0
; Stores zmm0 elements to indexed locations
; No equivalent in AVX2!
```

---

## Arithmetic Operations

### Basic Arithmetic

Same as AVX2 but with ZMM registers and optional masking:

```nasm
vaddps   zmm0, zmm1, zmm2           ; Add 16 floats
vsubps   zmm0, zmm1, zmm2           ; Subtract
vmulps   zmm0, zmm1, zmm2           ; Multiply
vdivps   zmm0, zmm1, zmm2           ; Divide

; With masking
vaddps   zmm0{k1}, zmm1, zmm2       ; Add only where k1 is set
vaddps   zmm0{k1}{z}, zmm1, zmm2    ; Add where k1 set, zero elsewhere

; With embedded broadcast
vmulps   zmm0, zmm1, [rdi]{1to16}   ; Multiply by broadcasted scalar
```

### Fused Multiply-Add (FMA)

```nasm
; Same variants as AVX2, but with ZMM
vfmadd231ps  zmm0, zmm1, zmm2        ; zmm0 = zmm1 * zmm2 + zmm0
vfmadd231ps  zmm0{k1}, zmm1, zmm2    ; Masked FMA

; With broadcast
vfmadd231ps  zmm0, zmm1, [rdi]{1to16}  ; One broadcast, 16 FMAs!
```

### Rounding Modes (EVEX)

```nasm
; Embedded rounding control (AVX-512 only)
vaddps   zmm0, zmm1, zmm2{rn-sae}   ; Round to nearest
vaddps   zmm0, zmm1, zmm2{rd-sae}   ; Round down (toward -∞)
vaddps   zmm0, zmm1, zmm2{ru-sae}   ; Round up (toward +∞)
vaddps   zmm0, zmm1, zmm2{rz-sae}   ; Round toward zero

; sae = Suppress All Exceptions
```

### Special Operations

```nasm
; Reduce: horizontal operations
vreduceps zmm0, zmm1, imm8         ; Reduce to specific range

; Range selection
vrangeps  zmm0, zmm1, zmm2, imm8   ; Select min/max/abs

; Scale by power of 2
vscalefps zmm0, zmm1, zmm2         ; zmm0 = zmm1 * 2^zmm2

; Reciprocal approximations (14 bits precision)
vrcp14ps  zmm0, zmm1               ; Fast 1/x
vrsqrt14ps zmm0, zmm1              ; Fast 1/sqrt(x)

; Full precision reciprocal (28 bits)
vrcp28ps  zmm0, zmm1               ; Only on Xeon Phi
```

---

## Masking Operations

### Comparison to Mask

```nasm
; Compare and put result in mask register
vcmpps   k1, zmm0, zmm1, 0         ; k1 = (zmm0 == zmm1)
vcmpps   k1, zmm0, zmm1, 1         ; k1 = (zmm0 < zmm1)
vcmpps   k1, zmm0, zmm1, 2         ; k1 = (zmm0 <= zmm1)
vcmpps   k1, zmm0, zmm1, 4         ; k1 = (zmm0 != zmm1)
vcmpps   k1, zmm0, zmm1, 5         ; k1 = (zmm0 >= zmm1)
vcmpps   k1, zmm0, zmm1, 6         ; k1 = (zmm0 > zmm1)

; Combine comparisons with mask operators
kandw    k3, k1, k2                ; k3 = k1 AND k2
korw     k3, k1, k2                ; k3 = k1 OR k2
kxorw    k3, k1, k2                ; k3 = k1 XOR k2
knotw    k2, k1                    ; k2 = NOT k1
```

### Count and Find

```nasm
; Count set bits in mask
kmovw    eax, k1
popcnt   eax, eax                  ; Count how many elements matched

; Find first set bit
kmovw    eax, k1
bsf      eax, eax                  ; Index of first set bit

; Check if any/all bits set
kortestw k1, k1
jz       .none_set                 ; ZF=1 if k1 is all zeros
jc       .all_set                  ; CF=1 if k1 is all ones
```

### Edge Handling with Masks

```nasm
; Process array of length 50 (not divisible by 16)
; First 48 elements: 3 full iterations
; Last 2 elements: masked iteration

mov     ecx, 50
mov     eax, ecx
and     eax, 15              ; eax = 50 mod 16 = 2
mov     ebx, 1
shl     ebx, cl              ; ebx = 2^2 = 4
dec     ebx                  ; ebx = 0b0011 (mask for 2 elements)
kmovw   k1, ebx

; Last iteration with mask
vmovups  zmm0{k1}{z}, [rdi + 48*4]   ; Load only last 2 floats
; ... process ...
vmovups  [rsi + 48*4]{k1}, zmm0       ; Store only last 2 floats
```

---

## Broadcast and Embedding

### Embedded Broadcast Syntax

```nasm
; Broadcast single element to all lanes during operation
vaddps   zmm0, zmm1, [rdi]{1to16}   ; Load 1 float, add to all 16

; Other broadcast sizes
vaddps   zmm0, zmm1, [rdi]{1to16}   ; 1 float → 16 floats
vaddpd   zmm0, zmm1, [rdi]{1to8}    ; 1 double → 8 doubles
```

### When to Use Embedded Broadcast

| Operation | Without Broadcast | With Broadcast |
|-----------|-------------------|----------------|
| Scale all elements | `vbroadcastss zmm2, [scale]` | `vmulps zmm0, zmm1, [scale]{1to16}` |
| | `vmulps zmm0, zmm1, zmm2` | (One instruction instead of two) |
| Add constant | `vbroadcastss zmm2, [one]` | `vaddps zmm0, zmm1, [one]{1to16}` |
| | `vaddps zmm0, zmm1, zmm2` | |

---

## Practical Examples

### Example 1: Vector Dot Product (512-bit)

```nasm
;-----------------------------------------------------------------------------
; vector_dot_avx512 - Compute dot product of two 32-float vectors
;
; Input:  rdi = pointer to vector A (16 floats, 64-byte aligned)
;         rsi = pointer to vector B (16 floats, 64-byte aligned)
; Output: xmm0 = dot product result (single float in xmm0[0])
;
; Uses: ZMM0-3 for computation
;-----------------------------------------------------------------------------
vector_dot_avx512:
    ; Load 32 floats from each vector
    vmovaps      zmm0, [rdi]          ; A[0..15]
    vmovaps      zmm1, [rdi + 64]     ; A[16..31]
    
    ; Multiply element-wise
    vmulps       zmm0, zmm0, [rsi]      ; A[0..15] * B[0..15]
    vfmadd231ps  zmm0, zmm1, [rsi + 64] ; += A[16..31] * B[16..31]
    
    ; Horizontal reduction: sum all 16 elements
    ; Step 1: Add high 256 bits to low 256 bits
    vextractf32x8 ymm1, zmm0, 1        ; Get high 256 bits
    vaddps       ymm0, ymm0, ymm1      ; Add to low 256 bits
    
    ; Step 2: Add high 128 bits to low 128 bits
    vextractf128 xmm1, ymm0, 1
    vaddps       xmm0, xmm0, xmm1
    
    ; Step 3: Horizontal add remaining 4 elements
    vhaddps      xmm0, xmm0, xmm0
    vhaddps      xmm0, xmm0, xmm0
    
    ; Result in xmm0[0]
    vzeroupper
    ret
```

### Example 2: Masked Array Processing

```nasm
;-----------------------------------------------------------------------------
; clamp_array_avx512 - Clamp array values to [min_val, max_val]
;
; Input:  rdi = pointer to array (float)
;         rsi = length
;         xmm0 = min_val
;         xmm1 = max_val
; Output: Array modified in place
;-----------------------------------------------------------------------------
clamp_array_avx512:
    ; Broadcast min/max to full ZMM registers
    vbroadcastss zmm2, xmm0           ; zmm2 = [min, min, ...]
    vbroadcastss zmm3, xmm1           ; zmm3 = [max, max, ...]
    
    ; Calculate number of full 16-element iterations
    mov     rcx, rsi
    shr     rcx, 4                    ; rcx = length / 16
    and     esi, 15                   ; rsi = length % 16 (remainder)
    
    ; Process 16 elements at a time
.full_loop:
    test    rcx, rcx
    jz      .remainder
    
    vmovups      zmm0, [rdi]          ; Load 16 floats
    vmaxps       zmm0, zmm0, zmm2     ; Clamp to min
    vminps       zmm0, zmm0, zmm3     ; Clamp to max
    vmovups      [rdi], zmm0          ; Store result
    
    add     rdi, 64
    dec     rcx
    jmp     .full_loop
    
.remainder:
    test    esi, esi
    jz      .done
    
    ; Create mask for remaining elements
    mov     eax, 1
    mov     ecx, esi
    shl     eax, cl
    dec     eax
    kmovw   k1, eax
    
    ; Process remainder with mask
    vmovups      zmm0{k1}{z}, [rdi]   ; Load only remaining elements
    vmaxps       zmm0{k1}, zmm0, zmm2
    vminps       zmm0{k1}, zmm0, zmm3
    vmovups      [rdi]{k1}, zmm0      ; Store only remaining elements
    
.done:
    vzeroupper
    ret
```

### Example 3: Matrix Transpose (4x4 tile)

```nasm
;-----------------------------------------------------------------------------
; transpose_4x4_avx512 - Transpose a 4x4 float matrix tile
;
; This demonstrates AVX-512 shuffle capabilities
; Input:  rdi = pointer to 4x4 matrix (row-major, 64 bytes)
; Output: Matrix transposed in place
;-----------------------------------------------------------------------------
transpose_4x4_avx512:
    ; Load 4 rows into one ZMM register
    vmovups      zmm0, [rdi]          ; [row0, row1, row2, row3]
    
    ; Use permute to transpose
    ; Original:  [a0 a1 a2 a3 | b0 b1 b2 b3 | c0 c1 c2 c3 | d0 d1 d2 d3]
    ; Target:    [a0 b0 c0 d0 | a1 b1 c1 d1 | a2 b2 c2 d2 | a3 b3 c3 d3]
    
    mov     rax, 0x0C08_0400_0D09_0501   ; Permutation indices (low 8)
    vmovq   xmm1, rax
    mov     rax, 0x0E0A_0602_0F0B_0703   ; Permutation indices (high 8)
    vmovq   xmm2, rax
    vpunpcklqdq xmm1, xmm1, xmm2
    vinserti128 ymm1, ymm1, xmm1, 1
    
    vpermd  zmm0, zmm1, zmm0          ; Permute to transpose
    vmovups [rdi], zmm0
    
    vzeroupper
    ret
```

---

## Performance Considerations

### Clock Speed Reduction

Some CPUs reduce clock speed when running AVX-512:

| Instruction Type | Speed Reduction (typical) |
|-----------------|---------------------------|
| AVX2 (256-bit) | 0-5% |
| AVX-512 Light (adds, moves) | 5-15% |
| AVX-512 Heavy (FMA, multiplies) | 10-25% |

**Implication**: For short bursts of AVX-512, the frequency penalty may outweigh the benefit. AVX-512 shines in sustained workloads.

### Latency and Throughput

| Operation | Latency (cycles) | Throughput (CPI) |
|-----------|-----------------|------------------|
| vmovaps (ZMM load) | ~6-7 | 0.5 |
| vaddps (ZMM) | 4 | 0.5 |
| vmulps (ZMM) | 4 | 0.5 |
| vfmadd231ps (ZMM) | 4 | 0.5 |
| vdivps (ZMM) | 14-16 | 8-10 |

### When to Use AVX-512 vs AVX2

**Use AVX-512 when:**
- Processing large arrays (amortize frequency penalty)
- Need mask registers for edge handling
- Doing heavy FMA (matrix multiplication)
- Register pressure is an issue (32 vs 16 registers)

**Stick with AVX2 when:**
- Short computational bursts
- Power/thermal constrained
- Maximum compatibility needed (more CPUs support AVX2)
- Memory bandwidth is the bottleneck (512-bit won't help)

---

## Checking CPU Support

```c
#include <cpuid.h>

int check_avx512f() {
    unsigned int eax, ebx, ecx, edx;
    if (__get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx)) {
        return (ebx & bit_AVX512F) != 0;  // bit 16
    }
    return 0;
}

int check_avx512vl() {
    unsigned int eax, ebx, ecx, edx;
    if (__get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx)) {
        return (ebx & bit_AVX512VL) != 0;  // bit 31
    }
    return 0;
}
```

```nasm
; Assembly CPUID check for AVX-512F
check_avx512f:
    mov     eax, 7
    xor     ecx, ecx
    cpuid
    bt      ebx, 16        ; AVX-512F is bit 16 of EBX
    setc    al
    movzx   eax, al
    ret
```

---

## Next Steps

- [MATRIX_MULT_ALGORITHM.md](MATRIX_MULT_ALGORITHM.md) - See how to apply these instructions to matrix multiplication
- Review the actual assembly implementation in `src/avx512/matrix_mult_avx512.asm`

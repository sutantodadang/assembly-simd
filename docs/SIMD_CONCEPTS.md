# SIMD Concepts: A Comprehensive Guide

## Table of Contents
1. [What is SIMD?](#what-is-simd)
2. [Why SIMD Matters](#why-simd-matters)
3. [x86_64 Vector Registers](#x86_64-vector-registers)
4. [Data Types and Packing](#data-types-and-packing)
5. [Memory Alignment](#memory-alignment)
6. [Common SIMD Patterns](#common-simd-patterns)
7. [Understanding FMA](#understanding-fma)
8. [Performance Considerations](#performance-considerations)

---

## What is SIMD?

**SIMD** stands for **S**ingle **I**nstruction, **M**ultiple **D**ata. It's a parallel computing paradigm where one instruction operates on multiple data elements simultaneously.

### Scalar vs SIMD

Consider adding two arrays of 8 floats:

```
SCALAR (Traditional):
─────────────────────
Instruction 1: a[0] + b[0] → c[0]
Instruction 2: a[1] + b[1] → c[1]
Instruction 3: a[2] + b[2] → c[2]
Instruction 4: a[3] + b[3] → c[3]
Instruction 5: a[4] + b[4] → c[4]
Instruction 6: a[5] + b[5] → c[5]
Instruction 7: a[6] + b[6] → c[6]
Instruction 8: a[7] + b[7] → c[7]
Total: 8 instructions

SIMD with AVX2 (256-bit):
─────────────────────────
               ┌─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┐
Array a:       │a[0] │a[1] │a[2] │a[3] │a[4] │a[5] │a[6] │a[7] │
               └─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┘
                  +     +     +     +     +     +     +     +
               ┌─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┐
Array b:       │b[0] │b[1] │b[2] │b[3] │b[4] │b[5] │b[6] │b[7] │
               └─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┘
                  ↓     ↓     ↓     ↓     ↓     ↓     ↓     ↓
               ┌─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┐
Result c:      │c[0] │c[1] │c[2] │c[3] │c[4] │c[5] │c[6] │c[7] │
               └─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┘
Total: 1 instruction (vaddps)
```

### Flynn's Taxonomy

SIMD is one of four categories in Flynn's taxonomy of computer architectures:

| Type | Description | Example |
|------|-------------|---------|
| **SISD** | Single Instruction, Single Data | Traditional scalar CPU |
| **SIMD** | Single Instruction, Multiple Data | AVX2, AVX512, NEON |
| **MISD** | Multiple Instruction, Single Data | Rare (fault tolerance) |
| **MIMD** | Multiple Instruction, Multiple Data | Multi-core CPUs, GPUs |

---

## Why SIMD Matters

### Performance Benefits

1. **Throughput**: Process 8-16x more data per instruction
2. **Energy Efficiency**: Fewer instructions = less power consumption
3. **Memory Bandwidth**: Better utilization of cache lines

### Real-World Applications

| Application | SIMD Benefit |
|-------------|--------------|
| Image Processing | Process multiple pixels per operation |
| Audio/Video Encoding | Parallel sample processing |
| Machine Learning | Matrix operations, convolutions |
| Physics Simulation | Vector math for positions/velocities |
| Cryptography | Parallel block processing |
| Database Operations | Parallel comparisons and aggregations |

### The Evolution of x86 SIMD

```
Timeline of x86 SIMD Extensions:
────────────────────────────────────────────────────────────────────────────────
1997 ─┬─ MMX          │ 64-bit  │ 8× 8-bit integers
      │               │ MM0-MM7 │ 4× 16-bit integers
      │               │         │ 2× 32-bit integers
      │               │         │
1999 ─┼─ SSE          │ 128-bit │ 4× 32-bit floats
      │               │ XMM0-7  │
      │               │         │
2001 ─┼─ SSE2         │ 128-bit │ 2× 64-bit doubles
      │               │ XMM0-7  │ 16× 8-bit integers, etc.
      │               │         │
2004 ─┼─ SSE3/SSSE3   │ 128-bit │ Horizontal operations
      │               │         │
2008 ─┼─ SSE4.1/4.2   │ 128-bit │ Dot product, string ops
      │               │         │
2011 ─┼─ AVX          │ 256-bit │ 8× 32-bit floats
      │               │ YMM0-15 │ 4× 64-bit doubles
      │               │         │
2013 ─┼─ AVX2         │ 256-bit │ 32× 8-bit integers
      │  + FMA3       │ YMM0-15 │ + Fused multiply-add
      │               │         │
2017 ─┴─ AVX-512      │ 512-bit │ 16× 32-bit floats
                      │ ZMM0-31 │ 8× 64-bit doubles
                      │ k0-k7   │ + Mask registers
```

---

## x86_64 Vector Registers

### Register Hierarchy

The vector registers in x86_64 are organized hierarchically:

```
512 bits ┌───────────────────────────────────────────────────────────────────┐
         │                            ZMM0                                   │ AVX-512
         └───────────────────────────────────────────────────────────────────┘
256 bits ┌───────────────────────────────────┐
         │              YMM0                 │                                 AVX/AVX2
         └───────────────────────────────────┘
128 bits ┌─────────────────┐
         │      XMM0       │                                                   SSE
         └─────────────────┘
         ↑                 ↑                 ↑                               ↑
         0                127              255                             511
```

**Key Point**: XMM, YMM, and ZMM registers overlap! Writing to XMM0 affects the lower 128 bits of YMM0 and ZMM0.

### Available Registers

| Extension | Registers | Size | Count |
|-----------|-----------|------|-------|
| SSE/SSE2 | XMM0-XMM15 | 128 bits | 16 |
| AVX/AVX2 | YMM0-YMM15 | 256 bits | 16 |
| AVX-512 | ZMM0-ZMM31 | 512 bits | 32 |
| AVX-512 | k0-k7 | 64 bits | 8 (mask registers) |

### Register Details

```
YMM0 (256 bits = 32 bytes):
┌────────┬────────┬────────┬────────┬────────┬────────┬────────┬────────┐
│ Lane 7 │ Lane 6 │ Lane 5 │ Lane 4 │ Lane 3 │ Lane 2 │ Lane 1 │ Lane 0 │
│ 32-bit │ 32-bit │ 32-bit │ 32-bit │ 32-bit │ 32-bit │ 32-bit │ 32-bit │
│ float  │ float  │ float  │ float  │ float  │ float  │ float  │ float  │
└────────┴────────┴────────┴────────┴────────┴────────┴────────┴────────┘
  High ←──────────────────────────────────────────────────────────→ Low

Bit positions:  255:224  223:192  191:160  159:128  127:96   95:64   63:32   31:0
```

---

## Data Types and Packing

### Packed Data Types

SIMD registers can hold different data types "packed" together:

```
256-bit YMM Register Packed Data Options:

8 × 32-bit single-precision floats (PS = Packed Single):
┌────────┬────────┬────────┬────────┬────────┬────────┬────────┬────────┐
│ float  │ float  │ float  │ float  │ float  │ float  │ float  │ float  │
│  [7]   │  [6]   │  [5]   │  [4]   │  [3]   │  [2]   │  [1]   │  [0]   │
└────────┴────────┴────────┴────────┴────────┴────────┴────────┴────────┘

4 × 64-bit double-precision floats (PD = Packed Double):
┌────────────────┬────────────────┬────────────────┬────────────────┐
│    double      │    double      │    double      │    double      │
│      [3]       │      [2]       │      [1]       │      [0]       │
└────────────────┴────────────────┴────────────────┴────────────────┘

32 × 8-bit integers (B = Bytes):
┌──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┐
│31│30│29│28│27│26│25│24│23│22│21│20│19│18│17│16│15│14│13│12│11│10│ 9│ 8│ 7│ 6│ 5│ 4│ 3│ 2│ 1│ 0│
└──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┘

16 × 16-bit integers (W = Words):
┌────┬────┬────┬────┬────┬────┬────┬────┬────┬────┬────┬────┬────┬────┬────┬────┐
│ 15 │ 14 │ 13 │ 12 │ 11 │ 10 │  9 │  8 │  7 │  6 │  5 │  4 │  3 │  2 │  1 │  0 │
└────┴────┴────┴────┴────┴────┴────┴────┴────┴────┴────┴────┴────┴────┴────┴────┘

8 × 32-bit integers (D = Double words):
┌────────┬────────┬────────┬────────┬────────┬────────┬────────┬────────┐
│  int   │  int   │  int   │  int   │  int   │  int   │  int   │  int   │
│  [7]   │  [6]   │  [5]   │  [4]   │  [3]   │  [2]   │  [1]   │  [0]   │
└────────┴────────┴────────┴────────┴────────┴────────┴────────┴────────┘

4 × 64-bit integers (Q = Quad words):
┌────────────────┬────────────────┬────────────────┬────────────────┐
│    int64       │    int64       │    int64       │    int64       │
│      [3]       │      [2]       │      [1]       │      [0]       │
└────────────────┴────────────────┴────────────────┴────────────────┘
```

### Instruction Naming Convention

Intel uses a consistent naming scheme for SIMD instructions:

```
v      add       p        s
↓       ↓        ↓        ↓
VEX   Operation  Packed  Single-precision
prefix           vs      vs
(AVX)           Scalar   Double-precision

Examples:
- vaddps  = VEX ADD Packed Single
- vmulpd  = VEX MUL Packed Double
- vfmadd231ps = VEX FMA (Fused Multiply-Add) 231 Packed Single
```

---

## Memory Alignment

### Why Alignment Matters

SIMD instructions perform best when data is aligned to vector boundaries:

```
Memory Layout (each cell = 4 bytes):

Aligned access (32-byte boundary):
Address:  0    32   64   96   128  160  192  224  256
          ↓     ↓    ↓    ↓     ↓    ↓    ↓    ↓    ↓
          ┌─────────────────────────────────────────────
Memory:   │▓▓▓▓│▓▓▓▓│▓▓▓▓│▓▓▓▓│▓▓▓▓│▓▓▓▓│▓▓▓▓│▓▓▓▓│░░░░│
          └─────────────────────────────────────────────
          ← ─ ─ ─ ─ 32 bytes ─ ─ ─ ─ →
          One vmovaps instruction loads all 8 floats
          
Unaligned access (starts at offset 12):
Address:  0    12        44        108
          ↓     ↓         ↓          ↓
          ┌─────────────────────────────────────────────
Memory:   │░░░░│▓▓▓▓│▓▓▓▓│▓▓▓▓│▓▓▓▓│▓▓▓▓│▓▓▓▓│▓▓▓▓│▓▓▓▓│
          └─────────────────────────────────────────────
                ← crosses cache line boundary →
          vmovups required, may span two cache lines
```

### Alignment Requirements

| Instruction Type | Minimum Alignment | Best Alignment |
|-----------------|-------------------|----------------|
| SSE aligned (movaps) | 16 bytes | 16 bytes |
| SSE unaligned (movups) | 1 byte | 16 bytes |
| AVX aligned (vmovaps) | 32 bytes | 32 bytes |
| AVX unaligned (vmovups) | 1 byte | 32 bytes |
| AVX-512 aligned (vmovaps) | 64 bytes | 64 bytes |
| AVX-512 unaligned (vmovups) | 1 byte | 64 bytes |

### Allocating Aligned Memory

```c
// C11 standard
float* ptr = aligned_alloc(32, size * sizeof(float));

// POSIX
float* ptr;
posix_memalign((void**)&ptr, 32, size * sizeof(float));

// Windows
float* ptr = _aligned_malloc(size * sizeof(float), 32);

// C++ 17
float* ptr = std::aligned_alloc(32, size * sizeof(float));
```

---

## Common SIMD Patterns

### Pattern 1: Load-Process-Store

The most basic SIMD pattern:

```nasm
; Load 8 floats from memory into YMM register
vmovaps ymm0, [rdi]           ; ymm0 = A[0..7]

; Process (e.g., add 1.0 to all elements)
vaddps  ymm0, ymm0, ymm1      ; ymm0 = ymm0 + ymm1

; Store back to memory
vmovaps [rsi], ymm0           ; B[0..7] = ymm0
```

### Pattern 2: Horizontal Reduction

Sum all elements in a vector:

```nasm
; Start: ymm0 = [a, b, c, d, e, f, g, h]

; Step 1: Add upper 128 bits to lower 128 bits
vextractf128 xmm1, ymm0, 1    ; xmm1 = [e, f, g, h]
vaddps       xmm0, xmm0, xmm1 ; xmm0 = [a+e, b+f, c+g, d+h]

; Step 2: Horizontal add within 128 bits
vhaddps      xmm0, xmm0, xmm0 ; xmm0 = [a+e+b+f, c+g+d+h, ...]
vhaddps      xmm0, xmm0, xmm0 ; xmm0 = [total, total, ...]

; Result: xmm0[0] contains sum of all 8 elements
```

### Pattern 3: Broadcast (Splat)

Copy a single value to all lanes:

```nasm
; Broadcast a single float to all 8 lanes
vbroadcastss ymm0, [rdi]      ; ymm0 = [x, x, x, x, x, x, x, x]

; Or from register (AVX2)
vbroadcastss ymm0, xmm1       ; Broadcast xmm1[0] to all lanes
```

### Pattern 4: Gather/Scatter

Load non-contiguous elements (AVX2+):

```nasm
; Gather: Load elements at indices stored in ymm1
; base address in rdi, indices in ymm1 (32-bit integers)
vgatherdps ymm0, [rdi + ymm1*4], ymm2  ; ymm2 is mask

; Scatter (AVX-512 only): Store to non-contiguous locations
vscatterdps [rdi + ymm1*4]{k1}, ymm0   ; k1 is mask register
```

### Pattern 5: Masked Operations (AVX-512)

```nasm
; Create a mask (e.g., first 5 elements)
mov     eax, 0b00011111
kmovw   k1, eax

; Conditional load
vmovups zmm0{k1}{z}, [rdi]    ; Load only first 5 elements, zero rest

; Conditional add
vaddps  zmm0{k1}, zmm0, zmm1  ; Only add where mask is 1

; Conditional store
vmovups [rsi]{k1}, zmm0       ; Only store where mask is 1
```

---

## Understanding FMA

### What is FMA?

**FMA** (Fused Multiply-Add) computes `a * b + c` in a single instruction with only one rounding step, providing:

1. **Better precision**: Single rounding vs two roundings
2. **Better performance**: One instruction instead of two
3. **Lower latency**: Typically 4-5 cycles vs 7-8 cycles for separate mul+add

### FMA Instruction Variants

The 3 digits in FMA instructions (132, 213, 231) specify operand ordering:

```
vfmadd132ps  ymm1, ymm2, ymm3    →  ymm1 = ymm1 * ymm3 + ymm2
vfmadd213ps  ymm1, ymm2, ymm3    →  ymm1 = ymm2 * ymm1 + ymm3  
vfmadd231ps  ymm1, ymm2, ymm3    →  ymm1 = ymm2 * ymm3 + ymm1  ← Most common!
```

### Why vfmadd231 is Preferred

`vfmadd231` is ideal for accumulation patterns:

```nasm
; Computing dot product: sum += a[i] * b[i]
; ymm0 = accumulator (sum)
; ymm1 = a[i..i+7]
; ymm2 = b[i..i+7]

vfmadd231ps ymm0, ymm1, ymm2    ; ymm0 += ymm1 * ymm2
                                ; ymm0 = ymm1 * ymm2 + ymm0
                                ; Accumulator stays in ymm0!
```

### FMA Variants

| Instruction | Operation | Use Case |
|-------------|-----------|----------|
| vfmadd231ps | d = a*b + d | Accumulation |
| vfnmadd231ps | d = -(a*b) + d | Subtraction |
| vfmsub231ps | d = a*b - d | Subtract accumulator |
| vfnmsub231ps | d = -(a*b) - d | Negate all |

---

## Performance Considerations

### Instruction Throughput and Latency

| Instruction | Latency (cycles) | Throughput (CPI) |
|-------------|-----------------|------------------|
| vmovaps | 4-7 (from memory) | 0.5 |
| vaddps | 4 | 0.5 |
| vmulps | 4 | 0.5 |
| vfmadd231ps | 4 | 0.5 |
| vdivps | 11-13 | 5-8 |
| vsqrtps | 12-18 | 6-12 |

*Values vary by CPU microarchitecture*

### Port Utilization

Modern CPUs have multiple execution ports. AVX instructions typically use:
- **Port 0**: FMA, multiply
- **Port 1**: FMA, add
- **Port 5**: Shuffles, permutes

To maximize throughput, interleave different instruction types:

```nasm
; Good: Different ports used
vfmadd231ps ymm0, ymm1, ymm2   ; Port 0 or 1
vpermps     ymm3, ymm4, ymm5   ; Port 5
vfmadd231ps ymm6, ymm7, ymm8   ; Port 0 or 1

; Less optimal: Same port contention
vfmadd231ps ymm0, ymm1, ymm2   ; Port 0 or 1
vfmadd231ps ymm3, ymm4, ymm5   ; Port 0 or 1 (may stall)
vfmadd231ps ymm6, ymm7, ymm8   ; Port 0 or 1 (may stall)
```

### Cache Optimization

```
L1 Cache: ~32KB, ~4 cycles latency
L2 Cache: ~256KB, ~12 cycles latency  
L3 Cache: ~8MB+, ~40 cycles latency
RAM: ~100+ cycles latency

Strategy for Matrix Multiplication:
1. Process small "tiles" that fit in L1 cache
2. Reuse data while it's in cache
3. Access data in sequential order (prefetch-friendly)
```

### AVX-SSE Transition Penalty

Mixing legacy SSE and AVX instructions causes performance penalties on older CPUs:

```nasm
; BAD: SSE instruction after AVX
vaddps  ymm0, ymm1, ymm2    ; AVX instruction
movaps  xmm3, [rdi]         ; Legacy SSE - causes penalty!
vaddps  ymm4, ymm5, ymm6    ; Back to AVX

; GOOD: Use VEX-encoded versions
vaddps  ymm0, ymm1, ymm2    ; AVX instruction
vmovaps xmm3, [rdi]         ; VEX-encoded SSE - no penalty
vaddps  ymm4, ymm5, ymm6    ; AVX instruction
```

Use `vzeroupper` before returning from AVX code to SSE code:

```nasm
; End of AVX function
vzeroupper                   ; Clear upper 128 bits of all YMM
ret
```

---

## Next Steps

Now that you understand SIMD fundamentals, continue to:
- [AVX2_REFERENCE.md](AVX2_REFERENCE.md) - Detailed AVX2 instruction reference
- [AVX512_REFERENCE.md](AVX512_REFERENCE.md) - AVX512 specifics
- [MATRIX_MULT_ALGORITHM.md](MATRIX_MULT_ALGORITHM.md) - Matrix multiplication optimization

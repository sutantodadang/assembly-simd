# x86-64 Assembly Guide for SIMD Developers

This guide explains the fundamental assembly language concepts used in this project. It is designed to help you understand the non-SIMD parts of the code, such as loop control, memory addressing, and function calls.

For specific SIMD instructions (like `vfmadd231ps`), please refer to:
- [**AVX2 Reference**](AVX2_REFERENCE.md)
- [**AVX-512 Reference**](AVX512_REFERENCE.md)

---

## 1. Registers

The CPU has different types of registers for different purposes.

### General Purpose Registers (GPRs) - 64-bit
Used for **pointers**, **counters**, and **integer arithmetic**.

| Register | Common Use in This Project | Notes |
|----------|----------------------------|-------|
| `rax`    | Return value, temporary math | Lower 32-bits is `eax` |
| `rbx`    | Base pointer for Matrix A | **Callee-saved** (must preserve) |
| `rcx`    | Loop counter, argument 4 | Used for `rep` string ops |
| `rdx`    | Argument 3 | |
| `rsi`    | Source index (Matrix B) | Argument 2 (System V) |
| `rdi`    | Destination index (Matrix A) | Argument 1 (System V) |
| `rbp`    | Base Pointer (Stack Frame) | **Callee-saved** |
| `rsp`    | Stack Pointer | Points to top of stack |
| `r8` - `r15` | General use, loop counters | `r12`-`r15` are **Callee-saved** |

> **Callee-saved**: If we use these registers, we MUST `push` them at the start and `pop` them at the end of the function to restore their original values for the caller.

### SIMD Registers (Vector Registers)
Used for **floating-point data** and **vector operations**.

| Name | Width | Holds |
|------|-------|-------|
| `xmm0` - `xmm15` | 128-bit | 4 floats |
| `ymm0` - `ymm15` | 256-bit | 8 floats (AVX2) |
| `zmm0` - `zmm31` | 512-bit | 16 floats (AVX-512) |

---

## 2. Basic Instructions

### Data Movement (`mov`)
Moves data between registers and memory.

```nasm
mov rax, 10          ; rax = 10
mov rbx, rax         ; rbx = rax
mov rax, [rbp - 8]   ; Load value from stack memory into rax
mov [rdi], eax       ; Store 32-bit value (eax) to memory at [rdi]
```

### Arithmetic
```nasm
add rax, 4           ; rax = rax + 4
sub rbx, 1           ; rbx = rbx - 1
inc rcx              ; rcx = rcx + 1 (increment)
dec rcx              ; rcx = rcx - 1 (decrement)
shl rax, 2           ; Shift Left: rax = rax * 4 (2^2)
xor rax, rax         ; Exclusive OR with self = 0 (Fastest way to zero a register)
```

### Comparison and Jumps
Assembly uses `cmp` (compare) followed by a conditional jump to create `if` statements and loops.

```nasm
cmp rax, rbx         ; Compare rax and rbx
je  .equal           ; Jump if Equal
rne .not_equal       ; Jump if Not Equal
jl  .less            ; Jump if Less 
jg  .greater         ; Jump if Greater
jz  .zero            ; Jump if Zero (result of previous arithmetic was 0)
jnz .not_zero        ; Jump if Not Zero
```

---

## 3. Addressing Modes

How we calculate memory addresses to read/write data.

### 1. Basic: `[reg]`
Access memory at the address stored in the register.
```nasm
mov rax, [rdi]       ; Read value from address in rdi
```

### 2. Displacement: `[reg + offset]`
Access memory at "address + constant". Useful for struct fields or stack variables.
```nasm
mov rax, [rbp + 16]  ; Read value at rbp + 16
```

### 3. Indexed: `[base + index * scale]`
Perfect for arrays. Scale can be 1, 2, 4, or 8.
```nasm
; rdi = array base, rax = index (0, 1, 2...)
movss xmm0, [rdi + rax*4]  ; Read float (4 bytes) at index rax
```

### 4. Complex: `[base + index * scale + offset]`
```nasm
vmovaps ymm0, [rdi + rax*4 + 32]
```

---

## 4. Common SIMD Instructions Explained

User asked specifically about these:

### `vmovaps` vs `vmovups` (Move Aligned/Unaligned Packed Singles)
- **`vmovaps`**: usage `vmovaps ymm0, [rax]`.
  - Expects the memory address in `rax` to be a **multiple of 32 bytes** (e.g., ends in ...00, ...20, ...40).
  - **CRASHES** if address is not aligned!
  - Slightly faster on older hardware.
- **`vmovups`**: usage `vmovups ymm0, [rax]`.
  - Works with **any** address.
  - Safe to use always. Modern CPUs run this almost as fast as aligned.

### `vxorps` (XOR Packed Singles)
- Usage: `vxorps ymm0, ymm0, ymm0`
- Logic: `A XOR A = 0`.
- Effect: **Sets the entire register to ZERO.**
- Why? It's the standard, most efficient idiom to clear a vector register.

### `vbroadcastss` (Broadcast Scalar Single)
- Usage: `vbroadcastss ymm0, [rax]`
- Effect: Reads **one** float (32-bit) from memory and **copies it** to all 8 positions in `ymm0`.
- Used for: Multiplying a whole vector by a single scalar value (like a matrix cell `A[i][k]`).

### `vfmadd231ps` (Fused Multiply-Add)
- Usage: `vfmadd231ps ymm0, ymm1, ymm2`
- Operation: `ymm0 = (ymm1 * ymm2) + ymm0`
- Why "Fused"? It does the multiply and add in a **single step**, retaining higher precision and performance than separate `vmulps` and `vaddps`.
- Why "231"? Refers to the operand order: `(2 * 3) + 1`.

---

## 5. Control Flow & Stack (The "Hidden" Logic)

### The Stack (`rsp`, `rbp`)
Memory usage for local variables and function calls.
- **`push reg`**: Decrements `rsp` by 8, saves `reg` value to memory.
- **`pop reg`**: Loads value from memory, increments `rsp` by 8.
- **`sub rsp, N`**: Allocates `N` bytes of local storage.

### Loops
A standard loop structure in our code:

```nasm
    xor rcx, rcx          ; rcx = 0 (Loop Counter)

.loop_start:
    cmp rcx, 100          ; Compare counter to limit
    jge .loop_end         ; If rcx >= 100, exit loop

    ; ... do work ...

    inc rcx               ; Counter++
    jmp .loop_start       ; Jump back to start

.loop_end:
```

### Function Calls & ABI (Application Binary Interface)
Why do we do `mov r8, rdx` etc?
- **Windows** and **Linux** pass arguments in different registers.
- **Windows**: `rcx`, `rdx`, `r8`, `r9`.
- **Linux**: `rdi`, `rsi`, `rdx`, `rcx`, `r8`, `r9`.
- Our code defines `matrix_mult_avx2` to be compatible by shuffling registers at the start so the common logic works on both.

---

## 6. How it all fits in Matrix Multiplication

1. **Outer Loops**: Standard x86 assembly (`cmp`, `jl`, `inc`) control the `i` (row) and `j` (column) counters.
2. **Inner Kernel**: 
   - Uses `vbroadcastss` to fill a register with `A[i][k]`.
   - Uses `vmovaps` to load a chunk of `B[k][j...j+7]`.
   - Uses `vfmadd231ps` to multiply them and add to the `Accumulator`.
3. **Storage**: Finally uses `vmovaps` to write the `Accumulator` to `C[i][j...j+7]`.


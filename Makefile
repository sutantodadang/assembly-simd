#==============================================================================
#                    SIMD Matrix Multiplication - Makefile
#==============================================================================
#
# Build targets:
#   make all       - Build everything (default)
#   make avx2      - Build only AVX2 version
#   make avx512    - Build only AVX-512 version
#   make test      - Build and run tests
#   make benchmark - Build and run benchmarks
#   make clean     - Remove build artifacts
#
# Configuration:
#   DEBUG=1        - Build with debug symbols
#   VERBOSE=1      - Show full command lines
#
#==============================================================================

#------------------------------------------------------------------------------
# Configuration
#------------------------------------------------------------------------------

# Compiler and assembler
CC = gcc
NASM = nasm

# Output directories
BUILD_DIR = build
OBJ_DIR = $(BUILD_DIR)/obj
BIN_DIR = $(BUILD_DIR)/bin

# Source files
AVX2_ASM = src/avx2/matrix_mult_avx2.asm
AVX512_ASM = src/avx512/matrix_mult_avx512.asm
TEST_SRC = test/test_main.c

# Output files
AVX2_OBJ = $(OBJ_DIR)/matrix_mult_avx2.o
AVX512_OBJ = $(OBJ_DIR)/matrix_mult_avx512.o
TEST_OBJ = $(OBJ_DIR)/test_main.o

TEST_BIN = $(BIN_DIR)/test_matrix_mult

# Detect OS
UNAME_S := $(shell uname -s)

ifeq ($(UNAME_S),Darwin)
    # macOS
    NASM_FORMAT = macho64
    NASM_FLAGS = -f $(NASM_FORMAT) -DMACHO
    # On macOS, C symbols don't have underscore prefix in 64-bit mode
    # But NASM needs to know about it
    CFLAGS_OS = -D_POSIX_VERSION=200112L
else ifeq ($(UNAME_S),Linux)
    # Linux
    NASM_FORMAT = elf64
    NASM_FLAGS = -f $(NASM_FORMAT)
    CFLAGS_OS = -D_POSIX_VERSION=200112L
else
    # Assume Windows (MinGW)
    NASM_FORMAT = win64
    NASM_FLAGS = -f $(NASM_FORMAT) -DWIN64
    CFLAGS_OS = 
endif

# Compiler flags
CFLAGS = -Wall -Wextra -std=c11 $(CFLAGS_OS)
CFLAGS += -I./src/include

# Debug/Release configuration
ifdef DEBUG
    CFLAGS += -g -O0 -DDEBUG
    NASM_FLAGS += -g
else
    CFLAGS += -O3 -march=native -DNDEBUG
endif

# Linker flags
LDFLAGS = -lm

# Verbose mode
ifdef VERBOSE
    Q =
else
    Q = @
endif

#------------------------------------------------------------------------------
# Targets
#------------------------------------------------------------------------------

.PHONY: all avx2 avx512 test benchmark clean help dirs

# Default target
all: dirs $(TEST_BIN)
	@echo "Build complete!"
	@echo "Run './$(TEST_BIN)' to execute tests"
	@echo "Run './$(TEST_BIN) --benchmark' to run benchmarks"

# Create output directories
dirs:
	$(Q)mkdir -p $(OBJ_DIR) $(BIN_DIR)

# Build only AVX2 object
avx2: dirs $(AVX2_OBJ)
	@echo "AVX2 object built: $(AVX2_OBJ)"

# Build only AVX-512 object
avx512: dirs $(AVX512_OBJ)
	@echo "AVX-512 object built: $(AVX512_OBJ)"

# Build and run tests
test: all
	@echo ""
	@echo "Running tests..."
	@echo "═══════════════════════════════════════"
	$(Q)./$(TEST_BIN)

# Build and run benchmarks
benchmark: all
	@echo ""
	@echo "Running benchmarks..."
	@echo "═══════════════════════════════════════"
	$(Q)./$(TEST_BIN) --benchmark

# Clean build artifacts
clean:
	$(Q)rm -rf $(BUILD_DIR)
	@echo "Build directory cleaned"

# Help
help:
	@echo "SIMD Matrix Multiplication Build System"
	@echo ""
	@echo "Targets:"
	@echo "  all       - Build everything (default)"
	@echo "  avx2      - Build only AVX2 assembly"
	@echo "  avx512    - Build only AVX-512 assembly"
	@echo "  test      - Build and run tests"
	@echo "  benchmark - Build and run benchmarks"
	@echo "  clean     - Remove build artifacts"
	@echo ""
	@echo "Options:"
	@echo "  DEBUG=1   - Build with debug symbols"
	@echo "  VERBOSE=1 - Show full command lines"
	@echo ""
	@echo "Examples:"
	@echo "  make"
	@echo "  make DEBUG=1 test"
	@echo "  make VERBOSE=1 benchmark"

#------------------------------------------------------------------------------
# Rules
#------------------------------------------------------------------------------

# Assemble AVX2 source
$(AVX2_OBJ): $(AVX2_ASM) | dirs
	@echo "  NASM    $<"
	$(Q)$(NASM) $(NASM_FLAGS) -o $@ $<

# Assemble AVX-512 source
$(AVX512_OBJ): $(AVX512_ASM) | dirs
	@echo "  NASM    $<"
	$(Q)$(NASM) $(NASM_FLAGS) -o $@ $<

# Compile test source
$(TEST_OBJ): $(TEST_SRC) src/include/matrix_mult.h | dirs
	@echo "  CC      $<"
	$(Q)$(CC) $(CFLAGS) -c -o $@ $<

# Link test binary
$(TEST_BIN): $(TEST_OBJ) $(AVX2_OBJ) $(AVX512_OBJ)
	@echo "  LINK    $@"
	$(Q)$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

#------------------------------------------------------------------------------
# Dependencies
#------------------------------------------------------------------------------

# Header dependencies
$(TEST_OBJ): src/include/matrix_mult.h

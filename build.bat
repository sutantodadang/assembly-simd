@echo off
REM ==============================================================================
REM                    SIMD Matrix Multiplication - Windows Build Script
REM ==============================================================================
REM
REM Prerequisites:
REM   - NASM (Netwide Assembler) installed and in PATH
REM   - MinGW-w64 (GCC) installed and in PATH
REM     OR
REM   - Visual Studio with MSVC (use build_msvc.bat instead)
REM
REM Usage:
REM   build.bat           - Build everything
REM   build.bat avx2      - Build only AVX2
REM   build.bat avx512    - Build only AVX-512
REM   build.bat test      - Build and run tests
REM   build.bat benchmark - Build and run benchmarks
REM   build.bat clean     - Clean build artifacts
REM   build.bat help      - Show this help
REM
REM ==============================================================================

setlocal enabledelayedexpansion

REM Configuration
set BUILD_DIR=build
set OBJ_DIR=%BUILD_DIR%\obj
set BIN_DIR=%BUILD_DIR%\bin

set AVX2_ASM=src\avx2\matrix_mult_avx2.asm
set AVX512_ASM=src\avx512\matrix_mult_avx512.asm
set TEST_SRC=test\test_main.c

set AVX2_OBJ=%OBJ_DIR%\matrix_mult_avx2.obj
set AVX512_OBJ=%OBJ_DIR%\matrix_mult_avx512.obj
set TEST_OBJ=%OBJ_DIR%\test_main.obj
set TEST_BIN=%BIN_DIR%\test_matrix_mult.exe

REM NASM settings (Windows uses win64 format)
set NASM_FLAGS=-f win64 -DWIN64

REM GCC settings
set CC=gcc
set CFLAGS=-Wall -Wextra -std=c11 -O3 -march=native -I.\src\include
set LDFLAGS=-lm

REM Parse command line argument
if "%1"=="" goto :all
if "%1"=="all" goto :all
if "%1"=="avx2" goto :avx2
if "%1"=="avx512" goto :avx512
if "%1"=="test" goto :test
if "%1"=="benchmark" goto :benchmark
if "%1"=="clean" goto :clean
if "%1"=="help" goto :help
echo Unknown target: %1
goto :help

:help
echo SIMD Matrix Multiplication - Windows Build Script
echo.
echo Usage: build.bat [target]
echo.
echo Targets:
echo   all       - Build everything (default)
echo   avx2      - Build only AVX2 assembly
echo   avx512    - Build only AVX-512 assembly
echo   test      - Build and run tests
echo   benchmark - Build and run benchmarks
echo   clean     - Remove build artifacts
echo   help      - Show this help
echo.
echo Prerequisites:
echo   - NASM (https://www.nasm.us/)
echo   - MinGW-w64 GCC (https://winlibs.com/)
echo.
goto :eof

:dirs
if not exist "%BUILD_DIR%" mkdir "%BUILD_DIR%"
if not exist "%OBJ_DIR%" mkdir "%OBJ_DIR%"
if not exist "%BIN_DIR%" mkdir "%BIN_DIR%"
goto :eof

:check_nasm
where nasm >nul 2>nul
if errorlevel 1 (
    echo ERROR: NASM not found in PATH!
    echo Please install NASM from https://www.nasm.us/
    exit /b 1
)
goto :eof

:check_gcc
where gcc >nul 2>nul
if errorlevel 1 (
    echo ERROR: GCC not found in PATH!
    echo Please install MinGW-w64 from https://winlibs.com/
    exit /b 1
)
goto :eof

:avx2
call :check_nasm
if errorlevel 1 exit /b 1
call :dirs
echo Assembling AVX2...
nasm %NASM_FLAGS% -o "%AVX2_OBJ%" "%AVX2_ASM%"
if errorlevel 1 (
    echo ERROR: Failed to assemble AVX2 source
    exit /b 1
)
echo AVX2 object built: %AVX2_OBJ%
goto :eof

:avx512
call :check_nasm
if errorlevel 1 exit /b 1
call :dirs
echo Assembling AVX-512...
nasm %NASM_FLAGS% -o "%AVX512_OBJ%" "%AVX512_ASM%"
if errorlevel 1 (
    echo ERROR: Failed to assemble AVX-512 source
    exit /b 1
)
echo AVX-512 object built: %AVX512_OBJ%
goto :eof

:all
call :check_nasm
if errorlevel 1 exit /b 1
call :check_gcc
if errorlevel 1 exit /b 1
call :dirs

echo.
echo ========================================
echo Building SIMD Matrix Multiplication
echo ========================================
echo.

echo [1/4] Assembling AVX2...
nasm %NASM_FLAGS% -o "%AVX2_OBJ%" "%AVX2_ASM%"
if errorlevel 1 (
    echo ERROR: Failed to assemble AVX2 source
    exit /b 1
)

echo [2/4] Assembling AVX-512...
nasm %NASM_FLAGS% -o "%AVX512_OBJ%" "%AVX512_ASM%"
if errorlevel 1 (
    echo ERROR: Failed to assemble AVX-512 source
    exit /b 1
)

echo [3/4] Compiling test harness...
%CC% %CFLAGS% -c -o "%TEST_OBJ%" "%TEST_SRC%"
if errorlevel 1 (
    echo ERROR: Failed to compile test source
    exit /b 1
)

echo [4/4] Linking...
%CC% %CFLAGS% -o "%TEST_BIN%" "%TEST_OBJ%" "%AVX2_OBJ%" "%AVX512_OBJ%" %LDFLAGS%
if errorlevel 1 (
    echo ERROR: Failed to link executable
    exit /b 1
)

echo.
echo Build complete!
echo.
echo Run '%TEST_BIN%' to execute tests
echo Run '%TEST_BIN% --benchmark' to run benchmarks
goto :eof

:test
call :all
if errorlevel 1 exit /b 1
echo.
echo ========================================
echo Running Tests
echo ========================================
echo.
"%TEST_BIN%"
goto :eof

:benchmark
call :all
if errorlevel 1 exit /b 1
echo.
echo ========================================
echo Running Benchmarks
echo ========================================
echo.
"%TEST_BIN%" --benchmark
goto :eof

:clean
echo Cleaning build directory...
if exist "%BUILD_DIR%" rd /s /q "%BUILD_DIR%"
echo Done.
goto :eof

#!/usr/bin/env bats

setup_file() {
  # locate taco source folder
  SRCDIR=${CMAKE_SOURCE_DIR}
  if [ x"${SRCDIR}" = x ]; then
    SRCDIR=$(dirname $0)/../../../..
  fi
  export SRCDIR

  # locate built taco executable
  TACO="${CMAKE_BUILD_DIR}/bin/taco"
  if [ ! -x "$TACO" ]; then
    TACO=`pwd`/../bin/taco
  fi
  if [ ! -x "$TACO" ]; then
    echo "No taco executable found.  Please set \$CMAKE_BUILD_DIR"
    exit 1
  fi
  export TACO

  # detect taco features
  HAS_OPENMP=0
  HAS_PYTHON=0
  HAS_CUDA=0
  rm -f $DEBUGFILE
  features=($($TACO -version | grep '^Built with' | grep 'support.$' | awk '{print $3}'))
  for FEATURE in "${features[@]}"; do
    case $FEATURE in
      OpenMP) HAS_OPENMP=1;;
      Python) HAS_PYTHON=1;;
      CUDA)   HAS_CUDA=1;;
      *) echo "Unknown taco feature '$FEATURE' found in 'taco -version' output";;
    esac
  done
  export HAS_OPENMP
  export HAS_PYTHON
  export HAS_CUDA

  # locate C compiler
  if [ x$CC = x ]; then
    CC=gcc
  fi
  if ! which ${CC}; then
    echo "No C compiler found.  Please add gcc to \$PATH or set \$CC"
    exit 1
  fi
  if [ x$CFLAGS = x ]; then
    CFLAGS="-O2 -Wall -pedantic"
    if [ $HAS_OPENMP = 1 ]; then
      CFLAGS="$CFLAGS -fopenmp"
    fi
  fi
  CC=$(which ${CC})
  export CC
  export CFLAGS
  export TACO_CC=$CC

  if [ $HAS_CUDA = 1 ]; then
    # locate nvcc
    if [ "x$NVCC" = x ]; then
      NVCC=nvcc
    fi
    if ! which ${NVCC}; then
      echo "No 'nvcc' found.  Please add nvcc to \$PATH or set \$NVCC"
      exit 1
    fi
    NVCC=$(which ${NVCC})
    if [ x$NVCCFLAGS = x ]; then
      NVCCFLAGS="--gpu-architecture=compute_61"
    fi
    export NVCC
    export NVCCFLAGS
  fi
}

@test 'taco command line exists' {
  echo command line tool should exist at "$TACO"
  test -x $TACO
}

@test 'invoking without arguments prints usage' {
  run $TACO
  [ "${lines[0]}" = "Usage: taco <index expression> [options]" ]
}

@test 'invoking with -help prints usage' {
  run $TACO -help
  [ "${lines[0]}" = "Usage: taco <index expression> [options]" ]
}

@test 'invoking with -help=scheduling prints scheduling directives' {
  run $TACO -help=scheduling
  [ "${lines[0]}" = "Scheduling commands modify the execution of the index expression." ]
}

@test 'invoking with -version prints version info' {
  run $TACO -version
  echo "${lines[0]}" | grep '^TACO version: '
}

tacogen() {
  run $TACO "$@" -print-nocolor
}

@test 'invoking with expression generates code' {
  tacogen 'a(i) = b(i)'
  [ "${status}" -eq 0 ]
  echo "${lines[0]}"
  [ "${lines[0]}" = '// Generated by the Tensor Algebra Compiler (tensor-compiler.org)' ]
}

tacocompile() {
  compiler="${CC} ${CFLAGS}"
  filename=${BATS_TMPDIR}/taco_test_program.c
  # TODO: use --write-source: see tensor-compiler/taco#438
  # $TACO "$@" -write-source=${BATS_TMPDIR}/taco_test_source.c
  $TACO "$@" -write-compute=${BATS_TMPDIR}/taco_test_source.c
  is_cuda_code=0
  if grep cudaDeviceSynchronize ${BATS_TMPDIR}/taco_test_source.c >/dev/null; then
    is_cuda_code=1
    if [ $HAS_CUDA = 0 ]; then
      echo skipping cuda build in non-cuda configuration
      return
    fi
  fi
  if [ $is_cuda_code -eq 1 ]; then
    compiler="${NVCC} ${NVCCFLAGS}"
    filename=${BATS_TMPDIR}/taco_test_program.cu
  fi
  (
    echo "#include \"${SRCDIR}/include/taco/taco_tensor_t.h\"";
    # TODO: remove this boilerplate in favor of --write-source: see tensor-compiler/taco#438
    cat <<EOF
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <math.h>
#include <complex.h>
#define TACO_MIN(_a,_b) ((_a) < (_b) ? (_a) : (_b))
#define TACO_MAX(_a,_b) ((_a) > (_b) ? (_a) : (_b))
EOF
    if [ $is_cuda_code -eq 1 ]; then true;
    fi
    cat ${BATS_TMPDIR}/taco_test_source.c
    echo "int main() {}"
  ) > ${filename}
  CMDLINE="${compiler} -o ${BATS_TMPDIR}/taco_test_program.exe ${filename}"
  echo compile command: $CMDLINE
  run $CMDLINE
  echo compiler returned "$status"
  echo output: "$output"
  [ "${status}" -eq 0 ]
}

@test 'generated code can compile' {
  tacocompile 'a(i) = b(i)'
  [ "${status}" -eq 0 ]
}

@test 'test -s=reorder' {
  expression="a(i,j) = b(i,k) * c(k,j)"
  scheduling_directives=(
    "-s=reorder(i,j,k)"
    "-s=reorder(k,i,j)"
  )
  for directive in "${scheduling_directives[@]}"; do
    echo "trying: taco '$expression' '$directive'"
    tacocompile "$expression" "$directive"
  done
}

@test 'test -s=split' {
  expression="a(i,j) = b(i,k) * c(k,j)"
  scheduling_directives=(
    "-s=split(i,i0,i1,4)"
  )
  for directive in "${scheduling_directives[@]}"; do
    echo "trying: taco '$expression' '$directive'"
    tacocompile "$expression" "$directive"
  done
}

@test 'test -s=divide' {
  expression="a(i,j) = b(i,k) * c(k,j)"
  scheduling_directives=(
    "-s=divide(i,i0,i1,4)"
  )
  for directive in "${scheduling_directives[@]}"; do
    echo "trying: taco '$expression' '$directive'"
    tacocompile "$expression" "$directive"
  done
}

@test 'test -s=fuse' {
  expression="a(i,j) = b(i,k) * c(k,j)"
  scheduling_directives=(
    "-s=reorder(i,j,k),fuse(i,j,ij)"
    #"-s=split(i,i0,i1,4),fuse(i0,i1,i)"
  )
  for directive in "${scheduling_directives[@]}"; do
    echo "trying: taco '$expression' '$directive'"
    tacocompile "$expression" "$directive"
  done
}

@test 'test -s=pos' {
  expression="a(i,j) = b(i,k) * c(k,j)"
  scheduling_directives=(
    "-s=pos(i,ipos,a)"
  )
  for directive in "${scheduling_directives[@]}"; do
    echo "trying: taco '$expression' '$directive'"
    tacocompile "$expression" "$directive"
  done
}

@test 'test -s=precompute' {
  expression="a(i) = b(i,j)*1 * c(j)"
  format="-f=b:ds"
  scheduling_directives=(
    "-s=precompute(b(i,j)*1*c(j),j,j)"
    "-s=precompute(b(i,j)*1*c(j),j,k)"
    "-s=precompute(b(i,j)*1,j,j)"
    "-s=precompute(b(i,j)*1,j,k)"
  )
  for directive in "${scheduling_directives[@]}"; do
    echo "trying: taco '$expression' '$format' '$directive'"
    tacocompile "$expression" "$format" "$directive"
  done
}

@test 'test -s=bound' {
  expression="a(i,j) = b(i,k) * c(k,j)"
  scheduling_directives=(
    "-s=bound(i,ibound,100,MaxExact)"
    "-s=bound(j,jbound,100,MaxExact)"
    "-s=bound(k,kbound,100,MaxExact)"
  )
  for directive in "${scheduling_directives[@]}"; do
    echo "trying: taco '$expression' '$directive'"
    tacocompile "$expression" "$directive"
  done

  # These should all die with "Not supported yet"
  scheduling_directives=(
    "-s=bound(k,kbound,100,MinExact)"
    "-s=bound(k,kbound,100,MaxConstraint)"
    "-s=bound(k,kbound,100,MinConstraint)"
  )

  for directive in "${TODO_scheduling_directives[@]}"; do
    echo "this should fail: taco '$expression' '$directive'"
    run $TACO "$expression" "$directive"
    echo status=$status
    [ $status -ne 0 ]
    echo "$output" | grep "Not supported yet"
  done

  # This should die with "Bound type not defined"
  directive="-s=bound(k,kbound,100,Unknown)"

  echo "this should fail: taco '$expression' '$directive'"
  run $TACO "$expression" "$directive"
  echo status=$status
  [ $status -ne 0 ]
  echo "output: $output"
  echo "$output" | grep "Bound type not defined"
}

@test 'test -s=unroll' {
  expression="a(i,j) = b(i,k) * c(k,j)"
  scheduling_directives=(
    "-s=reorder(i,j,k),split(k,k0,k1,32),unroll(k0,4)"
    "-s=reorder(i,j,k),bound(k,k0,32,MaxExact),unroll(k0,4)"
  )
  for directive in "${scheduling_directives[@]}"; do
    echo "trying: taco '$expression' '$directive'"
    tacocompile "$expression" "$directive"
  done

  # This should die with "Unable to vectorize or unroll loop over unbound variable"
  directive="-s=reorder(i,j,k),unroll(k,4)"

  echo "this should fail: taco '$expression' '$directive'"
  run $TACO "$expression" "$directive"
  echo status=$status
  [ $status -ne 0 ]
  echo "output: $output"
  echo "$output" | grep "unbound variable k"
}

@test 'test -s=assemble' {
  expression="a(i) = b(i,j)*1 * c(j)"
  format="-f=b:ds"
  scheduling_directives=(
    "-s=assemble(a,Append)"
    "-s=assemble(a,Insert)"
    "-s=assemble(a,Append,true)"
    "-s=assemble(a,Append,false)"
    "-s=assemble(a,Insert,true)"
    "-s=assemble(a,Insert,false)"
  )
  for directive in "${scheduling_directives[@]}"; do
    echo "trying: taco '$expression' '$format' '$directive'"
    tacocompile "$expression" "$format" "$directive"
  done

  directive="-s=assemble(b,Append)"
  echo "this should fail: taco '$expression' '$directive'"
  run $TACO "$expression" "$directive"
  echo status=$status
  [ $status -ne 0 ]
  echo "$output" | grep "Unable to find result tensor 'b'"

  directive="-s=assemble(d,Append)"
  echo "this should fail: taco '$expression' '$directive'"
  run $TACO "$expression" "$directive"
  echo status=$status
  [ $status -ne 0 ]
  echo "$output" | grep "Unable to find result tensor 'd'"

  directive="-s=assemble(a,Wombat)"
  echo "this should fail: taco '$expression' '$directive'"
  run $TACO "$expression" "$directive"
  echo status=$status
  [ $status -ne 0 ]
  echo "$output" | grep "Assemble strategy not defined"

  directive="-s=assemble(a,Append,wombat)"
  echo "this should fail: taco '$expression' '$directive'"
  run $TACO "$expression" "$directive"
  echo status=$status
  [ $status -ne 0 ]
  echo "$output" | grep "Incorrectly specified whether computation of result"

}

@test 'test -s=parallelize' {
  expression="a(i,j) = b(i,k) * c(k,j)"
  scheduling_directives=(
    "-s=parallelize(i,NotParallel,IgnoreRaces)"
    "-s=parallelize(i,NotParallel,NoRaces)"
    "-s=parallelize(i,NotParallel,Atomics)"
    "-s=parallelize(i,NotParallel,Temporary)"
    "-s=parallelize(i,NotParallel,ParallelReduction)"
    "-s=reorder(i,j,k),split(k,k0,k1,32),parallelize(k0,CPUVector,IgnoreRaces)"
    "-s=reorder(i,j,k),bound(k,k0,32,MaxExact),parallelize(k0,CPUVector,IgnoreRaces)"
    "-s=parallelize(i,CPUThread,IgnoreRaces)"
    "-s=parallelize(i,GPUBlock,IgnoreRaces),parallelize(j,GPUThread,IgnoreRaces)"
  )
  for directive in "${scheduling_directives[@]}"; do
    echo "trying: taco '$expression' '$directive'"
    tacocompile "$expression" "$directive"
  done

  # This should die with "Parallel hardware not defined"
  directive="-s=parallelize(i,Unknown,IgnoreRaces)"
  echo "this should fail: taco '$expression' '$directive'"
  run $TACO "$expression" "$directive"
  echo status=$status
  [ $status -ne 0 ]
  echo "$output" | grep "Parallel hardware not defined"

  # This should die with "Race strategy not defined"
  directive="-s=parallelize(i,NotParallel,Unknown)"
  echo "this should fail: taco '$expression' '$directive'"
  run $TACO "$expression" "$directive"
  echo status=$status
  [ $status -ne 0 ]
  echo "$output" | grep "Race strategy not defined"


  # This should die with "Unable to vectorize or unroll loop over unbound variable"
  directive="-s=reorder(i,j,k),parallelize(k,CPUVector,IgnoreRaces)"

  echo "this should fail: taco '$expression' '$directive'"
  run $TACO "$expression" "$directive"
  echo status=$status
  [ $status -ne 0 ]
  echo "output: $output"
  echo "$output" | grep "unbound variable k"

}

@test 'test -f (tensor layout directives)' {
  expression="a(i,j) = b(i,k) * c(k,j)"
  matrix_layouts=(
    "dd"
    "ds"
    "ss"
    "sd"
    "dd:1,0"
    "ds:1,0"
    "ss:1,0"
    "sd:1,0"
  )
  # TODO: sparse output
  for BFMT in "${matrix_layouts[@]}"; do
    for CFMT in "${matrix_layouts[@]}"; do
      Barg="-f=b:${BFMT}"
      Carg="-f=c:${CFMT}"
      echo "trying: taco '$expression' '$Barg' '$Carg'"
      tacocompile "$expression" "$Barg" "$Carg"
    done # CFMT
  done # BFMT
}

@test 'test -t (tensor element type directives)' {
  expression="a(i,j) = b(i,k) * c(k,j)"
  data_types=(
    "bool"
    "char"  "short"  "int"  "long"  "longlong"
    "uchar" "ushort" "uint" "ulong" "ulonglong"
    "int8"  "int16"  "int32"  "int64"
    "uint8" "uint16" "uint32" "uint64"
    "float" "double" "complexfloat" "complexdouble"
  )
  matrix_layouts=(
    "dd"
    "ss"
  )
  for DTYPE in "${data_types[@]}"; do
    Atype="-t=a:${DTYPE}"
    Btype="-t=b:${DTYPE}"
    Ctype="-t=c:${DTYPE}"
    for LAYOUT in "${matrix_layouts[@]}"; do
      Blayout="-f=b:${LAYOUT}"
      Clayout="-f=c:${LAYOUT}"
      # double = type * type
      echo "trying: taco '$expression' '$Blayout' '$Btype' '$Clayout' '$Ctype'"
      tacocompile "$expression" "$Blayout" "$Btype" "$Clayout" "$Ctype"
      # type = type * type
      echo "trying: taco '$expression' '$Atype' '$Blayout' '$Btype' '$Clayout' '$Ctype'"
      tacocompile "$expression" "$Atype" "$Blayout" "$Btype" "$Clayout" "$Ctype"
    done # LAYOUT
  done # DTYPE
}
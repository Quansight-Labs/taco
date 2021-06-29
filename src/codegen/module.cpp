#include "taco/codegen/module.h"

#include <iostream>
#include <fstream>
#include <dlfcn.h>
#include <unistd.h>
#if USE_OPENMP
#include <omp.h>
#endif

#include "taco/tensor.h"
#include "taco/error.h"
#include "taco/util/strings.h"
#include "taco/util/env.h"
#include "taco/util/print.h"
#include "codegen/codegen_c.h"
#include "codegen/codegen_cuda.h"
#include "taco/cuda.h"
#include "taco/llvm.h"

#ifdef HAVE_LLVM
#include "codegen/codegen_llvm.h"
#endif

using namespace std;

namespace taco {
namespace ir {

std::string Module::chars = "abcdefghijkmnpqrstuvwxyz0123456789";
std::default_random_engine Module::gen = std::default_random_engine();
std::uniform_int_distribution<int> Module::randint =
    std::uniform_int_distribution<int>(0, chars.length() - 1);

void Module::setJITTmpdir() {
  tmpdir = util::getTmpdir();
}

void Module::setJITLibname() {
  libname.resize(12);
  for (int i=0; i<12; i++)
    libname[i] = chars[randint(gen)];
}

void Module::addFunction(Stmt func) {
  funcs.push_back(func);
}

void Module::compileToSource(string path, string prefix) {
  if (!moduleFromUserSource) {
  
    // create a codegen instance and add all the funcs
    bool didGenRuntime = false;

    header.str("");
    header.clear();
    source.str("");
    source.clear();

    if (target.arch == Target::C99 or !should_use_LLVM_codegen()) {
      std::cout << "C99 codegen\n";
      std::shared_ptr<CodeGen> sourcegen =
          CodeGen::init_default(source, CodeGen::ImplementationGen);
      std::shared_ptr<CodeGen> headergen =
              CodeGen::init_default(header, CodeGen::HeaderGen);

      for (auto func: funcs) {
        sourcegen->compile(func, !didGenRuntime);
        headergen->compile(func, !didGenRuntime);
        didGenRuntime = true;
      }
    }
    else {
      std::cout << "LLVM codegen\n";
      // for any other arch we use LLVM
      auto sourcegen = std::static_pointer_cast<CodeGen_LLVM>(
        CodeGen::init_default(source, CodeGen::ImplementationGen));
      auto headergen = CodeGen::init_default(header, CodeGen::HeaderGen);

      for (auto func: funcs) {
        sourcegen->compile(func, !didGenRuntime);
        headergen->compile(func, !didGenRuntime);
        didGenRuntime = true;
      }

      std::string bc_filename = path + prefix + ".bc";
      std::cout << "bc_filename: " << bc_filename << "\n";
      sourcegen->writeModuleToFile(bc_filename);
      sourcegen->dumpModule();
    }
  }

  string file_ending;
  ofstream source_file;

  file_ending = should_use_CUDA_codegen() ? ".cu" : ".c";
  source_file.open(path+prefix+file_ending);
  source_file << source.str();
  source_file.close();

  ofstream header_file;
  std::cout << "header file: " << path + prefix + ".h" << std::endl;
  header_file.open(path+prefix+".h");
  header_file << header.str();
  header_file.close();
}

void Module::compileToStaticLibrary(string path, string prefix) {
  taco_tassert(false) << "Compiling to a static library is not supported";
}

namespace {

void writeShims(vector<Stmt> funcs, string path, string prefix) {
  stringstream shims;
  for (auto func: funcs) {
    if (should_use_CUDA_codegen()) {
      CodeGen_CUDA::generateShim(func, shims);
    }
    else {
      CodeGen_C::generateShim(func, shims);
    }
  }
  
  ofstream shims_file;
  if (should_use_CUDA_codegen()) {
    shims_file.open(path+prefix+"_shims.cpp");
  }
  else {
    shims_file.open(path+prefix+".c", ios::app);
  }
  shims_file << "#include \"" << path << prefix << ".h\"\n";
  shims_file << shims.str();
  shims_file.close();
}

} // anonymous namespace

string Module::compile() {
  string prefix = tmpdir+libname;
  string fullpath = prefix + ".so";

  std::cout << "--==--==--\n";
  std::cout << "should use LLVM codegen: " << should_use_LLVM_codegen() << "\n";
  
  string cc;
  string cflags;
  string file_ending;
  string shims_file;
  if (should_use_CUDA_codegen()) {
    cc = util::getFromEnv("TACO_NVCC", "nvcc");
    cflags = util::getFromEnv("TACO_NVCCFLAGS",
    get_default_CUDA_compiler_flags());
    file_ending = ".cu";
    shims_file = prefix + "_shims.cpp";
  }
  else if (should_use_LLVM_codegen() && !moduleFromUserSource) {
    cc = util::getFromEnv(target.compiler_env, target.compiler);
    cflags = util::getFromEnv("TACO_CFLAGS", "-O3 -ffast-math") + " -shared -fPIC";
    file_ending = ".o";
    shims_file = prefix + ".c";  // little hack to compile C files together
  }
  else {
    cc = util::getFromEnv(target.compiler_env, target.compiler);
    cflags = util::getFromEnv("TACO_CFLAGS",
    "-O3 -ffast-math -std=c99") + " -shared -fPIC";
#if USE_OPENMP
    cflags += " -fopenmp";
#endif
    file_ending = ".c";
    shims_file = "";
  }

  string cmd = cc + " " + cflags + " " +
    prefix + file_ending + " " + shims_file + " " + 
    "-o " + fullpath + " -lm";

  // open the output file & write out the source
  compileToSource(tmpdir, libname);
  
  // write out the shims
  writeShims(funcs, tmpdir, libname);

  // generate object files first
  if (should_use_LLVM_codegen() && !moduleFromUserSource) {
    std::string bc_filename = prefix + ".bc";
    std::string object_filename = prefix + ".o";
    string obj_cmd = "llc --filetype=obj " + bc_filename + " -o " + object_filename;
    std::cout << obj_cmd << "\n";

    int err = system(obj_cmd.data());
    taco_uassert(err == 0) << "Compilation command failed:\n"
                           << obj_cmd
                           << "\nreturned " << err;
  }

  std::cout << cmd << "\n";

  // now compile it
  int err = system(cmd.data());
  taco_uassert(err == 0) << "Compilation command failed:\n" << cmd
    << "\nreturned " << err;

  // use dlsym() to open the compiled library
  if (lib_handle) {
    dlclose(lib_handle);
  }
  lib_handle = dlopen(fullpath.data(), RTLD_NOW | RTLD_LOCAL);
  taco_uassert(lib_handle) << "Failed to load generated code, error is: " << dlerror();

  return fullpath;
}

void Module::setSource(string source) {
  this->source << source;
  moduleFromUserSource = true;
}

string Module::getSource() {
  return source.str();
}

void* Module::getFuncPtr(std::string name) {
  return dlsym(lib_handle, name.data());
}

int Module::callFuncPackedRaw(std::string name, void** args) {
  typedef int (*fnptr_t)(void**);
  static_assert(sizeof(void*) == sizeof(fnptr_t),
    "Unable to cast dlsym() returned void pointer to function pointer");
  void* v_func_ptr = getFuncPtr(name);
  fnptr_t func_ptr;
  *reinterpret_cast<void**>(&func_ptr) = v_func_ptr;

#if USE_OPENMP
  omp_sched_t existingSched;
  ParallelSchedule tacoSched;
  int existingChunkSize, tacoChunkSize;
  int existingNumThreads = omp_get_max_threads();
  omp_get_schedule(&existingSched, &existingChunkSize);
  taco_get_parallel_schedule(&tacoSched, &tacoChunkSize);
  switch (tacoSched) {
    case ParallelSchedule::Static:
      omp_set_schedule(omp_sched_static, tacoChunkSize);
      break;
    case ParallelSchedule::Dynamic:
      omp_set_schedule(omp_sched_dynamic, tacoChunkSize);
      break;
    default:
      break;
  }
  omp_set_num_threads(taco_get_num_threads());
#endif

  int ret = func_ptr(args);

#if USE_OPENMP
  omp_set_schedule(existingSched, existingChunkSize);
  omp_set_num_threads(existingNumThreads);
#endif

  return ret;
}

} // namespace ir
} // namespace taco

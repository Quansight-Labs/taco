/*
   This header file provides the following C++ template functions:

     toString(const T& v) -> std::string

       Convert object v of type T to string. Pretty-printing is
       enabled for objects that types define `toString` or `to_string`
       methods.

     typeName(const T* v) -> std::string

       Return the type name of an object passed in via its pointer
       value.

    and a convenience macro `PRINT(EXPR)` that sends the string
    representation of any expression to stdout.
*/

#pragma once

#ifndef __CUDACC__

#define HAVE_TOSTRING

#ifndef _WIN32
#include <cxxabi.h>
#endif

#include <chrono>
#include <iostream>
#include <set>
#include <sstream>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#ifdef ENABLE_TOSTRING_LLVM
#if __has_include("llvm/Support/raw_os_ostream.h")
#include <llvm/IR/Function.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Value.h>
#include <llvm/Support/raw_os_ostream.h>
#else
#undefine ENABLE_TOSTRING_LLVM
#endif
#endif

#define PRINT(EXPR)                                                            \
  std::cout << __FILE__ << ":" << __func__ << "#" << __LINE__                  \
            << ": " #EXPR "=" << ::toString(EXPR) << std::endl;

template <typename T> std::string typeName(const T *v) {
  std::stringstream stream;
  int status;
#ifdef _WIN32
  stream << std::string(typeid(T).name());
#else
  char *demangled = abi::__cxa_demangle(typeid(T).name(), 0, 0, &status);
  stream << std::string(demangled);
  free(demangled);
#endif
  return stream.str();
}

namespace {

template <typename T, typename = void> struct has_toString : std::false_type {};
template <typename T>
struct has_toString<T, decltype(std::declval<T>().toString(), void())>
    : std::true_type {};
template <class T>
inline constexpr bool has_toString_v = has_toString<T>::value;

template <typename T, typename = void>
struct get_has_toString : std::false_type {};
template <typename T>
struct get_has_toString<T, decltype(std::declval<T>().get()->toString(),
                                    void())> : std::true_type {};
template <class T>
inline constexpr bool get_has_toString_v = get_has_toString<T>::value;

#ifdef ENABLE_TOSTRING_to_string
template <typename T, typename = void>
struct has_to_string : std::false_type {};
template <typename T>
struct has_to_string<T, decltype(std::declval<T>().to_string(), void())>
    : std::true_type {};
template <class T>
inline constexpr bool has_to_string_v = has_to_string<T>::value;
#endif

#ifdef ENABLE_TOSTRING_str
template <typename T, typename = void> struct has_str : std::false_type {};
template <typename T>
struct has_str<T, decltype(std::declval<T>().str(), void())> : std::true_type {
};
template <class T> inline constexpr bool has_str_v = has_str<T>::value;
#endif

} // namespace

template <typename T> std::string toString(const T &v) {
  if constexpr (std::is_same_v<T, std::string>) {
    return "\"" + v + "\"";
#ifdef ENABLE_TOSTRING_LLVM
  } else if constexpr (std::is_same_v<T, llvm::Module>) {
    std::string type_str;
    llvm::raw_string_ostream rso(type_str);
    v.print(rso, nullptr);
    return "\n" + rso.str();
  } else if constexpr (std::is_same_v<T, llvm::Function>) {
    std::string type_str;
    llvm::raw_string_ostream rso(type_str);
    v.print(rso, nullptr);
    return "\n" + rso.str();
  } else if constexpr (std::is_same_v<T, llvm::Value>) {
    std::string type_str;
    llvm::raw_string_ostream rso(type_str);
    v.print(rso);
    return "(" + rso.str() + ")";
  } else if constexpr (std::is_same_v<T, llvm::Type>) {
    std::string type_str;
    llvm::raw_string_ostream rso(type_str);
    v.print(rso);
    return "(" + rso.str() + ")";
  } else if constexpr (std::is_same_v<T, llvm::Triple>) {
    return v.str();
#endif
  } else if constexpr (std::is_same_v<T, bool>) {
    return v ? "True" : "False";
  } else if constexpr (std::is_arithmetic_v<T>) {
    return std::to_string(v);
#ifdef ENABLE_TOSTRING_str
  } else if constexpr (has_str_v<T>) {
    return v.str();
#endif
#ifdef ENABLE_TOSTRING_to_string
  } else if constexpr (has_to_string_v<T>) {
    return v.to_string();
#endif
  } else if constexpr (has_toString_v<T>) {
    return v.toString();
  } else if constexpr (get_has_toString_v<T>) {
    auto ptr = v.get();
    return (ptr == NULL ? "NULL" : "&" + ptr->toString());
  } else if constexpr (std::is_same_v<T, void *>) {
    std::ostringstream ss;
    ss << std::hex << (uintptr_t)v;
    return "0x" + ss.str();
  } else if constexpr (std::is_pointer_v<T>) {
    return (v == NULL ? "NULL" : "&" + toString(*v));
  } else {
    return typeName(&v);
  }
}

template <typename T1, typename T2>
std::string toString(const std::pair<T1, T2> &v) {
  return "(" + toString(v.first) + ", " + toString(v.second) + ")";
}

template <typename T> std::string toString(const std::vector<T> &v) {
  auto result = std::string("[");
  for (size_t i = 0; i < v.size(); ++i) {
    if (i) {
      result += ", ";
    }
    result += toString(v[i]);
  }
  result += "]";
  return result;
}

template <typename T1, typename T2>
std::string toString(const std::unordered_map<T1, T2> &v) {
  auto result = std::string("{");
  size_t i = 0;
  for (const auto &p : v) {
    if (i) {
      result += ", ";
    }
    result += toString(p);
    i++;
  }
  result += "}";
  return result;
}

template <typename T> std::string toString(const std::unordered_set<T> &v) {
  auto result = std::string("{");
  size_t i = 0;
  for (const auto &p : v) {
    if (i) {
      result += ", ";
    }
    result += toString(p);
    i++;
  }
  result += "}";
  return result;
}

template <typename T> std::string toString(const std::set<T> &v) {
  auto result = std::string("{");
  size_t i = 0;
  for (const auto &p : v) {
    if (i) {
      result += ", ";
    }
    result += toString(p);
    i++;
  }
  result += "}";
  return result;
}

template <typename T> std::string toString(const std::tuple<T, T> &v) {
  T left, right;
  std::tie(left, right) = v;
  return std::string("(") + toString(left) + ", " + toString(right) + ")";
}

#endif // __CUDACC__
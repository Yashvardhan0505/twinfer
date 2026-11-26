#include "platform.h"

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <shellapi.h>  // CommandLineToArgvW

#include <fcntl.h>
#include <io.h>

#include <cstdio>
#ifdef _MSC_VER
#pragma comment(lib, "shell32.lib")
#endif
#endif

namespace tf {

#ifdef _WIN32
namespace {

std::string to_utf8(const wchar_t* w) {
  const int n = WideCharToMultiByte(CP_UTF8, 0, w, -1, nullptr, 0, nullptr, nullptr);
  if (n <= 1) return std::string();
  std::string s(static_cast<size_t>(n - 1), '\0');
  WideCharToMultiByte(CP_UTF8, 0, w, -1, &s[0], n, nullptr, nullptr);
  return s;
}

}  // namespace
#endif

std::vector<std::string> utf8_args(int argc, char** argv) {
#ifdef _WIN32
  int n = 0;
  LPWSTR* w = CommandLineToArgvW(GetCommandLineW(), &n);
  if (w) {
    std::vector<std::string> out;
    out.reserve(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) out.push_back(to_utf8(w[i]));
    LocalFree(w);
    if (n == argc) return out;  // otherwise the two parsers disagreed; fall back to the CRT's argv
  }
#endif
  return std::vector<std::string>(argv, argv + argc);
}

void setup_console() {
#ifdef _WIN32
  SetConsoleOutputCP(CP_UTF8);
  SetConsoleCP(CP_UTF8);
  _setmode(_fileno(stdout), _O_BINARY);
#endif
}

}  // namespace tf

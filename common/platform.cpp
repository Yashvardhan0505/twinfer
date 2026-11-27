#include "platform.h"

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

#include <fcntl.h>
#include <io.h>

#include <cstdio>
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

std::vector<std::string> split_command_line(const std::string& line) {
  const size_t n = line.size();
  auto blank = [&](size_t i) { return line[i] == ' ' || line[i] == '\t'; };
  std::vector<std::string> args(1);
  size_t i = 0;
  // argv[0], the program name: quotes group (and are dropped), nothing is escaped.
  for (bool quoted = false; i < n && (quoted || !blank(i)); ++i) {
    if (line[i] == '"') quoted = !quoted;
    else args[0] += line[i];
  }
  for (;;) {
    while (i < n && blank(i)) ++i;
    if (i == n) break;
    std::string a;
    bool quoted = false;
    while (i < n && (quoted || !blank(i))) {
      size_t slashes = 0;
      while (i < n && line[i] == '\\') {
        ++slashes;
        ++i;
      }
      if (i < n && line[i] == '"') {  // k backslashes then '"': k/2 backslashes, and the '"' is...
        a.append(slashes / 2, '\\');
        if (slashes % 2) {
          a += '"';  // ...a literal quote if k is odd
        } else if (quoted && i + 1 < n && line[i + 1] == '"') {
          a += '"';  // ...also literal if it's "" inside quotes
          ++i;
        } else {
          quoted = !quoted;  // ...otherwise it opens or closes a quoted part
        }
        ++i;
      } else {  // backslashes not before a quote are literal
        a.append(slashes, '\\');
        if (i < n && (quoted || !blank(i))) a += line[i++];
      }
    }
    args.push_back(a);
  }
  return args;
}

std::vector<std::string> utf8_args(int argc, char** argv) {
#ifdef _WIN32
  // Not CommandLineToArgvW: it also splits on '\n', '\r' and other control characters, which the C
  // runtime (and so every other program's argv) doesn't.
  const wchar_t* line = GetCommandLineW();
  if (line && *line) return split_command_line(to_utf8(line));
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

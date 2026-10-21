#include "server/server_internal.h"
#include "nlohmann/json.hpp"
#include <algorithm>
#include <charconv>
#include <fstream>
#include <mutex>
#ifndef _WIN32
#include <cctype>
#include <cstdio>
#include <sstream>
#include <string>
#endif
#ifdef _WIN32
#include <windows.h>
#include <psapi.h>
#endif

namespace mx {
using json = nlohmann::json;
namespace {
#ifdef _WIN32
struct Nvml {
  using Device = void*;
  struct Utilization { unsigned gpu, memory; };
  HMODULE dll = LoadLibraryExW(L"nvml.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);
  int (*init)() = nullptr;
  int (*handle)(const char*, Device*) = nullptr;
  int (*utilization)(Device, Utilization*) = nullptr;
  int (*temperature)(Device, unsigned, unsigned*) = nullptr;
  int (*power)(Device, unsigned*) = nullptr;
  bool ready = false;
  Nvml() {
    if (!dll) return;
    init = reinterpret_cast<decltype(init)>(GetProcAddress(dll, "nvmlInit_v2"));
    handle = reinterpret_cast<decltype(handle)>(GetProcAddress(dll, "nvmlDeviceGetHandleByPciBusId_v2"));
    utilization = reinterpret_cast<decltype(utilization)>(GetProcAddress(dll, "nvmlDeviceGetUtilizationRates"));
    temperature = reinterpret_cast<decltype(temperature)>(GetProcAddress(dll, "nvmlDeviceGetTemperature"));
    power = reinterpret_cast<decltype(power)>(GetProcAddress(dll, "nvmlDeviceGetPowerUsage"));
    ready = init && handle && init() == 0;
  }
  ~Nvml() {
    if (ready) {
      auto shutdown = reinterpret_cast<int(*)()>(GetProcAddress(dll, "nvmlShutdown"));
      if (shutdown) shutdown();
    }
    if (dll) FreeLibrary(dll);
  }
};
uint64_t ticks(FILETIME f) { return (uint64_t(f.dwHighDateTime) << 32) | f.dwLowDateTime; }
#else
bool read_cpu(uint64_t& total, uint64_t& idle) {
  std::ifstream f("/proc/stat");
  std::string label;
  if (!(f >> label) || label != "cpu") return false;
  uint64_t v[8]{};
  for (auto& x : v) if (!(f >> x)) return false;
  total = 0;
  for (auto x : v) total += x;
  idle = v[3] + v[4];
  return true;
}

bool read_process(uint64_t& cpu, uint64_t& bytes) {
  std::ifstream f("/proc/self/stat");
  std::string line;
  if (!std::getline(f, line)) return false;
  const auto end = line.rfind(')');
  if (end == std::string::npos) return false;
  std::istringstream fields(line.substr(end + 2));  // field 3 starts after the command name
  std::string value;
  uint64_t utime = 0, stime = 0, rss = 0;
  for (int i = 0; i <= 21; ++i) {
    if (!(fields >> value)) return false;
    if (i == 11) { if (std::from_chars(value.data(), value.data() + value.size(), utime).ec != std::errc{}) return false; }
    if (i == 12) { if (std::from_chars(value.data(), value.data() + value.size(), stime).ec != std::errc{}) return false; }
    if (i == 21) { if (std::from_chars(value.data(), value.data() + value.size(), rss).ec != std::errc{}) return false; }
  }
  cpu = utime + stime;
  bytes = rss * uint64_t(sysconf(_SC_PAGESIZE));
  return true;
}

std::vector<std::array<std::string, 4>> nvidia_metrics() {
  std::vector<std::array<std::string, 4>> rows;
  FILE* f = popen("nvidia-smi --query-gpu=pci.bus_id,utilization.gpu,temperature.gpu,power.draw --format=csv,noheader,nounits 2>/dev/null", "r");
  if (!f) return rows;
  char line[512];
  while (std::fgets(line, sizeof(line), f)) {
    std::istringstream s(line);
    std::array<std::string, 4> row;
    if (std::getline(s, row[0], ',') && std::getline(s, row[1], ',') &&
        std::getline(s, row[2], ',') && std::getline(s, row[3])) {
      for (auto& cell : row) {
        const auto first = cell.find_first_not_of(" \t\r\n");
        const auto last = cell.find_last_not_of(" \t\r\n");
        cell = first == std::string::npos ? "" : cell.substr(first, last - first + 1);
      }
      rows.push_back(std::move(row));
    }
  }
  pclose(f);
  return rows;
}

std::string normalize_bus(std::string bus) {
  const auto colon = bus.find(':');
  if (colon != std::string::npos) {
    const auto first = bus.find_first_not_of('0');
    bus = (first == std::string::npos || first > colon ? "0" : bus.substr(first, colon - first)) + bus.substr(colon);
  }
  std::transform(bus.begin(), bus.end(), bus.begin(), [](unsigned char c) { return char(std::tolower(c)); });
  return bus;
}
#endif
}  // namespace

std::string system_json(const std::vector<GpuInfo>& gpus, int queue_length, uint64_t requests,
                        uint64_t errors, double uptime, bool authenticated) {
  json host = {{"cpu_percent", nullptr}, {"ram_total", nullptr}, {"ram_used", nullptr}},
       process = {{"ram_used", nullptr}, {"cpu_percent", nullptr}}, arr = json::array();
#ifdef _WIN32
  // Windows 計數器與 NVML 都不啟動外部程式；首筆 CPU 還沒有採樣間隔，回 null。
  static std::mutex mu;
  std::lock_guard lk(mu);
  static Nvml nvml;
  static uint64_t prev_idle = 0, prev_total = 0, prev_process = 0;
  static json cpu = nullptr, process_cpu = nullptr;
  FILETIME idle{}, kernel{}, user{}, creation{}, exit{}, pk{}, pu{};
  if (GetSystemTimes(&idle, &kernel, &user)) {
    uint64_t total = ticks(kernel) + ticks(user), dt = total - prev_total;
    bool got_process = GetProcessTimes(GetCurrentProcess(), &creation, &exit, &pk, &pu) != 0;
    uint64_t pt = ticks(pk) + ticks(pu);
    if (prev_total && dt >= 1000000) {
      cpu = std::clamp(100.0 * (1.0 - double(ticks(idle) - prev_idle) / dt), 0.0, 100.0);
      if (got_process) process_cpu = std::clamp(100.0 * double(pt - prev_process) / dt, 0.0, 100.0);
    }
    if (!prev_total || dt >= 1000000) { prev_total = total; prev_idle = ticks(idle); prev_process = pt; }
    host["cpu_percent"] = cpu;
    process["cpu_percent"] = process_cpu;
  }
  MEMORYSTATUSEX mem{};
  mem.dwLength = sizeof(mem);
  if (GlobalMemoryStatusEx(&mem)) { host["ram_total"] = mem.ullTotalPhys; host["ram_used"] = mem.ullTotalPhys - mem.ullAvailPhys; }
  PROCESS_MEMORY_COUNTERS pm{};
  if (K32GetProcessMemoryInfo(GetCurrentProcess(), &pm, sizeof(pm))) process["ram_used"] = pm.WorkingSetSize;
#else
  static std::mutex mu;
  std::lock_guard lk(mu);
  static uint64_t prev_total = 0, prev_idle = 0, prev_process = 0;
  static json cpu = nullptr, process_cpu = nullptr;
  uint64_t total = 0, idle = 0, proc_ticks = 0, proc_bytes = 0;
  if (read_cpu(total, idle) && read_process(proc_ticks, proc_bytes)) {
    const uint64_t dt = total - prev_total;
    if (prev_total && dt) {
      cpu = std::clamp(100.0 * (1.0 - double(idle - prev_idle) / dt), 0.0, 100.0);
      process_cpu = std::clamp(100.0 * double(proc_ticks - prev_process) / dt, 0.0, 100.0);
    }
    prev_total = total;
    prev_idle = idle;
    prev_process = proc_ticks;
    host["cpu_percent"] = cpu;
    process["cpu_percent"] = process_cpu;
    process["ram_used"] = proc_bytes;
  }
  std::ifstream meminfo("/proc/meminfo");
  std::string key, unit;
  uint64_t total_kb = 0, available_kb = 0;
  while (meminfo >> key) {
    uint64_t value = 0;
    if (!(meminfo >> value >> unit)) break;
    if (key == "MemTotal:") total_kb = value;
    if (key == "MemAvailable:") available_kb = value;
  }
  if (total_kb) {
    host["ram_total"] = total_kb * 1024;
    host["ram_used"] = (total_kb - std::min(total_kb, available_kb)) * 1024;
  }
  const auto metrics = nvidia_metrics();
#endif
  for (const auto& g : gpus) {
    json gpu = {{"index", g.index}, {"name", g.name}, {"total", g.total}, {"free", g.free},
                {"utilization_percent", nullptr}, {"temperature_c", nullptr}, {"power_w", nullptr}};
#ifdef _WIN32
    Nvml::Device device{};
    if (nvml.ready && !g.pci_bus.empty() && nvml.handle(g.pci_bus.c_str(), &device) == 0) {
      Nvml::Utilization u{};
      unsigned value = 0;
      if (nvml.utilization && nvml.utilization(device, &u) == 0) gpu["utilization_percent"] = u.gpu;
      if (nvml.temperature && nvml.temperature(device, 0, &value) == 0) gpu["temperature_c"] = value;
      if (nvml.power && nvml.power(device, &value) == 0) gpu["power_w"] = value / 1000.0;
    }
#else
    for (const auto& row : metrics) if (normalize_bus(row[0]) == normalize_bus(g.pci_bus)) {
      try {
        if (row[1] != "[N/A]" && !row[1].empty()) gpu["utilization_percent"] = std::stoi(row[1]);
        if (row[2] != "[N/A]" && !row[2].empty()) gpu["temperature_c"] = std::stoi(row[2]);
        if (row[3] != "[N/A]" && !row[3].empty()) gpu["power_w"] = std::stod(row[3]);
      } catch (...) {}  // Telemetry is optional; leave unreadable values null.
      break;
    }
#endif
    arr.push_back(std::move(gpu));
  }
  return json{{"gpus", arr}, {"queue_length", queue_length}, {"host", host}, {"process", process},
              {"api", {{"requests", requests}, {"errors", errors}, {"uptime_seconds", uptime},
                       {"authentication", authenticated}}}}.dump();
}
}  // namespace mx

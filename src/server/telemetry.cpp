#include "server/server_internal.h"
#include "nlohmann/json.hpp"
#include <algorithm>
#include <mutex>
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
#endif
    arr.push_back(std::move(gpu));
  }
  return json{{"gpus", arr}, {"queue_length", queue_length}, {"host", host}, {"process", process},
              {"api", {{"requests", requests}, {"errors", errors}, {"uptime_seconds", uptime},
                       {"authentication", authenticated}}}}.dump();
}
}  // namespace mx

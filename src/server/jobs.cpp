#include "server/server_internal.h"

#include <algorithm>
#include <chrono>
#include <utility>

#include "server/image_io.h"

namespace mx {
namespace {

int64_t unix_now() {
  return std::chrono::duration_cast<std::chrono::seconds>(
             std::chrono::system_clock::now().time_since_epoch())
      .count();
}

std::string hex8(uint64_t n) {
  static const char* kHex = "0123456789abcdef";
  std::string s(8, '0');
  for (int i = 7; i >= 0; --i) {
    s[static_cast<size_t>(i)] = kHex[n & 0xf];
    n >>= 4;
  }
  return s;
}

void mark_cancelled(Job& job) {
  job.status = "cancelled";
  job.err_message = "已取消";
  job.err_type = "cancelled";
  job.err_code = "cancelled";
  job.seq++;
}

struct Outcome {
  std::string url;
  int64_t seed = 0;
  double seconds = 0;
  std::vector<uint8_t> png;
};

Outcome produce_image(ServerState& st, Job& job, const ProgressFn& cb) {
  auto result = st.engine.generate_image(job.image, cb, job.cancel.get());
  auto name = job.id + ".png";
  write_file(utf8_path(st.opt.out_dir) / name, result.png.data(), result.png.size());
  Outcome out;
  out.url = "/files/" + name;
  out.seed = result.seed;
  out.seconds = result.seconds;
  if (job.want_b64) out.png = std::move(result.png);
  return out;
}

void remove_temp_source(const std::filesystem::path& src, const std::filesystem::path& dst) {
  std::error_code ec;
  if (std::filesystem::equivalent(src, dst, ec)) return;
  auto tmp = std::filesystem::weakly_canonical(std::filesystem::temp_directory_path(), ec);
  auto canon = std::filesystem::weakly_canonical(src, ec);
  if (ec) return;
  auto rel = canon.lexically_relative(tmp);
  if (rel.empty() || *rel.begin() == "..") return;
  std::filesystem::remove(src, ec);
}

Outcome produce_video(ServerState& st, Job& job, const ProgressFn& cb) {
  auto result = st.engine.generate_video(job.video, cb, job.cancel.get());
  if (result.mp4_path.empty()) throw std::runtime_error("引擎沒有輸出影片");
  std::filesystem::path src(result.mp4_path);
  auto name = job.id + ".mp4";
  auto dst = utf8_path(st.opt.out_dir) / name;
  std::filesystem::copy_file(src, dst, std::filesystem::copy_options::overwrite_existing);
  remove_temp_source(src, dst);
  return Outcome{"/files/" + name, result.seed, result.seconds, {}};
}

void clear_temps(Job& job) {
  for (auto& p : job.temps) {
    std::error_code ec;
    std::filesystem::remove(p, ec);
  }
  job.temps.clear();
}

}  // namespace

ServerState::ServerState(Engine& engine_in, ServerOptions opt_in)
    : engine(engine_in), opt(std::move(opt_in)) {
  std::filesystem::create_directories(utf8_path(opt.out_dir));
  std::filesystem::create_directories(utf8_path(opt.out_dir) / "tmp");
  worker = std::thread([this] { worker_loop(); });
}

ServerState::~ServerState() { shutdown(); }

std::string ServerState::accept(const std::shared_ptr<Job>& job) {
  std::lock_guard lk(mu);
  job->id = "job_" + hex8(next_n++);
  job->status = "queued";
  job->created = unix_now();
  job->seq = 1;
  by_id[job->id] = job;
  all.push_back(job);
  queue.push_back(job);
  std::string body = job_to_json(*job);
  work_cv.notify_one();
  return body;
}

std::string ServerState::list_json() {
  std::lock_guard lk(mu);
  std::vector<std::shared_ptr<Job>> recent;
  for (auto it = all.rbegin(); it != all.rend() && recent.size() < 50; ++it) recent.push_back(*it);
  return jobs_to_json(recent);
}

std::string ServerState::job_json_by_id(const std::string& id, bool& found) {
  std::lock_guard lk(mu);
  auto it = by_id.find(id);
  if (it == by_id.end()) {
    found = false;
    return {};
  }
  found = true;
  return job_to_json(*it->second);
}

std::shared_ptr<Job> ServerState::find(const std::string& id) {
  std::lock_guard lk(mu);
  auto it = by_id.find(id);
  return it == by_id.end() ? nullptr : it->second;
}

bool ServerState::cancel(const std::string& id) {
  std::lock_guard lk(mu);
  auto it = by_id.find(id);
  if (it == by_id.end()) return false;
  auto job = it->second;
  job->cancel->store(true);
  if (job->status == "queued") {
    queue.erase(std::remove(queue.begin(), queue.end(), job), queue.end());
    mark_cancelled(*job);
    job_cv.notify_all();
  }
  return true;
}

int ServerState::queue_length() {
  std::lock_guard lk(mu);
  int n = 0;
  for (auto& job : all)
    if (job->status == "queued" || job->status == "running") ++n;
  return n;
}

void ServerState::prune_locked() {
  int finished = 0;
  for (auto& job : all)
    if (job_terminal(job->status)) ++finished;
  while (finished > 200) {
    auto it = std::find_if(all.begin(), all.end(),
                           [](const std::shared_ptr<Job>& job) { return job_terminal(job->status); });
    if (it == all.end()) break;
    by_id.erase((*it)->id);
    all.erase(it);
    --finished;
  }
}

void ServerState::execute(const std::shared_ptr<Job>& job) {
  auto cb = [this, job](const Progress& p) {
    std::lock_guard lk(mu);
    job->stage = p.stage;
    job->step = p.step;
    job->total = p.total;
    job->seq++;
    job_cv.notify_all();
  };
  try {
    Outcome out = job->type == "video" ? produce_video(*this, *job, cb) : produce_image(*this, *job, cb);
    std::lock_guard lk(mu);
    job->url = std::move(out.url);
    job->seed = out.seed;
    job->seconds = out.seconds;
    job->png = std::move(out.png);
    job->has_result = true;
    job->status = "succeeded";
    job->seq++;
    job_cv.notify_all();
    prune_locked();
  } catch (const std::exception& e) {
    std::lock_guard lk(mu);
    if (job->cancel->load() || stopping.load())
      mark_cancelled(*job);
    else {
      job->status = "failed";
      job->err_message = e.what();
      job->err_type = "server_error";
      job->err_code = "engine_error";
      job->seq++;
    }
    job_cv.notify_all();
    prune_locked();
  }
  clear_temps(*job);
}

void ServerState::worker_loop() {
  while (true) {
    std::shared_ptr<Job> job;
    {
      std::unique_lock lk(mu);
      work_cv.wait(lk, [&] { return stopping.load() || !queue.empty(); });
      if (queue.empty()) return;
      job = queue.front();
      queue.pop_front();
      if (job->cancel->load()) {
        mark_cancelled(*job);
        job_cv.notify_all();
        continue;
      }
      job->status = "running";
      job->seq++;
      job_cv.notify_all();
    }
    execute(job);
  }
}

void ServerState::shutdown() {
  {
    std::lock_guard lk(mu);
    stopping = true;
    for (auto& job : queue) mark_cancelled(*job);
    queue.clear();
    for (auto& job : all)
      if (job->status == "running") job->cancel->store(true);
  }
  work_cv.notify_all();
  job_cv.notify_all();
  if (worker.joinable() && worker.get_id() != std::this_thread::get_id()) worker.join();
}

}  // namespace mx

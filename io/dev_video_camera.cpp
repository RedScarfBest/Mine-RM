#include "dev_video_camera.hpp"

#include <stdexcept>
#include <thread>

#include "tools/logger.hpp"
#include "tools/yaml.hpp"

namespace io
{
DevVideoCamera::DevVideoCamera(const std::string & config_path)
{
  auto yaml = tools::load(config_path);
  auto video = yaml["video"];

  path_ = tools::read<std::string>(video, "path");
  loop_ = tools::read_or<bool>(video, "loop", true);
  auto fps = tools::read_or<double>(video, "fps", 0.0);

  cap_.open(path_);
  if (!cap_.isOpened()) {
    tools::logger()->error("[DevVideoCamera] Failed to open video: {}", path_);
    throw std::runtime_error("[DevVideoCamera] Failed to open video: " + path_);
  }

  if (fps <= 1.0) fps = cap_.get(cv::CAP_PROP_FPS);
  if (fps <= 1.0) fps = 30.0;

  period_ = std::chrono::duration_cast<std::chrono::steady_clock::duration>(
    std::chrono::duration<double>(1.0 / fps));
  next_time_ = std::chrono::steady_clock::now();
  ended_ = false;

  tools::logger()->info(
    "[DevVideoCamera] {} ({}x{}, {} frames, {:.2f} fps, loop={})", path_,
    static_cast<int>(cap_.get(cv::CAP_PROP_FRAME_WIDTH)),
    static_cast<int>(cap_.get(cv::CAP_PROP_FRAME_HEIGHT)),
    static_cast<int>(cap_.get(cv::CAP_PROP_FRAME_COUNT)), fps, loop_);
}

void DevVideoCamera::read(cv::Mat & img, std::chrono::steady_clock::time_point & timestamp)
{
  if (grab(frame_)) img = frame_;
  timestamp = std::chrono::steady_clock::now();
}

bool DevVideoCamera::try_read_for(
  cv::Mat & img, std::chrono::steady_clock::time_point & timestamp,
  std::chrono::milliseconds timeout)
{
  if (next_time_ > std::chrono::steady_clock::now() + timeout) return false;

  auto ok = grab(frame_);
  if (ok) img = frame_;
  timestamp = std::chrono::steady_clock::now();
  return ok;
}

bool DevVideoCamera::is_alive() const { return cap_.isOpened() && !ended_; }

bool DevVideoCamera::grab(cv::Mat & frame)
{
  std::this_thread::sleep_until(next_time_);

  auto now = std::chrono::steady_clock::now();
  next_time_ += period_;
  if (next_time_ < now) next_time_ = now + period_;

  cap_ >> frame;

  if (frame.empty() && loop_) {
    cap_.set(cv::CAP_PROP_POS_FRAMES, 0);
    cap_ >> frame;
  }

  if (frame.empty()) {
    if (!ended_) tools::logger()->warn("[DevVideoCamera] End of video: {}", path_);
    ended_ = true;
    return false;
  }

  return true;
}

}  // namespace io

#ifndef IO__DEV_VIDEO_CAMERA_HPP
#define IO__DEV_VIDEO_CAMERA_HPP

#include <chrono>
#include <opencv2/opencv.hpp>
#include <string>

#include "camera.hpp"

namespace io
{
class DevVideoCamera : public CameraBase
{
public:
  explicit DevVideoCamera(const std::string & config_path);

  void read(cv::Mat & img, std::chrono::steady_clock::time_point & timestamp) override;

  bool try_read_for(
    cv::Mat & img, std::chrono::steady_clock::time_point & timestamp,
    std::chrono::milliseconds timeout) override;

  bool is_alive() const override;

private:
  bool grab(cv::Mat & frame);

  cv::VideoCapture cap_;
  cv::Mat frame_;
  std::string path_;
  bool loop_;
  bool ended_;
  std::chrono::steady_clock::duration period_;
  std::chrono::steady_clock::time_point next_time_;
};

}  // namespace io

#endif  // IO__DEV_VIDEO_CAMERA_HPP

#include <chrono>
#include <opencv2/opencv.hpp>

#include "io/camera.hpp"
#include "io/gimbal/gimbal.hpp"
#include "tasks/auto_aim/solver.hpp"
#include "tasks/auto_aim/yolo.hpp"
#include "tools/exiter.hpp"
#include "tools/img_tools.hpp"
#include "tools/logger.hpp"
#include "tools/math_tools.hpp"
#include "tools/plotter.hpp"

const std::string keys =
  "{help h usage ? | | 输出命令行参数说明}"
  "{@config-path   | | yaml配置文件路径 }";

using namespace std::chrono_literals;

int main(int argc, char * argv[])
{
  cv::CommandLineParser cli(argc, argv, keys);
  auto config_path = cli.get<std::string>("@config-path");
  if (cli.has("help") || !cli.has("@config-path")) {
    cli.printMessage();
    return 0;
  }

  // 初始化工具类
  tools::Exiter exiter;
  tools::Plotter plotter;  // 注意plotter工具的使用

  // 初始化io类
  io::Camera camera(config_path);
  io::Gimbal gimbal(config_path);

  // 初始化auto_aim类
  auto_aim::YOLO yolo(config_path, true);
  auto_aim::Solver solver(config_path);

  cv::Mat img;
  Eigen::Quaterniond q;
  std::chrono::steady_clock::time_point t;

  while (!exiter.exit()) {
    // Your code start
    static int frame_count = 0;

    camera.read(img, t);

    if (!img.empty()) {
      auto armors = yolo.detect(img, frame_count);

      if (frame_count % 30 == 0) {
        tools::logger()->info("frame_count: {}, armors.size(): {}", frame_count, armors.size());

        for (const auto & armor : armors) {
          tools::logger()->info(
            "{} {} {} conf: {:.3f} center: ({:.0f}, {:.0f}) ratio: {:.2f}",
            auto_aim::COLORS[armor.color], auto_aim::ARMOR_TYPES[armor.type],
            auto_aim::ARMOR_NAMES[armor.name], armor.confidence, armor.center.x, armor.center.y,
            armor.ratio);
        }
      }
    }

    int key = cv::waitKey(1);
    if (key == 27 || key == 'q') break;
    frame_count++;
    // Your code end
  }

  return 0;
}
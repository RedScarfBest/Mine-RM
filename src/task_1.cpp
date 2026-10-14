#include <algorithm>  //add
#include <chrono>
#include <cmath>  //add
#include <opencv2/opencv.hpp>
#include <optional>   //add
#include <stdexcept>  //add

#include "io/camera.hpp"
#include "io/gimbal/gimbal.hpp"
#include "tasks/auto_aim/solver.hpp"
#include "tasks/auto_aim/yolo.hpp"
#include "tools/exiter.hpp"
#include "tools/img_tools.hpp"
#include "tools/logger.hpp"
#include "tools/math_tools.hpp"
#include "tools/plotter.hpp"
#include "tools/yaml.hpp"  //add

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
    struct Task1State
    {
      int frame = 0;
      bool locked = false;
      bool commanding = false;
      auto_aim::ArmorName name = auto_aim::not_armor;
      cv::Point2f center{};
      double aim_yaw = 0;
      double aim_pitch = 0;
      float command_yaw = 0;
      float command_pitch = 0;
      std::chrono::steady_clock::time_point started = std::chrono::steady_clock::now();
      std::chrono::steady_clock::time_point seen{};
      std::chrono::steady_clock::time_point output = started;
    };
    struct Task1Config
    {
      auto_aim::Color enemy;
      double yaw_offset;
      double pitch_offset;
    };
    struct Candidate
    {
      auto_aim::ArmorName name;
      cv::Point2f center;
      double confidence;
      Eigen::Vector3d ypd;
    };
    static Task1State state;
    static const Task1Config config = [&] {
      const auto yaml = tools::load(config_path);
      const auto enemy = tools::read<std::string>(yaml, "enemy_color");
      const double yaw_offset = tools::read<double>(yaml, "yaw_offset") * CV_PI / 180.0;
      const double pitch_offset = tools::read<double>(yaml, "pitch_offset") * CV_PI / 180.0;
      if (
        (enemy != "red" && enemy != "blue") || !std::isfinite(yaw_offset) ||
        !std::isfinite(pitch_offset) || std::abs(yaw_offset) > CV_PI ||
        std::abs(pitch_offset) > CV_PI) {
        throw std::runtime_error("Invalid Task 1 color or angular offsets");
      }
      return Task1Config{enemy == "red" ? auto_aim::red : auto_aim::blue, yaw_offset, pitch_offset};
    }();
    constexpr double hold_seconds = 0.150;
    constexpr double pitch_limit = 20.0 * CV_PI / 180.0;
    const bool print_log = state.frame % 30 == 0;
    auto clear_target = [&] {
      state.locked = false;
      state.commanding = false;
    };
    auto points_finite = [](const std::vector<cv::Point2f> & points) {
      return std::all_of(points.begin(), points.end(), [](const cv::Point2f & point) {
        return std::isfinite(point.x) && std::isfinite(point.y);
      });
    };

    camera.read(img, t);
    auto mode = gimbal.mode();
    const auto before_detection = std::chrono::steady_clock::now();
    if (
      mode != io::GimbalMode::AUTO_AIM ||
      (state.locked && tools::delta_time(before_detection, state.seen) > hold_seconds)) {
      clear_target();
    }
    bool invalid_frame = false;
    bool fresh_target = false;
    std::optional<Candidate> selected;
    double best_score = 0;
    if (!img.empty()) {
      const double age = tools::delta_time(before_detection, t);
      q = gimbal.q(t);
      const double norm = q.norm();
      if (
        age < 0 || age > hold_seconds || !q.coeffs().allFinite() || !std::isfinite(norm) ||
        norm <= 1e-6) {
        invalid_frame = true;
        if (print_log) tools::logger()->warn("invalid_quaternion_or_frame_time");
      } else {
        q.normalize();
        solver.set_R_gimbal2world(q);
        auto armors = yolo.detect(img, state.frame);
        cv::Mat img_show = img.clone();
        if (print_log) {
          tools::logger()->info("frame_count: {}, armors.size(): {}", state.frame, armors.size());
        }
        for (auto & armor : armors) {
          if (armor.color != config.enemy) continue;
          if (
            armor.points.size() != 4 || !points_finite(armor.points) ||
            !std::isfinite(armor.center.x) || !std::isfinite(armor.center.y) ||
            !std::isfinite(armor.confidence) || !std::isfinite(armor.ratio) ||
            std::abs(cv::contourArea(armor.points)) <= 1.0) {
            invalid_frame = true;
            if (print_log) tools::logger()->warn("invalid_points");
            continue;
          }
          try {
            solver.solve(armor);
          } catch (const cv::Exception & error) {
            invalid_frame = true;
            if (print_log) tools::logger()->warn("solve_failed: {}", error.what());
            continue;
          }
          if (
            !armor.xyz_in_gimbal.allFinite() || !armor.xyz_in_world.allFinite() ||
            !armor.ypr_in_gimbal.allFinite() || !armor.ypr_in_world.allFinite() ||
            !armor.ypd_in_world.allFinite() || armor.ypd_in_world[2] <= 0) {
            invalid_frame = true;
            if (print_log) tools::logger()->warn("invalid_pose");
            continue;
          }
          bool reprojection_valid = false;
          double reprojection_error = 0;
          tools::draw_points(img_show, armor.points, cv::Scalar(0, 255, 0), 3);
          try {
            const auto projected = solver.reproject_armor(
              armor.xyz_in_world, armor.ypr_in_world[0], armor.type, armor.name);
            if (projected.size() == 4 && points_finite(projected)) {
              double sum = 0;
              for (int i = 0; i < 4; ++i) sum += cv::norm(projected[i] - armor.points[i]);
              if (std::isfinite(sum)) {
                reprojection_error = sum / 4.0;
                reprojection_valid = true;
                tools::draw_points(img_show, projected, cv::Scalar(0, 0, 255), 3);
              }
            }
          } catch (const cv::Exception & error) {
            if (print_log) tools::logger()->warn("projection_unavailable: {}", error.what());
          }
          if (print_log) {
            tools::logger()->info(
              "{} {} {} conf={:.3f} ratio={:.2f} distance_m={:.3f} "
              "aim_yaw_rad={:.3f} aim_pitch_rad={:.3f} armor_yaw_rad={:.3f} "
              "armor_pitch_rad={:.3f} reproj_valid={} reproj_prior_px={:.3f}",
              auto_aim::COLORS[armor.color], auto_aim::ARMOR_NAMES[armor.name],
              auto_aim::ARMOR_TYPES[armor.type], armor.confidence, armor.ratio,
              armor.ypd_in_world[2], armor.ypd_in_world[0], armor.ypd_in_world[1],
              armor.ypr_in_world[0], armor.ypr_in_world[1], static_cast<int>(reprojection_valid),
              reprojection_error);
          }
          if (state.locked && armor.name != state.name) continue;
          const cv::Point2f reference =
            state.locked ? state.center : cv::Point2f(img.cols * 0.5f, img.rows * 0.5f);
          const double score = cv::norm(armor.center - reference);
          if (
            !selected || score < best_score ||
            (score == best_score && armor.confidence > selected->confidence)) {
            selected = Candidate{armor.name, armor.center, armor.confidence, armor.ypd_in_world};
            best_score = score;
          }
        }
        cv::resize(img_show, img_show, cv::Size(), 0.5, 0.5);
        cv::imshow("reprojected_points (pose prior)", img_show);
      }
    }

    const auto measured = gimbal.state();
    const bool feedback_valid = std::isfinite(measured.yaw) && std::isfinite(measured.pitch);
    const int key = cv::waitKey(1);
    const bool quitting = key == 27 || key == 'q' || exiter.exit();
    const auto now = std::chrono::steady_clock::now();
    mode = gimbal.mode();
    const double frame_age = tools::delta_time(now, t);
    if (!img.empty() && (frame_age < 0 || frame_age > hold_seconds)) invalid_frame = true;
    const bool permitted =
      mode == io::GimbalMode::AUTO_AIM && feedback_valid && !invalid_frame && !quitting;
    if (!permitted) {
      clear_target();
    } else if (selected) {
      state.locked = true;
      state.name = selected->name;
      state.center = selected->center;
      state.seen = t;
      state.aim_yaw = selected->ypd[0];
      state.aim_pitch = selected->ypd[1];
      const double desired_yaw =
        measured.yaw + tools::limit_rad(state.aim_yaw + config.yaw_offset - measured.yaw);
      const double desired_pitch =
        tools::limit_min_max(state.aim_pitch + config.pitch_offset, -pitch_limit, pitch_limit);
      const double dt = tools::limit_min_max(tools::delta_time(now, state.output), 0.0, 0.1);
      const double base_yaw = state.commanding ? state.command_yaw : measured.yaw;
      const double base_pitch = state.commanding ? state.command_pitch : measured.pitch;
      state.command_yaw = static_cast<float>(
        base_yaw +
        tools::limit_min_max(tools::limit_rad(desired_yaw - base_yaw), -3.0 * dt, 3.0 * dt));
      state.command_pitch = static_cast<float>(tools::limit_min_max(
        base_pitch + tools::limit_min_max(desired_pitch - base_pitch, -dt, dt), -pitch_limit,
        pitch_limit));
      state.commanding = true;
      fresh_target = true;
    } else if (!state.locked || tools::delta_time(now, state.seen) > hold_seconds) {
      clear_target();
    }
    const bool control = permitted && state.locked && state.commanding;
    const float command_yaw = control ? state.command_yaw : (feedback_valid ? measured.yaw : 0.0f);
    const float command_pitch =
      control ? state.command_pitch : (feedback_valid ? measured.pitch : 0.0f);
    const double observation_age = state.locked ? tools::delta_time(now, state.seen) : 0.0;
    const int tracking_state = !control ? 0 : (fresh_target ? 1 : 2);
    gimbal.send(control, false, command_yaw, command_pitch);
    plotter.plot(
      {{"frame", state.frame},
       {"time_s", tools::delta_time(now, state.started)},
       {"mode", static_cast<int>(mode)},
       {"tracking_state", tracking_state},
       {"target_valid", state.locked},
       {"fresh_target", fresh_target},
       {"observation_age_s", observation_age},
       {"aim_valid", state.locked},
       {"aim_yaw_rad", state.locked ? state.aim_yaw : 0.0},
       {"aim_pitch_rad", state.locked ? state.aim_pitch : 0.0},
       {"command_yaw_rad", command_yaw},
       {"command_pitch_rad", command_pitch},
       {"feedback_valid", feedback_valid},
       {"feedback_yaw_rad", feedback_valid ? measured.yaw : 0.0f},
       {"feedback_pitch_rad", feedback_valid ? measured.pitch : 0.0f},
       {"control", control},
       {"fire", false},
       {"invalid_frame", invalid_frame},
       {"quitting", quitting}});
    state.output = now;
    ++state.frame;
    if (quitting) break;

    // Your code end
  }

  return 0;
}
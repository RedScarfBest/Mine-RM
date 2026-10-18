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
    // Module 1: Persistent tracking state and per-frame candidate data.
    // Static state survives loop iterations; angles use radians and positions use meters.
    struct Task1State {
      int frame = 0;
      bool locked = false;      // Keep the selected armor identity until release or timeout.
      bool commanding = false;  // True after a valid control command has been generated.
      auto_aim::ArmorName name = auto_aim::not_armor;
      cv::Point2f center{};     // Last selected image center, used only for target association.
      double aim_yaw = 0;       // Target direction in world coordinates, before angular offsets.
      double aim_pitch = 0;     // Target elevation in world coordinates, before angular offsets.
      float command_yaw = 0;    // Last yaw command after offset and slew limiting (rad).
      float command_pitch = 0;  // Last pitch command after offset and limits (rad).
      std::chrono::steady_clock::time_point started = std::chrono::steady_clock::now();
      std::chrono::steady_clock::time_point seen{};  // Timestamp of the last valid target image.
      std::chrono::steady_clock::time_point last_frame{};  // Reject repeated or reversed timestamps.
      std::chrono::steady_clock::time_point output = started;  // Previous command-generation time.
    };
    struct Task1Config {
      auto_aim::Color enemy;
      double yaw_offset;                 // YAML yaw correction, converted from degrees to radians.
      double pitch_offset;               // YAML pitch correction, converted from degrees to radians.
      cv::Mat camera_matrix;             // Camera intrinsics for the actual image resolution.
      cv::Mat distortion;                // Five camera distortion coefficients.
      Eigen::Matrix3d camera_to_gimbal;  // Camera-to-gimbal rotation from calibration.
      Eigen::Vector3d camera_translation;  // Camera origin in the gimbal frame (m).
      Eigen::Vector3d yaw_to_pitch;        // Optional yaw-axis-to-pitch-axis offset (m).
    };
    struct Candidate {
      auto_aim::ArmorName name;
      cv::Point2f center;
      double confidence;
      Eigen::Vector3d ypd;  // Target yaw (rad), pitch (rad), and distance (m).
    };
    static Task1State state;
    // Module 2: Load configuration once and validate the local PnP inputs.
    // Use calibration for this robot; changing YAML requires restarting the program.
    static const Task1Config config = [&] {
      const auto yaml = tools::load(config_path);
      const auto enemy = tools::read<std::string>(yaml, "enemy_color");
      const double yaw_offset = tools::read<double>(yaml, "yaw_offset") * CV_PI / 180.0;
      const double pitch_offset = tools::read<double>(yaml, "pitch_offset") * CV_PI / 180.0;
      if ((enemy != "red" && enemy != "blue") || !std::isfinite(yaw_offset) ||
          !std::isfinite(pitch_offset) || std::abs(yaw_offset) > CV_PI ||
          std::abs(pitch_offset) > CV_PI) {
        throw std::runtime_error("Invalid Task 1 color or angular offsets");
      }
      // Check vector sizes before constructing matrices from their data pointers.
      auto read_vector = [&](const char* key, std::size_t size) {
        auto values = tools::read<std::vector<double>>(yaml, key);
        if (values.size() != size ||
            !std::all_of(values.begin(), values.end(), [](double v) { return std::isfinite(v); })) {
          throw std::runtime_error(std::string("Invalid Task 1 calibration: ") + key);
        }
        return values;
      };
      const auto k = read_vector("camera_matrix", 9);   // Row-major 3x3 intrinsic matrix.
      const auto d = read_vector("distort_coeffs", 5);  // k1, k2, p1, p2, k3.
      const auto r = read_vector("R_camera2gimbal", 9);
      const auto translation = read_vector("t_camera2gimbal", 3);
      Eigen::Vector3d yaw_to_pitch = Eigen::Vector3d::Zero();
      if (yaml["t_yaw2pitch"]) {
        const auto offset = read_vector("t_yaw2pitch", 3);
        yaw_to_pitch = Eigen::Vector3d(offset.data());
      }
      cv::Mat camera_matrix(3, 3, CV_64F);
      cv::Mat distortion(1, 5, CV_64F);
      std::copy(k.begin(), k.end(), camera_matrix.ptr<double>());
      std::copy(d.begin(), d.end(), distortion.ptr<double>());
      return Task1Config{enemy == "red" ? auto_aim::red : auto_aim::blue,
                         yaw_offset,
                         pitch_offset,
                         camera_matrix,
                         distortion,
                         Eigen::Matrix<double, 3, 3, Eigen::RowMajor>(r.data()),
                         Eigen::Vector3d(translation.data()),
                         yaw_to_pitch};
    }();
    // Module 3: Timing, motion limits, and validation helpers.
    static const double hold_seconds = [&] {
      const auto yaml = tools::load(config_path);
      const double seconds = tools::read_or<double>(yaml, "task1_hold_seconds", 0.300);

      if (!std::isfinite(seconds) || seconds < 0.0 || seconds > 0.5) {
        throw std::runtime_error("task1_hold_seconds must be within [0, 0.5] seconds");
      }

      return seconds;
    }();  // Keep the last command for ordinary detection misses (s).
    constexpr double frame_valid_seconds = 0.150;  // Maximum software image age at decision time (s).
    constexpr auto camera_timeout = std::chrono::milliseconds(50);  // Camera queue wait budget (ms).
    constexpr double pitch_limit =
        static_cast<float>(20.0 * CV_PI / 180.0);  // +/-20 deg, matching float feedback precision.
    const bool print_log = state.frame % 30 == 0;  // Print detailed diagnostics every 30 iterations.
    auto clear_target = [&] {
      state.locked = false;
      state.commanding = false;
    };
    auto points_finite = [](const std::vector<cv::Point2f>& points) {
      return std::all_of(points.begin(), points.end(), [](const cv::Point2f& point) {
        return std::isfinite(point.x) && std::isfinite(point.y);
      });
    };

    // Module 4: Read a frame, release stale input, and check the operating mode.
    // Camera gaps pause commands; target identity expires by its observation age.
    const auto cycle_started = std::chrono::steady_clock::now();
    const bool camera_ok = camera.try_read_for(img, t, camera_timeout);
    if (!camera_ok) img.release();
    const auto camera_finished = std::chrono::steady_clock::now();
    double detect_seconds = 0;
    double solve_seconds = 0;
    int rejected_candidates = 0;
    auto mode = gimbal.mode();
    const auto before_detection = std::chrono::steady_clock::now();
    if (mode != io::GimbalMode::AUTO_AIM ||
        (state.locked && tools::delta_time(before_detection, state.seen) > hold_seconds)) {
      clear_target();
    }
    bool invalid_frame = !camera_ok || img.empty();
    bool fresh_target = false;
    std::optional<Candidate> selected;
    double best_score = 0;
    // Module 5: Validate image time and obtain the corresponding gimbal orientation.
    // Detection and image display run only in AUTO_AIM; there is no observation-only mode.
    if (!invalid_frame && mode == io::GimbalMode::AUTO_AIM) {
      const double age = tools::delta_time(before_detection, t);
      if (age < 0 || age > frame_valid_seconds || t <= state.last_frame) {
        invalid_frame = true;
      } else {
        state.last_frame = t;
        // This framework call can block; camera_timeout does not limit its wait.
        q = gimbal.q(t);
      }
      const double norm = invalid_frame ? 0.0 : q.norm();
      // Reject nonfinite or near-zero quaternions before normalization (norm threshold: 1e-6).
      if (invalid_frame || !q.coeffs().allFinite() || !std::isfinite(norm) || norm <= 1e-6) {
        invalid_frame = true;
        if (print_log) tools::logger()->warn("invalid_quaternion_or_frame_time");
      } else {
        q.normalize();
        solver.set_R_gimbal2world(q);
        // Module 6: Detect armor and solve each relevant candidate independently.
        // One bad candidate must not discard another valid target in the same image.
        const auto detection_started = std::chrono::steady_clock::now();
        auto armors = yolo.detect(img, state.frame);
        detect_seconds = tools::delta_time(std::chrono::steady_clock::now(), detection_started);
        const auto solve_started = std::chrono::steady_clock::now();
        cv::Mat img_show = img.clone();
        if (print_log) {
          tools::logger()->info("frame_count: {}, armors.size(): {}", state.frame, armors.size());
        }
        for (auto& armor : armors) {
          // Filter color before PnP; associate valid candidates by world direction.
          if (armor.color != config.enemy) continue;
          if (armor.points.size() != 4 || !points_finite(armor.points) ||
              !std::isfinite(armor.center.x) || !std::isfinite(armor.center.y) ||
              !std::isfinite(armor.confidence) ||
              std::abs(cv::contourArea(armor.points)) <= 1.0) {  // Reject area <= 1 square pixel.
            ++rejected_candidates;
            if (print_log) tools::logger()->warn("invalid_points");
            continue;
          }
          std::vector<cv::Point2f> projected;
          try {
            // Physical plate widths: big 0.230 m, special 0.129 m, small 0.135 m.
            const float width = armor.type == auto_aim::big       ? 0.230f
                                : armor.type == auto_aim::special ? 0.129f
                                                                  : 0.135f;
            // Plate corners follow detector order; 0.028 m is half the 0.056 m height.
            const std::vector<cv::Point3f> object_points{{0, width / 2, 0.028f},
                                                         {0, -width / 2, 0.028f},
                                                         {0, -width / 2, -0.028f},
                                                         {0, width / 2, -0.028f}};
            cv::Vec3d rvec(0, 0, 0), tvec(0, 0, 0);
            const bool solved = cv::solvePnP(
                object_points, armor.points, config.camera_matrix, config.distortion, rvec, tvec,
                false, cv::SOLVEPNP_IPPE);  // Planar PnP, without an initial guess.
            // Accept only a successful finite solution in front of the camera (positive Z).
            if (!solved || !cv::checkRange(rvec) || !cv::checkRange(tvec) || tvec[2] <= 0) {
              ++rejected_candidates;
              continue;
            }
            // Transform the solved center: camera -> gimbal -> world, including axis offset.
            const Eigen::Vector3d camera_position(tvec[0], tvec[1], tvec[2]);
            armor.xyz_in_gimbal =
                config.camera_to_gimbal * camera_position + config.camera_translation;
            const Eigen::Matrix3d rotation = solver.R_gimbal2world();
            const double yaw = tools::eulers(rotation, 2, 1, 0)[0];
            const Eigen::Matrix3d yaw_rotation =
                Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ()).toRotationMatrix();
            armor.xyz_in_world = rotation * armor.xyz_in_gimbal + yaw_rotation * config.yaw_to_pitch;
            armor.ypd_in_world = tools::xyz2ypd(armor.xyz_in_world);
            // Raw-pose reprojection is diagnostic; projection failure does not veto position.
            try {
              cv::projectPoints(object_points, rvec, tvec, config.camera_matrix, config.distortion,
                                projected);
            } catch (const cv::Exception& error) {
              if (print_log) tools::logger()->warn("projection_unavailable: {}", error.what());
            }
          } catch (const cv::Exception& error) {
            ++rejected_candidates;
            if (print_log) tools::logger()->warn("solve_failed: {}", error.what());
            continue;
          }
          if (!armor.xyz_in_gimbal.allFinite() || !armor.xyz_in_world.allFinite() ||
              !armor.ypd_in_world.allFinite() || armor.ypd_in_world[2] <= 0) {
            ++rejected_candidates;
            if (print_log) tools::logger()->warn("invalid_pose");
            continue;
          }
          bool reprojection_valid = false;
          double reprojection_error = 0;
          tools::draw_points(img_show, armor.points, cv::Scalar(0, 255, 0),
                             3);  // Green: detected corners.
          try {
            if (projected.size() == 4 && points_finite(projected)) {
              double sum = 0;
              for (int i = 0; i < 4; ++i) sum += cv::norm(projected[i] - armor.points[i]);
              if (std::isfinite(sum)) {
                reprojection_error = sum / 4.0;
                reprojection_valid = true;
                tools::draw_points(img_show, projected, cv::Scalar(0, 0, 255),
                                   3);  // Red: raw PnP projection.
              }
            }
          } catch (const cv::Exception& error) {
            if (print_log) tools::logger()->warn("projection_unavailable: {}", error.what());
          }
          if (print_log) {
            tools::logger()->info(
                "{} {} {} conf={:.3f} ratio={:.2f} distance_m={:.3f} "
                "aim_yaw_rad={:.3f} aim_pitch_rad={:.3f} "
                "reproj_valid={} reproj_raw_px={:.3f}",
                auto_aim::COLORS[armor.color], auto_aim::ARMOR_NAMES[armor.name],
                auto_aim::ARMOR_TYPES[armor.type], armor.confidence, armor.ratio,
                armor.ypd_in_world[2], armor.ypd_in_world[0], armor.ypd_in_world[1],
                static_cast<int>(reprojection_valid), reprojection_error);
          }
          // Prefer the image center on acquisition, or the previous target center while locked.
          // Confidence breaks equal-distance ties; the image center is not the 3D aim point.
          const cv::Point2f reference =
              state.locked ? state.center : cv::Point2f(img.cols * 0.5f, img.rows * 0.5f);
          double score = cv::norm(armor.center - reference);
          if (state.locked) {
            const double yaw_delta = std::remainder(armor.ypd_in_world[0] - state.aim_yaw, 2 * CV_PI);
            const double pitch_delta = armor.ypd_in_world[1] - state.aim_pitch;
            const double separation = std::hypot(yaw_delta, pitch_delta);
            if (separation > 15.0 * CV_PI / 180.0) continue;
            score = separation + (armor.name == state.name ? 0.0 : 0.02);
          }
          if (!selected || score < best_score ||
              (score == best_score && armor.confidence > selected->confidence)) {
            selected = Candidate{armor.name, armor.center, armor.confidence, armor.ypd_in_world};
            best_score = score;
          }
        }
        solve_seconds = tools::delta_time(std::chrono::steady_clock::now(),
                                          solve_started);  // Includes candidate drawing.
        cv::resize(img_show, img_show, cv::Size(), 0.5,
                   0.5);  // Display at half size; PnP uses original pixels.
        cv::imshow("reprojected_points (raw PnP)", img_show);
      }
    }

    // Module 7: Apply feedback, exit, and end-of-processing freshness checks.
    const auto measured = gimbal.state();
    const bool feedback_valid = std::isfinite(measured.yaw) && std::isfinite(measured.pitch) &&
                                std::abs(measured.pitch) <= CV_PI / 2;
    const double feedback_yaw =
        feedback_valid ? std::remainder(double(measured.yaw), 2 * CV_PI) : 0.0;
    // Numeric validity does not prove serial feedback freshness; the framework exposes no age.
    const bool pitch_in_range = feedback_valid && std::abs(measured.pitch) <= pitch_limit;
    const int key = cv::waitKey(1);  // Pump GUI events; q/Esc works only when this point is reached.
    const bool quitting = key == 27 || key == 'q' || exiter.exit();
    const auto now = std::chrono::steady_clock::now();
    mode = gimbal.mode();
    const double frame_age = tools::delta_time(now, t);
    if (!img.empty() && (frame_age < 0 || frame_age > frame_valid_seconds)) invalid_frame = true;
    // AUTO_AIM, credible feedback, fresh input, and no exit request are required.
    const bool permitted =
        mode == io::GimbalMode::AUTO_AIM && feedback_valid && !invalid_frame && !quitting;
    // Module 8: Update tracking and generate offset-corrected, limited angle commands.
    // On takeover start from feedback; subsequent commands advance from the previous command.
    if (!permitted) {
      const bool short_camera_gap = (!camera_ok || img.empty()) && mode == io::GimbalMode::AUTO_AIM &&
                                    feedback_valid && !quitting && state.locked &&
                                    tools::delta_time(now, state.seen) <= hold_seconds;
      if (short_camera_gap)
        state.commanding = false;
      else
        clear_target();
    } else if (selected) {
      state.locked = true;
      state.name = selected->name;
      state.center = selected->center;
      state.seen = t;
      state.aim_yaw = selected->ypd[0];
      state.aim_pitch = selected->ypd[1];
      // Use circular yaw differences and keep pitch output within its soft range.
      const double desired_yaw = std::remainder(state.aim_yaw + config.yaw_offset, 2.0 * CV_PI);
      const double desired_pitch = tools::limit_min_max(-state.aim_pitch + config.pitch_offset,
                                                        -pitch_limit, pitch_limit);  // edit
      const double dt = tools::limit_min_max(tools::delta_time(now, state.output), 0.0,
                                             0.1);  // Cap slew time at 100 ms after delays.
      const double base_yaw = state.commanding ? state.command_yaw : feedback_yaw;
      const double base_pitch = state.commanding ? state.command_pitch : measured.pitch;
      state.command_yaw = static_cast<float>(std::remainder(
          base_yaw + tools::limit_min_max(std::remainder(desired_yaw - base_yaw, 2.0 * CV_PI),
                                          -3.0 * dt, 3.0 * dt),
          2.0 * CV_PI));  // Maximum yaw command slew: 3 rad/s.
      state.command_pitch = static_cast<float>(tools::limit_min_max(
          base_pitch + tools::limit_min_max(desired_pitch - base_pitch, -dt, dt), -pitch_limit,
          pitch_limit));  // +/-dt gives a 1 rad/s pitch slew limit.
      state.commanding = std::abs(state.command_pitch - base_pitch) <= dt + 1e-6;
      fresh_target = true;
    } else if (!state.locked || tools::delta_time(now, state.seen) >
                                    hold_seconds) {  // Release expired ordinary-miss holds.
      clear_target();
    }
    const bool control = permitted && state.locked && state.commanding;
    const float command_yaw =
        control ? static_cast<float>(std::remainder(double(state.command_yaw), 2.0 * CV_PI))
                : static_cast<float>(feedback_yaw);  // edit
    const float command_pitch =
        control ? state.command_pitch : (feedback_valid ? measured.pitch : 0.0f);
    const double observation_age = state.locked ? tools::delta_time(now, state.seen) : 0.0;
    const int tracking_state =
        !control ? 0 : (fresh_target ? 1 : 2);  // 0: off, 1: tracking, 2: holding.
    // Module 9: Send gimbal angles and publish matching diagnostic data.
    // Plotter sends UDP JSON; open PlotJuggler separately on 127.0.0.1:9870.
    gimbal.send(control, false, command_yaw,
                command_pitch);  // fire=false; no chassis movement command.
    // All *_rad fields use radians. aim_* is before offsets; command_* is the final command.
    // When aim_valid=false, the zero aim values are placeholders, not a valid target direction.
    // Software image age is not full exposure-to-actuation latency.
    // cycle_s ends before serial/UDP sending; solve_s includes candidate processing and drawing.
    plotter.plot({{"frame", state.frame},
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
                  {"internal_command_yaw_rad", state.command_yaw},
                  {"feedback_yaw_raw_rad", std::isfinite(measured.yaw) ? measured.yaw : 0.0f},
                  {"feedback_pitch_raw_rad", std::isfinite(measured.pitch) ? measured.pitch : 0.0f},
                  {"control_permitted", permitted},
                  {"command_ready", state.commanding},
                  {"control", control},
                  {"fire", false},
                  {"invalid_frame", invalid_frame},
                  {"camera_ok", camera_ok},
                  {"pitch_in_range", pitch_in_range},
                  {"rejected_candidates", rejected_candidates},
                  {"software_frame_age_s", img.empty() ? 0.0 : frame_age},
                  {"camera_wait_s", tools::delta_time(camera_finished, cycle_started)},
                  {"detect_s", detect_seconds},
                  {"solve_s", solve_seconds},
                  {"cycle_s", tools::delta_time(now, cycle_started)},
                  {"frame_valid_seconds", frame_valid_seconds},
                  {"hold_seconds", hold_seconds},
                  {"quitting", quitting}});
    state.output = now;
    ++state.frame;
    if (quitting)
      break;  // Reached q/Esc exits after send; blocked I/O or Ctrl+C may bypass final stop.

    // Your code end
  }

  return 0;
}
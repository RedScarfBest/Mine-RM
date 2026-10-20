#include <algorithm>  //add
#include <chrono>
#include <cmath>   //add
#include <deque>   //add
#include <limits>  //add
#include <opencv2/opencv.hpp>
#include <optional>   //add
#include <stdexcept>  //add

#include "io/camera.hpp"
#include "io/gimbal/gimbal.hpp"
#include "tasks/auto_aim/solver.hpp"
#include "tasks/auto_aim/target.hpp"
#include "tasks/auto_aim/yolo.hpp"
#include "tools/exiter.hpp"
#include "tools/img_tools.hpp"
#include "tools/logger.hpp"
#include "tools/math_tools.hpp"
#include "tools/plotter.hpp"
#include "tools/trajectory.hpp"  //add
#include "tools/yaml.hpp"        //add

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
    struct Task2State
    {
      int frame = 0;
      std::optional<auto_aim::Target> target;
      std::deque<std::pair<std::chrono::steady_clock::time_point, double>> speeds;
      std::chrono::steady_clock::time_point started = std::chrono::steady_clock::now();
      std::chrono::steady_clock::time_point seen{};
      std::chrono::steady_clock::time_point model_time{};
      std::chrono::steady_clock::time_point last_frame_time{};
      std::chrono::steady_clock::time_point output = started;
      std::chrono::steady_clock::time_point last_fire{};
      bool commanding = false;
      bool previous_fire = false;
      float command_yaw = 0;
      float command_pitch = 0;
      int selected_board = -1;
      int updates = 0;
      unsigned int fire_count = 0;
    };
    struct Task2Config
    {
      auto_aim::Color enemy;
      double yaw_offset;
      double pitch_offset;
      double execution_delay;
      double hold_seconds;
      double target_radius;
      bool fire_enabled;
    };
    static Task2State state;
    static const Task2Config config = [&] {
      const auto yaml = tools::load(config_path);
      const auto enemy = tools::read<std::string>(yaml, "enemy_color");
      const double yaw = tools::read<double>(yaml, "yaw_offset") * CV_PI / 180.0;
      const double pitch = tools::read<double>(yaml, "pitch_offset") * CV_PI / 180.0;
      const double delay = tools::read_or<double>(yaml, "task2_execution_delay_s", 0.0);
      const double hold = tools::read_or<double>(yaml, "task2_hold_seconds", 0.300);
      const double radius = tools::read_or<double>(yaml, "task2_target_radius_m", 0.2);
      if (!std::isfinite(radius) || radius <= 0.05 || radius >= 0.5) {
        throw std::runtime_error("task2_target_radius_m must be within (0.05, 0.5) metres");
      }
      if (!std::isfinite(hold) || hold < 0.0 || hold > 0.5) {
        throw std::runtime_error("task2_hold_seconds must be within [0, 0.5] seconds");
      }
      if (
        (enemy != "red" && enemy != "blue") || !std::isfinite(yaw) || !std::isfinite(pitch) ||
        std::abs(yaw) > CV_PI || std::abs(pitch) > CV_PI || !std::isfinite(delay) || delay < 0 ||
        delay > 0.5) {
        throw std::runtime_error("Invalid Task 2 configuration");
      }
      return Task2Config{
        enemy == "red" ? auto_aim::red : auto_aim::blue,        yaw, pitch, delay, hold, radius,
        tools::read_or<bool>(yaml, "task2_fire_enabled", false)};
    }();
    const double hold_seconds = config.hold_seconds;  // Ordinary misses; zero disables coasting.
    constexpr auto camera_timeout = std::chrono::milliseconds(50);
    constexpr double frame_valid_seconds = 0.150;
    // Match the accepted Task 1 float boundary; this limits the final aim, not yaw turns.
    constexpr double pitch_limit = static_cast<float>(20.0 * CV_PI / 180.0);
    auto reset = [&] {
      state.target.reset();
      state.speeds.clear();
      state.updates = 0;
      state.commanding = false;
      state.selected_board = -1;
    };
    auto valid_model = [](const auto_aim::Target & target) {
      const auto x = target.ekf_x();
      return x.allFinite() && target.ekf().P.allFinite() && !target.diverged() &&
             std::abs(x[7]) <= 20 && std::abs(x[10]) <= 0.5 &&
             target.ekf().P.diagonal().minCoeff() >= 0;
    };
    // A camera timeout pauses commands while a recent model may still be retained.
    img.release();
    const bool camera_ok = camera.try_read_for(img, t, camera_timeout);
    if (!camera_ok) img.release();
    auto mode = gimbal.mode();
    auto feedback = gimbal.state();
    bool feedback_valid = std::isfinite(feedback.yaw) && std::isfinite(feedback.pitch) &&
                          std::abs(feedback.pitch) <= CV_PI / 2;
    const auto before = std::chrono::steady_clock::now();
    if (
      mode != io::GimbalMode::AUTO_AIM || !feedback_valid ||
      (state.target && hold_seconds > 0 && tools::delta_time(before, state.seen) > hold_seconds))
      reset();
    const double input_age = tools::delta_time(before, t);
    const bool timestamp_valid =
      camera_ok && !img.empty() && input_age >= 0 && input_age <= frame_valid_seconds;
    bool invalid_frame =
      !timestamp_valid ||
      (state.last_frame_time.time_since_epoch().count() != 0 && t <= state.last_frame_time);
    bool fresh_target = false;
    int rejected_candidates = 0;
    std::string reason = "no_target";
    if (!img.empty()) {
      const double age = tools::delta_time(before, t);
      if (
        age < 0 || age > frame_valid_seconds ||
        (state.last_frame_time.time_since_epoch().count() != 0 && t <= state.last_frame_time)) {
        invalid_frame = true;
      } else {
        q = gimbal.q(t);
        const double norm = q.norm();
        if (!q.coeffs().allFinite() || !std::isfinite(norm) || norm <= 1e-6) {
          invalid_frame = true;
        } else {
          q.normalize();
          solver.set_R_gimbal2world(q);
          auto armors = yolo.detect(img, state.frame);
          if (state.target) {
            state.target->predict(t);
            state.model_time = t;
          }
          if (state.target && !valid_model(*state.target)) reset();
          auto_aim::Armor * selected = nullptr;
          double best_score = std::numeric_limits<double>::infinity();
          bool locked_pose_failed = false;
          for (auto & armor : armors) {
            if (
              armor.color != config.enemy || armor.type != auto_aim::small ||
              armor.name < auto_aim::two || armor.name > auto_aim::five ||
              (state.target && armor.name != state.target->name))
              continue;
            bool valid = armor.points.size() == 4 && std::isfinite(armor.center.x) &&
                         std::isfinite(armor.center.y) && std::isfinite(armor.confidence);
            for (const auto & point : armor.points) {
              valid = valid && std::isfinite(point.x) && std::isfinite(point.y);
            }
            valid = valid && std::abs(cv::contourArea(armor.points)) > 1.0;
            if (valid) {
              try {
                solver.solve(armor);
                valid = armor.xyz_in_gimbal.allFinite() && armor.xyz_in_world.allFinite() &&
                        armor.ypr_in_world.allFinite() &&
                        armor.ypr_in_world.cwiseAbs().maxCoeff() <= 4 * CV_PI &&
                        armor.ypd_in_world.allFinite() && armor.ypd_in_world[2] > 0.1 &&
                        armor.ypd_in_world[2] < 30;
              } catch (const cv::Exception &) {
                valid = false;
              }
            }
            if (!valid) {
              ++rejected_candidates;
              locked_pose_failed = locked_pose_failed || static_cast<bool>(state.target);
              continue;
            }
            double score = cv::norm(armor.center - cv::Point2f(img.cols * 0.5f, img.rows * 0.5f));
            if (state.target) {
              score = std::numeric_limits<double>::infinity();
              for (const auto & board : state.target->armor_xyza_list()) {
                const double distance = (board.head<3>() - armor.xyz_in_world).norm();
                const double angle =
                  std::abs(std::remainder(board[3] - armor.ypr_in_world[0], 2.0 * CV_PI));
                if (distance < 0.8 && angle < 1.0) score = std::min(score, distance + 0.2 * angle);
              }
            }
            if (
              std::isfinite(score) &&
              (!selected || score < best_score ||
               (score == best_score && armor.confidence > selected->confidence))) {
              selected = &armor;
              best_score = score;
            }
          }
          if (selected && mode == io::GimbalMode::AUTO_AIM && feedback_valid) {
            selected->priority = auto_aim::first;
            if (!state.target)
              state.target.emplace(*selected, t, config.target_radius, 4);
            else
              state.target->update(*selected);
            if (valid_model(*state.target)) {
              fresh_target = true;
              state.seen = t;
              state.model_time = t;
              ++state.updates;
              state.speeds.emplace_back(t, state.target->ekf_x()[7]);
              while (!state.speeds.empty() &&
                     tools::delta_time(t, state.speeds.front().first) > 0.5) {
                state.speeds.pop_front();
              }
            } else {
              invalid_frame = true;
            }
          } else if (locked_pose_failed && !selected) {
            invalid_frame = true;
          }
        }
      }
    }
    if (timestamp_valid && t > state.last_frame_time) state.last_frame_time = t;
    const int key = cv::waitKey(1);
    const bool quitting = key == 'q' || key == 27 || exiter.exit();
    const auto now = std::chrono::steady_clock::now();
    mode = gimbal.mode();
    feedback = gimbal.state();
    feedback_valid = std::isfinite(feedback.yaw) && std::isfinite(feedback.pitch) &&
                     std::abs(feedback.pitch) <= CV_PI / 2;
    // Normalize before adding small corrections, including for cumulative feedback yaw.
    const double feedback_yaw =
      feedback_valid ? std::remainder(double(feedback.yaw), 2.0 * CV_PI) : 0.0;
    const bool pitch_in_range = feedback_valid && std::abs(feedback.pitch) <= pitch_limit;
    const double frame_age = tools::delta_time(now, t);
    if (!img.empty() && (frame_age < 0 || frame_age > frame_valid_seconds)) invalid_frame = true;
    const bool permitted =
      mode == io::GimbalMode::AUTO_AIM && feedback_valid && !invalid_frame && !quitting;
    if (!permitted) {
      const bool short_camera_gap = (!camera_ok || img.empty()) &&
                                    mode == io::GimbalMode::AUTO_AIM && feedback_valid &&
                                    !quitting && state.target && hold_seconds > 0 &&
                                    tools::delta_time(now, state.seen) <= hold_seconds;
      if (short_camera_gap) {
        state.commanding = false;
        state.selected_board = -1;
      } else {
        reset();
      }
      fresh_target = false;
    } else if (state.target && !fresh_target && tools::delta_time(now, state.seen) > hold_seconds) {
      reset();
    }
    bool stable = false;
    double omega = 0;
    double omega_std = 0;
    if (state.target) {
      omega = state.target->ekf_x()[7];
      if (
        state.updates >= 10 && state.speeds.size() >= 10 &&
        tools::delta_time(state.speeds.back().first, state.speeds.front().first) >= 0.45) {
        double mean = 0;
        for (const auto & sample : state.speeds) mean += sample.second;
        mean /= state.speeds.size();
        for (const auto & sample : state.speeds) omega_std += std::pow(sample.second - mean, 2);
        omega_std = std::sqrt(omega_std / state.speeds.size());
        stable = state.target->jumped && omega_std <= 0.3 && state.target->ekf().P(7, 7) <= 1.0 &&
                 std::abs(omega) >= 1.0 && std::abs(omega) <= 12.0;
      }
    }
    bool aim_valid = false;
    bool facing = false;
    double aim_yaw = 0, aim_pitch = 0, horizon = 0;
    double yaw_tolerance = 0, pitch_tolerance = 0;
    double prediction_model_age = 0, flight_time = 0, selected_face_error = 0;
    int board_id = -1;
    const bool speed_valid = std::isfinite(feedback.bullet_speed) && feedback.bullet_speed >= 5 &&
                             feedback.bullet_speed <= 40;
    if (state.target && permitted) {
      double best = std::numeric_limits<double>::infinity();
      for (int id = 0; id < 4; ++id) {
        double fly = 0;
        double pitch = 0;
        Eigen::Vector4d board = Eigen::Vector4d::Zero();
        bool solved = false;
        double prediction_time = 0;
        if (speed_valid) {
          fly = state.target->armor_xyza_list()[id].head<3>().norm() / feedback.bullet_speed;
          for (int iteration = 0; iteration < 8; ++iteration) {
            prediction_time = tools::delta_time(now, state.model_time) + config.execution_delay + fly;
            if (prediction_time < 0 || prediction_time > 1.0) break;
            auto future = *state.target;
            future.predict(prediction_time);
            board = future.armor_xyza_list()[id];
            const double distance = board.head<2>().norm();
            if (!board.allFinite() || distance < 0.1) break;
            tools::Trajectory trajectory(feedback.bullet_speed, distance, board[2], 1);
            if (
              trajectory.unsolvable || !std::isfinite(trajectory.pitch) ||
              !std::isfinite(trajectory.fly_time) || trajectory.fly_time <= 0)
              break;
            pitch = trajectory.pitch;
            solved = std::abs(trajectory.fly_time - fly) < 1e-4;
            fly = trajectory.fly_time;
            if (solved) break;
          }
        } else {
          // Without measured bullet speed, follow the current line of sight only.
          // Advance a copy to now; no guessed flight time or ballistic compensation.
          prediction_time = tools::delta_time(now, state.model_time);
          if (!std::isfinite(prediction_time) || prediction_time < 0 || prediction_time > 1.0)
            continue;
          auto current = *state.target;
          current.predict(prediction_time);
          board = current.armor_xyza_list()[id];
          const double distance = board.head<2>().norm();
          if (!board.allFinite() || distance < 0.1) continue;
          pitch = std::atan2(board[2], distance);
          solved = true;
        }
        if (!solved) continue;
        const double yaw = std::atan2(board[1], board[0]);
        const double face_error = std::abs(std::remainder(board[3] - yaw, 2.0 * CV_PI));
        // The accepted robot command convention is opposite to world elevation.
        const double desired_pitch = -pitch + config.pitch_offset;
        if (face_error > CV_PI / 3 || std::abs(desired_pitch) > pitch_limit) continue;
        const double cost = face_error + (state.selected_board == id ? -0.1 : 0.0);
        if (cost >= best) continue;
        best = cost;
        aim_valid = true;
        facing = true;
        aim_yaw = std::remainder(yaw + config.yaw_offset, 2.0 * CV_PI);
        aim_pitch = desired_pitch;
        horizon = prediction_time;
        prediction_model_age = tools::delta_time(now, state.model_time);
        flight_time = fly;
        selected_face_error = face_error;
        board_id = id;
        const double distance = board.head<2>().norm();
        yaw_tolerance = std::atan2(0.135 * 0.25 * std::cos(face_error), distance);
        pitch_tolerance = std::atan2(0.056 * 0.25, distance);
      }
    }
    const auto command_time = std::chrono::steady_clock::now();
    if (
      (!img.empty() && tools::delta_time(command_time, t) > frame_valid_seconds) ||
      (state.target && !fresh_target &&
       tools::delta_time(command_time, state.seen) > hold_seconds)) {
      invalid_frame = true;
      reset();
      fresh_target = false;
      stable = false;
      aim_valid = false;
    }
    const bool ballistic_aim_valid = aim_valid && speed_valid;
    const bool control = permitted && state.target && aim_valid;
    const bool geometric_tracking = control && !speed_valid;
    if (control) {
      const double dt =
        tools::limit_min_max(tools::delta_time(command_time, state.output), 0.0, 0.1);
      const double base_yaw = state.commanding ? state.command_yaw : feedback_yaw;
      const double base_pitch = state.commanding ? state.command_pitch : feedback.pitch;
      // Shortest-angle slew and single-turn representation do not limit physical yaw turns.
      state.command_yaw = static_cast<float>(std::remainder(
        base_yaw + tools::limit_min_max(
                     std::remainder(aim_yaw - base_yaw, 2.0 * CV_PI), -3.0 * dt, 3.0 * dt),
        2.0 * CV_PI));
      // If feedback is outside +/-20 deg, return smoothly instead of clipping the first step.
      state.command_pitch =
        static_cast<float>(base_pitch + tools::limit_min_max(aim_pitch - base_pitch, -dt, dt));
      state.commanding = true;
      state.selected_board = board_id;
    } else {
      state.commanding = false;
      state.selected_board = -1;
    }
    const bool aligned =
      control && pitch_in_range && std::abs(state.command_pitch) <= pitch_limit &&
      std::abs(std::remainder(aim_yaw - feedback_yaw, 2.0 * CV_PI)) <= yaw_tolerance &&
      std::abs(aim_pitch - feedback.pitch) <= pitch_tolerance &&
      std::abs(std::remainder(aim_yaw - state.command_yaw, 2.0 * CV_PI)) <= yaw_tolerance &&
      std::abs(aim_pitch - state.command_pitch) <= pitch_tolerance;
    const bool cooldown =
      !state.previous_fire && (state.last_fire.time_since_epoch().count() == 0 ||
                               tools::delta_time(command_time, state.last_fire) >= 0.2);
    const bool fire =
      config.fire_enabled && control && ballistic_aim_valid && fresh_target && stable && facing &&
      aligned && cooldown;
    if (quitting)
      reason = "quitting";
    else if (mode != io::GimbalMode::AUTO_AIM)
      reason = "mode";
    else if (!feedback_valid)
      reason = "feedback_invalid";
    else if (invalid_frame)
      reason = "invalid_frame";
    else if (!state.target)
      reason = "no_target";
    else if (!fresh_target)
      reason = "coasting";
    else if (!speed_valid)
      reason = "bullet_speed";
    else if (!aim_valid)
      reason = "no_intercept";
    else if (!pitch_in_range)
      reason = "pitch_recovery";
    else if (!stable)
      reason = "acquiring";
    else if (!config.fire_enabled)
      reason = "fire_disabled";
    else if (!aligned)
      reason = "alignment";
    else if (!cooldown)
      reason = "cooldown";
    else
      reason = "ready";
    if (fire) {
      ++state.fire_count;
    }
    const float command_yaw =
      control ? static_cast<float>(std::remainder(double(state.command_yaw), 2.0 * CV_PI))
              : static_cast<float>(feedback_yaw);
    const float command_pitch =
      control ? state.command_pitch : (feedback_valid ? feedback.pitch : 0.0f);
    const int tracking_state = !state.target ? 0 : (!fresh_target ? 3 : (stable ? 2 : 1));
    const double speed_sample_span =
      state.speeds.size() >= 2
        ? tools::delta_time(state.speeds.back().first, state.speeds.front().first)
        : 0.0;
    const bool speed_window_ready =
      state.target && state.updates >= 10 && state.speeds.size() >= 10 && speed_sample_span >= 0.45;
    const auto sent_time = std::chrono::steady_clock::now();
    gimbal.send(control, fire, command_yaw, command_pitch);
    const auto send_finished = std::chrono::steady_clock::now();
    if (fire) state.last_fire = send_finished;
    plotter.plot(
      {{"frame", state.frame},
       {"time_s", tools::delta_time(sent_time, state.started)},
       {"mode", static_cast<int>(mode)},
       {"tracking_state", tracking_state},
       {"fresh_target", fresh_target},
       {"invalid_frame", invalid_frame},
       {"stable", stable},
       {"model_present", static_cast<bool>(state.target)},
       {"model_updates", state.updates},
       {"configured_target_radius_m", config.target_radius},
       {"geometry_frozen", state.target && state.target->ekf().P.block<3, 3>(8, 8).isZero(0.0)},
       {"model_radius_m", state.target ? state.target->ekf_x()[8] : 0.0},
       {"model_radius_difference_m", state.target ? state.target->ekf_x()[9] : 0.0},
       {"model_height_difference_m", state.target ? state.target->ekf_x()[10] : 0.0},
       {"speed_sample_count", state.speeds.size()},
       {"speed_sample_span_s", speed_sample_span},
       {"speed_window_ready", speed_window_ready},
       {"omega_std_ready", speed_window_ready},
       {"has_switched_board", state.target ? state.target->jumped : false},
       {"omega_covariance", state.target ? state.target->ekf().P(7, 7) : 0.0},
       {"bullet_speed_m_s", std::isfinite(feedback.bullet_speed) ? feedback.bullet_speed : 0.0f},
       {"bullet_speed_valid", speed_valid},
       {"fire_enabled", config.fire_enabled},
       {"execution_delay_s", config.execution_delay},
       {"aligned", aligned},
       {"facing", aim_valid && facing},
       {"cooldown_ready", cooldown},
       {"control", control},
       {"fire", fire},
       {"command_yaw_rad", command_yaw},
       {"command_pitch_rad", command_pitch},
       {"feedback_valid", feedback_valid},
       {"feedback_yaw_wrapped_rad", feedback_yaw},
       {"feedback_pitch_raw_rad", std::isfinite(feedback.pitch) ? feedback.pitch : 0.0f},
       {"pitch_in_range", pitch_in_range},
       {"camera_ok", camera_ok},
       {"control_permitted", permitted},
       {"command_ready", state.commanding},
       {"hold_seconds", hold_seconds},
       {"frame_valid_seconds", frame_valid_seconds},
       {"feedback_yaw_rad", feedback_valid ? feedback.yaw : 0.0f},
       {"feedback_pitch_rad", feedback_valid ? feedback.pitch : 0.0f},
       {"omega_rad_s", omega},
       {"omega_abs_rad_s", std::abs(omega)},
       {"omega_std_rad_s", omega_std},
       {"observation_age_s", state.target ? tools::delta_time(sent_time, state.seen) : 0.0},
       {"frame_age_s", tools::delta_time(sent_time, t)},
       {"aim_valid", aim_valid},
       {"ballistic_aim_valid", ballistic_aim_valid},
       {"geometric_tracking", geometric_tracking},
       {"aim_yaw_rad", aim_yaw},
       {"aim_pitch_rad", aim_pitch},
       {"prediction_time_s", horizon},
       {"prediction_model_age_s", aim_valid ? prediction_model_age : 0.0},
       {"flight_time_s", aim_valid ? flight_time : 0.0},
       {"face_error_rad", aim_valid ? selected_face_error : 0.0},
       {"yaw_tolerance_rad", aim_valid ? yaw_tolerance : 0.0},
       {"pitch_tolerance_rad", aim_valid ? pitch_tolerance : 0.0},
       {"feedback_yaw_error_rad",
        control ? std::remainder(aim_yaw - feedback_yaw, 2.0 * CV_PI) : 0.0},
       {"feedback_pitch_error_rad", control ? aim_pitch - feedback.pitch : 0.0},
       {"command_yaw_error_rad",
        control ? std::remainder(aim_yaw - command_yaw, 2.0 * CV_PI) : 0.0},
       {"command_pitch_error_rad", control ? aim_pitch - command_pitch : 0.0},
       {"selected_board", board_id},
       {"fire_command_count", state.fire_count},
       {"bullet_count", feedback.bullet_count},
       {"rejected_candidates", rejected_candidates},
       {"block_reason", reason},
       {"quitting", quitting}});
    if (state.frame % 30 == 0) {
      tools::logger()->info(
        "task2 frame={} state={} omega_rad_s={:.3f} stable={} control={} fire={} reason={} "
        "bullet_speed_m_s={:.3f} ballistic_aim_valid={}",
        state.frame, tracking_state, omega, stable, control, fire, reason,
        std::isfinite(feedback.bullet_speed) ? feedback.bullet_speed : 0.0f, ballistic_aim_valid);
    }
    state.previous_fire = fire;
    state.output = send_finished;
    ++state.frame;
    if (quitting) break;

    // Your code end
  }

  return 0;
}
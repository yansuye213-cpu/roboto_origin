// SPDX-License-Identifier: GPL-3.0
// Copyright (C) 2026 Luo1imasi

#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "robot_interface.hpp"

struct PolicyLogContext {
    float cmd_vx = 0.0f;
    float cmd_vy = 0.0f;
    float cmd_yaw = 0.0f;
    float imu_roll = 0.0f;
    float imu_pitch = 0.0f;
    float imu_wx = 0.0f;
    float imu_wy = 0.0f;
    float imu_wz = 0.0f;
    float gravity_z = -1.0f;
    float transition_phase = 1.0f;
    size_t clamped_joint_count = 0;
};

class RunLogger {
   public:
    RunLogger(std::filesystem::path output_directory,
              std::vector<std::string> joint_names);
    ~RunLogger();

    RunLogger(const RunLogger&) = delete;
    RunLogger& operator=(const RunLogger&) = delete;

    bool start(const std::string& policy_name, std::string& error_message);
    void finish(const std::string& reason, const std::string& detail, bool clean_exit);
    void record_control(const RobotInterface::TelemetrySnapshot& snapshot);
    void record_sample(const PolicyLogContext& context);
    void record_event(const std::string& event, const std::string& detail = "");

    bool active() const { return active_.load(std::memory_order_acquire); }
    std::filesystem::path current_path() const;

   private:
    struct JointRow {
        float q_actual = 0.0f;
        float dq_actual = 0.0f;
        float q_target = 0.0f;
        float tau_feedback_peak = 0.0f;
        float tau_demand_peak = 0.0f;
        float tau_feedback_utilization = 0.0f;
        float tau_demand_utilization = 0.0f;
        float motor_temperature = 0.0f;
        float mos_temperature = 0.0f;
        uint8_t error_code = 0;
    };

    struct CsvRow {
        std::chrono::system_clock::time_point wall_time;
        double elapsed_s = 0.0;
        bool is_sample = false;
        std::string event;
        std::string detail;
        uint64_t policy_step = 0;
        PolicyLogContext context;
        std::vector<JointRow> joints;
    };

    std::filesystem::path output_directory_;
    std::vector<std::string> joint_names_;
    std::filesystem::path active_path_;
    std::ofstream stream_;
    std::atomic<bool> active_{false};
    std::chrono::steady_clock::time_point start_steady_;
    uint64_t policy_step_ = 0;

    mutable std::mutex lifecycle_mutex_;
    std::mutex producer_mutex_;

    std::mutex telemetry_mutex_;
    RobotInterface::TelemetrySnapshot latest_telemetry_;
    std::vector<float> peak_feedback_tau_;
    std::vector<float> peak_demand_tau_;
    std::vector<uint8_t> last_fault_code_;
    std::vector<std::pair<size_t, uint8_t>> pending_faults_;
    bool has_telemetry_ = false;
    std::atomic<uint64_t> dropped_control_samples_{0};
    std::atomic<uint64_t> dropped_csv_rows_{0};

    std::mutex queue_mutex_;
    std::condition_variable queue_cv_;
    std::condition_variable drained_cv_;
    std::deque<CsvRow> queue_;
    bool writer_busy_ = false;
    bool writer_stop_ = false;
    std::thread writer_thread_;
    static constexpr size_t kMaxQueuedRows = 2048;

    void writer_loop();
    bool enqueue(CsvRow row, bool must_write);
    void wait_until_drained();
    CsvRow make_event_row(const std::string& event, const std::string& detail) const;
    void reset_aggregation();
    void recover_unclean_logs();
    void write_header();
    void write_row(const CsvRow& row);
    static std::string csv_escape(const std::string& value);
    static std::string format_wall_time(
        const std::chrono::system_clock::time_point& time_point);
    static std::string timestamp_for_filename();
    static std::string dm_error_name(uint8_t error_code);
};

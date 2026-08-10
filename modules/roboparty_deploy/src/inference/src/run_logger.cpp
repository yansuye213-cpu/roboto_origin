// SPDX-License-Identifier: GPL-3.0
// Copyright (C) 2026 Luo1imasi

#include "run_logger.hpp"

#include <algorithm>
#include <cmath>
#include <ctime>
#include <iomanip>
#include <limits>
#include <sstream>
#include <system_error>

namespace {

bool ends_with(const std::string& value, const std::string& suffix) {
    return value.size() >= suffix.size() &&
           value.compare(value.size() - suffix.size(), suffix.size(), suffix) == 0;
}

float signed_abs_max(float current, float candidate) {
    return std::abs(candidate) > std::abs(current) ? candidate : current;
}

}  // namespace

RunLogger::RunLogger(std::filesystem::path output_directory,
                     std::vector<std::string> joint_names)
    : output_directory_(std::move(output_directory)),
      joint_names_(std::move(joint_names)) {
    latest_telemetry_.resize(joint_names_.size());
    peak_feedback_tau_.assign(joint_names_.size(), 0.0f);
    peak_demand_tau_.assign(joint_names_.size(), 0.0f);
    last_fault_code_.assign(joint_names_.size(), 0);
    pending_faults_.reserve(joint_names_.size());
    writer_thread_ = std::thread(&RunLogger::writer_loop, this);
}

RunLogger::~RunLogger() {
    finish("node_shutdown", "logger destroyed while policy was active", false);
    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        writer_stop_ = true;
    }
    queue_cv_.notify_one();
    if (writer_thread_.joinable()) {
        writer_thread_.join();
    }
}

bool RunLogger::start(const std::string& policy_name, std::string& error_message) {
    std::lock_guard<std::mutex> lifecycle_lock(lifecycle_mutex_);
    if (active()) {
        error_message = "a policy run is already being recorded";
        return false;
    }

    try {
        std::filesystem::create_directories(output_directory_);
        recover_unclean_logs();
        active_path_ = output_directory_ /
                       ("run_" + timestamp_for_filename() + ".active.csv");
        stream_.open(active_path_, std::ios::out | std::ios::trunc);
        if (!stream_.is_open()) {
            error_message = "cannot open " + active_path_.string();
            return false;
        }
        write_header();
        stream_.flush();
    } catch (const std::exception& e) {
        error_message = e.what();
        if (stream_.is_open()) {
            stream_.close();
        }
        return false;
    }

    {
        std::lock_guard<std::mutex> producer_lock(producer_mutex_);
        start_steady_ = std::chrono::steady_clock::now();
        policy_step_ = 0;
        dropped_control_samples_.store(0);
        dropped_csv_rows_.store(0);
        reset_aggregation();
        active_.store(true, std::memory_order_release);
        enqueue(make_event_row("policy_start", policy_name), true);
    }
    error_message.clear();
    return true;
}

void RunLogger::finish(const std::string& reason, const std::string& detail,
                       bool clean_exit) {
    std::lock_guard<std::mutex> lifecycle_lock(lifecycle_mutex_);
    std::filesystem::path path_to_finalize;
    {
        std::lock_guard<std::mutex> producer_lock(producer_mutex_);
        if (!active()) {
            return;
        }
        active_.store(false, std::memory_order_release);
        std::string final_detail = detail;
        const uint64_t dropped_control = dropped_control_samples_.load();
        const uint64_t dropped_rows = dropped_csv_rows_.load();
        if (dropped_control > 0 || dropped_rows > 0) {
            if (!final_detail.empty()) {
                final_detail += "; ";
            }
            final_detail += "dropped_control_samples=" +
                            std::to_string(dropped_control) +
                            ", dropped_csv_rows=" + std::to_string(dropped_rows);
        }
        enqueue(make_event_row("run_end", reason +
                    (final_detail.empty() ? "" : ": " + final_detail)), true);
        path_to_finalize = active_path_;
    }

    wait_until_drained();
    if (stream_.is_open()) {
        stream_.flush();
        stream_.close();
    }

    const std::string active_suffix = ".active.csv";
    const std::string final_suffix = clean_exit ? ".ok.csv" : ".error.csv";
    std::string final_name = path_to_finalize.filename().string();
    if (ends_with(final_name, active_suffix)) {
        final_name.replace(final_name.size() - active_suffix.size(),
                           active_suffix.size(), final_suffix);
    }
    std::error_code ec;
    std::filesystem::rename(path_to_finalize,
                            path_to_finalize.parent_path() / final_name, ec);
}

void RunLogger::record_control(
    const RobotInterface::TelemetrySnapshot& snapshot) {
    if (!active()) {
        return;
    }
    std::lock_guard<std::mutex> lock(telemetry_mutex_);
    if (snapshot.joint_q.size() != joint_names_.size()) {
        dropped_control_samples_.fetch_add(1, std::memory_order_relaxed);
        return;
    }

    latest_telemetry_ = snapshot;
    has_telemetry_ = true;
    for (size_t i = 0; i < joint_names_.size(); ++i) {
        peak_feedback_tau_[i] =
            signed_abs_max(peak_feedback_tau_[i], snapshot.feedback_tau[i]);
        peak_demand_tau_[i] =
            signed_abs_max(peak_demand_tau_[i], snapshot.demand_tau[i]);
        const uint8_t error_code = snapshot.error_code[i];
        if (error_code >= 0x08 && error_code != last_fault_code_[i]) {
            pending_faults_.emplace_back(i, error_code);
        }
        last_fault_code_[i] = error_code;
    }
}

void RunLogger::record_sample(const PolicyLogContext& context) {
    std::lock_guard<std::mutex> producer_lock(producer_mutex_);
    if (!active()) {
        return;
    }

    CsvRow row;
    row.joints.resize(joint_names_.size());
    std::vector<std::pair<size_t, uint8_t>> faults;
    {
        std::lock_guard<std::mutex> telemetry_lock(telemetry_mutex_);
        if (!has_telemetry_) {
            return;
        }
        row.wall_time = std::chrono::system_clock::now();
        row.elapsed_s = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - start_steady_).count();
        row.is_sample = true;
        row.policy_step = policy_step_++;
        row.context = context;
        for (size_t i = 0; i < joint_names_.size(); ++i) {
            JointRow& joint = row.joints[i];
            joint.q_actual = latest_telemetry_.joint_q[i];
            joint.dq_actual = latest_telemetry_.joint_vel[i];
            joint.q_target = latest_telemetry_.joint_target[i];
            joint.tau_feedback_peak = peak_feedback_tau_[i];
            joint.tau_demand_peak = peak_demand_tau_[i];
            const float limit = latest_telemetry_.hardware_tau_limit[i];
            if (limit > std::numeric_limits<float>::epsilon()) {
                joint.tau_feedback_utilization =
                    std::abs(peak_feedback_tau_[i]) / limit;
                joint.tau_demand_utilization =
                    std::abs(peak_demand_tau_[i]) / limit;
            }
            joint.motor_temperature = latest_telemetry_.motor_temperature[i];
            joint.mos_temperature = latest_telemetry_.mos_temperature[i];
            joint.error_code = latest_telemetry_.error_code[i];
        }
        faults.assign(pending_faults_.begin(), pending_faults_.end());
        pending_faults_.clear();
        std::fill(peak_feedback_tau_.begin(), peak_feedback_tau_.end(), 0.0f);
        std::fill(peak_demand_tau_.begin(), peak_demand_tau_.end(), 0.0f);
    }

    enqueue(std::move(row), false);
    for (const auto& fault : faults) {
        const size_t joint_idx = fault.first;
        const uint8_t error_code = fault.second;
        std::ostringstream detail;
        detail << joint_names_[joint_idx] << ": " << dm_error_name(error_code)
               << " (0x" << std::hex << std::uppercase
               << static_cast<int>(error_code) << ")";
        enqueue(make_event_row("motor_fault", detail.str()), true);
    }
}

void RunLogger::record_event(const std::string& event,
                             const std::string& detail) {
    std::lock_guard<std::mutex> producer_lock(producer_mutex_);
    if (!active()) {
        return;
    }
    enqueue(make_event_row(event, detail), true);
}

std::filesystem::path RunLogger::current_path() const {
    std::lock_guard<std::mutex> lock(lifecycle_mutex_);
    return active_path_;
}

void RunLogger::writer_loop() {
    auto last_flush = std::chrono::steady_clock::now();
    while (true) {
        CsvRow row;
        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            queue_cv_.wait(lock, [this]() {
                return writer_stop_ || !queue_.empty();
            });
            if (writer_stop_ && queue_.empty()) {
                return;
            }
            row = std::move(queue_.front());
            queue_.pop_front();
            writer_busy_ = true;
        }

        if (stream_.is_open()) {
            write_row(row);
            const auto now = std::chrono::steady_clock::now();
            if (now - last_flush >= std::chrono::milliseconds(250)) {
                stream_.flush();
                last_flush = now;
            }
        }

        {
            std::lock_guard<std::mutex> lock(queue_mutex_);
            writer_busy_ = false;
            if (queue_.empty()) {
                drained_cv_.notify_all();
            }
        }
    }
}

bool RunLogger::enqueue(CsvRow row, bool must_write) {
    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        if (!must_write && queue_.size() >= kMaxQueuedRows) {
            dropped_csv_rows_.fetch_add(1, std::memory_order_relaxed);
            return false;
        }
        queue_.push_back(std::move(row));
    }
    queue_cv_.notify_one();
    return true;
}

void RunLogger::wait_until_drained() {
    std::unique_lock<std::mutex> lock(queue_mutex_);
    drained_cv_.wait(lock, [this]() {
        return queue_.empty() && !writer_busy_;
    });
}

RunLogger::CsvRow RunLogger::make_event_row(
    const std::string& event, const std::string& detail) const {
    CsvRow row;
    row.wall_time = std::chrono::system_clock::now();
    row.elapsed_s = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - start_steady_).count();
    row.event = event;
    row.detail = detail;
    return row;
}

void RunLogger::reset_aggregation() {
    std::lock_guard<std::mutex> telemetry_lock(telemetry_mutex_);
    has_telemetry_ = false;
    std::fill(peak_feedback_tau_.begin(), peak_feedback_tau_.end(), 0.0f);
    std::fill(peak_demand_tau_.begin(), peak_demand_tau_.end(), 0.0f);
    std::fill(last_fault_code_.begin(), last_fault_code_.end(), 0);
    pending_faults_.clear();
}

void RunLogger::recover_unclean_logs() {
    const std::string active_suffix = ".active.csv";
    for (const auto& entry : std::filesystem::directory_iterator(output_directory_)) {
        if (!entry.is_regular_file()) {
            continue;
        }
        const std::string name = entry.path().filename().string();
        if (!ends_with(name, active_suffix)) {
            continue;
        }
        std::string recovered_name = name;
        recovered_name.replace(recovered_name.size() - active_suffix.size(),
                               active_suffix.size(), ".unclean.csv");
        std::error_code ec;
        std::filesystem::rename(entry.path(),
                                entry.path().parent_path() / recovered_name, ec);
    }
}

void RunLogger::write_header() {
    stream_ << "timestamp,elapsed_s,row_type,event,event_detail,policy_step,"
               "cmd_vx_m_s,cmd_vy_m_s,cmd_yaw_rad_s,"
               "imu_roll_rad,imu_pitch_rad,imu_wx_rad_s,imu_wy_rad_s,"
               "imu_wz_rad_s,gravity_z,transition_phase,clamped_joint_count";
    for (const std::string& joint_name : joint_names_) {
        stream_ << ',' << joint_name << ".q_actual_rad"
                << ',' << joint_name << ".dq_actual_rad_s"
                << ',' << joint_name << ".q_target_rad"
                << ',' << joint_name << ".tau_feedback_peak_nm"
                << ',' << joint_name << ".tau_demand_peak_nm"
                << ',' << joint_name << ".tau_feedback_utilization"
                << ',' << joint_name << ".tau_demand_utilization"
                << ',' << joint_name << ".motor_temperature_c"
                << ',' << joint_name << ".mos_temperature_c"
                << ',' << joint_name << ".error_code";
    }
    stream_ << '\n';
}

void RunLogger::write_row(const CsvRow& row) {
    stream_ << csv_escape(format_wall_time(row.wall_time)) << ','
            << std::fixed << std::setprecision(6) << row.elapsed_s << ','
            << (row.is_sample ? "sample" : "event") << ','
            << csv_escape(row.event) << ',' << csv_escape(row.detail);
    if (!row.is_sample) {
        const size_t empty_columns = 12 + joint_names_.size() * 10;
        for (size_t i = 0; i < empty_columns; ++i) {
            stream_ << ',';
        }
        stream_ << '\n';
        return;
    }

    stream_ << ',' << row.policy_step
            << ',' << row.context.cmd_vx
            << ',' << row.context.cmd_vy
            << ',' << row.context.cmd_yaw
            << ',' << row.context.imu_roll
            << ',' << row.context.imu_pitch
            << ',' << row.context.imu_wx
            << ',' << row.context.imu_wy
            << ',' << row.context.imu_wz
            << ',' << row.context.gravity_z
            << ',' << row.context.transition_phase
            << ',' << row.context.clamped_joint_count;
    for (const JointRow& joint : row.joints) {
        stream_ << ',' << joint.q_actual
                << ',' << joint.dq_actual
                << ',' << joint.q_target
                << ',' << joint.tau_feedback_peak
                << ',' << joint.tau_demand_peak
                << ',' << joint.tau_feedback_utilization
                << ',' << joint.tau_demand_utilization
                << ',' << joint.motor_temperature
                << ',' << joint.mos_temperature
                << ',' << static_cast<int>(joint.error_code);
    }
    stream_ << '\n';
}

std::string RunLogger::csv_escape(const std::string& value) {
    if (value.find_first_of(",\"\n\r") == std::string::npos) {
        return value;
    }
    std::string escaped;
    escaped.reserve(value.size() + 2);
    escaped.push_back('"');
    for (char ch : value) {
        if (ch == '"') {
            escaped.push_back('"');
        }
        escaped.push_back(ch);
    }
    escaped.push_back('"');
    return escaped;
}

std::string RunLogger::format_wall_time(
    const std::chrono::system_clock::time_point& time_point) {
    const auto milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(
        time_point.time_since_epoch()) % 1000;
    const std::time_t raw_time = std::chrono::system_clock::to_time_t(time_point);
    std::tm local_time{};
    localtime_r(&raw_time, &local_time);
    std::ostringstream output;
    output << std::put_time(&local_time, "%Y-%m-%dT%H:%M:%S")
           << '.' << std::setw(3) << std::setfill('0') << milliseconds.count();
    return output.str();
}

std::string RunLogger::timestamp_for_filename() {
    const auto now = std::chrono::system_clock::now();
    const auto milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(
        now.time_since_epoch()) % 1000;
    const std::time_t raw_time = std::chrono::system_clock::to_time_t(now);
    std::tm local_time{};
    localtime_r(&raw_time, &local_time);
    std::ostringstream output;
    output << std::put_time(&local_time, "%Y%m%d_%H%M%S")
           << '_' << std::setw(3) << std::setfill('0') << milliseconds.count();
    return output.str();
}

std::string RunLogger::dm_error_name(uint8_t error_code) {
    switch (error_code) {
        case 0x08: return "over_voltage";
        case 0x09: return "under_voltage";
        case 0x0A: return "over_current";
        case 0x0B: return "mos_over_temperature";
        case 0x0C: return "coil_over_temperature";
        case 0x0D: return "lost_connection";
        case 0x0E: return "over_load";
        default: return "unknown_error";
    }
}

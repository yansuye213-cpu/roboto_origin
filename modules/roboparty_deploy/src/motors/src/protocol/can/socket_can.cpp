// SPDX-License-Identifier: GPL-3.0
// Copyright (C) 2025-2026 Luo1imasi
// Copyright (C) 2026 wentywenty

/**
 * @file
 * This file implements functions to receive
 * and transmit CAN frames via SocketCAN.
 */

#include "socket_can.hpp"
#include <chrono>
#include <spdlog/sinks/stdout_color_sinks.h>

// MotorsCAN static members
std::shared_ptr<MotorsCAN> MotorsCAN::get(const std::string& interface, const std::string& backend) {
    ensure_logger();
    if (backend == "socketcan") {
        return MotorsSocketCAN::get(interface);
    }
    throw std::runtime_error("Unknown CAN backend: " + backend);
}

// MotorsSocketCAN static members
std::unordered_map<std::string, std::shared_ptr<MotorsSocketCAN>> MotorsSocketCAN::instances_;

std::shared_ptr<MotorsSocketCAN> MotorsSocketCAN::get(const std::string& interface) {
    if (instances_.find(interface) == instances_.end()) instances_[interface] = createInstance(interface);
    return instances_[interface];
}

MotorsSocketCAN::MotorsSocketCAN(const std::string& interface)
    : interface_(interface), sockfd_(INIT_FD), receiving_(false), tx_queue_(TX_QUEUE_SIZE) {
    open(interface);
}

MotorsSocketCAN::~MotorsSocketCAN() { this->close(); }

int64_t MotorsSocketCAN::monotonic_time_ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

void MotorsSocketCAN::record_tx_timing(const CanTxQueueItem& item, int64_t write_ns,
                                       bool success, int write_errno) {
    std::lock_guard<std::mutex> lock(tx_timing_mutex_);
    if (tx_timing_events_.size() >= MAX_TIMING_EVENTS) {
        timing_event_drops_++;
        return;
    }
    tx_timing_events_.push_back({
        item.sequence, item.frame.can_id & CAN_SFF_MASK, item.enqueue_ns, write_ns,
        item.queue_depth, success, write_errno,
    });
}

void MotorsSocketCAN::record_rx_timing(const can_frame& frame, int64_t rx_ns) {
    std::lock_guard<std::mutex> lock(rx_timing_mutex_);
    if (rx_timing_events_.size() >= MAX_TIMING_EVENTS) {
        timing_event_drops_++;
        return;
    }
    rx_timing_events_.push_back({frame.can_id & CAN_SFF_MASK, rx_ns});
}

void MotorsSocketCAN::set_timing_enabled(bool enabled) {
    timing_enabled_ = false;
    if (!enabled) return;
    {
        std::lock_guard<std::mutex> lock(tx_timing_mutex_);
        tx_timing_events_.clear();
        tx_timing_events_.reserve(MAX_TIMING_EVENTS);
    }
    {
        std::lock_guard<std::mutex> lock(rx_timing_mutex_);
        rx_timing_events_.clear();
        rx_timing_events_.reserve(MAX_TIMING_EVENTS);
    }
    tx_sequence_ = 0;
    tx_queue_drops_ = 0;
    timing_event_drops_ = 0;
    timing_enabled_ = true;
}

CanTimingSnapshot MotorsSocketCAN::drain_timing() {
    CanTimingSnapshot snapshot;
    {
        std::lock_guard<std::mutex> lock(tx_timing_mutex_);
        snapshot.tx_events.swap(tx_timing_events_);
    }
    {
        std::lock_guard<std::mutex> lock(rx_timing_mutex_);
        snapshot.rx_events.swap(rx_timing_events_);
    }
    snapshot.tx_queue_drops = tx_queue_drops_.load();
    snapshot.timing_event_drops = timing_event_drops_.load();
    return snapshot;
}

void MotorsSocketCAN::open(const std::string& interface) {
    sockfd_ = socket(PF_CAN, SOCK_RAW, CAN_RAW);
    if (sockfd_ == INIT_FD) {
        logger_->error("Failed to create CAN socket");
        throw std::runtime_error("Failed to create CAN socket");
    }

    int bufsize = 1024 * 1024;  // 1MB
    setsockopt(sockfd_, SOL_SOCKET, SO_SNDBUF, &bufsize, sizeof(bufsize));

    strncpy(if_request_.ifr_name, interface.c_str(), IFNAMSIZ);
    if (ioctl(sockfd_, SIOCGIFINDEX, &if_request_) == -1) {
        logger_->error("Unable to detect CAN interface {}", interface);

        this->close();
        throw std::runtime_error("Unable to detect CAN interface " + interface);
    }

    // Bind the socket to the network interface
    addr_.can_family = AF_CAN;
    addr_.can_ifindex = if_request_.ifr_ifindex;
    int rc = ::bind(sockfd_, reinterpret_cast<struct sockaddr *>(&addr_), sizeof(addr_));
    if (rc == -1) {
        logger_->error("Failed to bind socket to network interface {}", interface);
        this->close();
        throw std::runtime_error("Failed to bind socket to network interface " + interface);
    }

    int flags = fcntl(sockfd_, F_GETFL, 0);
    if (flags == -1) {
        logger_->error("Failed to get socket flags");
        this->close();
        throw std::runtime_error("Failed to get socket flags");
    }
    if (fcntl(sockfd_, F_SETFL, flags | O_NONBLOCK) == -1) {
        logger_->error("Failed to set socket to non-blocking");
        this->close();
        throw std::runtime_error("Failed to set socket to non-blocking");
    }

    receiving_ = true;
    receiver_thread_ = std::thread([this]() {
        pthread_setname_np(pthread_self(), "can_rx");
        struct sched_param sp{}; sp.sched_priority = 80;
        if (pthread_setschedparam(pthread_self(), SCHED_FIFO, &sp) != 0) {
            logger_->error("Failed to set realtime priority for CAN RX thread");
        }

        int total_cores = std::thread::hardware_concurrency();
        if (total_cores == 0) total_cores = 4; // Fallback
        int cpu_id = total_cores - 1;

        char last_char = interface_.back();
        if (isdigit(last_char)) {
            int port_num = last_char - '0';
            cpu_id = total_cores - 1 - port_num;
            if (cpu_id < 0) cpu_id = 0; 
        }
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        CPU_SET(cpu_id, &cpuset);
        if (pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset) != 0) {
            logger_->error("Failed to bind CAN RX thread to Core {}", cpu_id);
        }

        fd_set descriptors;
        int maxfd = sockfd_;
        struct timeval timeout;
        can_frame rx_frame;

        while (receiving_) {
            FD_ZERO(&descriptors);
            FD_SET(sockfd_, &descriptors);

            timeout.tv_sec = TIMEOUT_SEC;
            timeout.tv_usec = TIMEOUT_USEC;

            int sel_ret = ::select(maxfd + 1, &descriptors, NULL, NULL, &timeout);
            if (sel_ret < 0) {
                if (errno == EINTR) continue;
                logger_->error("CAN select error: {}", strerror(errno));
                break;
            }
            if (sel_ret == 1) {
                while (true){
                    int len = ::read(sockfd_, &rx_frame, CAN_MTU);
                    if (len < 0) {
                        if (errno == EAGAIN || errno == EWOULDBLOCK) {
                            break; 
                        }
                        logger_->warn("CAN read error: {}", strerror(errno));
                        break;
                    }
                    if (len == 0){
                        break;
                    }
                    if (timing_enabled_) {
                        record_rx_timing(rx_frame, monotonic_time_ns());
                    }
                    CanCbkFunc callback_to_run;
                    {
                        std::lock_guard<std::mutex> lock(can_callback_mutex_);
                        CanCbkId key = key_extractor_(rx_frame);
                        auto it = can_callback_list_.find(key);
                        if (it != can_callback_list_.end()) {
                            callback_to_run = it->second;
                        }
                    }
                    if (callback_to_run) {
                        callback_to_run(rx_frame);
                    }
                }
            }
        }
    });

    sender_thread_ = std::thread([this]() {
        pthread_setname_np(pthread_self(), "can_tx");
        struct sched_param sp{}; sp.sched_priority = 80;
        if (pthread_setschedparam(pthread_self(), SCHED_FIFO, &sp) != 0) {
            logger_->error("Failed to set realtime priority for CAN TX thread");
        }

        int total_cores = std::thread::hardware_concurrency();
        if (total_cores == 0) total_cores = 4; // Fallback
        int cpu_id = total_cores - 1;
        
        char last_char = interface_.back();
        if (isdigit(last_char)) {
            int port_num = last_char - '0';
            cpu_id = total_cores - 1 - port_num;
            if (cpu_id < 0) cpu_id = 0; 
        }
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        CPU_SET(cpu_id, &cpuset);
        if (pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset) != 0) {
            logger_->error("Failed to bind CAN TX thread to Core {}", cpu_id);
        }

        CanTxQueueItem tx_item;
        while (receiving_) {
            {
                std::unique_lock<std::mutex> lock(tx_mutex_);
                tx_cv_.wait(lock, [this]() { return !tx_queue_.empty() || !receiving_; });
                if (!receiving_) break;
                if (!tx_queue_.pop(tx_item)) continue;
                if (tx_item.sequence != 0) tx_queue_depth_.fetch_sub(1);
            }
            int retry_count = 0;
            int write_errno = 0;
            bool write_success = false;
            int64_t write_ns = 0;
            while (retry_count < MAX_RETRY_COUNT) {
                const ssize_t written = ::write(sockfd_, &tx_item.frame, sizeof(can_frame));
                if (written == static_cast<ssize_t>(sizeof(can_frame))) {
                    write_ns = monotonic_time_ns();
                    write_success = true;
                    break;
                }
                write_errno = errno;
                retry_count += 1;
                if (retry_count < MAX_RETRY_COUNT) {
                    std::this_thread::sleep_for(std::chrono::microseconds(1000));
                }
            }
            if (tx_item.sequence != 0) {
                record_tx_timing(tx_item, write_ns, write_success, write_errno);
            }
            if (!write_success) {
                logger_->error("Failed to transmit CAN frame");
            } else if (send_sleep_us_ > 0) {
                std::this_thread::sleep_for(std::chrono::microseconds(send_sleep_us_));
            }
        }
    });
}

void MotorsSocketCAN::close() {
    receiving_ = false;
    tx_cv_.notify_one();
    if (receiver_thread_.joinable()) receiver_thread_.join();
    if (sender_thread_.joinable()) sender_thread_.join();

    if (sockfd_ != INIT_FD) {
        if (::close(sockfd_) < 0) {
            if (logger_) logger_->warn("Failed to close socket {}: {}", interface_, strerror(errno));
        } else {
            if (logger_) logger_->info("CAN interface {} closed successfully.", interface_);
        }
    }
    sockfd_ = INIT_FD;
}

void MotorsSocketCAN::transmit(const can_frame &frame) {
    if (sockfd_ == INIT_FD) {
        logger_->error("Unable to transmit: Socket not open");
        return;
    }
    CanTxQueueItem item;
    item.frame = frame;
    if (timing_enabled_) {
        item.sequence = tx_sequence_.fetch_add(1) + 1;
        item.enqueue_ns = monotonic_time_ns();
        item.queue_depth = tx_queue_depth_.fetch_add(1) + 1;
    }
    if (!tx_queue_.bounded_push(item)) {
        if (item.sequence != 0) {
            tx_queue_depth_.fetch_sub(1);
            tx_queue_drops_++;
            record_tx_timing(item, 0, false, ENOBUFS);
        }
        logger_->error("CAN TX queue full; frame dropped on {}", interface_);
        return;
    }
    tx_cv_.notify_one();
}

void MotorsSocketCAN::add_can_callback(const CanCbkFunc& callback, const CanCbkId id) {
    std::lock_guard<std::mutex> lock(can_callback_mutex_);
    can_callback_list_[id] = callback;
}

void MotorsSocketCAN::remove_can_callback(CanCbkId id) {
    std::lock_guard<std::mutex> lock(can_callback_mutex_);
    can_callback_list_.erase(id);
}

void MotorsSocketCAN::clear_can_callbacks() {
    std::lock_guard<std::mutex> lock(can_callback_mutex_);
    can_callback_list_.clear();
}

void MotorsSocketCAN::set_can_key_extractor(CanCbkKeyExtractor extractor) {
    std::lock_guard<std::mutex> lock(can_callback_mutex_);
    key_extractor_ = std::move(extractor);
}

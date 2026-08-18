// SPDX-License-Identifier: GPL-3.0
// Copyright (C) 2025-2026 Luo1imasi
// Copyright (C) 2026 wentywenty

/**
 * @file
 * This file declares an interface to SocketCAN,
 * to facilitates frame transmission and reception.
 */

#pragma once

#include "can_iso.hpp"

#include <linux/can.h>
#include <net/if.h>
#include <pthread.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>
#include <sys/ioctl.h>
#include <sys/socket.h>
#include <unistd.h>
#include <fcntl.h>

#include <atomic>
#include <boost/lockfree/queue.hpp>
#include <condition_variable>
#include <cstdbool>
#include <cstdio>
#include <cstring>
#include <cstdint>
#include <memory>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

constexpr const int INIT_FD = -1;
constexpr const int TIMEOUT_SEC = 0;
constexpr const int TIMEOUT_USEC = 1000;
constexpr const int TX_QUEUE_SIZE = 4096;
constexpr const int MAX_RETRY_COUNT = 3;
constexpr const size_t MAX_TIMING_EVENTS = 262144;

struct CanTxQueueItem {
    can_frame frame{};
    uint64_t sequence = 0;
    int64_t enqueue_ns = 0;
    uint32_t queue_depth = 0;
};

struct CanTxTimingEvent {
    uint64_t sequence = 0;
    uint32_t can_id = 0;
    int64_t enqueue_ns = 0;
    int64_t write_ns = 0;
    uint32_t queue_depth = 0;
    bool write_success = false;
    int write_errno = 0;
};

struct CanRxTimingEvent {
    uint32_t can_id = 0;
    int64_t rx_ns = 0;
};

struct CanTimingSnapshot {
    std::vector<CanTxTimingEvent> tx_events;
    std::vector<CanRxTimingEvent> rx_events;
    uint64_t tx_queue_drops = 0;
    uint64_t timing_event_drops = 0;
};

using LFQueue = boost::lockfree::queue<CanTxQueueItem, boost::lockfree::fixed_sized<true>>;

class MotorsSocketCAN : public MotorsCAN {
   private:
    std::string interface_;  // The network interface name
    int sockfd_ = -1;        // The file descriptor for the CAN socket
    std::atomic<bool> receiving_;
    LFQueue tx_queue_;
    std::atomic<uint32_t> tx_queue_depth_{0};
    std::mutex tx_mutex_;
    std::condition_variable tx_cv_;

    sockaddr_can addr_;      // The address of the CAN socket
    ifreq if_request_;       // The network interface request

    /// Receiving
    std::thread receiver_thread_;
    CanCbkMap can_callback_list_;
    std::mutex can_callback_mutex_;
    CanCbkKeyExtractor key_extractor_ = [](const can_frame &frame) -> CanCbkId {
        return static_cast<CanCbkId>(frame.can_id);
    };

    /// Transmitting
    std::thread sender_thread_;
    std::atomic<int> send_sleep_us_{0};

    /// Optional timing diagnostics. Disabled during normal robot operation.
    std::atomic<bool> timing_enabled_{false};
    std::atomic<uint64_t> tx_sequence_{0};
    std::atomic<uint64_t> tx_queue_drops_{0};
    std::atomic<uint64_t> timing_event_drops_{0};
    std::mutex tx_timing_mutex_;
    std::mutex rx_timing_mutex_;
    std::vector<CanTxTimingEvent> tx_timing_events_;
    std::vector<CanRxTimingEvent> rx_timing_events_;

    static int64_t monotonic_time_ns();
    void record_tx_timing(const CanTxQueueItem& item, int64_t write_ns,
                          bool success, int write_errno);
    void record_rx_timing(const can_frame& frame, int64_t rx_ns);

    MotorsSocketCAN(const std::string& interface);

    static std::shared_ptr<MotorsSocketCAN> createInstance(const std::string& interface) {
        return std::shared_ptr<MotorsSocketCAN>(new MotorsSocketCAN(interface));
    }
    static std::unordered_map<std::string, std::shared_ptr<MotorsSocketCAN>> instances_;

    void open(const std::string& interface);
    void close();

   public:
    MotorsSocketCAN(const MotorsSocketCAN &) = delete;
    MotorsSocketCAN &operator=(const MotorsSocketCAN &) = delete;
    ~MotorsSocketCAN() override;

    static std::shared_ptr<MotorsSocketCAN> get(const std::string& interface);

    void transmit(const can_frame& frame) override;
    void add_can_callback(const CanCbkFunc& callback, CanCbkId id) override;
    void remove_can_callback(CanCbkId id) override;
    void clear_can_callbacks() override;
    void set_can_key_extractor(CanCbkKeyExtractor extractor) override;

    void set_send_sleep(int us) { send_sleep_us_ = us; }

    void set_timing_enabled(bool enabled);
    CanTimingSnapshot drain_timing();

    /// SocketCAN-specific: direct socket fd access (for advanced use).
    int get_fd() const { return sockfd_; }
};

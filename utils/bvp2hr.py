#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：Non-contact-human-physiological-monitoring 
@File    ：bvp2hr.py
@IDE     ：PyCharm 
@Author  ：王洪飞
@Date    ：2025/12/29 17:47 
'''
import numpy as np
from scipy.signal import butter, filtfilt, find_peaks, welch
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import interp1d
import warnings


def butter_bandpass_filter(signal, lowcut, highcut, fs, order=4):
    """
    巴特沃斯带通滤波（增加参数校验，避免Wn超出0~1）
    :param signal: 输入信号数组
    :param lowcut: 低截止频率 (Hz)
    :param highcut: 高截止频率 (Hz)
    :param fs: 采样率 (Hz)
    :param order: 滤波器阶数
    :return: 滤波后的信号
    """
    # 校验采样率有效性
    if fs <= 0:
        raise ValueError(f"采样率fs={fs}无效，必须为正数！")
    nyq = 0.5 * fs  # 奈奎斯特频率

    # 修正截止频率：确保lowcut < highcut < nyq，且lowcut > 0
    lowcut = max(0.001, lowcut)  # 避免lowcut≤0
    highcut = min(nyq - 0.001, highcut)  # 避免highcut≥nyq
    if lowcut >= highcut:
        raise ValueError(f"低截止频率({lowcut}Hz) ≥ 高截止频率({highcut}Hz)，且需小于奈奎斯特频率({nyq}Hz)！")

    # 归一化截止频率（确保0 < Wn < 1）
    low = lowcut / nyq
    high = highcut / nyq

    b, a = butter(order, [low, high], btype='band')
    filtered_signal = filtfilt(b, a, signal)  # 零相位滤波
    return filtered_signal


def remove_baseline_drift(signal, fs, cutoff=0.5, order=2):
    """
    去除基线漂移（增加参数校验）
    :param signal: 输入信号数组
    :param fs: 采样率 (Hz)
    :param cutoff: 基线提取的低截止频率 (Hz)
    :param order: 滤波器阶数
    :return: 去除基线后的信号
    """
    if fs <= 0:
        raise ValueError(f"采样率fs={fs}无效，必须为正数！")
    nyq = 0.5 * fs

    # 修正截止频率：确保0 < cutoff < nyq
    cutoff = np.clip(cutoff, 0.001, nyq - 0.001)
    cutoff_norm = cutoff / nyq  # 归一化

    b, a = butter(order, cutoff_norm, btype='low')
    baseline = filtfilt(b, a, signal)
    return signal - baseline


import numpy as np
from scipy.signal import butter, filtfilt, find_peaks
from scipy.ndimage import uniform_filter1d


def robust_bvp_to_hr(bvp_signal, sampling_rate=25.0,
                     bandpass_low=0.7, bandpass_high=2.5,
                     min_heart_rate=40, max_heart_rate=180,
                     min_peak_prominence=0.1,
                     smooth_window=3,
                     verbose=False):
    """
    鲁棒的BVP到心率转换算法
    专为25FPS信号优化，兼顾稳定性和准确性
    """
    # -------------------- 1. 输入验证 --------------------
    if sampling_rate <= 0:
        raise ValueError(f"无效的采样率: {sampling_rate}")

    if len(bvp_signal) < 30:  # 至少1.2秒数据（25FPS）
        if verbose:
            print(f"信号过短: {len(bvp_signal)} 个点 (< 30)")
        return np.array([]), np.array([])

    # -------------------- 2. 信号预处理 --------------------
    try:
        # 2.1 去除基线漂移
        bvp_no_baseline = remove_baseline_drift(
            bvp_signal, sampling_rate, cutoff=0.5
        )

        # 2.2 自适应调整滤波参数
        # 确保不超过奈奎斯特频率
        nyquist = 0.5 * sampling_rate
        safe_highcut = min(bandpass_high, nyquist * 0.9)

        # 带通滤波
        bvp_filtered = butter_bandpass_filter(
            bvp_no_baseline,
            lowcut=bandpass_low,
            highcut=safe_highcut,
            fs=sampling_rate,
            order=4
        )

        # 2.3 归一化（使用稳健的方法）
        signal_median = np.median(bvp_filtered)
        signal_iqr = np.percentile(bvp_filtered, 75) - np.percentile(bvp_filtered, 25)

        if signal_iqr < 1e-6:  # 避免除零
            if verbose:
                print("信号变化太小")
            return np.array([]), np.array([])

        bvp_normalized = (bvp_filtered - signal_median) / signal_iqr

    except Exception as e:
        if verbose:
            print(f"预处理失败: {e}")
        return np.array([]), np.array([])

    # -------------------- 3. 峰值检测 --------------------
    # 计算最小峰值距离（基于最高心率）
    min_distance_samples = int((60 / max_heart_rate) * sampling_rate)
    min_distance_samples = max(min_distance_samples, 3)  # 至少3个采样点

    # 计算自适应阈值
    signal_mean = np.mean(bvp_normalized)
    signal_std = np.std(bvp_normalized)

    # 如果信号太"平"，降低阈值要求
    if signal_std < 0.3:
        height_threshold = signal_mean + 0.05
    else:
        height_threshold = signal_mean + 0.15 * signal_std

    try:
        # 检测峰值
        peaks, properties = find_peaks(
            bvp_normalized,
            height=height_threshold,
            distance=min_distance_samples,
            prominence=min_peak_prominence
        )
    except Exception as e:
        if verbose:
            print(f"峰值检测失败: {e}")
        return np.array([]), np.array([])

    if len(peaks) < 3:  # 至少需要3个峰值
        if verbose:
            print(f"峰值数量不足: {len(peaks)} (< 3)")
        return np.array([]), np.array([])

    # -------------------- 4. 计算心率 --------------------
    peak_times = peaks / sampling_rate
    rr_intervals = np.diff(peak_times)
    hr_raw = 60 / rr_intervals

    # -------------------- 5. 异常值过滤 --------------------
    # 5.1 基于生理范围过滤
    valid_mask = (hr_raw >= min_heart_rate) & (hr_raw <= max_heart_rate)

    if np.sum(valid_mask) < 2:  # 需要至少2个有效心率点
        if verbose:
            print(f"有效心率点不足: {np.sum(valid_mask)} (< 2)")
        return np.array([]), np.array([])

    hr_filtered = hr_raw[valid_mask]
    peak_times_filtered = peak_times[1:][valid_mask]

    # 5.2 基于相邻心率变化的过滤
    if len(hr_filtered) >= 3:
        hr_changes = np.abs(np.diff(hr_filtered))
        median_change = np.median(hr_changes)

        # 过滤变化过大的点（超过中位数变化的3倍）
        if median_change > 0:  # 避免除零
            change_ratios = hr_changes / median_change
            stable_mask = change_ratios < 3.0
            stable_mask = np.concatenate([[True, True], stable_mask])[:len(hr_filtered)]

            hr_filtered = hr_filtered[stable_mask]
            peak_times_filtered = peak_times_filtered[stable_mask]

    if len(hr_filtered) < 2:
        if verbose:
            print(f"过滤后心率点不足: {len(hr_filtered)}")
        return np.array([]), np.array([])

    # -------------------- 6. 平滑处理 --------------------
    if len(hr_filtered) > smooth_window:
        hr_smoothed = uniform_filter1d(hr_filtered, size=smooth_window, mode='mirror')
    else:
        hr_smoothed = hr_filtered

    # -------------------- 7. 输出统计（可选） --------------------
    if verbose and len(hr_smoothed) > 0:
        mean_hr = np.mean(hr_smoothed)
        std_hr = np.std(hr_smoothed)
        median_hr = np.median(hr_smoothed)

        print(f"心率统计: {mean_hr:.1f} ± {std_hr:.1f} BPM (中位数: {median_hr:.1f} BPM)")
        print(f"峰值数量: {len(peaks)}, 有效心率点: {len(hr_smoothed)}")

    return peak_times_filtered, hr_smoothed
# 信号插值增强版本（提高25FPS信号的时间分辨率）
def enhance_bvp_temporal_resolution(bvp_signal, original_fps=25, target_fps=100):
    """
    通过插值增强BVP信号的时间分辨率
    用于改善低采样率信号的峰值检测精度
    """
    if original_fps >= target_fps:
        return bvp_signal, original_fps

    # 原始时间轴
    original_time = np.arange(len(bvp_signal)) / original_fps

    # 目标时间轴
    target_length = int(len(bvp_signal) * target_fps / original_fps)
    target_time = np.linspace(original_time[0], original_time[-1], target_length)

    # 三次样条插值
    interp_func = interp1d(original_time, bvp_signal, kind='cubic')
    enhanced_signal = interp_func(target_time)

    return enhanced_signal, target_fps


# 完整的心率计算流程（包含信号增强）
def calculate_heart_rate_from_bvp(bvp_signal, sampling_rate=25.0,
                                  enhance_resolution=True,
                                  min_segment_length=5.0,  # 最小分析段长度（秒）
                                  verbose=False):
    """
    从BVP信号计算心率的完整流程
    """
    # 检查信号长度是否足够
    required_points = int(min_segment_length * sampling_rate)
    if len(bvp_signal) < required_points:
        if verbose:
            print(f"信号长度不足: {len(bvp_signal)} < {required_points}")
        return None, None, None

    # 可选：增强时间分辨率
    if enhance_resolution and sampling_rate < 50:
        bvp_enhanced, enhanced_fps = enhance_bvp_temporal_resolution(
            bvp_signal,
            target_fps=100
        )
        if verbose:
            print(f"信号分辨率从 {sampling_rate}FPS 增强到 {enhanced_fps}FPS")
    else:
        bvp_enhanced = bvp_signal
        enhanced_fps = sampling_rate

    # 计算心率
    hr_times, hr_values = robust_bvp_to_hr(
        bvp_enhanced,
        sampling_rate=enhanced_fps,
        verbose=verbose
    )

    if len(hr_values) == 0:
        return None, None, None

    # 计算统计信息
    hr_mean = np.mean(hr_values)
    hr_std = np.std(hr_values)
    hr_median = np.median(hr_values)

    if verbose:
        print(f"心率统计: {hr_mean:.1f} ± {hr_std:.1f} BPM (中位数: {hr_median:.1f} BPM)")

    return hr_times, hr_values, {
        'mean': hr_mean,
        'std': hr_std,
        'median': hr_median,
        'min': np.min(hr_values),
        'max': np.max(hr_values),
        'num_points': len(hr_values)
    }


# 批量处理函数
def process_bvp_in_windows(bvp_signal, sampling_rate=25.0,
                           window_seconds=10.0, overlap=0.5,
                           enhance_resolution=True):
    """
    滑动窗口处理BVP信号，计算连续心率
    """
    window_points = int(window_seconds * sampling_rate)
    step_points = int(window_points * (1 - overlap))

    if len(bvp_signal) < window_points:
        return [], []

    all_hr_times = []
    all_hr_values = []

    for start in range(0, len(bvp_signal) - window_points + 1, step_points):
        end = start + window_points
        window_signal = bvp_signal[start:end]

        # 计算窗口中心时间
        window_center_time = (start + end) / (2 * sampling_rate)

        # 计算心率
        hr_times, hr_values, stats = calculate_heart_rate_from_bvp(
            window_signal,
            sampling_rate=sampling_rate,
            enhance_resolution=enhance_resolution,
            verbose=False
        )

        if hr_values is not None and len(hr_values) > 0:
            # 使用窗口中心时间作为该段心率的参考时间
            avg_hr = np.mean(hr_values)
            all_hr_times.append(window_center_time)
            all_hr_values.append(avg_hr)

    return np.array(all_hr_times), np.array(all_hr_values)
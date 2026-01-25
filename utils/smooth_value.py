from collections import deque
from math import sqrt as m_sqrt
import numpy as np


def filter_by_mad(data: list, target_count=80):
    """
    使用MAD（中位数绝对偏差）方法，对异常值最鲁棒
    """
    data = np.array(data)
    if len(data) <= target_count:
        return np.mean(data)

    # 计算中位数绝对偏差
    median_val = np.median(data)
    mad = np.median(np.abs(data - median_val))

    # 计算标准化距离（使用MAD）
    if mad > 0:
        mad_distances = np.abs(data - median_val) / mad
    else:
        mad_distances = np.abs(data - median_val)

    # 选择距离最小的target_count个点
    sorted_indices = np.argsort(mad_distances)
    selected_indices = sorted_indices[:target_count]
    filtered_data = data[selected_indices]

    return np.mean(filtered_data)


def find_max_count(data):
    max_count = max([data.count(i) for i in set(data)])
    max_number = [x for x in set(data) if data.count(x) == max_count]
    return max_number[0]


def smooth_temp_list(data: deque, temp_min, temp_max) -> float:
    """
    对温度列表进行异常值的剔除
    """
    # ===================== 测试 临时环境 =======================

    # # 首先 剔除温度过低的 低于36.的 默认为错误数据
    data = [i if i >= temp_min else temp_min for i in data]
    # 剔除温度过高的数据 剔除大于 37.5的数据
    data = [i for i in data if i <= temp_max]
    # real_temp = find_max_count(data)    # 中位数 为温度
    if len(data) == 0:
        return 0
    real_temp = filter_by_mad(data, max(1, max(1, len(data) // 4)))
    return real_temp
    # return sum(data)/len(data)
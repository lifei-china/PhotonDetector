import numpy as np
import matplotlib.pyplot as plt
 

def add_gaussian_pulse(signal, t0, amplitude, fwhm, dt):
    """
    在 signal 上叠加一个以 t0 为中心的高斯脉冲。
    参数:
    --------
    signal : np.ndarray
        要叠加的目标信号数组（原地修改）
    t0 : float
        脉冲中心时间 (s)
    amplitude : float
        脉冲峰值幅度（高斯峰值）
    fwhm : float
        脉冲全宽半高 (Full Width at Half Maximum, s)
    dt : float
        采样时间步长 (s)
    """
    # FWHM 转换为 sigma
    sigma = fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))

    # 取 ±3σ 的高斯窗口（超过这个范围的值可以忽略）
    half = int(np.ceil(3.0 * sigma / dt))
    tw = (np.arange(-half, half + 1) * dt)

    # 峰值归一化（最大值=1），面积随 σ 改变
    g = np.exp(-0.5 * (tw / sigma) ** 2)

    # 找到中心索引并写入信号
    ic = int(round(t0 / dt))
    i0, i1 = ic - half, ic + half + 1
    j0, j1 = max(0, i0), min(len(signal), i1)
    if j0 < j1:
        gj0 = j0 - i0
        gj1 = gj0 + (j1 - j0)
        signal[j0:j1] += amplitude * g[gj0:gj1]


def generate_signal(spectrum_bins, spectrum_pdf, rate, T_total=400e-6,
                    pulse_width=25e-9, dt=1e-9, gain=1.0,
                    pulse_shape='rect',          # 'rect' 或 'gauss'
                    ):
    """"
    | 参数        | 含义                                                  | 典型值          
    -----------   | -----------------------------------------------------  
    spectrum_bins | 能谱的能量区间(单位 keV),例如 [0, 1, 2, ..., 100]      | 实验能谱能量边界       
    spectrum_pdf  | 每个能量区间对应的概率或强度(未必归一化)                | 来自实验或模拟的能谱     
    rat           | 入射光子计数率(单位:counts/s)                          | 1e5 ~ 1e7     
    T_total       | 采样积分总时间(秒)                                     | 400e-6 (400 μs)   
    pulse_width   | 每个脉冲宽度(秒)                                       | 25e-9 (25 ns) 
    dt            | 采样时间步长(秒)                                       | 1e-9 (1 ns)   
    gain          | 幅度放大系数，用于将能量转成信号幅度                    | 可设 1.0       
    """
    # 归一化谱
    pdf = spectrum_pdf / np.sum(spectrum_pdf)
    # pdf = spectrum_pdf
    cdf = np.cumsum(pdf)
    bin_centers = (spectrum_bins[:-1] + spectrum_bins[1:]) / 2
 

    # 泊松过程生成到达时间
    t, times = 0.0, []      # t 表示当前的时间; times 保存每个光子事件的到达时刻。
    # 循环执行， 得到从 0 到 T_total 内所有的随机到达时间。
    while t < T_total:
        # “模拟一次新的光子到达” → “时间向前跳一个随机的间隔 Δt”。
        # np.random.exponential(1.0 / rate) 为一个随机时间间隔 Δt
        t += np.random.exponential(1.0 / rate)  # rate 光子到达的平均时间间隔
        if t < T_total:
            times.append(t)
    times = np.array(times)
 

    # 根据谱抽样能量
    u = np.random.rand(len(times))  # 随机均匀地抽取 N 个光子，赋予一个随机数，这些随机数在0-1之间
    idx = np.searchsorted(cdf, u)   # 


    # energies = bin_centers[np.clip(idx, 0, len(bin_centers) - 1)]
    # amplitudes = gain * energies
    # 假设你有 bin_edges 和为每个事件找到的 bin 索引 idx            # second method  
    left  = spectrum_bins[idx]                                    # second method
    right = spectrum_bins[idx + 1]                                # second method
    energies = left + np.random.rand(len(idx)) * (right - left)  # second method # 在区间内均匀抽样
    amplitudes = gain * energies                                  # second method    


    # 生成时间轴与信号
    t_axis = np.arange(0, T_total, dt)
    signal = np.zeros_like(t_axis)

    # pulse_samples = int(pulse_width / dt)   # 每个脉冲持续 25 个采样时间点
    # for ti, Ai in zip(times, amplitudes):
    #     i0 = int(ti / dt)
    #     i1 = min(i0 + pulse_samples, len(signal))
    #     signal[i0:i1] += Ai


    if pulse_shape == 'rect':
        pulse_samples = max(1, int(round(pulse_width / dt)))
        for ti, Ai in zip(times, amplitudes):
            i0 = int(ti / dt)
            i1 = min(i0 + pulse_samples, len(signal))
            if i0 < len(signal):
                signal[i0:i1] += Ai

    elif pulse_shape == 'gauss':
        for ti, Ai in zip(times, amplitudes):
            add_gaussian_pulse(signal, ti, Ai, pulse_width, dt)
    else:
        raise ValueError("pulse_shape must be 'rect' or 'gauss'.")

 
    return t_axis, signal, times, amplitudes



def count_crossings(sig, thr ):
    """计算信号从 <thr 到 >=thr 的上升沿个数（事件数）"""
    above = sig >=  (thr )           #找出大于阈值的值
    # 上升沿：当前为 True 且前一采样为 False
    rise = above & np.roll(~above, 1)   # 检测为上升沿的值
    # 防止首样本被误计
    rise[0] = False
 
    return int(rise.sum())

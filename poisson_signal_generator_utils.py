import numpy as np
import matplotlib.pyplot as plt
import math 

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


def generate_signal(spectrum_bins, spectrum_pdf, photon_rate, T_total=400e-6,
                    pulse_width=15e-9, dt=1e-9, g_mat = 1, 
                    noise_ENC = 0, Cf = 40e-15,
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
    g_mat         | 1 keV 的光子产生的 电子数                               | 可设 1.0     
    noise_ENC     | 芯片的等效噪声电荷数
    Cf            | 典型的 CdTe/CZT PCD 前端电容量级                        | 40e-15 F        
  
    """
    # 1）归一化谱
    pdf = spectrum_pdf / np.sum(spectrum_pdf) 
    cdf = np.cumsum(pdf) 
 

    # 2）泊松过程生成到达时间
    t, times = 0.0, []      # t 表示当前的时间; times 保存每个光子事件的到达时刻。
    # 循环执行， 得到从 0 到 T_total 内所有的随机到达时间。
    while t < T_total:
        # “模拟一次新的光子到达” → “时间向前跳一个随机的间隔 Δt”。
        #  这里，np.random.exponential(1.0 / photon_rate) 为一个随机时间间隔 Δt
        t += np.random.exponential(1.0 / photon_rate)  # photon_rate 光子到达的平均时间间隔
        if t < T_total:
            times.append(t)
    times = np.array(times)
 

    # 3）根据谱抽样能量
    u = np.random.rand(len(times))  # 随机均匀地抽取 N(len(times)) 个光子，赋予一个随机数，这些随机数在0-1之间
    idx = np.searchsorted(cdf, u)   # 通过u里的随机数，与cdf表匹配，生成idx，这是不同的能量区间


    # 4）对能谱做更细的抽样，得到连续谱，而非阶梯状的谱        
    # spectrum_bins[idx] 是一个数组，是前面生成的 N 个光子的能量，但是是整数。
    # 接下来我们得到一个更连续的能量值数组          
    left  = spectrum_bins[idx] - 0.5                             
    right = spectrum_bins[idx] + 0.5
    # 在区间内均匀抽样，np.random.rand( len(idx))生成 一个 1 以内 的浮点随机数。中心值加这个随机数，细化能量谱                               
    energies = left + np.random.rand( len(idx)) * (right - left)  # energies 是一个数组，是被细化后的，更连续的，N个光子的能量值
    
    Ne = energies * g_mat  
    Ne_noisy = Ne  + np.random.normal(0, noise_ENC, size=Ne.shape)     # kv能量的光子在晶体中产生相应的电子数 
    Ne_noisy = np.maximum(Ne_noisy, 0.0)

    amplitudes = Ne_noisy * 1.602e-19 / Cf     # 1.602e-19 为单个电子的电荷量
    print(Ne)
    print(Ne_noisy)
    print(amplitudes)
                              
    # 5）生成时间轴与信号
    t_axis = np.arange(0, T_total, dt)
    signal = np.zeros_like(t_axis)
 
     # 6）生成 方波 或 高斯波 时间序列
    if pulse_shape == 'rect':
        pulse_samples = max(1, int(round(pulse_width / dt)))
        print('pulse_samples:', pulse_samples)

        for ti, Ai in zip(times, amplitudes):
            i0 = int(ti / dt)           # 在帧率 dt 的刻度下，于i0个点作为起始开始采集点
            i1 = min(i0 + pulse_samples, len(signal))   # 采样 i0 + pulse_samples 个点，pulse_samples 是脉宽
            if i0 < len(signal):
                signal[i0:i1] += Ai

    elif pulse_shape == 'gauss':
        for ti, Ai in zip(times, amplitudes):
            add_gaussian_pulse(signal, ti, Ai, pulse_width, dt)
    else:
        raise ValueError("pulse_shape must be 'rect' or 'gauss'.")
 
 
    return t_axis, signal, times, amplitudes




def make_threshold_grid_from_energy(bins, gain=1.0, n_thr=256, vmin=None, vmax=None):
    """
    bins: 能量边界 (keV), 长度 N+1
    gain: 幅度= gain * 能量。若 gain=1 → 阈值轴单位为 keV 等效；否则为 Volt
    vmin/vmax: 阈值最小/最大（与幅度同单位）。若不提供，自动按能量范围映射。
    """
    E_min, E_max = bins[0], 256 #bins[-1]
    if vmin is None: vmin = max(0.0, gain * E_min)
    if vmax is None: vmax = gain * E_max
    return np.linspace(vmin, vmax, n_thr)

import numpy as np

def build_energy_bins_and_pdf(energies, values, normalize=True):
    """
    根据能量点和强度值构建能量边界 bins 与归一化概率密度 pdf。

    参数
    ----
    energies :  ndarray 能量采样点（单位 keV），长度 N
    values :    ndarray 每个能量点的光谱强度（任意单位），长度 N
    normalize : bool,   可选  是否对 pdf 归一化为 1，默认为 True

    返回  
    bins : ndarray 能量边界数组（长度 N+1）
    pdf :  ndarray 归一化后的概率密度（长度 N）
    dE :   ndarray 每个 bin 的能量宽度
    """

    # ---------- 计算能量边界 ----------
    mid = (energies[:-1] + energies[1:]) / 2
    bins = np.empty(len(energies) + 1)
    bins[1:-1] = mid
    bins[0] = energies[0] - (mid[0] - energies[0])
    bins[-1] = energies[-1] + (energies[-1] - mid[-1])

    # ---------- 计算 bin 宽度 ----------
    dE = np.diff(bins)

    # ---------- 计算 pdf ----------
    pdf = values * dE
    if normalize:
        pdf = pdf / np.sum(pdf)

    return bins, pdf, dE


def count_crossings(sig, thr, eps=1e-12 ):
    """计算信号从 <thr 到 >=thr 的上升沿个数（事件数）"""
    above = sig >=  (thr + eps)          #找出大于阈值的值
    # 上升沿：当前为 True 且前一采样为 False
    rise = above & np.roll(~above, 1)    # 检测为上升沿的值
    # 防止首样本被误计
    rise[0] = False
 
    return int(rise.sum())


def spectrum_from_signal(sig, thr_grid): 
    integral_counts = np.array(
        [count_crossings(sig, thr) for thr in thr_grid], dtype=float
        )
    differential = -np.gradient(integral_counts, thr_grid)
    return integral_counts, differential
 
def plot_signals_over_time(
    t, signals, *,
    max_plot=None,        # 限制绘制次数（防止太密）
    alpha=0.4,            # 单条曲线透明度 
    zoom=None,            # 时间窗口 (start_us, end_us)
    figsize=(10, 4)       # 图大小
):
    """
    绘制多次模拟信号随时间的叠加图，可选显示平均波形，并支持放大时间窗口。

    参数
    ----
    t : ndarray
        时间轴 (单位: 秒)
    signals : list[ndarray]
        多次模拟信号，例如 [sig1, sig2, sig3, ...]
    max_plot : int, 可选
        仅绘制前 max_plot 条曲线
    alpha : float, 可选
        线条透明度
    show_avg : bool, 可选
        是否绘制平均波形
    zoom : tuple(float, float), 可选
        时间窗口，单位 μs，例如 zoom=(5.0, 5.5)
    figsize : tuple, 可选
        图形大小
    """

    n_total = len(signals)
    if max_plot is not None:
        signals = signals[:max_plot]

    # 单位转换
    t_us = t * 1e6

    # 如果指定 zoom，裁剪时间区间
    if zoom is not None:
        t_start_us, t_end_us = zoom
        mask = (t_us >= t_start_us) & (t_us <= t_end_us)
        t_us = t_us[mask]
        signals = [sig[mask] for sig in signals]

    # 绘制
    plt.figure(figsize=figsize)
    for i, sig in enumerate(signals):
        plt.plot(t_us, sig, alpha=alpha, lw=1, label=f'Run {i+1}' if i < 5 else None)
 

    plt.xlabel('Time (μs)')
    plt.ylabel('Signal Amplitude (arb. unit)')
    title = f'Signal Waveforms ({n_total} runs)'
    if zoom:
        title += f' — Zoomed: {zoom[0]:.2f}–{zoom[1]:.2f} μs'
    plt.title(title)
    plt.grid(alpha=0.3)
    if len(signals) <= 5 :
        plt.legend(loc="upper right")
    plt.tight_layout()
    plt.show()

def plot_signals_grid(
    t, signals, *,
    max_plot=9,         # 最多显示几条信号（默认9）
    zoom=None,          # 时间窗口 (start_us, end_us) 
    figsize=(12, 8)     # 画布大小
):
    """
    在一个画布上以子图形式绘制多条信号（每条单独一个 subplot）

    参数
    ----
    t : ndarray
        时间轴 (单位: 秒)
    signals : list[ndarray]
        多次模拟信号，例如 [sig1, sig2, ...]
    max_plot : int
        最多绘制的信号数量（避免太多）
    zoom : tuple(float, float)
        放大显示的时间范围 (μs)
    show_avg : bool
        是否叠加平均波形
    avg_color : str
        平均波形颜色
    figsize : tuple
        整个画布大小
    """

    # 限制数量
    n_show = min(max_plot, len(signals))
    t_us = t * 1e6

    # 如果 zoom，则裁剪区间
    if zoom is not None:
        t_start_us, t_end_us = zoom
        mask = (t_us >= t_start_us) & (t_us <= t_end_us)
        t_us = t_us[mask]
        signals = [sig[mask] for sig in signals[:n_show]]
    else:
        signals = signals[:n_show]

    # 自动计算子图布局（尽量接近正方形）
    # n_cols = int(np.ceil(np.sqrt(n_show)))
    n_cols = 1
    n_rows = int(np.ceil(n_show / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, sharex=True, sharey=True)
    axes = np.ravel(axes)
 

    # 绘制每个信号
    for i, (ax, sig) in enumerate(zip(axes, signals)):
        ax.plot(t_us, sig, lw=1, alpha=0.8, color='steelblue') 
        ax.set_title(f'Run {i+1}', fontsize=10)
        ax.grid(alpha=0.3)

    # 去掉多余子图
    for ax in axes[n_show:]:
        ax.axis('off')

    # 统一坐标
    fig.text(0.5, 0.04, 'Time (μs)', ha='center')
    fig.text(0.04, 0.5, 'Amplitude (arb. unit)', va='center', rotation='vertical')
    fig.suptitle(f'Signal Waveforms ({n_show} runs)', fontsize=14)
    plt.tight_layout(rect=[0.04, 0.04, 1, 0.95])
    plt.show()


 

def report_source_max(amps, gain=1.0):
    amax = float(np.max(amps)) if len(amps) else 0.0
    print(f"[源最大幅度] max(amps) = {amax:.3f}  (单位 = {'keV等效' if gain==1.0 else 'Volt'})")
    return amax

def contiguous_segments(mask):
    """把布尔掩码里 True 的连续区间分段，返回 [(start_idx, end_idx), ...]（end为开区间）"""
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return []
    diff = np.diff(mask.astype(int))
    starts = np.where(diff == 1)[0] + 1
    ends   = np.where(diff == -1)[0] + 1
    if mask[0]:  starts = np.r_[0, starts]
    if mask[-1]: ends   = np.r_[ends, len(mask)]
    return list(zip(starts, ends))

def gaussian_sigma_from_fwhm(fwhm):
    # FWHM = 2*sqrt(2 ln2)*sigma
    return fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))

def analyze_peaks_above_threshold(
    t, sig, times, amps, *,
    threshold=120.0,
    pulse_width=25e-9,
    pulse_shape='rect',        # 'rect' 或 'gauss'（高斯默认 FWHM = pulse_width）
    top_k=5,
    gain=1.0
):
    """
    找到所有超过阈值的峰；对每个峰统计：峰时刻、峰值、贡献的事件数量与幅度。
    返回一个列表，每个元素为字典。
    """
    t = np.asarray(t)
    sig = np.asarray(sig)

    # 1) 找到超过阈值的连续片段，并取各片段的峰与峰时刻
    above = sig > threshold
    segs = contiguous_segments(above)
    peak_infos = []
    for s, e in segs:
        if e - s <= 0: 
            continue
        i_peak_local = s + int(np.argmax(sig[s:e]))
        t_peak = t[i_peak_local]
        v_peak = float(sig[i_peak_local])

        # 2) 计算贡献窗口（矩形取 ±pulse_width/2；高斯取 ±3σ）
        if pulse_shape == 'gauss':
            sigma = gaussian_sigma_from_fwhm(pulse_width)
            half_window = 3.0 * sigma
        else:
            half_window = 0.5 * pulse_width

        # 3) 统计窗口内的到达事件
        in_win = (times >= (t_peak - half_window)) & (times <= (t_peak + half_window))
        contrib_idx = np.where(in_win)[0]
        contrib_amps = amps[contrib_idx] if len(contrib_idx) else np.array([])
        n_contrib = int(len(contrib_idx))
        single_max = float(contrib_amps.max()) if n_contrib else 0.0

        peak_infos.append({
            "t_peak": t_peak,
            "v_peak": v_peak,
            "n_contrib": n_contrib,
            "single_max_amp": single_max,
            "contrib_indices": contrib_idx,
            "contrib_amps": contrib_amps
        })

    # 4) 只打印最显著的 top_k 个（按峰值倒序）
    peak_infos_sorted = sorted(peak_infos, key=lambda x: x["v_peak"], reverse=True)[:top_k]

    # 5) 控制台报告
    print(f"\n[超过阈值 {threshold} 的峰统计] 共 {len(peak_infos)} 个片段，展示 top {len(peak_infos_sorted)}：")
    for j, info in enumerate(peak_infos_sorted, 1):
        us = info["t_peak"] * 1e6
        print(f"  #{j}: t_peak={us:.3f} μs, v_peak={info['v_peak']:.3f}, "
              f"n_contrib={info['n_contrib']}, single_max_amp={info['single_max_amp']:.3f}")

    return peak_infos_sorted

def quick_overlap_metrics(times, pulse_width, rate=None):
    times = np.sort(times)
    frac = float(np.mean(np.diff(times) < pulse_width)) if len(times) > 1 else 0.0
    if rate is not None:
        print(f"[λτ 与重叠比例]  λτ≈{rate*pulse_width:.3g}， 经验 overlap_frac≈{frac:.3f}")
    else:
        print(f"[重叠比例]  overlap_frac≈{frac:.3f}")
    return frac


# ---------------------------------------------------------
# 以下是读取不同管压和滤片生成的光谱的数据
# ---------------------------------------------------------

# ---------------------------------------------------------
# 工具函数：自动读取两列数字（SpekCalc 与 XCOM CSV 通用）
# ---------------------------------------------------------
def load_two_cols(path):
    E, Y = [], []
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        for ln in f:
            s = ln.strip().split()
            if len(s) < 2:
                continue
            try:
                e = float(s[0]); y = float(s[1])
            except ValueError:
                continue
            E.append(e); Y.append(y)
    E = np.asarray(E, float); Y = np.asarray(Y, float)
    idx = np.argsort(E)
    return E[idx], Y[idx]


# ---------------------------------------------------------
# ⭐ 主函数：光谱衰减计算
# ---------------------------------------------------------
def compute_filtered_spectrum(
    spek_path,
    xcom_filter_path,
    density,
    thickness_mm_list,
    plot=True
):
    """
    参数:
        spek_path:          spekcalc 导出的光谱 txt
        xcom_filter_path:   滤片材料的 (mu/rho) CSV
        density:            滤片密度 [g/cm^3]
        thickness_mm_list:  滤片厚度 (可单值 或 list)
        plot:               是否绘制光谱图
    
    返回:
        E_keV:             能量 (keV)
        Phi_in:            原始谱
        Phi_out_list:      对应每个厚度的衰减光谱(list)
    """

    # ------------------ 1. 读光源光谱 ------------------
    E_keV, Phi_in = load_two_cols(spek_path)

    # ------------------ 2. 读 XCOM μ/ρ ------------------
    Ex_MeV, mu_over_rho = load_two_cols(xcom_filter_path)
    Ex_keV = Ex_MeV * 1000.0  # MeV → keV

    # μ/ρ 插值到 spek 能量上
    mu_rho_interp = np.interp(
        E_keV, Ex_keV, mu_over_rho,
        left=mu_over_rho[0], right=mu_over_rho[-1]
    )

    # μ(E) = (μ/ρ) × 密度
    mu_cm_inv = mu_rho_interp * density

    # 若 thickness_mm_list 是单值 → 转换成 list
    if not isinstance(thickness_mm_list, (list, tuple, np.ndarray)):
        thickness_mm_list = [thickness_mm_list]

    Phi_out_list = []

    # ------------------ 3. 计算每个厚度的衰减谱 ------------------
    for t_mm in thickness_mm_list:
        x_cm = t_mm * 0.1    # mm → cm
        T = np.exp(-mu_cm_inv * x_cm)
        Phi_out = Phi_in * T
        Phi_out_list.append(Phi_out)

    # ------------------ 4. 是否绘图 ------------------
    if plot:
        plt.figure(figsize=(10, 6))
        plt.plot(E_keV, Phi_in, 'k-', linewidth=2, label="Source Spectrum")

        colors = ['red','blue','green','orange','purple','cyan']
        styles = ['-','--','-.',':','-', '--']

        for i, (t_mm, Phi_out) in enumerate(zip(thickness_mm_list, Phi_out_list)):
            plt.plot(E_keV, Phi_out,
                     color=colors[i % len(colors)],
                     linestyle=styles[i % len(styles)],
                     linewidth=1.8,
                     label=f"Filter {t_mm} mm")

        plt.xlabel("Energy (keV)")
        plt.ylabel("Relative Intensity (a.u.)")
        plt.title("Filtered X-ray Spectrum")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    return E_keV, Phi_in, Phi_out_list

 

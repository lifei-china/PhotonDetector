import numpy as np
import matplotlib.pyplot as plt
import math 
from pathlib import Path
e_charge = 1.602e-19


def make_threshold_grid_from_LSB(a_keV_per_LSB,
                                 b_keV,
                                 g_mat,
                                 Cf,
                                 n_thr=256,
                                 clip_E_min=None):
    """
    根据像素的能量标定 (a,b)、材料转换系数 g_mat、反馈电容 Cf，
    把 LSB -> E_thr -> N_e_thr -> V_thr，生成真正的“阈值电压”数组。

    返回:
        thr_grid_V : 形状 (n_thr,) 的数组，单位 V
        lsb_values : 对应的 LSB 0..n_thr-1
        E_thr_keV  : 对应的阈值能量 (keV)，方便后续画图或分析
    """
    lsb_values = np.arange(n_thr)  # 0,1,...,255

    # 1) LSB -> 能量 (keV)
    E_thr_keV = a_keV_per_LSB * lsb_values + b_keV

    # 2) 能量 -> 电子数
    Ne_thr = g_mat * E_thr_keV  # g_mat = 1000/W, W=4.5 eV

    # 3) 电子数 -> 阈值电压 (V)
    thr_grid_V = Ne_thr * e_charge / Cf

    return thr_grid_V, lsb_values, E_thr_keV
 


def add_baseline_noise(signal, noise_ENC, Cf, dt, tau_n):
    sigma_V = noise_ENC * e_charge / Cf
    alpha = np.exp(-dt / tau_n)
    sigma_w = sigma_V * np.sqrt(1 - alpha**2)

    noise = np.zeros_like(signal)
    for i in range(1, len(signal)):
        noise[i] = alpha * noise[i-1] + sigma_w * np.random.randn()
    return signal + noise



# ----------------------------
# 1) Poisson 到达时间  
# ----------------------------
def poisson_arrivals(photon_rate, T_total, rng=np.random):
    t = 0.0     # t 表示当前的时间; 
    times = []  # times 保存每个光子事件的到达时刻。
    while t < T_total:
        # “模拟一次新的光子到达” → “时间向前跳一个随机的间隔 Δt”。
        #  这里，np.random.exponential(1.0 / photon_rate) 为一个随机时间间隔 Δt
        t += rng.exponential(1.0 / photon_rate)  # photon_rate 光子到达的平均时间间隔
        if t < T_total:
            times.append(t)
    return np.array(times)

# ----------------------------
# 2）根据谱抽样能量
# ----------------------------
def sample_energies_from_pdf(spectrum_bins, spectrum_pdf, n_events):
    pdf = spectrum_pdf / np.sum(spectrum_pdf)
    cdf = np.cumsum(pdf)

    u = np.random.rand(n_events)    # 随机均匀地抽取 N(len(times)) 个光子，赋予一个随机数，这些随机数在0-1之间
    idx = np.searchsorted(cdf, u)   # 通过u里的随机数，与cdf表匹配，生成idx，这是不同的能量区间

    # 对能谱做更细的抽样，得到连续谱，而非阶梯状的谱        
    # spectrum_bins[idx] 是一个数组，是前面生成的 N 个光子的能量，但是是整数。
    # 接下来我们得到一个更连续的能量值数组
    left = spectrum_bins[idx] - 0.5 
    right = spectrum_bins[idx] + 0.5
    # 在区间内均匀抽样，np.random.rand( len(idx))生成 一个 1 以内 的浮点随机数。中心值加这个随机数，细化能量谱
    E = left + np.random.rand(n_events) * (right - left)
    return E

# ----------------------------
# 3) 积分节点：电荷注入到 Cf + 指数释放（τ_decay）
#    Vint[n] = alpha*Vint[n-1] + (Q_inj[n]/Cf)
# ----------------------------
def integrate_with_decay(event_times, event_volt_steps, t_axis, dt, tau_decay):
    """
    event_volt_steps: 每个事件在积分节点造成的瞬时电压阶跃 ΔV = Q/Cf (单位 V)
    """
    n = len(t_axis)
    alpha = np.exp(-dt / tau_decay)  # 衰减系数
    Vint = np.zeros(n, dtype=float)

    # 把事件投到采样点上（同一个采样点可叠加多个事件）
    inj = np.zeros(n, dtype=float)
    idx = (event_times / dt).astype(int)   # 将事件时间轴上，以dt为间隔，得到一个整数的离散化的序列号索引
    idx = idx[(idx >= 0) & (idx < n)]
    for i, dv in zip(idx, event_volt_steps[:len(idx)]):
        inj[i] += dv

    for k in range(1, n):
        Vint[k] = alpha * Vint[k - 1] + inj[k]
    return Vint
 
# ----------------------------
# 4) shaper（整形滤波器）CR-RC
#     
# ----------------------------
def cr_filter(x, dt, tau_cr):
    """一阶高通（CR），去除 DC / 基线漂移"""
    a = np.exp(-dt / tau_cr)
    y = np.zeros_like(x, dtype=float)
    for n in range(1, len(x)):
        y[n] = a * (y[n-1] + x[n] - x[n-1])
    return y

def rc_filter(x, dt, tau_rc):
    """一阶低通（RC），平滑成钟形"""
    a = np.exp(-dt / tau_rc)
    y = np.zeros_like(x, dtype=float)
    for n in range(1, len(x)):
        y[n] = a * y[n-1] + (1 - a) * x[n]
    return y

def shaper_cr_rc(vint, dt, tau_cr, tau_rc):
    """CR-RC: 高通后低通"""
    return rc_filter(cr_filter(vint, dt, tau_cr), dt, tau_rc)

def measure_fwhm(t, y):
    y = y - np.min(y)
    if np.max(y) <= 0:
        return np.nan
    half = 0.5 * np.max(y)
    idx = np.where(y >= half)[0]
    if len(idx) < 2:
        return np.nan
    return t[idx[-1]] - t[idx[0]]

def impulse_response_fwhm(dt, tau_cr, tau_rc, n=2000):
    # 对单位冲激响应求整形后的 FWHM
    x = np.zeros(n); x[0] = 1.0
    y = shaper_cr_rc(x, dt, tau_cr, tau_rc)
    t = np.arange(n) * dt
    return measure_fwhm(t, y)

def tune_tau_rc_for_fwhm(dt, target_fwhm=15e-9, tau_cr=2e-9, lo=0.5e-9, hi=20e-9, iters=30):
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        f = impulse_response_fwhm(dt, tau_cr, mid)
        if np.isnan(f):
            lo = mid
            continue
        if f < target_fwhm:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)



def generate_signal(spectrum_bins, spectrum_pdf, photon_rate, T_total=400e-6,
                    pulse_width=15e-9, dt=1e-9, g_mat = 1, 
                    noise_ENC = 0, Cf = 40e-15,
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

  
    # 1）泊松过程生成到达时间 
    event_times = poisson_arrivals(photon_rate, T_total)
    n_events = len(event_times)

    # 2）归一化谱 根据谱抽样能量 
    energies = sample_energies_from_pdf(spectrum_bins, spectrum_pdf, n_events)

    # 3) 对应生成的电子数、电荷量、电压
    Ne = energies * g_mat   
    Q_charge = Ne * e_charge
    V_pulse = Q_charge / Cf         # Cf 为反馈电容

    # 4）积分节点动力学（指数释放）
    t_axis = np.arange(0, T_total, dt)
    Vint = integrate_with_decay(event_times, V_pulse, t_axis, dt, tau_decay=150e-9 )
 
    tau_cr = 2e-9       # 经验参数，待确认
    tau_rc = tune_tau_rc_for_fwhm(dt, target_fwhm= pulse_width, tau_cr=tau_cr)
    signal = shaper_cr_rc(Vint, dt, tau_cr, tau_rc)
    print("tau_rc tuned to:", tau_rc, "s")

    signal = add_baseline_noise(signal, noise_ENC, Cf, dt, tau_n=pulse_width)
  

    return t_axis, signal, event_times, Vint


def plot_vint_prefilter(
    t_axis, Vint,
    t0=5.0e-6,
    t1=6.3e-6, 
):
    """
    Plot integrator node voltage Vint(t) BEFORE shaping / filtering.
    Top: full time window
    Bottom: zoomed-in window [t0, t1]
    Parameters
    ----------
    t_axis : ndarray
        Time axis [s]
    Vint : ndarray
        Integrator voltage [V]
    t0, t1 : float
        Zoom window start/end [s]
    save_path : str or Path or None
        If given, save figure
    show : bool
        Whether to plt.show()
    """

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1,
        figsize=(10, 6),
        sharex=False
    )

    # =========================
    # Top panel: full waveform
    # =========================
    ax_top.plot(t_axis * 1e6, Vint * 1e3, lw=1.2)
    ax_top.set_xlabel("Time (µs)")
    ax_top.set_ylabel("Vint (mV)")
    ax_top.set_title("Integrator Node Voltage Vint(t) — Pre-filter")
    ax_top.grid(alpha=0.3)

    # =========================
    # Bottom panel: zoomed view
    # =========================
    mask = (t_axis >= t0) & (t_axis <= t1)
    ax_bot.plot(
        (t_axis[mask] - t0) * 1e9,
        Vint[mask] * 1e3,
        lw=1.5
    )
    ax_bot.set_xlabel("Time relative to t₀ (ns)")
    ax_bot.set_ylabel("Vint (mV)")
    ax_bot.set_title(
        f"Zoomed view (pre-filter): {t0*1e6:.1f}–{t1*1e6:.1f} µs"
    )
    ax_bot.grid(alpha=0.3)

    fig.tight_layout()
  
    return fig


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


def spectrum_from_signal(sig, thr_grid, lsb_values=None): 
    integral_counts = np.array(
        [count_crossings(sig, thr) for thr in thr_grid], dtype=float
        )
    if lsb_values is None:
        # 默认当 thr_grid 等间隔时
        differential = -np.gradient(integral_counts)
    else:
        # 用 LSB 作为自变量，而不是 thr_grid（V）
        differential = -np.gradient(integral_counts, lsb_values)
    # differential = -np.gradient(integral_counts, thr_grid)
    return integral_counts, differential



def plot_signals_over_time(
    t, signals, *,
    max_plot=None,        # 限制绘制次数（防止太密）
    alpha=0.4,            # 单条曲线透明度
    zoom=None,            # 时间窗口 (start_us, end_us)
    figsize=(10, 4),      # 图大小
    save_path=None,       # 保存路径（str / Path / None）
    show=True             # 是否显示
    ):
    """
    绘制多次模拟信号随时间的叠加图，并可选择保存到文件。

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
    zoom : tuple(float, float), 可选
        时间窗口，单位 μs，例如 zoom=(5.0, 5.5)
    figsize : tuple, 可选
        图形大小
    save_path : str or Path, 可选
        若提供，则保存图片到该路径
    show : bool, 可选
        是否调用 plt.show()
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
    fig = plt.figure(figsize=figsize)
    for i, sig in enumerate(signals):
        plt.plot(
            t_us, sig,
            alpha=alpha,
            lw=1,
            label=f'Run {i+1}' if i < 5 else None
        )

    plt.xlabel('Time (μs)')
    plt.ylabel('Signal Amplitude (arb. unit)')
    title = f'Signal Waveforms ({n_total} runs)'
    if zoom:
        title += f' — Zoomed: {zoom[0]:.2f}–{zoom[1]:.2f} μs'
    plt.title(title)
    plt.grid(alpha=0.3)

    if len(signals) <= 5:
        plt.legend(loc="upper right")

    plt.tight_layout()

    # ✅ 保存
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    # ✅ 显示 or 关闭
    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig


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
        fig = plt.figure(figsize=(10, 6)) 
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
        plt.close(fig)

    return E_keV, Phi_in, Phi_out_list, fig

 

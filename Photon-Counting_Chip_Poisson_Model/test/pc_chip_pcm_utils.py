import yaml
import json
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import math 
from pathlib import Path
from scipy.ndimage import gaussian_filter1d
e_charge = 1.602e-19


# -------------------------
#  1） 参数配置
# -------------------------
def load_yaml_config(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found:\n{path.resolve()}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def make_run_dir(results_root: str, exp_name: str) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = Path(results_root) / exp_name / ts
    out.mkdir(parents=True, exist_ok=True)
    return out

def save_run_artifacts(out: Path, cfg_dict: dict, summary: dict):
    # 1) 配置快照
    with open(out / "config_used.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg_dict, f, allow_unicode=True, sort_keys=False)

    # 2) 摘要信息
    with open(out / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # 3)  
    with open(out / "summary.txt", "w", encoding="utf-8") as f:
        for k, v in summary.items():
            f.write(f"{k}: {v}\n")

def postprocess(differential_avg, energy_axis, smooth_sigma: float):
    secd = -np.gradient(differential_avg, energy_axis)
    secd_smooth = gaussian_filter1d(secd, sigma=smooth_sigma)
    return secd, secd_smooth


#   求取电路参数：t_rc, k_shaper
def init_frontend_shaper_params(
    dt,
    pulse_width,
    tau_decay,
    n=8000,
):
    """
    - 自动求 tau_rc（满足目标 FWHM）
    - 用“单位 CSA 衰减”计算 k_shaper（与 dt 仅有数值误差关系）
    """

    # 1) 根据目标脉宽，确定 RC 时间常数
    tau_rc = tune_tau_rc_for_fwhm_pz_rc(
        dt,
        target_fwhm=pulse_width,
        tau_decay=tau_decay
    )

    # 2) 构造单位幅度的 CSA 输出（V0 = 1）
    t = np.arange(n) * dt
    vint_unit = np.exp(-t / tau_decay)   # 等效于 Q/Cf = 1
    # vint_unit = np.ones_like(t)
    print("max_vint_unit", np.max(vint_unit))

    # 3) 通过你当前的整形链
    v_pz = pzc_from_vint(vint_unit, dt, tau_decay=tau_decay)
    y = rc_filter(v_pz, dt, tau_rc)

    # 4) 峰值即为 k_shaper
    k_shaper = float(np.max(y))

    return {
        "tau_rc": tau_rc,
        "k_shaper": k_shaper,
    }





# -------------------------
# 2) 光子生成
# -------------------------

def load_two_cols(path):    # 自动读取两列数字（SpekCalc 与 XCOM CSV 通用）
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
#   光谱衰减计算
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
        plt.rcParams.update({
        "font.size": 16,
        "axes.labelsize": 18,
        "axes.titlesize": 20,
        "legend.fontsize": 14,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
    })
        fig = plt.figure(figsize=(10, 6)) 
        plt.plot(E_keV, Phi_in, 'k-', linewidth=2, label="Source Spectrum")

        colors = ['red','blue','green','orange','purple','cyan']
        styles = ['-','--','-.',':','-', '--']

        for i, (t_mm, Phi_out) in enumerate(zip(thickness_mm_list, Phi_out_list)):
            plt.plot(E_keV, Phi_out,
                     color=colors[i % len(colors)],
                     linestyle=styles[i % len(styles)],
                     linewidth=1.8,
                     label=f"Filter GOS {t_mm} mm")

        plt.xlabel("Energy (keV)")
        plt.ylabel(r"$N\;(\mathrm{keV\;cm^2\;mAs})^{-1}$ @ 1 m")
        # plt.ylabel("Relative Intensity (a.u.)")
        plt.title("Filtered X-ray Spectrum@ 120 kVp")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        plt.close(fig)

    return E_keV, Phi_in, Phi_out_list, fig

 

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

def load_spectrum_dict(spec_cfg: dict, source_cfg: dict, base_dir: Path):
    spek_path = base_dir / spec_cfg["spek_path"]
    xcom_path = base_dir / spec_cfg["xcom_filter_path"]
 
    if not spek_path.exists():
        raise FileNotFoundError(f"spek_path not found: {spek_path}")
    if not xcom_path.exists():
        raise FileNotFoundError(f"xcom_filter_path not found: {xcom_path}")

    energies, Phi_in, Phi_out_list, fig_spec = compute_filtered_spectrum(
        spek_path=str(spek_path),
        xcom_filter_path=str(xcom_path),
        density=spec_cfg["density"],
        thickness_mm_list=spec_cfg["thickness_mm_list"],
        plot=spec_cfg.get("plot_filtered_spectrum", False),
    )

    values = Phi_out_list[spec_cfg["thickness_index"]]
    bins, pdf, dE = build_energy_bins_and_pdf(energies, values)

    photon_rate = compute_photon_rate_dict(values, dE, source_cfg)
    # lambda_tau = photon_rate * cfg["chip"]["pulse_s"]
    # print(f"photon_rate (lambda) = {photon_rate:.6g} [1/s]")
    # print(f"lambda*tau = {lambda_tau:.6g} (dimensionless)")


    return energies, Phi_in, values, photon_rate, bins, pdf, dE, fig_spec


def compute_photon_rate_dict(values, dE, src_cfg: dict):
    pixel_length_cm = src_cfg["pixel_length_m"] * 100.0
    A_cm2 = pixel_length_cm ** 2
    lam = np.sum(values * dE * src_cfg["current_mA"] * (1.0 / src_cfg["distance_m"]) ** 2 * A_cm2)
    return lam

# ----------------------------
#   - 1. Poisson 到达时间  
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
#   -2. 根据谱抽样能量
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
# 3) 探测器响应 CSA
#    积分节点：电荷注入到 Cf + 指数释放（τ_decay）
#    Vint[n] = alpha*Vint[n-1] + (Q_inj[n]/Cf)
# ----------------------------

def integrate_with_decay(event_times, event_volt_steps, t_axis, dt, tau_decay,
                         tau_rise=3e-9):
    """
    event_volt_steps: 每个事件在积分节点造成的“总电压增量” ΔV_total = Q/Cf (单位 V)
    tau_decay: 积分节点释放时间常数
    tau_rise : 注入/电荷收集的上升时间常数（让初期更平滑）
    """
    n = len(t_axis)
    alpha = np.exp(-dt / tau_decay)  # 衰减系数
    Vint = np.zeros(n, dtype=float)

    # ---- 关键改动：构造一个“快速上升”的离散注入核，使总面积=1 ----
    L = max(1, int(np.ceil(5 * tau_rise / dt)))        # 核长度（~5个tau_rise）
    k = np.arange(L)
    kernel = np.exp(-k * dt / tau_rise)                # 从大到小的“注入尾巴”
    kernel = kernel / kernel.sum()                     # 归一化：sum=1

    inj = np.zeros(n, dtype=float)
    idx = (event_times / dt).astype(int)
    valid = (idx >= 0) & (idx < n)
    idx = idx[valid]
    dvs = np.asarray(event_volt_steps, dtype=float)[valid]

    # ---- 关键改动：把每个 dv 分摊到接下来 L 个采样点里 ----
    for i0, dv in zip(idx, dvs):
        j1 = min(n, i0 + L)
        inj[i0:j1] += dv * kernel[:j1 - i0]

    for k in range(1, n):
        Vint[k] = alpha * Vint[k - 1] + inj[k]
    return Vint



# ----------------------------
# 4) shaper（整形滤波器）CR-RC
# ----------------------------
# Pole-Zero Cancellation (PZC)

def pzc_from_vint(vint, dt, tau_decay):
    """
    精确离散 PZC：v_pz[n] = vint[n] - k*vint[n-1], k=exp(-dt/tau_decay)
    对应把 CSA 的极点 (1/(1-k z^-1)) 抵消掉。
    """
    k = np.exp(-dt / tau_decay)
    v_pz = np.zeros_like(vint, dtype=float)
    v_pz[0] = vint[0]
    for n in range(1, len(vint)):
        v_pz[n] = vint[n] - k * vint[n-1]
    return v_pz 
 
def rc_filter(x, dt, tau_rc):
    a = np.exp(-dt / tau_rc)
    y = np.zeros_like(x, dtype=float)
    for n in range(1, len(x)):
        y[n] = a * y[n-1] + (1 - a) * x[n]
    return y

def measure_fwhm_peak(t, y, tail_frac=0.2, baseline_mode="median", interp=True):
    t = np.asarray(t, float)
    y = np.asarray(y, float)

    k = max(5, int(len(y) * tail_frac))
    tail = y[-k:]
    baseline = np.median(tail) if baseline_mode == "median" else np.mean(tail)
    yy = y - baseline

    imax = int(np.argmax(yy))
    peak = yy[imax]
    if peak <= 0:
        return np.nan
    half = 0.5 * peak

    iL = imax
    while iL > 0 and yy[iL] >= half:
        iL -= 1
    iR = imax
    while iR < len(yy)-1 and yy[iR] >= half:
        iR += 1
    if iL == imax or iR == imax:
        return np.nan

    def cross_time(i1, i2):
        y1, y2 = yy[i1], yy[i2]
        t1, t2 = t[i1], t[i2]
        if y2 == y1:
            return t1
        return t1 + (half - y1) * (t2 - t1) / (y2 - y1)

    if interp:
        tL = cross_time(iL, iL+1)
        tR = cross_time(iR-1, iR)
    else:
        tL = t[iL+1]
        tR = t[iR-1]

    if tR <= tL:
        return np.nan
    return tR - tL

def impulse_response_fwhm_pz_rc(dt, tau_decay, tau_rc, n=4000):
    # 单事件：在 t=0 注入单位电压台阶（幅度无关，只看宽度）
    t = np.arange(n) * dt
    event_times = np.array([0.0])
    V_pulse = np.array([1.0])

    Vint = integrate_with_decay(event_times, V_pulse, t, dt, tau_decay=tau_decay)

    v_pz = pzc_from_vint(Vint, dt, tau_decay)
    y = rc_filter(v_pz, dt, tau_rc)

    return measure_fwhm_peak(t, y)

def tune_tau_rc_for_fwhm_pz_rc(dt, target_fwhm, tau_decay, lo=0.5e-9, hi=100e-9, iters=40):
    for _ in range(iters):
        mid = 0.5*(lo + hi)
        f = impulse_response_fwhm_pz_rc(dt, tau_decay, mid)
        if np.isnan(f):
            lo = mid
            continue
        if f < target_fwhm:
            lo = mid
        else:
            hi = mid
    return 0.5*(lo + hi)


# ----------------------------



# -------------------------
# 5) 能量标定与阈值比较
# -------------------------
def make_threshold_grid_from_LSB(  a_keV_per_LSB,  b_keV,  g_mat,
    Cf, n_thr=256,  clip_E_min=None,    k_shaper=1.0, 
):
    """
    生成阈值数组，默认返回“比较器看到的阈值”（即 shaper 输出端等效阈值）。

    物理链路：
      LSB(code) -> E_thr(keV) -> Ne_thr -> Q_thr -> Vstep_CSA = Q_thr/Cf
      然后用 k_shaper 把 Vstep_CSA 映射到 shaper 输出端阈值： Vthr_shaper

    返回:
      thr_grid      : (n_thr,) 阈值数组（与 sig 同单位；如果 sig 是电压则单位 V）
      lsb_values    : 0..n_thr-1
      E_thr_keV     : 对应能量阈值 (keV)
      thr_csa_Vstep : CSA 节点阶跃幅度 (V)，用于debug/检查
    """
    lsb_values = np.arange(n_thr, dtype=float)  # 0..255

    
    E_thr_keV = a_keV_per_LSB * lsb_values + b_keV  # 1) LSB -> 能量 (keV)
    if clip_E_min is not None:
        E_thr_keV = np.maximum(E_thr_keV, clip_E_min)

    
    Ne_thr = g_mat * E_thr_keV  # 2) 能量 -> 电子数
    Q_thr = Ne_thr * e_charge   # 3) 电子数 -> CSA 阶跃电压 (V)
    thr_csa_Vstep = Q_thr / Cf
    thr_grid = k_shaper * thr_csa_Vstep # 4) 映射到 shaper 输出端等效阈值

    return thr_grid, lsb_values.astype(int), E_thr_keV 

 
def build_threshold_grid_dict(chip_cfg: dict, calib_cfg: dict, k_shaper=1.0):
    g_mat = 1000.0 / chip_cfg["W_eV"]
    thr_grid, lsb_values, E_thr_keV = make_threshold_grid_from_LSB(
        a_keV_per_LSB=calib_cfg["a_keV_per_LSB"],
        b_keV=calib_cfg["b_keV"],
        g_mat=g_mat,
        Cf=chip_cfg["Cf_F"],
        n_thr=calib_cfg.get("n_thr", 256),    
        clip_E_min=None,
        k_shaper=k_shaper,          # CSA阶跃 -> shaper峰值 的幅度系数
    ) 

    return thr_grid, lsb_values, E_thr_keV


def count_crossings(sig, thr, eps=1e-12):
    """上升沿过阈计数（below -> above）"""
    above = sig >= (thr + eps)
    rise = above[1:] & (~above[:-1])   # 不用 roll
    return int(np.count_nonzero(rise))

def spectrum_from_signal(sig, thr_grid, x_axis=None):
    """
    sig: 1D shaped waveform
    thr_grid: thresholds in same unit as sig (e.g. V or arb unit)
    x_axis: 用来做微分的自变量（通常就是 thr_grid；如果你要 keV，就传 keV 轴）
    """
    integral_counts = np.array([count_crossings(sig, thr) for thr in thr_grid], dtype=float)

    if x_axis is None:
        x_axis = thr_grid

    differential = -np.gradient(integral_counts)#, x_axis)
    return integral_counts, differential
 




# -------------------------
# 6) 探测器主体
# -------------------------

def generate_signal(spectrum_bins, spectrum_pdf, photon_rate, T_total=400e-6,
                    pulse_width=15e-9, dt=1e-9, g_mat = 1, 
                    noise_ENC = 0, Cf = 40e-15, tau_rc = 1
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

    debug_plot_energy_hist(energies)    # 画出采样得到的能谱图
 

    # 3) 对应生成的电子数、电荷量、电压
    Ne = energies * g_mat   
    Q_charge = Ne * e_charge
    #  在事件电荷上加 （等效输入电荷噪声）
    sigma_Q = noise_ENC * e_charge
    Q_charge_noisy = Q_charge + sigma_Q * np.random.randn(len(Q_charge))


    V_pulse = Q_charge_noisy / Cf         # Cf 为反馈电容

    # 4）积分节点动力学（指数释放）
    t_axis = np.arange(0, T_total, dt)
    Vint = integrate_with_decay(event_times, V_pulse, t_axis, dt, tau_decay=150e-9 )        # measurement 书籍里，tau_decay ~ 50us，不过这个pile up 会比较严重。150ns相对轻一些。
 


    # tau_cr = 2e-9       # 经验参数，待确认
    # tau_rc = tune_tau_rc_for_fwhm(dt, target_fwhm= pulse_width, tau_cr=tau_cr)
    # # signal = shaper_cr_rc(Vint, dt, tau_cr, tau_rc)
    # signal = shaper_pzc_cr_rc(Vint, dt, tau_decay=150e-9, tau_cr=tau_cr, tau_rc=tau_rc, pzc_gain=5.5)
 
    v_pz = pzc_from_vint(Vint, dt, tau_decay=150e-9)
    signal = rc_filter(v_pz, dt, tau_rc)


    print("tau_rc tuned to:", tau_rc, "s")
    peak = signal.max()
    print("signal max =", peak, "min =", signal.min())
    print("p1,p50,p99 =", np.percentile(signal, [1,50,99]))

  

    return t_axis, signal, event_times, Vint







# -------------------------
# 7) 画图区域
# -------------------------



def debug_plot_energy_hist(
    energies,
    out_root="Photon-Counting_Chip_Poisson_Model/results",
    bins=200,
    fname="energy_hist.png",
):
    """
    Debug: plot and save sampled photon energy histogram.
    """

    energies = np.asarray(energies)

    debug_dir = Path(out_root) / "DEBUG"
    debug_dir.mkdir(parents=True, exist_ok=True)

    debug_path = (debug_dir / fname).resolve()
    print("[DEBUG] energy_hist path =", debug_path)

    # 哨兵文件：证明函数被调用
    (debug_dir / "ENERGY_HIST_REACHED.txt").write_text(
        "reached\n", encoding="utf-8"
    )

    fig, ax = plt.subplots(1, 1, figsize=(7, 4))
    ax.hist(energies, bins=bins)
    ax.set_xlabel("Photon energy (keV)")
    ax.set_ylabel("Counts")
    ax.set_title("Sampled photon energies")
    fig.tight_layout()

    fig.savefig(debug_path, dpi=150)
    plt.close(fig)

    print(
        "[DEBUG] energy_hist saved =",
        debug_path.exists(),
        "size =",
        debug_path.stat().st_size if debug_path.exists() else None,
    )

    return debug_path



def plot_vint_prefilter(t_axis, Vint,   t0=5.0e-6,  t1=6.3e-6, ):
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
        (t_axis[mask]) * 1e6,
        Vint[mask] * 1e3,
        lw=1.5
    )
    ax_bot.set_xlabel("Time (µs)")
    ax_bot.set_ylabel("Vint (mV)")
    ax_bot.set_title(
        f"Zoomed view (pre-filter): {t0*1e6:.1f}–{t1*1e6:.1f} µs"
    )
    ax_bot.grid(alpha=0.3)

    fig.tight_layout()
  
    return fig



def plot_signals_over_time(
    t, signals, *,
    max_plot=None,        # 限制绘制次数（防止太密）
    alpha=0.4,            # 单条曲线透明度
    zoom=None,            # 时间窗口 (start_us, end_us)
    figsize=(10, 4),      # 图大小
    save_path=None,       # 保存路径（str / Path / None）
    show=False             # 是否显示
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



def save_or_show_fig(fig, save_path=None, show=False, dpi=150):
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)



def plot_and_save_dict(out: Path, cfg: dict, energy_axis, integral_avg, differential_avg, secd, secd_smooth, t, signals):
    # 信号叠加图（如果 utils 内部直接 show，那就只负责 show，不强制保存）
    if cfg["run"].get("plot_signals", True):
        plot_signals_over_time(
            t, signals,
            max_plot=cfg["run"].get("max_plot_signals", 5),
            zoom=tuple(cfg["run"].get("zoom", (5.0, 5.2)))
        )

    # 三联图保存
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 10), sharex=True)

    ax1.plot(energy_axis, integral_avg, lw=1.8)
    ax1.set_ylabel("Integral counts")
    ax1.set_title("Integral Spectrum")

    ax2.plot(energy_axis, differential_avg, lw=1.8, label="Differential (-dN/dT)")
    ax2.set_ylabel("Counts per keV")
    ax2.set_title("Differential Spectrum")
    ax2.set_ylim(-100, 100)
    ax2.legend()

    ax3.plot(energy_axis, secd, lw=1.0, label="raw")
    ax3.plot(energy_axis, secd_smooth, lw=1.8, label=f"Gaussian σ={cfg['run']['smooth_sigma']}")
    ax3.set_ylabel("2nd-derivative proxy")
    ax3.set_title("Second Derivative (smoothed)")
    ax3.set_ylim(-100, 30)
    ax3.legend()

    for ax in (ax1, ax2, ax3):
        ax.set_xticks(np.arange(0, 300, 20))
    ax3.set_xlabel("Threshold / Energy (keV)")

    fig.tight_layout()
    fig.savefig(out / "spectra.png", dpi=150, bbox_inches="tight")

    if cfg["run"].get("show_plots", False):
        plt.show()
    else:
        plt.close(fig)
 

def save_all_figures_and_plots(
    out: Path,
    cfg: dict,
    *,
    t,
    signals,
    debug_figs: dict,
    fig_spec=None,
):
    """
    统一保存/显示本次 run 的所有图。
    - out: 本次运行输出目录
    - cfg: 总配置 dict（用到 cfg["run"] 的开关）
    - t, signals: 波形时间轴与波形集合
    - debug_figs: 调试图 dict，例如 {"Vint_prefilter": fig, ...}
    - fig_spec: 光谱图 figure（可为 None）
    """

    show_plots = cfg["run"].get("show_plots", False)

    # 1) Vint 预滤波图
    fig_vint = debug_figs.get("Vint_prefilter")
    if fig_vint is not None:
        save_or_show_fig(
            fig_vint,
            save_path=out / "Vint_prefilter.png",
            # show=show_plots
        )

    # 2) 滤片/光谱图
    if fig_spec is not None:
        save_or_show_fig(
            fig_spec,
            save_path=out / "filtered_spectrum.png",
            # show=show_plots
        )

    # 3) 信号波形（zoom）
    plot_signals_over_time(
        t,
        signals,
        max_plot=cfg["run"].get("max_plot_signals", 5),
        zoom=tuple(cfg["run"].get("zoom", ())),
        save_path=out / "signals_zoom.png",
        # show=show_plots
    )

    # 4) 信号波形（full）
    plot_signals_over_time(
        t,
        signals,
        max_plot=cfg["run"].get("max_plot_signals", 5),
        save_path=out / "signals_full.png",
        # show=show_plots
    )

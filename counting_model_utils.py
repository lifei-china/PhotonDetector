
import numpy as np
import matplotlib.pyplot as plt
 

def load_signal_from_spek_abs(filename, I_mA=1.0, t_s = 400e-6,
                              A_pix=0.05*0.05, d=1.0,
                              res_sigma=0.0,
                              E_out=None,  # 可选：目标能量网格（keV），若给出将插值到该网格
                              plot=True, return_spectrum=True):
    """
    读取 Spek 光谱（E, flux[#/(keV·cm²·mAs)@1m]），保持绝对量，不做归一化。
    计算像素上的期望计数谱，并（可选）按 Poisson 采样出光子能量。
    返回:
      energies  : 采样得到的光子能量数组（可能为空，取决于期望计数）
      E_grid    : 能量网格（原始或插值后）
      flux_pix  : 像素处的通量谱  [#/keV]（已乘 A_pix, d, mAs）
      counts_exp: 每个能量 bin 的期望计数 [#]
    """

    # 1) 读 Spek 数据（假设两列：E[keV], flux）
    data = np.loadtxt(filename, skiprows=1)
    E_src   = data[:, 0].astype(float)
    flux_src = data[:, 1].astype(float)  # #/(keV·cm²·mAs) @1 m

    # 2) 距离/面积/曝光换算到像素上的单位能量通量
    mAs = I_mA * t_s
    flux_pix_src = flux_src * A_pix * (1.0 / d**2) * mAs   # #/keV per pixel (for this exposure)

    # 3) 可选：把谱插值到目标能量网格（比如更细的 0.5 keV）
    if E_out is not None:
        E_grid = np.asarray(E_out, float)
        flux_pix = np.interp(E_grid, E_src, flux_pix_src, left=0.0, right=0.0)
    else:
        E_grid = E_src
        flux_pix = flux_pix_src

    # 4) 计算各 bin 宽度 ΔE（最后一个 bin 取与前一相同）
    dE = np.diff(E_grid, append=E_grid[-1] + (E_grid[-1] - E_grid[-2] if E_grid.size>1 else 1.0) )
   
    # 5) 每个 bin 的期望**绝对计数**（不归一化）
    counts_exp = flux_pix * dE  # [# per pixel]

    # 6) Poisson 采样得到实际光子数，并在各 bin 内均匀撒点
    n_per_bin = np.random.poisson(np.maximum(counts_exp, 0.0))
    total_n = int(n_per_bin.sum())
    energies = np.empty(total_n, dtype=float)
    idx = 0
    for i, n in enumerate(n_per_bin):
        if n <= 0: 
            continue
        e0 = E_grid[i]
        e1 = E_grid[i] + dE[i]
        energies[idx:idx+n] = e0 + np.random.rand(n) * (e1 - e0)
        idx += n

    # 7) 能量分辨率（可选）
    if res_sigma > 0 and energies.size > 0:
        energies = np.random.normal(energies, res_sigma)
        energies = energies[energies > 0]

    # 8) 画图（绝对量）
    if plot:
        # fig, ax = plt.subplots(1, 1, figsize=(8, 5))
        # ax.step(E_grid, flux_pix, where='post', label="Flux per pixel [#/keV]", linewidth=1.6)
        # ax.set_xlabel("Energy (keV)")
        # ax.set_ylabel("Flux per pixel [#/keV]")
        # ax.grid(True, alpha=0.3)
        # ax.legend(loc="best")
        # plt.tight_layout()
        # plt.show()

        # # 期望计数谱
        # fig, ax = plt.subplots(1, 1, figsize=(8, 4))
        # ax.step(E_grid, counts_exp, where='post', label="Expected counts per bin", linewidth=1.6)
        # ax.set_xlabel("Energy bin start (keV)")
        # ax.set_ylabel("Counts per bin [#]")
        # ax.grid(True, alpha=0.3)
        # ax.legend(loc="best")
        # plt.tight_layout()
        # plt.show()

        # 抽样到的能量直方图（同样画**绝对计数**）
        if energies.size > 0:
            ax = plt.figure(figsize=(8,4)).gca()
            ax.hist(energies, bins=len(E_grid), range=(E_grid[0], E_grid[-1]+dE[-1]),
                    density=False, alpha=0.5, label=f"Sampled (N={energies.size})")
            ax.set_xlabel("Energy (keV)")
            ax.set_ylabel("Counts")
            ax.legend(); ax.grid(True, alpha=0.3)
            plt.tight_layout(); plt.show()

    return (energies, E_grid, flux_pix, counts_exp) if return_spectrum else energies


# ---------------- 积分谱计算函数（考虑死时间） ----------------
def integral_spectrum(events, thresholds, tau, T, mode="ideal"):
    spectrum = []
    for thr in thresholds:
        # 阈值以上的光子数
        N_thr = np.sum(events >= thr)
        # 当前输入光子率
        R_thr = N_thr / T  

        if mode == "ideal":
            R_obs = R_thr
        elif mode == "nonpar":
            R_obs = R_thr / (1 + R_thr * tau)
        elif mode == "par":
            R_obs = R_thr * np.exp(-R_thr * tau)
        else:
            raise ValueError("Unknown mode")
        
 # 转换回总计数
        N_obs = R_obs * T
  
        # 转换回计数率
        R_final = N_obs 
        # R_final = N_obs / T
        # 转换回总计数数目
        spectrum.append(R_final)
    return np.array(spectrum)



def double_specturm(events, thresholds, 
                    dead_time, integral_time, 
                    mode="par"):

    Par_int = integral_spectrum(events, thresholds, 
                                    dead_time, integral_time, mode="par")
    Par_diff =     -np.gradient(Par_int, thresholds)
    return Par_int, Par_diff

def plot_spectrum_subplot(ax, x, curves, xlabel, ylabel, title):
    """
    在子图 ax 上绘制多条曲线
    参数:
        ax     : matplotlib subplot 对象
        x      : 横坐标 (ndarray)
        curves : [(y, style, label)] 的列表
        xlabel, ylabel, title : 坐标轴标题 & 图标题
    """
    # mcps = 1e6
    mcps = 1
    for y, style, label in curves:
        ax.plot(x, y/mcps, style, label=label)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True)

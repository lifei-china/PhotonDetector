import numpy as np
import matplotlib.pyplot as plt
from poisson_signal_generator_utils import generate_signal, build_energy_bins_and_pdf, compute_filtered_spectrum

# 参数列表：
        #   density_Ce = 6.77
        #   density_Gd = 7.90
        #   density_W  = 19.3
        #   density_Pb = 11.34
        #   density_Al = 2.70


# 1) 读取理论光谱文件,
density_Gd = 7.90
energies, Phi_in, Phi_out_list = compute_filtered_spectrum(
    spek_path="spectrum_file/spek_120.txt",
    xcom_filter_path="spectrum_file/Gd_mu_over_rho.csv",
    density= density_Gd,
    thickness_mm_list=[0.001, 0.3, 1.20],
    plot=True
)

# 1) 读取所需光谱
values = Phi_out_list[1]
bins, pdf, dE = build_energy_bins_and_pdf(energies, 
                                          values) # 构建坐标轴 计算入射计数率 λ（counts/s）# ---------- 用中心能量构造“边界” -> 保证 bins 长度 = N+1, pdf 长度 = N ----------

values_source = Phi_in


# 2) 光源物理参数
Voltage = 120                   # 管压              [kvp]
I_mA   = 10.00                  # 管电流            [mA] 
r_m    = 0.30                   # 源到探测器距离     [m]
pixel_length  = 340e-6           # 单像素尺寸340um， [m]  
pixel_length_cm = pixel_length * 100.0            # [cm]

# 2-1) 光子信号在芯片上的假设参数
pulse   = 15e-9                 # 信号脉宽 15ns      [s]
W_eV = 4.5                      # CdTe 1eV 产生一对电子 [eV/pair]
g_mat = 1000.0 / W_eV           # ~ 227 e-h pairs per keV

# 3） 芯片参数设置
delta_t = 1e-9                    # 采样帧频率 1ns     [s]
integral_time = 400e-6             # 信号积分时间 400us [s]
# Capacitance = 40e-15            # 典型的 CdTe/CZT PCD 前端电容量级 [F]
Capacitance = 40e-15           # 典型的 CdTe/CZT PCD 前端电容量级 [F]
 
noise_ENC_BLRoff = 330                 # Equivalent Noise Charge 芯片上的等效噪声电荷数
noise_ENC_BLRon_LowRange   = 350
noise_ENC_BLRon_HighRange  = 390

# 3.5) 实验参数
a_keV_per_LSB = 0.85       # keV / LSB
b_keV        = -36.85      # keV


# 4) 计算参数
A_cm2  = pixel_length_cm**2    # m->cm，面积用 cm^2
photon_rate = np.sum(values * dE * I_mA * (1.0/r_m)**2 * A_cm2)   # 射线源的发射率 

print("lambda*tau =", photon_rate * pulse)  # 无量纲


from poisson_signal_generator_utils import make_threshold_grid_from_LSB
# -------- 固定全局阈值轴（重要！）-------- 

# thr_grid = make_threshold_grid_from_energy(bins, gain=1, n_thr=256)
# --- 生成真实的“阈值电压表” ---
thr_grid, lsb_values, E_thr_keV = make_threshold_grid_from_LSB(
    a_keV_per_LSB=a_keV_per_LSB,
    b_keV=b_keV,
    g_mat=g_mat,
    Cf=Capacitance,
    n_thr=256,             # 对应 LSB=0..255
) 



from poisson_signal_generator_utils import make_threshold_grid_from_energy, spectrum_from_signal
from poisson_signal_generator_utils import plot_signals_over_time, plot_signals_grid

 
# 5） 根据 归一化的 谱，产生时间信号
n_runs = 1   # 重复次数
integrals, differentials = [], []
signals = []  # 存每次生成的波形
 

for _ in range(n_runs):
    t, sig, times, amps = generate_signal(bins, pdf, photon_rate, 
                                      integral_time , 
                                      pulse,     # 脉宽 
                                      delta_t,              # 1ns
                                      g_mat = g_mat,
                                      noise_ENC = noise_ENC_BLRoff,
                                      Cf = Capacitance,
                                      pulse_shape='gauss'
                                      )
    integ, diff = spectrum_from_signal(sig, thr_grid, lsb_values=lsb_values)
    integrals.append(integ)
    differentials.append(diff) 
    signals.append(sig)     # ✅ 保存每次的波形

integral_avg     = np.mean(integrals, axis=0)
differential_avg = np.mean(differentials, axis=0)

# ✅ 画出所有信号叠加图  
plot_signals_over_time(t, signals, max_plot=5, zoom=(5.0, 5.2))
plot_signals_over_time(t, signals, max_plot=5 )

from scipy.ndimage import gaussian_filter1d

energy_axis = lsb_values   # keV（在你的 generate_signal 里 gain=1.0）
sigma_smooth = 3.50
secd_differential_avg = - np.gradient( differential_avg, energy_axis )
secd_differential_smooth = gaussian_filter1d(secd_differential_avg, sigma=sigma_smooth)

# --- 画图：积分谱 & 微分谱（与输入pdf作形状对比）---
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 10), sharex=True)

# 积分谱（阈值越高，计数越少）
ax1.plot(energy_axis, integral_avg, lw=1.8)
ax1.set_ylabel('Integral counts')
ax1.set_title('Integral Spectrum')
ax1.set_xlabel('Threshold / Energy (keV)')

# 微分谱（应近似输入能谱形状；仅差一个尺度因子）
ax2.plot(energy_axis, differential_avg, lw=1.8, label='Differential ( -dN/dT )')

# 可选：把输入 pdf 归一后缩放到可见范围作对比（仅形状对比）
# 这里简单按最大值对齐
orig_centers = (bins[:-1] + bins[1:]) / 2
 
ax2.set_xlabel('Threshold / Energy (keV)')
ax2.set_ylabel('Counts per keV')
ax2.set_title('Differential Spectrum')
ax2.set_ylim(-100, 100)
ax2.legend()

# 积分谱（阈值越高，计数越少）
ax3.plot(energy_axis, secd_differential_avg, lw=1.0, label='raw')
ax3.plot(energy_axis, secd_differential_smooth, lw=1.8, label=f'Gaussian smooth σ={sigma_smooth}')
ax3.set_ylabel('secd_differential ')
ax3.set_title('secd_differential ')
ax3.set_xlabel('Threshold / Energy (keV)')
ax3.set_ylim(-100, 30)
ax3.legend()

for ax in (ax1, ax2, ax3):
    # ax.set_xlim(0, 165)
    ax.set_xticks(np.arange(0, 300, 20))   # 每隔 5 keV 一根刻度

plt.tight_layout()
plt.show()

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path 
import plotly.graph_objects as go 

def analyze_and_plot_pixel(integral, thr, thr_mid, row, col, condition, outdir=None, savefig=False):
    """
    分析单像素积分/微分曲线并绘图

    参数
    ----
    integral : np.ndarray
        积分曲线 (1D array, 长度=阈值个数)
    thr : np.ndarray
        阈值序列 (e.g. np.arange(n_thresh))
    thr_mid : np.ndarray
        阈值中点序列 (长度 = len(thr)-1)
    row, col : int
        像素坐标 (行, 列)
    condition : str
        测试条件名，用于标题
    outdir : Path 或 str, optional
        输出文件夹路径
    savefig : bool
        是否保存图像 (True 保存，False 仅显示)
    """
    # 微分曲线
    diff = integral[:-1] - integral[1:]

    # --------- 关键点分析 ---------

    # 0. 积分曲线的最右边阈值 (积分值最后不为0的位置)  
    nonzero_indices = np.where(integral > 0)[0]
    if len(nonzero_indices) > 0:
        idx_emax_int = nonzero_indices[-1]   # 最后一个非零点
        LSB_emax_int = thr[idx_emax_int]     # 用原始阈值序号
    else:
        LSB_emax_int = np.nan 

    # 1. 零点
    # idx_zero = np.argmin(np.abs(diff[5:50]))
    # LSB_zero = thr[idx_zero]

    search_start, search_end = 15, 50
    idx_zero_local = np.argmin(np.abs(diff[search_start:search_end]))
    idx_zero = search_start + idx_zero_local   # 转换为全局索引
    LSB_zero = thr_mid[idx_zero]               # 注意用 thr_mid 对齐微分曲线

    # 2. 主峰点 (阈值 >50)
    valid_range = thr_mid > 50
    idx_peak = np.argmax(diff[valid_range])
    idx_peak = np.where(valid_range)[0][idx_peak]
    LSB_peak = thr_mid[idx_peak]
    N_max = diff[idx_peak]

    # 3. 能量最大点 (下降段外推)
    search_region = np.arange(idx_peak, len(diff))
    idx1 = search_region[np.argmin(np.abs(diff[search_region] - 0.1*N_max))]
    idx2 = search_region[np.argmin(np.abs(diff[search_region] - 0.2*N_max))]
    x1, y1 = thr_mid[idx1], diff[idx1]
    x2, y2 = thr_mid[idx2], diff[idx2]

    if y2 != y1:
        LSB_emax = x1 - y1 * (x2 - x1) / (y2 - y1)
    else:
        LSB_emax = np.nan

    # --------- 绘图 ---------
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # 积分图
    axes[0].plot(thr, integral, '-o', markersize=3)
    axes[0].set_title(f"Integral curve\nPixel ({row},{col})")
    axes[0].set_xlabel("Threshold index")
    axes[0].set_ylabel("Counts")
    axes[0].grid(True)

    # 微分图
    axes[1].plot(thr_mid, diff, '-o', markersize=3, label="Differential")
    axes[1].plot(LSB_zero, diff[idx_zero], 'ro', label=f"Zero={LSB_zero:.1f}")
    axes[1].plot(LSB_peak, N_max, 'ro', label=f"Peak={LSB_peak:.1f}")
    if not np.isnan(LSB_emax):
        axes[1].plot(LSB_emax, 0, 'ro', label=f"Emax={LSB_emax:.1f}")
    if not np.isnan(LSB_emax_int):
        axes[1].plot(LSB_emax_int, 0, 'ro', label=f"Emax={LSB_emax_int:.1f}")
    axes[1].plot(x1, y1, 'rx', markersize=8, label="0.1Nmax")
    axes[1].plot(x2, y2, 'rx', markersize=8, label="0.2Nmax")
    axes[1].plot([x1, x2], [y1, y2], 'r--', linewidth=1.5, label="Extrapolation")

    axes[1].set_title(f"Differential curve\nPixel ({row},{col})")
    axes[1].set_xlabel("Threshold index")
    axes[1].set_ylabel("ΔCounts")
    axes[1].grid(True)
    axes[1].legend()

    fig.suptitle(f"{condition}", fontsize=12, fontweight='bold')
    plt.tight_layout()

    # 保存 or 显示
    if savefig and outdir is not None:
        Path(outdir).mkdir(parents=True, exist_ok=True)
        outpath = Path(outdir) / f"pixel_{row}_{col}.png"
        fig.savefig(outpath, dpi=150)
        plt.close(fig)
    else:
        plt.show()

    return LSB_zero, LSB_peak, LSB_emax, LSB_emax_int



 
import numpy as np
import plotly.graph_objects as go
import os
from pathlib import Path
def make_clickable_map(map_data, title, maps_dir, pixels_dir, nrow, ncol, html_name="map.html"):
    import numpy as np, os
    import plotly.graph_objects as go
    from pathlib import Path

    maps_dir = Path(maps_dir)
    pixels_dir = Path(pixels_dir)

    # 计算相对路径: 从 maps_dir 到 pixels_dir
    rel_path = os.path.relpath(pixels_dir, maps_dir)

    # 每个像素对应的 PNG 文件（相对路径）
    customdata = np.empty((nrow, ncol), dtype=object)
    for r in range(nrow):
        for c in range(ncol):
            customdata[r, c] = str(Path(rel_path) / f"pixel_{r}_{c}.png")

    # map_data = np.flipud(map_data) 
    y_coords = np.arange(nrow-1, -1, -1)  # [23,22,...,0]

    fig = go.Figure(data=go.Heatmap(
        z=map_data,    
        x=np.arange(ncol),
        y=y_coords,   # y 轴索引 (0..23)
        text=[[f"{map_data[r, c]:.1f}" for c in range(ncol)] for r in range(nrow)],
        texttemplate="%{text}",
        colorscale="RdBu",
        customdata=customdata,
        hovertemplate="Row %{y}, Col %{x}<br>Value=%{z}<extra></extra>",
    ))

 
    # ✅ 正确设置 y 轴翻转 + tick label
    fig.update_yaxes(
        # autorange="reversed",
        tickmode="array",
        tickvals=y_coords,
        ticktext=np.arange(nrow)   # 倒序显示 0..23
    )

    fig.update_layout(
        title=title,
        clickmode="event+select"
    )

    # 输出 HTML 文件
    html_file = maps_dir / html_name
    fig.write_html(html_file, include_plotlyjs="cdn")

    # 注入 JS 脚本
    with open(html_file, "r", encoding="utf-8") as f:
        html_text = f.read()

    js_code = """
<script>
window.onload = function() {
    var plot = document.getElementsByClassName('plotly-graph-div')[0];
    plot.on('plotly_click', function(data){
        var point = data.points[0];
        var img_path = point.customdata;
        if (img_path) {
            window.open(img_path, '_blank');
        }
    });
};
</script>
"""
    html_text = html_text.replace("</body>", js_code + "\n</body>")

    with open(html_file, "w", encoding="utf-8") as f:
        f.write(html_text)

    print(f"已生成: {html_file}")

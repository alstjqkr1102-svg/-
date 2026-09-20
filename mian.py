import streamlit as st
import numpy as np
import scipy.optimize as opt
import plotly.graph_objects as go
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# Page Setup
# ---------------------------------------------------------
st.set_page_config(page_title="구면 표면 최소 에너지 경로 시뮬레이터", layout="wide")

st.title("🌐 구면 표면 최소 에너지 경로 (Minimum-energy surface path)")
st.markdown("이동 거리($L$)와 수직항력($N$)을 분리하여 마찰 에너지 및 총 에너지를 정밀 계산합니다.")

# ---------------------------------------------------------
# Top Input Panel
# ---------------------------------------------------------
col_ax, col_ay, col_az, col_bx, col_by, col_bz = st.columns(6)

with col_ax:
    Ax = st.number_input("A x", value=-4.0, step=0.5, format="%.1f")
with col_ay:
    Ay = st.number_input("A y", value=0.0, step=0.5, format="%.1f")
with col_az:
    Az = st.number_input("A z", value=3.0, step=0.5, format="%.1f")

with col_bx:
    Bx = st.number_input("B x", value=4.0, step=0.5, format="%.1f")
with col_by:
    By = st.number_input("B y", value=0.0, step=0.5, format="%.1f")
with col_bz:
    Bz = st.number_input("B z", value=3.0, step=0.5, format="%.1f")

col_m, col_mu, col_algo, col_btn = st.columns([1.5, 1.5, 3, 2])

with col_m:
    m = st.number_input("질량 (kg)", value=1.0, min_value=0.1, step=0.1, format="%.1f")
with col_mu:
    mu = st.number_input("마찰계수 μ", value=0.20, min_value=0.00, max_value=1.00, step=0.01, format="%.2f")
with col_algo:
    algorithm_option = st.selectbox(
        "탐색 알고리즘",
        [
            "개선 방식 1: 다중 초기값 SLSQP (Multi-start)",
            "기존 방식: 단일 SLSQP (최단경로 초기값)",
            "개선 방식 2: 전역 최적화 (Differential Evolution)"
        ]
    )
with col_btn:
    st.write("")
    st.write("")
    run_opt = st.button("🚀 최적화 실행", use_container_width=True, type="primary")

# ---------------------------------------------------------
# Geometry & Physical Calculations
# ---------------------------------------------------------
g = 9.81
RA = np.sqrt(Ax**2 + Ay**2 + Az**2)
RB = np.sqrt(Bx**2 + By**2 + Bz**2)
R = (RA + RB) / 2.0 if (RA > 0 and RB > 0) else 5.0

phi_A = np.arcsin(np.clip(Az / RA, -1.0, 1.0)) if RA > 0 else 0.0
theta_A = np.arctan2(Ay, Ax)

phi_B = np.arcsin(np.clip(Bz / RB, -1.0, 1.0)) if RB > 0 else 0.0
theta_B = np.arctan2(By, Bx)

dtheta = (theta_B - theta_A + np.pi) % (2 * np.pi) - np.pi
theta_target = theta_A + dtheta

num_intermediate = 7
N_points = 150
t_arr = np.linspace(0, 1, N_points)
theta_arr = np.linspace(theta_A, theta_target, N_points)

def spherical_to_cartesian(r, theta, phi):
    x = r * np.cos(phi) * np.cos(theta)
    y = r * np.cos(phi) * np.sin(theta)
    z = r * np.sin(phi)
    return x, y, z

def calculate_path_energy(phi_nodes):
    t_control = np.linspace(0, 1, len(phi_nodes))
    phi_arr = np.interp(t_arr, t_control, phi_nodes)
    
    x, y, z = spherical_to_cartesian(R, theta_arr, phi_arr)
    
    # 1. 상승 위치 에너지 (Rise Energy)
    z_max = np.max(z)
    delta_h = max(0.0, z_max - z[0])
    E_rise = m * g * delta_h
    
    # 2. 이동 거리 및 미소 구간별 수직항력 분리 계산
    dx = np.diff(x)
    dy = np.diff(y)
    dz = np.diff(z)
    ds_arr = np.sqrt(dx**2 + dy**2 + dz**2)
    total_length = np.sum(ds_arr)  # L (이동 거리)
    
    # 각 지점에서의 수직항력 N = m * g * cos(phi)
    phi_mid = (phi_arr[:-1] + phi_arr[1:]) / 2.0
    N_arr = m * g * np.cos(phi_mid)
    
    # 평균 수직항력 N_avg 산출
    N_avg = np.sum(N_arr * ds_arr) / total_length if total_length > 0 else m * g
    
    # 마찰 에너지 = μ * N_avg * L
    E_friction = mu * N_avg * total_length
    
    E_total = E_rise + E_friction
    return E_total, E_rise, E_friction, total_length, N_avg, x, y, z

# 대원 경로 초기화
uA = np.array([Ax, Ay, Az]) / (RA if RA > 0 else 1)
uB = np.array([Bx, By, Bz]) / (RB if RB > 0 else 1)
t_vals = np.linspace(0, 1, num_intermediate + 2)
init_path_3d = np.array([(1 - t) * uA + t * uB for t in t_vals])
init_path_3d = init_path_3d / np.linalg.norm(init_path_3d, axis=1, keepdims=True) * R
init_phi_nodes = np.arcsin(np.clip(init_path_3d[:, 2] / R, -1.0, 1.0))

# 초기 상태 계산
e_init_tot, _, _, init_L, init_N, init_x, init_y, init_z = calculate_path_energy(init_phi_nodes)

# ---------------------------------------------------------
# Optimization Execution Logic
# ---------------------------------------------------------
bounds = [(-np.pi/2 + 0.05, np.pi/2 - 0.05)] * num_intermediate
opt_options = {'maxiter': 60, 'ftol': 1e-4}

def objective_func(phi_mid):
    phi_full = np.concatenate([[phi_A], phi_mid, [phi_B]])
    e_tot, _, _, _, _, _, _, _ = calculate_path_energy(phi_full)
    return e_tot

if "has_run" not in st.session_state or run_opt:
    with st.spinner("🚀 최적화 경로 계산 중..."):
        st.session_state.has_run = True
        iterations = 0
        
        if algorithm_option == "기존 방식: 단일 SLSQP (최단경로 초기값)":
            init_mid = init_phi_nodes[1:-1]
            res = opt.minimize(objective_func, init_mid, method='SLSQP', bounds=bounds, options=opt_options)
            best_phi_mid = res.x
            iterations = res.nit
            
        elif algorithm_option == "개선 방식 1: 다중 초기값 SLSQP (Multi-start)":
            candidates = [
                init_phi_nodes[1:-1],
                np.full(num_intermediate, (phi_A + phi_B)/2),
                np.linspace(phi_A, np.pi/3, num_intermediate),
                np.full(num_intermediate, 0.0)
            ]
            best_e = float('inf')
            total_nit = 0
            for init_p in candidates:
                res = opt.minimize(objective_func, init_p, method='SLSQP', bounds=bounds, options=opt_options)
                total_nit += res.nit
                if res.fun < best_e:
                    best_e = res.fun
                    best_phi_mid = res.x
            iterations = total_nit
            
        else:
            res = opt.differential_evolution(
                objective_func, bounds=bounds, seed=42, maxiter=100, popsize=12, polish=True
            )
            best_phi_mid = res.x
            iterations = res.nit

        opt_phi_full = np.concatenate([[phi_A], best_phi_mid, [phi_B]])
        e_tot, e_rise, e_fric, path_len, n_avg, px, py, pz = calculate_path_energy(opt_phi_full)
        
        st.session_state.opt_results = {
            "iterations": iterations,
            "e_tot": e_tot,
            "e_rise": e_rise,
            "e_fric": e_fric,
            "path_len": path_len,
            "n_avg": n_avg,
            "px": px, "py": py, "pz": pz
        }

# ---------------------------------------------------------
# Results Display
# ---------------------------------------------------------
res = st.session_state.opt_results

st.success(
    f"**Optimization terminated successfully** | 반복 {res['iterations']}회 | "
    f"**총 에너지 = {res['e_tot']:.4f} J** (오르막 {res['e_rise']:.4f} J + 마찰 {res['e_fric']:.4f} J)"
)

# 이동 거리와 수직항력 분리 표시 메트릭
c1, c2, c3 = st.columns(3)
with c1:
    st.metric(label="📏 이동 거리 (L)", value=f"{res['path_len']:.3f} m")
with c2:
    st.metric(label="⚖️ 평균 수직항력 (N_avg)", value=f"{res['n_avg']:.3f} N")
with c3:
    st.metric(label="🔥 마찰 손실 에너지", value=f"{res['e_fric']:.4f} J")

# ---------------------------------------------------------
# 3D Visualization
# ---------------------------------------------------------
tab1, tab2 = st.tabs(["🌐 3D 인터랙티브 그래프 (Plotly)", "📈 3D 정적 그래프 (Matplotlib)"])

with tab1:
    u_mesh = np.linspace(0, 2 * np.pi, 40)
    v_mesh = np.linspace(-np.pi / 2, np.pi / 2, 40)
    sx = R * np.outer(np.cos(v_mesh), np.cos(u_mesh))
    sy = R * np.outer(np.cos(v_mesh), np.sin(u_mesh))
    sz = R * np.outer(np.sin(v_mesh), np.ones(np.size(u_mesh)))

    fig = go.Figure()

    fig.add_trace(go.Surface(x=sx, y=sy, z=sz, opacity=0.30, colorscale='Blues', showscale=False, hoverinfo='skip'))
    fig.add_trace(go.Scatter3d(
        x=init_x, y=init_y, z=init_z, mode='lines+markers',
        line=dict(color='gray', width=3, dash='dash'), marker=dict(size=2, color='gray'),
        name='Initial projected path'
    ))
    fig.add_trace(go.Scatter3d(
        x=res['px'], y=res['py'], z=res['pz'], mode='lines+markers',
        line=dict(color='crimson', width=5), marker=dict(size=3.5, color='crimson'),
        name='Optimized path'
    ))

    xA, yA, zA = spherical_to_cartesian(R, theta_A, phi_A)
    xB, yB, zB = spherical_to_cartesian(R, theta_B, phi_B)

    fig.add_trace(go.Scatter3d(x=[xA], y=[yA], z=[zA], mode='markers+text', marker=dict(size=8, color='blue'), text=['A (시작점)'], name='A (시작점)'))
    fig.add_trace(go.Scatter3d(x=[xB], y=[yB], z=[zB], mode='markers+text', marker=dict(size=8, color='green'), text=['B (도착점)'], name='B (도착점)'))

    fig.update_layout(
        title=dict(text="<b>Minimum-energy surface path (구면 표면 최소 에너지 경로)</b>", font=dict(size=16)),
        scene=dict(xaxis_title='x', yaxis_title='y', zaxis_title='z', aspectmode='data'),
        legend=dict(x=0.02, y=0.98, bgcolor='rgba(255,255,255,0.7)'),
        height=600, margin=dict(l=10, r=10, b=10, t=40)
    )

    st.plotly_chart(fig, use_container_width=True, theme=None)

with tab2:
    fig_mpl = plt.figure(figsize=(9, 7))
    ax = fig_mpl.add_subplot(111, projection='3d')

    u_m = np.linspace(0, 2 * np.pi, 30)
    v_m = np.linspace(-np.pi / 2, np.pi / 2, 30)
    sx_m = R * np.outer(np.cos(v_m), np.cos(u_m))
    sy_m = R * np.outer(np.cos(v_m), np.sin(u_m))
    sz_m = R * np.outer(np.sin(v_m), np.ones(np.size(u_m)))

    ax.plot_surface(sx_m, sy_m, sz_m, color='lightblue', alpha=0.25, edgecolor='none')
    ax.plot(init_x, init_y, init_z, color='gray', linestyle='--', linewidth=1.5, label='Initial path')
    ax.plot(res['px'], res['py'], res['pz'], color='crimson', linewidth=2.5, label='Optimized path')
    ax.scatter(res['px'], res['py'], res['pz'], color='crimson', s=12)
    ax.scatter([xA], [yA], [zA], color='blue', s=60, label='A (Start)')
    ax.scatter([xB], [yB], [zB], color='green', s=60, label='B (End)')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title("Minimum-energy surface path (3D)")
    ax.legend(loc='upper left')
    st.pyplot(fig_mpl)
    plt.close(fig_mpl)
    

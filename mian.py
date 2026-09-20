import streamlit as st
import numpy as np
from scipy.optimize import minimize, differential_evolution
import plotly.graph_objects as go

# 페이지 레이아웃 설정
st.set_page_config(page_title="구면 표면 최소 에너지 경로", layout="wide")

st.title("Minimum-energy surface path (구면 표면 최소 에너지 경로)")

# --- 1. 입력 폼 레이아웃 ---
col1, col2, col3, col4, col5, col6 = st.columns(6)

with col1:
    xA = st.number_input("A x", value=-4.0, step=0.1)
with col2:
    yA = st.number_input("A y", value=0.0, step=0.1)
with col3:
    zA = st.number_input("A z", value=3.0, step=0.1)

with col4:
    xB = st.number_input("B x", value=4.0, step=0.1)
with col5:
    yB = st.number_input("B y", value=0.0, step=0.1)
with col6:
    zB = st.number_input("B z", value=3.0, step=0.1)

col_m, col_mu, col_algo, col_btn = st.columns([1.5, 1.5, 3, 2])

with col_m:
    mass = st.number_input("질량 (kg)", value=1.0, min_value=0.1, step=0.1)
with col_mu:
    mu = st.number_input("마찰계수 μ", value=0.08, min_value=0.0, step=0.01)
with col_algo:
    algo_option = st.selectbox(
        "최적화 알고리즘 선택",
        options=[
            "기존 방식: 단일 SLSQP (최단경로 국소 최적점 고착)",
            "개선 방식 1: Multi-start SLSQP (다중 초기 경로 탐색)",
            "개선 방식 2: Differential Evolution (전역 최적화)"
        ]
    )

with col_btn:
    st.write(" ") 
    run_button = st.button("🚀 최적화 실행", type="primary", use_container_width=True)

# 기본 물리 상수 및 함수 정의
raw_A = np.array([xA, yA, zA], dtype=float)
raw_B = np.array([xB, yB, zB], dtype=float)
R_default = float(np.mean([np.linalg.norm(raw_A), np.linalg.norm(raw_B)]))
R = R_default if R_default > 0 else 5.0

g = 9.81 
N_POINTS = 15 # 중간 제어점 개수

def project_to_sphere(p):
    """단일 점 또는 점들의 집합을 반지름 R 구면 위로 투영"""
    p = np.array(p, dtype=float)
    if p.ndim == 1:
        norm = np.linalg.norm(p)
        return p * (R / norm) if norm > 0 else p
    else:
        norms = np.linalg.norm(p, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return p * (R / norms)

def calculate_path_energy(P_flat, pA, pB, m, mu_val):
    """에너지 계산 (오르막 위치에너지 + 마찰 손실 에너지)"""
    P_mid = project_to_sphere(np.array(P_flat).reshape((-1, 3)))
    full_path = np.vstack(([pA], P_mid, [pB]))
    
    climb_energy = 0.0
    friction_energy = 0.0
    
    for i in range(len(full_path) - 1):
        p1 = full_path[i]
        p2 = full_path[i+1]
        
        ds_vec = p2 - p1
        ds = np.linalg.norm(ds_vec)
        if ds < 1e-9:
            continue
            
        dz = p2[2] - p1[2]
        
        # 1. 위치 에너지 (오르막 구간만 추가)
        if dz > 0:
            climb_energy += m * g * dz
            
        # 2. 마찰 에너지
        p_mid_seg = (p1 + p2) / 2.0
        r_norm = np.linalg.norm(p_mid_seg)
        n_hat = p_mid_seg / r_norm if r_norm > 0 else np.array([0, 0, 1])
            
        cos_theta = max(0.0, n_hat[2])
        N_force = m * g * cos_theta
        friction_energy += mu_val * N_force * ds
        
    return climb_energy + friction_energy, climb_energy, friction_energy

def get_initial_path(pA, pB, mode="geodesic"):
    """시작점 pA와 도착점 pB를 정확히 잇는 초기 경로 생성"""
    t = np.linspace(0, 1, N_POINTS + 2)[1:-1] # 양 끝점 제외한 내적점
    
    if mode == "geodesic":
        x = (1 - t) * pA[0] + t * pB[0]
        y = (1 - t) * pA[1] + t * pB[1]
        z = (1 - t) * pA[2] + t * pB[2]
    elif mode == "horizontal_detour":
        x = (1 - t) * pA[0] + t * pB[0]
        y = (1 - t) * pA[1] + t * pB[1] + np.sin(np.pi * t) * 3.0
        z = (1 - t) * pA[2] + t * pB[2]

    P_init = np.column_stack((x, y, z))
    return project_to_sphere(P_init).flatten()

# --- 최적화 실행 조건 및 세션 초기화 ---
ptA_curr = project_to_sphere(raw_A)
ptB_curr = project_to_sphere(raw_B)

if run_button or "best_path_flat" not in st.session_state:
    with st.spinner("최적 에너지 경로를 계산 중입니다..."):
        def objective(P_flat):
            tot, _, _ = calculate_path_energy(P_flat, ptA_curr, ptB_curr, mass, mu)
            return tot

        if "기존 방식" in algo_option:
            p0 = get_initial_path(ptA_curr, ptB_curr, "geodesic")
            res = minimize(objective, p0, method='SLSQP', options={'maxiter': 100, 'ftol': 1e-6})
            best_path_flat = res.x
            n_iter = res.nit
        elif "Multi-start" in algo_option:
            candidates = ["geodesic", "horizontal_detour"]
            best_val = float('inf')
            best_path_flat = None
            total_nit = 0
            
            for cand in candidates:
                p0 = get_initial_path(ptA_curr, ptB_curr, cand)
                res = minimize(objective, p0, method='SLSQP', options={'maxiter': 100, 'ftol': 1e-6})
                total_nit += res.nit
                if res.fun < best_val:
                    best_val = res.fun
                    best_path_flat = res.x
            n_iter = total_nit
        else:
            bounds = [(-R, R)] * (N_POINTS * 3)
            res = differential_evolution(objective, bounds, maxiter=40, popsize=10, seed=42, polish=True)
            best_path_flat = res.x
            n_iter = res.nit

        st.session_state["best_path_flat"] = best_path_flat
        st.session_state["init_path_geodesic"] = get_initial_path(ptA_curr, ptB_curr, "geodesic")
        st.session_state["n_iter"] = n_iter
        st.session_state["ptA"] = ptA_curr
        st.session_state["ptB"] = ptB_curr
        st.session_state["R"] = R

# 세션 데이터 안전하게 불러오기
if "best_path_flat" not in st.session_state:
    st.session_state["R"] = R
    st.session_state["ptA"] = ptA_curr
    st.session_state["ptB"] = ptB_curr
    st.session_state["init_path_geodesic"] = get_initial_path(ptA_curr, ptB_curr, "geodesic")
    st.session_state["best_path_flat"] = st.session_state["init_path_geodesic"]
    st.session_state["n_iter"] = 0

best_path_flat = st.session_state["best_path_flat"]
init_path_geodesic = st.session_state["init_path_geodesic"]
n_iter = st.session_state["n_iter"]
ptA = st.session_state["ptA"]
ptB = st.session_state["ptB"]
R = st.session_state["R"]

tot_e, climb_e, fric_e = calculate_path_energy(best_path_flat, ptA, ptB, mass, mu)

# --- 2. 결과 상태 출력 ---
st.success(
    f"Optimization terminated successfully | 반복 {n_iter}회 | "
    f"총 에너지 = {tot_e:.4f} J (오르막 {climb_e:.4f} J, 마찰 {fric_e:.4f} J)"
)

# --- 3. Plotly 3D 시각화 ---
u = np.linspace(0, 2 * np.pi, 40)
v = np.linspace(0, np.pi, 40)
x_sphere = R * np.outer(np.cos(u), np.sin(v))
y_sphere = R * np.outer(np.sin(u), np.sin(v))
z_sphere = R * np.outer(np.ones(np.size(u)), np.cos(v))

P_init_mid = project_to_sphere(np.array(init_path_geodesic).reshape((-1, 3)))
full_init = np.vstack(([ptA], P_init_mid, [ptB]))

P_opt_mid = project_to_sphere(np.array(best_path_flat).reshape((-1, 3)))
full_opt = np.vstack(([ptA], P_opt_mid, [ptB]))

fig = go.Figure()

# 반투명 구 표면
fig.add_trace(go.Surface(
    x=x_sphere, y=y_sphere, z=z_sphere,
    colorscale=[[0, '#3366cc'], [1, '#3366cc']],
    opacity=0.2,
    showscale=False,
    hoverinfo='skip'
))

# 초기 직진 경로 (기준 점선)
fig.add_trace(go.Scatter3d(
    x=full_init[:, 0], y=full_init[:, 1], z=full_init[:, 2],
    mode='lines',
    line=dict(color='gray', width=3, dash='dash'),
    name='Initial Path (초기 직진 경로)'
))

# 최적화 경로 (실선)
fig.add_trace(go.Scatter3d(
    x=full_opt[:, 0], y=full_opt[:, 1], z=full_opt[:, 2],
    mode='lines+markers',
    line=dict(color='crimson', width=6),
    marker=dict(size=3, color='red'),
    name='Optimized Path (최적화 경로)'
))

# 시작점 A / 도착점 B 마커
fig.add_trace(go.Scatter3d(
    x=[ptA[0]], y=[ptA[1]], z=[ptA[2]],
    mode='markers+text',
    marker=dict(size=10, color='blue'),
    text=["A (시작점)"],
    textposition="top center",
    name='A (시작점)'
))

fig.add_trace(go.Scatter3d(
    x=[ptB[0]], y=[ptB[1]], z=[ptB[2]],
    mode='markers+text',
    marker=dict(size=10, color='green'),
    text=["B (도착점)"],
    textposition="top center",
    name='B (도착점)'
))

fig.update_layout(
    scene=dict(xaxis=dict(title='X'), yaxis=dict(title='Y'), zaxis=dict(title='Z'), aspectmode='data'),
    margin=dict(l=0, r=0, b=0, t=20),
    legend=dict(x=0.02, y=0.98),
    height=650
)

st.plotly_chart(fig, use_container_width=True)

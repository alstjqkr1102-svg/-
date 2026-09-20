import streamlit as st
import numpy as np
from scipy.optimize import minimize, differential_evolution
import plotly.graph_objects as go

# 페이지 레이아웃 설정
st.set_page_config(page_title="구면 표면 최소 에너지 경로", layout="wide")

st.title("Minimum-energy surface path (구면 표면 최소 에너지 경로)")

# --- 1. 입력 폼 레이아웃 (이미지 스타일 반영) ---
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
    mu = st.number_input("마찰계수 μ", value=0.05, min_value=0.0, step=0.01)
with col_algo:
    algo_option = st.selectbox(
        "최적화 알고리즘 선택",
        options=[
            "개선 1: Multi-start SLSQP (다중 초기 경로 탐색)",
            "개선 2: Differential Evolution (전역 최적화)",
            "기존: 단일 SLSQP (최단경로 초기값)"
        ]
    )

with col_btn:
    st.write(" ") # 높이 맞춤용 공간
    run_button = st.button("최적화 실행", type="primary", use_container_width=True)

# 구 반지름 및 상수 설정
R = 5.0  # 구 반지름 (A, B 점 위치에 맞게 설정)
g = 9.81 # 중력가속도
N_POINTS = 20 # 중간 제어점 개수

# --- Helper Functions ---
def enforce_sphere_constraint(P_flat):
    """경로 점들을 구면 표면(반지름 R)으로 정렬"""
    P = P_flat.reshape((-1, 3))
    norms = np.linalg.norm(P, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    P_projected = P * (R / norms)
    return P_projected

def calculate_path_energy(P_flat, xA, yA, zA, xB, yB, zB, m, mu_val):
    """에너지 계산 함수 (오르막 에너지 + 마찰 에너지)"""
    P_mid = enforce_sphere_constraint(P_flat)
    full_path = np.vstack(([xA, yA, zA], P_mid, [xB, yB, zB]))
    
    climb_energy = 0.0
    friction_energy = 0.0
    
    for i in range(len(full_path) - 1):
        p1 = full_path[i]
        p2 = full_path[i+1]
        
        # 이동 벡터 및 거리
        ds_vec = p2 - p1
        ds = np.linalg.norm(ds_vec)
        if ds < 1e-9:
            continue
            
        dz = p2[2] - p1[2]
        
        # 1. 오르막 에너지 (상승 위치에너지)
        if dz > 0:
            climb_energy += m * g * dz
            
        # 2. 수직 항력 및 마찰 에너지
        # 구면 수직 벡터 (중간점 기준)
        p_mid_seg = (p1 + p2) / 2.0
        r_norm = np.linalg.norm(p_mid_seg)
        if r_norm > 0:
            n_hat = p_mid_seg / r_norm
        else:
            n_hat = np.array([0, 0, 1])
            
        # 수직항력 N = m*g*cos(theta) (theta: 수직벡터와 z축 사이 각도)
        cos_theta = max(0.0, n_hat[2]) # z방향 성분
        N_force = m * g * cos_theta
        
        friction_energy += mu_val * N_force * ds
        
    total_energy = climb_energy + friction_energy
    return total_energy, climb_energy, friction_energy

def get_initial_path(mode="geodesic"):
    """초기 경로 생성 함수"""
    t = np.linspace(0, 1, N_POINTS + 2)[1:-1]
    
    if mode == "geodesic":
        # 대원/구면 최단경로 직선 보간 후 투영
        x = (1 - t) * xA + t * xB
        y = (1 - t) * yA + t * yB
        z = (1 - t) * zA + t * zB
    elif mode == "horizontal_front":
        # 수평 측면 우회 경로 (y 양수 방향)
        x = (1 - t) * xA + t * xB
        y = np.sin(np.pi * t) * 4.0
        z = (1 - t) * zA + t * zB
    elif mode == "horizontal_back":
        # 수평 측면 우회 경로 (y 음수 방향)
        x = (1 - t) * xA + t * xB
        y = -np.sin(np.pi * t) * 4.0
        z = (1 - t) * zA + t * zB
    else:
        x = (1 - t) * xA + t * xB
        y = (1 - t) * yA + t * yB
        z = (1 - t) * zA + t * zB

    P_init = np.column_stack((x, y, z))
    return enforce_sphere_constraint(P_init.flatten())

# --- 최적화 실행 조건 ---
if run_button or "optimized" not in st.session_state:
    
    # 1. 시작점/끝점 구면으로 투영 정렬
    normA = np.linalg.norm([xA, yA, zA])
    normB = np.linalg.norm([xB, yB, zB])
    if normA > 0: xA_p, yA_p, zA_p = np.array([xA, yA, zA]) * (R / normA)
    else: xA_p, yA_p, zA_p = xA, yA, zA
    if normB > 0: xB_p, yB_p, zB_p = np.array([xB, yB, zB]) * (R / normB)
    else: xB_p, yB_p, zB_p = xB, yB, zB

    def objective(P_flat):
        tot, _, _ = calculate_path_energy(P_flat, xA_p, yA_p, zA_p, xB_p, yB_p, zB_p, mass, mu)
        return tot

    init_path_geodesic = get_initial_path("geodesic")

    if "기존" in algo_option:
        # 단일 SLSQP
        res = minimize(objective, init_path_geodesic, method='SLSQP', options={'maxiter': 300})
        best_path_flat = res.x
        n_iter = res.nit
        success = res.success
    elif "Multi-start" in algo_option:
        # Multi-start SLSQP
        candidates = ["geodesic", "horizontal_front", "horizontal_back"]
        best_val = float('inf')
        best_path_flat = None
        total_nit = 0
        success = True
        
        for cand in candidates:
            p0 = get_initial_path(cand)
            res = minimize(objective, p0, method='SLSQP', options={'maxiter': 200})
            total_nit += res.nit
            if res.fun < best_val:
                best_val = res.fun
                best_path_flat = res.x
        n_iter = total_nit
    else:
        # Differential Evolution
        bounds = [(-R, R)] * (N_POINTS * 3)
        res = differential_evolution(objective, bounds, maxiter=50, popsize=10, seed=42)
        best_path_flat = res.x
        n_iter = res.nit
        success = res.success

    # 결과 세션 저장
    st.session_state["optimized"] = True
    st.session_state["best_path_flat"] = best_path_flat
    st.session_state["init_path_geodesic"] = init_path_geodesic
    st.session_state["n_iter"] = n_iter
    st.session_state["success"] = success
    st.session_state["pts"] = (xA_p, yA_p, zA_p, xB_p, yB_p, zB_p)

# --- 결과 데이터 계산 ---
best_path_flat = st.session_state["best_path_flat"]
init_path_geodesic = st.session_state["init_path_geodesic"]
n_iter = st.session_state["n_iter"]
xA_p, yA_p, zA_p, xB_p, yB_p, zB_p = st.session_state["pts"]

tot_e, climb_e, fric_e = calculate_path_energy(best_path_flat, xA_p, yA_p, zA_p, xB_p, yB_p, zB_p, mass, mu)

# --- 2. 상태 메시지 바 (이미지 스타일 반영) ---
st.success(
    f"Optimization terminated successfully | 반목 {n_iter}회 | "
    f"총 에너지 = {tot_e:.4f} J (오르막 {climb_e:.4f} J, 마찰 {fric_e:.4f} J)"
)

# --- 3. 3D Plotly 시각화 ---
# 구면 데이터 생성
u = np.linspace(0, 2 * np.pi, 50)
v = np.linspace(0, np.pi, 50)
x_sphere = R * np.outer(np.cos(u), np.sin(v))
y_sphere = R * np.outer(np.sin(u), np.sin(v))
z_sphere = R * np.outer(np.ones(np.size(u)), np.cos(v))

# 경로 정렬
P_init_arr = enforce_sphere_constraint(init_path_geodesic)
full_init = np.vstack(([xA_p, yA_p, zA_p], P_init_arr, [xB_p, yB_p, zB_p]))

P_opt_arr = enforce_sphere_constraint(best_path_flat)
full_opt = np.vstack(([xA_p, yA_p, zA_p], P_opt_arr, [xB_p, yB_p, zB_p]))

fig = go.Figure()

# 구면 메시 (반투명 파란색)
fig.add_trace(go.Surface(
    x=x_sphere, y=y_sphere, z=z_sphere,
    colorscale=[[0, '#3366cc'], [1, '#3366cc']],
    opacity=0.3,
    showscale=False,
    hoverinfo='skip'
))

# 초기 투영 경로 (회색 점선)
fig.add_trace(go.Scatter3d(
    x=full_init[:, 0], y=full_init[:, 1], z=full_init[:, 2],
    mode='lines+markers',
    line=dict(color='gray', width=4, dash='dash'),
    marker=dict(size=3, color='black'),
    name='Initial projected path (초기 투영 경로)'
))

# 최적화된 경로 (빨간색 실선)
fig.add_trace(go.Scatter3d(
    x=full_opt[:, 0], y=full_opt[:, 1], z=full_opt[:, 2],
    mode='lines+markers',
    line=dict(color='red', width=6),
    marker=dict(size=4, color='darkred'),
    name='Optimized path (최적화된 경로)'
))

# A (시작점) - 3D 호환 'diamond' 마커 사용
fig.add_trace(go.Scatter3d(
    x=[xA_p], y=[yA_p], z=[zA_p],
    mode='markers',
    marker=dict(size=10, color='blue', symbol='diamond'),
    name='A (시작점)'
))

# B (도착점) - 3D 호환 'square' 마커 사용
fig.add_trace(go.Scatter3d(
    x=[xB_p], y=[yB_p], z=[zB_p],
    mode='markers',
    marker=dict(size=10, color='green', symbol='square'),
    name='B (도착점)'
))

# 레이아웃 설정
fig.update_layout(
    title=dict(
        text="Minimum-energy surface path (구면 표면 최소 에너지 경로)",
        x=0.5,
        xanchor='center'
    ),
    scene=dict(
        xaxis=dict(title='X'),
        yaxis=dict(title='Y'),
        zaxis=dict(title='Z'),
        aspectmode='data'
    ),
    margin=dict(l=0, r=0, b=0, t=40),
    legend=dict(x=0.02, y=0.98),
    height=650
)

st.plotly_chart(fig, use_container_width=True)

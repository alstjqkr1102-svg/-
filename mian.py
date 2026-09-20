import streamlit as st
import numpy as np
import scipy.optimize as opt
import plotly.graph_objects as go

# ---------------------------------------------------------
# Page Setup
# ---------------------------------------------------------
st.set_page_config(page_title="구면 최적 경로 시뮬레이터 (개선판)", layout="wide")

st.title("🌐 구면 위 진짜 최소 에너지 경로 탐구 시뮬레이터")
st.markdown("""
본 시뮬레이션은 **구면 위의 두 점 $A(-4,0,3)$와 $B(4,0,3)$를 잇는 최소 에너지 경로**를 탐색합니다.
기존 단일 SLSQP의 국소 최적해(Local Minima) 문제를 극복하기 위해 **다중 초기값(Multi-start)** 및 **전역 최적화(Differential Evolution)** 기법을 제공합니다.
""")

# ---------------------------------------------------------
# Physical Parameters & Sphere Setup
# ---------------------------------------------------------
R = 5.0  # 구의 반지름 ( (-4)^2 + 0^2 + 3^2 = 25 = 5^2 )
m = 1.0  # 질량 (kg)
g = 9.81 # 중력 가속도 (m/s^2)

# 시작점 A와 도착점 B (구면 좌표계: r=R, theta=경도, phi=위도)
# A: (-4, 0, 3) -> z=3, r=5 -> sin(phi)=0.6, phi=arcsin(0.6), theta=pi
# B: (4, 0, 3)  -> z=3, r=5 -> sin(phi)=0.6, phi=arcsin(0.6), theta=0
phi_A = np.arcsin(3.0 / R)  # 약 0.6435 rad (36.87도)
theta_A = np.pi             # 180도

phi_B = np.arcsin(3.0 / R)  # 약 0.6435 rad (36.87도)
theta_B = 0.0               # 0도

# ---------------------------------------------------------
# Helper Functions: Coordinate Conversions & Energy Calc
# ---------------------------------------------------------
def spherical_to_cartesian(r, theta, phi):
    """구면좌표계 (r, theta, phi) -> 직교좌표계 (x, y, z)"""
    x = r * np.cos(phi) * np.cos(theta)
    y = r * np.cos(phi) * np.sin(theta)
    z = r * np.sin(phi)
    return x, y, z

def calculate_path_energy(phi_nodes, N=50, mu=0.2):
    """
    경로 제어점(phi_nodes)을 받아 총 에너지를 계산하는 함수
    - theta는 A(pi)에서 B(0)까지 등간격 배치
    - phi_nodes: 경유지의 위도 값들
    """
    theta_arr = np.linspace(theta_A, theta_B, N)
    
    # 제어점을 바탕으로 전체 경로 위도(phi) 보간
    num_controls = len(phi_nodes)
    control_theta = np.linspace(theta_A, theta_B, num_controls)
    phi_arr = np.interp(theta_arr, control_theta, phi_nodes)
    
    # 3D 위치 계산
    x, y, z = spherical_to_cartesian(R, theta_arr, phi_arr)
    
    # 1. 상승 에너지 (Rise Energy)
    # 시작점(z_A = 3.0) 대비 경로 상 최고 높이(z_max)에 도달하기 위한 위치 에너지
    z_max = np.max(z)
    delta_h = max(0.0, z_max - z[0])
    E_rise = m * g * delta_h
    
    # 2. 마찰 에너지 (Friction Energy)
    # 경로의 총 길이 L = sum(sqrt(dx^2 + dy^2 + dz^2))
    dx = np.diff(x)
    dy = np.diff(y)
    dz = np.diff(z)
    segment_lengths = np.sqrt(dx**2 + dy**2 + dz**2)
    total_length = np.sum(segment_lengths)
    
    E_friction = mu * m * g * total_length
    
    # 총 에너지
    E_total = E_rise + E_friction
    
    return E_total, E_rise, E_friction, total_length, x, y, z

# ---------------------------------------------------------
# Sidebar Controls
# ---------------------------------------------------------
st.sidebar.header("⚙️ 실험 및 알고리즘 설정")

mu = st.sidebar.slider("마찰 계수 (μ)", min_value=0.01, max_value=0.50, value=0.20, step=0.01)
num_intermediate = st.sidebar.slider("경로 중간 제어점 개수", min_value=3, max_value=15, value=7)

algorithm_option = st.sidebar.selectbox(
    "탐색 알고리즘 선택",
    [
        "기존 방식: 단일 SLSQP (최단경로 초기값)",
        "개선 방식 1: 다중 초기값 SLSQP (Multi-start)",
        "개선 방식 2: 전역 최적화 (Differential Evolution)"
    ]
)

# ---------------------------------------------------------
# Optimization Execution Logic
# ---------------------------------------------------------
N_points = 100
# 양 끝점(A, B)은 고정 (phi = phi_A = phi_B)
bounds = [(-np.pi/2 + 0.1, np.pi/2 - 0.1)] * num_intermediate

def objective_func(phi_mid):
    phi_full = np.concatenate([[phi_A], phi_mid, [phi_B]])
    e_tot, _, _, _, _, _, _ = calculate_path_energy(phi_full, N=N_points, mu=mu)
    return e_tot

best_phi_mid = None
opt_status_text = ""

if algorithm_option == "기존 방식: 단일 SLSQP (최단경로 초기값)":
    # 구면 위쪽을 넘어가는 대원(최단거리) 경로 초기화
    init_phi = np.linspace(phi_A, np.pi/2 * 0.8, num_intermediate // 2 + 1)
    init_phi = np.concatenate([init_phi, init_phi[::-1][1:]])
    if len(init_phi) > num_intermediate:
        init_phi = init_phi[:num_intermediate]
    elif len(init_phi) < num_intermediate:
        init_phi = np.pad(init_phi, (0, num_intermediate - len(init_phi)), 'edge')
        
    res = opt.minimize(objective_func, init_phi, method='SLSQP', bounds=bounds)
    best_phi_mid = res.x
    opt_status_text = "단일 최단경로 초기값에서 SLSQP 수행 완료 (지역 최적해 정체 가능성 높음)"

elif algorithm_option == "개선 방식 1: 다중 초기값 SLSQP (Multi-start)":
    # 4가지 서로 다른 초기 경로 생성 (상단, 등고선 수평, 하단, 사선)
    candidates = [
        np.full(num_intermediate, phi_A),                          # 수평 등고선 경로
        np.linspace(phi_A, np.pi/2.5, num_intermediate),            # 상단 우회 경로
        np.full(num_intermediate, 0.0),                             # 적도/하단 우회 경로
        np.linspace(phi_A, -np.pi/4, num_intermediate)             # 남반구 우회 경로
    ]
    
    best_energy = float('inf')
    for idx, init_p in enumerate(candidates):
        res = opt.minimize(objective_func, init_p, method='SLSQP', bounds=bounds)
        if res.fun < best_energy:
            best_energy = res.fun
            best_phi_mid = res.x
    opt_status_text = "4가지 다양한 초기 경로(상단, 수평, 하단)에서 SLSQP 수행 후 최적 경로 선택 완료"

else: # Differential Evolution
    res = opt.differential_evolution(objective_func, bounds=bounds, seed=42, maxiter=100)
    best_phi_mid = res.x
    opt_status_text = "전역 탐색 알고리즘(Differential Evolution) 수행 완료"

# 최적 경로 풀 생성 및 결과 계산
opt_phi_full = np.concatenate([[phi_A], best_phi_mid, [phi_B]])
e_tot, e_rise, e_fric, path_len, px, py, pz = calculate_path_energy(opt_phi_full, N=N_points, mu=mu)

# 참고용 비교 데이터 (순수 수평 등고선 경로 vs 순수 상단 최단경로)
flat_phi = np.full(num_intermediate + 2, phi_A)
e_tot_flat, e_rise_flat, e_fric_flat, len_flat, fx, fy, fz = calculate_path_energy(flat_phi, N=N_points, mu=mu)

# ---------------------------------------------------------
# Dashboard Metrics
# ---------------------------------------------------------
st.info(f"💡 **알고리즘 진행 상태:** {opt_status_text}")

col1, col2, col3, col4 = st.columns(4)
col1.metric("총 에너지 (Total Energy)", f"{e_tot:.2f} J", delta=f"{e_tot - e_tot_flat:.2f} J vs 수평경로", delta_color="inverse")
col2.metric("상승 위치에너지 (Rise)", f"{e_rise:.2f} J")
col3.metric("마찰 손실에너지 (Friction)", f"{e_fric:.2f} J")
col4.metric("이동 경로 길이 (Length)", f"{path_len:.2f} m")

# ---------------------------------------------------------
# 3D Visualization using Plotly
# ---------------------------------------------------------
st.subheader("📌 3D 경로 비교 시각화")

# 구면 메시 생성
u = np.linspace(0, 2 * np.pi, 60)
v = np.linspace(-np.pi / 2, np.pi / 2, 60)
sx = R * np.outer(np.cos(v), np.cos(u))
sy = R * np.outer(np.cos(v), np.sin(u))
sz = R * np.outer(np.sin(v), np.ones(np.size(u)))

fig = go.Figure()

# 1. 반투명 구면 렌더링
fig.add_trace(go.Surface(
    x=sx, y=sy, z=sz,
    opacity=0.25,
    colorscale='Blues',
    showscale=False,
    hoverinfo='skip'
))

# 2. 시작점 A & 도착점 B 표시
xA, yA, zA = spherical_to_cartesian(R, theta_A, phi_A)
xB, yB, zB = spherical_to_cartesian(R, theta_B, phi_B)

fig.add_trace(go.Scatter3d(
    x=[xA, xB], y=[yA, yB], z=[zA, zB],
    mode='markers+text',
    marker=dict(size=8, color=['red', 'green']),
    text=['A (-4,0,3)', 'B (4,0,3)'],
    textposition="top center",
    name='시작점/도착점'
))

# 3. 기준 수평 경로 (등고선 경로 z=3)
fig.add_trace(go.Scatter3d(
    x=fx, y=fy, z=fz,
    mode='lines',
    line=dict(color='gray', width=4, dash='dash'),
    name=f'수평 우회 경로 (z=3 고정, E={e_tot_flat:.2f}J)'
))

# 4. 탐색된 최적화 경로
fig.add_trace(go.Scatter3d(
    x=px, y=py, z=pz,
    mode='lines+markers',
    line=dict(color='crimson', width=7),
    marker=dict(size=3),
    name=f'현재 최적 탐색 경로 (E={e_tot:.2f}J)'
))

fig.update_layout(
    scene=dict(
        xaxis_title='X (m)',
        yaxis_title='Y (m)',
        zaxis_title='Z (m)',
        aspectmode='data'
    ),
    height=650,
    margin=dict(l=0, r=0, b=0, t=30)
)

st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------
# Educational Analysis / Summary Section
# ---------------------------------------------------------
st.subheader("🔍 탐구 분석 및 알고리즘 비교")

tab1, tab2 = st.tabs(["📊 알고리즘별 결과 비교", "📐 물리적/수학적 원리"])

with tab1:
    st.markdown("""
    ### 왜 기존 SLSQP는 수평 경로를 찾지 못했는가?
    - **경사하강법의 한계**: SLSQP와 같은 경사기반(Gradient-based) 최적화 알고리즘은 **초기 탐색 지점 근처의 기울기**만 따라 이동합니다.
    - **지역 최적해(Local Minima) 덫**: 구 위쪽을 지나는 최단거리 경로에서 탐색을 시작하면, 조금 이동했을 때 상승 높이가 크게 줄지 않아 경사면을 벗어나지 못하고 산봉우리 근처에 갇히게 됩니다.
    
    ### 해결책 비교:
    1. **Multi-start SLSQP**: 수평 우회 경로($z=3$)를 초기값 중 하나로 넣어줌으로써 상승 에너지가 0에 가까운 전역 최적해 영역에 빠르게 수렴합니다.
    2. **Differential Evolution**: 무작위 변이와 교배를 통해 구면 전체 공간을 탐색하므로 초기 경로 설정 없이도 스스로 수평 우회 경로를 발견합니다.
    """)

with tab2:
    st.markdown("""
    ### 에너지 방정식을 통한 분석
    $$ E_{total} = m g \Delta h_{max} + \mu m g L $$
    
    - **경로 1 (구 상단 최단경로)**: 
      - 거리 $L \approx 9.27\text{m}$ (짧음)
      - 상승 높이 $\Delta h \approx 2.0\text{m}$ ($z=3 \rightarrow z=5$)
      - $E_{total} \approx 19.62\text{J} + 9.10\text{J} = \mathbf{28.72\text{J}}$
      
    - **경로 2 (수평 우회 경로 $z=3$)**:
      - 거리 $L = R \cos(\phi) \cdot \pi = 5 \times 0.8 \times \pi \approx 12.57\text{m}$ (더 닒)
      - 상승 높이 $\Delta h = 0\text{m}$
      - $E_{total} = 0\text{J} + (0.2 \times 1.0 \times 9.81 \times 12.57) \approx \mathbf{12.33\text{J}}$
      
    👉 **결론**: 거리는 약 35% 길어지지만, 상승 에너지를 완전 배제함으로써 **총 소모 에너지를 50% 이상 절감**할 수 있습니다.
    """)

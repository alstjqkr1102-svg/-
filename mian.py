import streamlit as st
import numpy as np
import plotly.graph_objects as go
from scipy.optimize import minimize

# ============================================================
# 기본 설정
# ============================================================

R = 5.0
N = 25
G = 9.81


# ============================================================
# 구면 관련 함수
# ============================================================

def project_to_sphere(p):
    """임의의 3차원 점을 반지름 R인 구면 위로 투영"""
    p = np.asarray(p, dtype=float)
    norm = np.linalg.norm(p)

    if norm < 1e-12:
        raise ValueError("원점 (0,0,0)은 구면 위의 점으로 투영할 수 없습니다.")

    return R * p / norm


def cartesian_to_spherical(p):
    """
    Cartesian -> spherical
    phi   : 0 ~ pi
    theta : -pi ~ pi
    """
    x, y, z = p

    phi = np.arccos(np.clip(z / R, -1.0, 1.0))
    theta = np.arctan2(y, x)

    return theta, phi


def spherical_to_cartesian(theta, phi):
    """spherical -> Cartesian"""
    x = R * np.sin(phi) * np.cos(theta)
    y = R * np.sin(phi) * np.sin(theta)
    z = R * np.cos(phi)

    return np.array([x, y, z])


def path_from_angles(A, B, variables):
    """
    최적화 변수로부터 전체 경로 생성

    variables:
        theta 23개
        phi   23개
        uphill auxiliary variables 24개
    """

    intermediate_count = N - 2

    theta = variables[:intermediate_count]
    phi = variables[
        intermediate_count:
        2 * intermediate_count
    ]

    points = [A]

    for t, p in zip(theta, phi):
        points.append(spherical_to_cartesian(t, p))

    points.append(B)

    return np.array(points)


def surface_distance(p1, p2):
    """구면 위 두 점 사이의 최단 호 길이"""
    dot = np.dot(p1, p2) / (R * R)
    dot = np.clip(dot, -1.0, 1.0)

    angle = np.arccos(dot)

    return R * angle


def path_energy(points, mass, mu):
    """전체 에너지 계산"""

    z = points[:, 2]

    dz = np.diff(z)

    uphill = mass * G * np.sum(np.maximum(dz, 0))

    distances = [
        surface_distance(points[i], points[i + 1])
        for i in range(len(points) - 1)
    ]

    friction = mu * mass * G * np.sum(distances)

    total = uphill + friction

    return total, uphill, friction


# ============================================================
# 초기 경로
# ============================================================

def create_initial_path(A, B):
    """
    A -> B 직선 구간을 N개 점으로 나눈 후
    각각을 구면 위로 투영
    """

    path = []

    for t in np.linspace(0, 1, N):
        p = A * (1 - t) + B * t
        path.append(project_to_sphere(p))

    return np.array(path)


# ============================================================
# SLSQP 최적화
# ============================================================

def optimize_path(A, B, mass, mu):

    initial_path = create_initial_path(A, B)

    intermediate_count = N - 2

    theta = []
    phi = []

    for p in initial_path[1:-1]:
        t, f = cartesian_to_spherical(p)
        theta.append(t)
        phi.append(f)

    theta = np.array(theta)
    phi = np.array(phi)

    # 상승량 보조변수
    dz = np.diff(initial_path[:, 2])
    uphill_variables = np.maximum(dz, 0)

    x0 = np.concatenate([
        theta,
        phi,
        uphill_variables
    ])

    # --------------------------------------------------------
    # 목적함수
    # --------------------------------------------------------

    def objective(x):

        points = path_from_angles(A, B, x)

        uphill_vars = x[
            2 * intermediate_count:
        ]

        distances = [
            surface_distance(points[i], points[i + 1])
            for i in range(N - 1)
        ]

        friction = mu * mass * G * np.sum(distances)

        uphill = mass * G * np.sum(uphill_vars)

        return uphill + friction

    # --------------------------------------------------------
    # 제약조건
    #
    # u_i >= z_(i+1) - z_i
    # u_i >= 0
    # --------------------------------------------------------

    def constraints(x):

        points = path_from_angles(A, B, x)

        u = x[
            2 * intermediate_count:
        ]

        dz = np.diff(points[:, 2])

        return u - dz

    # phi는 0 ~ pi
    bounds = []

    # theta
    for _ in range(intermediate_count):
        bounds.append((-10 * np.pi, 10 * np.pi))

    # phi
    for _ in range(intermediate_count):
        bounds.append((0, np.pi))

    # uphill auxiliary variables
    for _ in range(N - 1):
        bounds.append((0, None))

    result = minimize(
        objective,
        x0,
        method="SLSQP",
        bounds=bounds,
        constraints={
            "type": "ineq",
            "fun": constraints
        },
        options={
            "maxiter": 1000,
            "ftol": 1e-9,
            "disp": False
        }
    )

    optimized_path = path_from_angles(A, B, result.x)

    # 실제 물리식으로 다시 계산
    total, uphill, friction = path_energy(
        optimized_path,
        mass,
        mu
    )

    # 최적화 실패 또는 초기값보다 나쁜 경우
    # 초기 경로를 결과로 사용
    initial_energy = path_energy(
        initial_path,
        mass,
        mu
    )

    if (not result.success) or total > initial_energy[0]:
        optimized_path = initial_path.copy()

        total, uphill, friction = path_energy(
            optimized_path,
            mass,
            mu
        )

        success = False
    else:
        success = True

    return (
        initial_path,
        optimized_path,
        total,
        uphill,
        friction,
        result,
        success
    )


# ============================================================
# Plotly 3D 시각화
# ============================================================

def create_figure(
    initial_path,
    optimized_path,
    A,
    B
):

    fig = go.Figure()

    # --------------------------------------------------------
    # 구
    # --------------------------------------------------------

    u = np.linspace(0, 2 * np.pi, 60)
    v = np.linspace(0, np.pi, 30)

    x = R * np.outer(np.cos(u), np.sin(v))
    y = R * np.outer(np.sin(u), np.sin(v))
    z = R * np.outer(np.ones_like(u), np.cos(v))

    fig.add_trace(
        go.Surface(
            x=x,
            y=y,
            z=z,
            opacity=0.35,
            showscale=False,
            name="Sphere (구)"
        )
    )

    # --------------------------------------------------------
    # 초기 경로
    # --------------------------------------------------------

    fig.add_trace(
        go.Scatter3d(
            x=initial_path[:, 0],
            y=initial_path[:, 1],
            z=initial_path[:, 2],
            mode="lines+markers",
            name="Initial Path (초기 경로)",
            line=dict(
                color="gray",
                width=3,
                dash="dot"
            ),
            marker=dict(
                size=3
            )
        )
    )

    # --------------------------------------------------------
    # 최적 경로
    # --------------------------------------------------------

    fig.add_trace(
        go.Scatter3d(
            x=optimized_path[:, 0],
            y=optimized_path[:, 1],
            z=optimized_path[:, 2],
            mode="lines+markers",
            name="Optimized Path (최적 경로)",
            line=dict(
                color="red",
                width=6
            ),
            marker=dict(
                size=4
            )
        )
    )

    # --------------------------------------------------------
    # A
    # --------------------------------------------------------

    fig.add_trace(
        go.Scatter3d(
            x=[A[0]],
            y=[A[1]],
            z=[A[2]],
            mode="markers+text",
            text=["A"],
            textposition="top center",
            name="Start A (시작점 A)",
            marker=dict(
                size=12,
                color="black",
                symbol="star"
            )
        )
    )

    # --------------------------------------------------------
    # B
    # --------------------------------------------------------

    fig.add_trace(
        go.Scatter3d(
            x=[B[0]],
            y=[B[1]],
            z=[B[2]],
            mode="markers+text",
            text=["B"],
            textposition="top center",
            name="Goal B (도착점 B)",
            marker=dict(
                size=12,
                color="black",
                symbol="star"
            )
        )
    )

    fig.update_layout(
        title="Minimum-Energy Path on a Sphere (구면 위 최소 에너지 경로)",
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
            aspectmode="cube"
        ),
        height=700,
        legend=dict(
            x=0,
            y=1
        )
    )

    return fig


# ============================================================
# Streamlit UI
# ============================================================

st.set_page_config(
    page_title="Spherical Minimum-Energy Simulator",
    page_icon="🌐",
    layout="wide"
)

st.title(
    "Spherical Minimum-Energy Path Simulator "
    "(구면 위 최소 에너지 경로 시뮬레이터)"
)

st.write(
    "구의 표면 위에서 시작점 A에서 도착점 B까지 이동할 때 "
    "중력에 의한 상승 에너지와 마찰 에너지를 고려하여 "
    "에너지가 작은 경로를 계산합니다."
)

# ============================================================
# 사이드바
# ============================================================

st.sidebar.header("Simulation Settings (시뮬레이션 설정)")

st.sidebar.write(f"Sphere Radius R (구 반지름): **{R}**")
st.sidebar.write(f"Stepping Points N (점 개수): **{N}**")

st.sidebar.subheader("Point A (시작점 A)")

Ax = st.sidebar.number_input(
    "A x",
    value=5.0,
    step=0.1
)

Ay = st.sidebar.number_input(
    "A y",
    value=0.0,
    step=0.1
)

Az = st.sidebar.number_input(
    "A z",
    value=0.0,
    step=0.1
)

st.sidebar.subheader("Point B (도착점 B)")

Bx = st.sidebar.number_input(
    "B x",
    value=0.0,
    step=0.1
)

By = st.sidebar.number_input(
    "B y",
    value=5.0,
    step=0.1
)

Bz = st.sidebar.number_input(
    "B z",
    value=0.0,
    step=0.1
)

st.sidebar.subheader("Physics (물리량)")

mass = st.sidebar.number_input(
    "Mass m (질량, kg)",
    min_value=0.001,
    value=1.0,
    step=0.1
)

mu = st.sidebar.number_input(
    "Friction μ (마찰계수)",
    min_value=0.0,
    value=0.05,
    step=0.01
)

run = st.sidebar.button(
    "Run Simulation (시뮬레이션 실행)",
    type="primary"
)

# ============================================================
# 실행
# ============================================================

if run:

    A_raw = np.array([
        Ax,
        Ay,
        Az
    ])

    B_raw = np.array([
        Bx,
        By,
        Bz
    ])

    try:

        A = project_to_sphere(A_raw)
        B = project_to_sphere(B_raw)

        (
            initial_path,
            optimized_path,
            total,
            uphill,
            friction,
            result,
            success
        ) = optimize_path(
            A,
            B,
            mass,
            mu
        )

        # ----------------------------------------------------
        # 투영 결과
        # ----------------------------------------------------

        st.subheader(
            "Projected Points (구면 위로 투영된 점)"
        )

        col1, col2 = st.columns(2)

        with col1:
            st.write(
                f"A → "
                f"({A[0]:.4f}, {A[1]:.4f}, {A[2]:.4f})"
            )

        with col2:
            st.write(
                f"B → "
                f"({B[0]:.4f}, {B[1]:.4f}, {B[2]:.4f})"
            )

        # ----------------------------------------------------
        # 에너지
        # ----------------------------------------------------

        st.subheader(
            "Energy Result (에너지 결과)"
        )

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Total Energy (전체)",
                f"{total:.6f} J"
            )

        with col2:
            st.metric(
                "Uphill Energy (상승)",
                f"{uphill:.6f} J"
            )

        with col3:
            st.metric(
                "Friction Energy (마찰)",
                f"{friction:.6f} J"
            )

        if success:
            st.success(
                "SLSQP optimization completed. "
                "(SLSQP 최적화가 완료되었습니다.)"
            )
        else:
            st.warning(
                "Optimization did not improve the initial path, "
                "so the initial path is displayed. "
                "(최적화 결과가 초기 경로보다 좋지 않아 초기 경로를 표시합니다.)"
            )

        # ----------------------------------------------------
        # 3D
        # ----------------------------------------------------

        st.subheader(
            "3D Simulation (3차원 시뮬레이션)"
        )

        fig = create_figure(
            initial_path,
            optimized_path,
            A,
            B
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        # ----------------------------------------------------
        # 최적화 정보
        # ----------------------------------------------------

        st.subheader(
            "Optimization Information (최적화 정보)"
        )

        st.write(
            f"- Method: **SLSQP**"
        )

        st.write(
            f"- Iterations: **{result.nit}**"
        )

        st.write(
            f"- Function evaluations: **{result.nfev}**"
        )

        st.write(
            f"- Sphere constraint: "
            f"all points remain on R = {R}"
        )

    except ValueError as e:

        st.error(str(e))

else:

    st.info(
        "왼쪽에서 A, B와 물리량을 설정한 후 "
        "**Run Simulation** 버튼을 눌러주세요."
    )

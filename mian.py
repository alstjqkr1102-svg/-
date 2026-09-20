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
# 구면 투영
# ============================================================

def project_to_sphere(point):
    point = np.asarray(point, dtype=float)
    norm = np.linalg.norm(point)

    if norm < 1e-12:
        raise ValueError(
            "원점 (0, 0, 0)은 구면 위로 투영할 수 없습니다."
        )

    return R * point / norm


# ============================================================
# Cartesian <-> Spherical
# ============================================================

def cartesian_to_spherical(point):
    x, y, z = point

    phi = np.arccos(
        np.clip(z / R, -1.0, 1.0)
    )

    theta = np.arctan2(y, x)

    return theta, phi


def spherical_to_cartesian(theta, phi):
    x = R * np.sin(phi) * np.cos(theta)
    y = R * np.sin(phi) * np.sin(theta)
    z = R * np.cos(phi)

    return np.array([x, y, z])


# ============================================================
# 초기 경로 생성
# ============================================================

def create_initial_path(A, B):
    path = []

    for t in np.linspace(0.0, 1.0, N):
        point = A * (1.0 - t) + B * t
        point = project_to_sphere(point)
        path.append(point)

    return np.array(path)


# ============================================================
# 구면 위 거리
# ============================================================

def surface_distance(p1, p2):
    dot = np.dot(p1, p2) / (R * R)

    dot = np.clip(dot, -1.0, 1.0)

    angle = np.arccos(dot)

    return R * angle


# ============================================================
# 에너지 계산
# ============================================================

def calculate_energy(path, mass, mu):

    z = path[:, 2]

    dz = np.diff(z)

    # 상승할 때만 위치에너지 증가
    uphill = (
        mass
        * G
        * np.sum(np.maximum(dz, 0.0))
    )

    # 구면 위 이동거리
    distances = []

    for i in range(N - 1):
        distances.append(
            surface_distance(
                path[i],
                path[i + 1]
            )
        )

    total_distance = np.sum(distances)

    # 마찰 에너지
    friction = (
        mu
        * mass
        * G
        * total_distance
    )

    total = uphill + friction

    return total, uphill, friction, total_distance


# ============================================================
# 최적화 변수로부터 경로 생성
# ============================================================

def build_path(A, B, variables):

    intermediate_count = N - 2

    theta = variables[
        :intermediate_count
    ]

    phi = variables[
        intermediate_count:
        2 * intermediate_count
    ]

    path = [A]

    for i in range(intermediate_count):

        point = spherical_to_cartesian(
            theta[i],
            phi[i]
        )

        path.append(point)

    path.append(B)

    return np.array(path)


# ============================================================
# SLSQP 최적화
# ============================================================

def optimize_path(A, B, mass, mu):

    initial_path = create_initial_path(A, B)

    intermediate_count = N - 2

    # --------------------------------------------------------
    # 초기 spherical coordinate
    # --------------------------------------------------------

    theta_initial = []
    phi_initial = []

    for point in initial_path[1:-1]:

        theta, phi = cartesian_to_spherical(
            point
        )

        theta_initial.append(theta)
        phi_initial.append(phi)

    theta_initial = np.array(theta_initial)
    phi_initial = np.array(phi_initial)

    # --------------------------------------------------------
    # 상승량 보조 변수
    # --------------------------------------------------------

    dz = np.diff(
        initial_path[:, 2]
    )

    uphill_initial = np.maximum(
        dz,
        0.0
    )

    # --------------------------------------------------------
    # 전체 초기 변수
    # --------------------------------------------------------

    x0 = np.concatenate([
        theta_initial,
        phi_initial,
        uphill_initial
    ])

    # --------------------------------------------------------
    # 목적함수
    # --------------------------------------------------------

    def objective(x):

        path = build_path(
            A,
            B,
            x
        )

        uphill_variables = x[
            2 * intermediate_count:
        ]

        # 마찰 에너지
        distances = []

        for i in range(N - 1):

            distances.append(
                surface_distance(
                    path[i],
                    path[i + 1]
                )
            )

        friction = (
            mu
            * mass
            * G
            * np.sum(distances)
        )

        # 상승 에너지
        uphill = (
            mass
            * G
            * np.sum(uphill_variables)
        )

        return uphill + friction

    # --------------------------------------------------------
    # 제약조건
    #
    # u >= z(i+1) - z(i)
    # --------------------------------------------------------

    def constraint_function(x):

        path = build_path(
            A,
            B,
            x
        )

        uphill_variables = x[
            2 * intermediate_count:
        ]

        dz = np.diff(
            path[:, 2]
        )

        return uphill_variables - dz

    # --------------------------------------------------------
    # 변수 범위
    # --------------------------------------------------------

    bounds = []

    # theta
    for _ in range(intermediate_count):

        bounds.append(
            (-10 * np.pi, 10 * np.pi)
        )

    # phi
    for _ in range(intermediate_count):

        bounds.append(
            (0.0, np.pi)
        )

    # 상승량
    for _ in range(N - 1):

        bounds.append(
            (0.0, None)
        )

    # --------------------------------------------------------
    # SLSQP
    # --------------------------------------------------------

    result = minimize(

        objective,

        x0,

        method="SLSQP",

        bounds=bounds,

        constraints={
            "type": "ineq",
            "fun": constraint_function
        },

        options={
            "maxiter": 1000,
            "ftol": 1e-9,
            "disp": False
        }
    )

    # --------------------------------------------------------
    # 최적 경로
    # --------------------------------------------------------

    optimized_path = build_path(
        A,
        B,
        result.x
    )

    # 실제 에너지 계산
    optimized_energy = calculate_energy(
        optimized_path,
        mass,
        mu
    )

    initial_energy = calculate_energy(
        initial_path,
        mass,
        mu
    )

    # --------------------------------------------------------
    # 최적화 결과가 초기 경로보다 나쁜 경우
    # --------------------------------------------------------

    if (
        not result.success
        or optimized_energy[0] > initial_energy[0]
    ):

        optimized_path = initial_path.copy()

        final_energy = initial_energy

        optimization_success = False

    else:

        final_energy = optimized_energy

        optimization_success = True

    return (
        initial_path,
        optimized_path,
        initial_energy,
        final_energy,
        result,
        optimization_success
    )


# ============================================================
# 3D Plotly 그래프
# ============================================================

def create_3d_figure(
    initial_path,
    optimized_path,
    A,
    B
):

    fig = go.Figure()

    # --------------------------------------------------------
    # 구 생성
    # --------------------------------------------------------

    u = np.linspace(
        0,
        2 * np.pi,
        60
    )

    v = np.linspace(
        0,
        np.pi,
        30
    )

    x = R * np.outer(
        np.cos(u),
        np.sin(v)
    )

    y = R * np.outer(
        np.sin(u),
        np.sin(v)
    )

    z = R * np.outer(
        np.ones_like(u),
        np.cos(v)
    )

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
                size=3,
                color="gray"
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
                size=4,
                color="red"
            )
        )
    )

    # --------------------------------------------------------
    # 시작점 A
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

                # Scatter3d에서 지원되는 모양
                symbol="diamond"
            )
        )
    )

    # --------------------------------------------------------
    # 도착점 B
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

                # Scatter3d에서 지원되는 모양
                symbol="diamond"
            )
        )
    )

    # --------------------------------------------------------
    # 그래프 설정
    # --------------------------------------------------------

    fig.update_layout(

        title=(
            "Minimum-Energy Path on a Sphere "
            "(구면 위 최소 에너지 경로)"
        ),

        height=700,

        scene=dict(

            xaxis_title="X",

            yaxis_title="Y",

            zaxis_title="Z",

            aspectmode="cube"
        ),

        legend=dict(

            x=0,

            y=1
        )
    )

    return fig


# ============================================================
# Streamlit 기본 설정
# ============================================================

st.set_page_config(

    page_title=(
        "Spherical Minimum-Energy Simulator"
    ),

    page_icon="🌐",

    layout="wide"
)


# ============================================================
# 제목
# ============================================================

st.title(

    "Spherical Minimum-Energy Path Simulator "
    "(구면 위 최소 에너지 경로 시뮬레이터)"
)

st.write(

    """
    구의 표면 위에서 시작점 A부터 도착점 B까지 이동할 때
    상승 에너지와 마찰 에너지를 고려하여
    총 에너지가 작은 경로를 계산합니다.
    """
)


# ============================================================
# 사이드바
# ============================================================

st.sidebar.header(
    "Simulation Settings (시뮬레이션 설정)"
)

st.sidebar.write(
    f"Sphere Radius R (구 반지름): **{R}**"
)

st.sidebar.write(
    f"Stepping Points N (점 개수): **{N}**"
)


# ============================================================
# A 입력
# ============================================================

st.sidebar.subheader(
    "Point A (시작점 A)"
)

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


# ============================================================
# B 입력
# ============================================================

st.sidebar.subheader(
    "Point B (도착점 B)"
)

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


# ============================================================
# 물리량
# ============================================================

st.sidebar.subheader(
    "Physics (물리량)"
)

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


# ============================================================
# 실행 버튼
# ============================================================

run = st.sidebar.button(

    "Run Simulation "
    "(시뮬레이션 실행)",

    type="primary"
)


# ============================================================
# 시뮬레이션 실행
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

        # ----------------------------------------------------
        # 입력점을 구면 위로 투영
        # ----------------------------------------------------

        A = project_to_sphere(
            A_raw
        )

        B = project_to_sphere(
            B_raw
        )

        # ----------------------------------------------------
        # 최적화
        # ----------------------------------------------------

        (
            initial_path,
            optimized_path,
            initial_energy,
            final_energy,
            result,
            optimization_success
        ) = optimize_path(

            A,
            B,
            mass,
            mu
        )

        # ----------------------------------------------------
        # 투영된 A/B
        # ----------------------------------------------------

        st.subheader(
            "Projected Points "
            "(구면 위로 투영된 점)"
        )

        col1, col2 = st.columns(2)

        with col1:

            st.write(
                f"""
                **A**

                ({A[0]:.4f},
                {A[1]:.4f},
                {A[2]:.4f})
                """
            )

        with col2:

            st.write(
                f"""
                **B**

                ({B[0]:.4f},
                {B[1]:.4f},
                {B[2]:.4f})
                """
            )

        # ----------------------------------------------------
        # 에너지 계산
        # ----------------------------------------------------

        initial_total = initial_energy[0]

        final_total = final_energy[0]

        uphill = final_energy[1]

        friction = final_energy[2]

        distance = final_energy[3]

        # ----------------------------------------------------
        # 결과
        # ----------------------------------------------------

        st.subheader(
            "Energy Result "
            "(에너지 결과)"
        )

        col1, col2, col3 = st.columns(3)

        with col1:

            st.metric(
                "Total Energy (전체)",
                f"{final_total:.6f} J"
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

        st.write(
            f"Initial Energy (초기 에너지): "
            f"**{initial_total:.6f} J**"
        )

        st.write(
            f"Final Path Distance (최종 경로 길이): "
            f"**{distance:.6f} m**"
        )

        # ----------------------------------------------------
        # 최적화 상태
        # ----------------------------------------------------

        if optimization_success:

            st.success(
                "SLSQP optimization completed successfully. "
                "(SLSQP 최적화가 완료되었습니다.)"
            )

        else:

            st.warning(
                "최적화 결과가 초기 경로보다 좋지 않아 "
                "초기 경로를 최종 결과로 사용했습니다."
            )

        # ----------------------------------------------------
        # 3D 시뮬레이션
        # ----------------------------------------------------

        st.subheader(
            "3D Simulation "
            "(3차원 시뮬레이션)"
        )

        fig = create_3d_figure(

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
            "Optimization Information "
            "(최적화 정보)"
        )

        st.write(
            f"**Method:** SLSQP"
        )

        st.write(
            f"**Number of points:** {N}"
        )

        st.write(
            f"**Iterations:** {result.nit}"
        )

        st.write(
            f"**Function evaluations:** {result.nfev}"
        )

        st.write(
            f"**Sphere radius:** {R}"
        )

        # ----------------------------------------------------
        # 에너지 비교 그래프
        # ----------------------------------------------------

        st.subheader(
            "Energy Comparison "
            "(에너지 비교)"
        )

        energy_fig = go.Figure()

        energy_fig.add_trace(

            go.Bar(

                x=[
                    "Initial Path (초기 경로)",
                    "Optimized Path (최적 경로)"
                ],

                y=[
                    initial_total,
                    final_total
                ],

                text=[
                    f"{initial_total:.4f} J",
                    f"{final_total:.4f} J"
                ],

                textposition="auto"
            )
        )

        energy_fig.update_layout(

            title=(
                "Initial vs Optimized Energy "
                "(초기 경로와 최적 경로의 에너지 비교)"
            ),

            yaxis_title="Energy (J)",

            height=450
        )

        st.plotly_chart(

            energy_fig,

            use_container_width=True
        )

    except ValueError as error:

        st.error(
            str(error)
        )

else:

    st.info(
        """
        왼쪽에서 시작점 A와 도착점 B를 입력하고
        질량과 마찰계수를 설정한 뒤
        **Run Simulation (시뮬레이션 실행)** 버튼을 눌러주세요.
        """
    )

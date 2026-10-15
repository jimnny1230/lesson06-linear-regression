import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# 페이지 설정
st.set_page_config(
    page_title="서울 기온 예측기", page_icon="🌡️", layout="wide"
)

st.title("🌡️ 서울 연평균 기온 분석 및 예측기")
st.markdown(
    "서울의 과거 기온 데이터(1900년~2025년 기준)를 바탕으로 연도별 평균기온을 분석하고, 선형 회귀 분석을 통해 미래 기온을 예측합니다."
)


# 데이터 로드 함수 (캐싱 적용)
@st.cache_data
def load_data():
  url = "https://raw.githubusercontent.com/greatsong/modudata/bb860932644270ad1199f10d3e7670e30231bce4/data/seoul.csv"
  df = pd.read_csv(url, encoding="utf-8")
  df["날짜"] = pd.to_datetime(df["날짜"])
  df["연도"] = df["날짜"].dt.year
  return df


try:
  df = load_data()
except Exception as e:
  st.error(f"데이터를 불러오는 중 오류가 발생했습니다: {e}")
  st.stop()

# 1. 기준 기간 2025년까지 필터링
df_filtered = df[df["연도"] <= 2025]

# 2. 관측일이 300일 미만인 해 제외
year_counts = df_filtered.groupby("연도")["평균기온"].count()
valid_years = year_counts[year_counts >= 300].index
df_valid = df_filtered[df_filtered["연도"].isin(valid_years)]

# 3. 연도별 평균기온 계산
annual_mean = (
    df_valid.groupby("연도")["평균기온"].mean().reset_index(name="연평균기온")
)

# 4. 전체 기간 선형 회귀 분석 및 상관계수 계산
X_all = annual_mean["연도"]
y_all = annual_mean["연평균기온"]
slope_all, intercept_all = np.polyfit(X_all, y_all, 1)
correlation = X_all.corr(y_all)

# 100년당 기온 상승 폭 계산 (°C / 100년)
slope_100_all = slope_all * 100

# 5. 최근 20년 기간 선형 회귀 분석 (예: 최근 20개년 데이터)
recent_years_df = annual_mean[
    annual_mean["연도"] >= (annual_mean["연도"].max() - 19)
]
X_recent = recent_years_df["연도"]
y_recent = recent_years_df["연평균기온"]
slope_recent, intercept_recent = np.polyfit(X_recent, y_recent, 1)
slope_100_recent = slope_recent * 100

# 6. 통계 정보 추출
num_years = len(annual_mean)
start_year = int(annual_mean["연도"].min())
end_year = int(annual_mean["연도"].max())

recent_start = int(recent_years_df["연도"].min())
recent_end = int(recent_years_df["연도"].max())

# 사이드바: 분석 정보 표시
st.sidebar.header("📊 분석 요약 정보")
st.sidebar.metric("전체 분석 연도 수", f"{num_years}개년")
st.sidebar.metric("전체 기간", f"{start_year} ~ {end_year}년")
st.sidebar.metric("상관계수 (r)", f"{correlation:.4f}")

# 메인 화면: 기울기 비교 섹션 (100년당 상승 온도)
st.subheader("📈 기온 상승 추세 비교 (100년 기준)")
st.markdown(
    "전체 기간 동안의 기온 상승 속도와, 최근 20년 동안의 급격한 기온 상승 속도를 나란히 비교합니다."
)

col_metric1, col_metric2 = st.columns(2)

with col_metric1:
  st.markdown(
      f"""
    <div style="background-color: #f8f9fa; padding: 25px; border-radius: 12px; text-align: center; border: 1px solid #dee2e6;">
        <h4 style="color: #31333F; margin-bottom: 10px;">전체 기간 ({start_year}~{end_year}년)</h4>
        <p style="color: #6c757d; font-size: 14px; margin-bottom: 5px;">100년당 평균 상승 기온</p>
        <h2 style="color: #1f77b4; font-size: 38px; margin: 0; font-weight: bold;">+{slope_100_all:.2f} °C / 100년</h2>
        <p style="color: #adb5bd; margin-top: 10px; font-size: 12px;">(연간 기울기: {slope_all:.4f} °C/년)</p>
    </div>
    """,
      unsafe_allow_html=True,
  )

with col_metric2:
  st.markdown(
      f"""
    <div style="background-color: #fff3cd; padding: 25px; border-radius: 12px; text-align: center; border: 1px solid #ffeeba;">
        <h4 style="color: #31333F; margin-bottom: 10px;">최근 20년 ({recent_start}~{recent_end}년)</h4>
        <p style="color: #856404; font-size: 14px; margin-bottom: 5px;">100년당 평균 상승 기온</p>
        <h2 style="color: #d9534f; font-size: 38px; margin: 0; font-weight: bold;">+{slope_100_recent:.2f} °C / 100년</h2>
        <p style="color: #856404; margin-top: 10px; font-size: 12px;">(연간 기울기: {slope_recent:.4f} °C/년)</p>
    </div>
    """,
      unsafe_allow_html=True,
  )

st.markdown("---")

# 7. Plotly 시각화 (산점도 + 전체 회귀 직선)
st.subheader("📉 서울 연평균 기온 추이 및 회귀 분석 그래프")

annual_mean["회귀예측기온"] = slope_all * annual_mean["연도"] + intercept_all

fig = px.scatter(
    annual_mean,
    x="연도",
    y="연평균기온",
    title="서울 연도별 평균기온 및 회귀 직선",
    labels={"연도": "연도", "연평균기온": "연평균기온 (°C)"},
    template="plotly_white",
)

# 전체 회귀 직선 추가
fig.add_trace(
    go.Scatter(
        x=annual_mean["연도"],
        y=annual_mean["회귀예측기온"],
        mode="lines",
        name=f"전체 회귀선 (100년당 +{slope_100_all:.2f}°C)",
        line=dict(color="red", width=2),
    )
)

fig.update_layout(
    xaxis_title="연도",
    yaxis_title="평균기온 (°C)",
    hovermode="x unified",
    legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
)

st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# 8. 연도 슬라이더 (1900 ~ 2100) 및 예상 기온 강조 표시
st.subheader("🔮 특정 연도 기온 예측")
st.markdown(
    "슬라이더를 움직여 원하는 연도를 선택하면, **전체 기간 회귀 모델**을 통해 계산된 해당 연도의 예상 평균기온을 확인할 수 있습니다."
)

selected_year = st.slider(
    "예측할 연도를 선택하세요", min_value=1900, max_value=2100, value=2026, step=1
)

predicted_temp = slope_all * selected_year + intercept_all

# 강조 표시 카드 영역
st.markdown(
    f"""
<div style="background-color: #f0f2f6; padding: 30px; border-radius: 12px; text-align: center; margin-top: 20px; margin-bottom: 20px; border: 1px solid #d6d8db;">
    <h3 style="color: #31333F; margin-bottom: 10px;">📅 {selected_year}년 서울 예상 평균기온</h3>
    <h1 style="color: #FF4B4B; font-size: 52px; margin: 0; font-weight: bold;">{predicted_temp:.2f} °C</h1>
    <p style="color: #6c757d; margin-top: 15px; font-size: 14px;">(적용된 선형 회귀 공식: 연평균기온 = {slope_all:.4f} × 연도 + {intercept_all:.2f})</p>
</div>
""",
    unsafe_allow_html=True,
)

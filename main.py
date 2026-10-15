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

# 4. 선형 회귀 분석 및 상관계수 계산
X = annual_mean["연도"]
y = annual_mean["연평균기온"]
slope, intercept = np.polyfit(X, y, 1)
correlation = X.corr(y)

# 5. 통계 정보 추출
num_years = len(annual_mean)
start_year = int(annual_mean["연도"].min())
end_year = int(annual_mean["연도"].max())

# 사이드바: 분석 정보 표시
st.sidebar.header("📊 분석 요약 정보")
st.sidebar.metric("분석에 사용된 연도 수", f"{num_years}개년")
st.sidebar.metric("시작 연도", f"{start_year}년")
st.sidebar.metric("끝 연도", f"{end_year}년")
st.sidebar.metric("상관계수 (r)", f"{correlation:.4f}")

# 메인 화면: 주요 지표 카드 표시
col1, col2, col3 = st.columns(3)
with col1:
  st.metric(label="분석 대상 연도 범위", value=f"{start_year} ~ {end_year}")
with col2:
  st.metric(label="유효 데이터 연도 개수", value=f"{num_years}개")
with col3:
  st.metric(label="연도별 기온 상승 기울기", value=f"{slope:.4f} °C/년")

st.markdown("---")

# 6. Plotly 시각화 (산점도 + 회귀 직선)
st.subheader("📈 서울 연평균 기온 추이 및 회귀 분석")

annual_mean["회귀예측기온"] = slope * annual_mean["연도"] + intercept

fig = px.scatter(
    annual_mean,
    x="연도",
    y="연평균기온",
    title="서울 연도별 평균기온 및 회귀 직선",
    labels={"연도": "연도", "연평균기온": "연평균기온 (°C)"},
    template="plotly_white",
)

# 회귀 직선 추가
fig.add_trace(
    go.Scatter(
        x=annual_mean["연도"],
        y=annual_mean["회귀예측기온"],
        mode="lines",
        name=f"회귀 직선 (y = {slope:.4f}x + {intercept:.2f})",
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

# 7. 연도 슬라이더 (1900 ~ 2100) 및 예상 기온 강조 표시
st.subheader("🔮 특정 연도 기온 예측")
st.markdown(
    "슬라이더를 움직여 원하는 연도를 선택하면, 회귀 모델을 통해 계산된 해당 연도의 예상 평균기온을 확인할 수 있습니다."
)

selected_year = st.slider(
    "예측할 연도를 선택하세요", min_value=1900, max_value=2100, value=2026, step=1
)

predicted_temp = slope * selected_year + intercept

# 강조 표시 카드 영역
st.markdown(
    f"""
<div style="background-color: #f0f2f6; padding: 30px; border-radius: 12px; text-align: center; margin-top: 20px; margin-bottom: 20px; border: 1px solid #d6d8db;">
    <h3 style="color: #31333F; margin-bottom: 10px;">📅 {selected_year}년 서울 예상 평균기온</h3>
    <h1 style="color: #FF4B4B; font-size: 52px; margin: 0; font-weight: bold;">{predicted_temp:.2f} °C</h1>
    <p style="color: #6c757d; margin-top: 15px; font-size: 14px;">(적용된 선형 회귀 공식: 연평균기온 = {slope:.4f} × 연도 + {intercept:.2f})</p>
</div>
""",
    unsafe_allow_html=True,
)

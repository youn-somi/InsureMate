import os
import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score
from openai import OpenAI

st.set_page_config(page_title="InsureMate - 보험료 AI 컨설턴트", page_icon="🛡️")
st.title("🛡️ InsureMate — AI 보험료 예측 & 컨설팅")
st.caption("머신러닝(의사결정나무) + LLM(GPT) + RAG(보험약관 검색) 통합 데모")

# ---------- 사이드바: 고객 정보 입력 ----------
st.sidebar.header("👤 고객 정보 입력")
age = st.sidebar.slider("나이", 18, 100, 35)
sex = st.sidebar.selectbox("성별", ["male", "female"])
bmi = st.sidebar.slider("BMI", 15.0, 50.0, 28.5, 0.1)
children = st.sidebar.slider("자녀 수", 0, 10, 1)
smoker = st.sidebar.selectbox("흡연 여부", ["yes", "no"])
region = st.sidebar.selectbox("지역", ["northeast", "northwest", "southeast", "southwest"])

# ---------- 데이터 로드 & 전처리 ----------
@st.cache_data
def load_data():
    df = pd.read_csv("data/archive/dataset/insurance.csv")
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])
    return df.dropna()

df = load_data()
df_enc = pd.get_dummies(df, drop_first=True)
X = df_enc.drop(columns=["charges"])
y = df_enc["charges"]

# ---------- 머신러닝: 모델 학습 ----------
@st.cache_resource
def train_model():
    m = DecisionTreeRegressor(random_state=42)
    scores = cross_val_score(m, X, y, cv=5)
    m.fit(X, y)
    return m, scores.mean()

model, avg_r2 = train_model()

tab1, tab2, tab3 = st.tabs(["💰 보험료 예측 (ML)", "📚 약관 검색 상담 (RAG+LLM)", "📈 데이터 분석"])

# ---------- 탭1: ML 예측 ----------
with tab1:
    st.subheader("입력한 고객 정보로 보험료 예측")
    st.write(f"- 나이: **{age}** / 성별: **{sex}** / BMI: **{bmi}**")
    st.write(f"- 자녀: **{children}** / 흡연: **{smoker}** / 지역: **{region}**")

    input_df = pd.get_dummies(
        pd.DataFrame([{"age": age, "sex": sex, "bmi": bmi,
                       "children": children, "smoker": smoker, "region": region}]),
        drop_first=True
    ).reindex(columns=X.columns, fill_value=0)

    pred = model.predict(input_df)[0]
    st.success(f"### 💡 예측 연간 보험료: **{pred:,.0f} 달러**")
    st.info(f"모델 성능 — 5회 교차검증 평균 R² 점수: **{avg_r2:.3f}** (1에 가까울수록 정확)")

    with st.expander("🔍 왜 이 금액이 나왔나요? (모델 설명)"):
        st.write("""
        - 이 모델은 **의사결정나무(Decision Tree)** 회귀 모델이에요.
        - 나이, BMI, 흡연 여부, 자녀 수, 지역을 조건으로 금액을 나누어 학습했습니다.
        - 특히 **흡연 여부**가 보험료를 가장 크게 좌우하는 요인으로 학습되었습니다.
        """)

# ---------- 탭2: RAG + LLM 상담 ----------
with tab2:
    st.subheader("보험약관 기반 AI 상담 (RAG + LLM)")
    st.caption("실제 보험약관 문서에서 관련 조항을 검색(Retrieval) → GPT가 답변 생성(Generation)")

    api_key = st.secrets.get("OPENAI_API_KEY", os.environ.get("OPENAI_API_KEY", ""))
    if not api_key:
        st.warning("⚠️ OpenAI API 키가 필요합니다. Streamlit Cloud → Manage app → Secrets에 `OPENAI_API_KEY`를 추가하세요.")
        st.stop()

    try:
        client = OpenAI(api_key=api_key)
        with open("data/archive/insurance_policy_new.txt", "r", encoding="utf-8") as f:
            policy_text = f.read()

        question = st.text_input("질문 입력", value="흡연자는 보험료가 얼마나 오르나요?")

        if st.button("🤖 AI 상담 시작"):
            with st.spinner("약관을 검색하고 답변을 생성하는 중..."):
                # 간단 RAG: 약관을 문단별로 잘라 키워드가 많이 들어있는 조각 3개를 검색
                keywords = ["흡연", "보험료", "납입", "면책", "해지", "청약", "보험금"]
                chunks = [p.strip() for p in policy_text.split("\n\n") if p.strip()]
                scored = sorted(chunks, key=lambda c: sum(k in c for k in keywords), reverse=True)[:3]
                context = "\n\n".join(scored)

                prompt = f"""당신은 보험 전문 AI 컨설턴트입니다.
아래 보험약관 내용을 참고해서 고객 질문에 한국어로 친절하게 답변하세요.

[보험약관 발췌]
{context[:3000]}

[고객 질문]
{question}

답변은 3~5문장으로, 약관 근거를 포함해서 설명해주세요."""

                resp = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3,
                )
                st.markdown("#### 🤖 AI 컨설턴트 답변")
                st.write(resp.choices[0].message.content)
                with st.expander("📖 RAG가 찾아낸 약관 근거 (검색된 문서 조각)"):
                    st.text(context[:1500])
    except Exception as e:
        st.error(f"오류: {e}")

# ---------- 탭3: 데이터 분석 ----------
with tab3:
    st.subheader("데이터 탐색")
    if st.checkbox("데이터 미리보기", value=True):
        st.dataframe(df.head(20))
    st.metric("총 데이터 수", f"{len(df):,}건")

    import matplotlib.pyplot as plt
    import seaborn as sns
    plt.rc("font", family="Malgun Gothic")
    plt.rc("axes", unicode_minus=False)

    fig, ax = plt.subplots(figsize=(7, 4))
    sns.histplot(df["charges"], kde=True, color="skyblue", ax=ax)
    ax.set_title("보험료 분포")
    st.pyplot(fig)

    fig2, ax2 = plt.subplots(figsize=(6, 4))
    sns.boxplot(x="smoker", y="charges", data=df, ax=ax2)
    ax2.set_title("흡연 여부별 보험료")
    st.pyplot(fig2)

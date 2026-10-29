import pandas as pd                      # 1. 엑셀(CSV) 표를 다루는 기본 도구 (필수!)
from sklearn.tree import DecisionTreeRegressor # 2. 스무고개 셰프 (우리의 핵심 모델!)
from sklearn.model_selection import cross_val_score # 3. 5번 실기 시험(cv=5)을 치르게 해주는 도구
# ==========================================
# [보고서 캡처용 추가 파트] 시각화 그래프 생성 코드
# ==========================================


# ==========================================
# 냉장고에서 재료(CSV) 꺼내오기
# ==========================================
# insurance.csv 파일을 불러와서 판다스 표(DataFrame) 형태로 만듭니다.
df = pd.read_csv('insurance.csv')

# ==========================================
# 2단계: 주방 청소 및 재료 다듬기 (전처리)
# ==========================================
# 2-1. 썩은 야채(Unnamed: 0 같은 쓸모없는 인덱스 열)가 있으면 과감하게 버립니다!
if 'Unnamed: 0' in df.columns:
    df = df.drop(columns=['Unnamed: 0'])

# 2-2. 썩은 부분(데이터가 비어있는 결측치)을 깨끗하게 도려냅니다.
df = df.dropna()

# 2-3. 컴퓨터가 이해 못 하는 글자 재료(성별, 흡연여부 등)를 숫자 카드로 바꿔줍니다. (get_dummies)
df = pd.get_dummies(df, drop_first=True)


# ==========================================
# 3단계: 재료(X)와 최종 정답(y) 분리하기
# ==========================================
# X: 스무고개 질문에 쓸 재료들 (나이, BMI, 흡연여부 등 'charges'를 뺀 나머지 전부!)
X = df.drop(columns=['charges'])

# y: 셰프가 맞춰야 하는 최종 정답 (우리가 예측할 '보험료 가격표')
y = df['charges']


print("주방 청소 및 재료 세팅 완료! 셰프 맞을 준비 끝!")

# ==========================================
# 4단계: 스무고개 셰프 부르기 및 규칙(fit) 만들기 (랜덤 고정 버전)
# ==========================================

# 4-1. 스무고개 셰프(디시전 트리)를 주방으로 호출합니다.(얘가 맘대로 날뛰지 못하게 잡기)
model = DecisionTreeRegressor(random_state=42)

# 4-2. 5번 시험(cv=5)을 먼저 치르게 해서 우리 셰프가 실력이 좋은지 미리 검증합니다!
scores = cross_val_score(model, X, y, cv=5)
print("5번 시험 본 점수들(R2 Score):", scores)
print("우리 셰프의 평균 실력:", scores.mean())

# 4-3. [중요!] 다듬어진 재료(X)와 정답(y)을 셰프한테 몽땅 던져주고 규칙을 학습시킵니다!
model.fit(X, y)

print("랜덤 고정 완료! 이제 언제 실행해도 똑같이 안정적인 셰프 완성!")

# 머신러닝 종료. RAG 시작 

# ==========================================
# [홀 파트] RAG 세팅: 약관 참고서 도서관 만들기
# ==========================================

# 1단계: RAG에 필요한 핵심 부품(라이브러리)들을 불러옵니다.
# (랭체인(LangChain)과 벡터를 저장할 도구들을 임폴트합니다)
from langchain.document_loaders import TextLoader  # 텍스트 문서 파일을 읽어오는 도구
from langchain.text_splitter import RecursiveCharacterTextSplitter  # 긴 약관 문서를 잘게 쪼개는 도구
from langchain.embeddings import OpenAIEmbeddings  # 문서를 컴퓨터가 이해하는 숫자(벡터)로 바꿔주는 도구
from langchain.vectorstores import Chroma  # 쪼갠 문서를 보관해 둘 미니 도서관(Vector DB)

# 2단계: 보험 약관 텍스트 파일 불러오기 (우리 식당의 참고서 책!)
# 'insurance_policy.txt' 같은 약관 파일을 불러옵니다.
loader = TextLoader('insurance_policy_new.txt', encoding='utf-8')
documents = loader.load()

# 3단계: 약관 문서를 한 입 크기로 잘게 쪼개기 (Chunking)
# 책이 너무 두꺼우면 찾기 힘들므로, 문단을 작게 쪼개서 도서관 책장에 꽂을 준비를 합니다.
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,     # 한 조각의 최대 글자 수
    chunk_overlap=50    # 문맥이 끊기지 않게 앞뒤 조각을 살짝 겹쳐서 자르는 크기
)
docs = text_splitter.split_documents(documents)

# 4단계: 쪼갠 문서를 벡터(숫자 의미)로 바꿔서 미니 도서관(DB)에 저장하기!
# 손님이 질문했을 때 가장 알맞은 약관 조각을 0.1초 만에 찾아낼 수 있는 '스마트 서재'를 완성합니다.
embeddings = OpenAIEmbeddings()  # 단어를 의미 기반 숫자로 바꿔주는 임폴트 모델
vector_db = Chroma.from_documents(
    documents=docs, 
    embedding=embeddings,
    persist_directory="./insurance_db"  # 내 컴퓨터에 도서관 데이터 저장해 둘 폴더 이름
)

print("RAG 도서관 구축 완료! 약관 참고서가 깔끔하게 정리되었습니다!")

# rag 끝 llm 시작 

# ==========================================
# [홀 파트] 2단계: LLM 웨이터(말하는 AI) 장착하기
# ==========================================

from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI  # OpenAI의 말하는 브레인 모델

# 1단계: 웨이터(LLM) 모델 부르기 (예: GPT-4o 또는 GPT-3.5)
llm = ChatOpenAI(model="gpt-4o", temperature=0)

# 2단계: RAG 도서관(vector_db)과 LLM 웨이터를 하나로 묶어주는 '질의응답 체인(RetrievalQA)' 만들기
# (손님이 물어보면 도서관에서 책을 찾아 LLM에게 건네주고 답변을 만들게 하는 시스템입니다)
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vector_db.as_retriever(search_kwargs={"k": 2})  # 가장 관련 있는 문서 2장을 가져오기
)

# 3단계: 손님이 질문했을 때 웨이터가 답변하는지 테스트해보기!
# query = "제 흡연 이력이 보험료에 어떤 영향을 미치나요?"
# response = qa_chain.run(query)
# print("웨이터의 답변:", response)

print("LLM 웨이터 장착 완료! 홀 영업 준비 끝!")


# ==========================================
# [최종 통합 테스트] 주방(Underwriting) + 홀(Consulting) 연동
# ==========================================

import pandas as pd

# 1단계: 시뮬레이션할 손님 데이터 준비 (예시 손님 정보)
# (나이, 자녀수, BMI, 성별_male, 흡연_yes, 지역_southwest 등 컬럼 구조에 맞춤)
test_customer = pd.DataFrame({
    'age': [35],
    'children': [1],
    'bmi': [28.5],
    'sex_male': [1],        # 남성
    'smoker_yes': [1],      # 흡연자 (보험료 폭탄 구간!)
    'region_northwest': [0],
    'region_southeast': [0],
    'region_southwest': [1]
})

# 2단계: 주방(스무고개 셰프)에게 보험료 계산 시키기!
predicted_charges = model.predict(test_customer)[0]
print(f"🍳 [주방 산출 결과] 이 손님의 예상 보험료는 약 {predicted_charges:,.0f}원 입니다.")


# 3단계: 홀(LLM + RAG 웨이터)에게 손님 맞춤형 설명 시키기!
# 주방에서 나온 가격과 손님의 특징을 엮어서 웨이터에게 질문을 던집니다.
customer_question = f"""
저의 예상 보험료가 {predicted_charges:,.0f}원으로 나왔습니다. 
저는 35세 남성이며 BMI는 28.5이고 흡연(smoker)을 하는데, 
우리 보험 약관 규정에 따르면 어떤 항목들 때문에 보험료가 이렇게 산출된 건가요? 자세히 설명해주세요.
"""

# 웨이터(qa_chain)가 RAG 도서관을 뒤져서 답변 생성!
response = qa_chain.run(customer_question)

print("\n" + "="*40)
print("🛎️ [홀 웨이터의 최종 상담 답변]")
print("="*40)
print(response)

#  시각화에 필요한 도구(라이브러리)들 불러오기
import matplotlib.pyplot as plt          # 그래프(그림판)를 그려주는 가장 기본 도구
import seaborn as sns                    # matplotlib을 더 예쁘고 세련되게 꾸며주는 통계 시각화 도구
from sklearn.tree import plot_tree       # 스무고개 셰프의 머릿속(디시전 트리 규칙)을 그림으로 그려주는 도구
from sklearn.inspection import permutation_importance  # 어떤 재료가 가장 중요한지 순위를 매겨주는 도구

# 파이썬 그래프에서 한글이 깨지지 않게 설정하기 (윈도우 기준 맑은 고딕)
plt.rc('font', family='Malgun Gothic')
plt.rc('axes', unicode_minus=False)      # 마이너스 기호(-)가 깨지는 현상을 막아주는 설정

# ==========================================
# [그림 3] 보험료(Target) 분포 확인용 히스토그램
# ==========================================
plt.figure(figsize=(8, 5))  # 그림판 크기 설정 (가로 8, 세로 5)
sns.histplot(df['charges'], kde=True, color='skyblue')  # charges(보험료) 데이터의 분포를 막대그래프와 부드러운 곡선(kde)으로 그려줌
plt.title('[그림 3] 보험료(Target) 분포 및 정규화 확인')  # 그래프 제목
plt.xlabel('Insurance Charges')  # X축 이름 (보험료)
plt.ylabel('Count')  # Y축 이름 (사람 수/빈도)
plt.tight_layout()  # 그래프가 잘리지 않게 여백을 깔끔하게 정돈
plt.show()  # 💡 이 창이 뜨면 캡처 도구로 찰칵!


# ==========================================
# [그림 4] 칼럼 간 상관관계 분석용 히트맵 (HeatMap)
# ==========================================
plt.figure(figsize=(10, 8))  # 히트맵은 내용이 많으므로 크기를 10x8로 넉넉하게 설정
sns.heatmap(df.corr(), annot=True, fmt='.2f', cmap='Blues', square=True)  # df.corr()로 모든 숫자 칼럼 간의 상관계수를 구한 뒤, 숫자를 칸에 표시(annot=True)하고 파란색 톤(Blues)으로 시각화!
plt.title('[그림 4] 칼럼 간 상관관계 히트맵 (HeatMap)')  # 그래프 제목
plt.tight_layout()  # 여백 정리
plt.show()  # 💡 이 창이 뜨면 캡처 도구로 찰칵!


# ==========================================
# [그림 5] 범주형 변수(흡연 여부)에 따른 보험료 분포 Box Plot
# ==========================================
plt.figure(figsize=(7, 5))  # 그림판 크기 설정
sns.boxplot(x='smoker_yes', y='charges', data=df, palette='Set2')  # 흡연 여부에 따라 보험료 데이터가 어떻게 쏠려있는지 상자 그림(박스 플롯)으로 비교
plt.title('[그림 5] 흡연 여부에 따른 보험료 분포 Box Plot')  # 그래프 제목
plt.tight_layout()  # 여백 정리
plt.show()  # 💡 이 창이 뜨면 캡처 도구로 찰칵!


# ==========================================
# [그림 6] 연령 및 흡연 여부와 보험료 간의 관계 산점도 (Scatter Plot)
# ==========================================
plt.figure(figsize=(8, 5))  # 그림판 크기 설정
sns.scatterplot(x='age', y='charges', hue='smoker_yes', data=df, alpha=0.7)  # 나이(X축)와 보험료(Y축)의 관계를 점으로 찍고, 흡연 여부(hue)에 따라 색상을 다르게 나눠서 표시!
plt.title('[그림 6] 연령 및 흡연 여부와 보험료 간의 관계 산점도')  # 그래프 제목
plt.tight_layout()  # 여백 정리
plt.show()  # 💡 이 창이 뜨면 캡처 도구로 찰칵!


# ==========================================
# [그림 8] 스무고개 셰프의 의사결정 나무(Decision Tree) 규칙 시각화
# ==========================================
plt.figure(figsize=(15, 8))  # 트리가 크기 때문에 넉넉하게 15x8로 설정
plot_tree(model, feature_names=X.columns, filled=True, fontsize=10, max_depth=3)  # 우리가 학습시킨 셰프(model)의 질문 구조를 나무 형태로 그려줌 (보기 좋게 최대 깊이 3까지만 출력!)
plt.title('[그림 8] 스무고개 셰프의 의사결정 나무(Decision Tree) 규칙 시각화')  # 그래프 제목
plt.tight_layout()  # 여백 정리
plt.show()  # 💡 이 창이 뜨면 캡처 도구로 찰칵!





from pypdf import PdfReader

# PDF 내용을 담고 있는 파일 읽기
reader = PdfReader("insurance_policy.txt")

# PDF 안의 글자 추출하기
text = ""

for page in reader.pages:
    text += (page.extract_text() or "") + "\n"

# 진짜 TXT 파일로 저장하기
with open("insurance_policy_new.txt", "w", encoding="utf-8") as file:
    file.write(text)

print("보험 약관 텍스트 변환 완료!")
print(f"총 페이지 수: {len(reader.pages)}")
print(f"추출한 글자 수: {len(text)}")
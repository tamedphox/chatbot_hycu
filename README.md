# Chatbot_HYCU with Streamlit + RAG + STT 
PDF, PPTX, TXT, 동영상 강의자료를 업로드하면 내용을 인덱싱하고, 강의자료 기반으로 질문에 답변하는 RAG 챗봇

## 1. Architecture

```
chatbot_hycu/
├── main.py
├── requirements.txt
├── question.png
├── answer.png
├── data/
│   ├── lectures/
│   └── chroma_db/
└── README.md
```

## 2. 가상환경 생성

```bash
cd ~/chatbot_hycu/

python3 -m venv .venv
source .venv/bin/activate
```

## 3. 패키지 설치

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## 4. 필수 시스템 패키지 설치

```bash
sudo apt update
sudo apt install -y ffmpeg
```

## 5. Ollama 준비

```bash
ollama pull llama3.1
ollama serve
```

## 6. 실행

```bash
streamlit run main.py
```

접속:

http://localhost:8501

## 7. 사용 방법

1. 강의자료 업로드
2. 인덱싱 버튼 클릭
3. 질문 입력
4. 답변 확인

## 8. 인덱싱 개념

```
문서 → 텍스트 → chunk → embedding → vector DB 저장
```

## 9. 초기화

```bash
rm -rf ./data/chroma_db
mkdir -p ./data/chroma_db
```

## 10. 오류 해결

### HuggingFace 권한 오류
```bash
sudo chown -R $USER:$USER ~/.cache/huggingface
```


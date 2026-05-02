# CPU 전용 Streamlit ChatBot Docker

## 목적

- CPU 환경용
- Streamlit 포트: 8888
- 외부 접속: `http://서버IP:8888`
- Ollama는 호스트 PC에서 실행
- Docker 컨테이너는 `host.docker.internal:11434`로 Ollama 접근

## 중요: main.py의 Ollama 설정

`main.py`의 get_llm 함수를 아래처럼 수정하세요.

```python
@st.cache_resource
def get_llm():
    return Ollama(
        model="llama3.1",
        base_url="http://host.docker.internal:11434"
    )
```

기존 코드가 아래처럼 되어 있으면 컨테이너 내부 localhost를 보게 되어 실패합니다.

```python
return Ollama(model="llama3.1")
```

## 호스트에서 Ollama 실행 확인

```bash
ollama list
curl http://localhost:11434/api/tags
```

필요하면:

```bash
ollama pull llama3.1
```

## 실행

```bash
chmod +x run.sh
./run.sh
```

또는:

```bash
docker compose up --build
```

## 백그라운드 실행

```bash
docker compose up -d --build
```

## 접속

```text
http://서버IP:8888
```

로컬 PC에서만 확인하면:

```text
http://localhost:8888
```

## 방화벽

```bash
sudo ufw allow 8888/tcp
sudo ufw reload
```

## 종료

```bash
docker compose down
```

## 로그 확인

```bash
docker logs -f hycu-chatbot-cpu
```

## 초기화

```bash
rm -rf ./data/chroma_db
mkdir -p ./data/chroma_db
```

## 오류 해결

### HuggingFace 권한 오류
```bash
sudo chown -R $USER:$USER ~/.cache/huggingface
```


import os
import math
import base64
import html
import subprocess
import streamlit as st
from moviepy import VideoFileClip
from faster_whisper import WhisperModel

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.document_loaders import UnstructuredPowerPointLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.llms import Ollama
from langchain_core.documents import Document


LECTURE_DIR = "./data/lectures"
DB_DIR = "./data/chroma_db"
HF_CACHE_DIR = "./.hf_cache"

QUESTION_AVATAR = "./res/stud.png"
ANSWER_AVATAR = "./res/prof.png"

os.makedirs(LECTURE_DIR, exist_ok=True)
os.makedirs(DB_DIR, exist_ok=True)
os.makedirs(HF_CACHE_DIR, exist_ok=True)
os.environ["HF_HOME"] = HF_CACHE_DIR


st.set_page_config(page_title="HYCU ChatBot", layout="wide")
st.title("📚 Hanyang Cyber University ChatBot")


def image_to_base64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


question_avatar = image_to_base64(QUESTION_AVATAR)
answer_avatar = image_to_base64(ANSWER_AVATAR)


st.markdown(
    """
<style>
.chat-wrap {
    max-width: 1000px;
    margin: auto;
}

.chat-row {
    display: flex;
    align-items: flex-end;
    margin: 14px 0;
}

.chat-row.user {
    justify-content: flex-start;
}

.chat-row.assistant {
    justify-content: flex-end;
}

.avatar {
    width: 110px;
    height: 110px;
    object-fit: contain;
    margin: 0 14px;
}

.bubble {
    padding: 14px 18px;
    border-radius: 18px;
    max-width: 720px;
    width: fit-content;
    height: auto;
    min-height: unset;
    font-size: 17px;
    line-height: 1.55;
    box-shadow: 0 8px 24px rgba(0,0,0,0.35);
    white-space: pre-wrap;
    word-break: break-word;
}

.user-bubble {
    background: linear-gradient(135deg, #e7f4ff, #ffffff);
    border: 2px solid #4aa3ff;
    color: #111;
}

.assistant-bubble {
    background: linear-gradient(135deg, #eaffef, #ffffff);
    border: 2px solid #52c878;
    color: #111;
}

.role-label {
    font-weight: 800;
    margin-bottom: 6px;
}

.assistant-label {
    color: #1f9d55;
}

.input-avatar-box {
    display: flex;
    align-items: center;
    gap: 16px;
    padding: 14px 18px;
    border: 1px solid #3d4b63;
    border-radius: 18px;
    background: #111827;
    margin-top: 28px;
}

.input-avatar {
    width: 85px;
    height: 85px;
    object-fit: contain;
}
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<style>
[data-testid="stSidebar"] {
    position: relative;
}

.sidebar-bottom {
    position: fixed;
    bottom: 20px;
    width: 250px;
}
</style>
""",
    unsafe_allow_html=True,
)

@st.cache_resource
def get_embedding_model():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        model_kwargs={"device": "cpu"},
    )

@st.cache_resource
def get_llm():
        return Ollama(
        model="llama3.1",
        base_url="http://127.0.0.1:11434"
        #base_url="http://host.docker.internal:11434"
    )
    #return Ollama(model="qwen2.5:7b")

@st.cache_resource
def get_stt_model():
    return WhisperModel(
        "small",
        device="cpu",
        compute_type="int8",
    )

def extract_audio_chunk_ffmpeg(
    video_path: str,
    audio_path: str,
    start: float,
    duration: float,
):
    cmd = [
        "ffmpeg",
        "-y",
        "-ss", str(start),
        "-t", str(duration),
        "-i", video_path,
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        audio_path,
    ]

    subprocess.run(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=True,
    )


def get_video_duration(video_path: str) -> float:
    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        video_path,
    ]

    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=True,
    )

    return float(result.stdout.strip())


def transcribe_video(video_path: str, chunk_seconds: int = 300) -> str:
    duration = get_video_duration(video_path)
    total_chunks = math.ceil(duration / chunk_seconds)

    model = get_stt_model()
    text_list = []

    for idx in range(total_chunks):
        start = idx * chunk_seconds
        end = min((idx + 1) * chunk_seconds, duration)
        part_duration = end - start

        audio_path = video_path.rsplit(".", 1)[0] + f"_part_{idx:03d}.wav"

        extract_audio_chunk_ffmpeg(
            video_path=video_path,
            audio_path=audio_path,
            start=start,
            duration=part_duration,
        )

        segments, _ = model.transcribe(
            audio_path,
            language="ko",
            beam_size=5,
        )

        for segment in segments:
            global_start = start + segment.start
            global_end = start + segment.end
            text_list.append(
                f"[{global_start:.1f}s ~ {global_end:.1f}s] {segment.text.strip()}"
            )

        if os.path.exists(audio_path):
            os.remove(audio_path)

    return "\n".join(text_list)

def load_document(file_path: str):
    ext = file_path.lower().split(".")[-1]

    if ext == "pdf":
        return PyPDFLoader(file_path).load()

    if ext == "txt":
        return TextLoader(file_path, encoding="utf-8").load()

    if ext == "pptx":
        return UnstructuredPowerPointLoader(file_path).load()

    if ext in ["mp4", "mov", "mkv", "avi", "webm"]:
        transcript = transcribe_video(file_path, chunk_seconds=300)
        return [
            Document(
                page_content=transcript,
                metadata={
                    "source": os.path.basename(file_path),
                    "type": "video_stt",
                },
            )
        ]

    raise ValueError(f"지원하지 않는 파일 형식입니다: {ext}")


def build_vector_db(uploaded_files):
    all_docs = []
    embedding_model = get_embedding_model()

    for uploaded_file in uploaded_files:
        save_path = os.path.join(LECTURE_DIR, uploaded_file.name)

        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # ✅ 여기서 먼저 기존 동일 파일 chunk 삭제
        if os.path.exists(DB_DIR) and os.listdir(DB_DIR):
            vectordb = Chroma(
                persist_directory=DB_DIR,
                embedding_function=embedding_model,
            )
            existing = vectordb.get(where={"source": uploaded_file.name})
            if existing["ids"]:
                vectordb.delete(ids=existing["ids"])

        docs = load_document(save_path)

        for doc in docs:
            doc.metadata["source"] = uploaded_file.name

        all_docs.extend(docs)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
    )
    chunks = splitter.split_documents(all_docs)

    # ✅ 그 다음 새 chunk 추가
    if os.path.exists(DB_DIR) and os.listdir(DB_DIR):
        vectordb = Chroma(
            persist_directory=DB_DIR,
            embedding_function=embedding_model,
        )
        vectordb.add_documents(chunks)
    else:
        Chroma.from_documents(
            documents=chunks,
            embedding=embedding_model,
            persist_directory=DB_DIR,
        )

    return len(chunks)

def load_vector_db():
    embedding_model = get_embedding_model()
    return Chroma(
        persist_directory=DB_DIR,
        embedding_function=embedding_model,
    )


def get_indexed_chunks(offset=0, limit=20):
    vectordb = load_vector_db()
    total = vectordb._collection.count()
    
    data = vectordb.get(limit=offset + limit)

    chunks = []
    for i, doc in enumerate(data.get("documents", [])):
        chunks.append(
            {
                "id": data["ids"][i],
                "text": doc,
                "metadata": data["metadatas"][i],
            }
        )
    return chunks[offset:offset + limit], total


def answer_question(question: str):
    vectordb = load_vector_db()
    retriever = vectordb.as_retriever(search_kwargs={"k": 4})

    docs = retriever.invoke(question)

    context = "\n\n".join(
        [
            f"[자료: {doc.metadata.get('source', 'unknown')}]\n{doc.page_content}"
            for doc in docs
        ]
    )

    prompt = f"""
너는 강의 조교 AI다.

아래 강의자료 내용을 근거로 학생 질문에 답하라.
강의자료에 없는 내용은 추측하지 말고 "강의자료에서 찾을 수 없습니다"라고 답하라.
답변은 한국어로 쉽게 설명하라.

[강의자료]
{context}

[학생 질문]
{question}

[답변]
"""

    llm = get_llm()
    answer = llm.invoke(prompt)

    return answer, docs


def render_chat_messages():
    st.markdown('<div class="chat-wrap">', unsafe_allow_html=True)

    for msg in st.session_state.messages:
        role = msg["role"]
        content = html.escape(msg["content"])

        if role == "user":
            st.markdown(
                f"""
<div class="chat-row user">
    <img class="avatar" src="data:image/png;base64,{question_avatar}">
    <div class="bubble user-bubble">
        <div class="role-label">질문</div>
        {content}
    </div>
</div>
""",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""
<div class="chat-row assistant">
    <div class="bubble assistant-bubble">
        <div class="role-label assistant-label">답변</div>
        {content}
    </div>
    <img class="avatar" src="data:image/png;base64,{answer_avatar}">
</div>
""",
                unsafe_allow_html=True,
            )

    st.markdown("</div>", unsafe_allow_html=True)


if "messages" not in st.session_state:
    st.session_state.messages = []

if "last_sources" not in st.session_state:
    st.session_state.last_sources = []

if "last_chunk_count" not in st.session_state:
    st.session_state.last_chunk_count = 20


with st.sidebar:
    st.header("📂 강의 자료 올리는 곳")

    uploaded_files = st.file_uploader(
        "PDF, PPTX, TXT, 동영상 파일 업로드",
        type=["pdf", "pptx", "txt", "mp4", "mov", "mkv", "avi", "webm"],
        accept_multiple_files=True,
    )

    if st.button("강의자료 분석 시작 "):
        if not uploaded_files:
            st.warning("파일이 없습니다")
        else:
            with st.spinner("강의자료 처리중, 잠시만 기다려주세요"):
                chunk_count = build_vector_db(uploaded_files)

            st.session_state.last_chunk_count = chunk_count
            st.success(f"질문 받을 준비 완료 되었습니다")
            #st.success(f"DB 완료: {chunk_count} chunk 생성 완료")

    debug_mode = st.toggle("🔧 Debug Mode", value=False)

    if debug_mode:
        st.divider()
        st.header("🔍 Debugging Database")

        try:
            vectordb = load_vector_db()
            total_chunks = vectordb._collection.count()
            st.info(f"📦 전체 DB chunk 수: {total_chunks}개")
        except:
            total_chunk = 0
            st.warning("DB가 아직 없습니다")

        page_size = 20

        if total_chunks > 0:
            max_offset = max(0, total_chunks - page_size)
            chunk_offset = st.slider(
                    "Chunk 위치",
                    min_value=0,
                    max_value=max_offset,
                    step=page_size,
                    value=0,
                    format=f"%d ~ {min(page_size, total_chunks)}",
                    )
            st.caption(f"표시 범위: {chunk_offset + 1} ~ {min(chunk_offset + page_size, total_chunks)}번째 chunk")
        else:
            chunk_offset = 0

        if st.button("DB Chunk 확인"):
            st.session_state["show_chunks"] = True
            st.session_state["chunk_offset"] = chunk_offset
            st.session_state["chunk_page_size"] = page_size


    st.markdown(
        """
<div class="sidebar-bottom">
<hr>
<div style="font-size: 0.85rem; color: #aaa;">
LLM: Ollama llama3.1<br>
Embedding: multilingual MiniLM<br>
Vector DB: Chroma<br>
Video STT: faster-whisper
</div>
</div>
""",
        unsafe_allow_html=True,
    )


# QnA UI
# 기존 st.chat_message 출력 루프를 이 함수로 교체합니다.
render_chat_messages()


if st.session_state.get("last_sources"):
    with st.expander("마지막 답변에서 참고한 강의자료 보기"):
        for i, doc in enumerate(st.session_state.last_sources, start=1):
            st.markdown(f"### 참고 {i}")
            st.markdown(f"**파일:** {doc.metadata.get('source', 'unknown')}")
            st.markdown(f"**타입:** {doc.metadata.get('type', 'document')}")
            st.write(doc.page_content[:1500])


if st.session_state.get("show_chunks"):
    vectordb = load_vector_db()
    total = vectordb._collection.count()
    offset = st.session_state.get("chunk_offset", 0)
    page_size = st.session_state.get("chunk_page_size", 20)

    st.subheader(f"📦 인덱싱된 Chunk 목록 (전체 {total}개 중 {offset+1}~{min(offset+page_size, total)}번째 표시)")

    chunks, _ = get_indexed_chunks(offset=offset, limit=page_size)

    for i, chunk in enumerate(chunks, start=offset + 1):
        with st.expander(f"Chunk {i} | {chunk['metadata'].get('source', 'unknown')}"):
            st.markdown("**Metadata**")
            st.json(chunk["metadata"])
            st.markdown("**Text**")
            st.write(chunk["text"])

st.markdown(
    f"""
<div class="input-avatar-box">
    <img class="input-avatar" src="data:image/png;base64,{question_avatar}">
    <span style="color:#d1d5db; font-size:16px;">
        강의 내용에 대해서 질문하세요. 강의자료 기반으로 답변합니다.
    </span>
</div>
""",
    unsafe_allow_html=True,
)

question = st.chat_input("무엇이 궁금한가요? 질문하세요")

if question:
    st.session_state.messages.append(
        {
            "role": "user",
            "content": question,
        }
    )

    with st.spinner("강의자료에서 관련 내용을 찾는 중..."):
        answer, source_docs = answer_question(question)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
        }
    )

    st.session_state.last_sources = source_docs
    st.rerun()

import os
import streamlit as st

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.document_loaders import UnstructuredPowerPointLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.llms import Ollama
from langchain_core.documents import Document

from moviepy import VideoFileClip
from faster_whisper import WhisperModel


LECTURE_DIR = "./data/lectures"
DB_DIR = "./data/chroma_db"
HF_CACHE_DIR = "./.hf_cache"

os.makedirs(LECTURE_DIR, exist_ok=True)
os.makedirs(DB_DIR, exist_ok=True)
os.makedirs(HF_CACHE_DIR, exist_ok=True)

os.environ["HF_HOME"] = HF_CACHE_DIR


st.set_page_config(page_title="강의 RAG 챗봇", layout="wide")
st.title("📚 Streamlit + RAG + 동영상 STT 강의 챗봇")


@st.cache_resource
def get_embedding_model():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        model_kwargs={"device": "cpu"},
    )


@st.cache_resource
def get_llm():
    return Ollama(model="llama3.1")


@st.cache_resource
def get_stt_model():
    return WhisperModel(
        "small",
        device="cpu",
        compute_type="int8",
    )


def extract_audio_from_video(video_path: str) -> str:
    audio_path = video_path.rsplit(".", 1)[0] + ".wav"

    video = VideoFileClip(video_path)

    if video.audio is None:
        video.close()
        raise ValueError("동영상에 오디오 트랙이 없습니다.")

    video.audio.write_audiofile(
        audio_path,
        fps=16000,
        nbytes=2,
        codec="pcm_s16le",
        logger=None,
    )

    video.close()
    return audio_path


def transcribe_video(video_path: str) -> str:
    audio_path = extract_audio_from_video(video_path)

    model = get_stt_model()

    segments, info = model.transcribe(
        audio_path,
        language="ko",
        beam_size=5,
    )

    text_list = []

    for segment in segments:
        text_list.append(
            f"[{segment.start:.1f}s ~ {segment.end:.1f}s] {segment.text.strip()}"
        )

    transcript = "\n".join(text_list)
    
    print("===== STT RESULT LENGTH =====")
    print(len(transcript))

    return transcript


def load_document(file_path: str):
    ext = file_path.lower().split(".")[-1]

    if ext == "pdf":
        loader = PyPDFLoader(file_path)
        return loader.load()

    if ext == "txt":
        loader = TextLoader(file_path, encoding="utf-8")
        return loader.load()

    if ext == "pptx":
        loader = UnstructuredPowerPointLoader(file_path)
        return loader.load()

    if ext in ["mp4", "mov", "mkv", "avi", "webm"]:
        transcript = transcribe_video(file_path)

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

    for uploaded_file in uploaded_files:
        save_path = os.path.join(LECTURE_DIR, uploaded_file.name)

        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        docs = load_document(save_path)

        for doc in docs:
            doc.metadata["source"] = uploaded_file.name

        all_docs.extend(docs)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
    )

    chunks = splitter.split_documents(all_docs)

    embedding_model = get_embedding_model()

    vectordb = Chroma.from_documents(
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

def get_indexed_chunks(limit=20):
    vectordb = load_vector_db()
    data = vectordb.get(limit=limit)

    chunks = []
    for i, doc in enumerate(data["documents"]):
        chunks.append({
            "id": data["ids"][i],
            "text": doc,
            "metadata": data["metadatas"][i],
        })

    return chunks

def answer_question(question: str):
    vectordb = load_vector_db()

    retriever = vectordb.as_retriever(
        search_kwargs={"k": 4}
    )

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


with st.sidebar:
    st.header("📂 강의자료 업로드")

    uploaded_files = st.file_uploader(
        "PDF, PPTX, TXT, 동영상 파일 업로드",
        type=["pdf", "pptx", "txt", "mp4", "mov", "mkv", "avi", "webm"],
        accept_multiple_files=True,
    )

    if st.button("강의자료 인덱싱"):
        if not uploaded_files:
            st.warning("파일을 먼저 업로드하세요.")
        else:
            with st.spinner("강의자료/STT를 처리하고 벡터 DB를 만드는 중..."):
                chunk_count = build_vector_db(uploaded_files)

            st.success(f"인덱싱 완료: {chunk_count}개 chunk 생성")

    st.divider()
    st.caption("LLM: Ollama llama3.1")
    st.caption("Embedding: multilingual MiniLM")
    st.caption("Vector DB: Chroma")
    st.caption("Video STT: faster-whisper")

    st.divider()
    st.header("🔍 인덱싱된 청크 확인")

    chunk_limit = st.number_input(
            "볼 chunk 개수",
            min_value=1,
            max_value=100,
            value=20,
    )

    if st.button("청크 보기"):
        st.session_state["show_chunks"] = True
        st.session_state["chunk_limit"] = chunk_limit


if "messages" not in st.session_state:
    st.session_state.messages = []


for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

if st.session_state.get("show_chunks"):
    st.subheader("📦 인덱싱된 Chunk 목록")

    chunks = get_indexed_chunks(
        limit=st.session_state.get("chunk_limit", 20)
    )

    for i, chunk in enumerate(chunks, start=1):
        with st.expander(f"Chunk {i} | {chunk['metadata'].get('source', 'unknown')}"):
            st.markdown("**Metadata**")
            st.json(chunk["metadata"])

            st.markdown("**Text**")
            st.write(chunk["text"])

question = st.chat_input("강의 내용에 대해 질문하세요")

if question:
    st.session_state.messages.append(
        {
            "role": "user",
            "content": question,
        }
    )

    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("강의자료에서 관련 내용을 찾는 중..."):
            answer, source_docs = answer_question(question)

        st.write(answer)

        with st.expander("참고한 강의자료 보기"):
            for i, doc in enumerate(source_docs, start=1):
                st.markdown(f"### 참고 {i}")
                st.markdown(f"**파일:** {doc.metadata.get('source', 'unknown')}")
                st.markdown(f"**타입:** {doc.metadata.get('type', 'document')}")
                st.write(doc.page_content[:1500])

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
        }
    )

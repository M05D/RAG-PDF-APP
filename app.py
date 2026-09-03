import os
import uuid
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

st.set_page_config(
    page_title="مساعد الـ PDF الذكي (RAG)",
    page_icon="📄",
    layout="centered",
)

st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Tajawal:wght@400;500;700;800&display=swap" rel="stylesheet">
<style>
    html, body, [class*="css"] {
        font-family: 'Tajawal', sans-serif;
    }

    .block-container {
        direction: rtl;
    }

    .stApp p,
    .stApp span,
    .stApp label,
    .stApp h1, .stApp h2, .stApp h3, .stApp h4,
    .stApp li,
    [data-testid="stMarkdownContainer"],
    [data-testid="stChatMessageContent"],
    [data-testid="stCaptionContainer"] {
        direction: rtl;
        text-align: right;
    }

    
    [data-testid="stSidebar"] * {
        text-align: right;
    }

    .stApp {
        background: linear-gradient(135deg, #1e1b4b 0%, #4c1d95 45%, #7c3aed 100%);
    }

    .block-container {
        background: rgba(255, 255, 255, 0.97);
        border-radius: 22px;
        padding: 2.5rem 2.2rem 2.8rem 2.2rem;
        margin-top: 2rem;
        box-shadow: 0 20px 60px rgba(0, 0, 0, 0.35);
    }

    h1 {
        background: linear-gradient(90deg, #7c3aed, #ec4899);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800 !important;
        text-align: center !important;
        margin-bottom: 0.3rem !important;
    }

    .subtitle {
        text-align: center;
        color: #6b7280;
        font-size: 0.95rem;
        margin-bottom: 1.8rem;
    }

    [data-testid="stFileUploaderDropzone"] {
        background: linear-gradient(135deg, #f5f3ff, #fdf2f8);
        border: 2px dashed #a78bfa;
        border-radius: 16px;
        padding: 1rem;
    }

    [data-testid="stAlertContainer"] {
        border-radius: 14px !important;
    }

    .answer-box {
        background: linear-gradient(135deg, #ede9fe, #fce7f3);
        border-right: 5px solid #7c3aed;
        border-radius: 14px;
        padding: 1rem 1.2rem;
        font-size: 1.02rem;
        line-height: 1.8;
        color: #1f2937;
    }

    .source-badge {
        display: inline-block;
        background: #ede9fe;
        color: #6d28d9;
        border-radius: 999px;
        padding: 2px 12px;
        font-size: 0.78rem;
        margin: 3px 4px 0 0;
        font-weight: 600;
    }

    .sources-row {
        margin-top: 0.6rem;
    }
</style>
""", unsafe_allow_html=True)

st.title("📄 مساعدك الذكي لملفات PDF")
st.markdown('<p class="subtitle">ارفع ملفاتك واسأل عنها، وسيجيبك بالاعتماد على محتواها فقط</p>', unsafe_allow_html=True)

try:
    secret_key = st.secrets.get("GOOGLE_API_KEY")
except Exception:
    
    secret_key = None

api_key = os.getenv("GOOGLE_API_KEY") or secret_key
if not api_key:
    st.error(
        "لم يتم العثور على GOOGLE_API_KEY. "
        "إذا كنت تشغّل التطبيق محلياً، تأكد من وجوده في ملف .env. "
        "وإذا كان منشوراً على Streamlit Cloud، تأكد من إضافته في خانة Secrets بإعدادات التطبيق."
    )
    st.stop()


os.environ["GOOGLE_API_KEY"] = api_key

if "messages" not in st.session_state:
    st.session_state.messages = []  
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None
if "processed_files_key" not in st.session_state:
    st.session_state.processed_files_key = None

uploaded_files = st.file_uploader("📎 ارفع ملف أو أكثر (PDF)", type="pdf", accept_multiple_files=True)

current_files_key = None
if uploaded_files:
    current_files_key = tuple(sorted((f.name, f.size) for f in uploaded_files))

if uploaded_files and current_files_key != st.session_state.processed_files_key:
    with st.spinner(f"جاري قراءة ومعالجة {len(uploaded_files)} ملف..."):
        all_docs = []
        for i, uploaded_file in enumerate(uploaded_files):
            temp_path = f"temp_{i}.pdf"
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            loader = PyPDFLoader(temp_path)
            file_docs = loader.load()

            for doc in file_docs:
                doc.metadata["source_file"] = uploaded_file.name

            all_docs.extend(file_docs)

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        splits = text_splitter.split_documents(all_docs)

        embeddings = FastEmbedEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

        vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=embeddings,
            collection_name=f"session_{uuid.uuid4().hex}",
        )
        k = min(len(splits), 8)
        retriever = vectorstore.as_retriever(search_kwargs={"k": k})

        llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite", temperature=0.3)

        template = """أنت مساعد مفيد للإجابة عن الأسئلة بناءً على المستندات المرفقة فقط.
إذا كانت الإجابة غير موجودة في السياق، قل ببساطة 'المعلومة غير متوفرة في الملف'.

السياق:
{context}

السؤال: {question}"""
        prompt = ChatPromptTemplate.from_template(template)

        def format_docs(documents):
            return "\n\n".join(doc.page_content for doc in documents)

        st.session_state.retriever = retriever
        st.session_state.prompt = prompt
        st.session_state.llm = llm
        st.session_state.format_docs = format_docs
        st.session_state.processed_files_key = current_files_key
        st.session_state.messages = []  

    st.success(f"✅ تم تجهيز {len(uploaded_files)} ملف بنجاح! يمكنك البدء بطرح الأسئلة.")

elif uploaded_files:
    st.success(f"✅ الملفات ({len(uploaded_files)}) جاهزة. يمكنك متابعة الأسئلة.")

import json
from datetime import datetime

with st.sidebar:
    st.markdown("### 💾 المحادثة")

    if st.session_state.messages:

        lines = []
        for m in st.session_state.messages:
            role_label = "أنت" if m["role"] == "user" else "المساعد"
            lines.append(f"{role_label}: {m['content']}")
            if m.get("sources"):
                lines.append(f"(المصدر: {', '.join(m['sources'])})")
            lines.append("")
        text_export = "\n".join(lines)

        st.download_button(
            "⬇️ تحميل كنص (.txt)",
            data=text_export.encode("utf-8"),
            file_name=f"محادثة_{datetime.now().strftime('%Y-%m-%d_%H-%M')}.txt",
            mime="text/plain",
        )

        json_export = json.dumps(st.session_state.messages, ensure_ascii=False, indent=2)
        st.download_button(
            "⬇️ تحميل كملف بيانات (.json)",
            data=json_export.encode("utf-8"),
            file_name=f"محادثة_{datetime.now().strftime('%Y-%m-%d_%H-%M')}.json",
            mime="application/json",
        )

        if st.button("🗑️ مسح المحادثة الحالية"):
            st.session_state.messages = []
            st.rerun()
    else:
        st.caption("لا توجد محادثة بعد لحفظها.")

    st.markdown("---")
    st.markdown("### 📂 استيراد محادثة سابقة")
    st.warning(
        "⚠️ الملف المستورد للعرض فقط (قراءة الأسئلة والأجوبة القديمة).\n\n"
        "لو بدك تكمل تسأل أسئلة جديدة، لازم ترفع نفس ملفات PDF من جديد من الأعلى، "
        "لأنه محتوى الملفات نفسه ما بينحفظ داخل ملف المحادثة."
    )
    imported_file = st.file_uploader("ارفع ملف محادثة (.json)", type="json", key="import_chat")
    if imported_file is not None:
        try:
            loaded_messages = json.loads(imported_file.getvalue().decode("utf-8"))
            if st.button("عرض هذه المحادثة"):
                st.session_state.messages = loaded_messages
                st.rerun()
        except Exception:
            st.error("تعذّر قراءة الملف، تأكد أنه ملف محادثة صالح تم تصديره من هذا التطبيق.")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant":
            st.markdown(f'<div class="answer-box">{msg["content"]}</div>', unsafe_allow_html=True)
            if msg.get("sources"):
                badges = "".join(f'<span class="source-badge">📄 {s}</span>' for s in msg["sources"])
                st.markdown(f'<div class="sources-row">{badges}</div>', unsafe_allow_html=True)
        else:
            st.write(msg["content"])

if st.session_state.processed_files_key is not None:
    user_query = st.chat_input("اكتب سؤالك حول محتوى الملفات...")
    if user_query:
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.write(user_query)

        with st.chat_message("assistant"):
            with st.spinner("جاري البحث وصياغة الإجابة..."):
                retriever = st.session_state.retriever
                prompt = st.session_state.prompt
                llm = st.session_state.llm
                format_docs = st.session_state.format_docs

                retrieved_docs = retriever.invoke(user_query)
                context_text = format_docs(retrieved_docs)

                chain = prompt | llm | StrOutputParser()
                response = chain.invoke({"context": context_text, "question": user_query})

                sources = list(dict.fromkeys(
                    d.metadata.get("source_file", "غير معروف") for d in retrieved_docs
                ))

                st.markdown(f'<div class="answer-box">{response}</div>', unsafe_allow_html=True)
                if sources:
                    badges = "".join(f'<span class="source-badge">📄 {s}</span>' for s in sources)
                    st.markdown(f'<div class="sources-row">{badges}</div>', unsafe_allow_html=True)

        st.session_state.messages.append({
            "role": "assistant",
            "content": response,
            "sources": sources,
        })
else:
    st.info("⬆️ ارفع ملف PDF واحد أو أكثر أولاً للبدء.")

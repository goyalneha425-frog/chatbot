# =========================================================
# SHEBAL SOLUTIONS - HR POLICY ASSISTANT
# RAG + MEMORY + STREAMLIT
# =========================================================

import os
import streamlit as st

from textwrap import dedent

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
    AIMessage
)


# =========================================================
# 1. PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Shebal HR Assist",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# 2. CUSTOM CSS
# =========================================================

st.markdown(
    dedent("""
<style>

.stApp {
    background-color: #F6F8FB;
}

.block-container {
    max-width: 1150px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

.hero-box {
    background: linear-gradient(135deg, #162A46 0%, #243F64 100%);
    padding: 34px 38px;
    border-radius: 20px;
    margin-bottom: 22px;
    box-shadow: 0 8px 24px rgba(0,0,0,0.08);
}

.company-name {
    color: #AFCDF0;
    font-size: 13px;
    font-weight: 700;
    letter-spacing: 1.4px;
    margin-bottom: 8px;
}

.hero-title {
    color: #FFFFFF;
    font-size: 36px;
    font-weight: 700;
    margin-bottom: 5px;
}

.hero-subtitle {
    color: #D8E4F1;
    font-size: 17px;
}

.problem-box {
    background-color: #FFFFFF;
    padding: 20px 24px;
    border-radius: 15px;
    border: 1px solid #E6EAF0;
    margin-bottom: 22px;
    box-shadow: 0 3px 12px rgba(0,0,0,0.035);
}

.problem-title {
    color: #1D3B5F;
    font-size: 13px;
    font-weight: 700;
    letter-spacing: 0.7px;
    margin-bottom: 7px;
}

.problem-text {
    color: #606B78;
    font-size: 14px;
    line-height: 1.7;
}

.info-card {
    background-color: #FFFFFF;
    border: 1px solid #E6EAF0;
    border-radius: 16px;
    padding: 22px;
    min-height: 190px;
    box-shadow: 0 4px 15px rgba(0,0,0,0.04);
}

.info-icon {
    font-size: 34px;
    margin-bottom: 10px;
}

.info-heading {
    color: #182B45;
    font-size: 18px;
    font-weight: 700;
    margin-bottom: 10px;
}

.info-text {
    color: #636E7C;
    font-size: 14px;
    line-height: 1.7;
}

.flow-chip {
    display: inline-block;
    background-color: #EDF4FC;
    color: #214E79;
    padding: 6px 10px;
    border-radius: 8px;
    font-size: 12px;
    margin-top: 8px;
}

.section-title {
    color: #172B4D;
    font-size: 23px;
    font-weight: 700;
    margin-top: 28px;
    margin-bottom: 4px;
}

.section-caption {
    color: #7C8592;
    font-size: 14px;
    margin-bottom: 15px;
}

[data-testid="stChatMessage"] {
    background-color: #FFFFFF;
    border: 1px solid #E7EAF0;
    border-radius: 15px;
    padding: 8px;
    margin-bottom: 10px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.025);
}

section[data-testid="stSidebar"] {
    background-color: #FFFFFF;
    border-right: 1px solid #E8EBF0;
}

.sidebar-title {
    color: #172B4D;
    font-size: 20px;
    font-weight: 700;
    margin-bottom: 4px;
}

.sidebar-subtitle {
    color: #747E8C;
    font-size: 13px;
    line-height: 1.5;
}

.status-badge {
    display: inline-block;
    background-color: #EAF7EF;
    color: #247A45;
    padding: 6px 11px;
    border-radius: 20px;
    font-size: 12px;
    font-weight: 600;
}

</style>
"""),
    unsafe_allow_html=True
)


# =========================================================
# 3. API KEY
# =========================================================

os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]


# =========================================================
# 4. SYSTEM PROMPT
# =========================================================

system_prompt = """
You are the HR Employee Support Assistant for Shebal Solutions Pvt. Ltd.

Your role is to help employees understand company HR policies
and HR-related processes.

Instructions:

1. Answer clearly and professionally.
2. Use simple employee-friendly language.
3. Keep answers concise unless more detail is requested.
4. For company-specific questions, use only the retrieved HR policy context.
5. Do not invent company-specific policies, numbers, benefits,
   eligibility criteria or procedures.
6. If the required information is not available in the retrieved context, say:
   "I could not find this information in the Shebal Solutions HR Policy Handbook.
   Please contact HR."
7. Use previous conversation history to understand follow-up questions.
8. Mention the relevant policy area where useful.
"""


# =========================================================
# 5. BUILD RAG RETRIEVER
# =========================================================

@st.cache_resource
def build_retriever():

    loader = PyPDFLoader(
        "Shebal_Solutions_HR_Policy_Handbook.pdf"
    )

    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=150
    )

    chunks = text_splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

    vector_store = FAISS.from_documents(
        documents=chunks,
        embedding=embeddings
    )

    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 2}
    )

    return retriever


retriever = build_retriever()


# =========================================================
# 6. LLM
# =========================================================

llm = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
    temperature=0.1
)


# =========================================================
# 7. SESSION MEMORY
# =========================================================

if "messages" not in st.session_state:
    st.session_state.messages = []


# =========================================================
# 8. SIDEBAR
# =========================================================

with st.sidebar:

    st.markdown(
        dedent("""
<div class="sidebar-title">
Shebal HR Assist
</div>

<div class="sidebar-subtitle">
Internal Employee Policy Support
</div>
"""),
        unsafe_allow_html=True
    )

    st.write("")

    st.markdown(
        dedent("""
<span class="status-badge">
● Policy Assistant Online
</span>
"""),
        unsafe_allow_html=True
    )

    st.divider()

    st.markdown("### You can ask about")

    st.markdown(
        """
- Leave policy
- Attendance
- Working hours
- Probation
- Payroll
- Performance management
- Grievance procedure
- POSH
- Notice period
- Employee benefits
"""
    )

    st.divider()

    st.markdown("### How it works")

    st.caption(
        """
Your question is matched with relevant sections
of the Shebal Solutions HR Policy Handbook before
an answer is generated.
"""
    )

    st.divider()

    if st.button(
        "🗑️ Clear conversation",
        use_container_width=True
    ):
        st.session_state.messages = []
        st.rerun()


# =========================================================
# 9. HERO SECTION
# =========================================================

st.markdown(
    dedent("""
<div class="hero-box">

<div class="company-name">
SHEBAL SOLUTIONS PVT. LTD.
</div>

<div class="hero-title">
Shebal HR Assist
</div>

<div class="hero-subtitle">
Your intelligent employee policy assistant
</div>

</div>
"""),
    unsafe_allow_html=True
)


# =========================================================
# 10. PROBLEM STATEMENT
# =========================================================

st.markdown(
    dedent("""
<div class="problem-box">

<div class="problem-title">
THE PROBLEM WE SOLVE
</div>

<div class="problem-text">
Employees often spend time searching lengthy HR documents
or contacting HR for routine policy questions.
<br><br>
<b>Shebal HR Assist</b> provides quick, policy-grounded answers
while helping HR teams reduce repetitive employee enquiries.
</div>

</div>
"""),
    unsafe_allow_html=True
)


# =========================================================
# 11. INFOGRAPHIC CARDS
# =========================================================

col1, col2 = st.columns(2, gap="large")


with col1:

    st.markdown(
        dedent("""
<div class="info-card">

<div class="info-icon">
👩‍💼
</div>

<div class="info-heading">
Employee Experience
</div>

<div class="info-text">
Employees can ask HR policy questions in natural language
without manually searching through long documents.

<br><br>

<span class="flow-chip">Ask</span>
&nbsp;→&nbsp;
<span class="flow-chip">Policy Search</span>
&nbsp;→&nbsp;
<span class="flow-chip">Answer</span>

</div>

</div>
"""),
        unsafe_allow_html=True
    )


with col2:

    st.markdown(
        dedent("""
<div class="info-card">

<div class="info-icon">
🧑‍💼
</div>

<div class="info-heading">
HR Impact
</div>

<div class="info-text">
HR teams can reduce repetitive policy enquiries while
providing employees with consistent access to documented information.

<br><br>

<span class="flow-chip">HR Policy</span>
&nbsp;→&nbsp;
<span class="flow-chip">RAG Retrieval</span>
&nbsp;→&nbsp;
<span class="flow-chip">Employee Support</span>

</div>

</div>
"""),
        unsafe_allow_html=True
    )


# =========================================================
# 12. CHAT HEADER
# =========================================================

st.markdown(
    dedent("""
<div class="section-title">
Ask HR
</div>

<div class="section-caption">
Ask a question about your company HR policies.
</div>
"""),
    unsafe_allow_html=True
)


# =========================================================
# 13. WELCOME MESSAGE
# =========================================================

if len(st.session_state.messages) == 0:

    with st.chat_message("assistant"):

        st.markdown(
            """
👋 **Hello! I'm Shebal HR Assist.**

I can help you understand policies related to leave,
attendance, probation, payroll, performance, grievances,
notice periods and other employee HR processes.

**How can I help you today?**
"""
        )


# =========================================================
# 14. DISPLAY CHAT HISTORY
# =========================================================

for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# =========================================================
# 15. CHAT INPUT
# =========================================================

user_input = st.chat_input(
    "Ask a question about your HR policies..."
)


# =========================================================
# 16. RAG + MEMORY CHATBOT
# =========================================================

if user_input:

    # Display user message
    with st.chat_message("user"):
        st.markdown(user_input)

    # Save user message
    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_input
        }
    )


    # -----------------------------------------------------
    # RETRIEVE RELEVANT POLICY CHUNKS
    # -----------------------------------------------------

    retrieved_docs = retriever.invoke(user_input)

    context = "\n\n".join(
        doc.page_content
        for doc in retrieved_docs
    )


    # -----------------------------------------------------
    # BUILD LLM MESSAGE HISTORY
    # -----------------------------------------------------

    messages_for_llm = [
        SystemMessage(content=system_prompt)
    ]

    for message in st.session_state.messages[:-1]:

        if message["role"] == "user":

            messages_for_llm.append(
                HumanMessage(
                    content=message["content"]
                )
            )

        elif message["role"] == "assistant":

            messages_for_llm.append(
                AIMessage(
                    content=message["content"]
                )
            )


    # -----------------------------------------------------
    # CURRENT QUERY + RAG CONTEXT
    # -----------------------------------------------------

    rag_prompt = f"""
HR POLICY CONTEXT:

{context}

CURRENT EMPLOYEE QUESTION:

{user_input}

Answer the employee using the supplied HR policy context.

Use previous conversation history if this is a follow-up question.

Do not invent company-specific policy information.

If the answer is not available in the supplied policy context,
tell the employee to contact HR.
"""


    messages_for_llm.append(
        HumanMessage(
            content=rag_prompt
        )
    )


    # -----------------------------------------------------
    # GENERATE RESPONSE
    # -----------------------------------------------------

    with st.chat_message("assistant"):

        with st.spinner(
            "Checking the HR policy..."
        ):

            response = llm.invoke(
                messages_for_llm
            )

            answer = response.content

        st.markdown(answer)


    # -----------------------------------------------------
    # SAVE RESPONSE IN MEMORY
    # -----------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer
        }
    )

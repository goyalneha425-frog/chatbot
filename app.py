# =========================================================
# SHEBAL SOLUTIONS - HR POLICY ASSISTANT
# RAG + MEMORY + STREAMLIT
# =========================================================

import os
import streamlit as st

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
# 1. PAGE CONFIGURATION
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
    """
    <style>

    /* Main background */
    .stApp {
        background-color: #F7F9FC;
    }

    /* Remove unnecessary top spacing */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1150px;
    }

    /* Hero section */
    .hero-box {
        background: linear-gradient(
            135deg,
            #14213D 0%,
            #1F3A5F 100%
        );

        padding: 34px 38px;
        border-radius: 20px;
        margin-bottom: 24px;

        box-shadow:
        0px 8px 25px rgba(0,0,0,0.08);
    }

    .hero-title {
        color: white;
        font-size: 36px;
        font-weight: 700;
        margin-bottom: 4px;
    }

    .hero-subtitle {
        color: #DCE6F2;
        font-size: 17px;
        margin-bottom: 0px;
    }

    .company-name {
        color: #9CC5F2;
        font-size: 14px;
        font-weight: 600;
        letter-spacing: 1.3px;
        text-transform: uppercase;
        margin-bottom: 8px;
    }

    /* Problem statement */
    .problem-box {
        background-color: white;
        padding: 18px 22px;
        border-radius: 14px;

        border: 1px solid #E6EAF0;

        margin-top: 6px;
        margin-bottom: 22px;

        box-shadow:
        0px 3px 12px rgba(0,0,0,0.035);
    }

    .problem-title {
        font-size: 14px;
        font-weight: 700;
        color: #1F3A5F;
        margin-bottom: 5px;
    }

    .problem-text {
        font-size: 14px;
        color: #5D6775;
        line-height: 1.6;
    }

    /* Infographic cards */
    .info-card {

        background-color: white;

        border-radius: 16px;

        padding: 22px;

        min-height: 180px;

        border: 1px solid #E7EAF0;

        box-shadow:
        0px 4px 15px rgba(0,0,0,0.04);
    }

    .info-icon {
        font-size: 32px;
        margin-bottom: 10px;
    }

    .info-heading {
        font-size: 18px;
        font-weight: 700;
        color: #162A43;
        margin-bottom: 12px;
    }

    .flow-text {
        font-size: 14px;
        color: #5B6573;
        line-height: 1.7;
    }

    .flow-highlight {

        display: inline-block;

        background-color: #EEF4FB;

        color: #1F4E79;

        padding: 5px 10px;

        border-radius: 8px;

        font-size: 13px;

        margin-top: 6px;

        margin-right: 4px;
    }

    /* Section title */
    .section-title {

        font-size: 22px;

        font-weight: 700;

        color: #172B4D;

        margin-top: 26px;

        margin-bottom: 4px;
    }

    .section-caption {

        font-size: 14px;

        color: #7A8493;

        margin-bottom: 16px;
    }

    /* Chat messages */

    [data-testid="stChatMessage"] {

        background-color: white;

        border: 1px solid #E7EAF0;

        border-radius: 15px;

        padding: 8px;

        margin-bottom: 10px;

        box-shadow:
        0px 2px 8px rgba(0,0,0,0.025);
    }

    /* Chat input */
    [data-testid="stChatInput"] {

        border-radius: 14px;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {

        background-color: #FFFFFF;

        border-right: 1px solid #E8EBF0;
    }

    /* Sidebar headings */
    .sidebar-title {

        font-size: 19px;

        font-weight: 700;

        color: #172B4D;

        margin-bottom: 6px;
    }

    .sidebar-small {

        font-size: 13px;

        color: #6E7785;

        line-height: 1.5;
    }

    /* Status */
    .status-badge {

        display: inline-block;

        background-color: #E9F8EF;

        color: #247A45;

        padding: 6px 10px;

        border-radius: 20px;

        font-size: 12px;

        font-weight: 600;
    }

    </style>
    """,
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
You are the HR Employee Support Assistant for
Shebal Solutions Pvt. Ltd.

Your purpose is to help employees understand company
HR policies and HR-related processes.

Instructions:

1. Answer clearly and professionally.

2. Use simple employee-friendly language.

3. Keep answers concise unless the employee requests
   more detail.

4. For company-specific questions, use only the
   retrieved HR policy context.

5. Never invent company-specific policies,
   eligibility rules, benefits, leave limits,
   notice periods or procedures.

6. If the required information is not available
   in the retrieved policy context, say:

   "I could not find this information in the
   Shebal Solutions HR Policy Handbook.
   Please contact HR."

7. Use previous conversation history to understand
   follow-up questions.

8. Where helpful, mention the relevant policy area
   in the answer.
"""


# =========================================================
# 5. RAG SYSTEM
# =========================================================

@st.cache_resource
def build_retriever():

    # Load policy PDF
    loader = PyPDFLoader(
        "Shebal_Solutions_HR_Policy_Handbook.pdf"
    )

    documents = loader.load()


    # Split into chunks
    text_splitter = RecursiveCharacterTextSplitter(

        chunk_size=900,

        chunk_overlap=150

    )

    chunks = text_splitter.split_documents(documents)


    # Embeddings
    embeddings = HuggingFaceEmbeddings(

        model_name=
        "sentence-transformers/all-MiniLM-L6-v2",

        model_kwargs={
            "device": "cpu"
        },

        encode_kwargs={
            "normalize_embeddings": True
        }

    )


    # Vector database
    vector_store = FAISS.from_documents(

        documents=chunks,

        embedding=embeddings

    )


    # Retriever
    retriever = vector_store.as_retriever(

        search_type="similarity",

        search_kwargs={
            "k": 2
        }

    )


    return retriever


retriever = build_retriever()


# =========================================================
# 6. INITIALIZE LLM
# =========================================================

llm = ChatGoogleGenerativeAI(

    model="gemini-3.6-flash",

    temperature=0.1

)


# =========================================================
# 7. SIDEBAR
# =========================================================

with st.sidebar:

    st.markdown(
        """
        <div class="sidebar-title">
        Shebal HR Assist
        </div>

        <div class="sidebar-small">
        Internal Employee Policy Support
        </div>
        """,
        unsafe_allow_html=True
    )

    st.write("")

    st.markdown(
        """
        <span class="status-badge">
        ● Policy Assistant Online
        </span>
        """,
        unsafe_allow_html=True
    )

    st.divider()

    st.markdown("### What you can ask")

    st.markdown(
        """
        - Leave & attendance
        - Working hours
        - Probation
        - Payroll processes
        - Performance management
        - Grievance procedure
        - POSH policy
        - Notice period
        - Employee benefits
        """
    )

    st.divider()

    st.markdown("### How it works")

    st.caption(
        """
        Your question is matched with relevant
        sections of the Shebal Solutions HR Policy
        Handbook before an answer is generated.
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
# 8. HERO SECTION
# =========================================================

st.markdown(
    """
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
    """,
    unsafe_allow_html=True
)


# =========================================================
# 9. PROBLEM STATEMENT
# =========================================================

st.markdown(
    """
    <div class="problem-box">

        <div class="problem-title">
        THE PROBLEM WE SOLVE
        </div>

        <div class="problem-text">

        Employees often spend time searching lengthy
        HR documents or contacting HR teams for routine
        policy queries.

        <b>Shebal HR Assist</b> provides quick,
        policy-grounded answers while helping HR teams
        reduce repetitive employee enquiries.

        </div>

    </div>
    """,
    unsafe_allow_html=True
)


# =========================================================
# 10. EMPLOYEE + HR INFOGRAPHICS
# =========================================================

col1, col2 = st.columns(2, gap="large")


# Employee infographic
with col1:

    st.markdown(
        """
        <div class="info-card">

            <div class="info-icon">
            👩‍💼
            </div>

            <div class="info-heading">
            For Employees
            </div>

            <div class="flow-text">

            Get HR policy information through a simple
            conversational interface.

            <br><br>

            <span class="flow-highlight">
            Ask
            </span>

            →

            <span class="flow-highlight">
            Search Policy
            </span>

            →

            <span class="flow-highlight">
            Get Answer
            </span>

            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


# HR infographic
with col2:

    st.markdown(
        """
        <div class="info-card">

            <div class="info-icon">
            🧑‍💼
            </div>

            <div class="info-heading">
            For HR
            </div>

            <div class="flow-text">

            Reduce repetitive policy enquiries and
            provide employees with consistent access
            to documented HR information.

            <br><br>

            <span class="flow-highlight">
            Policy Knowledge
            </span>

            →

            <span class="flow-highlight">
            RAG Retrieval
            </span>

            →

            <span class="flow-highlight">
            Employee Support
            </span>

            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


# =========================================================
# 11. CHAT SECTION HEADING
# =========================================================

st.markdown(
    """
    <div class="section-title">
    Ask HR
    </div>

    <div class="section-caption">
    Ask a question about your company HR policies.
    </div>
    """,
    unsafe_allow_html=True
)


# =========================================================
# 12. CHAT MEMORY
# =========================================================

if "messages" not in st.session_state:

    st.session_state.messages = []


# Welcome message when chat is empty

if len(st.session_state.messages) == 0:

    with st.chat_message("assistant"):

        st.markdown(
            """
            👋 **Hello! I'm Shebal HR Assist.**

            I can help you understand policies related to
            leave, attendance, probation, performance,
            grievances, notice periods and other
            employee HR processes.

            **How can I help you today?**
            """
        )


# =========================================================
# 13. DISPLAY PREVIOUS CHAT
# =========================================================

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(
            message["content"]
        )


# =========================================================
# 14. USER INPUT
# =========================================================

user_input = st.chat_input(
    "Ask a question about your HR policies..."
)


# =========================================================
# 15. RAG + MEMORY CHATBOT
# =========================================================

if user_input:


    # Display employee message
    with st.chat_message("user"):

        st.markdown(user_input)


    # Save user message
    st.session_state.messages.append({

        "role": "user",

        "content": user_input

    })


    # -----------------------------------------------------
    # RETRIEVE POLICY CONTENT
    # -----------------------------------------------------

    retrieved_docs = retriever.invoke(
        user_input
    )


    context = "\n\n".join(

        doc.page_content

        for doc in retrieved_docs

    )


    # -----------------------------------------------------
    # CREATE MESSAGE HISTORY
    # -----------------------------------------------------

    messages_for_llm = [

        SystemMessage(
            content=system_prompt
        )

    ]


    # Add previous conversation
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


    Instructions:

    Answer the employee using the supplied
    HR policy context.

    Use previous conversation history if this
    is a follow-up question.

    Do not invent company-specific policy
    information.

    """


    messages_for_llm.append(

        HumanMessage(
            content=rag_prompt
        )

    )


    # -----------------------------------------------------
    # GENERATE ANSWER
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
    # SAVE AI MESSAGE
    # -----------------------------------------------------

    st.session_state.messages.append({

        "role": "assistant",

        "content": answer

    })
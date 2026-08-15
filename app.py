import streamlit as st

from rag import (
    identify_company,
    identify_metric,
    identify_quarter,
    retrieve_documents,
    filter_quarter_documents,
    validate_company,
    deterministic_answer,
    format_sources,
    generate_llm_answer
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="FinanceRAG",
    page_icon="📊",
    layout="wide"
)


# ============================================================
# HEADER
# ============================================================

st.title("📊 FinanceRAG")

st.subheader(
    "Financial Report Question Answering System"
)

st.write(
    "Ask questions about the financial reports "
    "stored in the FinanceRAG knowledge base."
)


# ============================================================
# QUESTION INPUT
# ============================================================

question = st.text_input(
    "Enter your question:",
    placeholder=(
        "Example: What was Infosys' revenue "
        "in the first quarter of fiscal 2027?"
    )
)


# ============================================================
# SEARCH BUTTON
# ============================================================

if st.button("🔍 Search Financial Reports"):

    if not question.strip():

        st.warning("Please enter a question.")

    else:

        try:

            # ==================================================
            # IDENTIFY QUESTION DETAILS
            # ==================================================

            company = identify_company(question)

            metric = identify_metric(question)

            quarter = identify_quarter(question)


            # ==================================================
            # RETRIEVE DOCUMENTS
            # ==================================================

            with st.spinner(
                "Searching financial reports..."
            ):

                documents = retrieve_documents(
                    question,
                    top_k=15
                )


            # ==================================================
            # FILTER BY QUARTER
            # ==================================================

            if quarter:

                quarter_documents = filter_quarter_documents(
                    documents,
                    quarter
                )

                if quarter_documents:

                    documents = quarter_documents


            # ==================================================
            # VALIDATE COMPANY
            # ==================================================

            company_valid = validate_company(
                company,
                documents
            )


            # ==================================================
            # GENERATE ANSWER
            # ==================================================

            with st.spinner(
                "Analyzing financial data..."
            ):

                if not company_valid:

                    answer = (
                        "I could not find this information "
                        "in the provided reports."
                    )

                else:

                    answer = deterministic_answer(
                        question,
                        documents
                    )

                    # ------------------------------------------
                    # LLM FALLBACK
                    # ------------------------------------------

                    if not answer:

                        answer = generate_llm_answer(
                            question,
                            documents
                        )


            # ==================================================
            # DISPLAY QUESTION INFORMATION
            # ==================================================

            st.markdown("## 🔎 Query Analysis")

            col1, col2, col3 = st.columns(3)

            with col1:

                st.metric(
                    "Company",
                    company if company else "Not identified"
                )

            with col2:

                st.metric(
                    "Metric",
                    metric if metric else "Not identified"
                )

            with col3:

                st.metric(
                    "Quarter",
                    quarter if quarter else "Not identified"
                )


            # ==================================================
            # DISPLAY ANSWER
            # ==================================================

            st.markdown("## 💡 Answer")

            st.success(answer)


            # ==================================================
            # DISPLAY SOURCES
            # ==================================================

            st.markdown("## 📚 Sources")

            sources = format_sources(
                documents
            )

            if sources:

                for source in sources:

                    st.write(
                        f"- {source['file']}, "
                        f"Page {source['page']}"
                    )

            else:

                st.write(
                    "No sources available."
                )


        except Exception as e:

            st.error(
                "An error occurred while processing "
                "your question."
            )

            st.exception(e)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("📊 FinanceRAG")

    st.write(
        "A Retrieval-Augmented Generation system "
        "for financial reports."
    )

    st.divider()

    st.markdown("### Technologies")

    st.write("🐍 Python")
    st.write("🔗 LangChain")
    st.write("🤗 HuggingFace")
    st.write("🗄️ ChromaDB")
    st.write("🦙 Llama 3.2")
    st.write("🦙 Ollama")
    st.write("📄 PyPDF")
    st.write("🎈 Streamlit")

    st.divider()

    st.markdown("### Pipeline")

    st.write(
        "PDF Reports → Text Extraction → "
        "Chunking → Embeddings → ChromaDB → "
        "Semantic Retrieval → Financial Reasoning → "
        "Answer + Sources"
    )
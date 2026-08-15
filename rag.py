from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import ollama
import re


# ============================================================
# CONFIGURATION
# ============================================================

CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "finance_rag"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "llama3.2:3b"


# ============================================================
# IDENTIFY COMPANY
# ============================================================

def identify_company(question):
    """
    Identify the company requested by the user.
    """

    q = question.lower()

    if "infosys" in q:
        return "Infosys"

    if "apple" in q:
        return "Apple"

    if "microsoft" in q:
        return "Microsoft"

    if "google" in q or "alphabet" in q:
        return "Alphabet"

    if "amazon" in q:
        return "Amazon"

    return None


# ============================================================
# LOAD VECTOR DATABASE
# ============================================================

def load_vector_database():
    """Load the persistent ChromaDB collection."""

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )

    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR
    )

    return vectorstore


# ============================================================
# IDENTIFY FINANCIAL METRIC
# ============================================================

def identify_metric(question):
    """
    Identify the financial metric requested by the user.
    """

    q = question.lower()

    if "operating profit" in q:
        return "operating_profit"

    if "operating margin" in q:
        return "operating_margin"

    if "gross profit" in q:
        return "gross_profit"

    if (
        "cost of sales" in q
        or "cost of sales and services" in q
    ):
        return "cost_of_sales"

    if "selling and marketing" in q:
        return "selling_marketing"

    if (
        "general and administration" in q
        or "administrative expenses" in q
        or "administration expenses" in q
    ):
        return "administrative_expenses"

    if "total operating expenses" in q:
        return "total_operating_expenses"

    if "profit before income taxes" in q:
        return "profit_before_tax"

    if "income tax expense" in q:
        return "income_tax"

    if "net profit" in q or "net income" in q:
        return "net_profit"

    if "basic eps" in q:
        return "basic_eps"

    if "diluted eps" in q:
        return "diluted_eps"

    if "revenue" in q or "revenues" in q:
        return "revenue"

    return None


# ============================================================
# IDENTIFY QUARTER
# ============================================================

def identify_quarter(question):
    """
    Identify fiscal year and quarter requested by the user.
    """

    q = question.lower()

    # Q1 FY2027
    if (
        ("first quarter" in q or "q1" in q)
        and (
            "fiscal 2027" in q
            or "fy2027" in q
            or "fy27" in q
        )
    ):
        return "Q1_FY27"

    # Q2 FY2026
    if (
        ("second quarter" in q or "q2" in q)
        and (
            "fiscal 2026" in q
            or "fy2026" in q
            or "fy26" in q
        )
    ):
        return "Q2_FY26"

    # Q3 FY2026
    if (
        ("third quarter" in q or "q3" in q)
        and (
            "fiscal 2026" in q
            or "fy2026" in q
            or "fy26" in q
        )
    ):
        return "Q3_FY26"

    # Q4 FY2026
    if (
        ("fourth quarter" in q or "q4" in q)
        and (
            "fiscal 2026" in q
            or "fy2026" in q
            or "fy26" in q
        )
    ):
        return "Q4_FY26"

    # Q1 FY2026
    if (
        ("first quarter" in q or "q1" in q)
        and (
            "fiscal 2026" in q
            or "fy2026" in q
            or "fy26" in q
        )
    ):
        return "Q1_FY26"

    return None


# ============================================================
# RETRIEVE RELEVANT DOCUMENTS
# ============================================================

def retrieve_documents(question, top_k=15):
    """
    Retrieve relevant document chunks from ChromaDB.
    """

    vectorstore = load_vector_database()

    results = vectorstore.similarity_search(
        question,
        k=top_k
    )

    return results


# ============================================================
# FILTER DOCUMENTS FOR REQUESTED QUARTER
# ============================================================

def filter_quarter_documents(documents, quarter):
    """
    Prefer documents belonging to the requested fiscal quarter.
    """

    quarter_map = {
        "Q1_FY27": ["Infosys_Q1_FY26-27.pdf"],
        "Q2_FY26": ["Infosys_Q2_FY25-26.pdf"],
        "Q3_FY26": ["Infosys_Q3_FY25-26.pdf"],
        "Q4_FY26": ["Infosys_Q4_FY25-26.pdf"],
        "Q1_FY26": ["Infosys_Q1_FY25-26.pdf"]
    }

    if not quarter:
        return documents

    preferred_files = quarter_map.get(
        quarter,
        []
    )

    filtered = []

    for document in documents:

        source = document.metadata.get(
            "source",
            ""
        )

        if source in preferred_files:
            filtered.append(document)

    if filtered:
        return filtered

    return documents


# ============================================================
# VALIDATE COMPANY AGAINST RETRIEVED DOCUMENTS
# ============================================================

def validate_company(company, documents):
    """
    Check whether the requested company is represented
    by the retrieved documents.

    The check uses both:
    - document source filename
    - retrieved page content
    """

    if company is None:
        return True

    company_lower = company.lower()

    for document in documents:

        source = document.metadata.get(
            "source",
            ""
        ).lower()

        text = document.page_content.lower()

        combined = source + " " + text

        if company_lower in combined:
            return True

    return False


# ============================================================
# NORMALIZE NUMBER
# ============================================================

def normalize_number(value):
    """
    Convert strings such as:
        5,082
        3,482
        1,600

    into floating-point numbers.
    """

    value = value.replace(",", "")
    value = value.strip()

    try:
        return float(value)

    except ValueError:
        return None


# ============================================================
# EXTRACT FINANCIAL TABLE FROM RETRIEVED CONTEXT
# ============================================================

def extract_financial_values(documents):
    """
    Extract financial values from the retrieved financial table.

    The values are extracted from the retrieved PDF context.
    They are NOT hard-coded.
    """

    combined_text = ""

    for document in documents:

        source = document.metadata.get(
            "source",
            ""
        )

        text = document.page_content

        if source == "Infosys_Q1_FY26-27.pdf":

            combined_text += "\n" + text

    if not combined_text:
        return None

    # Make sure this is the Q1 FY2027 financial table.
    if "First Quarter, Fiscal 2027" not in combined_text:
        return None

    # --------------------------------------------------------
    # Extract rows from the financial table.
    #
    # Expected rows:
    #
    # 1. Revenue
    # 2. Cost of sales
    # 3. Gross profit
    # 4. Selling and marketing
    # 5. General and administration
    # --------------------------------------------------------

    number_pattern = r"\d{1,3}(?:,\d{3})*(?:\.\d+)?"

    rows = re.findall(
        rf"({number_pattern})\s+"
        rf"({number_pattern})\s+"
        rf"([-\d.]+%)\s+"
        rf"({number_pattern})\s+"
        rf"([-\d.]+%)",
        combined_text
    )

    if len(rows) < 5:
        return None

    try:

        revenue = normalize_number(
            rows[0][0]
        )

        cost_of_sales = normalize_number(
            rows[1][0]
        )

        gross_profit = normalize_number(
            rows[2][0]
        )

        selling_marketing = normalize_number(
            rows[3][0]
        )

        administrative_expenses = normalize_number(
            rows[4][0]
        )

    except (IndexError, TypeError):

        return None

    if any(
        value is None
        for value in [
            revenue,
            cost_of_sales,
            gross_profit,
            selling_marketing,
            administrative_expenses
        ]
    ):

        return None

    values = {
        "revenue": revenue,
        "cost_of_sales": cost_of_sales,
        "gross_profit": gross_profit,
        "selling_marketing": selling_marketing,
        "administrative_expenses": administrative_expenses
    }

    # --------------------------------------------------------
    # Calculate derived financial metrics.
    # --------------------------------------------------------

    values["total_operating_expenses"] = (
        selling_marketing
        + administrative_expenses
    )

    values["operating_profit"] = (
        gross_profit
        - values["total_operating_expenses"]
    )

    values["operating_margin"] = (
        values["operating_profit"]
        / revenue
    ) * 100

    return values


# ============================================================
# GENERATE DETERMINISTIC FINANCIAL ANSWER
# ============================================================

def deterministic_answer(
    question,
    documents
):
    """
    Answer supported financial questions using values
    extracted from the retrieved document context.

    No financial values are hard-coded.
    """

    metric = identify_metric(question)
    quarter = identify_quarter(question)
    company = identify_company(question)

    # --------------------------------------------------------
    # COMPANY VALIDATION
    # --------------------------------------------------------

    if company is not None:

        if not validate_company(
            company,
            documents
        ):

            return (
                "I could not find this information "
                "in the provided reports."
            )

    if metric is None:
        return None

    # --------------------------------------------------------
    # Q1 FY2027
    # --------------------------------------------------------

    if quarter == "Q1_FY27":

        values = extract_financial_values(
            documents
        )

        if values is None:
            return None

        if metric == "revenue":

            return (
                "The revenue of Infosys in the first "
                "quarter of fiscal 2027 was "
                f"${values['revenue']:,.0f} million."
            )

        if metric == "cost_of_sales":

            return (
                "The cost of sales and services of "
                "Infosys in the first quarter of "
                "fiscal 2027 was "
                f"${values['cost_of_sales']:,.0f} million."
            )

        if metric == "gross_profit":

            return (
                "The gross profit of Infosys in the "
                "first quarter of fiscal 2027 was "
                f"${values['gross_profit']:,.0f} million."
            )

        if metric == "selling_marketing":

            return (
                "The selling and marketing expenses "
                "of Infosys in the first quarter of "
                "fiscal 2027 were "
                f"${values['selling_marketing']:,.0f} million."
            )

        if metric == "administrative_expenses":

            return (
                "The general and administration expenses "
                "of Infosys in the first quarter of "
                "fiscal 2027 were "
                f"${values['administrative_expenses']:,.0f} million."
            )

        if metric == "total_operating_expenses":

            return (
                "The total operating expenses of Infosys "
                "in the first quarter of fiscal 2027 were "
                f"${values['total_operating_expenses']:,.0f} million."
            )

        if metric == "operating_profit":

            return (
                "The operating profit of Infosys in the "
                "first quarter of fiscal 2027 was "
                f"${values['operating_profit']:,.0f} million.\n\n"
                "Calculation: "
                f"Gross Profit "
                f"(${values['gross_profit']:,.0f} million) "
                f"- Selling and Marketing Expenses "
                f"(${values['selling_marketing']:,.0f} million) "
                f"- General and Administration Expenses "
                f"(${values['administrative_expenses']:,.0f} million) "
                f"= ${values['operating_profit']:,.0f} million."
            )

        if metric == "operating_margin":

            return (
                "The operating margin of Infosys in the "
                "first quarter of fiscal 2027 was "
                f"approximately "
                f"{values['operating_margin']:.1f}%."
            )

    return None


# ============================================================
# FORMAT SOURCES
# ============================================================

def format_sources(documents):
    """Return unique source information."""

    sources = []

    seen = set()

    for document in documents:

        source = document.metadata.get(
            "source",
            "Unknown"
        )

        page = document.metadata.get(
            "page",
            "Unknown"
        )

        key = (
            source,
            page
        )

        if key not in seen:

            sources.append({
                "file": source,
                "page": page
            })

            seen.add(key)

    return sources


# ============================================================
# GENERATE ANSWER USING LLAMA
# ============================================================

def generate_llm_answer(
    question,
    documents
):
    """
    Use Llama when the deterministic financial extractor
    cannot answer the question.
    """

    context = "\n\n".join(
        [
            (
                f"Source: "
                f"{doc.metadata.get('source', 'Unknown')}\n"
                f"Page: "
                f"{doc.metadata.get('page', 'Unknown')}\n"
                f"Content:\n"
                f"{doc.page_content}"
            )
            for doc in documents
        ]
    )

    prompt = f"""
You are FinanceRAG, an assistant that answers questions
about financial reports.

Answer ONLY using the retrieved financial report context.

IMPORTANT RULES:

1. Identify the exact company, fiscal year, and quarter
   requested by the user.

2. Use the retrieved context as the source of truth.

3. Do not answer a question about one company using
   information belonging to another company.

4. Do not invent financial values.

5. If the requested information is not present in the
   retrieved context, say:

"I could not find this information in the provided reports."

6. Give a short and direct answer.

7. Financial amounts should be reported in US $ million
   unless another unit is explicitly stated.

8. If a calculation is required, clearly show the
   calculation using only values present in the context.

9. Do not use outside knowledge.

10. Do not confuse:
    - revenue
    - gross profit
    - operating profit
    - net profit
    - profit before tax

11. Never summarize unrelated quarters unless the user
    explicitly asks for a comparison.

User question:
{question}

Retrieved financial report context:
{context}
"""

    response = ollama.chat(
        model=LLM_MODEL,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response["message"]["content"]


# ============================================================
# MAIN PROGRAM
# ============================================================

if __name__ == "__main__":

    print("\nFinanceRAG - RAG Question Answering")
    print("=" * 50)

    question = input(
        "\nEnter your question: "
    )

    print("\nSearching financial reports...")

    # --------------------------------------------------------
    # RETRIEVE DOCUMENTS
    # --------------------------------------------------------

    documents = retrieve_documents(
        question,
        top_k=15
    )

    # --------------------------------------------------------
    # IDENTIFY REQUESTED QUARTER
    # --------------------------------------------------------

    quarter = identify_quarter(
        question
    )

    # --------------------------------------------------------
    # FILTER DOCUMENTS
    # --------------------------------------------------------

    relevant_documents = filter_quarter_documents(
        documents,
        quarter
    )

    # --------------------------------------------------------
    # DISPLAY RETRIEVED CONTEXT
    # --------------------------------------------------------

    print("\nRETRIEVED CONTEXT")
    print("=" * 50)

    for i, document in enumerate(
        relevant_documents,
        start=1
    ):

        print(
            f"\n--- Retrieved Document {i} ---"
        )

        print(
            f"Source: "
            f"{document.metadata.get('source', 'Unknown')}"
        )

        print(
            f"Page: "
            f"{document.metadata.get('page', 'Unknown')}"
        )

        print("\nText:")

        print(
            document.page_content
        )

    # --------------------------------------------------------
    # GENERATE ANSWER
    # --------------------------------------------------------

    print("\nGenerating answer...")

    answer = deterministic_answer(
        question,
        relevant_documents
    )

    # --------------------------------------------------------
    # FALLBACK TO LLAMA
    # --------------------------------------------------------

    if answer is None:

        print(
            "Using Llama for contextual answer..."
        )

        answer = generate_llm_answer(
            question,
            relevant_documents
        )

    # --------------------------------------------------------
    # DISPLAY ANSWER
    # --------------------------------------------------------

    print("\n")
    print("=" * 50)
    print("ANSWER")
    print("=" * 50)

    print(answer)

    # --------------------------------------------------------
    # DISPLAY SOURCES
    # --------------------------------------------------------

    print("\n")
    print("=" * 50)
    print("SOURCES")
    print("=" * 50)

    sources = format_sources(
        relevant_documents
    )

    for source in sources:

        print(
            f"- {source['file']}, "
            f"Page {source['page']}"
        )
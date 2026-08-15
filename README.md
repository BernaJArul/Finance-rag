# FinanceRAG - Financial Report Question Answering System

## 1. Project Overview

FinanceRAG is a Retrieval-Augmented Generation (RAG) based question-answering system designed to answer questions from financial reports.

The system processes financial PDF reports, converts them into searchable text chunks, creates vector embeddings, stores them in ChromaDB, retrieves relevant information, and generates answers using the retrieved financial context.

The project also performs deterministic financial calculations when the required values are available in the retrieved financial report.

---

## 2. Objectives

The main objectives of FinanceRAG are:

- Extract information from financial PDF reports.
- Split financial documents into searchable chunks.
- Generate embeddings using HuggingFace.
- Store embeddings in ChromaDB.
- Retrieve relevant financial information using semantic search.
- Identify the requested fiscal year and quarter.
- Identify the requested financial metric.
- Calculate derived financial metrics when required.
- Use Llama as a fallback language model.
- Provide source document and page information.
- Reduce hallucination by restricting answers to retrieved financial context.

---

## 3. Technologies Used

- Python
- LangChain
- HuggingFace Sentence Transformers
- ChromaDB
- Ollama
- Llama 3.2
- PyPDF
- Streamlit

### Embedding Model

sentence-transformers/all-MiniLM-L6-v2

### Language Model

llama3.2:3b

---

## 4. System Architecture

The FinanceRAG workflow is:

```text
PDF Financial Reports
        ↓
PDF Text Extraction
        ↓
Text Chunking
        ↓
HuggingFace Embeddings
        ↓
ChromaDB Vector Database
        ↓
User Question
        ↓
Semantic Retrieval
        ↓
Relevant Financial Context
        ↓
Metric and Quarter Identification
        ↓
Financial Calculation or Llama
        ↓
Answer + Sources
```

---

## 5. Project Files

### ingest.py

Reads PDF reports from the data folder, extracts text, splits the text into chunks, creates embeddings, and stores them in ChromaDB.

### rag.py

The main RAG question-answering engine.

It:

1. Receives the user's question.
2. Retrieves relevant document chunks.
3. Identifies the requested fiscal quarter.
4. Filters documents for the requested quarter.
5. Extracts financial values from the retrieved context.
6. Calculates derived metrics when required.
7. Uses Llama when deterministic extraction cannot answer the question.
8. Displays the answer and sources.

### app.py

Provides the user-facing Streamlit interface for FinanceRAG.

### data/

Contains the financial PDF reports used by the system.

### chroma_db/

Contains the persistent ChromaDB vector database.

### requirements.txt

Contains the Python dependencies required to run the project.

### README.md

Contains the project documentation, setup instructions, architecture, examples, limitations, and future enhancements.

---

## 6. Example Questions

The system can answer questions such as:

- What was Infosys' revenue in the first quarter of fiscal 2027?
- What was Infosys' gross profit in the first quarter of fiscal 2027?
- What was Infosys' operating profit in the first quarter of fiscal 2027?
- What was Infosys' operating margin in the first quarter of fiscal 2027?

Example:

Question:

What was Infosys' operating profit in the first quarter of fiscal 2027?

Answer:

The operating profit of Infosys in the first quarter of fiscal 2027 was $1,072 million.

Calculation:

Gross Profit ($1,600 million)
- Selling and Marketing Expenses ($270 million)
- General and Administration Expenses ($258 million)
= $1,072 million

Source:

Infosys_Q1_FY26-27.pdf, Page 3

---

## 7. Retrieval-Augmented Generation

FinanceRAG uses Retrieval-Augmented Generation instead of relying only on the language model.

The system first retrieves relevant information from the financial reports using semantic similarity search.

The retrieved information is then provided as context to the answering system.

This helps the system answer questions using the actual financial reports rather than relying only on the language model's pre-trained knowledge.

---

## 8. Deterministic Financial Calculation

For supported financial tables, FinanceRAG extracts financial values directly from the retrieved document context.

For example:

Operating Profit = Gross Profit - Selling and Marketing Expenses - General and Administration Expenses

For Q1 FY2027:

Operating Profit = 1600 - 270 - 258

Operating Profit = 1072 million US dollars

The values used in the calculation are extracted from the retrieved financial report context.

No financial values are hard-coded into the calculation.

---

## 9. Source Attribution

FinanceRAG displays the source PDF and page number used for the retrieved information.

Example:

- Infosys_Q1_FY26-27.pdf, Page 3

This allows the user to verify the answer against the original financial report.

---

## 10. Error Handling

If the requested information is not available in the provided reports, the system responds:

"I could not find this information in the provided reports."

The system also avoids answering questions about companies that are not represented in the retrieved financial reports.

For example, when asked about Apple's operating profit while only Infosys reports are available, the system does not provide an Infosys value as an answer.

---

## 11. How to Run the Project

### Step 1: Activate the virtual environment

```bash
source .venv/bin/activate
```

### Step 2: Install the required dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Make sure the financial PDFs are inside the data folder

```text
Finance-rag/
└── data/
    ├── Financial_Report_1.pdf
    ├── Financial_Report_2.pdf
    ├── Financial_Report_3.pdf
    └── Financial_Report_4.pdf
```

### Step 4: Create or update the vector database

```bash
python ingest.py
```

The ingestion process:

- Reads the financial PDFs.
- Extracts the text.
- Splits the text into chunks.
- Generates embeddings.
- Stores the embeddings in ChromaDB.

### Step 5: Run the RAG question-answering system

```bash
python rag.py
```

### Step 6: Enter a financial question

Example:

```text
What was Infosys' revenue in the first quarter of fiscal 2027?
```

---

## 12. Example Output

### Example 1: Revenue

Question:

What was Infosys' revenue in the first quarter of fiscal 2027?

Answer:

The revenue of Infosys in the first quarter of fiscal 2027 was $5,082 million.

Sources:

- Infosys_Q1_FY26-27.pdf, Page 2
- Infosys_Q1_FY26-27.pdf, Page 3

### Example 2: Gross Profit

Question:

What was Infosys' gross profit in the first quarter of fiscal 2027?

Answer:

The gross profit of Infosys in the first quarter of fiscal 2027 was $1,600 million.

Source:

- Infosys_Q1_FY26-27.pdf, Page 3

### Example 3: Operating Profit

Question:

What was Infosys' operating profit in the first quarter of fiscal 2027?

Answer:

The operating profit of Infosys in the first quarter of fiscal 2027 was $1,072 million.

Calculation:

Gross Profit = $1,600 million

Selling and Marketing Expenses = $270 million

General and Administration Expenses = $258 million

Operating Profit
= 1,600 - 270 - 258
= $1,072 million

Source:

- Infosys_Q1_FY26-27.pdf, Page 3

### Example 4: Operating Margin

Question:

What was Infosys' operating margin in the first quarter of fiscal 2027?

Answer:

The operating margin of Infosys in the first quarter of fiscal 2027 was approximately 21.1%.

Source:

- Infosys_Q1_FY26-27.pdf, Page 1

### Example 5: Company Mismatch

Question:

What was Apple's operating profit in the first quarter of fiscal 2027?

Answer:

I could not find this information in the provided reports.

The system does not incorrectly use the Infosys financial data to answer an Apple question.

---

## 13. Supported Financial Metrics

The system currently supports financial metrics including:

- Revenue
- Cost of sales and services
- Gross profit
- Selling and marketing expenses
- General and administration expenses
- Total operating expenses
- Operating profit
- Operating margin
- Profit before income taxes
- Income tax expense
- Net profit
- Basic EPS
- Diluted EPS

---

## 14. Advantages

FinanceRAG provides the following advantages:

- Uses real financial reports.
- Uses semantic retrieval.
- Provides source attribution.
- Supports financial calculations.
- Reduces hallucination.
- Uses a local LLM through Ollama.
- Uses persistent ChromaDB storage.
- Can be extended with additional financial reports.
- Provides a clear separation between document retrieval and answer generation.
- Allows users to ask questions using natural language.
- Provides answers based on retrieved financial report context.
- Prevents company-mismatch answers when the requested company is not available in the reports.

---

## 15. Limitations

The current deterministic financial extraction logic is designed around the financial table structure available in the indexed reports.

Additional financial report formats may require improved table extraction and metric detection.

The current project has limited support for automatic identification of arbitrary companies and fiscal quarters.

The system currently works best when the required financial information is available in the indexed reports.

The system does not automatically retrieve financial reports from the internet. Reports must first be added to the data folder and indexed.

---

## 16. Future Enhancements

Possible future improvements include:

- Support for multiple companies.
- Automatic company identification.
- Support for more fiscal quarters.
- Improved financial table extraction.
- Quarter-to-quarter comparison.
- Company-to-company comparison.
- Interactive Streamlit dashboard.
- Financial trend charts.
- Improved citation and document highlighting.
- Automatic report ingestion.
- More advanced financial reasoning.
- Support for additional financial metrics.
- Improved handling of complex financial tables.
- Automatic financial report discovery and downloading.
- Exporting answers and financial analysis to PDF or CSV.

---

## 17. Project Structure

```text
Finance-rag/
│
├── app.py
├── ingest.py
├── rag.py
├── rag_final_working.py
├── rag_before_company_fix.py
├── requirements.txt
├── README.md
│
├── data/
│   └── Financial PDF reports
│
└── chroma_db/
    └── ChromaDB vector database
```

### Description of Important Files

app.py

User-facing Streamlit application interface.

ingest.py

Reads PDF reports, extracts text, creates chunks, generates embeddings, and stores them in ChromaDB.

rag.py

Main question-answering system that performs retrieval, financial extraction, calculations, company validation, and LLM-based answering.

rag_final_working.py

Backup copy of the working RAG implementation.

rag_before_company_fix.py

Backup copy of the RAG implementation before company-mismatch handling was added.

requirements.txt

Contains the Python packages required to run the project.

data/

Stores the financial PDF reports.

chroma_db/

Stores the persistent vector database created from the financial reports.

README.md

Contains the project documentation and instructions.

---

## 18. Conclusion

FinanceRAG demonstrates how Retrieval-Augmented Generation can be applied to financial document analysis.

The system combines:

- PDF processing
- Text extraction
- Text chunking
- Semantic embeddings
- Vector database storage
- Semantic retrieval
- Financial value extraction
- Deterministic financial calculations
- Local language model reasoning
- Company and fiscal-period validation
- Source attribution

The system retrieves relevant information from financial reports before generating an answer. This helps reduce hallucination and allows the user to verify the answer using the displayed source document and page number.

The project demonstrates an end-to-end AI pipeline:

Document
↓
Text Extraction
↓
Chunking
↓
Embeddings
↓
Vector Database
↓
Semantic Retrieval
↓
Company and Quarter Identification
↓
Financial Metric Identification
↓
Financial Reasoning
↓
Answer Generation
↓
Source Attribution

FinanceRAG provides a practical example of how RAG technology can be used to make financial reports easier to search, understand, and analyze.

The project can be further extended to support multiple companies, additional fiscal quarters, financial comparisons, interactive dashboards, charts, automatic report ingestion, and more advanced financial reasoning.

---

## Project Status

The core FinanceRAG pipeline has been implemented and tested.

Current verified capabilities include:

- PDF ingestion
- Document chunking
- HuggingFace embeddings
- ChromaDB vector storage
- Semantic retrieval
- Fiscal quarter identification
- Financial metric identification
- Deterministic financial calculations
- Operating profit calculation
- Revenue extraction
- Gross profit extraction
- Operating margin extraction
- Company mismatch protection
- Llama fallback answering
- Source and page attribution
- Dependency documentation
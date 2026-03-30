# Department Document Generation & Rename Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generate 60 realistic consulting firm PDFs across three departments (engineering, accounting, hr) and rename `company_id` to `department` throughout the codebase.

**Architecture:** A standalone `generate_docs.py` script calls Claude API to produce a JSON fixture, then converts it to PDFs using fpdf2. Separately, `tutorial.py` is updated to use `department` instead of `company_id` in all metadata, filters, and user context.

**Tech Stack:** `anthropic` (already installed), `fpdf2` (new), `json`, `re` (stdlib)

---

### Task 1: Install fpdf2 and delete old PDFs

**Files:**
- Modify: `requirements.txt`
- Delete: `pdfs/companyA/`, `pdfs/companyB/`, `pdfs/companyC/`
- Delete: `chroma_db/` (must be rebuilt with new metadata)

- [ ] **Step 1: Install fpdf2**

```bash
pip install fpdf2
```

- [ ] **Step 2: Delete old company PDF folders and Chroma index**

```bash
rm -rf pdfs/companyA pdfs/companyB pdfs/companyC chroma_db/
```

- [ ] **Step 3: Create department folders**

```bash
mkdir -p pdfs/engineering pdfs/accounting pdfs/hr
```

- [ ] **Step 4: Commit**

```bash
git add pdfs/
git commit -m "chore: replace company folders with department folders, remove old PDFs"
```

---

### Task 2: Create the document generation script

**Files:**
- Create: `generate_docs.py`

- [ ] **Step 1: Create `generate_docs.py`**

Write the full script to `generate_docs.py`:

```python
import json
import os
import re
from anthropic import Anthropic
from fpdf import FPDF

DEPARTMENTS = ["engineering", "accounting", "hr"]

PERMISSION_DISTRIBUTION = {
    "public": 10,
    "internal": 7,
    "confidential": 3,
}

DEPARTMENT_CONTEXT = {
    "engineering": (
        "a software engineering department at a 50-person consulting firm. "
        "Topics include: coding standards, architecture decision records, project specs, "
        "deployment procedures, tech stack guidelines, incident response playbooks, "
        "code review policies, sprint processes, security practices, API guidelines, "
        "testing standards, CI/CD procedures, on-call rotations, tech debt policies, "
        "client engagement technical processes, development environment setup, "
        "monitoring and alerting procedures, data handling policies, "
        "performance review criteria for engineers, and knowledge sharing practices."
    ),
    "accounting": (
        "an accounting department at a 50-person consulting firm. "
        "Topics include: expense report policies, invoice procedures, budget templates, "
        "audit procedures, financial reporting guidelines, vendor payment processes, "
        "travel reimbursement rules, procurement policies, tax compliance procedures, "
        "accounts receivable processes, accounts payable workflows, month-end close procedures, "
        "client billing guidelines, timesheet policies, petty cash procedures, "
        "financial approval thresholds, cost center guidelines, revenue recognition policies, "
        "credit card usage policies, and charitable donation matching procedures."
    ),
    "hr": (
        "an HR department at a 50-person consulting firm. "
        "Topics include: PTO policies, onboarding checklists, performance review procedures, "
        "benefits guides, termination procedures, employee handbook sections, "
        "interview guidelines, compensation philosophy, remote work policies, "
        "diversity and inclusion initiatives, training and development programs, "
        "workplace safety procedures, grievance procedures, promotion criteria, "
        "parental leave policies, employee assistance programs, dress code policies, "
        "social media policies, referral bonus programs, and exit interview procedures."
    ),
}

FIXTURE_PATH = os.path.join(os.path.dirname(__file__), "docs_fixture.json")
PDF_BASE = os.path.join(os.path.dirname(__file__), "pdfs")


def generate_fixture():
    """Call Claude API to generate document content for all departments."""
    if os.path.exists(FIXTURE_PATH):
        print(f"Fixture already exists at {FIXTURE_PATH}, skipping generation.")
        return

    client = Anthropic()
    all_docs = []

    for department in DEPARTMENTS:
        permission_list = []
        for level, count in PERMISSION_DISTRIBUTION.items():
            permission_list.extend([level] * count)

        prompt = (
            f"Generate exactly 20 internal documents for {DEPARTMENT_CONTEXT[department]}\n\n"
            f"Each document should be 1-2 paragraphs (roughly 200-400 words) of realistic, "
            f"specific content — not generic filler. Include concrete details like specific "
            f"numbers, dates, procedures, and names where appropriate.\n\n"
            f"The 20 documents must have these exact permission levels in this order:\n"
            f"{json.dumps(permission_list)}\n\n"
            f"Return a JSON array of 20 objects, each with:\n"
            f'- "title": a descriptive document title (3-6 words)\n'
            f'- "department": "{department}"\n'
            f'- "permission_level": the permission level from the list above (in order)\n'
            f'- "content": the document text (200-400 words)\n\n'
            f"Return ONLY the JSON array, no other text."
        )

        print(f"Generating documents for {department}...")
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=16000,
            messages=[{"role": "user", "content": prompt}],
        )

        raw = response.content[0].text
        # Strip markdown code fences if present
        raw = re.sub(r"^```(?:json)?\s*\n?", "", raw.strip())
        raw = re.sub(r"\n?```\s*$", "", raw.strip())

        docs = json.loads(raw)
        all_docs.extend(docs)
        print(f"  Generated {len(docs)} documents for {department}")

    with open(FIXTURE_PATH, "w") as f:
        json.dump(all_docs, f, indent=2)

    print(f"Fixture saved to {FIXTURE_PATH} ({len(all_docs)} documents)")


def slugify(text):
    """Convert a title to a filename-safe slug."""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[\s_]+", "-", text)
    text = re.sub(r"-+", "-", text)
    return text.strip("-")


def generate_pdfs():
    """Convert the fixture JSON into PDF files."""
    with open(FIXTURE_PATH) as f:
        docs = json.load(f)

    for doc in docs:
        department = doc["department"]
        permission_level = doc["permission_level"]
        title = doc["title"]
        content = doc["content"]

        dept_dir = os.path.join(PDF_BASE, department)
        os.makedirs(dept_dir, exist_ok=True)

        filename = f"{slugify(title)}-{permission_level}.pdf"
        filepath = os.path.join(dept_dir, filename)

        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 16)
        pdf.cell(0, 10, title, new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "", 9)
        pdf.set_text_color(128, 128, 128)
        pdf.cell(0, 6, f"Department: {department} | Classification: {permission_level} | Owner: hr-team", new_x="LMARGIN", new_y="NEXT")
        pdf.set_text_color(0, 0, 0)
        pdf.ln(4)
        pdf.set_font("Helvetica", "", 11)
        pdf.multi_cell(0, 6, content)

        pdf.output(filepath)

    print(f"Generated {len(docs)} PDFs across {len(set(d['department'] for d in docs))} departments")


if __name__ == "__main__":
    generate_fixture()
    generate_pdfs()
```

- [ ] **Step 2: Verify the script is syntactically valid**

Run: `python3 -c "import ast; ast.parse(open('generate_docs.py').read()); print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add generate_docs.py
git commit -m "feat: add document generation script with Claude API + fpdf2"
```

---

### Task 3: Run the generation script

- [ ] **Step 1: Run the script**

```bash
python3 generate_docs.py
```

Expected output:
```
Generating documents for engineering...
  Generated 20 documents for engineering
Generating documents for accounting...
  Generated 20 documents for accounting
Generating documents for hr...
  Generated 20 documents for hr
Fixture saved to docs_fixture.json (60 documents)
Generated 60 PDFs across 3 departments
```

- [ ] **Step 2: Verify the output**

```bash
ls pdfs/engineering/ | wc -l
ls pdfs/accounting/ | wc -l
ls pdfs/hr/ | wc -l
```

Expected: 20 files in each directory.

- [ ] **Step 3: Verify permission levels in filenames**

```bash
ls pdfs/engineering/ | grep -c "public"
ls pdfs/engineering/ | grep -c "internal"
ls pdfs/engineering/ | grep -c "confidential"
```

Expected: 10, 7, 3 respectively.

- [ ] **Step 4: Commit the fixture and generated PDFs**

```bash
git add docs_fixture.json pdfs/
git commit -m "feat: generate 60 department documents (20 per department)"
```

---

### Task 4: Rename `company_id` to `department` in tutorial.py

**Files:**
- Modify: `tutorial.py:21-25` (current_user)
- Modify: `tutorial.py:59-66` (Chroma filter in prompt_with_context)
- Modify: `tutorial.py:106-157` (vector store loading section)
- Modify: `tutorial.py:113` (collection name)
- Modify: `tutorial.py:161` (startup message)

- [ ] **Step 1: Update `current_user`**

Replace:
```python
current_user = {
    "company_id": "companyA",
    "permission_level": "internal",
    "name": "Alice",
}
```

With:
```python
current_user = {
    "department": "engineering",
    "permission_level": "internal",
    "name": "Alice",
}
```

- [ ] **Step 2: Update the Chroma filter in `prompt_with_context`**

Replace:
```python
    retrieved_docs = vector_store.similarity_search(
        last_query,
        filter={
            "$and": [
                {"company_id": current_user["company_id"]},
                {"permission_level": {"$in": allowed_levels}},
            ]
        },
    )
```

With:
```python
    retrieved_docs = vector_store.similarity_search(
        last_query,
        filter={
            "$and": [
                {"department": current_user["department"]},
                {"permission_level": {"$in": allowed_levels}},
            ]
        },
    )
```

- [ ] **Step 3: Update the Chroma collection name**

Replace:
```python
vector_store = Chroma(
    collection_name="company_docs",
    embedding_function=embeddings,
    persist_directory=chroma_path,
)
```

With:
```python
vector_store = Chroma(
    collection_name="department_docs",
    embedding_function=embeddings,
    persist_directory=chroma_path,
)
```

- [ ] **Step 4: Update the vector store loading section**

Replace:
```python
# 3. Vector Store (single Chroma index, all companies)
base_dir = os.path.dirname(__file__)
pdf_base = os.path.join(base_dir, "pdfs")
chroma_path = os.path.join(base_dir, "chroma_db")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, add_start_index=True)

vector_store = Chroma(
    collection_name="department_docs",
    embedding_function=embeddings,
    persist_directory=chroma_path,
)

if vector_store._collection.count() == 0:
    print("No existing Chroma data found. Building index from PDFs...")
    company_dirs = sorted(
        d for d in os.listdir(pdf_base)
        if os.path.isdir(os.path.join(pdf_base, d))
    )

    all_splits = []
    for company_id in company_dirs:
        company_pdf_dir = os.path.join(pdf_base, company_id)
        pdf_files = sorted(glob.glob(os.path.join(company_pdf_dir, "*.pdf")))

        if not pdf_files:
            print(f"  No PDFs found for {company_id}, skipping.")
            continue

        docs = []
        for pdf_path in pdf_files:
            filename = os.path.basename(pdf_path)
            permission_level = parse_permission_level(filename)
            print(f"  Loading {filename} (company: {company_id}, permission: {permission_level})...")

            loader = PyPDFLoader(pdf_path)
            loaded_docs = loader.load()

            for doc in loaded_docs:
                doc.metadata["company_id"] = company_id
                doc.metadata["permission_level"] = permission_level
                doc.metadata["owner"] = "hr-team"

            docs.extend(loaded_docs)

        print(f"  Loaded {len(docs)} pages from {len(pdf_files)} PDF(s) for {company_id}")
        all_splits.extend(text_splitter.split_documents(docs))

    print(f"Total splits across all companies: {len(all_splits)}")
    vector_store.add_documents(all_splits)
    print("Chroma index built and persisted.")
else:
    print(f"Loaded existing Chroma index ({vector_store._collection.count()} documents).")
```

With:
```python
# 3. Vector Store (single Chroma index, all departments)
base_dir = os.path.dirname(__file__)
pdf_base = os.path.join(base_dir, "pdfs")
chroma_path = os.path.join(base_dir, "chroma_db")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, add_start_index=True)

vector_store = Chroma(
    collection_name="department_docs",
    embedding_function=embeddings,
    persist_directory=chroma_path,
)

if vector_store._collection.count() == 0:
    print("No existing Chroma data found. Building index from PDFs...")
    department_dirs = sorted(
        d for d in os.listdir(pdf_base)
        if os.path.isdir(os.path.join(pdf_base, d))
    )

    all_splits = []
    for department in department_dirs:
        dept_pdf_dir = os.path.join(pdf_base, department)
        pdf_files = sorted(glob.glob(os.path.join(dept_pdf_dir, "*.pdf")))

        if not pdf_files:
            print(f"  No PDFs found for {department}, skipping.")
            continue

        docs = []
        for pdf_path in pdf_files:
            filename = os.path.basename(pdf_path)
            permission_level = parse_permission_level(filename)
            print(f"  Loading {filename} (department: {department}, permission: {permission_level})...")

            loader = PyPDFLoader(pdf_path)
            loaded_docs = loader.load()

            for doc in loaded_docs:
                doc.metadata["department"] = department
                doc.metadata["permission_level"] = permission_level
                doc.metadata["owner"] = "hr-team"

            docs.extend(loaded_docs)

        print(f"  Loaded {len(docs)} pages from {len(pdf_files)} PDF(s) for {department}")
        all_splits.extend(text_splitter.split_documents(docs))

    print(f"Total splits across all departments: {len(all_splits)}")
    vector_store.add_documents(all_splits)
    print("Chroma index built and persisted.")
else:
    print(f"Loaded existing Chroma index ({vector_store._collection.count()} documents).")
```

- [ ] **Step 5: Update the startup message**

Replace:
```python
print(f"\nReady! Logged in as {current_user['name']} ({current_user['company_id']}, {current_user['permission_level']} access).")
print("Ask questions about your company's PDFs (type 'quit' to exit).")
```

With:
```python
print(f"\nReady! Logged in as {current_user['name']} ({current_user['department']}, {current_user['permission_level']} access).")
print("Ask questions about your department's documents (type 'quit' to exit).")
```

- [ ] **Step 6: Update the system prompt in prompt_with_context**

Replace:
```python
        "You are an HR representative answering questions from employees about current PTO policies. "
```

With:
```python
        "You are a helpful assistant answering questions from employees at a consulting firm. "
```

- [ ] **Step 7: Verify the file is syntactically valid**

Run: `python3 -c "import ast; ast.parse(open('tutorial.py').read()); print('OK')"`
Expected: `OK`

- [ ] **Step 8: Commit**

```bash
git add tutorial.py
git commit -m "feat: rename company_id to department, update to consulting firm context"
```

---

### Task 5: End-to-end verification

- [ ] **Step 1: Delete Chroma index to force rebuild**

```bash
rm -rf chroma_db/
```

- [ ] **Step 2: Run the chatbot**

```bash
python3 tutorial.py
```

Expected:
- PDFs loaded from `pdfs/engineering/`, `pdfs/accounting/`, `pdfs/hr/`
- Metadata shows `department:` instead of `company:`
- Startup: `Logged in as Alice (engineering, internal access)`

- [ ] **Step 3: Test department filtering**

Ask: "What are the coding standards?" — should return engineering docs.
Change `current_user["department"]` to `"hr"`, delete `chroma_db/`, restart.
Ask: "What is the PTO policy?" — should return HR docs.

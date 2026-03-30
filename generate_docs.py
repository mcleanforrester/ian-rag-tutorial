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

import os
import glob

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document

from identity.permissions import parse_permission_level


def load_department_docs(pdf_base: str) -> list[Document]:
    """Scan pdfs/{department}/ folders, load all PDFs, attach metadata."""
    department_dirs = sorted(
        d for d in os.listdir(pdf_base)
        if os.path.isdir(os.path.join(pdf_base, d))
    )

    all_docs = []
    for department in department_dirs:
        dept_pdf_dir = os.path.join(pdf_base, department)
        pdf_files = sorted(glob.glob(os.path.join(dept_pdf_dir, "*.pdf")))

        if not pdf_files:
            print(f"  No PDFs found for {department}, skipping.")
            continue

        for pdf_path in pdf_files:
            filename = os.path.basename(pdf_path)
            permission_level = parse_permission_level(filename)
            print(f"  Loading {filename} (department: {department}, permission: {permission_level.name})...")

            loader = PyPDFLoader(pdf_path)
            loaded_docs = loader.load()

            for doc in loaded_docs:
                doc.metadata["department"] = department
                doc.metadata["permission_level"] = permission_level.name
                doc.metadata["owner"] = "hr-team"

            all_docs.extend(loaded_docs)

        print(f"  Loaded docs from {len(pdf_files)} PDF(s) for {department}")

    return all_docs

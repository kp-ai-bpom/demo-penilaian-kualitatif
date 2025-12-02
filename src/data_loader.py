import os
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain.schema import Document

def load_pdf_documents(pdf_path):
    """Load PDF documents"""
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    
    loader = PyPDFLoader(pdf_path)
    return loader.load()

def load_skj_documents(skj_folder):
    """Load all SKJ documents from folder"""
    if not os.path.exists(skj_folder):
        print(f"⚠️ SKJ folder not found: {skj_folder}")
        return []
    
    all_docs = []
    for filename in os.listdir(skj_folder):
        if filename.endswith('.docx'):
            file_path = os.path.join(skj_folder, filename)
            loader = Docx2txtLoader(file_path)
            docs = loader.load()
            for doc in docs:
                doc.metadata['source'] = filename
                doc.metadata['type'] = 'SKJ'
            all_docs.extend(docs)
            print(f"✅ Loaded: {filename}")
    
    return all_docs
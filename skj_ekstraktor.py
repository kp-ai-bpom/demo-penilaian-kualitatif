# skj_extractor.py
import os
import json
import re
from datetime import datetime
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
import sys
import os

# Add current directory to path to import prompt
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from prompt.prompt import EXTRACT_SKJ_PROMPT

class SKJExtractor:
    def __init__(self, llm, skj_folder="data/skj_documents"):
        self.llm = llm
        self.skj_folder = Path(skj_folder)
        self.skj_folder.mkdir(parents=True, exist_ok=True)
        self.extraction_chain = LLMChain(llm=llm, prompt=EXTRACT_SKJ_PROMPT)
        
    def load_skj_document(self, file_path):
        """Load SKJ document based on file type"""
        file_path = Path(file_path)
        
        if file_path.suffix.lower() == '.pdf':
            loader = PyPDFLoader(str(file_path))
        elif file_path.suffix.lower() in ['.docx', '.doc']:
            loader = Docx2txtLoader(str(file_path))
        elif file_path.suffix.lower() == '.txt':
            loader = TextLoader(str(file_path), encoding='utf-8')
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")
            
        documents = loader.load()
        full_text = "\n".join([doc.page_content for doc in documents])
        return self.preprocess_text(full_text)
    
    def preprocess_text(self, text):
        """Preprocess SKJ text"""
        # Remove extra whitespaces and normalize
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'-\s*\d+\s*-', '', text)  # Remove page numbers
        return text.strip()
    
    def extract_skj(self, file_path):
        """Extract SKJ data from document"""
        try:
            print(f"Memproses file: {file_path}")
            skj_text = self.load_skj_document(file_path)
            
            # Extract using LLM
            result = self.extraction_chain.invoke({"skj_text": skj_text[:8000]})  # Limit text length
            extracted_data = self.parse_json_output(result['text'])
            
            # Add metadata
            extracted_data['metadata']['sumber_file'] = os.path.basename(file_path)
            extracted_data['metadata']['extracted_at'] = datetime.now().isoformat()
            
            return extracted_data
            
        except Exception as e:
            print(f"Error extracting SKJ from {file_path}: {str(e)}")
            return None
    
    def parse_json_output(self, text):
        """Parse JSON output from LLM"""
        try:
            # Extract JSON from text
            json_match = re.search(r'\{.*\}', text, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                return json.loads(json_str)
            else:
                raise ValueError("No JSON found in LLM output")
        except json.JSONDecodeError as e:
            print(f"JSON parsing error: {e}")
            # Return empty structure
            return self.get_empty_skj_structure()
    
    def get_empty_skj_structure(self):
        """Return empty SKJ structure"""
        return {
            "jabatan": "",
            "kode_jabatan": "",
            "unit_organisasi": "",
            "ringkasan_tugas": "",
            "kompetensi_manajerial": [],
            "kompetensi_teknis": [],
            "persyaratan_jabatan": {
                "pendidikan": "",
                "pelatihan": "",
                "pengalaman": ""
            },
            "metadata": {
                "sumber_file": "",
                "extracted_at": "",
                "extractor_version": "v1"
            }
        }
    
    def get_available_skj_files(self):
        """Get list of available SKJ files"""
        skj_files = []
        for ext in ['*.pdf', '*.docx', '*.doc', '*.txt']:
            skj_files.extend(self.skj_folder.glob(ext))
        return [f.name for f in skj_files]
    
    def load_skj_data(self, filename):
        """Load extracted SKJ data from file"""
        # Simpan di data/skj/extracted untuk konsistensi
        extracted_dir = Path("data/skj/extracted")
        skj_file = extracted_dir / f"{Path(filename).stem}.json"
        if skj_file.exists():
            with open(skj_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return None
    
    def save_skj_data(self, skj_data, filename):
        """Save extracted SKJ data"""
        # Simpan di data/skj/extracted untuk konsistensi
        output_dir = Path("data/skj/extracted")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_file = output_dir / f"{Path(filename).stem}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(skj_data, f, indent=2, ensure_ascii=False)
        
        print(f"✅ SKJ data tersimpan di: {output_file}")
        return output_file

class SKJManager:
    def __init__(self, extractor):
        self.extractor = extractor
        self.selected_skj = None
        
    def list_available_skj(self):
        """List available SKJ files"""
        return self.extractor.get_available_skj_files()
    
    def select_skj(self, filename):
        """Select SKJ for assessment"""
        # Try to load existing extracted data
        skj_data = self.extractor.load_skj_data(filename)
        
        if not skj_data:
            # Extract from original file
            file_path = self.extractor.skj_folder / filename
            if file_path.exists():
                skj_data = self.extractor.extract_skj(file_path)
                if skj_data:
                    self.extractor.save_skj_data(skj_data, filename)
        
        self.selected_skj = skj_data
        return skj_data
    
    def get_selected_skj_context(self):
        """Get context from selected SKJ for RAG"""
        if not self.selected_skj:
            return ""
        
        context_parts = []
        
        # Add basic job information
        context_parts.append(f"JABATAN: {self.selected_skj.get('jabatan', '')}")
        context_parts.append(f"KODE: {self.selected_skj.get('kode_jabatan', '')}")
        context_parts.append(f"UNIT: {self.selected_skj.get('unit_organisasi', '')}")
        context_parts.append(f"RINGKASAN TUGAS: {self.selected_skj.get('ringkasan_tugas', '')}")
        
        # Add managerial competencies
        if self.selected_skj.get('kompetensi_manajerial'):
            context_parts.append("\nKOMPETENSI MANAJERIAL:")
            for comp in self.selected_skj['kompetensi_manajerial']:
                context_parts.append(f"- {comp.get('nama_kompetensi', '')}: {comp.get('definisi', '')}")
                for level in ['level_1', 'level_2', 'level_3', 'level_4']:
                    if comp.get(level):
                        context_parts.append(f"  {level.upper()}: {comp[level].get('deskripsi', '')}")
                        for indicator in comp[level].get('indikator_perilaku', []):
                            context_parts.append(f"    • {indicator}")
        
        # Add technical competencies
        if self.selected_skj.get('kompetensi_teknis'):
            context_parts.append("\nKOMPETENSI TEKNIS:")
            for comp in self.selected_skj['kompetensi_teknis']:
                context_parts.append(f"- {comp.get('nama_kompetensi', '')}: {comp.get('definisi', '')}")
        
        return "\n".join(context_parts)
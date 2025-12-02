# src/assessment_engine.py
import json
import os
import re
from datetime import datetime
from typing import Dict, List, Any, Optional

from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter


class JobCompetencyExtractor:
    def __init__(self, llm, embedding_model):
        self.llm = llm
        self.embedding_model = embedding_model
        self.job_mapping = {}

    def extract_from_documents(self, documents: List[Document]) -> Dict[str, Any]:
        """Extract job competencies from documents using LLM"""
        print("🤖 LLM extracting job competencies from documents...")
        
        # Take sample content for analysis (first 5000 chars from first 5 docs)
        sample_content = ""
        for doc in documents[:5]:
            sample_content += doc.page_content[:1000] + "\n\n"

        extraction_prompt = f"""
        ANALISIS DOKUMEN STANDAR KOMPETENSI JABATAN:

        {sample_content}

        TUGAS: Identifikasi semua JABATAN dan KOMPETENSI yang disebutkan dalam dokumen.

        FORMAT OUTPUT JSON:
        {{
          "nama_jabatan": {{
            "level": ["level1", "level2", ...],
            "kompetensi_teknis": ["komp1", "komp2", ...],
            "kompetensi_manajerial": ["komp1", "komp2", ...], 
            "kompetensi_sosial_kultural": ["komp1", "komp2", ...],
            "indikator_perilaku": {{
              "level_1": ["indikator1", "indikator2"],
              "level_2": ["indikator1", "indikator2"],
              "level_3": ["indikator1", "indikator2"],
              "level_4": ["indikator1", "indikator2"]
            }}
          }}
        }}

        HANYA output JSON, tanpa penjelasan tambahan.
        """

        try:
            response = self.llm.invoke(extraction_prompt)
            mapping_text = response.content
            
            # Extract JSON from response
            start_idx = mapping_text.find('{')
            end_idx = mapping_text.rfind('}') + 1
            
            if start_idx != -1 and end_idx != -1:
                json_str = mapping_text[start_idx:end_idx]
                job_mapping = json.loads(json_str)
                print(f"✅ LLM berhasil extract {len(job_mapping)} jabatan")
                self.job_mapping = job_mapping
                return job_mapping
            else:
                raise ValueError("JSON tidak ditemukan dalam response LLM")
                
        except Exception as e:
            print(f"❌ Error extracting with LLM: {e}")
            print("🔄 Using fallback mapping...")
            self.job_mapping = self._create_fallback_mapping()
            return self.job_mapping

    def _create_fallback_mapping(self) -> Dict[str, Any]:
        """Create fallback mapping when LLM extraction fails"""
        return {
            "Analis Kepegawaian": {
                "level": ["Ahli Pertama", "Ahli Muda", "Ahli Madya", "Ahli Utama"],
                "kompetensi_teknis": ["Manajemen SDM", "Analisis Jabatan", "Rekrutmen dan Seleksi", "Pengembangan Kompetensi", "Evaluasi Kinerja"],
                "kompetensi_manajerial": ["Perencanaan", "Pengorganisasian", "Pengawasan", "Evaluasi Kinerja", "Pengambilan Keputusan"],
                "kompetensi_sosial_kultural": ["Komunikasi", "Kerjasama", "Pelayanan Publik", "Integritas", "Adaptabilitas"],
                "indikator_perilaku": {
                    "level_1": [
                        "Menyelesaikan tugas rutin sesuai prosedur",
                        "Berkomunikasi efektif dengan atasan langsung",
                        "Mengumpulkan data yang diperlukan"
                    ],
                    "level_2": [
                        "Mengkoordinasikan tim kecil dalam penyelesaian tugas",
                        "Melakukan analisis data sederhana",
                        "Menyusun laporan periodik yang komprehensif"
                    ],
                    "level_3": [
                        "Menyusun strategi pengembangan SDM",
                        "Memimpin unit kerja dengan efektif",
                        "Melakukan analisis kebijakan kepegawaian"
                    ],
                    "level_4": [
                        "Mengembangkan kebijakan nasional di bidang kepegawaian",
                        "Memimpin organisasi dengan visi yang jelas",
                        "Melakukan inovasi sistem manajemen SDM"
                    ]
                }
            },
            "Pengawas Pemerintahan": {
                "level": ["Pengawas", "Pengawas Madya", "Pengawas Utama"],
                "kompetensi_teknis": ["Pengawasan", "Audit", "Evaluasi Kinerja", "Analisis Risiko", "Investigasi"],
                "kompetensi_manajerial": ["Kepemimpinan", "Pengambilan Keputusan", "Pengendalian", "Manajemen Konflik", "Strategi Pengawasan"],
                "kompetensi_sosial_kultural": ["Integritas", "Objektivitas", "Ketegasan", "Komunikasi Assertif", "Keberanian Moral"],
                "indikator_perilaku": {
                    "level_1": [
                        "Melakukan pengawasan rutin sesuai checklist",
                        "Menyusun laporan temuan pengawasan",
                        "Mengikuti prosedur audit standar"
                    ],
                    "level_2": [
                        "Memimpin tim pengawasan kecil",
                        "Melakukan analisis temuan kompleks",
                        "Menyusun rekomendasi perbaikan yang actionable"
                    ],
                    "level_3": [
                        "Mengembangkan sistem pengawasan yang efektif",
                        "Memimpin investigasi kasus kompleks",
                        "Melakukan evaluasi kebijakan organisasi"
                    ],
                    "level_4": [
                        "Menyusun kebijakan pengawasan nasional",
                        "Memimpin organisasi pengawasan yang kredibel",
                        "Mengembangkan standar audit nasional"
                    ]
                }
            },
            "Pranata Komputer": {
                "level": ["Pranata Komputer", "Pranata Komputer Madya", "Pranata Komputer Utama"],
                "kompetensi_teknis": ["Pengembangan Sistem", "Manajemen Basis Data", "Keamanan Informasi", "Analisis Kebutuhan", "Arsitektur Enterprise"],
                "kompetensi_manajerial": ["Manajemen Proyek TI", "Koordinasi Tim", "Perencanaan Teknologi", "Penganggaran TI", "Manajemen Vendor"],
                "kompetensi_sosial_kultural": ["Kolaborasi", "Komunikasi Teknis", "Adaptabilitas", "Service Orientation", "Problem Solving"],
                "indikator_perilaku": {
                    "level_1": [
                        "Mengembangkan modul aplikasi sederhana",
                        "Melakukan pemeliharaan sistem rutin",
                        "Membuat dokumentasi teknis yang jelas"
                    ],
                    "level_2": [
                        "Menganalisis kebutuhan pengguna secara komprehensif",
                        "Mengkoordinasikan pengembangan sistem terintegrasi",
                        "Mengelola basis data dengan optimal"
                    ],
                    "level_3": [
                        "Mendesain arsitektur sistem enterprise",
                        "Memanajemen proyek TI yang kompleks",
                        "Menyusun strategi transformasi digital"
                    ],
                    "level_4": [
                        "Mengembangkan kebijakan teknologi nasional",
                        "Melakukan inovasi sistem pemerintahan digital",
                        "Menyusun standarisasi TI nasional"
                    ]
                }
            }
        }

    def save_mapping(self, filepath: str):
        """Save job mapping to file"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.job_mapping, f, indent=2, ensure_ascii=False)
        print(f"✅ Mapping disimpan: {filepath}")

    def create_vector_store(self, documents: List[Document]) -> FAISS:
        """Create vector store from documents"""
        print("🏗️ Creating vector store from documents...")
        
        # Preprocess documents
        for doc in documents:
            doc.page_content = self._preprocess_text(doc.page_content)
        
        # Split documents
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        split_docs = text_splitter.split_documents(documents)
        
        print(f"✅ Created {len(split_docs)} document chunks")
        return FAISS.from_documents(split_docs, self.embedding_model)

    def _preprocess_text(self, text: str) -> str:
        """Preprocess text for better embedding"""
        # Remove page number footers
        text = re.sub(r'-\s*\d+\s*-', '', text)
        # Normalize whitespace
        text = re.sub(r'\n\s*\n', '\n\n', text)
        text = re.sub(r'\n(?!\n)', ' ', text)
        text = re.sub(r' +', ' ', text)
        return text.strip().lower()


class RealAssessmentSystem:
    def __init__(self, vector_db: FAISS, llm):
        self.vector_db = vector_db
        self.llm = llm
        self.job_mapping = self._load_mapping()

    def _load_mapping(self) -> Dict[str, Any]:
        """Load job mapping from file"""
        mapping_path = "../models/job_competency_mapping.json"
        if os.path.exists(mapping_path):
            with open(mapping_path, 'r', encoding='utf-8') as f:
                mapping = json.load(f)
                print(f"✅ Loaded job mapping: {len(mapping)} positions")
                return mapping
        else:
            print("❌ Mapping file not found, using empty mapping")
            return {}

    def show_available_options(self):
        """Display all available options from mapping"""
        print("\n" + "="*60)
        print("🎯 OPTIONS AVAILABLE (From LLM Extraction)")
        print("="*60)
        
        print(f"\n📋 TOTAL JABATAN: {len(self.job_mapping)}")
        
        for job_name, job_info in self.job_mapping.items():
            print(f"\n🔹 {job_name.upper()}")
            print(f"   📈 Level: {', '.join(job_info['level'])}")
            print(f"   🔧 Teknis: {', '.join(job_info['kompetensi_teknis'][:3])}" + 
                  ("..." if len(job_info['kompetensi_teknis']) > 3 else ""))
            print(f"   💼 Manajerial: {', '.join(job_info['kompetensi_manajerial'][:3])}" +
                  ("..." if len(job_info['kompetensi_manajerial']) > 3 else ""))
            print(f"   🤝 Sosial Kultural: {', '.join(job_info['kompetensi_sosial_kultural'][:3])}" +
                  ("..." if len(job_info['kompetensi_sosial_kultural']) > 3 else ""))

    def get_job_list(self) -> List[str]:
        """Get list of available jobs"""
        return list(self.job_mapping.keys())

    def get_job_info(self, job_name: str) -> Dict[str, Any]:
        """Get detailed info for a specific job"""
        return self.job_mapping.get(job_name, {})

    def get_competencies_by_type(self, job_name: str, comp_type: str) -> List[str]:
        """Get competencies by type for a job"""
        job_info = self.job_mapping.get(job_name, {})
        return job_info.get(f"kompetensi_{comp_type}", [])

    def generate_questions_with_llm(self, jabatan: str, level: str, kompetensi: str) -> str:
        """Generate assessment questions using LLM based on mapping"""
        print(f"🤖 GENERATING QUESTIONS: {kompetensi} ({level})")
        
        # Get behavioral indicators from mapping
        level_key = self._get_level_key(level)
        indicators = self.job_mapping[jabatan]["indikator_perilaku"].get(level_key, [])
        
        prompt = f"""
        BUATKAN SOAL ASSESSMENT untuk mengukur kompetensi: {kompetensi}
        
        KONTEKS:
        - Jabatan: {jabatan}
        - Level: {level} 
        - Indikator Perilaku: {', '.join(indicators[:3])}
        
        INSTRUKSI:
        1. Buat 1 studi kasus REALISTIC yang relevan dengan jabatan {jabatan}
        2. Kasus harus mengukur indikator perilaku: {', '.join(indicators[:3])}
        3. Berikan pertanyaan yang membutuhkan jawaban essay/uraian
        4. Situasi harus challenging sesuai level {level}
        5. Kasus harus kontekstual dengan pekerjaan sehari-hari
        
        FORMAT OUTPUT:
        ### STUDI KASUS:
        [Deskripsi situasi realistic dan menantang]
        
        ### PERTANYAAN:
        [Pertanyaan yang mengukur kompetensi secara spesifik]
        
        ### INDIKATOR YANG DIUKUR:
        - {indicators[0] if indicators else 'Relevansi jawaban dengan konteks'}
        - {indicators[1] if len(indicators) > 1 else 'Kedalaman analisis'}
        - {indicators[2] if len(indicators) > 2 else 'Aplikasi konsep'}
        """
        
        try:
            response = self.llm.invoke(prompt)
            return response.content
        except Exception as e:
            print(f"❌ Error generating question: {e}")
            return self._get_fallback_question(jabatan, level, kompetensi)

    def _get_level_key(self, level: str) -> str:
        """Convert level name to mapping key"""
        level_lower = level.lower()
        if 'pertama' in level_lower or '1' in level_lower:
            return "level_1"
        elif 'muda' in level_lower or '2' in level_lower:
            return "level_2"
        elif 'madya' in level_lower or '3' in level_lower:
            return "level_3"
        elif 'utama' in level_lower or '4' in level_lower:
            return "level_4"
        else:
            return "level_2"  # default

    def _get_fallback_question(self, jabatan: str, level: str, kompetensi: str) -> str:
        """Fallback question when LLM fails"""
        return f"""
        ### STUDI KASUS:
        Anda adalah {jabatan} level {level}. Dalam pelaksanaan tugas sehari-hari, Anda dihadapkan pada situasi yang membutuhkan penerapan kompetensi {kompetensi}.

        ### PERTANYAAN:
        Jelaskan bagaimana Anda akan menerapkan kompetensi {kompetensi} dalam menyelesaikan tantangan pekerjaan sebagai {jabatan} level {level}? Berikan contoh konkret dari pengalaman atau pendekatan yang akan Anda lakukan.

        ### INDIKATOR YANG DIUKUR:
        - Pemahaman konseptual tentang {kompetensi}
        - Kemampuan aplikasi dalam konteks nyata
        - Kedalaman analisis dan solusi
        """

    def assess_with_llm(self, nama: str, jabatan: str, jawaban: str, kompetensi: str, level_target: str) -> Dict[str, Any]:
        """Use LLM to assess answers based on mapping"""
        # Get relevant documents from vector DB
        query = f"{kompetensi} {jabatan} level {level_target} indikator perilaku"
        relevant_docs = self.vector_db.similarity_search(query, k=6)
        context = "\n".join([doc.page_content[:500] for doc in relevant_docs])  # Limit context length

        # Get indicators from mapping
        level_key = self._get_level_key(level_target)
        indicators = self.job_mapping[jabatan]["indikator_perilaku"].get(level_key, [])
        
        assessment_prompt = PromptTemplate(
            input_variables=["context", "nama", "jabatan", "jawaban", "kompetensi", "level_target", "indicators"],
            template="""
            STANDAR PENILAIAN KOMPETENSI:
            {context}

            INDIKATOR PERILAKU LEVEL {level_target}:
            {indicators}

            DATA PENILAIAN:
            - Nama: {nama}
            - Jabatan: {jabatan}
            - Kompetensi: {kompetensi} 
            - Level Target: {level_target}
            - Jawaban Peserta: {jawaban}

            TUGAS PENILAIAN:
            1. Beri skor 1-5 berdasarkan kesesuaian dengan indikator di atas
            2. Analisis DETAIL kesesuaian dengan setiap indikator
            3. Identifikasi kekuatan spesifik dalam jawaban
            4. Berikan rekomendasi pengembangan yang actionable
            5. Tentukan level pencapaian (1-4) berdasarkan skor

            KRITERIA SKOR:
            - 5: Sangat Baik (melebihi ekspektasi level)
            - 4: Baik (memenuhi semua indikator level)  
            - 3: Cukup (memenuhi sebagian besar indikator)
            - 2: Perlu Perbaikan (hanya memenuhi beberapa indikator)
            - 1: Tidak Memadai (tidak memenuhi indikator)

            FORMAT OUTPUT:
            ### HASIL PENILAIAN
            #### SKOR: [1-5]
            #### LEVEL PENCAPAIAN: [1-4]
            #### ANALISIS INDIKATOR:
            - [Indikator 1]: [Analisis kesesuaian dan evidence dari jawaban]
            - [Indikator 2]: [Analisis kesesuaian dan evidence dari jawaban]
            - [Indikator 3]: [Analisis kesesuaian dan evidence dari jawaban]
            #### KEKUATAN:
            - [Kekuatan 1 dengan contoh dari jawaban]
            - [Kekuatan 2 dengan contoh dari jawaban]
            #### AREA PERBAIKAN:
            - [Area 1 yang perlu dikembangkan]
            - [Area 2 yang perlu dikembangkan]
            #### REKOMENDASI PENGEMBANGAN:
            - [Rekomendasi 1 yang spesifik dan actionable]
            - [Rekomendasi 2 yang spesifik dan actionable]

            Gunakan Bahasa Indonesia profesional dan objektif.
            """
        )
        
        try:
            assessment_chain = LLMChain(llm=self.llm, prompt=assessment_prompt)
            
            result = assessment_chain.invoke({
                "context": context,
                "nama": nama,
                "jabatan": jabatan,
                "jawaban": jawaban, 
                "kompetensi": kompetensi,
                "level_target": level_target,
                "indicators": "\n".join([f"- {ind}" for ind in indicators])
            })
            
            return {
                "hasil": result['text'],
                "sumber": relevant_docs,
                "kompetensi": kompetensi,
                "level_target": level_target
            }
            
        except Exception as e:
            print(f"❌ Error in assessment: {e}")
            return {
                "hasil": "### HASIL PENILAIAN\n#### ERROR: Terjadi kesalahan dalam penilaian",
                "sumber": [],
                "kompetensi": kompetensi,
                "level_target": level_target
            }

    def generate_comprehensive_report(self, assessment_data: Dict[str, Any], results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive assessment report"""
        print(f"\n📊 GENERATING REPORT: {assessment_data['nama_pegawai']}")
        
        report = {
            'metadata': {
                'tanggal_assessment': datetime.now().isoformat(),
                'nama_pegawai': assessment_data['nama_pegawai'],
                'nip': assessment_data.get('nip', ''),
                'jabatan': assessment_data['jabatan'],
                'level_target': assessment_data['level_target'],
                'kompetensi_dinilai': assessment_data['kompetensi_dinilai']
            },
            'summary': {
                'total_kompetensi': len(results),
                'rata_rata_skor': 0,
                'status_kelayakan': 'BELUM_DITENTUKAN'
            },
            'detailed_results': results
        }
        
        # Calculate average score
        total_score = 0
        count = 0
        for kompetensi, result in results.items():
            hasil_text = result['hasil']
            # Simple score extraction (you might want to make this more robust)
            if 'SKOR:' in hasil_text:
                try:
                    score_line = [line for line in hasil_text.split('\n') if 'SKOR:' in line][0]
                    score = int(score_line.split('SKOR:')[1].strip().split()[0])
                    total_score += score
                    count += 1
                except:
                    continue
        
        if count > 0:
            report['summary']['rata_rata_skor'] = round(total_score / count, 2)
            # Determine eligibility
            avg_score = report['summary']['rata_rata_skor']
            if avg_score >= 4.0:
                report['summary']['status_kelayakan'] = 'KOMPETEN'
            elif avg_score >= 3.0:
                report['summary']['status_kelayakan'] = 'CUKUP_KOMPETEN'
            else:
                report['summary']['status_kelayakan'] = 'PERLU_PENGEMBANGAN'
        
        # Save report
        reports_dir = "../reports"
        os.makedirs(reports_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M')
        filename = f"assessment_{assessment_data['nama_pegawai'].replace(' ', '_')}_{timestamp}.json"
        report_file = os.path.join(reports_dir, filename)
        
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Report disimpan: {report_file}")
        return report

    def get_system_status(self) -> Dict[str, Any]:
        """Get system status information"""
        return {
            "vector_db_ready": self.vector_db is not None,
            "llm_ready": self.llm is not None,
            "job_mapping_loaded": len(self.job_mapping) > 0,
            "total_jobs": len(self.job_mapping),
            "available_jobs": list(self.job_mapping.keys()),
            "timestamp": datetime.now().isoformat()
        }
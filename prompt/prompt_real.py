# Template untuk penilaian kompetensi manajerial
#Prompt
from langchain.prompts import PromptTemplate



# Template ini akan diisi dengan dokumen dari retriever dan pertanyaan dari pengguna (query)
prompt_template = """
Gunakan potongan-potongan konteks berikut untuk menjawab pertanyaan pengguna.
Jawablah secara ringkas dan jelas dalam Bahasa Indonesia.
Jika Anda tidak tahu jawabannya berdasarkan konteks yang diberikan, katakan saja bahwa Anda tidak dapat menemukan informasinya, jangan mencoba mengarang jawaban.

KONTEKS:
{context}

PERTANYAAN:
{question}

JAWABAN YANG MEMBANTU:
"""

CUSTOM_PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)


# Template untuk query dasar RAG
BASIC_RAG_PROMPT = PromptTemplate(
    template="""
Gunakan potongan-potongan konteks berikut untuk menjawab pertanyaan pengguna.
Jawablah secara ringkas dan jelas dalam Bahasa Indonesia.
Jika Anda tidak tahu jawabannya berdasarkan konteks yang diberikan, katakan saja bahwa Anda tidak dapat menemukan informasinya, jangan mencoba mengarang jawaban.

KONTEKS:
{context}

PERTANYAAN:
{question}

JAWABAN YANG MEMBANTU:
""",
    input_variables=["context", "question"]
)
MANAGERIAL_ASSESSMENT_PROMPT = PromptTemplate(
    template="""
ANDA ADALAH PENILAI KOMPETENSI MANAJERIAL BERDASARKAN PERMENPAN RB NO. 38 TAHUN 2017
DAN STANDAR KOMPETENSI JABATAN (SKJ) YANG RELEVAN.

KONTEKS RESMI:
{context}

DATA TEST CASE:
- Nama: {nama}
- Jabatan: {jabatan}
- Kompetensi yang Dinilai: {kompetensi}
- Level Target: {level_target}
- Soal: {soal}
- Jawaban Peserta: {jawaban}

INSTRUKSI PENILAIAN:
1. Gunakan KONTEKS RESMI sebagai dasar penilaian (PermenPAN + SKJ).
2. Analisis jawaban peserta terhadap indikator perilaku dan level target.
3. Berikan skor 1-4:
   - 1: Tidak memenuhi
   - 2: Cukup
   - 3: Baik
   - 4: Sangat Baik
4. Berikan justifikasi yang terhubung dengan indikator perilaku.
5. Berikan rekomendasi pengembangan yang spesifik dan realistis.

FORMAT OUTPUT (WAJIB DIIKUTI, JANGAN MENAMBAH FORMAT LAIN):
SKOR: [angka 1-4]
JUSTIFIKASI: [penjelasan berdasarkan indikator perilaku dan konteks]
REKOMENDASI: [saran pengembangan kompetensi]

HASIL PENILAIAN:
""",
    input_variables=[
        "context",
        "nama",
        "jabatan",
        "kompetensi",
        "level_target",
        "soal",
        "jawaban",
    ],
)


# Template untuk evaluasi kompetensi teknis
TECHNICAL_COMPETENCY_PROMPT = PromptTemplate(
    template="""
ANDA ADALAH PENILAI KOMPETENSI TEKNIS BERDASARKAN PERMENPAN RB NO. 38 TAHUN 2017.

KONTEKS STANDAR KOMPETENSI TEKNIS:
{context}

DATA PENILAIAN:
- Nama: {nama}
- Jabatan: {jabatan}
- Bidang Teknis: {bidang_teknis}
- Jawaban/Karya: {jawaban}
- Level yang Dinilai: {level_target}

INSTRUKSI:
1. Evaluasi berdasarkan standar kompetensi teknis untuk bidang {bidang_teknis}
2. Berikan penilaian kualitatif
3. Identifikasi kekuatan dan kelemahan
4. Rekomendasikan pengembangan teknis

HASIL EVALUASI TEKNIS:
""",
    input_variables=["context", "nama", "jabatan", "bidang_teknis", "jawaban", "level_target"]
)

# Template untuk analisis gap kompetensi
COMPETENCY_GAP_ANALYSIS_PROMPT = PromptTemplate(
    template="""
ANDA ADALAH ANALIS KOMPETENSI ASN BERDASARKAN PERMENPAN RB NO. 38 TAHUN 2017.

KONTEKS STANDAR KOMPETENSI:
{context}

DATA PROFIL:
- Nama: {nama}
- Jabatan: {jabatan}
- Level Saat Ini: {level_sekarang}
- Level Target: {level_target}
- Hasil Assessment: {hasil_assessment}

INSTRUKSI ANALISIS:
1. Identifikasi gap kompetensi antara level sekarang dan target
2. Rekomendasikan program pengembangan spesifik
3. Saran timeline pengembangan
4. Prioritas kompetensi yang perlu ditingkatkan

FORMAT OUTPUT:
KOMPETENSI MEMADAI: [daftar kompetensi yang sudah memadai]
KOMPETENSI PERLU DITINGKATKAN: [daftar kompetensi yang perlu ditingkatkan]
PROGRAM PENGEMBANGAN: [rekomendasi program spesifik]
TIMELINE: [estimasi timeline pengembangan]

HASIL ANALISIS:
""",
    input_variables=["context", "nama", "jabatan", "level_sekarang", "level_target", "hasil_assessment"]
)

# === EXTRACT SKJ ===
EXTRACT_SKJ_PROMPT = PromptTemplate(
    template="""
Instruksi:
Anda adalah extractor yang memproses dokumen Standar Kompetensi Jabatan (SKJ).
Tugas: baca teks SKJ dan kembalikan **hanya** JSON valid (tidak ada teks penjelas lain).
Gunakan Bahasa Indonesia.

Format JSON yang WAJIB dikembalikan:
{
  "jabatan": "<nama jabatan>",
  "source_file": "<nama_file>",
  "kompetensi":[
    {
      "nama_kompetensi": "<nama kompetensi>",
      "definisi": "<ringkasan definisi (1-2 kalimat)>",
      "indikator_perilaku": [],
      "contoh_perilaku": [],
      "level_mapping": {
         "1": "",
         "2": "",
         "3": "",
         "4": ""
      }
    }
  ],
  "metadata": {
     "extracted_at": "",
     "extractor_version": "v1"
  }
}

Instruksi tambahan:
- Jika suatu field tidak ada di dokumen, isi dengan empty string atau empty list.
- Jangan mengarang indikator baru.
- Kembalikan output **ONLY JSON**.
""",
    input_variables=[]
)

# === GENERATE SOAL ===
CREATE_SOAL_SKJ_PROMPT = PromptTemplate(
    template="""
Instruksi:
Berdasarkan objek JSON SKJ, buat:
1) 5 soal multiple choice untuk tiap kompetensi.
2) 3 soal essay per kompetensi.

Format soal:
- id_soal
- tipe ("mcq" / "essay")
- soal
- pilihan (A-D, hanya untuk mcq)
- jawaban_benar (mcq)
- kunci_penilaian
- kompetensi_target
- level_target

Output dalam **JSON array**.
""",
    input_variables=["context"]
)
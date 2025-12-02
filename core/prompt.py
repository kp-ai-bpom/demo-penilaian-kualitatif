# prompt.py
from langchain.prompts import PromptTemplate

# Template untuk query dasar RAG
BASIC_RAG_PROMPT = PromptTemplate(
    template="""
Gunakan potongan-potongan konteks berikut untuk menjawab pertanyaan pengguna.
Jawablah secara ringkas dan jelas dalam Bahasa Indonesia.
Jika Anda tidak tahu jawabannya berdasarkan konteks yang diberikan,
katakan saja bahwa Anda tidak dapat menemukan informasinya,
jangan mengarang.

KONTEKS:
{context}

PERTANYAAN:
{question}

JAWABAN:
""",
    input_variables=["context", "question"],
)

# Prompt khusus penilaian kompetensi, DIISI KONTEX RAG (PermenPAN + SKJ)
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

# # === EXTRACT SKJ ===
# EXTRACT_SKJ_PROMPT = PromptTemplate(
#     template="""
# Instruksi:
# Anda adalah extractor yang memproses dokumen Standar Kompetensi Jabatan (SKJ).
# Tugas: baca teks SKJ dan kembalikan **hanya** JSON valid (tidak ada teks penjelas lain).
# Gunakan Bahasa Indonesia.

# Format JSON yang WAJIB dikembalikan:
# {{
#   "jabatan": "<nama jabatan>",
#   "kode_jabatan": "<kode jabatan>",
#   "unit_organisasi": "<unit organisasi>",
#   "ringkasan_tugas": "<ringkasan tugas jabatan>",
#   "kompetensi_manajerial": [
#     {{
#       "nama_kompetensi": "<nama kompetensi>",
#       "definisi": "<definisi kompetensi>",
#       "level_1": {{
#         "deskripsi": "<deskripsi level 1>",
#         "indikator_perilaku": ["<indikator1>", "<indikator2>"]
#       }},
#       "level_2": {{
#         "deskripsi": "<deskripsi level 2>",
#         "indikator_perilaku": ["<indikator1>", "<indikator2>"]
#       }},
#       "level_3": {{
#         "deskripsi": "<deskripsi level 3>",
#         "indikator_perilaku": ["<indikator1>", "<indikator2>"]
#       }},
#       "level_4": {{
#         "deskripsi": "<deskripsi level 4>",
#         "indikator_perilaku": ["<indikator1>", "<indikator2>"]
#       }}
#     }}
#   ],
#   "kompetensi_teknis": [
#     {{
#       "nama_kompetensi": "<nama kompetensi teknis>",
#       "definisi": "<definisi kompetensi teknis>",
#       "level_1": {{
#         "deskripsi": "<deskripsi level 1>",
#         "indikator_perilaku": ["<indikator1>", "<indikator2>"]
#       }},
#       "level_2": {{
#         "deskripsi": "<deskripsi level 2>",
#         "indikator_perilaku": ["<indikator1>", "<indikator2>"]
#       }},
#       "level_3": {{
#         "deskripsi": "<deskripsi level 3>",
#         "indikator_perilaku": ["<indikator1>", "<indikator2>"]
#       }},
#       "level_4": {{
#         "deskripsi": "<deskripsi level 4>",
#         "indikator_perilaku": ["<indikator1>", "<indikator2>"]
#       }}
#     }}
#   ],
#   "persyaratan_jabatan": {{
#     "pendidikan": "<persyaratan pendidikan>",
#     "pelatihan": "<persyaratan pelatihan>",
#     "pengalaman": "<persyaratan pengalaman>"
#   }},
#   "metadata": {{
#     "sumber_file": "<nama_file>",
#     "extracted_at": "<timestamp>",
#     "extractor_version": "v1"
#   }}
# }}

# Instruksi tambahan:
# - Jika suatu field tidak ada di dokumen, isi dengan empty string atau empty list/object.
# - Jangan mengarang indikator baru.
# - Kembalikan output **ONLY JSON**.
# - Pastikan format JSON valid.

# Teks SKJ yang akan diproses:
# {skj_text}
# """,
#     input_variables=["skj_text"]
# )
# # === GENERATE SOAL ===
# CREATE_SOAL_SKJ_PROMPT = PromptTemplate(
#     template="""
# Instruksi:
# Berdasarkan data SKJ berikut, buat soal assessment dalam format JSON array.

# DATA SKJ:
# {skj_data}

# INSTRUKSI PEMBUATAN SOAL:
# 1. Buat 3 soal multiple choice untuk setiap kompetensi manajerial
# 2. Buat 2 soal essay untuk setiap kompetensi manajerial  
# 3. Buat 2 soal multiple choice untuk setiap kompetensi teknis
# 4. Buat 1 soal essay untuk setiap kompetensi teknis

# FORMAT SOAL MCQ:
# {{
#   "id_soal": "MCQ_[kompetensi]_[nomor]",
#   "tipe": "mcq",
#   "soal": "<pertanyaan>",
#   "pilihan": {{
#     "A": "<pilihan A>",
#     "B": "<pilihan B>", 
#     "C": "<pilihan C>",
#     "D": "<pilihan D>"
#   }},
#   "jawaban_benar": "<A/B/C/D>",
#   "kunci_penilaian": "<penjelasan jawaban benar>",
#   "kompetensi_target": "<nama kompetensi>",
#   "level_target": "<level 1-4>",
#   "bobot": 1
# }}

# FORMAT SOAL ESSAY:
# {{
#   "id_soal": "ESSAY_[kompetensi]_[nomor]",
#   "tipe": "essay",
#   "soal": "<pertanyaan>",
#   "kunci_penilaian": "<kriteria penilaian essay>",
#   "kompetensi_target": "<nama kompetensi>", 
#   "level_target": "<level 1-4>",
#   "bobot": 2
# }}

# ATURAN:
# - Soal harus mengacu pada indikator perilaku di SKJ
# - Level target disesuaikan dengan kompleksitas kompetensi
# - Gunakan Bahasa Indonesia yang formal
# - Kembalikan **ONLY JSON ARRAY**

# OUTPUT:
# """,
#     input_variables=["skj_data"]
# )

# # Template untuk RAG dengan SKJ
# SKJ_RAG_PROMPT = PromptTemplate(
#     template="""
# Gunakan konteks berikut dari PERMENPAN dan SKJ terpilih untuk menjawab pertanyaan.

# KONTEKS PERMENPAN:
# {context_permenpan}

# KONTEKS SKJ TERPILIH:
# {context_skj}

# PERTANYAAN:
# {question}

# JAWABAN YANG MEMBANTU:
# """,
#     input_variables=["context_permenpan", "context_skj", "question"]
# )
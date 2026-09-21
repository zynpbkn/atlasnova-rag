__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import os
import shutil

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings



# --------------------------------------------------
# ENV
# --------------------------------------------------

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

if not OPENROUTER_API_KEY:
    raise ValueError(
        "OPENROUTER_API_KEY bulunamadı. .env dosyanı kontrol et."
    )


# --------------------------------------------------
# AYARLAR
# --------------------------------------------------

PDF_PATH = "data/AtlasNova_Calisan_El_Kitabi_Duzenli.pdf"
CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "atlasnova_hr"


# --------------------------------------------------
# 1. PDF OKU
# --------------------------------------------------

print("📄 PDF okunuyor...")

loader = PyPDFLoader(PDF_PATH)
documents = loader.load()

print(f"✅ {len(documents)} sayfa okundu.")


# --------------------------------------------------
# 2. METNİ PARÇALA
# --------------------------------------------------

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=900,
    chunk_overlap=150,
)

chunks = text_splitter.split_documents(documents)

print(f"✂️ {len(chunks)} chunk oluşturuldu.")


# --------------------------------------------------
# 3. METADATA
# --------------------------------------------------

for chunk in chunks:
    page = chunk.metadata.get("page", 0)

    chunk.metadata.update(
        {
            "page_number": page + 1,
            "company": "AtlasNova Teknoloji ve Danışmanlık Ltd. Şti.",
            "document_type": "employee_handbook",
        }
    )


# --------------------------------------------------
# 4. EMBEDDING
# --------------------------------------------------

print("🧠 Embedding modeli hazırlanıyor...")


# embeddings = OpenAIEmbeddings(
#    model="openai/text-embedding-3-small",
#    openai_api_key=OPENROUTER_API_KEY,
#   openai_api_base="https://openrouter.ai/api/v1",
# )

embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-2-preview",
    google_api_key=os.getenv("GEMINI_API_KEY")
)

# --------------------------------------------------
# 5. ESKİ CHROMA'YI TEMİZLE
# --------------------------------------------------

if os.path.exists(CHROMA_PATH):
    print("🗑️ Eski Chroma verisi temizleniyor...")
    shutil.rmtree(CHROMA_PATH)


# --------------------------------------------------
# 6. CHROMA'YA KAYDET
# --------------------------------------------------

print("💾 Chroma oluşturuluyor...")

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    collection_name=COLLECTION_NAME,
    persist_directory=CHROMA_PATH,
)

print("✅ Ingestion tamamlandı.")
print(f"📦 Chroma: {CHROMA_PATH}")
print(f"📚 Collection: {COLLECTION_NAME}")
print(f"🔢 Toplam chunk: {len(chunks)}")
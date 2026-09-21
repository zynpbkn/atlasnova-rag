# AtlasNova RAG - İnsan Kaynakları Asistanı 🏢

AtlasNova Teknoloji ve Danışmanlık Ltd. Şti. çalışan el kitabı (PDF) üzerinden soruları yanıtlayan; **LangChain**, **LangGraph**, **ChromaDB**, **Google Gemini Embeddings** ve **OpenRouter** tabanlı bir Retrieval-Augmented Generation (RAG) uygulamasıdır.

Hafıza (memory/checkpoint) desteği sayesinde çoklu konuşma turlarını (multi-turn chat) takip edebilir ve çalışanların kıdem/izin hesaplamalarını geçmiş konuşma bağlamını koruyarak yapabilir.

---

## 📁 Proje Yapısı

```text
atlasnova-rag/
│
├── data/
│   └── AtlasNova_Calisan_El_Kitabi_Duzenli.pdf   # İndekslenecek çalışan el kitabı
│
├── chroma_db/                                    # Vektör veritabanı (otomatik oluşturulur)
│
├── .env                                          # API anahtarları ve ortam değişkenleri
├── requirements.txt                              # Proje bağımlılıkları
├── ingest.py                                     # PDF'i işleyip ChromaDB'ye aktaran betik
└── rag.py                                        # LangGraph & Agent tabanlı terminal asistanı

🛠️ Teknolojiler ve Bileşenler
Paket Yöneticisi: uv

Orkestrasyon & Agent Framework: LangChain & LangGraph (InMemorySaver ile hafıza yönetimi)

Vektör Veritabanı: ChromaDB (langchain-chroma)

Embedding Modeli: Google Gemini (gemini-embedding-2-preview)

LLM / Chat Modeli: OpenRouter (openrouter/free)

SQLite Uyumluluğu: pysqlite3-binary

🚀 Kurulum ve Çalıştırma Adımları
1. Projeyi Başlatma ve Sanal Ortamı Hazırlama
Projeyi uv ile ilklendirin ve bağımlılıkları yükleyin:

Bash
uv init .
uv add -r requirements.txt
source .venv/bin/activate
2. Çevre Değişkenlerinin Hazırlanması (.env)
Projenin kök dizininde bir .env dosyası oluşturun ve gerekli API anahtarlarını ekleyin:

Kod snippet'i
OPENROUTER_API_KEY=your_openrouter_api_key_here
GEMINI_API_KEY=your_gemini_api_key_here

3. Vektör Veritabanının Oluşturulması (ingest.py)
PDF dokümanını parçalayıp (RecursiveCharacterTextSplitter), gemini-embedding-2-preview modeli ile embedding alarak chroma_db dizinine kaydetmek için:

Bash
python ingest.py

4. HR Asistanının Çalıştırılması (rag.py)
Terminal üzerinden etkileşimli İK asistanını başlatın:

Bash
python rag.py
💬 Örnek Çalışma ve Terminal Ekran Çıktısı
Plaintext
[train@LAPTOP-OEKGJ8TI atlasnova-rag]$ uv init .
Initialized project `atlasnova-rag` at `/home/train/atlasnova-rag`
[train@LAPTOP-OEKGJ8TI atlasnova-rag]$ uv add -r requirements.txt
[train@LAPTOP-OEKGJ8TI atlasnova-rag]$ source .venv/bin/activate
(atlasnova-rag) [train@LAPTOP-OEKGJ8TI atlasnova-rag]$ python ingest.py
📄 PDF okunuyor...
✅ 20 sayfa okundu.
✂️ 75 chunk oluşturuldu.
🧠 Embedding modeli hazırlanıyor...
🗑️ Eski Chroma verisi temizleniyor...
💾 Chroma oluşturuluyor...
✅ Ingestion tamamlandı.
📦 Chroma: ./chroma_db
📚 Collection: atlasnova_hr
🔢 Toplam chunk: 75

(atlasnova-rag) [train@LAPTOP-OEKGJ8TI atlasnova-rag]$ python rag.py

============================================================
🏢 ATLASNOVA HR ASISTANI
============================================================
Çıkmak için: exit / quit

👤 Siz: doğum izni kaç gün

🤖 AtlasNova HR:
Bilgi tabanında "doğum izni" konusunda iki farklı kategori bulunmaktadır:

**1. Babalık İzni (Çalışanın çocuğunun doğumu):**
AtlasNova'nın şirket içi standardı **5 iş günüdür**. Mevzuat veya bireysel sözleşme daha farklı/lehine bir düzenleme getiriyorsa ilgili hüküm uygulanır.

**2. Doğum / Analık İzni:**
Bilgi tabanında bu konu için **"Mevzuattaki izin süreleri + şirket ek uygulamaları"** olarak belirtilmiştir. Ancak bilgi tabanında analık izni için kesin bir gün sayısı verilmemiştir.

---

**Kaynak:** Çalışan El Kitabı, Sayfa 5 (Özel İzinler ve Aile Yaşamı bölümü)

Hangi doğum izninden bahsettiğinizi belirtirseniz (babalık mı, analık mı), daha ayrıntılı yardımcı olabilirim. Analık izni için mevzuat sürelerini de ekleyerek bilgi verebilirim — ancak bu, bilgi tabanında değil, mevzuat kaynaklarından gelen bir değerdir.

👤 Siz: babalık izni kaç gündür

🤖 AtlasNova HR:
AtlasNova’nın şirket içi standardı **5 iş günü** babalık izni olarak belirlenmiştir. Mevzuat veya bireysel sözleşme daha farklı/lehine bir düzenleme getiriyorsa ilgili hüküm uygulanır.

**Kaynak:** Çalışan El Kitabı, Sayfa 5 (Özel İzinler ve Aile Yaşamı bölümü)

👤 Siz: 7 yıldır AtlasNova'da çalışıyorum.

🤖 AtlasNova HR:
Anladım, 7 yıldır çalışıyorsunuz. Buna göre **yıllık izin hakkınız 20 iş günüdür**.

**Kaynak:** Çalışan El Kitabı, Sayfa 17

👤 Siz: Geçen yıldan 4 gün kullanılmamış iznim kaldı.
Bu yıl toplam kaç gün izin kullanabilirim?

🤖 AtlasNova HR:
7 yıldır çalıştığınıza göre yıllık izin hakkınız **20 iş günüdür**.  
Geçen yıldan kullanmamış **4 gün** izinizin de bu yıl kullanılması mümkün, çünkü şirket politikası kullanılmamış izin bakiyesinin sonraki döneme devre edilmesine izin verir.

**Toplam kullanabileceğiniz izin:**  
20 (yıllık hak) + 4 (devre edilen) = **24 iş günü**

**Kaynaklar:**  
- 7 yıl çalışan için 20 iş günü yıllık izin: Çalışan El Kitabı, Sayfa 17  
- Geçen yıldan kalan izinlerin bu yıl kullanılması: Çalışan El Kitabı, Sayfa 5 (Özel İzinler ve Aile Yaşamı bölümü)  
- Senaryo B – 7 yıllık çalışan, 4 gün devriyle toplam 24 gün: Çalışan El Kitabı, Sayfa 9 (İzin Hesaplama Örnekleri)

👤 Siz: exit

Görüşmek üzere.
(atlasnova-rag) [train@LAPTOP-OEKGJ8TI atlasnova-rag]$
# 📊 Cara Running Dashboard

Panduan lengkap untuk menjalankan Dashboard Visualisasi Data Al-Baleed Resort.

---

## 🎯 Prasyarat

Sebelum menjalankan dashboard, pastikan Anda sudah menginstall:

- **Python 3.8 atau lebih baru**
  - Download dari: https://python.org/downloads
  - ⚠️ **PENTING**: Saat install, centang opsi **"Add Python to PATH"**

---

## 🚀 Cara Menjalankan Dashboard

### **Opsi 1: Cara Otomatis (Recommended)**

#### Untuk Windows:
1. Double-click file `run_dashboard.bat`
2. Tunggu instalasi dependencies selesai
3. Browser akan otomatis membuka dashboard
4. Selesai! 🎉

#### Untuk Mac/Linux:
1. Buka Terminal di folder project
2. Jalankan perintah:
   ```bash
   chmod +x run_dashboard.sh
   ./run_dashboard.sh
   ```
3. Tunggu instalasi dependencies selesai
4. Browser akan otomatis membuka dashboard
5. Selesai! 🎉

---

### **Opsi 2: Cara Manual**

#### Langkah 1: Install Dependencies

Buka Terminal/Command Prompt di folder project, lalu jalankan:

```bash
pip install -r requirements.txt
```

Atau install satu per satu:

```bash
pip install streamlit pandas numpy plotly scikit-learn openpyxl wordcloud matplotlib seaborn statsmodels
```

#### Langkah 2: Jalankan Dashboard

```bash
streamlit run streamlit_dashboard.py
```

#### Langkah 3: Buka Dashboard

Dashboard akan otomatis membuka di browser. Jika tidak, buka secara manual:
- URL: http://localhost:8501

---

## 📦 Dependencies yang Dibutuhkan

Dashboard ini membutuhkan library berikut:

| Library | Versi | Fungsi |
|---------|-------|--------|
| `streamlit` | 1.28.0 | Framework dashboard |
| `pandas` | 2.1.0 | Analisis data |
| `numpy` | 1.24.3 | Komputasi numerik |
| `plotly` | 5.17.0 | Visualisasi interaktif |
| `scikit-learn` | 1.3.0 | Machine learning |
| `openpyxl` | 3.1.2 | Baca file Excel |
| `wordcloud` | 1.9.2 | Word cloud visualization |
| `matplotlib` | 3.7.2 | Plotting |
| `seaborn` | 0.12.2 | Statistical visualization |
| `statsmodels` | 0.14.0 | Statistical modeling |

---

## 🛠️ Troubleshooting

### Problem: "python is not recognized"

**Solusi:**
1. Install Python dari https://python.org/downloads
2. Saat install, **WAJIB centang** "Add Python to PATH"
3. Restart Terminal/Command Prompt
4. Test dengan `python --version`

---

### Problem: "Permission denied: run_dashboard.sh"

**Solusi (Mac/Linux):**
```bash
chmod +x run_dashboard.sh
./run_dashboard.sh
```

---

### Problem: "Port 8501 sudah dipakai"

**Solusi - Gunakan port lain:**
```bash
streamlit run streamlit_dashboard.py --server.port 8502
```

Kemudian buka: http://localhost:8502

---

### Problem: Error "ModuleNotFoundError"

**Solusi:**
```bash
pip install --upgrade -r requirements.txt
```

---

### Problem: Dashboard tidak otomatis membuka di browser

**Solusi:**
1. Lihat output di Terminal
2. Copy URL yang ditampilkan (biasanya http://localhost:8501)
3. Paste di browser Anda

---

## 💡 Tips Penggunaan Dashboard

### 1. Upload Dataset
- Klik sidebar di kiri atas
- Upload file Excel Anda
- Format yang didukung: `.xlsx`, `.xls`

### 2. Gunakan Filter
Filter yang tersedia di sidebar:
- **Date Range**: Pilih rentang tanggal
- **Trip Type**: Solo, Couple, Family, Friends, Business
- **Rating**: 1-5 stars
- **Season**: Spring, Summer, Fall, Winter

### 3. Eksplorasi 6 Tab Utama

| Tab | Deskripsi |
|-----|-----------|
| **Overview** | Ringkasan KPI dan metrics utama |
| **Descriptive Analytics** | Statistik deskriptif dan distribusi data |
| **Diagnostic Analytics** | Analisis причин dan korelasi |
| **Predictive Analytics** | Model machine learning dan prediksi |
| **Prescriptive Analytics** | Rekomendasi dan action items |
| **Raw Data** | Data mentah yang dapat difilter |

### 4. Interaksi dengan Chart
- **Hover**: Lihat detail data point
- **Zoom**: Klik dan drag pada chart
- **Pan**: Shift + drag untuk geser chart
- **Download**: Klik icon kamera untuk screenshot
- **Fullscreen**: Klik icon expand

---

## 🌐 Deploy Dashboard Online (Opsional)

Jika ingin membuat dashboard dapat diakses online:

1. Upload project ke GitHub
2. Buka https://share.streamlit.io
3. Login dengan GitHub
4. Pilih repository Anda
5. Deploy!

URL publik akan dibuat otomatis: `https://your-app.streamlit.app`

📖 **Detail lengkap**: Lihat file `DEPLOYMENT_GUIDE.md`

---

## ⚙️ Konfigurasi Advanced (Opsional)

### Ubah Port Default

Edit file `.streamlit/config.toml` atau jalankan:
```bash
streamlit run streamlit_dashboard.py --server.port 8080
```

### Ubah Theme

Edit `.streamlit/config.toml`:
```toml
[theme]
primaryColor = "#FF4B4B"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
textColor = "#262730"
```

### Disable Auto-Open Browser

```bash
streamlit run streamlit_dashboard.py --server.headless true
```

---

## 📞 Bantuan Lebih Lanjut

- **README**: Lihat `README.md` untuk overview project
- **Quick Start**: Lihat `QUICK_START.md` untuk panduan ringkas
- **Deployment**: Lihat `DEPLOYMENT_GUIDE.md` untuk deploy ke cloud
- **Package Info**: Lihat `DASHBOARD_PACKAGE_INFO.txt` untuk detail library

---

## ✅ Checklist Sebelum Running

- [ ] Python 3.8+ sudah terinstall
- [ ] Python sudah ditambahkan ke PATH
- [ ] Dependencies sudah terinstall (`pip install -r requirements.txt`)
- [ ] Terminal/Command Prompt dibuka di folder project yang benar
- [ ] Port 8501 tidak sedang digunakan aplikasi lain

---

## 🎓 Contoh Workflow

1. **Buka Terminal** di folder project
2. **Install dependencies** (sekali saja):
   ```bash
   pip install -r requirements.txt
   ```
3. **Jalankan dashboard**:
   ```bash
   streamlit run streamlit_dashboard.py
   ```
4. **Upload dataset** via sidebar
5. **Filter data** sesuai kebutuhan
6. **Explore tabs** untuk insight berbeda
7. **Download charts** untuk presentasi/laporan

---

## 🔄 Update Dashboard

Jika ada update pada code:

```bash
# Pull update terbaru
git pull

# Update dependencies jika ada perubahan
pip install --upgrade -r requirements.txt

# Restart dashboard
streamlit run streamlit_dashboard.py
```

---

**Happy Analyzing! 📊✨**

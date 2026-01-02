# 🚀 QUICK START GUIDE

## Untuk Pemula (3 Langkah!)

### Windows

1. **Install Python** (jika belum): https://python.org/downloads
2. **Double-click** file `run_dashboard.bat`
3. **Browser otomatis membuka** → Upload dataset Excel Anda!

### Mac/Linux

1. **Install Python** (jika belum): https://python.org/downloads
2. **Buka Terminal** di folder ini
3. **Run**:
   ```bash
   ./run_dashboard.sh
   ```
4. **Browser otomatis membuka** → Upload dataset Excel Anda!

---

## Troubleshooting Cepat

**"python is not recognized"**
→ Install Python dari https://python.org/downloads
   (Centang "Add Python to PATH" saat install!)

**"Permission denied: run_dashboard.sh"**
```bash
chmod +x run_dashboard.sh
./run_dashboard.sh
```

**Port sudah dipakai**
```bash
streamlit run streamlit_dashboard.py --server.port 8502
```

---

## Cara Pakai Dashboard

1. **Upload Dataset** via sidebar (kiri atas)
2. **Pilih Filters** yang Anda mau:
   - Date Range
   - Trip Type
   - Rating
   - Season
3. **Explore 6 Tabs**:
   - Overview (mulai dari sini!)
   - Descriptive Analytics
   - Diagnostic Analytics
   - Predictive Analytics (lihat ML model!)
   - Prescriptive Analytics (recommendations!)
   - Raw Data
4. **Semua chart interactive** - hover untuk detail!

---

## Deploy ke Cloud (Opsional)

Buat dashboard online yang bisa di-share via URL:

1. Upload file ke GitHub
2. Buka https://share.streamlit.io
3. Connect GitHub → Deploy!
4. Done! Dapat URL public: `https://your-app.streamlit.app`

Lihat **DEPLOYMENT_GUIDE.md** untuk detail lengkap.

---

## Tips

💡 **Untuk presentasi**: Demo filter real-time di tab Overview dulu, baru masuk ke ML insights di tab Predictive!

💡 **Untuk laporan**: Screenshot key charts dengan klik kanan pada chart.

💡 **Untuk portfolio**: Deploy ke cloud dan masukkan URL di CV/LinkedIn Anda!

---

**Butuh bantuan?** Baca README.md atau DEPLOYMENT_GUIDE.md

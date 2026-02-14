# Aerial Object Detection for Autonomous UAV Systems (YOLOv8) - TEKNOFEST

This repository hosts a comprehensive computer vision and dashboard solution for autonomous UAV systems, developed for the TEKNOFEST competition. The system integrates real-time object detection with an advanced telemetry-aware dashboard.

## 🚀 Key Features
- **Real-Time Detection:** Optimized YOLOv8 implementation for aerial vehicle and person detection.
- **Advanced Dashboard:** High-performance PyQt5-based HUD featuring video streaming, telemetry data, and real-time metrics.
- **Data Pipeline:** Specialized scripts for UAVDT dataset conversion and validation against competition standards.
- **Extensible Architecture:** Modular code structure for seamless integration of additional sensors and flight data.

## 🤝 Contribution Guidelines
We follow a standard collaborative workflow to ensure code quality and stability:
1. **Branching:** Please create feature-specific branches for any updates.
2. **Pull Requests:** Submit a PR for any changes. All code is subject to review before being merged into the main branch.

## 🛠️ Installation
```bash
git clone https://github.com/Egemennayhan/Aerial-Object-Detection-for-Autonomous-UAV-Systems-YOLOv8-.git
cd Aerial-Object-Detection-for-Autonomous-UAV-Systems-YOLOv8-
pip install -r requirements.txt
```

---

# Otonom İHA Sistemleri için Hava Nesnesi Tespiti (YOLOv8) - TEKNOFEST

Selam ekip! Bu proje, TEKNOFEST için geliştirdiğimiz otonom İHA nesne tespiti ve gelişmiş dashboard sistemini içermektedir.

## 🤝 Birlikte Çalışma Kültürümüz
Projemizi düzenli ve güvenli bir şekilde ilerletmek adına şu yöntemi izliyoruz:

1. **Geliştirme:** Yeni bir özellik veya hata düzeltmesi üzerinde çalışırken yeni bir dal (branch) açmamız ana kodun kararlılığını korur.
2. **Gözden Geçirme:** Çalışmanızı tamamladığınızda bir 'Pull Request' (Değişiklik İsteği) oluşturarak hepimizin incelemesine sunabilirsiniz.
3. **Onay ve Birleştirme:** Değişiklikleri birlikte değerlendirdikten sonra ana projeye dahil ederek yolumuza devam edebiliriz. Karar merci olarak tüm PR'lar Ege (Egemennayhan) onayından geçecektir.

## 🛠️ Kurulum ve Veri Seti
* **Kurulum:** `pip install -r requirements.txt` komutu ile gerekli kütüphaneleri yükleyebilirsiniz.
* **Veri Seti:** 35GB'lık veri seti GitHub'da bulunmamaktadır. Lütfen ekip içinde paylaşılan bağlantıdan indirip `data/` klasörüne yerleştirin.

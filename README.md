# VLM Hallucination Detection (Object / Color / Quantity)

본 프로젝트는 Vision-Language Model이 생성한 캡션에서  
**객체(Object)**, **색상(Color)**, **수량(Quantity)** 단위로 환각(Hallucination)을 탐지하고  
각 토큰을 Match / Uncertain / Hallucination으로 하이라이트하는 시스템입니다.

---

## 📌 Pipeline

<p align="center">
  <img src="figures/pipeline.png" width="1000">
</p>

<p align="center"><em>
Figure 1. 전체 파이프라인 (BLIP2 캡션 생성 → spaCy 파싱 → GroundingDINO 객체 검출 → 색상·수량 속성 검증 → 토큰 하이라이트)
</em></p>

---

## ⚙️ Requirements (CPU 환경 기준)

본 실험은 **CPU 환경**에서 수행되었습니다.

### Python Version
- Python 3.9+

### Install

```bash
# PyTorch (CPU-only)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# BLIP2 (transformers)
pip install transformers==4.36.0

# GroundingDINO & open_clip
pip install git+https://github.com/IDEA-Research/GroundingDINO.git
pip install git+https://github.com/mlfoundations/open_clip.git

# spaCy
pip install spacy
python -m spacy download en_core_web_sm

# Utils
pip install numpy pandas pillow opencv-python matplotlib tqdm scikit-learn

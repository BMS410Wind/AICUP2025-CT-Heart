<div align="center">
  <img src="https://github.com/BMS410Wind/AICUP2025-CT-Heart/blob/main/heart.png" width="50%" />

  <br />

  <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" />

  <br />
  <hr />
</div>

## 🌟 Project Overview

本專案是針對 **AI CUP 2025 秋季賽 — 電腦斷層心臟肌肉影像分割競賽 (II)** 的高效解決方案。
使用 **"Vibe Coding"** 開發 **Multi-Agent 協作架構** ，完成醫學影像處理的半自動化流程。

---

## 🏗️ Multi-Agent Workflow

系統由四個專業 AI Agents 組成：


### 🤖 代理人職責說明
1.  **Preprocessing Agent**: 負責將原始CT影像(.nii.gz格式)轉換為模型可處理的標準化數據格式,確保數據品質和一致性。
2.  **Segmentation Agent**: 使用3D U-Net模型對心臟CT影像進行精準分割,識別心臟肌肉(Segment_1)、主動脈瓣膜(Segment_2)和鈣化區域(Segment_3)。
3.  **Post-Correction Agent**: 優化分割結果的品質,修正錯誤預測,並將輸出格式化為競賽要求的.nii.gz檔案。
4.  **Reporting Agent**: 生成綜合評估報告,確保提交檔案符合競賽格式。

---

## 📊 Benchmarks

模型在驗證集 (Validation Set) 上的數據：

| 🏥 Anatomical Structure | Metric (Dice) | Status |
| :--- | :---: | :--- |
| **Myocardium (心肌)** | `0.9051` | <img src="https://img.shields.io/badge/-Optimal-success?style=flat-square" /> |
| **Aortic Valve (主動脈瓣)** | `0.7530` | <img src="https://img.shields.io/badge/-Fine--tuning-important?style=flat-square" /> |
| **Calcification (鈣化)** | `In Progress` | <img src="https://img.shields.io/badge/-Under%20Dev-lightgrey?style=flat-square" /> |

---

## 🐳 Deployment & Usage

### Dockerized Environment
正在建置中

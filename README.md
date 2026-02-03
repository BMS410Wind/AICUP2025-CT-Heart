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
我們首創將 **"Vibe Coding"** 開發哲學與 **Multi-Agent 協作架構** 結合，大幅提升了醫學影像處理的自動化程度與精確度。

---

## 🏗️ Multi-Agent Workflow

系統由四個專業 AI Agents 組成：


### 🤖 代理人職責說明
1.  **Preprocessing Agent**: 自動解析 NIfTI 標籤，執行 Resampling 與 Windowing (窗寬窗位) 調整。
2.  **Segmentation Agent**: 驅動核心 **nnU-Net V2**，針對心肌與瓣膜進行 3D 體素級推理。
3.  **Post-Correction Agent**: 執行 3D 連通域分析與空洞填充，確保解剖學結構完整。
4.  **Reporting Agent**: 自動檢核輸出格式，生成符合競賽要求的提交檔。

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

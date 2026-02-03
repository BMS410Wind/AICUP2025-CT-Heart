<div align="center">
  <img src="https://capsule-render.vercel.app/render?type=waving&color=gradient&height=280&section=header&text=CT%20Heart%20Segmentation&fontSize=70&animation=fadeIn&fontAlignY=38&desc=AI%20CUP%202025%20|%20Vibe%20Coding%20Solution&descSize=25&descAlignY=55" width="100%" />

  <br />

  <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" />
  <img src="https://img.shields.io/badge/nnU--Net-V2-008000?style=for-the-badge&logo=nvidia&logoColor=white" />
  <img src="https://img.shields.io/badge/Docker-Enabled-2496ED?style=for-the-badge&logo=docker&logoColor=white" />

  <br />
  <hr />
</div>

## 🌟 Project Overview

本專案是針對 **AI CUP 2025 秋季賽 — 電腦斷層心臟肌肉影像分割競賽 (II)** 的高效解決方案。
我們首創將 **"Vibe Coding"** 開發哲學與 **Multi-Agent 協作架構** 結合，大幅提升了醫學影像處理的自動化程度與精確度。

> [!TIP]
> **Vibe Coding:** 核心開發邏輯由 AI 代理人輔助生成，專注於高效迭代與跨模組協同，解決醫學影像中複雜的類別不平衡問題。

---

## 🏗️ Multi-Agent Workflow

系統由四個專業 AI Agents 組成，模擬專業醫師與放射科人員的協作流程：



### 🤖 代理人職責說明
1.  **Preprocessing Agent**: 自動解析 NIfTI 標籤，執行 Resampling 與 Windowing (窗寬窗位) 調整。
2.  **Segmentation Agent**: 驅動核心 **nnU-Net V2**，針對心肌與瓣膜進行 3D 體素級推理。
3.  **Post-Correction Agent**: 執行 3D 連通域分析與空洞填充，確保解剖學結構完整。
4.  **Reporting Agent**: 自動檢核輸出格式，生成符合競賽要求的提交檔。

---

## 📊 Benchmarks

模型在驗證集 (Validation Set) 上的最新數據：

| 🏥 Anatomical Structure | Metric (Dice) | Status |
| :--- | :---: | :--- |
| **Myocardium (心肌)** | `0.9051` | <img src="https://img.shields.io/badge/-Optimal-success?style=flat-square" /> |
| **Aortic Valve (主動脈瓣)** | `0.7530` | <img src="https://img.shields.io/badge/-Fine--tuning-important?style=flat-square" /> |
| **Calcification (鈣化)** | `In Progress` | <img src="https://img.shields.io/badge/-Under%20Dev-lightgrey?style=flat-square" /> |

---

## 🐳 Deployment & Usage

### Dockerized Environment
為了避免醫學影像庫 (CUDA/SimpleITK) 環境衝突，強烈建議使用 Docker：

```bash
# 1. 快速啟動環境
docker-compose up -d --build

# 2. 執行端到端分割流程 (All-in-one Agent Flow)
docker exec -it aicup_container python main.py --mode run_all

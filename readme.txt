<div align="center">
  <img src="https://capsule-render.vercel.app/render?type=waving&color=auto&height=250&section=header&text=AICUP%202025&fontSize=80&animation=fadeIn&fontAlignY=35" width="100%" />

  # 🫀 CT Heart Segmentation Pipeline
  ### *The "Vibe Coding" Approach with Multi-Agent Systems*

  <p align="center">
    <a href="#-project-overview">Overview</a> •
    <a href="#-system-architecture">Architecture</a> •
    <a href="#-experimental-results">Results</a> •
    <a href="#-quick-start">Quick Start</a> •
    <a href="#-docker-deployment">Docker</a>
  </p>

  [![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
  [![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org/)
  [![nnU-Net](https://img.shields.io/badge/nnU--Net-V2-008000?style=flat-square&logo=nvidia&logoColor=white)](https://github.com/MIC-DKFZ/nnUNet)
  [![License](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](https://opensource.org/licenses/MIT)

  <br/>
</div>

## 🌟 Project Overview

本專案專注於 **AI CUP 2025 秋季賽 — 電腦斷層心臟肌肉影像分割競賽 (II)**。
透過 **Multi-Agent 協作架構** 與 **nnU-Net** 深度學習框架，實現對心臟 CT 影像中微小結構的高精度自動分割。

> [!IMPORTANT]
> **本專案特色：** 結合了 **Vibe Coding** 的開發哲學，利用 AI 代理人自動化處理從預處理到報告生成的繁瑣工作流。

---

## 🏗️ System Architecture

我們將醫學分割任務拆解為四個核心代理人，形成一個閉環的 **Agentic Workflow**：



| 🤖 Agent | 職責與描述 |
| :--- | :--- |
| **Preprocessing** | 執行 `NIfTI` 影像標準化、Windowing 調整及 Resampling。 |
| **Segmentation** | 核心 `nnU-Net V2` 模型，執行 3D 卷積神經網絡推論。 |
| **Post-Correction** | 形態學優化、邊界平滑及雜訊過濾。 |
| **Reporter** | 自動化品質評估 (Dice/IoU) 與提交檔封裝。 |

---

## 📊 Experimental Results

目前模型在驗證集上的量化指標：

| Label | Structure | Dice Score | Status |
| :---: | :--- | :---: | :--- |
| 01 | **Myocardium (心肌)** | **0.9051** | 🟢 Optimal |
| 02 | **Aortic Valve (主動脈瓣)** | **0.7530** | 🟡 Improving |
| 03 | **Calcification (鈣化)** | **0.0000** | 🔴 In Progress |

---

## 🚀 Quick Start

### 🐍 Local Installation
```bash
# Clone
git clone [https://github.com/BMS410Wind/AICUP2025-CT-Heart.git](https://github.com/BMS410Wind/AICUP2025-CT-Heart.git) && cd AICUP2025-CT-Heart

# Setup environment
pip install -r requirements.txt

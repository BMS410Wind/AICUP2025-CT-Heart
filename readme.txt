# AI CUP 2025: CT Heart Segmentation with Vibe Coding

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)
![nnU-Net](https://img.shields.io/badge/nnU--Net-V2-green)
![Status](https://img.shields.io/badge/Status-Development-yellow)

> **AI CUP 2025 秋季賽 — 電腦斷層心臟肌肉影像分割競賽 (II)**
> An automated medical segmentation pipeline powered by **Multi-Agent Systems** and **nnU-Net**.

## 📖 專案簡介 (Introduction)

本專案是針對 2025 AI CUP 心臟 CT 影像分割競賽的解決方案。我們採用 **"Vibe Coding"** 方法論，設計了一套由 AI 驅動的**多代理人協作系統 (Multi-Agent System)**，將繁瑣的醫學影像處理流程自動化，以精確分割以下目標：

1.  **全心臟肌肉 (Myocardium)** - (Class 1)
2.  **主動脈瓣膜 (Aortic Valve)** - (Class 2)
3.  **鈣化區域 (Calcification)** - (Class 3)

## 🏗️ 系統架構 (System Architecture)

本系統打破傳統單一腳本的限制，將任務拆解為四個專門化的 Agents：

```mermaid
graph TD
    Input[CT Images .nii.gz] --> Pre[🤖 Preprocessing Agent]
    Pre --> Seg[🧠 Segmentation Agent]
    Seg --> Post[🔧 Postprocessing Agent]
    Post --> Report[📊 Reporting Agent]
    Report --> Output[Submission File]
    
    Database[(📚 RAG Knowledge Base)] -.-> Pre
    Database -.-> Seg
    Database -.-> Post

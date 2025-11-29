*本程式最終在AI CUP 2025秋季賽－電腦斷層心臟肌肉影像分割競賽 I－心臟肌肉影像分割 取得0.7的socre

如果要訓練&測試資料 可去AICUP拿，如果拿不到可跟我拿(email:windspeak122@gmail.com)，因為資料太大就不上傳

code:
1.Trainall 可用於訓練nnUnet模型 需要去AICUP下載資料集 並修改資料儲存位址 才可訓練
2.predall 可用於預測 並產生繳交壓縮檔 其產生檔案可直接上傳AICUP

data_split: 內涵json檔案，用於區分訓練與驗證資料

experiments: 訓練結果將存成log 與 jpg供使用者調整訓練策略，同時記錄最佳權重

predictions: 產生預測結果並壓縮成zip檔案供使用者上傳

data_diagnosis: 檢查train/val類別、每個類別的像素占比、建議可行方案

requirements: 環境需求

本操作環境在 conda、windows、本地端、monai:1.5.1 python:3.9 cuda version:13.0、3080GPU10G、i7 12700K CPU、48G RAM


P.S. 報告內容附在PDF

權重太大就不放

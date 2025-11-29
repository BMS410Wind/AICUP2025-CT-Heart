code:
1.Trainall 可用於訓練nnUnet模型 需要去AICUP下載資料集 並修改資料儲存位址 才可訓練
2.predall 可用於預測 並產生繳交壓縮檔 其產生檔案可直接上傳AICUP
-------------
data_split: 涵json檔案，用於區分訓練與驗證資料
-------------
experiments: 訓練結果將存成log與jpg供使用者調整訓練策略，同時記錄最佳權重
-------------
predictions: 產生預測結果並壓縮成zip檔案(由predall產生)
-------------
data_diagnosis: 檢查train/val類別、每個類別的像素占比、建議可行方案
-------------
requirements: 環境需求
-------------
P.S. 操作環境: windows、local、monai:1.5.1 python:3.9 cuda version:13.0、3080GPU10G、i7 12700K CPU、48G RAM


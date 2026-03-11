# Parkinson's Freezing of Gait Prediction - 深度時序卷積優化方案

本專案參加 Kaggle 競賽 [「Parkinson's Freezing of Gait Prediction」](https://www.kaggle.com/competitions/tlvmc-parkinsons-freezing-gait-prediction)。本文件詳細紀錄如何將原始的線性基準模型（Baseline）改良為具備時序感知能力的深度學習架構，以精準預測帕金森氏症患者的凍結步態（FoG）現象。

## 為什麼需要改良 Baseline？

原始基準模型採用 **Rolling Window MLP (多層感知器)**，在處理生物感測信號時存在以下技術瓶頸：

1. **空間特徵缺失**：Baseline 將 32 幀感測數據直接拉平（Flatten）輸入線性層，完全忽略了三軸加速度計（AccV, AccML, AccAP）之間的空間耦合關係。
2. **時序規律流失**：線性層難以捕捉加速度信號中的波形特徵，尤其是凍結步態常伴隨的 3-8Hz 高頻微顫（Tremor）特徵。
3. **特徵工程單薄**：原始方案僅依賴原始數值，缺乏對信號變化率與能量分佈的描述，難以應對不同實驗場景（defog 與 tdcsfog）的數值分佈差異。

---

## 模型優化策略：從線性到殘差卷積 (Res-CNN)

本專案核心改動為引入 **1D-CNN (一維卷積神經網路)** 與 **殘差連結 (Residual Connection)**，使模型具備自動提取病患步態特徵的能力。

### 1. 一維卷積特徵提取 (1D-Convolutional Blocks)
捨棄 Baseline 的拉平做法，改用 `nn.Conv1d` 在時間軸上滑動掃描：
* **局部規律識別**：卷積核能自動識別加速度信號中的微小模式變化（如步幅縮短或特定頻率震顫）。
* **多通道融合處理**：透過維度轉換，模型能同時分析三軸加速度的交互影響，捕捉步態在三維空間中的協同特徵。


### 2. 標籤權重與遮罩策略 (Label Masking)
針對實驗數據中包含大量非運動片段的問題，我們優化了損失函數的權重控制：
* **有效性加權**：模型在計算損失（Loss）時，會嚴格過濾 `Valid` 與 `Task` 標籤。
* **抗噪訓練**：確保模型僅在真實的運動障礙測試區間進行權重更新，有效避免隨機背景噪聲干擾模型收斂。

---

## 數據處理與驗證機制

### 1. 類別平衡與驗證策略
* **類別平衡優化**：凍結步態（FoG）屬於極稀有事件。我們透過 **Stratified Group K-Fold** 進行分層劃分，確保每一折（Fold）中的 `StartHesitation`、`Turn`、`Walking` 事件比例一致。

### 2. 性能優化技術
* **多來源特徵對齊**：針對 `defog`（居家環境）與 `tdcsfog`（實驗室環境）數據進行標準化處理，提升模型跨數據集的表現。

---

# Training-Unit7：基於 CNN 的道路與障礙物影像分類 / CNN-Based Road and Obstacle Image Classification

## 1. 任務概述 (Introduction)
本任務需使用 CNN 模型（例如：VGG16、ResNet18、ResNet50）完成影像分類任務，並使用 Weights & Biases (wandb) 進行訓練監控。
This assignment requires using CNN models (e.g., VGG16, ResNet18, ResNet50) to complete an image classification task, with Weights & Biases (wandb) utilized for training monitoring and logging.

* **基本任務（二分類） / Basic Task (Binary Classification)**：
* **進階任務 / Advanced Task**：
### 學習目標 / Learning Objectives

- 理解 VGG 與 ResNet 架構差異（參數量、訓練速度、效果）
  Understand the architectural differences between VGG and ResNet (parameter count, training speed, performance)
- 熟悉圖像分類任務的訓練流程（loss function、optimizer、early stopping）
  Familiarize yourself with the image classification training pipeline (loss function, optimizer, early stopping)
- 使用 [Weights & Biases (wandb)](https://wandb.ai/) 可視化訓練過程與超參數實驗結果
  Use [Weights & Biases (wandb)](https://wandb.ai/) to visualize the training process and hyperparameter experiment results


---

## 2. 資料集準備 (Dataset Preparation)
本次任務的影像數據由實驗室統一提供，**每位同學分發到的圖片數據皆不相同**，請看群組訊息檔案位置，勿互相拷貝資料。
The image data for this task will be provided by the lab. **Each student will receive different image data.** Please check the group message file location and do not copy the data from each other.

### 分類標籤定義 / Classification Label Definition
模型需要辨識圖片的好壞，好的圖片須包含：
The model needs to distinguish between good and bad images; a good image must include:

* **道路 (Road)**：清楚且看起來正常的道路。 / Clear and normal-looking roads.
* **落石 (Rockfall) / 泥土 (Mud)**：在道路上的落石或泥土。 / Rockfalls or mud on the road.

### 驗證標準（人工測試集）/ Validation Criteria (Manual Test Set)
* 每位同學須在模型訓練前，**手動挑選 100 張圖片**作為訓練與測試好壞的依據。
  Each student must **manually select 100 images** before model training to serve as the baseline for evaluating model performance.
* 這 100 張圖片將作為評估模型好壞的最終依據。
  These 100 images will be used as the golden standard for the final assessment of the model.

---

## 3. 模型架構參考 / Model Architecture Reference
在實作時請運用以下的模型：
Please utilize the following models in your implementation:

| 模型名稱 / Model | 參數數量 / Parameters | 特點 / Features |
| :--- | :--- | :--- |
| **VGG16** | 138M | 傳統大型卷積網路 / Classic large convolutional network |
| **ResNet18** | 11M | 較小但引入殘差模組 / Smaller with residual blocks |
| **ResNet50** | 25M | 更深更準確 / Deeper and more accurate |

---

## 4. 作業繳交規範 / Assignment Submission Guidelines

### 必須完成的項目 / Required Tasks
1. 使用三種模型（VGG16、ResNet18、ResNet50）訓練資料集，並記錄 loss 與 accuracy 曲線。
   Train the dataset using three models (VGG16, ResNet18, ResNet50) and record the loss and accuracy curves.
2. 至少嘗試兩種超參數設定（例如不同的 learning rate 或 batch size），請在訓練主程式碼中手動修改並記錄結果。
   Try at least two hyperparameter configurations (e.g., different learning rates or batch sizes), modify them manually in the main training script, and record the results.
3. 全程使用 wandb 記錄訓練過程。
   Use wandb throughout the training process to record and log logs.
4. 提交一份報告總結觀察（含圖表與文字說明）。
   Submit a report summarizing your observations (including charts, screenshots, and textual explanations).

### 📂 繳交內容 / Submission Contents
請提交以下檔案 / Please submit the following files:
* **程式碼** / Source Code
* **訓練結果報告**（含 wandb 圖表截圖與簡要說明）/ Training result report (including wandb chart screenshots and brief explanations)
* **你的 wandb 專案公開連結** / Your public wandb project link

### Overleaf 報告撰寫要求 / Report Requirements in Overleaf
* 本任務要求撰寫一份完整的技術報告，且**必須使用 Overleaf (LaTeX) 進行撰寫**。
  This assignment requires writing a comprehensive technical report, which **must be authored using Overleaf (LaTeX)**.
* 報告中必須包含清晰的章節標籤（Label）以便內文引用，並包含實驗結果表格。
  The report must include clear section labels (`\label{...}`) for in-text cross-referencing, along with tables summarizing experimental results.
* **報告必要章節 / Required Sections:**
  * **模型架構 (Methodology & Model Architecture)**：說明所選用的 CNN 架構與超參數設定。 / Describe the chosen CNN architectures and hyperparameter configurations.
  * **實驗結果與分析 (Experimental Results & Analysis)**
  * **結論與心得 (Conclusion & Discussion)**
  * **如何使用 AI 幫忙 (AI Tools Utilization)**


## ☁️ GitHub 繳交方式 / GitHub Submission Method

請每位同學**自行建立一個 GitHub Repository**，將本次作業所有內容上傳，並在報告中附上你的 GitHub Repo 連結。
Each student should **create their own GitHub Repository**, upload all assignment content, and include the GitHub Repo link in the report.

### ✅ Repository 命名建議 / Naming Convention

請依照以下格式命名你的個人作業 Repo / Please name your personal assignment repo in the following format：

```
training-Unit7-{學號}
```

例如：

```
training-Unit7-613410112
```

### 📌 注意事項 / Notes

- 請確認你的 GitHub Repo 為 **公開狀態** 或提供我們存取權限。
  Please ensure your GitHub Repo is **public** or grant us access.
- 請勿上傳大型模型檔案（例如 .pt 檔）
  Do not upload large model files (e.g., .pt files).
- 報告內容需清楚呈現訓練結果與分析（可含 loss/accuracy 圖表、wandb 截圖）
  The report must clearly present training results and analysis (may include loss/accuracy charts and wandb screenshots).

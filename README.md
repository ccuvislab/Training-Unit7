# Training-Unit7：基於 CNN 的道路與障礙物影像分類 / CNN-Based Road and Obstacle Image Classification

## 1. 任務概述 / Introduction
本任務需使用 CNN 模型（例如：VGG16、ResNet18、ResNet50）完成影像分類任務，並使用 Weights & Biases (wandb) 進行訓練監控。作業分為「基本任務」與「進階任務」兩部分。
This assignment requires using CNN models (e.g., VGG16, ResNet18, ResNet50) to complete an image classification task, with Weights & Biases (wandb) utilized for training monitoring and logging.The assignment consists of a basic task and an advanced task.

* **基本任務（二分類） / Basic Task (Binary Classification)**：
使用模型判斷每張分配到的圖片屬於：

Use a model to determine whether each assigned image belongs to one of the following classes:

- **好影像（Good）**
- **壞影像（Bad）**

「好」與「壞」的具體判定方式及區分數值由每位同學自行制定。例如，可先建立一套影像品質評分方式，並規定總分高於某個數值時為好影像，低於該數值時為壞影像。

Each student must define the criteria and numerical threshold used to distinguish Good images from Bad images. For example, students may establish an image-quality scoring system and define images with scores above a selected threshold as Good and those below the threshold as Bad.

同學制定的規則必須符合以下要求：

The student-defined criteria must satisfy the following requirements:

1. 判定規則與數值門檻必須在標註前明確訂定。  
   The decision rules and numerical threshold must be determined before annotation begins.
2. 所有圖片必須使用相同標準，不可依模型預測結果任意修改規則。  
   The same criteria must be applied to all images and must not be modified based on model predictions.
3. 判定方式必須可以被他人理解與重現，不可只以個人直覺作為依據。  
   The criteria must be understandable and reproducible by others rather than based only on personal intuition.
4. 報告中必須列出完整判定規則、計分方式及好／壞區分門檻。  
   The report must clearly describe the complete decision rules, scoring method, and Good/Bad threshold.
5. 報告中至少提供 3 張好影像與 3 張壞影像作為標註範例。  
   The report must include at least three Good images and three Bad images as annotation examples.

可參考道路是否清楚、落石或泥土是否可辨識、影像是否模糊、曝光是否正常及重要區域是否被遮擋等因素制定評分規則，但最終仍只需產生 **Good** 與 **Bad** 兩種標籤。

Possible factors include road visibility, whether rockfalls or mud can be recognized, image blur, exposure quality, and occlusion of important regions. However, the final annotation must contain only the two labels **Good** and **Bad**.

* **進階任務 / Advanced Task**：
學生需保留模型對每張圖片輸出的 logits、Softmax 機率、Sigmoid 分數或其他連續信心分數，並使用論文常見的評估指標，對模型表現進行量化評分與分析。若某項指標需要離散預測結果，學生可以設定 confidence threshold 將連續分數轉換成預測類別。

Students must retain the logits, Softmax probabilities, Sigmoid scores, or other continuous confidence scores produced by the model for each image. These scores must then be evaluated using metrics commonly reported in papers. When a metric requires discrete predictions, students may apply a confidence threshold to convert continuous scores into predicted classes. 

報告中至少需要計算並分析下列五項評估項目：

The report must calculate and analyze at least the following five evaluation items:

| 評估項目 / Metric | 定義與代表意義 / Definition and Interpretation | 報告分析重點 / Required Analysis |
| :--- | :--- | :--- |
| **Accuracy** | 在選定 confidence threshold 後，模型預測正確的樣本比例。 / The proportion of samples predicted correctly after applying the selected confidence threshold. | 說明整體正確率，並討論類別不平衡時 Accuracy 是否可能高估模型表現。 / Explain the overall correctness and discuss whether Accuracy may overestimate performance when the classes are imbalanced. |
| **Precision** | 在模型判斷為目標類別的樣本中，實際屬於該類別的比例，公式為 `TP / (TP + FP)`。 / The proportion of samples predicted as the target class that actually belong to that class, calculated as `TP / (TP + FP)`. | 說明 False Positive 對模型評分的影響。 / Explain how false positives affect the model score. |
| **Recall** | 在所有實際屬於目標類別的樣本中，被模型成功找出的比例，公式為 `TP / (TP + FN)`。 / The proportion of actual target-class samples successfully identified by the model, calculated as `TP / (TP + FN)`. | 說明 False Negative 對模型評分的影響。 / Explain how false negatives affect the model score. |
| **F1-score** | Precision 與 Recall 的調和平均，公式為 `2 × Precision × Recall / (Precision + Recall)`。 / The harmonic mean of Precision and Recall, calculated as `2 × Precision × Recall / (Precision + Recall)`. | 分析模型是否能兼顧 Precision 與 Recall，而非只在單一指標取得高分。 / Analyze whether the model balances Precision and Recall rather than performing well on only one metric. |
| **AP 與 mAP** | AP 根據連續信心分數，彙整不同 threshold 下的 Precision–Recall 表現；mAP 為各類別 AP 的平均值。 / AP summarizes Precision–Recall performance across confidence thresholds using continuous confidence scores, while mAP is the mean AP across classes. | 繪製 Precision–Recall curve，並說明 AP、各類別 AP 與 mAP 的差異。 / Plot a Precision–Recall curve and explain the differences among AP, class-wise AP, and mAP. |

#### 指標計算時的目標類別 / Target Class for Metric Computation

Precision、Recall、F1-score 與單一類別 AP 都需要指定一個目標類別。學生必須在報告中清楚說明計算時將哪一類視為目標類別，並在不同模型之間保持一致。此設定僅用於指標計算，不代表進階任務要求再次執行好／壞圖片分類。

Precision, Recall, F1-score, and single-class AP require a target class to be specified. Students must clearly state which class is treated as the target class during metric computation and use the same definition for all models. This setting is used only for evaluation and does not mean that the advanced task requires another round of Good/Bad image classification.

### 學習目標 / Learning Objectives

- 理解 VGG 與 ResNet 架構差異（參數量、訓練速度、效果）
  Understand the architectural differences between VGG and ResNet (parameter count, training speed, performance)
- 熟悉圖像分類任務的訓練流程（loss function、optimizer、early stopping）
  Familiarize yourself with the image classification training pipeline (loss function, optimizer, early stopping)
- 使用 [Weights & Biases (wandb)](https://wandb.ai/) 可視化訓練過程與超參數實驗結果
  Use [Weights & Biases (wandb)](https://wandb.ai/) to visualize the training process and hyperparameter experiment results
- 學習根據模型的連續輸出分數，使用 Accuracy、Precision、Recall、F1-score、AP 與 mAP 進行量化評分與結果分析。  
  Learn to use continuous model output scores for quantitative evaluation and result analysis with Accuracy, Precision, Recall, F1-score, AP, and mAP.


---

## 2. 資料集準備 / Dataset Preparation
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

同學需在報告中清楚說明：
The report must clearly state:

- 是否使用預訓練權重。  
  Whether pretrained weights are used.
- 輸入影像尺寸。  
  The input image size.
- 是否凍結 backbone。  
  Whether the backbone is frozen.
- optimizer、learning rate、batch size 與 epoch 設定。  
  The optimizer, learning rate, batch size, and number of epochs settings.
- 模型輸出為 logits、Softmax 機率或 Sigmoid 分數。  
  Whether the model output is represented as logits, Softmax probabilities, or Sigmoid scores.

進階任務以模型輸出的分數為主要分析對象，因此測試程式必須保留每張圖片的**連續信心分數**。
The advanced task focuses on the scores produced by the model. Therefore, the evaluation program must preserve the **continuous confidence score** for every image.

---

## 4. 實驗與評估要求 / Experimental and Evaluation Requirements

### 4.1 基本任務要求 / Basic Task Requirements

1. 使用 **VGG16、ResNet18 與 ResNet50** 完成好／壞二分類。  
   Use **VGG16, ResNet18, and ResNet50** to perform Good/Bad binary classification.
2. 每種模型至少完成一組有效訓練，並比較三種模型的分類結果。  
   Complete at least one valid training run for each model and compare their classification results.
3. 至少選擇其中一種模型嘗試兩組以上超參數設定，例如不同 learning rate、batch size、optimizer 或資料增強方式。  
   Select at least one model and evaluate two or more hyperparameter configurations, such as different learning rates, batch sizes, optimizers, or augmentation methods.
4. 使用 wandb 記錄 training loss、validation loss、training accuracy、validation accuracy 與 learning rate。  
   Use wandb to record training loss, validation loss, training accuracy, validation accuracy, and learning rate.
5. 使用驗證集選擇最佳模型，不得使用最終測試集選擇模型。  
   Use the validation set to select the best model. The final test set must not be used for model selection.
6. 列出至少 5 個正確案例與 5 個錯誤案例，分析模型判斷的可能原因。  
   Present at least five correct predictions and five incorrect predictions, and analyze the possible reasons for the model's decisions.

### 4.2 進階任務要求 / Advanced Task Requirements

進階任務只要求同學使用模型輸出的分數完成量化評估與分析，不需要重新訓練一個好／壞分類模型，也不需要另外提交逐張圖片的分類結果。可直接使用基本任務中已完成訓練的模型，或選擇其中表現最佳的模型進行深入分析。

The advanced task requires only quantitative evaluation and analysis of the scores produced by the model. Students do not need to train another Good/Bad classifier or submit an additional per-image classification result. They may directly use the models trained in the basic task or select the best-performing model for a more detailed analysis.

至少完成以下內容：

Complete at least the following items:

1. 保留每張測試圖片的連續模型分數，例如 logits、Softmax probability 或 Sigmoid score。  
   Preserve the continuous model score for every test image, such as logits, Softmax probabilities, or Sigmoid scores.
2. 使用模型分數計算並回報以下五項評估項目：Accuracy、Precision、Recall、F1-score，以及 AP／mAP。  
   Use the model scores to compute and report the following five evaluation items: Accuracy, Precision, Recall, F1-score, and AP/mAP.
3. 對 Accuracy、Precision、Recall 與 F1-score，說明使用的 confidence threshold，以及該 threshold 的選擇方式。  
   For Accuracy, Precision, Recall, and F1-score, state the confidence threshold used and explain how it was selected.
4. AP 與 mAP 必須直接根據連續模型分數計算，不可只使用固定 threshold 後的類別結果。  
   AP and mAP must be computed directly from continuous model scores rather than only from class predictions obtained after applying a fixed threshold.
5. 清楚說明指標計算時使用的目標類別。  
   Clearly state the target class used for metric computation.
6. 繪製至少一張 Precision–Recall curve，並說明曲線與 AP 的關係。  
   Plot at least one Precision–Recall curve and explain its relationship with AP.
7. 比較不同模型或不同超參數設定在五項評估項目上的差異，不可只根據單一指標判斷模型優劣。  
   Compare different models or hyperparameter settings across all five evaluation items. Model quality must not be judged using only one metric.
8. 分析類別不平衡、confidence threshold 與高信心錯誤樣本如何影響最終評分。  
    Analyze how class imbalance, the confidence threshold, and high-confidence errors affect the final scores.

### 4.3 建議結果表格 / Suggested Result Table

| Model | Accuracy | Precision | Recall | F1-score | AP50 | mAP |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| VGG16 |  |  |  |  |  |  |
| ResNet18 |  |  |  |  |  |  |
| ResNet50 |  |  |  |  |  |  |

此表格呈現的是模型的整體評估分數，不是要求同學再次輸出每張圖片的好／壞分類結果。

This table summarizes overall model evaluation scores and does not require students to produce another Good/Bad prediction for each image.

---

## 5. Weights & Biases 使用要求 / Weights & Biases Requirements

所有正式訓練實驗皆須使用 wandb 記錄，建議每次 run 至少包含：

All formal training experiments must be logged with wandb. Each run should include at least:

- 模型名稱與是否使用預訓練權重。  
  Model name and whether pretrained weights are used.
- learning rate、batch size、optimizer、epoch、input size 與 random seed。  
  Learning rate, batch size, optimizer, number of epochs, input size, and random seed.
- training loss 與 validation loss。  
  Training loss and validation loss.
- training accuracy 與 validation accuracy。  
  Training accuracy and validation accuracy.
- validation Precision、Recall 與 F1-score。  
  Validation Precision, Recall, and F1-score.
- 最佳 epoch 與最佳驗證結果。  
  The best epoch and best validation result.
- 最終測試集 Accuracy、Precision、Recall、F1-score、AP 與 mAP。  
  Final test-set Accuracy, Precision, Recall, F1-score, AP, and mAP.
- confusion matrix 或 prediction table。  
  A confusion matrix or prediction table.

wandb run 的名稱應能辨識模型與設定，例如：

The name of each wandb run should identify the model and experimental configuration, for example:

```text
vgg16_pretrained_lr1e-4_bs32
resnet18_pretrained_lr1e-4_bs32
resnet50_scratch_lr1e-3_bs16
```

測試集結果應在模型與超參數確定後再執行一次並記錄，不應反覆查看測試結果來調整模型。

The test set should be evaluated and logged only after the model and hyperparameters have been finalized. Students must not repeatedly inspect test-set results to tune the model.

---

## 6. 作業繳交規範 / Assignment Submission Guidelines

### 必須完成的項目 / Required Tasks
1. 使用三種模型（VGG16、ResNet18、ResNet50）訓練資料集，並記錄 loss 與 accuracy 曲線。
   Train the dataset using three models (VGG16, ResNet18, ResNet50) and record the loss and accuracy curves.
2. 至少嘗試兩種超參數設定（例如不同的 learning rate 或 batch size），請在訓練主程式碼中手動修改並記錄結果。
   Try at least two hyperparameter configurations (e.g., different learning rates or batch sizes), modify them manually in the main training script, and record the results.
3. 全程使用 wandb 記錄訓練過程。
   Use wandb throughout the training process to record and log logs.
4. 完成基本任務的好／壞二分類。  
   Complete the Good/Bad binary-classification basic task
5. 使用模型輸出的連續分數，完成進階任務的五項量化評估與結果分析；不需另外提交逐張圖片的好／壞分類結果。  
   Use continuous model scores to complete the five quantitative evaluations and result analyses in the advanced task; no additional per-image Good/Bad classification file is required.
6. 提交一份報告總結觀察（含圖表與文字說明）。
   Submit a report summarizing your observations (including charts, screenshots, and textual explanations).

### 📂 繳交內容 / Submission Contents
請提交以下檔案 / Please submit the following files:
* **程式碼** / Source Code
* **環境與套件說明 / Environment File**（例如 `requirements.txt` 或 `environment.yml` / e.g., `requirements.txt` or `environment.yml`）
* **訓練結果報告**（含 wandb 圖表截圖與簡要說明）/ Training result report (including wandb chart screenshots and brief explanations)
* **你的 wandb 專案公開連結** / Your public wandb project link

### Overleaf 報告撰寫要求 / Report Requirements in Overleaf
* 本任務要求撰寫一份完整的技術報告，且**必須使用 Overleaf (LaTeX) 進行撰寫**。
  This assignment requires writing a comprehensive technical report, which **must be authored using Overleaf (LaTeX)**.
* 報告中必須包含清晰的章節標籤（Label）以便內文引用，並包含實驗結果表格。
  The report must include clear section labels (`\label{...}`) for in-text cross-referencing, along with tables summarizing experimental results.
* **報告必要章節 / Required Sections:**
  * **模型架構 (Methodology & Model Architecture)**：說明所選用的 CNN 架構與超參數設定。 / Describe the chosen CNN architectures and hyperparameter configurations.
  * **任務定義與標註規則 (Task Definition and Annotation Criteria)** ：說明好／壞影像的定義、自訂品質分數、區分數值門檻及代表性標註範例。 / Describe the definitions of Good and Bad images, the custom quality score, the decision threshold, and representative annotation examples.
  * **基本任務實驗結果與分析 (Basic Task Results & Analysis)** ：呈現 VGG16、ResNet18、ResNet50 的訓練過程與好／壞分類結果。 / Present the training process and Good/Bad classification results of VGG16, ResNet18, and ResNet50.
  * **進階任務：模型分數與評估指標分析 (Advanced Task: Model Scores and Evaluation Metrics)** ：說明模型輸出的連續分數、confidence threshold 的設定方式，以及 Accuracy、Precision、Recall、F1-score、AP 與 mAP 的意義與分析結果。進階任務不需另外呈現逐張圖片的好／壞分類清單。 / Describe the continuous scores produced by the model, the method used to select the confidence threshold, and the meanings and results of Accuracy, Precision, Recall, F1-score, AP, and mAP. The advanced task does not require an additional per-image Good/Bad prediction list.
  * **結論與心得 (Conclusion & Discussion)**
  * **如何使用 AI 幫忙 (AI Tools Utilization)**

### 報告中至少應包含的圖表 / Minimum Required Figures and Tables

- 訓練與驗證 loss 曲線。  
  Training and validation loss curves.
- 訓練與驗證 accuracy 曲線。  
  Training and validation accuracy curves.
- VGG16、ResNet18、ResNet50 的模型比較表。  
  A comparison table for VGG16, ResNet18, and ResNet50.
- Precision–Recall curve。  
  A Precision–Recall curve.
- Accuracy、Precision、Recall、F1-score、AP_Good、AP_Bad 與 mAP 結果表。  
  A result table containing Accuracy, Precision, Recall, F1-score, AP_Good, AP_Bad, and mAP.
- 正確與錯誤預測案例。  
  Correct and incorrect prediction examples.
- wandb 實驗頁面截圖或圖表匯出結果。  
  Screenshots or exported plots from the wandb experiment page.

### 進階任務的文字分析要求 / Written Analysis Requirements for the Advanced Task

進階任務的重點是解釋模型分數與評估結果，而不是重新列出圖片的好／壞分類。針對每項評估指標，報告至少需要回答：

The advanced task focuses on interpreting model scores and evaluation results rather than listing Good/Bad predictions again. For each evaluation metric, the report must answer at least the following questions:

1. 此指標衡量模型的哪一種能力？  
   What aspect of model performance does this metric measure?
2. 此指標的數值是多少？  
   What is the value of this metric?
3. 三種模型中哪一個模型在此指標表現最好？  
   Which of the three models performs best according to this metric?
4. 此結果是否受到類別不平衡或 confidence threshold 影響？  
   Is the result affected by class imbalance or the confidence threshold?
5. 此指標與其他指標是否呈現不同結論？原因可能是什麼？  
   Does this metric lead to a different conclusion from the other metrics? What may explain the difference?

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

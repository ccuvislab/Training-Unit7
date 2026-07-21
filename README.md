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

* **進階任務（品質分數與 Top-100 排序） / Advanced Task (Quality Scoring and Top-100 Ranking)**：
學生需先從固定的候選圖片集中，人工挑選 100 張「表現最好且最接近真實應用需求」的圖片，作為人工最佳 100 張參考集合。人工挑選規則必須在查看模型結果前制定，且人工最佳 100 張不得用於模型訓練、驗證、權重選擇或超參數調整。

Students must first manually select 100 images that perform best and most closely match real-world application requirements from a fixed candidate image pool. These images form the human-selected Top-100 reference set. The human selection criteria must be defined before reviewing model results, and the human-selected Top-100 images must not be used for model training, validation, checkpoint selection, or hyperparameter tuning.

接著，同學需使用人工參考集合以外的品質訓練資料，分別訓練 VGG16、ResNet18 與 ResNet50，使每個模型能對候選圖片產生可排序的品質分數。模型可採用連續分數回歸或 1 至 5 級序位分類，但三個模型必須使用相同的輸出形式，並統一定義為「分數越高代表圖片越好、越接近真實需求」。

Students must then train VGG16, ResNet18, and ResNet50 using quality-training data outside the human reference set, so that each model produces a sortable quality score for every candidate image. The models may use continuous-score regression or 1-to-5 ordinal classification, but all three models must use the same output format and the same score direction: a higher score indicates a better image and a closer match to real-world requirements.

模型完成訓練後，需使用不同的單張圖片排序分數，將完整候選圖片集由高至低排序，並針對每個模型與排序方式取出前 100 名，再與人工最佳 100 張進行相似度與差異分析。

After training, students must use different per-image ranking scores to sort the complete candidate pool from highest to lowest. The Top-100 images produced by each model and ranking method must then be compared with the human-selected Top-100 set.

報告中至少需要建立並分析下列分數與評估項目：

The report must construct and analyze at least the following scoring methods and evaluation items:

| 項目 / Item | 類型 / Type | 定義與代表意義 / Definition and Interpretation |
| :--- | :--- | :--- |
| **原始品質分數 / Raw Quality Score** | 單張圖片排序分數 / Per-image Ranking Score | 直接使用模型對每張圖片輸出的品質分數，依分數由高至低排序。 / Directly use the quality score predicted by the model for each image and rank images from highest to lowest. |
| **信心修正分數 / Confidence-Adjusted Score** | 單張圖片排序分數 / Per-image Ranking Score | 根據測試時增強、MC Dropout、預測熵或機率差距所估計的不確定性，降低高品質但不穩定預測的排序分數。 / Adjust the quality score using uncertainty estimated from test-time augmentation, MC Dropout, prediction entropy, or probability margins, reducing the rank of high-scoring but unstable predictions. |
| **集成品質分數 / Ensemble Quality Score** | 單張圖片排序分數 / Per-image Ranking Score | 將 VGG16、ResNet18 與 ResNet50 的分數正規化後取平均，產生三模型集成排序。 / Normalize and average the scores from VGG16, ResNet18, and ResNet50 to create an ensemble ranking. |
| **Overlap** | Top-100 集合評估 / Top-100 Set Evaluation | 人工最佳 100 張與模型 Top-100 的交集數量除以 100。 / The number of images shared by the human and model Top-100 sets, divided by 100. |
| **Jaccard** | Top-100 集合評估 / Top-100 Set Evaluation | 人工與模型 Top-100 交集大小除以兩集合聯集大小。 / The intersection size of the human and model Top-100 sets divided by their union size. |
| **AP** | 完整排序評估 / Full-Ranking Evaluation | 將人工最佳 100 張視為 relevant，使用完整候選圖片集的連續模型分數評估人工選中圖片是否普遍位於排名前段。 / Treat the human-selected Top-100 images as relevant and use continuous scores over the full candidate pool to evaluate whether those images are generally ranked near the top. |

#### 單張圖片分數與整體評估指標 / Per-Image Scores and Overall Evaluation Metrics

原始品質分數、信心修正分數與集成品質分數可用來替單張圖片排序；Overlap、Jaccard 與 AP 則是比較整體排序結果的評估指標，不能直接當成單張圖片分數。若只有一組人工最佳 100 張參考集合，應回報 **AP**；只有在建立多組獨立人工參考集合並平均各組 AP 時，才可稱為 **mAP**。

Raw quality scores, confidence-adjusted scores, and ensemble quality scores can be used to rank individual images. Overlap@100, Jaccard@100, and AP evaluate the overall ranking result and must not be treated as per-image scores. When only one human-selected Top-100 reference set is available, the result must be reported as **AP**. The term **mAP** is appropriate only when AP is computed for multiple independent reference sets and then averaged.

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

進階任務以每張圖片的品質排序分數為主要分析對象，因此評分程式必須保留每張候選圖片的**連續品質分數**，並確保分數越高代表圖片越好、越接近真實應用需求。
The advanced task focuses on per-image quality-ranking scores. Therefore, the scoring program must preserve a **continuous quality score** for every candidate image and ensure that higher scores consistently represent better images and a closer match to real-world application requirements.

---

## 4. 必要前置軟體與帳號 / Prerequisites

開始訓練前，請準備：

Before starting, prepare the following:

- Python 3.10 或 3.11 / Python 3.10 or 3.11
- Git
- Miniconda、Anaconda 或 Python 虛擬環境  
  Miniconda, Anaconda, or another Python virtual environment
- GitHub 帳號 / GitHub account
- Weights & Biases 帳號 / Weights & Biases account
- Overleaf 帳號 / Overleaf account
- 可執行 PyTorch 的電腦 / A computer capable of running PyTorch
- NVIDIA GPU（非必要，但建議使用）  
  NVIDIA GPU (optional but recommended)

使用 NVIDIA GPU 時，先確認驅動程式與 GPU 狀態：

When using an NVIDIA GPU, first verify the driver and GPU status:

```bash
nvidia-smi
```

若此指令無法執行，請先檢查 NVIDIA 驅動程式、遠端節點、容器或 GPU 權限。

If this command fails, check the NVIDIA driver, remote node allocation, container settings, or GPU permissions.

---

## 5. Python 環境建置 / Python Environment Setup

### 5.1 建立 Conda 環境 / Create a Conda Environment

```bash
conda create -n training-unit7 python=3.11 -y
conda activate training-unit7
python -m pip install --upgrade pip
```

確認目前使用的 Python：

Verify the active Python interpreter:

```bash
python --version
```

Linux 或 macOS：

Linux or macOS:

```bash
which python
```

Windows：

Windows:

```powershell
where python
```

### 5.2 安裝 PyTorch / Install PyTorch

PyTorch 的安裝方式與作業系統、GPU 及 CUDA 版本有關。使用 GPU 的同學應依 [PyTorch 官方安裝頁面](https://pytorch.org/get-started/locally/) 選擇正確指令。

The PyTorch installation command depends on the operating system, GPU, and CUDA version. GPU users should select the correct command from the [official PyTorch installation page](https://pytorch.org/get-started/locally/).

僅使用 CPU 時，可先使用：

For CPU-only execution:

```bash
pip install torch torchvision
```

確認安裝：

Verify the installation:

```bash
python -c "import torch; print(torch.__version__); print('CUDA available:', torch.cuda.is_available())"
```

若使用 GPU：

For GPU users:

```bash
python -c "import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No CUDA device')"
```

### 5.3 安裝其他套件 / Install Additional Packages

```bash
pip install numpy pandas scikit-learn matplotlib pillow tqdm pyyaml wandb
```

### 5.4 登入 wandb / Log in to wandb

```bash
wandb login
```

請勿將 wandb API key 寫入程式碼、README、公開 Repository 或任何會被提交的檔案。

Do not place the wandb API key in source code, the README, a public repository, or any submitted file.

### 5.5 保存環境 / Save the Environment

```bash
pip freeze > requirements.txt
```

或使用 Conda：

Or export the Conda environment:

```bash
conda env export --no-builds > environment.yml
```

至少提交 `requirements.txt` 或 `environment.yml` 其中一個。

Submit at least one of `requirements.txt` or `environment.yml`.

### 5.6 環境檢查 / Environment Check

建立 `check_environment.py`：

Create `check_environment.py`:

```python
import torch
import torchvision
import numpy
import pandas
import sklearn
import matplotlib
import PIL
import wandb

print("PyTorch:", torch.__version__)
print("Torchvision:", torchvision.__version__)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

print("Environment check completed.")
```

執行：

Run:

```bash
python check_environment.py
```

---

## 6. 實驗與評估要求 / Experimental and Evaluation Requirements

### 6.1 基本任務要求 / Basic Task Requirements

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

### 6.2 進階任務要求 / Advanced Task Requirements

進階任務需另外建立固定候選圖片集與人工最佳 100 張參考集合，並訓練 VGG16、ResNet18 與 ResNet50 產生圖片品質排序分數。人工最佳 100 張只可用於最終比較，不可用於模型訓練、驗證、模型權重選擇、超參數調整或排序指標係數調整。

The advanced task requires a fixed candidate image pool and a human-selected Top-100 reference set. Students must train VGG16, ResNet18, and ResNet50 to generate image-quality ranking scores. The human-selected Top-100 images may only be used for final comparison and must not be used for training, validation, checkpoint selection, hyperparameter tuning, or adjustment of ranking-score coefficients.

至少完成以下內容：

Complete at least the following items:

1. 固定候選圖片集，確保人工、三個模型與所有排序方法使用完全相同的候選圖片。  
   Fix the candidate image pool so that the human evaluator, all three models, and all ranking methods use exactly the same candidate images.
2. 在查看模型結果前制定「表現最好且最接近真實需求」的人工挑選規則，並從候選圖片集中人工挑選 100 張圖片。  
   Define the criteria for images that perform best and most closely match real-world requirements before viewing model results, and manually select 100 images from the candidate pool.
3. 鎖定人工最佳 100 張參考集合，不可將其加入訓練集或驗證集，也不可在查看模型排名後更換圖片。  
   Lock the human-selected Top-100 reference set. These images must not enter the training or validation sets and must not be replaced after model rankings are reviewed.
4. 使用人工參考集合以外的圖片品質分數、1 至 5 級品質等級或其他可重現的品質目標，訓練 VGG16、ResNet18 與 ResNet50。  
   Train VGG16, ResNet18, and ResNet50 using reproducible quality targets outside the human reference set, such as continuous quality scores or 1-to-5 quality levels.
5. 三個模型必須採用相同輸出形式，例如連續分數回歸或序位分類，並統一定義分數越高代表圖片越好。  
   All three models must use the same output format, such as continuous-score regression or ordinal classification, and higher scores must consistently represent better images.
6. 對完整候選圖片集產生原始品質分數，並確認每張圖片均有唯一、有效且非 `NaN` 的分數。  
   Generate raw quality scores for the complete candidate pool and verify that every image has one unique, valid, non-`NaN` score.
7. 至少建立原始品質分數、信心修正分數與三模型集成品質分數，並針對每個模型與排序方法由高至低選出前 100 名。  
   Construct at least raw quality scores, confidence-adjusted scores, and a three-model ensemble quality score, then select the Top-100 images for each model and ranking method.
8. 比較模型 Top-100 與人工最佳 100 張，計算交集數量、Overlap 與 Jaccard。  
   Compare each model Top-100 set with the human-selected Top-100 set and compute the intersection count, Overlap, and Jaccard.
9. 使用完整候選圖片集的連續排序分數計算 AP，不可只使用前 100 名的 0／1 結果。若只有一組人工參考集合，應回報 AP 而非 mAP。  
   Compute AP using continuous ranking scores over the full candidate pool rather than binary Top-100 membership alone. When only one human reference set exists, report AP rather than mAP.
10. 分析至少三類圖片：人工與模型共同選中、人工選中但模型漏掉，以及模型選中但人工未選；每類至少展示 5 張圖片。  
    Analyze at least three image groups: images selected by both the human and model, images selected only by the human, and images selected only by the model. Show at least five images from each group.

### 6.3 建議結果表格 / Suggested Result Table

| Model | Ranking Indicator | Intersection Count | Overlap | Jaccard | AP |
| :--- | :--- | ---: | ---: | ---: | ---: |
| VGG16 | Raw Quality Score |  |  |  |  |
| VGG16 | Confidence-Adjusted Score |  |  |  |  |
| ResNet18 | Raw Quality Score |  |  |  |  |
| ResNet50 | Raw Quality Score |  |  |  |  |
| Three-Model Ensemble | Ensemble Quality Score |  |  |  |  |

此表格比較的是各模型與各排序方式所選出的 Top-100 和人工最佳 100 張之間的相似度。AP 必須使用完整候選圖片集的連續分數計算，不能將 AP、Overlap 或 Jaccard 當成單張圖片的品質分數。

This table compares the similarity between the Top-100 images selected by each model or ranking method and the human-selected Top-100 set. AP must be computed using continuous scores over the complete candidate pool. AP, Overlap, and Jaccard must not be treated as per-image quality scores.

---

## 7. Weights & Biases 使用要求 / Weights & Biases Requirements

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
- 基本任務的 validation Precision、Recall 與 F1-score。  
  Validation Precision, Recall, and F1-score for the basic task.
- 進階任務的輸出形式、品質分數範圍、損失函數與品質訓練目標來源。  
  Output format, quality-score range, loss function, and source of quality-training targets for the advanced task.
- 進階任務若採回歸，記錄 validation MAE、RMSE 與 Spearman correlation；若採序位分類，記錄 validation Accuracy、Macro F1 與等級 MAE。  
  For advanced-task regression, log validation MAE, RMSE, and Spearman correlation; for ordinal classification, log validation Accuracy, Macro F1, and level MAE.
- 最佳 epoch 與最佳驗證結果。  
  The best epoch and best validation result.
- 基本任務的最終分類指標，以及進階任務的交集數量、Overlap、Jaccard 與 AP。  
  Final classification metrics for the basic task, and the intersection count, Overlap, Jaccard, and AP for the advanced task.
- confusion matrix、prediction table 或 Top-100 ranking table。  
  A confusion matrix, prediction table, or Top-100 ranking table.

wandb run 的名稱應能辨識模型與設定，例如：

The name of each wandb run should identify the model and experimental configuration, for example:

```text
vgg16_pretrained_lr1e-4_bs32
resnet18_pretrained_lr1e-4_bs32
resnet50_scratch_lr1e-3_bs16
```

最終人工參考集的比較結果應在模型、超參數與排序分數定義確定後再執行一次並記錄，不應反覆查看人工參考結果來調整模型或排序指標。

The final comparison with the human reference set must be performed and logged only after the models, hyperparameters, and ranking-score definitions have been finalized. Students must not repeatedly inspect human-reference results to tune the models or ranking methods.

---

## 8. 作業繳交規範 / Assignment Submission Guidelines

### 必須完成的項目 / Required Tasks
1. 使用三種模型（VGG16、ResNet18、ResNet50）訓練資料集，並記錄 loss 與 accuracy 曲線。
   Train the dataset using three models (VGG16, ResNet18, ResNet50) and record the loss and accuracy curves.
2. 至少嘗試兩種超參數設定（例如不同的 learning rate 或 batch size），請在訓練主程式碼中手動修改並記錄結果。
   Try at least two hyperparameter configurations (e.g., different learning rates or batch sizes), modify them manually in the main training script, and record the results.
3. 全程使用 wandb 記錄訓練過程。
   Use wandb throughout the training process to record and log logs.
4. 完成基本任務的好／壞二分類。  
   Complete the Good/Bad binary-classification basic task
5. 完成進階任務的人工最佳 100 張參考集合、三模型品質分數與 Top-100 排序，並使用交集數量、Overlap、Jaccard 與 AP 比較人工和模型結果。  
   Complete the human-selected Top-100 reference set, quality scoring, and Top-100 ranking for the three models, and compare human and model results using intersection count, Overlap, Jaccard, and AP.
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
  * **進階任務：品質分數與人工 Top-100 比較 (Advanced Task: Quality Scores and Human Top-100 Comparison)** ：說明人工最佳圖片挑選規則、品質訓練目標、模型輸出形式、原始／信心修正／集成品質分數，以及各模型 Top-100 和人工最佳 100 張在 Overlap、Jaccard 與 AP 上的比較結果。 / Describe the human image-selection criteria, quality-training targets, model output format, raw/confidence-adjusted/ensemble quality scores, and the comparison between each model Top-100 and the human-selected Top-100 set using Overlap, Jaccard, and AP.
  * **結論與心得 (Conclusion & Discussion)**
  * **如何使用 AI 幫忙 (AI Tools Utilization)**

### 報告中至少應包含的圖表 / Minimum Required Figures and Tables

- 訓練與驗證 loss 曲線。  
  Training and validation loss curves.
- 訓練與驗證 accuracy 曲線。  
  Training and validation accuracy curves.
- VGG16、ResNet18、ResNet50 的模型比較表。  
  A comparison table for VGG16, ResNet18, and ResNet50.
- 各模型與各排序方式的 Top-100 結果表。  
  A Top-100 result table for each model and ranking method.
- 交集數量、Overlap、Jaccard 與 AP 比較表。  
  A comparison table containing intersection count, Overlap, Jaccard, and AP.
- 人工與模型共同選中、人工選中但模型漏掉，以及模型選中但人工未選的圖片案例。  
  Image examples selected by both the human and model, selected only by the human, and selected only by the model.
- 正確與錯誤預測案例。  
  Correct and incorrect prediction examples.
- wandb 實驗頁面截圖或圖表匯出結果。  
  Screenshots or exported plots from the wandb experiment page.

### 進階任務的文字分析要求 / Written Analysis Requirements for the Advanced Task

進階任務的重點是比較人工最佳 100 張與模型品質排序結果，而不是將 Overlap、Jaccard 或 AP 當成單張圖片的分數。報告至少需要回答：

The advanced task focuses on comparing the human-selected Top-100 set with model-generated quality rankings. Overlap, Jaccard, and AP must not be treated as per-image scores. The report must answer at least the following questions:

1. 人工如何定義「表現最好且最接近真實需求」，挑選規則是否在查看模型結果前固定？  
   How was “best-performing and closest to real-world requirements” defined, and were the human selection criteria fixed before reviewing model results?
2. 模型使用何種品質訓練目標與輸出形式？分數越高是否一致代表圖片越好？  
   What quality-training targets and output format were used, and do higher scores consistently indicate better images?
3. VGG16、ResNet18、ResNet50 在原始品質分數下各自選出的 Top-100，與人工最佳 100 張有多少交集？  
   How many images overlap between the human-selected Top-100 set and the Top-100 sets produced by VGG16, ResNet18, and ResNet50 using raw quality scores?
4. 信心修正分數或集成品質分數是否改善 Overlap、Jaccard 或 AP？可能原因為何？  
   Did confidence-adjusted or ensemble quality scores improve Overlap, Jaccard, or AP, and what may explain the result?
5. 為什麼 AP 必須使用完整候選圖片集的連續分數？本實驗是否只有一組人工參考集合，因此應回報 AP 而非 mAP？  
   Why must AP be computed using continuous scores over the complete candidate pool, and does this experiment contain only one human reference set, requiring AP rather than mAP?
6. 人工與模型共同選中的圖片具有哪些特徵？  
   What characteristics are shared by images selected by both the human and the model?
7. 人工選中但模型漏掉的圖片，是否反映模型忽略真實性、任務資訊或特殊場景？  
   Do images selected only by the human reveal that the model overlooks realism, task-relevant information, or uncommon scenes?
8. 模型選中但人工未選的圖片，是否顯示模型偏好高亮度、高對比、特定背景或其他非預期特徵？  
   Do images selected only by the model indicate a preference for high brightness, high contrast, particular backgrounds, or other unintended features?

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

# Training-Unit7 Rundown（一）：人工與模型的好／壞二分類比較

## 0. 任務目的

本任務要求同學先從實驗室提供的資料集中挑選 100 張圖片，依照自行制定且可重現的標準，人工標註為「好影像」或「壞影像」。這 100 張圖片作為最終人工參考集，不參與模型訓練或驗證。

接著，同學需使用人工參考集以外的訓練資料，分別訓練 VGG16、ResNet18 與 ResNet50。模型訓練完成後，先對完整候選資料集執行分類，再使用三個模型對固定的 100 張人工參考圖片進行推論，最後比較人工標註與模型分類結果的一致性與差異。

整體流程如下：

```text
建立環境
→ 檢查資料集
→ 制定好／壞標註規則
→ 挑選並人工標註 100 張圖片
→ 鎖定人工參考集
→ 準備其他訓練與驗證資料
→ 訓練 VGG16、ResNet18、ResNet50
→ 對完整資料集執行分類
→ 對人工 100 張執行推論
→ 比較人工與模型結果
→ 撰寫報告並繳交
```

> 100 張人工標註圖片只能作為最終比較依據，不可用於訓練、選擇模型權重或調整超參數。

---

## 1. 必要帳號與軟體

開始作業前，請準備：

- Python 3.10 或 3.11
- Git
- Miniconda、Anaconda 或 Python 虛擬環境
- GitHub 帳號
- Weights & Biases 帳號
- Overleaf 帳號
- 可執行 PyTorch 的電腦
- NVIDIA GPU（非必要，但建議使用）

使用 NVIDIA GPU 時，先確認驅動程式與 GPU 是否正常：

```bash
nvidia-smi
```

若指令無法執行，可能代表：

- 尚未安裝 NVIDIA 驅動程式
- 驅動程式未正確載入
- 使用的電腦沒有 NVIDIA GPU
- 目前位於無法存取 GPU 的容器或遠端節點

---

## 2. 建立 Python 環境

### 2.1 使用 Conda 建立環境

```bash
conda create -n training-unit7 python=3.11 -y
conda activate training-unit7
python -m pip install --upgrade pip
```

確認目前使用的 Python：

```bash
which python
python --version
```

Windows 可使用：

```powershell
where python
python --version
```

### 2.2 安裝 PyTorch

PyTorch 的安裝方式與作業系統、GPU 及 CUDA 版本有關。請依官方安裝頁面選擇符合電腦環境的指令。

僅使用 CPU 時，可先使用：

```bash
pip install torch torchvision
```

完成後測試：

```bash
python -c "import torch; print(torch.__version__); print('CUDA available:', torch.cuda.is_available())"
```

若使用 GPU，還可檢查：

```bash
python -c "import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No CUDA device')"
```

### 2.3 安裝其他套件

```bash
pip install numpy pandas scikit-learn matplotlib pillow tqdm pyyaml wandb
```

### 2.4 登入 wandb

```bash
wandb login
```

請勿將 wandb API key 上傳至 GitHub，也不要寫在程式碼中。

### 2.5 建立環境檔案

```bash
pip freeze > requirements.txt
```

也可使用 Conda：

```bash
conda env export --no-builds > environment.yml
```

### 2.6 完整環境測試

建立 `check_environment.py`：

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

print("環境檢查完成")
```

執行：

```bash
python check_environment.py
```

---

## 3. 建立 GitHub 專案

Repository 建議命名為：

```text
training-Unit7-Binary-{學號}
```

例如：

```text
training-Unit7-Binary-613410112
```

初始化專案：

```bash
git init
```

建議資料夾結構：

```text
training-Unit7-Binary-{學號}/
├── README.md
├── requirements.txt
├── environment.yml
├── configs/
│   ├── vgg16.yaml
│   ├── resnet18.yaml
│   └── resnet50.yaml
├── data/
│   ├── raw/
│   ├── annotations/
│   │   ├── binary_reference_100.csv
│   │   └── training_labels.csv
│   └── splits/
│       ├── train.csv
│       ├── val.csv
│       └── reference_100.csv
├── src/
│   ├── dataset.py
│   ├── models.py
│   ├── train_binary.py
│   ├── infer_dataset.py
│   ├── infer_reference.py
│   └── evaluate_agreement.py
├── outputs/
│   ├── checkpoints/
│   ├── predictions/
│   ├── figures/
│   └── logs/
└── report/
    └── report.pdf
```

Windows 無法使用 `mkdir -p` 時，可手動建立資料夾。

建議 `.gitignore`：

```gitignore
__pycache__/
*.pyc
.env
wandb/
data/raw/
outputs/checkpoints/
*.pt
*.pth
*.ckpt
```

大型資料集與模型權重不可直接上傳至 GitHub。

---

## 4. 檢查原始資料集

在人工挑選與訓練前，先檢查：

- 圖片是否可以正常開啟
- 是否存在空檔案或損壞檔案
- 是否有重複圖片
- 是否有高度相似的連續影格
- 圖片解析度是否異常
- 圖片方向是否錯誤
- 檔名是否重複

---

## 5. 制定人工好／壞判定規則

### 5.1 規則要求

人工規則必須在標註前完成，並符合：

1. 所有圖片使用相同規則。
2. 規則必須可由他人理解與重現。
3. 不可看完模型結果後修改規則。
4. 必須有明確計分方式或門檻。
5. 報告中必須提供規則、門檻及標註範例。

### 5.2 評分規則範例

| 評分項目 | 0 分 | 1 分 | 2 分 |
|---|---|---|---|
| 道路可見程度 | 幾乎不可見 | 部分可見 | 清楚可見 |
| 障礙物可辨識程度 | 無法判讀 | 部分可判讀 | 清楚可辨識 |
| 影像清晰度 | 嚴重模糊 | 輕微模糊 | 清楚 |
| 曝光品質 | 嚴重過曝或過暗 | 輕微異常 | 正常 |
| 遮擋程度 | 重要區域嚴重遮擋 | 部分遮擋 | 幾乎無遮擋 |

總分範圍為 0 至 10 分。範例門檻：

```text
總分大於或等於 7：好影像
總分小於 7：壞影像
```

這只是示例，同學可自行修改項目、權重與門檻，但必須在開始標註前固定。

---

## 6. 挑選並人工標註 100 張圖片

### 6.1 挑選原則

100 張圖片應盡量涵蓋：

- 不同場景
- 不同亮度
- 不同模糊程度
- 不同道路狀態
- 不同障礙物狀態
- 不同遮擋程度
- 足夠數量的好影像與壞影像
- 容易判斷與邊界案例

人工參考集不一定需要好／壞各 50 張，但類別不宜嚴重失衡。

### 6.2 建立人工標註檔案

建立：

```text
data/annotations/binary_reference_100.csv
```

格式：

```csv
image_id,image_path,human_score,human_label,reason
000001,data/raw/000001.jpg,8,Good,道路清楚且曝光正常
000002,data/raw/000002.jpg,4,Bad,畫面模糊且道路資訊不完整
```

建議標籤編碼：

```text
Bad = 0
Good = 1
```

### 6.3 鎖定人工參考集

完成後：

- 不可加入訓練集
- 不可加入驗證集
- 不可用來選擇模型權重
- 不可用來調整 learning rate、batch size 或 epoch
- 不可因模型結果而修改人工標籤
- 不可更換人工參考圖片

---

## 7. 準備模型訓練資料

### 7.1 訓練資料必須有標籤

VGG16、ResNet18 與 ResNet50 是監督式分類模型，訓練資料必須具有好／壞標籤。

標籤來源可以是：

1. 實驗室提供的好／壞標籤。
2. 同學使用相同人工規則，標註人工參考集以外的另一批圖片。
3. 由實驗室提供既有模型產生初步標籤，再由同學人工檢查。

不可使用人工參考集的 100 張作為訓練資料。

### 7.2 切分訓練集與驗證集

建議比例：

```text
訓練集：80%
驗證集：20%
```

以 `source_group` 為單位切分，避免同一影片的連續影格同時出現在訓練集與驗證集。

### 7.3 檢查類別分布

| 資料集合 | 好影像 | 壞影像 | 總數 |
|---|---:|---:|---:|
| 訓練集 |  |  |  |
| 驗證集 |  |  |  |
| 人工參考集 |  |  | 100 |

若訓練資料類別不平衡，可使用：

- 類別權重
- 加權取樣器
- 適度資料增強

不可修改人工參考集的分布以提高結果。

---

## 8. 資料前處理

三個模型應使用相同輸入設定，以確保公平比較。

建議：

```text
輸入尺寸：依照資料集中圖片大小為準
色彩格式：RGB
正規化：ImageNet 平均值與標準差
```

訓練資料可使用：

- `RandomResizedCrop`
- `RandomHorizontalFlip`
- 輕微 `ColorJitter`
- 輕微 `RandomRotation`
- `Normalize`

驗證集、完整資料集推論及人工參考集推論只能使用確定性前處理，例如：

- `Resize`
- `CenterCrop`
- `Normalize`

不可對人工參考集使用隨機資料增強。

---

## 9. 建立三種模型

必須完成：

- VGG16
- ResNet18
- ResNet50

將最後分類層修改為兩個輸出：

```text
輸出 0：壞影像
輸出 1：好影像
```

建議使用：

```text
損失函數：CrossEntropyLoss
```

報告中必須說明：

- 是否使用 ImageNet 預訓練權重
- 是否凍結 backbone
- 輸入影像尺寸
- 最後分類層的修改方式
- 損失函數
- 最佳化器
- learning rate
- batch size
- epoch
- random seed

---

## 10. 正式訓練前快速測試

正式訓練前，先完成小型測試：

1. 使用 8 至 16 張圖片。
2. 執行 1 個 epoch。
3. 確認程式可完成前向與反向傳播。
4. 確認 loss 不是 `NaN`。
5. 確認 GPU 或 CPU 裝置正確。
6. 確認模型權重可以儲存與載入。
7. 確認 wandb 有收到資料。
8. 確認訓練與驗證資料不包含人工 100 張。

---

## 11. 訓練 VGG16、ResNet18 與 ResNet50

### 11.1 建議基準設定

```text
預訓練權重：ImageNet
最佳化器：AdamW
Learning rate：1e-4
Batch size：32
Epoch：30
Weight decay：1e-4
Early stopping patience：5
Random seed：42
```

可依 GPU 記憶體與資料量調整。

### 11.2 超參數比較

至少選擇一種模型完成兩組以上設定，例如：

| 實驗 | 模型 | Learning rate | Batch size | 最佳化器 |
|---|---|---:|---:|---|
| A | ResNet18 | 1e-4 | 32 | AdamW |
| B | ResNet18 | 1e-3 | 32 | AdamW |

每次盡量只修改一至兩個設定，才能分析差異來源。

### 11.3 選擇最佳模型權重

只能使用驗證集選擇最佳權重，例如：

- validation loss 最低
- validation F1-score 最高
- validation accuracy 最高

必須在訓練前決定選擇規則。

---

## 12. wandb 紀錄要求

每次實驗的設定至少包含：

```text
model_name
pretrained
freeze_backbone
input_size
optimizer
learning_rate
batch_size
epochs
weight_decay
random_seed
augmentation
```

每個 epoch 至少記錄：

```text
train/loss
train/accuracy
val/loss
val/accuracy
val/precision
val/recall
val/f1
learning_rate
```

建議實驗名稱：

```text
binary_vgg16_pretrained_lr1e-4_bs32
binary_resnet18_pretrained_lr1e-4_bs32
binary_resnet50_pretrained_lr1e-4_bs32
```

---

## 13. 對完整資料集執行分類

三個模型訓練完成並固定權重後，分別對候選資料集執行分類。

輸出格式：

```csv
image_id,model,predicted_label,score_good,score_bad
000001,VGG16,Good,0.91,0.09
000001,ResNet18,Good,0.87,0.13
000001,ResNet50,Bad,0.42,0.58
```

此步驟可用於：

- 觀察完整資料集的分類分布
- 比較三個模型的預測差異
- 找出三個模型一致或不一致的圖片
- 建立後續錯誤案例分析

完整資料集推論結果不能取代人工參考集評估。

---

## 14. 對人工 100 張圖片執行推論

三個模型必須使用：

- 相同的 100 張圖片
- 相同圖片順序
- 相同前處理
- 各自固定的最佳模型權重
- 相同標籤編碼

輸出格式：

```csv
image_id,human_label,model,predicted_label,score_good,score_bad,match
000001,Good,VGG16,Good,0.91,0.09,1
000001,Good,ResNet18,Bad,0.41,0.59,0
000001,Good,ResNet50,Good,0.88,0.12,1
```

每個模型必須輸出 100 筆結果。

---

## 15. 比較人工與模型分類結果

### 15.1 一致數量與一致率

```text
一致數量 = 人工標籤與模型預測相同的圖片數
不一致數量 = 100 − 一致數量
一致率 = 一致數量 ÷ 100
```

### 15.2 混淆矩陣

若將壞影像視為正類：

|  | 模型判斷為壞 | 模型判斷為好 |
|---|---:|---:|
| 人工標註為壞 | TP | FN |
| 人工標註為好 | FP | TN |

### 15.3 Precision、Recall 與 F1-score

三個模型必須使用相同的正類定義。

```text
Precision = TP / (TP + FP)
Recall = TP / (TP + FN)
F1-score = 2 × Precision × Recall / (Precision + Recall)
```

### 15.4 Cohen’s Kappa

Cohen’s Kappa 用於評估人工與模型超出隨機情況的一致程度。

不可只根據 Kappa 判斷模型，仍需結合：

- 一致率
- 混淆矩陣
- Precision
- Recall
- F1-score
- 個別圖片案例

### 15.5 建議結果表格

| 模型 | 一致數量 | 一致率 | Precision | Recall | F1-score | Cohen’s Kappa |
|---|---:|---:|---:|---:|---:|---:|
| VGG16 |  |  |  |  |  |  |
| ResNet18 |  |  |  |  |  |  |
| ResNet50 |  |  |  |  |  |  |

---

## 16. 圖片案例差異分析

每個模型至少挑選：

- 5 張人工與模型一致的圖片
- 5 張人工與模型不一致的圖片
- 至少 2 張高信心錯誤圖片

分析內容至少包含：

- 人工標籤
- 模型預測
- 模型信心分數
- 圖片特徵
- 模型可能判斷錯誤的原因
- 人工規則是否存在邊界案例
- 訓練資料與人工參考集是否有分布差異

常見原因可能包括：

- 影像模糊
- 過曝或過暗
- 道路被遮擋
- 障礙物太小
- 背景特徵誤導模型
- 畫面中存在模型未見過的場景
- 人工評分規則仍有主觀邊界

---

## 17. Overleaf 報告建議結構

```text
1. 作業目的
2. 執行環境
3. 人工好／壞判定規則
4. 人工 100 張參考集
5. 訓練與驗證資料
6. 模型架構與超參數
7. VGG16、ResNet18、ResNet50 訓練結果
8. 完整資料集分類結果
9. 人工與模型一致性比較
10. 圖片案例差異分析
11. 結論與心得
12. AI 工具使用說明
```

報告至少包含：

- 人工標註規則與門檻
- 3 張好影像與 3 張壞影像範例
- 訓練、驗證與人工參考集數量
- 三個模型的設定
- loss 與 accuracy 曲線
- wandb 圖表
- 人工與模型比較表
- 混淆矩陣
- 一致與不一致圖片
- GitHub 與 wandb 連結

---

## 18. 最終檢查表

### 環境

- [ ] Python 環境可正常啟用
- [ ] PyTorch 與 torchvision 可正常匯入
- [ ] scikit-learn 與 wandb 可正常匯入
- [ ] GPU 或 CPU 裝置已確認
- [ ] `requirements.txt` 或 `environment.yml` 已建立

### 人工參考集

- [ ] 已制定好／壞標註規則
- [ ] 已挑選並標註 100 張圖片
- [ ] 已保存分數、標籤與原因
- [ ] 100 張圖片未進入訓練或驗證
- [ ] 未根據模型結果修改人工標籤

### 模型訓練

- [ ] 已訓練 VGG16
- [ ] 已訓練 ResNet18
- [ ] 已訓練 ResNet50
- [ ] 至少完成一組超參數比較
- [ ] 已使用驗證集選擇最佳模型權重
- [ ] 正式實驗均記錄於 wandb

### 最終評估

- [ ] 三個模型均對完整資料集執行分類
- [ ] 三個模型均對人工 100 張執行推論
- [ ] 每個模型均有 100 筆結果
- [ ] 已計算一致率
- [ ] 已建立混淆矩陣
- [ ] 已計算 Precision、Recall 與 F1-score
- [ ] 已計算 Cohen’s Kappa
- [ ] 已分析一致、不一致與高信心錯誤案例

# Training-Unit7 Rundown（二）：模型指標分數與人工最佳 100 張比較

## 0. 任務目的

本任務要求同學先從實驗室提供的候選資料集中，人工挑選 100 張「表現最好且最接近真實應用需求」的圖片，作為人工最佳 100 張參考集合。

接著，同學需使用人工參考集合以外的訓練資料，分別訓練 VGG16、ResNet18 與 ResNet50，使模型能對每張候選圖片產生可排序的品質分數。模型訓練完成後，需根據不同的單張圖片指標排序完整候選資料集，分別取出分數最高的 100 張圖片，再與人工挑選的 100 張進行相似度與差異分析。

整體流程如下：

```text
建立環境
→ 固定候選圖片集
→ 制定人工最佳圖片挑選規則
→ 人工挑選最佳 100 張
→ 鎖定人工參考集合
→ 準備其他圖片的品質訓練目標
→ 訓練 VGG16、ResNet18、ResNet50
→ 對候選圖片產生品質分數
→ 建立不同的單張圖片排序指標
→ 各模型與各指標取前 100 名
→ 與人工最佳 100 張比較
→ 分析模型與人工的差異
→ 撰寫報告並繳交
```

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

使用 NVIDIA GPU 時先執行：

```bash
nvidia-smi
```

---

## 2. 建立 Python 環境

### 2.1 建立 Conda 環境

```bash
conda create -n training-unit7 python=3.11 -y
conda activate training-unit7
python -m pip install --upgrade pip
```

### 2.2 安裝 PyTorch

依作業系統與 CUDA 環境選擇正確的 PyTorch 安裝方式。

僅使用 CPU 時，可先使用：

```bash
pip install torch torchvision
```

確認安裝：

```bash
python -c "import torch; print(torch.__version__); print('CUDA available:', torch.cuda.is_available())"
```

### 2.3 安裝其他套件

```bash
pip install numpy pandas scikit-learn matplotlib pillow tqdm pyyaml wandb
```

這些套件可用於：

- 資料處理
- 模型訓練
- 圖表繪製
- AP 計算
- wandb 紀錄

### 2.4 登入 wandb

```bash
wandb login
```

### 2.5 保存環境

```bash
pip freeze > requirements.txt
```

或：

```bash
conda env export --no-builds > environment.yml
```

### 2.6 環境測試

建立 `check_environment.py`：

```python
import torch
import torchvision
import sklearn
import pandas
import numpy
import matplotlib
import wandb

print("PyTorch:", torch.__version__)
print("Torchvision:", torchvision.__version__)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

print("環境檢查完成")
```

---

## 3. 建立專案結構

Repository 建議命名為：

```text
training-Unit7-Metrics-{學號}
```

建議結構：

```text
training-Unit7-Metrics-{學號}/
├── README.md
├── requirements.txt
├── configs/
│   ├── ranking_vgg16.yaml
│   ├── ranking_resnet18.yaml
│   └── ranking_resnet50.yaml
├── data/
│   ├── raw/
│   ├── annotations/
│   │   ├── ranking_reference_top100.csv
│   │   └── ranking_training_targets.csv
│   └── splits/
│       ├── train.csv
│       ├── val.csv
│       └── candidate_pool.csv
├── src/
│   ├── dataset.py
│   ├── models.py
│   ├── train_ranking.py
│   ├── score_images.py
│   ├── build_indicators.py
│   └── evaluate_top100.py
├── outputs/
│   ├── checkpoints/
│   ├── scores/
│   ├── rankings/
│   ├── figures/
│   └── logs/
└── report/
    └── report.pdf
```

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

---

## 4. 固定候選圖片集

人工與所有模型必須從同一批候選圖片中選出最佳 100 張。

建立：

```text
data/splits/candidate_pool.csv
```

格式：

```csv
image_id,image_path
000001,data/raw/000001.jpg
000002,data/raw/000002.jpg
```

候選圖片集固定後：

- 不可增加或刪除圖片
- 不可因模型結果而更換圖片
- 所有模型必須使用相同圖片
- 所有指標必須使用相同圖片
- 人工最佳 100 張必須從此集合中挑選

---

## 5. 定義「最佳且最接近現實」

人工最佳 100 張不需要標註為好／壞，但必須先制定明確的挑選規則。

可考慮：

- 道路資訊是否完整
- 障礙物資訊是否清楚
- 影像是否清晰
- 曝光是否自然
- 畫面是否接近真實拍攝狀況
- 重要區域是否被遮擋
- 是否因過度後製而失真
- 是否包含實際任務需要的內容
- 是否只是畫面漂亮，但缺乏道路或障礙物資訊

人工挑選原則必須：

1. 在查看模型結果前制定。
2. 對所有圖片使用相同標準。
3. 可以被他人理解。
4. 在報告中完整說明。
5. 不可因模型排名結果而修改。

---

## 6. 人工挑選最佳 100 張圖片

### 6.1 建立人工參考集合

建立：

```text
data/annotations/ranking_reference_top100.csv
```

若只需要比較集合，可使用：

```csv
image_id,image_path,selected_by_human,selection_reason
000010,data/raw/000010.jpg,1,畫面清楚且符合實際道路狀況
```

若另外提供人工名次，可使用：

```csv
human_rank,image_id,image_path,selection_reason
1,000010,data/raw/000010.jpg,道路與障礙物資訊最完整
2,000051,data/raw/000051.jpg,曝光自然且場景合理
```

### 6.2 未排序集合與完整排名的差異

預設情況下，人工只需要挑選 100 張，不需要將其排序為第 1 至第 100 名。

若人工沒有提供完整名次：

- 可使用 Overlap
- 可使用 AP

若人工提供完整名次，才可額外分析排名相關性。

### 6.3 鎖定人工參考集合

完成後：

- 不可直接用人工最佳 100 張訓練模型
- 不可加入驗證集
- 不可用來選擇模型權重
- 不可用來調整超參數
- 不可看完模型結果後更換人工圖片

---

## 7. 準備模型品質訓練目標

### 7.1 為什麼需要額外訓練目標

VGG16、ResNet18 與 ResNet50 必須有訓練目標，才能學習圖片品質或現實接近程度。

人工最佳 100 張是最終參考集合，不能直接作為模型訓練資料。

可使用：

1. 實驗室提供的圖片品質分數。
2. 人工最佳 100 張以外的其他圖片品質分數。
3. 1 至 5 級品質等級。
4. 0 至 100 的連續品質分數。
5. 成對比較標註，例如圖片 A 是否比圖片 B 更好。

不可直接將人工最佳 100 張設為正類，其他圖片全部設為負類，因為未被選中的圖片不一定是壞圖片。

### 7.2 建議訓練目標格式

連續分數：

```csv
image_id,image_path,quality_score
000101,data/raw/000101.jpg,82
000102,data/raw/000102.jpg,47
```

序位等級：

```csv
image_id,image_path,quality_level
000101,data/raw/000101.jpg,5
000102,data/raw/000102.jpg,2
```

---

## 8. 選擇模型輸出形式

三個模型必須使用相同的輸出形式。

例如：

模型輸出一個連續品質分數：

```text
輸出範圍：0 至 1，或 0 至 100
```

可使用：

- `MSELoss`
- `L1Loss`
- `SmoothL1Loss`

或是也可：

輸出 1 至 5 級品質等級：

```text
1：最低
2：偏低
3：普通
4：良好
5：最佳
```

可使用：

- `CrossEntropyLoss`
- 序位分類損失函數

模型可依各等級機率計算期望品質分數：

```text
期望品質分數 = Σ P(等級 = k) × k
```

總之請必須統一定義：

```text
分數越高，代表圖片越好、越接近真實需求
```

三個模型與所有指標必須使用相同方向。

---

## 9. 切分訓練集與驗證集

要求：

- 人工最佳 100 張不得進入訓練集
- 人工最佳 100 張不得進入驗證集
- 使用驗證集選擇模型權重與超參數
- 不可使用人工最佳 100 張調整模型

建議：

```text
訓練集：80%
驗證集：20%
```

---

## 10. 資料前處理

建議三個模型使用：

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

驗證與候選圖片評分必須使用固定前處理。

---

## 11. 訓練三種模型

必須完成：

- VGG16
- ResNet18
- ResNet50

### 11.1 回歸任務建議設定

```text
損失函數：SmoothL1Loss
最佳化器：AdamW
Learning rate：1e-4
Batch size：32
Epoch：30
Early stopping patience：5
Random seed：42
```

### 11.2 序位分類建議設定

```text
損失函數：CrossEntropyLoss
最佳化器：AdamW
Learning rate：1e-4
Batch size：32
Epoch：30
Early stopping patience：5
Random seed：42
```

### 11.3 驗證指標

回歸可記錄：

- MAE
- RMSE
- Spearman correlation
- validation loss

序位分類可記錄：

- Accuracy
- Macro F1
- 預測等級與真實等級的 MAE
- validation loss

最佳模型權重只能由驗證集決定。

---

## 12. 正式訓練前快速測試

1. 使用少量訓練圖片。
2. 執行 1 個 epoch。
3. 確認每張圖片可輸出一個品質分數。
4. 確認分數不是全部相同。
5. 確認分數不是 `NaN`。
6. 確認分數方向正確。
7. 確認模型權重可儲存與載入。
8. 確認 wandb 有收到資料。
9. 確認人工最佳 100 張未參與訓練與驗證。

---

## 13. wandb 紀錄要求

設定至少包含：

```text
task
model_name
output_type
score_range
loss_function
pretrained
input_size
optimizer
learning_rate
batch_size
epochs
random_seed
augmentation
```

每個 epoch 至少記錄：

```text
train/loss
val/loss
val/mae
val/rmse
val/spearman
learning_rate
```

序位分類可改為：

```text
train/loss
val/loss
val/accuracy
val/macro_f1
val/level_mae
learning_rate
```

建議實驗名稱：

```text
ranking_vgg16_regression_lr1e-4_bs32
ranking_resnet18_regression_lr1e-4_bs32
ranking_resnet50_regression_lr1e-4_bs32
```

---

## 14. 對候選圖片集產生原始品質分數

三個模型分別對所有候選圖片輸出品質分數。

格式：

```csv
image_id,image_path,model,raw_quality_score
000001,data/raw/000001.jpg,VGG16,0.8124
000001,data/raw/000001.jpg,ResNet18,0.8462
000001,data/raw/000001.jpg,ResNet50,0.8390
```

必須確認：

- 每個模型的輸出數量等於候選圖片數量
- 每個圖片編號只出現一次
- 沒有缺失分數
- 沒有 `NaN`
- 三個模型均為分數越高代表越好

---

## 15. 建立單張圖片排序指標

各項指標都必須為每張圖片產生一個可排序分數。

### 15.1 原始品質分數

直接使用模型輸出的品質分數：

```text
原始品質分數 = 模型預測分數
```

### 15.2 信心修正分數

回歸模型可使用：

- 測試時增強標準差
- MC Dropout 標準差

序位分類模型可使用：

- 預測熵
- 最高與第二高機率差距

範例：

```text
信心修正分數
= 品質分數 ×（1 − 正規化不確定性）
```

### 15.3 集成品質分數

先將 VGG16、ResNet18 與 ResNet50 的分數正規化，再取平均：

```text
集成品質分數
= 三個模型正規化分數的平均值
```

不可查看人工最佳 100 張後再調整模型權重。

---

## 16. 各模型與各指標選出前 100 名

排序規則：

```text
分數越高代表圖片越好
依分數由高至低排序
取前 100 張
```

輸出格式：

```csv
rank,image_id,image_path,model,indicator,score
1,000010,data/raw/000010.jpg,ResNet50,raw_quality,0.9631
2,000044,data/raw/000044.jpg,ResNet50,raw_quality,0.9574
```

至少應產生：

- VGG16 原始品質分數前 100 名
- ResNet18 原始品質分數前 100 名
- ResNet50 原始品質分數前 100 名
- 各模型信心修正分數前 100 名
- 集成品質分數前 100 名

若分數相同，必須預先制定排序規則，例如以 `image_id` 由小至大排列。

---

## 17. 比較模型前 100 名與人工最佳 100 張

定義：

```text
H = 人工最佳 100 張集合
M = 模型或指標選出的前 100 張集合
I = H 與 M 的交集數量
```

### 17.1 交集數量

```text
交集數量 = |H ∩ M|
```

### 17.2 Overlap

```text
Overlap@100 = 交集數量 ÷ 100
```

由於人工與模型集合均為 100 張，因此：

```text
Precision = Recall = Overlap
```

### 17.3 Jaccard

```text
Jaccard
= 交集數量 ÷（200 − 交集數量）
```

### 17.4 AP

將候選圖片集中：

```text
人工最佳 100 張：relevant = 1
其他圖片：relevant = 0
```

使用完整模型連續分數計算：

```python
from sklearn.metrics import average_precision_score

ap = average_precision_score(
    human_relevance,
    model_scores
)
```

不可只用模型前 100 名的 0／1 結果計算 AP。

### 17.5 AP 與 mAP

若只有一組人工最佳 100 張，應回報 AP。

只有在依不同場景或查詢建立多組獨立人工參考集合，再平均各組 AP 時，才稱為 mAP。

---

## 18. 建議結果表格

### 18.1 三個模型的原始品質分數

| 模型 | 交集數量 | Overlap | Jaccard | AP |
|---|---:|---:|---:|---:|
| VGG16 |  |  |  |  |
| ResNet18 |  |  |  |  |
| ResNet50 |  |  |  |  |

### 18.2 各指標比較

| 模型 | 排序指標 | 交集數量 | Overlap | Jaccard | AP |
|---|---|---:|---:|---:|---:|
| VGG16 | 原始品質分數 |  |  |  |  |
| VGG16 | 信心修正分數 |  |  |  |  |
| ResNet18 | 原始品質分數 |  |  |  |  |
| ResNet50 | 原始品質分數 |  |  |  |  |
| 三模型集成 | 集成品質分數 |  |  |  |  |

---

## 19. 圖片差異分析

針對每個主要模型與最佳指標，建立三組圖片：

### 19.1 人工與模型共同選中

```text
H ∩ M
```

代表人工與模型都認為是最佳圖片。

### 19.2 人工選中但模型漏掉

```text
H − M
```

可能表示模型沒有重視某些真實性或任務資訊。

### 19.3 模型選中但人工未選

```text
M − H
```

可能表示模型偏好：

- 高亮度
- 高對比
- 特定背景
- 過度清晰
- 畫面漂亮但缺少任務資訊
- 與訓練資料相似但不符合實際需求的圖片

每組至少展示 5 張圖片。

---

## 20. 指標解讀

### Overlap

反映模型前 100 張與人工最佳 100 張的集合重疊程度，但不反映共同圖片在模型排名中的位置。

### Jaccard

反映兩個集合的整體相似度。

### AP

反映人工選中圖片在完整候選圖片排名中是否普遍獲得較高分。

### 單張圖片排序分數

用來對每張圖片排序，例如：

- 原始品質分數
- 測試時增強平均分數
- 穩定度修正分數
- 信心修正分數
- 集成品質分數

不可將 AP、Overlap 或 Jaccard 當成單張圖片分數。

---

## 21. 常見錯誤

### 錯誤一：使用人工最佳 100 張訓練模型

這會造成資料洩漏，使最終比較失去意義。

### 錯誤二：把未被人工選中的圖片全部視為壞圖片

未進入人工最佳 100 張不代表品質差，只代表沒有進入最終 100 張集合。

### 錯誤三：將 AP 當成單張圖片分數

AP 是完整排序結果的評估值，不能直接用 AP 對圖片排序。

### 錯誤四：只用前 100 名的 0／1 結果計算 AP

AP 必須使用完整候選圖片集的連續模型分數。

### 錯誤五：三個模型使用不同候選圖片集

人工、VGG16、ResNet18、ResNet50 與所有排序指標必須使用同一候選圖片集。

### 錯誤六：查看人工最佳 100 張後調整模型

人工最佳 100 張只能用於最終比較，不能用來選擇模型權重、超參數或指標係數。

### 錯誤七：只有一組人工參考集合卻使用 mAP

只有一組人工最佳 100 張時應回報 AP。

---

## 22. Overleaf 報告建議結構

```text
1. 作業目的
2. 執行環境
3. 候選圖片集
4. 人工最佳圖片挑選規則
5. 人工最佳 100 張參考集合
6. 品質訓練目標
7. 模型架構與訓練設定
8. 單張圖片排序指標
9. 各模型與各指標前 100 名
10. 與人工最佳 100 張的比較
11. 圖片差異分析
12. 結論與心得
13. AI 工具使用說明
```

報告至少包含：

- 人工挑選規則
- 品質訓練目標來源
- 三個模型設定
- 訓練與驗證曲線
- wandb 圖表
- 各模型與各指標結果表
- Overlap
- Jaccard
- AP
- 人工限定與模型限定圖片
- GitHub 與 wandb 連結

---

## 23. 最終檢查表

### 環境

- [ ] Python 環境可正常啟用
- [ ] PyTorch 與 torchvision 可正常匯入
- [ ] scikit-learn 與 wandb 可正常匯入
- [ ] GPU 或 CPU 裝置已確認
- [ ] `requirements.txt` 或 `environment.yml` 已建立

### 候選圖片與人工參考

- [ ] 候選圖片集已固定
- [ ] 人工挑選規則已在查看模型結果前制定
- [ ] 已人工選出 100 張圖片
- [ ] 已記錄選擇原因
- [ ] 人工最佳 100 張未參與訓練或驗證
- [ ] 未根據模型結果更換人工圖片

### 模型訓練

- [ ] 已說明品質訓練目標來源
- [ ] 已訓練 VGG16
- [ ] 已訓練 ResNet18
- [ ] 已訓練 ResNet50
- [ ] 已使用驗證集選擇模型權重
- [ ] 已使用 wandb 紀錄正式實驗

### 分數與排序

- [ ] 每張候選圖片均有模型分數
- [ ] 已確認分數越高代表越好
- [ ] 已產生三模型原始品質分數前 100 名
- [ ] 已產生信心修正分數
- [ ] 已產生集成品質分數
- [ ] 已預先定義同分時的排序方式

### 最終比較

- [ ] 已計算交集數量
- [ ] 已計算 Overlap
- [ ] 已計算 Jaccard
- [ ] 已使用完整連續分數計算 AP
- [ ] 已分析人工與模型共同選中圖片
- [ ] 已分析人工選中但模型漏掉的圖片
- [ ] 已分析模型選中但人工未選的圖片
- [ ] 未將 AP 當成單張圖片分數

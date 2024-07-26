## Data Preprocess
- **截面Z-Score 标准化（CSZScore）** 对所有数据按日期聚合后进行 Z-Score 处理，主要目的在于保证每日横截面数据的可比性。  
*Z-score = (x - μ)/σ*

- **截面排序标准化（CSRank）** 对所有数据按日期聚合后进行排序处理，将排序结果作为模型输入。此方法主要目的在于排除异常值的影响，但缺点也很明显，丧失了数据间相对大小关系的刻画。

- **数据集整体Z-Score标准化（ZScore）** 截面标准化会使数据损失时序变化信息，而整个数据集做标准化可以将不同日期的相对大小关系也喂入模型进行学习。当然此处需要注意数据泄露问题，我们使用训练集算出均值和标准差后，将其用于整个数据集进行标准化。

- **数据集整体 Minmax 标准化（MinMax）** 相较于 ZScore 标准化而言，MinMax 能使数据严格限制在规定的上下限范围内，且保留了数据间的大小关系。

- **数据集整体 Robust Z-Score 标准化（RobustZScore）** 由于标准差的计算需要对数据均值偏差进行平方运算，会使数据对极值更敏感。而𝑀𝐴𝐷 = M𝑒𝑑𝑖𝑎𝑛(|𝑥 − 𝑀𝑒𝑑𝑖𝑎𝑛(𝑥)|)能有效解决这一问题，使得到的均值标准差指标更加稳健。

针对 GBDT 类模型，我们使用超额收益率作为预测目标，特征和标签均使用RobustZscore 处理。  
针对（时序）神经网络类模型，我们选择超额收益率作为预测目标，特征采用 RobustZScore 方式处理，标签使用 CSRank 处理。

## 全A训练还是成分股训练？

>成分股：  
> - 沪深300指数：包括了沪深两市中市值最大、流动性最好的300只股票，覆盖了各个行业。
> - 上证50指数：包括了上海证券交易所市值最大、流动性最好的50只股票。
> - 中证500指数：包括了沪深两市中市值排名在沪深300指数之外、较为中型的500只股票

在针对沪深 300 的训练过程中，LGBM 和 GRU 展现出了不同的规律。对于更需要大量样本进行投喂训练的 GRU 而言，明显使用全 A 股票会对预测结果
带来明显的提升。而对于具有少量样本就能充分学习的 LightGBM 而言，使用沪深 300 成分股能够有效使模型学到大市值股票的选股逻辑和规律，相较于全 A 而言有明显优势。

对于中证 500 而言，情况略有不同，对于 GRU 模型，同样是大样本量的全 A 训练更具优势。而 LightGBM 模型使用两种成分股样本训练效果已经比较接近，使用成分股训练时，虽然 IC 相关指标略低一些，但多头和多空的最大回撤明显更低，具有在不同市场环境中更稳定的优势。  

由于中证 1000 成分股在市值上已经非常接近全 A 股票的中位数水平，且成分股本身数量较多，因此在中证 1000 上，使用成分股或全 A 训练的预测效果已经非常接近。在样本特征极其相似的情况下，LightGBM 使用全 A 训练效果略微更优。GRU 模型则差异极小，当样本量上升一定水平后，继续扩大样本量所带来的提升已经比较有限。

## 一次性、滚动还是扩展训练
> 1. 完整一次性训练  
>  使用固定的时间区间数据集进行一次性训练。数据划分：将整个时间区间的数据划分为训练集、验证集和测试集。例如，使用前70%的数据作为训练集，接下来的15%作为验证集，最后的15%作为测试集。  

> 2. 向前滚动训练  
> 这种方法通过逐年滚动数据窗口来训练模型，使得模型能够不断更新和改进。  
> 初始划分：选择一个初始时间段作为训练集。  
> 滚动训练：第一年：使用初始时间段的数据训练模型，在下一时间段的数据上验证和测试; 第二年：向前滚动时间窗口，使用新的时间段数据（包括上一年的数据）训练模型，并在下一时间段的数据上验证和测试。重复上述步骤，直到覆盖整个数据集。

> 3. 扩展训练  
> 这种方法保持训练集的起始时间不变，逐年扩展训练集的数据量。  
> 初始划分：选择一个初始时间段作为训练集。  
> 扩展训练：第一年：使用初始时间段的数据训练模型，在下一时间段的数据上验证和测试; 第二年：保持训练集的起始时间不变，扩展训练集，加入新一年的数据，重新训练模型，并在下一时间段的数据上验证和测试。重复上述步骤，直到覆盖整个数据集。

对于 LightGBM 而言，一次性训练效果明显更优，无论从 IC、多空相关指标来看，均要好于滚动或扩展训练集的方式。  
而对于 GRU 而言，三种训练效果差距缩窄，一次性训练的预测结果主要在回撤控制上具有一定优势。  

由于训练过程中为了避免过拟合并找到合适的参数，我们都会设置一定的早停轮数 N，验证集上的损失大小若连续 N 轮没有下降就停止训练。因此在滚动或扩展训练的情况下，验证集的不断更新会使模型的早停标准跟随市场交易逻辑的变化而变化，在碰到极端市场行情时，或过去两年的交易逻辑在当年不再适用时，可能导致测试集上效果出现较大下滑，这在更容易过拟合的 LightGBM 模型上会更加明显。

## 批次和损失函数
1. 均方误差（MSE）  
均方误差（Mean Squared Error，MSE）是衡量模型预测值与真实值之间差异的常用指标。MSE是预测误差的平方的平均值。

2. IC  
IC（Information Coefficient）通常指Pearson相关系数，用于衡量两个变量之间的线性关系。IC = corr(y_pred, y_true)

3. Spearman秩相关系数（RankIC）  
RankIC（Rank Information Coefficient）通常指Spearman秩相关系数，用于衡量两个变量之间的单调关系（无论是线性还是非线性）。RankIC = corr(R(y_pred), R(y_true))


- TotalBatch-TotalLoss: 不分交易日划分 Batch 且整个样本内计算损失函数  
- DailyBatch-TotalLoss: 按照交易日划分 Batch 且整个样本内计算损失函数  
- DailyBatch-DailyLoss: 按照交易日划分 Batch 且日度计算损失函数后求均值

# LightGBM

## lightgbm.Dataset
Bases: object  
Dataset in LightGBM.  
Constract Dataset.  

### Parameters:
- **data** *(string, numpy array or scipy.sparse)* – Data source of Dataset. If string, it represents the path to txt file.
- **label** *(list, numpy 1-D array or None) (default=None)* – Label of the data.
- **max_bin** *(int or None) (default=None)* – Max number of discrete bins for features. If None, default value from parameters of CLI-version will be used.
- **reference** *(Dataset or None) (default=None)* – If this is Dataset for validation, training data should be used as reference.
- **weight** *(list, numpy 1-D array or None) (default=None)* – Weight for each instance.
- **group** *(list, numpy 1-D array or None) (default=None)* – Group/query size for Dataset.
- **init_score** *(list, numpy 1-D array or None) (default=None)* – Init score for Dataset.
- **silent** *(bool) (default=False)* – Whether to print messages during construction.
- **feature_name** *(list of strings or 'auto') (default="auto")* – Feature names. If ‘auto’ and data is pandas DataFrame, data columns names are used.
- **categorical_feature** *(list of strings or int, or 'auto')(default="auto")* – Categorical features. If list of int, interpreted as indices. If list of strings, interpreted as feature names (need to specify feature_name as well). If ‘auto’ and data is pandas DataFrame, pandas categorical columns are used.
- **params** *(dict or None) (default=None)* – Other parameters.
- **free_raw_data** *(bool) (default=True)* – If True, raw data is freed after constructing inner Dataset

### Method
#### **construct**()
Lazy init.  
Returns: self  
Return type: Dataset  

#### **create_valid**(data, label=None, weight=None, group=None, init_score=None, silent=False, params=None)
Create validation data align with current Dataset.
- **data** *(string, numpy array or scipy.sparse)* – Data source of Dataset. If string, it represents the path to txt file.
- **label** *(list or numpy 1-D array) (default=None)* – Label of the training data.
- **weight** *(list, numpy 1-D array or None) (default=None)* – Weight for each instance.
- **group** *(list, numpy 1-D array or None) (default=None)* – Group/query size for Dataset.
- **init_score** *(list, numpy 1-D array or None) (default=None)* – Init score for Dataset.
- **silent** *(bool) (default=False)* – Whether to print messages during construction.
- **params** *(dict or None) (default=None)* – Other parameters.  

Returns: self  
Return type: Dataset

#### **get_field**(field_name)
Get property from the Dataset.

- **field_name** (string) – The field name of the information.  

Returns: info – A numpy array with information from the Dataset.  
Return type: numpy array

#### **get_group**()
Get the group of the Dataset.  

Returns: group – Group size of each group.  
Return type: numpy array  

#### **get_init_score**()
Get the initial score of the Dataset.  

Returns: init_score – Init score of Booster.  
Return type: numpy array

#### **get_label**()
Get the label of the Dataset.  

Returns: label – The label information from the Dataset.  
Return type: numpy array  

#### **get_ref_chain**(ref_limit=100)
Get a chain of Dataset objects, starting with r, then going to r.reference if exists, then to r.reference.reference, etc. until we hit ref_limit or a reference loop.  

- **ref_limit** *(int) (default=100)* – The limit number of references.  

Returns: ref_chain – Chain of references of the Datasets.  
Return type: set of Dataset  

#### **get_weight()**
Get the weight of the Dataset. 

Returns: weight – Weight for each data point from the Dataset.  
Return type: numpy array  

#### **num_data()**
Get the number of rows in the Dataset.  

Returns: number_of_rows – The number of rows in the Dataset.  
Return type: int

#### **num_feature()**
Get the number of columns (features) in the Dataset.  

Returns: number_of_columns – The number of columns (features) in the Dataset.  
Return type: int  

#### **save_binary**(filename)
Save Dataset to binary file.  

- **filename** *(string)* – Name of the output file.

No returns

#### **set_categorical_feature**(categorical_feature)
Set categorical features.

- **categorical_feature** *(list of int or strings)* – Names or indices of categorical features.

No returns

#### **set_feature_name**(feature_name)
Set feature name.

- **feature_name** *(list of strings)* – Feature names.

No returns

#### **set_field**(field_name, data)
Set property into the Dataset.

- **field_name** *(string)* – The field name of the information.
- **data** *(list, numpy array or None)* – The array of data to be set.

No returns

#### **set_group**(group)
Set group size of Dataset (used for ranking).

- **group** *(list, numpy array or None)* – Group size of each group.

No returns

#### **set_init_score**(init_score)
Set init score of Booster to start from.

- **init_score** *(list, numpy array or None)* – Init score for Booster.

No returns

#### **set_label**(label)
Set label of Dataset

- **label** *(list, numpy array or None)* – The label information to be set into Dataset.

No returns

#### **set_reference**(reference)
Set reference Dataset.

- **reference** *(Dataset)* – Reference that is used as a template to consturct the current Dataset.

No returns

#### **set_weight**(weight)
Set weight of each instance.

- **weight** *(list, numpy array or None)* – Weight to be set for each data point.

No returns

#### **subset**(used_indices, params=None)
Get subset of current Dataset.

- **used_indices** *(list of int)* – Indices used to create the subset.
- **params** *(dict or None) (default=None)* – Other parameters.  

Returns: subset – Subset of the current Dataset.   
Return type: Dataset  

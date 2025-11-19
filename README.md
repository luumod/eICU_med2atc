> If you need any data, please open an issue and I will handle it promptly.

# eICU_med2atc    

The goal of this repository is to solve the problem that medications in the `medications.csv` table of the **`eICU`** dataset cannot be converted to ATC level 3 codes, which prevents coarse-grained analyses and discussions.

> The `medications.csv` table contains: `drugname` and `drughiclseqno`  
> * `drugname` includes redundant information such as drug ingredients, dosage, and dosage form.  
> * `drughiclseqno` is the drug’s HICL code, but HICL codes cannot be directly converted to RXCUI, NDC, ATC, or other coding systems.

## Conversion Strategy

1. Perform fuzzy matching between `drugname` in `medications.csv` and the `RXSTRING` field in the `rxnorm_mappings.txt` file to obtain the corresponding RXCUI codes.
2. RXCUI → NDC / ATC → drugbank_id → drug_name → moldb_smiles

Implementation steps:

1. In `1_get_unique_name.ipynb`, obtain the unique original drug names.
2. In `2_data_matching.py`, perform fuzzy matching to obtain the mapping from original `drugname` to RXCUI.
3. In `3_mappings.ipynb`, based on RXCUI, build mappings to all common drug coding systems.
4. In `4_med_mappings.ipynb`, generate the mapped `medications_mappings.csv` file.
5. In `5_processing_eicu.py`, implement the processing of the eICU dataset for **medication recommendation**, enabling coarse-grained (ATC level 3) analysis.

The files in the `output` directory are all the results produced by `5_processing_eicu.py` and can be used directly.

============================ Chinese Version / 中文版本 ===========================
# eICU_med2atc

此仓库实现的目标是：解决 `eICU` 数据集中的 medications.csv 表中的药物无法转换为 ATC3 级别的代码，导致无法从粗粒度的角度对其进行一系列的讨论。
> medications.csv 表中含有：drugname 和 drughiclseqno
> * drugname 包含有药物成分、剂量、剂型等冗余信息。
> * drughiclseqno是药物的 HICL 编码，但是HICL编码无法直接转换为 RXCUI、NDC、ATC等编码。

## 转换思路

1. 将 medications.csv 表中的 drugname 与 `rxnorm_mappings.txt` 文件中的 RXSTRING 内容进行模糊匹配，得到对应的 RXCUI 编码.
2. RXCUI -> NDC / ATC -> drugbank_id -> drug_name -> moldb_smiles

实现步骤：
1. `1_get_unique_name.ipynb` 文件，得到了独一无二的药物原始名称
2. `2_data_matching.py` 文件，实现了模糊匹配，得到了原始的 drugname 到 RXCUI 的映射
3. `3_mappings.ipynb` 文件，根据RXCUI，实现了所有常见药物编码的映射
4. `4_med_mappings.ipynb` 文件，获得了映射后的 medications_mappings.csv 文件
5. `5_processing_eicu.py` 文件，实现了**药物推荐**中对于eICU数据集的操作，并且可以对其进行粗粒度(ATC3)级别的分析

output 里面的文件是所有经过 `5_processing_eicu.py` 处理后得到的输出结果，可以直接使用

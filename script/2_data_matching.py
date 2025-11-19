import pandas as pd
import re
from difflib import SequenceMatcher
from collections import defaultdict
import numpy as np
from tqdm import tqdm

# -------------------- 按权重匹配 ----------------------
# 成分名称匹配：提取主要药物成分 (权重: 60%)
# 剂量匹配：考虑剂量强度和单位（权重: 20%）
# 剂型匹配：片剂、注射液、溶液等（权重: 10%）
# 整体字符串相似度 (权重: 10%)
# -----------------------------------------------------

class DrugNameMatcher:
    def __init__(self):
        # 常见的剂型缩写映射
        self.dose_forms = {
            'TABS': 'tablet', 'TBEC': 'tablet', 'CAPS': 'capsule',
            'SOLN': 'solution', 'SUSP': 'suspension', 'OINT': 'ointment',
            'CREA': 'cream', 'SOLR': 'solution', 'SUPP': 'suppository',
            'SYRP': 'syrup', 'INJ': 'injection', 'AERO': 'aerosol'
        }
        
        # 给药途径缩写
        self.routes = {
            'PO': 'oral', 'IV': 'intravenous', 'IM': 'intramuscular',
            'SC': 'subcutaneous', 'SL': 'sublingual', 'TOP': 'topical',
            'OP': 'ophthalmic', 'OT': 'otic', 'RE': 'rectal',
            'IJ': 'injection', 'IN': 'intranasal'
        }
    
    def preprocess_drug_name(self, name):
        """预处理药物名称，提取关键信息"""
        if pd.isna(name):
            return None
        
        name = str(name).upper().strip()
        
        # 提取主要成分（通常在最前面）
        # 移除剂量信息但保留用于后续匹配
        parts = name.split()
        
        # 识别并提取成分名
        ingredient = []
        dose_info = []
        form_info = []
        route_info = []
        
        i = 0
        while i < len(parts):
            part = parts[i]
            
            # 检查是否是数字或剂量单位
            if re.match(r'^\d+(\.\d+)?$', part) or part in ['MG', 'ML', 'G', 'MCG', 'UNIT', '%']:
                dose_info.append(part)
            # 检查是否是剂型
            elif part in self.dose_forms:
                form_info.append(self.dose_forms[part])
            # 检查是否是给药途径
            elif part in self.routes:
                route_info.append(self.routes[part])
            # 否则认为是成分名的一部分
            elif not re.match(r'^[\d\.\-/]+$', part):
                ingredient.append(part)
            
            i += 1
        
        return {
            'ingredient': ' '.join(ingredient),
            'dose': ' '.join(dose_info),
            'form': ' '.join(form_info),
            'route': ' '.join(route_info),
            'original': name
        }
    
    def normalize_rxnorm_string(self, rxstring):
        """标准化RxNorm字符串"""
        if pd.isna(rxstring):
            return ''
        return str(rxstring).lower().strip()
    
    def extract_core_ingredient(self, text):
        """提取核心成分名称（去除盐、酯等）"""
        # 移除常见的盐和化合物后缀
        suffixes = [
            r'\s+sodium', r'\s+potassium', r'\s+calcium', r'\s+chloride',
            r'\s+sulfate', r'\s+succinate', r'\s+tartrate', r'\s+citrate',
            r'\s+acetate', r'\s+hcl', r'\s+hydrochloride', r'\s+maleate',
            r'\s+fumarate', r'\s+mesylate', r'\s+na', r'\s+pf',
            r'\s+\(human\)', r'\s+ec'
        ]
        
        core = text.lower()
        for suffix in suffixes:
            core = re.sub(suffix, '', core)
        
        return core.strip()
    
    def calculate_similarity(self, drug_info, rxnorm_string):
        """计算相似度得分"""
        if not drug_info or not rxnorm_string:
            return 0.0
        
        rxnorm_lower = rxnorm_string.lower()
        
        # 提取核心成分进行匹配
        core_ingredient = self.extract_core_ingredient(drug_info['ingredient'])
        
        # 1. 成分名称匹配 (权重: 60%)
        ingredient_score = 0.0
        if core_ingredient:
            # 尝试完整匹配
            if core_ingredient in rxnorm_lower:
                ingredient_score = 1.0
            else:
                # 分词匹配
                ingredient_words = core_ingredient.split()
                matched_words = sum(1 for word in ingredient_words 
                                  if len(word) > 2 and word in rxnorm_lower)
                if ingredient_words:
                    ingredient_score = matched_words / len(ingredient_words)
        
        # 2. 剂量匹配 (权重: 20%)
        dose_score = 0.0
        if drug_info['dose']:
            # 提取数字进行匹配
            dose_numbers = re.findall(r'\d+(?:\.\d+)?', drug_info['dose'])
            if dose_numbers:
                matched_doses = sum(1 for num in dose_numbers if num in rxnorm_lower)
                dose_score = matched_doses / len(dose_numbers)
        
        # 3. 剂型匹配 (权重: 10%)
        form_score = 0.0
        if drug_info['form']:
            form_keywords = ['tablet', 'capsule', 'solution', 'injection', 
                           'cream', 'ointment', 'spray', 'suspension']
            for keyword in form_keywords:
                if keyword in drug_info['form'] and keyword in rxnorm_lower:
                    form_score = 1.0
                    break
        
        # 4. 整体字符串相似度 (权重: 10%)
        string_similarity = SequenceMatcher(None, 
                                          drug_info['ingredient'].lower(), 
                                          rxnorm_lower).ratio()
        
        # 综合得分
        total_score = (ingredient_score * 0.6 + 
                      dose_score * 0.2 + 
                      form_score * 0.1 + 
                      string_similarity * 0.1)
        
        return total_score
    
    def find_best_match(self, drug_name, rxnorm_df):
        """为单个药物名称找到最佳匹配"""
        drug_info = self.preprocess_drug_name(drug_name)
        
        if not drug_info:
            return None
        
        best_score = 0.0
        best_rxcui = None
        
        # 计算每个RxNorm条目的相似度
        for idx, row in tqdm(rxnorm_df.iterrows(), desc=f'Matching "{drug_name}"', total=len(rxnorm_df)):
            rxnorm_string = row['RXSTRING']
            score = self.calculate_similarity(drug_info, rxnorm_string)
            
            if score > best_score:
                best_score = score
                best_rxcui = row['RXCUI']
        
        return best_rxcui if best_score > 0.3 else None  # 设置最低阈值
    
    def match_all_drugs(self, drugs_file, rxnorm_file, output_file='/home/qlunlp/ylh/database/tmp/drug_rxcui_mapping.txt'):
        """匹配所有药物并输出结果"""
        print("Loading data...")
        
        # 读取数据
        drugs_df = pd.read_csv(drugs_file)
        rxnorm_df = pd.read_csv(rxnorm_file)
        
        print(f"Loaded {len(drugs_df)} unique drugs")
        print(f"Loaded {len(rxnorm_df)} RxNorm entries")
        
        # 创建映射字典
        mapping_dict = {}
        
        print("\nMatching drugs to RxNorm codes...")
        for idx, row in drugs_df.iterrows():
            drug_name = row['drugname']
            
            if idx % 100 == 0:
                print(f"Progress: {idx}/{len(drugs_df)}")
            
            rxcui = self.find_best_match(drug_name, rxnorm_df)
            print(f'========= Process No {idx}: Drug="{drug_name}" -> RXCUI="{rxcui} ========="')
            mapping_dict[drug_name] = rxcui
        
        # 输出结果
        print(f"\nSaving results to {output_file}...")
        
        # 保存为易读格式
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("Drug Name\tRXCUI\n")
            for drug, rxcui in mapping_dict.items():
                f.write(f"{drug}\t{rxcui if rxcui else 'NO_MATCH'}\n")
        
        # 统计信息
        matched = sum(1 for v in mapping_dict.values() if v is not None)
        print(f"\nMatching completed!")
        print(f"Total drugs: {len(mapping_dict)}")
        print(f"Matched: {matched} ({matched/len(mapping_dict)*100:.1f}%)")
        print(f"Unmatched: {len(mapping_dict) - matched}")
        
        return mapping_dict


def main():
    matcher = DrugNameMatcher()
    
    # 执行匹配
    mapping = matcher.match_all_drugs(
        '/home/qlunlp/ylh/database/tmp/unique_drugname.csv',
        '/home/qlunlp/ylh/database/tmp/processed_rxnorm_mappings.csv'
    )
    
    # 显示前10个匹配结果示例
    print("\nSample matches:")
    for i, (drug, rxcui) in enumerate(list(mapping.items())[:10]):
        print(f"{drug} -> {rxcui}")

if __name__ == "__main__":
    main()
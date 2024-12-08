import pandas as pd
import numpy as np
import re
import os
import matplotlib.pyplot as plt
from collections import defaultdict

class RuleExtractor:
    
    def __init__(self, model) -> None:
        self.model = model.dt # 민철: max_depth_dt --> max_param_dt 수정
        # self.model = model.max_depth_dt
        self.feature_names = model.feature_names
        self.class_names = model.class_names
        self.route = {}
    

    def get_route(self):        
        model = self.model
        leaf_ids = model.tree_.children_left == -1
        leaf_node_indicies = np.where(leaf_ids)[0]

        feature = model.tree_.feature
        threshold = model.tree_.threshold
        n_nodes = model.tree_.node_count
        children_left = model.tree_.children_left
        children_right = model.tree_.children_right
        
        route = {}
        
        for leaf_idx in leaf_node_indicies:
            path = []
            node_idx = leaf_idx
            while node_idx != 0:
                parent_idx = -1
                for j in range(n_nodes):
                    if children_left[j] == node_idx or children_right[j] == node_idx:
                        parent_idx = j
                        break
                if parent_idx == -1:
                    break
                if node_idx == children_left[parent_idx]:
                    path.append(f"{self.feature_names[feature[parent_idx]]} <= {threshold[parent_idx]}")
                else:
                    path.append(f"{self.feature_names[feature[parent_idx]]} > {threshold[parent_idx]}")
                node_idx = parent_idx
            path.reverse()
            route[leaf_idx] = path
        
        return route, leaf_node_indicies
    

    def extract_rule(self, segment_num, strict=True):
        if len(self.route) == 0 or self.model.n_classes_ == 2:
            self.route, self.leaf_node_indicies = self.get_route()

        leaf_node_values = self.model.tree_.value[self.leaf_node_indicies]
        leaf_node_classes = self.model.classes_[np.argmax(leaf_node_values, axis=2)]

        segment_rule_list = []
        leaf_node_class = np.squeeze(leaf_node_classes)
        segment_leaf_node_indicies = self.leaf_node_indicies[np.where(leaf_node_class==segment_num)]
        
        for leaf_idx in segment_leaf_node_indicies:
            segment_rule_list.append(' [AND] '.join(self.route[leaf_idx]))
        rule_str = ' [OR] \n'.join(segment_rule_list)

        segment_rule = re.sub(r'\d+\.\d+', lambda x: str(round(float(x.group()), 3)), rule_str)

        if strict:
            segment_rule = self.apply_strict_rule(segment_rule)
        
        return segment_rule
    

    def apply_strict_rule(self, segment_rule):
        or_list = segment_rule.split('[OR]')

        ls = [[r.split() for r in l.split('[AND]')] for l in or_list]
        new_ls = []
        for node in ls:
            new_ls.append([])
            rule_dic = {}
            for r in node:
                key_name = f'{r[0]} {r[1]} '
                rule_dic[key_name] = rule_dic.get(key_name, [])
                rule_dic[key_name].append(float(r[2]))    
            
            for k, v in rule_dic.items():
                if k in '>':
                    rule_dic[k] = max(v)
                else:
                    rule_dic[k] = min(v)
            
            for k, v in rule_dic.items():
                new_rule = k.split()
                new_rule.append(str(v))
                new_ls[-1].append(new_rule)
        
        node_list = []
        for node in new_ls:
            rule_list = []
            for rule in node:
                rule_list.append(''.join(rule))
            node_rule = ' [AND] '.join(rule_list)
            node_list.append(node_rule)
        segment_rule = ' [OR]\n'.join(node_list)
        
        return segment_rule
    
    def save_rule(self, rule, save_file):
        with open(save_file, 'w') as f:
            f.writelines(rule)
    

class RuleStatisticsCalculator:
    
    def __init__(self, model, pivot_df):
        self.model = model.dt
        self.pivot_df = pivot_df
        self.k = self.model.n_classes_


    def calculate_statistics(self, save_dir, file_name):
        segment_dict = defaultdict(list)
        genre_dict = defaultdict(list)

        pred = self.model.predict(self.pivot_df)
        pred_df = pd.DataFrame(pred).value_counts().sort_index()
        for i in range(self.k):
            with open(save_dir + f'/{file_name}{i}.txt', 'r') as f:
                full_text = f.read()

            or_list = full_text.split('[OR]')

            depth_list = []
            for or_rule in or_list:
                and_list = or_rule.split('[AND]')
                depth_list.append(len(and_list))
                for and_rule in and_list:
                    and_rule = and_rule.strip()
                    
                    if '<=' in and_rule:
                        key, value = and_rule.split('<=')
                        value = float(value)
                        if value not in genre_dict[key]:
                            genre_dict[key].append(value)

            segment_dict['user_num'].append(pred_df[i])
            segment_dict['rule_num'].append(len(or_list))
            segment_dict['depth_min'].append(min(depth_list))
            segment_dict['depth_max'].append(max(depth_list))
            segment_dict['depth_mean'].append(np.mean(depth_list))
        
        self.segment_dict = segment_dict
        self.genre_dict = genre_dict

        return round(pd.DataFrame(segment_dict), 2)


    def plot_rule_distribution(self):
        visual_df = self.pivot_df.copy()

        for genre, rule_list in self.genre_dict.items():

            genre_mean = visual_df[genre].mean()
            genre_std = visual_df[genre].std()

            visual_cond = visual_df[genre] >= genre_mean + genre_std
            visual_df.loc[visual_cond, genre] = genre_mean + genre_std
            
            min_rule = sorted(rule_list)[0]
            count_zero = sum(visual_df[genre] == 0)
            count_min = sum(visual_df[genre] < min_rule)

            print(f'0인 개수: {count_zero}')
            print(f'{min_rule} 이하인 개수: {count_min}')
            print(f'0인 비율: {(count_zero/count_min)*100:.3f} %')

            plt.figure(figsize=(20,6))
            visual_df[genre].plot.hist(bins=50)
            plt.title(genre)
            
            for rule in rule_list:
                plt.axvline(rule, color='red', linestyle='--')
            plt.show()
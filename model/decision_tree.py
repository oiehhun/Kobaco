from sklearn.cluster import KMeans, AgglomerativeClustering
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split


class DecisionTree:

    def __init__(self, pivot_df, embedding, validation=False, k=15, cluster='kmeans') -> None:
        self.df = pivot_df
        self.embedding = embedding

        self.alpha = 0.0
        self.k = k
        self.validation = validation

        X = np.array(pivot_df.iloc[:,:])
        if cluster=='kmeans':
            Y = np.array(self.kmeans(k))
        else:
            Y = np.array(self.agglomerative_clustering(k))

        if validation:
            self.x_train, self.x_valid, self.y_train, self.y_valid = train_test_split(X, Y, test_size=0.1, random_state=42)
        else:
            self.x_train, self.y_train = X, Y
        
        self.X = self.x_train
        self.Y = self.y_train

        self.feature_names = self.df.columns.tolist()[:]
        self.class_names = [str(i) for i in list(self.Y)]


    def kmeans(self, k):
        kmeans = KMeans(n_clusters=k, random_state=42)
        embedding = self.embedding[self.df.index.values.tolist()].copy()
        kmeans.fit(embedding)
        return kmeans.predict(embedding)
    

    def agglomerative_clustering(self, k):
        agg_cluster = AgglomerativeClustering(n_clusters=k, affinity='euclidean', linkage='ward')
        embedding = self.embedding[self.df.index.values.tolist()].copy()
        return agg_cluster.fit_predict(embedding)
    

    def kmeans_target(self, tgt_n): 
        self.tgt_n = tgt_n
        self.X = self.x_train
        self.Y = (self.y_train==self.tgt_n).astype(int)
    

    def make_dt(self, random_state=42, **params):
        if self.validation:
            self.X = self.x_train
            if len(np.unique(self.Y)) == 2:
                self.Y = (self.y_train==self.tgt_n).astype(int)
            else:
                self.Y = self.y_train
        model = DecisionTreeClassifier(random_state=random_state, **params)
        model = model.fit(self.X, self.Y)
        return model
    

    def get_valid_score(self, model, scoring, average='macro'):
        self.X = self.x_valid
        if model.n_classes_ == 2:
            self.Y = (self.y_valid==self.tgt_n).astype(int)
        else:
            self.Y = self.y_valid
        return self.get_score(model, scoring, average)


    def get_score(self, model, scoring, average='macro'):
        prediction = model.predict(self.X)
        if model.n_classes_ == 2:   
            average='binary'
        
        recall = recall_score(self.Y, prediction, average=average)
        precision = precision_score(self.Y, prediction, average=average)
        f1 = f1_score(self.Y, prediction, average=average)
        accuracy = accuracy_score(self.Y, prediction)

        if scoring == 'recall': return recall
        elif scoring == 'precision': return precision
        elif scoring == 'f1_score': return f1
        elif scoring == 'accuracy': return accuracy
        else:   return recall, precision, f1, accuracy


    def get_proper_param(self, search_space, target_score=0.6, scoring='f1_score', check_param='max_depth', **params):

        self.max_param = self.make_dt(**params).get_params()[check_param]
        print(f'최대 score의 {check_param} = {self.max_param}')
        
        passed_param = []
        score_list = []
        model_params_list = []
        i = 0
        
        for param in search_space:
            print(f'testing {check_param}: {param}...', end='\r')
            model = self.make_dt(**{check_param:param}, **params)
            score = self.get_score(model, scoring)
            now_params = model.get_params()
            
            passed_param.append(param)
            score_list.append(score)
            model_params_list.append(now_params)

            if (score < target_score) and (i > 0):
                print(f'{check_param} {param}에서 target_socre {target_score}을 달성하지 못하여 종료합니다.')
                break
            i += 1
            
        self.max_param = passed_param[-2]
        self.max_score = score_list[-2]
        
        # plt.plot(passed_param,score_list)
        # plt.xlabel(check_param)
        # plt.ylabel(scoring)
        # # max_score 수평선 그리기
        # plt.hlines(y=target_score, xmin=passed_param[-1], xmax=passed_param[0], colors='r')
        # plt.show()

        self.dt = self.make_dt(**{check_param:passed_param[-2]}, **params)
        # 해당 모델의 파라미터, 스코어 반환
        return model_params_list[-2], score_list[-2]  
    

    def get_all_depth(self, scoring='all', visualize=True):
        score_list = []
        val_score_list = []


        for depth in range(1, self.make_dt().get_depth()):
            model = self.make_dt(max_depth = depth)

            score = self.get_score(model, scoring)
            score_list.append(score)

            if self.validation:
                val_score = self.get_valid_score(model, scoring)
                val_score_list.append(val_score)

            if visualize:   
                self.visualize_tree(model)
        
        if self.validation:
            return score_list, val_score_list
        else:
            return score_list
    

    def visualize_tree(self, model):
        plt.figure(figsize=(70, 50))
        class_names = [str(i) for i in np.unique(self.Y)]
        plot_tree(model, feature_names=self.feature_names, class_names=class_names, filled=True)
        plt.show()
from sklearnex import patch_sklearn

patch_sklearn()

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from imblearn.over_sampling import SMOTE
from scipy.stats import mode
from skin_lesion_cad.data import TOP_FEATURES
from skin_lesion_cad.data.BOVW import DescriptorsTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, cohen_kappa_score
from sklearn.svm import SVC

classifiers = [(SVC(kernel='rbf', C=1, random_state=42, probability=True, class_weight='balanced'), 'full', 'svc_full'),
               (RandomForestClassifier(bootstrap=False, max_depth=10, class_weight='balanced'), 'smote', 'rf_smote'),
               (SVC(C=10, class_weight='balanced', gamma=0.1, probability=True), 'rf_fs', 'svc_rf_fs'),
               (RandomForestClassifier(bootstrap=False, max_depth=20, class_weight='balanced'), 'pca', 'rf_pca'),
               (SVC(kernel='rbf', C=1, probability=True, class_weight='balanced'), 'smote', 'svc_smote')]

class EnsemblingClssfifier:
    def __init__(self, class_data: list, ensembling=None) -> None:
        self.class_data = class_data
        self.model_names = [x[2] for x in class_data]
        self.smote = SMOTE(random_state=42, k_neighbors=5, n_jobs=-1, sampling_strategy='not majority')
        self.tf_fs_set = TOP_FEATURES
        self.transformer = DescriptorsTransformer(None)
        self.pca = PCA(n_components=0.95)
        self.ensembling = ensembling
        
    def fit(self, X, y, columns_X):
        """X - raw 442 shape array with features
        columngs_X - list of columns names used to make X
        only once pca per class  data"""
        # normalize data
        X = self.transformer.fit_transform(X)
        
        for model, dataset, name in self.class_data:
            if dataset == 'full':
                model.fit(X, y)
            elif dataset == 'smote':
                X_smote, y_smote = self.smote.fit_resample(X, y)
                model.fit(X_smote, y_smote)
            elif dataset == 'rf_fs':
                X_rf = X[:,np.isin(columns_X, self.tf_fs_set)]
                model.fit(X_rf, y)           
            elif dataset == 'pca':
                X_pca = self.pca.fit_transform(X)
                model.fit(X_pca, y)
            else:
                raise ValueError('Unknown dataset')
        
        if 'stacking' in self.ensembling:
            self.meta_model = LogisticRegression()
            predictions, predicitons_proba = self.predict(X, columns_X, False)
            predictions = {k: v for k, v in predictions.items() if k in self.model_names}
            predicitons_proba = {k: v for k, v in predicitons_proba.items() if k in self.model_names}
            self.meta_model.fit(np.hstack(predicitons_proba.values()), y)
        
    def predict(self, X, columns_X, with_meta=True):
        X = self.transformer.fit_transform(X)
        
        predictions = dict()
        predicitons_proba = dict()
        
        for model, dataset, name in self.class_data:
            if dataset == 'full':
                predictions[name] = model.predict(X)
                predicitons_proba[name] = model.predict_proba(X)

            elif dataset == 'smote':
                predictions[name] = model.predict(X)
                predicitons_proba[name] = model.predict_proba(X)
            
            elif dataset == 'rf_fs':
                X_rf = X[:,np.isin(columns_X, self.tf_fs_set)]
                predictions[name] = model.predict(X_rf)
                predicitons_proba[name] = model.predict_proba(X_rf)
            elif dataset == 'pca':
                X_pca = self.pca.transform(X)
                predictions[name] = model.predict(X_pca)
                predicitons_proba[name] = model.predict_proba(X_pca)
            else:
                raise ValueError('Unknown dataset')
        
        if 'stacking' in self.ensembling and with_meta:
            predictions['stacking'] = self.meta_model.predict(np.hstack(predicitons_proba.values()))
            predicitons_proba['stacking'] = self.meta_model.predict_proba(np.hstack(predicitons_proba.values()))
            
        if 'hard_voting' in self.ensembling:
            predictions['hard_voting'] = mode(np.stack([v for k,v in predictions.items() if k in self.model_names]), axis=0)[0][0]
            predicitons_proba['hard_voting'] = None
            
        if 'soft_voting' in self.ensembling:
            predicitons_proba['soft_voting'] = np.asarray([v for k, v in predicitons_proba.items() if k in self.model_names]).mean(axis=0)
            predictions['soft_voting'] = predicitons_proba['soft_voting'].argmax(axis=1)
            
        return predictions, predicitons_proba
    
    def evaluate(self, X, y, columns_X, plot=True):
        predictions, predicitons_proba = self.predict(X, columns_X)
        kappa_scores = {name: cohen_kappa_score(y, pred) for name, pred in predictions.items()}
        balacc_scores = {name: balanced_accuracy_score(y, pred) for name, pred in predictions.items()}
        
        if plot:
            fig, axs = plt.subplots(1, 2, figsize=(18, 6))

            # plot bar plot of the dict
            sns.barplot(y=list(kappa_scores.values()), x=list(kappa_scores.keys()), ax=axs[0])
            axs[0].set_title('Kappa scores')
            axs[0].bar_label(axs[0].containers[0])

            sns.barplot(y=list(balacc_scores.values()), x=list(balacc_scores.keys()), ax=axs[1])
            axs[1].set_title('Balanced Accuracy scores')
            axs[1].bar_label(axs[1].containers[0])


            plt.show()
        return kappa_scores, balacc_scores

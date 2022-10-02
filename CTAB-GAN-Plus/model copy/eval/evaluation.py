import numpy as np
import pandas as pd 
from sklearn import metrics
from sklearn import model_selection
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import svm,tree
from sklearn.ensemble import RandomForestClassifier
from dython.nominal import compute_associations
from scipy.stats import wasserstein_distance
from scipy.spatial import distance
import warnings

warnings.filterwarnings("ignore")

def supervised_model_training(x_train, y_train, x_test, 
                              y_test, model_name):
  
     
  if model_name == 'lr':
    model  = LogisticRegression(random_state=42,max_iter=500) 
  elif model_name == 'svm':
    model  = svm.SVC(random_state=42,probability=True)
  elif model_name == 'dt':
    model  = tree.DecisionTreeClassifier(random_state=42)
  elif model_name == 'rf':      
    model = RandomForestClassifier(random_state=42)
  elif model_name == "mlp":
    model = MLPClassifier(random_state=42,max_iter=100)
  
  model.fit(x_train, y_train)
  pred = model.predict(x_test)

  if len(np.unique(y_train))>2:
    predict = model.predict_proba(x_test)        
    acc = metrics.accuracy_score(y_test,pred)*100
    auc = metrics.roc_auc_score(y_test, predict,average="weighted",multi_class="ovr")
    f1_score = metrics.precision_recall_fscore_support(y_test, pred,average="weighted")[2]
    return [acc, auc,f1_score] 

  else:
    predict = model.predict_proba(x_test)[:,1]    
    acc = metrics.accuracy_score(y_test,pred)*100
    auc = metrics.roc_auc_score(y_test, predict)
    f1_score = metrics.precision_recall_fscore_support(y_test,pred)[2].mean()
    return [acc, auc,f1_score] 


def get_utility_metrics(real_path,fake_paths,scaler="MinMax",classifiers=["lr","dt","rf","mlp"],test_ratio=.20):

    data_real = pd.read_csv(real_path).to_numpy()
    data_dim = data_real.shape[1]

    data_real_y = data_real[:,-1]
    data_real_X = data_real[:,:data_dim-1]
    X_train_real, X_test_real, y_train_real, y_test_real = model_selection.train_test_split(data_real_X ,data_real_y, test_size=test_ratio, stratify=data_real_y,random_state=42) 

    if scaler=="MinMax":
        scaler_real = MinMaxScaler()
    else:
        scaler_real = StandardScaler()
        
    scaler_real.fit(data_real_X)
    X_train_real_scaled = scaler_real.transform(X_train_real)
    X_test_real_scaled = scaler_real.transform(X_test_real)

    all_real_results = []
    for classifier in classifiers:
      real_results = supervised_model_training(X_train_real_scaled,y_train_real,X_test_real_scaled,y_test_real,classifier)
      all_real_results.append(real_results)
      
    all_fake_results_avg = []
    
    for fake_path in fake_paths:
      data_fake  = pd.read_csv(fake_path).to_numpy()
      data_fake_y = data_fake[:,-1]
      data_fake_X = data_fake[:,:data_dim-1]
      X_train_fake, _ , y_train_fake, _ = model_selection.train_test_split(data_fake_X ,data_fake_y, test_size=test_ratio, stratify=data_fake_y,random_state=42) 

      if scaler=="MinMax":
        scaler_fake = MinMaxScaler()
      else:
        scaler_fake = StandardScaler()
      
      scaler_fake.fit(data_fake_X)
      
      X_train_fake_scaled = scaler_fake.transform(X_train_fake)
      
      all_fake_results = []
      for classifier in classifiers:
        fake_results = supervised_model_training(X_train_fake_scaled,y_train_fake,X_test_real_scaled,y_test_real,classifier)
        all_fake_results.append(fake_results)

      all_fake_results_avg.append(all_fake_results)
    
    diff_results = np.array(all_real_results)- np.array(all_fake_results_avg).mean(axis=0)

    return diff_results

def stat_sim(real_path,fake_path,cat_cols=None):
    
    Stat_dict={}
    
    real = pd.read_csv(real_path)
    fake = pd.read_csv(fake_path)

    really = real.copy()
    fakey = fake.copy()

    real_corr = compute_associations(real, nominal_columns=cat_cols)

    fake_corr = compute_associations(fake, nominal_columns=cat_cols)

    corr_dist = np.linalg.norm(real_corr - fake_corr)
    
    cat_stat = []
    num_stat = []
    
    for column in real.columns:
        
        if column in cat_cols:

            real_pdf=(really[column].value_counts()/really[column].value_counts().sum())
            fake_pdf=(fakey[column].value_counts()/fakey[column].value_counts().sum())
            categories = (fakey[column].value_counts()/fakey[column].value_counts().sum()).keys().tolist()
            sorted_categories = sorted(categories)
            
            real_pdf_values = [] 
            fake_pdf_values = []

            for i in sorted_categories:
                real_pdf_values.append(real_pdf[i])
                fake_pdf_values.append(fake_pdf[i])
            
            if len(real_pdf)!=len(fake_pdf):
                zero_cats = set(really[column].value_counts().keys())-set(fakey[column].value_counts().keys())
                for z in zero_cats:
                    real_pdf_values.append(real_pdf[z])
                    fake_pdf_values.append(0)
            Stat_dict[column]=(distance.jensenshannon(real_pdf_values,fake_pdf_values, 2.0))
            cat_stat.append(Stat_dict[column])        
        else:
            scaler = MinMaxScaler()
            scaler.fit(real[column].values.reshape(-1,1))
            l1 = scaler.transform(real[column].values.reshape(-1,1)).flatten()
            l2 = scaler.transform(fake[column].values.reshape(-1,1)).flatten()
            Stat_dict[column]= (wasserstein_distance(l1,l2))
            num_stat.append(Stat_dict[column])

    return [np.mean(num_stat),np.mean(cat_stat),corr_dist]

def privacy_metrics(real_path,fake_path,data_percent=15):
    
    real = pd.read_csv(real_path).drop_duplicates(keep=False)
    fake = pd.read_csv(fake_path).drop_duplicates(keep=False)

    real_refined = real.sample(n=int(len(real)*(.01*data_percent)), random_state=42).to_numpy()
    fake_refined = fake.sample(n=int(len(fake)*(.01*data_percent)), random_state=42).to_numpy()

    scalerR = StandardScaler()
    scalerR.fit(real_refined)
    scalerF = StandardScaler()
    scalerF.fit(fake_refined)
    df_real_scaled = scalerR.transform(real_refined)
    df_fake_scaled = scalerF.transform(fake_refined)
    
    dist_rf = metrics.pairwise_distances(df_real_scaled, Y=df_fake_scaled, metric='minkowski', n_jobs=-1)
    dist_rr = metrics.pairwise_distances(df_real_scaled, Y=None, metric='minkowski', n_jobs=-1)
    rd_dist_rr = dist_rr[~np.eye(dist_rr.shape[0],dtype=bool)].reshape(dist_rr.shape[0],-1)
    dist_ff = metrics.pairwise_distances(df_fake_scaled, Y=None, metric='minkowski', n_jobs=-1)
    rd_dist_ff = dist_ff[~np.eye(dist_ff.shape[0],dtype=bool)].reshape(dist_ff.shape[0],-1) 
    smallest_two_indexes_rf = [dist_rf[i].argsort()[:2] for i in range(len(dist_rf))]
    smallest_two_rf = [dist_rf[i][smallest_two_indexes_rf[i]] for i in range(len(dist_rf))]       
    smallest_two_indexes_rr = [rd_dist_rr[i].argsort()[:2] for i in range(len(rd_dist_rr))]
    smallest_two_rr = [rd_dist_rr[i][smallest_two_indexes_rr[i]] for i in range(len(rd_dist_rr))]
    smallest_two_indexes_ff = [rd_dist_ff[i].argsort()[:2] for i in range(len(rd_dist_ff))]
    smallest_two_ff = [rd_dist_ff[i][smallest_two_indexes_ff[i]] for i in range(len(rd_dist_ff))]
    nn_ratio_rr = np.array([i[0]/i[1] for i in smallest_two_rr])
    nn_ratio_ff = np.array([i[0]/i[1] for i in smallest_two_ff])
    nn_ratio_rf = np.array([i[0]/i[1] for i in smallest_two_rf])
    nn_fifth_perc_rr = np.percentile(nn_ratio_rr,5)
    nn_fifth_perc_ff = np.percentile(nn_ratio_ff,5)
    nn_fifth_perc_rf = np.percentile(nn_ratio_rf,5)

    min_dist_rf = np.array([i[0] for i in smallest_two_rf])
    fifth_perc_rf = np.percentile(min_dist_rf,5)
    min_dist_rr = np.array([i[0] for i in smallest_two_rr])
    fifth_perc_rr = np.percentile(min_dist_rr,5)
    min_dist_ff = np.array([i[0] for i in smallest_two_ff])
    fifth_perc_ff = np.percentile(min_dist_ff,5)
    
    return np.array([fifth_perc_rf,fifth_perc_rr,fifth_perc_ff,nn_fifth_perc_rf,nn_fifth_perc_rr,nn_fifth_perc_ff]).reshape(1,6)    
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, BaggingClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier, RidgeClassifier
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score
import pandas as pd


df = pd.read_csv('output_data.csv')
metrics = pd.DataFrame()
X = df[['Area', 'Radius', 'Length']]
y = df['Octane Number']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
X_train = X_train.set_index('Area')
X_train = X_train.reset_index()
X_test = X_test.set_index('Area')
X_test = X_test.reset_index()

# Models

cb_model = CatBoostClassifier(5000, random_seed=42)
forest_model = RandomForestClassifier(n_estimators=1000, max_depth=90, bootstrap=True, min_samples_leaf=1, min_samples_split=5, random_state=42)
# knn = KNeighborsClassifier(n_neighbors=600)
# tree = DecisionTreeClassifier(random_state=42)
# svc = SVC(random_state=42)
# sgd = SGDClassifier(random_state=42)
# ridge = RidgeClassifier(max_iter=600, random_state=42)
# gb = GradientBoostingClassifier(max_depth=90, random_state=42)
# ada = AdaBoostClassifier(random_state=42)
# bagging = BaggingClassifier(random_state=42)

# Training

print("========= TRAINING CATBOOST =========")
cb_model.fit(X_train, y_train, verbose=False)
print("========= TRAINING RANDOM FOREST =========")
forest_model.fit(X_train, y_train)
# print("========= TRAINING KNN =========")
# knn.fit(X_train, y_train)
# print("========= TRAINING DECISION TREE =========")
# tree.fit(X_train, y_train)
# print("========= TRAINING SVC =========")
# svc.fit(X_train, y_train)
# print("========= TRAINING SGD =========")
# sgd.fit(X_train, y_train)
# print("========= TRAINING RIDGE =========")
# ridge.fit(X_train, y_train)
# print("========= TRAINING GRADIENT BOOSTING =========")
# gb.fit(X_train, y_train)
# print("========= TRAINING BAGGING =========")
# bagging.fit(X_train, y_train)
# print("========= TRAINING ADA =========")
# ada.fit(X_train, y_train)

# Predictions

cb_pred = cb_model.predict(X_test)
forest_pred = forest_model.predict(X_test)
# knn_pred = knn.predict(X_test)
# tree_pred = tree.predict(X_test)
# svc_pred = svc.predict(X_test)
# sgd_pred = sgd.predict(X_test)
# ridge_pred = ridge.predict(X_test)
# gb_pred = gb.predict(X_test)
# bagging_pred = bagging.predict(X_test)
# ada_pred = ada.predict(X_test)

# Creating dataframe

metrics = pd.concat([metrics, pd.DataFrame({
    'Model': 'CatBoostClassifier',
    'Precision score': round(precision_score(y_test, cb_pred, labels=[92, 95, 98], average='weighted'), 2),
    'Recall score': round(recall_score(y_test, cb_pred, labels=[92, 95, 98], average='weighted'), 2),
    'F1 score': round(f1_score(y_test, cb_pred, labels=[92, 95, 98], average='weighted'), 2)
}, index=[0])], ignore_index=True)

metrics = pd.concat([metrics, pd.DataFrame({
    'Model': 'RandomForestClassifier',
    'Precision score': round(precision_score(y_test, forest_pred, labels=[92, 95, 98], average='weighted'), 2),
    'Recall score': round(recall_score(y_test, forest_pred, labels=[92, 95, 98], average='weighted'), 2),
    'F1 score': round(f1_score(y_test, forest_pred, labels=[92, 95, 98], average='weighted'), 2)
}, index=[0])], ignore_index=True)

# metrics = pd.concat([metrics, pd.DataFrame({
#     'Model': 'KNeighbours',
#     'Precision score': round(precision_score(y_test, knn_pred, labels=[92, 95, 98], average='weighted'), 2),
#     'Recall score': round(recall_score(y_test, knn_pred, labels=[92, 95, 98], average='weighted'), 2),
#     'F1 score': round(f1_score(y_test, knn_pred, labels=[92, 95, 98], average='weighted'), 2)
# }, index=[0])], ignore_index=True)

# metrics = pd.concat([metrics, pd.DataFrame({
#     'Model': 'Decision Tree',
#     'Precision score': round(precision_score(y_test, tree_pred, labels=[92, 95, 98], average='weighted'), 2),
#     'Recall score': round(recall_score(y_test, tree_pred, labels=[92, 95, 98], average='weighted'), 2),
#     'F1 score': round(f1_score(y_test, tree_pred, labels=[92, 95, 98], average='weighted'), 2)
# }, index=[0])], ignore_index=True)

# metrics = pd.concat([metrics, pd.DataFrame({
#     'Model': 'Support Vector Machines',
#     'Precision score': round(precision_score(y_test, svc_pred, labels=[92, 95, 98], average='weighted'), 2),
#     'Recall score': round(recall_score(y_test, svc_pred, labels=[92, 95, 98], average='weighted'), 2),
#     'F1 score': round(f1_score(y_test, svc_pred, labels=[92, 95, 98], average='weighted'), 2)
# }, index=[0])], ignore_index=True)

# metrics = pd.concat([metrics, pd.DataFrame({
#     'Model': 'Stochastic Gradient Descent',
#     'Precision score': round(precision_score(y_test, sgd_pred, labels=[92, 95, 98], average='weighted'), 2),
#     'Recall score': round(recall_score(y_test, sgd_pred, labels=[92, 95, 98], average='weighted'), 2),
#     'F1 score': round(f1_score(y_test, sgd_pred, labels=[92, 95, 98], average='weighted'), 2)
# }, index=[0])], ignore_index=True)

# metrics = pd.concat([metrics, pd.DataFrame({
#     'Model': 'RidgeClassifier',
#     'Precision score': round(precision_score(y_test, ridge_pred, labels=[92, 95, 98], average='weighted'), 2),
#     'Recall score': round(recall_score(y_test, ridge_pred, labels=[92, 95, 98], average='weighted'), 2),
#     'F1 score': round(f1_score(y_test, ridge_pred, labels=[92, 95, 98], average='weighted'), 2)
# }, index=[0])], ignore_index=True)

# metrics = pd.concat([metrics, pd.DataFrame({
#     'Model': 'GradientBoostingClassifier',
#     'Precision score': round(precision_score(y_test, gb_pred, labels=[92, 95, 98], average='weighted'), 2),
#     'Recall score': round(recall_score(y_test, gb_pred, labels=[92, 95, 98], average='weighted'), 2),
#     'F1 score': round(f1_score(y_test, gb_pred, labels=[92, 95, 98], average='weighted'), 2)
# }, index=[0])], ignore_index=True)

# metrics = pd.concat([metrics, pd.DataFrame({
#     'Model': 'BaggingClassifier',
#     'Precision score': round(precision_score(y_test, bagging_pred, labels=[92, 95, 98], average='weighted'), 2),
#     'Recall score': round(recall_score(y_test, bagging_pred, labels=[92, 95, 98], average='weighted'), 2),
#     'F1 score': round(f1_score(y_test, bagging_pred, labels=[92, 95, 98], average='weighted'), 2)
# }, index=[0])], ignore_index=True)

# metrics = pd.concat([metrics, pd.DataFrame({
#     'Model': 'AdaClassifier',
#     'Precision score': round(precision_score(y_test, ada_pred, labels=[92, 95, 98], average='weighted'), 2),
#     'Recall score': round(recall_score(y_test, ada_pred, labels=[92, 95, 98], average='weighted'), 2),
#     'F1 score': round(f1_score(y_test, ada_pred, labels=[92, 95, 98], average='weighted'), 2)
# }, index=[0])], ignore_index=True)

# Save dataframe

metrics.to_csv('output_metrics.csv', index=False)
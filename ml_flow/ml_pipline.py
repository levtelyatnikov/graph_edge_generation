import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score

def ml_flow(X_train, X_val, y_train, y_val):
    X_train, X_val, y_train, y_val = X_train.cpu().numpy(), X_val.cpu().numpy(), y_train.cpu().numpy(), y_val.cpu().numpy()
    clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
    clf.fit(X_train, y_train)
    preds = clf.predict(X_val)
    f1_svm = np.float(np.round(f1_score(preds, y_val, average='macro'), 4))
    acc_svm = np.float(np.round(accuracy_score(preds, y_val), 4))

    clf = make_pipeline(StandardScaler(), LogisticRegression(random_state=0, max_iter=1000))
    clf.fit(X_train, y_train)
    preds = clf.predict(X_val)
    f1_lin = np.float(np.round(f1_score(preds, y_val, average='macro'), 4))
    acc_lin = np.float(np.round(accuracy_score(preds, y_val), 4))
    return f1_svm, acc_svm, f1_lin, acc_lin
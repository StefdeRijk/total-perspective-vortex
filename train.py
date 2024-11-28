# coding: utf-8

import matplotlib

from mne.decoding import SPoC
from joblib import dump

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import ShuffleSplit, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression

matplotlib.use('TkAgg')


def create_pipeline(X, y, transformer1, transformer2=None, transformer3=None):
    cv = ShuffleSplit(10, test_size=0.2, random_state=42)

    lda = LDA(solver='lsqr', shrinkage='auto')
    log_reg = LogisticRegression(penalty='l1', solver='liblinear', multi_class='auto')
    rfc = RandomForestClassifier(n_estimators=100, random_state=42)

    final_result = []

    pipeline1 = make_pipeline(transformer1, lda)
    scores1 = cross_val_score(pipeline1, X, y, cv=cv, n_jobs=1)
    final_result.append(('LDA ', pipeline1, scores1))
    if transformer2:
        pipeline2 = make_pipeline(transformer2, log_reg)
        scores2 = cross_val_score(pipeline2, X, y, cv=cv, n_jobs=1)
        final_result.append(('LOGR', pipeline2, scores2))
    if transformer3:
        pipeline3 = make_pipeline(transformer3, rfc)
        scores3 = cross_val_score(pipeline3, X, y, cv=cv, n_jobs=1)
        final_result.append(('RFC', pipeline3, scores3))

    return final_result


def save_pipeline(pipe, epochs_data_train, labels, subjectID, experiment_name):
    pipe = pipe.fit(epochs_data_train, labels)
    fileName = f"../data/models/model_subject_{subjectID}_{experiment_name}.joblib"
    dump(pipe, fileName)
    return


def train_data(X, y, transformer="CSP", run_all_pipelines=False):
    if transformer == "CSP":
        from csp import CSP
        csp1 = CSP()

        if run_all_pipelines:
            csp2 = CSP()
            csp3 = CSP()
            return create_pipeline(X, y, csp1, csp2, csp3)

        return create_pipeline(X, y, csp1)

    elif transformer == "SPoC":
        Spoc1 = SPoC(n_components=15, reg='oas', log=True, rank='full')

        if run_all_pipelines:
            Spoc2 = SPoC(n_components=15, reg='oas', log=True, rank='full')
            Spoc3 = SPoC(n_components=15, reg='oas', log=True, rank='full')
            return create_pipeline(X, y, Spoc1, Spoc2, Spoc3)
        else:
            return create_pipeline(X, y, Spoc1)

    else:
        raise ValueError(f"Unknown transformer, please enter valid one.")
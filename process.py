# coding: utf-8
import matplotlib
import matplotlib.pyplot as plt

import mne

from mne import events_from_annotations, pick_types
from mne.io import concatenate_raws
from mne.io.edf import read_raw_edf
from mne.channels import make_standard_montage
from mne.datasets import eegbci
from mne.preprocessing import ICA

mne.set_log_level("CRITICAL")
matplotlib.use('TkAgg')
DATA_SAMPLE_PATH = "../data/"


def get_data(subject_number):
    run_execution = [5, 9, 13]
    run_imagery = [6, 10, 14]

    raw_files = []

    for i, j in zip(run_execution, run_imagery):
        raw_files_execution = [read_raw_edf(f, preload=True, stim_channel='auto') for f in
                               eegbci.load_data(subject_number, i, DATA_SAMPLE_PATH)]
        raw_execution = concatenate_raws(raw_files_execution)

        raw_files_imagery = [read_raw_edf(f, preload=True, stim_channel='auto') for f in
                             eegbci.load_data(subject_number, j, DATA_SAMPLE_PATH)]
        raw_imagery = concatenate_raws(raw_files_imagery)

        events, _ = mne.events_from_annotations(raw_execution, event_id=dict(T0=1, T1=2, T2=3))
        mapping = {1: 'rest', 2: 'do/feet', 3: 'do/hands'}
        annot_from_events = mne.annotations_from_events(
            events=events, event_desc=mapping, sfreq=raw_execution.info['sfreq'],
            orig_time=raw_execution.info['meas_date'])
        raw_execution.set_annotations(annot_from_events)

        events, _ = mne.events_from_annotations(raw_imagery, event_id=dict(T0=1, T1=2, T2=3))
        mapping = {1: 'rest', 2: 'imagine/feet', 3: 'imagine/hands'}
        annot_from_events = mne.annotations_from_events(
            events=events, event_desc=mapping, sfreq=raw_imagery.info['sfreq'],
            orig_time=raw_imagery.info['meas_date'])
        raw_imagery.set_annotations(annot_from_events)

        raw_files.append(raw_execution)
        raw_files.append(raw_imagery)
    raw = concatenate_raws(raw_files)

    event, event_dict = events_from_annotations(raw)
    picks = pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False, exclude='bads')

    return [raw, event, event_dict, picks]


def prepare_data(raw, plotIt=False):
    eegbci.standardize(raw)
    montage = make_standard_montage("biosemi64")
    raw.set_montage(montage, on_missing='ignore')

    if plotIt:
        montage = raw.get_montage()
        p = montage.plot()
        p = mne.viz.plot_raw(raw, scalings={"eeg": 75e-6})

    return raw


def filter_data(raw, plotIt=None):
    raw.filter(7, 30, fir_design='firwin', skip_by_annotation='edge')
    if plotIt:
        p = mne.viz.plot_raw(raw, scalings={"eeg": 75e-6})
        plt.show()
    return raw


def filter_eye_artifacts(raw, picks, method, plotIt=None):
    raw_corrected = raw.copy()
    n_components = 20

    ica = ICA(n_components=n_components, method=method, fit_params=None, random_state=97)

    ica.fit(raw_corrected, picks=picks)

    [eog_indicies, scores] = ica.find_bads_eog(raw, ch_name='Fpz', threshold=1.5)
    ica.exclude.extend(eog_indicies)
    ica.apply(raw_corrected, n_pca_components=n_components, exclude=ica.exclude)

    if plotIt:
        ica.plot_components()
        ica.plot_scores(scores, exclude=eog_indicies)

        plt.show()

    return raw_corrected


def get_events(data_filtered, tmin=-1., tmax=4.):
    events, event_ids = events_from_annotations(data_filtered)
    picks = mne.pick_types(data_filtered.info, meg=False, eeg=True, stim=False, eog=False, exclude='bads')
    epochs = mne.Epochs(data_filtered, events, event_ids, tmin, tmax, proj=True,
                        picks=picks, baseline=None, preload=True)
    labels = epochs.events[:, -1]
    return labels, epochs, picks


def pre_process_data(subjectID, experiments):
    [raw, event, event_dict, picks] = get_data(subjectID)

    raw_prepared = prepare_data(raw)

    raw_filtered = filter_data(raw_prepared)

    labels, epochs, picks = get_events(raw_filtered)

    selected_epochs = epochs[experiments]

    X = selected_epochs.get_data()
    y = selected_epochs.events[:, -1] - 1

    return [X, y, epochs]

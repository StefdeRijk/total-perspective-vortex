from enums import EXPERIMENTS
from predict import predict
from parse_args import load_config
from process import pre_process_data
from train import train_data, save_pipeline

import json
import time
import hashlib
import numpy as np


def hash_list_secure(my_list):
    sorted_tuple = tuple(sorted(my_list))
    hash_object = hashlib.sha256(str(sorted_tuple).encode())
    return hash_object.hexdigest()


def process_subject(subjectID, args, isSingleSubject=False):
    start_time_inner = time.time()
    [X, y, epochs] = pre_process_data(subjectID, EXPERIMENTS[args['EXPERIMENT']])

    result_inner = [0, 0]
    output = []
    output.append(f"-----------------------------[Subject {subjectID}]")
    stats = {
        'subject_id': subjectID,
        'pipelines': [],
        'cross_val_score': 0,
        'accuracy': 0
    }
    if args['MODE'] == "train" or args['MODE'] == "all":
        pipelines = train_data(X=X, y=y, transformer=args['TRANSFORMER'], run_all_pipelines=True)
        best_pipeline = {'cross_val_score': -1}

        for pipel in pipelines:
            cross_val_score = pipel[2].mean()
            pipeline_name = pipel[0]
            pipeline = pipel[1]
            output.append(f":--- [S{subjectID}] {pipeline_name} cross_val_score : {cross_val_score.round(2)}")

            if cross_val_score > best_pipeline['cross_val_score']:
                best_pipeline = {'name': pipeline_name, 'cross_val_score': cross_val_score, 'pipeline': pipeline}
            stats['pipelines'].append((pipeline_name, cross_val_score))
        save_pipeline(best_pipeline['pipeline'], X, y, subjectID, args['EXPERIMENT'])
        result_inner[0] = best_pipeline['cross_val_score']
        stats['cross_val_score'] = result_inner[0]

    if args['MODE'] == "predict" or args['MODE'] == "all":
        prediction_result = predict(X, y, subjectID, args['EXPERIMENT'], isSingleSubject)
        output.append(
            f":--- [S{subjectID}] Prediction accurracy: {'{:.2%}'.format(prediction_result).rstrip('0').rstrip('.')}")
        result_inner[1] = prediction_result
        stats['accuracy'] = result_inner[1]

    end_time_inner = time.time()

    time_cost_inner = end_time_inner - start_time_inner
    stats['time_cost'] = time_cost_inner
    print(*output, sep="\n")
    print(f":--- [S{subjectID}] time cost: {round(stats['time_cost'], 2)} seconds")
    return stats


def calculate_all_means(cross_val_scores, accuracy_scores, final_stats):
    print("\n-------------------[Mean Scores for all subjects]------------------")
    if len(cross_val_scores) > 1:
        print(f":--- Mean cross value : {np.mean(cross_val_scores).round(2)}")
        final_stats['mean_cross_val_score'] = np.mean(cross_val_scores)
    if len(accuracy_scores) > 1:
        print(f":--- Mean accuracy  : {np.mean(accuracy_scores).round(2)}")
        final_stats['mean_accuracy'] = np.mean(accuracy_scores)


def result_to_json(final_stats, args):
    results_filename = \
        f"../data/results/results-{args['MODE']}-{args['EXPERIMENT']}-{time.time()}-{args['TRANSFORMER']}-{final_stats['subjects_hash']}.json"

    with open(results_filename, 'w', encoding='utf-8') as f:
        json.dump(final_stats, f, ensure_ascii=False, indent=4)

    print(f"Result of training and prediction are stored in \n[{results_filename}]")


def main():
    start_time = time.time()
    args = load_config("config.yaml")
    print(args)

    print(
        f"Experiment in study: ({EXPERIMENTS[args['EXPERIMENT']][0]}) <--VS--> ({EXPERIMENTS[args['EXPERIMENT']][1]})")
    CALC_MEAN_FOR_ALL = True if len(args['SUBJECTS']) > 1 else False

    cross_val_scores = []
    accuracy_scores = []
    final_stats = {
        'subjects_hash': "all" if len(args['SUBJECTS']) == 109 else ''.join(map(str, args['SUBJECTS'])),
        'config': args,
        'events': EXPERIMENTS[args['EXPERIMENT']],
        'subjects': [],
        'time_unit': "seconds"
    }

    for subjectID in args['SUBJECTS']:
        result = process_subject(subjectID, args, isSingleSubject=not CALC_MEAN_FOR_ALL)
        if args['MODE'] == "train" or args['MODE'] == "all":
            cross_val_scores.append(result['cross_val_score'])

        if args['MODE'] == "predict" or args['MODE'] == "all":
            accuracy_scores.append(result['accuracy'])

        final_stats['subjects'].append(result)

    if CALC_MEAN_FOR_ALL:
        calculate_all_means(cross_val_scores, accuracy_scores, final_stats)

    final_stats['time_cost'] = time.time() - start_time
    print(f"Time Elapsed for all : {round(final_stats['time_cost'], 2)}")

    result_to_json(final_stats, args)


if __name__ == "__main__":
    main()

from os import listdir
from os.path import isfile, join
import json
import sys
import os

sys.path.insert(0,'/data/users/sstauden/dev/nn_robustness/tasks/QuestionAnswering/experiments/evaluation')
from PerformanceEvaluation import create_predictions_csv, EvaluationReport

sys.path.insert(0,'/data/users/sstauden/dev/nn_robustness/datasets')
import DatasetPaths


# Model Paths
model_base = "/nethome/sstauden/bi-att-flow/training_out/"
prepro_base = "/nethome/sstauden/bi-att-flow/prepro_data/"
test_base = "/nethome/sstauden/bi-att-flow/testing/"

# Model Information
model_dict = {
  'BiDAF-original': {
    'path': model_base + 'original',
    'train_data': DatasetPaths.SQUAD_ORIGINAL_V1_train,
    'prepro_data': prepro_base + 'original'
  },
  'BiDAF-AddSent': {
    'path': model_base + 'AddSent',
    'train_data': DatasetPaths.SQUAD_ADDSENT_V1_train,
    'prepro_data': prepro_base + 'AddSent'
  },
  'BiDAF-AddSentDiv': {
    'path': model_base + 'AddSentDiv',
    'train_data': DatasetPaths.SQUAD_ADDSENTDIV_V1_train,
    'prepro_data': prepro_base + 'AddSentDiv'
  },
  'BiDAF-WordSwap': {
    'path': model_base + 'WordSwap',
    'train_data': DatasetPaths.SQUAD_WORDSWAP_train,
    'prepro_data': prepro_base + 'WordSwap'
  },
}

attack_dict = {
  'original': {
    'train': DatasetPaths.SQUAD_ORIGINAL_V1_train, 
    'test': DatasetPaths.SQUAD_ORIGINAL_V1_dev
  },
  'AddSent': {
    'train': DatasetPaths.SQUAD_ADDSENT_V1_train, 
    'test': DatasetPaths.SQUAD_ADDSENT_V1_dev
  },
  'AddSentDiv': {
    'train': DatasetPaths.SQUAD_ADDSENTDIV_V1_train, 
    'test': DatasetPaths.SQUAD_ADDSENTDIV_V1_dev
  },
  'WordSwap': {
    'train': DatasetPaths.SQUAD_WORDSWAP_train, 
    'test': DatasetPaths.SQUAD_WORDSWAP_dev
  },
}

def _get_latest_file(folder, prefix, suffix):

  all_files = [f for f in listdir(folder) if ( isfile(join(folder, f)) and f.startswith(prefix) and f.endswith(suffix) )]

  # get all versions
  all_versions = []
  for single_file in all_files:
    
    # remove prefix
    cur_num = single_file.split(prefix)[-1]

    # remove file extension
    cur_num = cur_num.split('.')[0]

    # add version to list
    all_versions.append(int(cur_num))

  # get latest_version
  return prefix + str(max(all_versions)).zfill(6) + suffix

def create_predictions_json(answer_folder, out_path):

  # get latest answer file
  answer_file = _get_latest_file(answer_folder, "dev-", ".json")

  # create new dict
  preds = {}

  with open(join(answer_folder, answer_file)) as json_file:
    answer_data = json.load(json_file)

    for qas_id in answer_data:

      if qas_id == "scores":
        continue

      print(qas_id)
      
      # set answer
      preds[qas_id] = answer_data[qas_id]

  
  if not os.path.exists(os.path.dirname(out_path)):
      os.makedirs(os.path.dirname(out_path))

  # save results
  with open(out_path, 'w') as outfile:
    json.dump(preds, outfile, indent=4)

  print("Data from " + answer_file + " successfully saved to " + out_path)

def create_prediction_files(model_name, attack_name):

  base_path = "/data/users/sstauden/dev/nn_robustness/tasks/QuestionAnswering/experiments/"

  # get model path
  model_path = model_dict[model_name]['path']

  test_base = "/nethome/sstauden/bi-att-flow/testing/"

  answer_folder = test_base + model_name + "/" + attack_name + "/answer"

  # create predictions_.json
  print("Creating Predictions json")
  pred_json_file = base_path + "testing/{}/test_{}/predictions_.json".format(model_name, attack_name)
  create_predictions_json(answer_folder, out_path=pred_json_file)

  # -- evaluation stage --
  print("Creating Predictions CSV")
  dataset_file = attack_dict[attack_name]['test']
  pred_csv_file = base_path + 'evaluation/{}/test_{}/predictions.csv'.format(model_name, attack_name)

  # create predictions.csv
  create_predictions_csv(dataset_file, pred_json_file, outfile=pred_csv_file)

  print("Creating Report")
  eval_report = EvaluationReport(model_output_folder=model_name, 
                                  train_file=model_dict[model_name]['train_data'], 
                                  dataset_json_file=dataset_file, 
                                  prediction_json_file=pred_json_file,
                                  prediction_csv_file=pred_csv_file,
                                  outfile=base_path + "evaluation/{}/test_{}/report.csv".format(model_name, attack_name))

  eval_report.create_report()

if __name__ == "__main__":

  model_name = sys.argv[1]
  attack_name = sys.argv[2]

  create_prediction_files(model_name, attack_name)



  




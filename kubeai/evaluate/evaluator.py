from abc import abstractmethod
import os
from kubeai.evaluate.mysql_handler import MysqlHandler

class Evaluator:

    def __init__(self):
        model_dir = os.getenv('MODEL_PATH')
        dataset_dir = os.getenv('DATASET_PATH')
        report_dir = os.getenv('METRICS_PATH')
        self.model_dir = model_dir
        self.dataset_dir = dataset_dir
        self.report_dir = report_dir

    def test_init(self, model_dir, dataset_dir, report_dir):
        self.model_dir = model_dir
        self.dataset_dir = dataset_dir
        self.report_dir = report_dir

    @abstractmethod
    def preprocess_dataset(self):
        pass

    @abstractmethod
    def load_model(self):
        pass

    @abstractmethod
    def evaluate_model(self, dataset):
        pass

    @abstractmethod
    def report_metrics(self, metrics):
        pass

    def report_process(self, metrics):
        self.metrics = metrics
        self.to_json_file()

        flag = os.getenv('ENABLE_MYSQL')
        if flag == "True":
            print("Writing mysql")
            host = os.getenv('MYSQL_HOST')
            port = int(os.getenv('MYSQL_PORT'))
            user = os.getenv('MYSQL_USERNAME')
            password = os.getenv('MYSQL_PASSWORD')

            handler = MysqlHandler(host=host, port=port, user=user, password=password)
            handler.writeMetrics(metrics=metrics)


    def to_json_file(self):
        import json
        with open(self.report_dir + '/metrics.json', "w") as f:
            f.write(json.dumps(self.metrics, ensure_ascii=False, indent=4, separators=(',', ':')))

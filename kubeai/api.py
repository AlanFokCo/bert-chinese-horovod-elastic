# !/usr/bin/env python
# coding=utf-8

class KubeAI:

    @staticmethod
    def evaluate(evaluator):
        evaluator.load_model()
        dataset = evaluator.preprocess_dataset()
        metrics = evaluator.evaluate_model(dataset)
        evaluator.report_metrics(metrics)
        evaluator.report_process(metrics)

    @staticmethod
    def test(evaluator, model_dir, dataset_dir, report_dir):
        evaluator.test_init(model_dir=model_dir, dataset_dir=dataset_dir, report_dir=report_dir)
        evaluator.load_model()
        dataset = evaluator.preprocess_dataset()
        metrics = evaluator.evaluate_model(dataset)
        evaluator.report_metrics(metrics)
        evaluator.report_process(metrics)
FROM registry-vpc.cn-beijing.aliyuncs.com/huozx/bert-elastic-demo:v1.5-test
RUN pip install kubeai
COPY ./train_bert.py /examples/elastic/pytorch/train_bert.py
import os
import json
import tensorflow as tf
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('./')
tfevents = "./events.out.tfevents.1673248626.bert-no-elastic-test-worker-5.1688160.0"
elastic_tfevents = "events.out.tfevents.1673248637.bert-elastic-scaleout-test-worker-11.1952516.0"
scale_out_in_tfevents = "events.out.tfevents.1673265090.bert-elastic-scaleout-scalein-test-worker-9.2008897.0"

events = tf.compat.v1.train.summary_iterator(tfevents)
for event in events:
    for value in event.summary.value:
        print(value)

index = 1
cycle = 0
epoch = 0
events = tf.compat.v1.train.summary_iterator(tfevents)
for event in events:
    for value in event.summary.value:
        if cycle == 4:
            cycle = 0
            index += 1
        value = str(value)
        arr = value.split("\n")
        if arr[0].split()[1].replace("\"", "") == "time":
            writer.add_scalars(arr[0].split()[1].replace("\"", ""), {"no-elastic": float(arr[1].split()[1])}, epoch)
            epoch += 1
            cycle -= 1
        writer.add_scalars(arr[0].split()[1].replace("\"", ""), {"no-elastic": float(arr[1].split()[1])}, index)
        print(arr[0].split()[1].replace("\"", ""), arr[1].split()[1])
        cycle += 1

index = 1
cycle = 0
epoch = 0
events = tf.compat.v1.train.summary_iterator(elastic_tfevents)
for event in events:
    for value in event.summary.value:
        if cycle == 4:
            cycle = 0
            index += 1
        value = str(value)
        arr = value.split("\n")
        if arr[0].split()[1].replace("\"", "") == "time":
            writer.add_scalars(arr[0].split()[1].replace("\"", ""), {"elastic": float(arr[1].split()[1])}, epoch)
            epoch += 1
            cycle -= 1
        writer.add_scalars(arr[0].split()[1].replace("\"", ""), {"elastic": float(arr[1].split()[1])}, index)
        print(arr[0].split()[1].replace("\"", ""), arr[1].split()[1])
        cycle += 1

index = 1
cycle = 0
epoch = 0
events = tf.compat.v1.train.summary_iterator(scale_out_in_tfevents)
for event in events:
    for value in event.summary.value:
        if cycle == 4:
            cycle = 0
            index += 1
        value = str(value)
        arr = value.split("\n")
        if arr[0].split()[1].replace("\"", "") == "time":
            writer.add_scalars(arr[0].split()[1].replace("\"", ""), {"elastic-out-in": float(arr[1].split()[1])}, epoch)
            epoch += 1
            cycle -= 1
        writer.add_scalars(arr[0].split()[1].replace("\"", ""), {"elastic-out-in": float(arr[1].split()[1])}, index)
        print(arr[0].split()[1].replace("\"", ""), arr[1].split()[1])
        cycle += 1

writer.close()


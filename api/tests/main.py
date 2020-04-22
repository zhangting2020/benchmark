# copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import print_function

import argparse
import os
import feeder
import json

import sys
sys.path.append("..")
from common import utils


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--task',
        type=str,
        default="speed",
        help='Specify the task: [speed|accuracy]')
    parser.add_argument(
        '--framework',
        type=str,
        default="paddle",
        help='Specify the framework: [paddle|tensorflow|tf|both]')
    parser.add_argument(
        '--dtype',
        type=str,
        default="float32",
        help='Specify the data type of api')
    parser.add_argument(
        '--run_with_executor',
        type=str2bool,
        default=True,
        help='Whether running with executor [True|False]')
    parser.add_argument(
        '--check_output',
        type=str2bool,
        default=True,
        help='Whether checking the consistency of outputs [True|False]')
    parser.add_argument(
        '--profiler',
        type=str,
        default="none",
        help='Choose which profiler to use [\"none\"|\"Default\"|\"OpDetail\"|\"AllOpDetail\"|\"nvprof\"]')
    parser.add_argument(
        '--backward',
        type=str2bool,
        default=False,
        help='Whether appending grad ops [True|False]')
    parser.add_argument(
        '--use_gpu',
        type=str2bool,
        default=False,
        help='Whether using gpu [True|False]')
    parser.add_argument(
        '--repeat',
        type=int,
        default=1,
        help='Iterations of Repeat running')
    parser.add_argument(
        '--log_level',
        type=int,
        default=0,
        help='level of logging')
    parser.add_argument(
        '--gpu_id',
        type=int,
        default=0,
        help='GPU id when benchmarking for GPU')
    args = parser.parse_args()
    gpu_id = args.gpu_id if args.gpu_id > 0 else 0
    if os.environ.get("CUDA_VISIBLE_DEVICES", None) is None:
        print("CUDA_VISIBLE_DEVICES is None, set to CUDA_VISIBLE_DEVICES={}".format(gpu_id))
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    if args.task not in ["speed", "accuracy"]:
        raise ValueError("task should be speed, accuracy")
    if args.framework not in ["paddle", "tensorflow", "tf", "both"]:
        raise ValueError("task should be paddle, tensorflow, tf, both")
    if args.dtype not in ["float32", "float16"]:
        raise ValueError("dtype should be float32, float16")
    return args


def test_paddle(task, obj, args, feed_list=None):
    feed = None
    if feed_list is not None:
        assert len(feed_list) == len(obj.feed_vars)

        feed = {}
        for i in range(len(obj.feed_vars)):
            feed[obj.feed_vars[i].name] = feed_list[i]

    if task == "speed":
        if args.run_with_executor:
            obj.run_with_executor(use_gpu=args.use_gpu,
                                  feed=feed,
                                  repeat=args.repeat,
                                  log_level=args.log_level,
                                  check_output=args.check_output,
                                  profiler=args.profiler)
        else:
            obj.run_with_core_executor(use_gpu=args.use_gpu,
                                       feed=feed,
                                       repeat=args.repeat,
                                       log_level=args.log_level,
                                       check_output=args.check_output,
                                       profiler=args.profiler)
        return None
    elif task == "accuracy":
        if feed is None:
            raise ValueError("feed should not be None when checking accuracy.")
        outputs = obj.run_with_executor(use_gpu=args.use_gpu,
                                        feed=feed,
                                        check_output=False)
        return outputs


def test_tensorflow(task, obj, args, feed_list=None):
    feed = None
    if feed_list is not None:
        assert len(feed_list) == len(obj.feed_list)

        feed = {}
        for i in range(len(obj.feed_list)):
            feed[obj.feed_list[i]] = feed_list[i]

    if task == "speed":
        profile = True if args.profiler != "none" else False
        obj.run(use_gpu=args.use_gpu,
                feed=feed,
                repeat=args.repeat,
                log_level=args.log_level,
                check_output=args.check_output,
                profile=profile)
        return None
    elif task == "accuracy":
        if feed is None:
            raise ValueError("feed should not be None when checking accuracy.")
        outputs = obj.run(use_gpu=args.use_gpu,
                          feed=feed,
                          check_output=False)
        return outputs


def test_speed_main(obj):
    args = parse_args()
    obj.build_program(backward=args.backward, dtype=args.dtype)
    test_paddle("speed", obj, args)


def test_per_config(pd_obj=None, tf_obj=None, feed_spec=None, api_config=None, args=None):
    feed_list = None
    if args.task == "accuracy" or args.framework in ["paddle", "both"]:
        if pd_obj is None:
            raise ValueError("Paddle object is None.")
        pd_obj.api_config = api_config
        pd_obj.api_config = pd_obj.to_pd_api_config()
        pd_obj.create_progrom()
        pd_obj.build_program(backward=args.backward, dtype=args.dtype)
        feed_list = feeder.feed_paddle(pd_obj, feed_spec)
        pd_outputs = test_paddle(args.task, pd_obj, args, feed_list)

    if args.task == "accuracy" or args.framework in ["tensorflow", "tf", "both"]:
        if tf_obj is None:
            raise ValueError("TensorFlow object is None.")
        tf_obj.api_config = api_config
        tf_obj.api_config = tf_obj.to_tf_api_config()
        tf_obj.build_graph(backward=args.backward)
        feed_list = feeder.feed_tensorflow(tf_obj, feed_list, feed_spec)
        tf_outputs = test_tensorflow(args.task, tf_obj, args, feed_list)

    if args.task == "accuracy":
        utils.check_outputs(pd_outputs, tf_outputs, name=pd_obj.name)


def test_main(pd_obj=None, tf_obj=None, feed_spec=None):
    args = parse_args()
    config_file = os.path.join('APIConfig', pd_obj.name + '.json')
    with open(config_file, 'r') as f:
        api_configs = json.load(f)

    for api_config in api_configs:
        test_per_config(pd_obj, tf_obj, feed_spec, api_config, args)


    
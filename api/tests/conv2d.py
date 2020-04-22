#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

from main import test_main

import sys
sys.path.append("..")
from common import paddle_api_benchmark as paddle_api
from common import tensorflow_api_benchmark as tensorflow_api


class PDConv2d(paddle_api.PaddleAPIBenchmarkBase):
    def __init__(self):
        super(PDConv2d, self).__init__()
        self.name = "conv2d"

    def to_pd_api_config(self, api_config):
        if api_config['data_format'] == "NCHW":
            num_channels = api_config['input_shape'][1]
        elif api_config['data_format'] == "NHWC":
            num_channels = api_config['input_shapw'][4]

        api_config['filter_tensor_shape'] = [
            api_config['num_filters'], num_channels,
            api_config['filter_size'][0], api_config['filter_size'][1]
        ]
        return api_config

    def build_program(self, backward=False, dtype=None):
        import paddle.fluid as fluid

        with fluid.program_guard(self.main_program, self.startup_program):
            input = fluid.data(
                name='input',
                shape=self.api_config['input_shape'],
                dtype=dtype,
                lod_level=0)
            filter = fluid.layers.create_parameter(
                name='filter',
                shape=self.api_config['filter_tensor_shape'],
                dtype=dtype)
            input.stop_gradient = False
            result = fluid.layers.conv2d(
                input=input,
                num_filters=self.api_config['num_filters'],
                filter_size=self.api_config['filter_size'],
                stride=self.api_config['stride'],
                padding=self.api_config['padding'],
                dilation=self.api_config['dilation'],
                groups=self.api_config['groups'],
                param_attr='filter',
                bias_attr=False,
                use_cudnn=self.api_config['use_cudnn'],
                act=None,
                data_format=self.api_config['data_format'])

            self.feed_vars = [input, filter]
            self.fetch_vars = [result]
            if backward:
                self.append_gradients(result, [input])


class TFConv2d(tensorflow_api.TensorflowAPIBenchmarkBase):
    def __init__(self):
        super(TFConv2d, self).__init__()
        self.name = "conv2d"

    def to_tf_api_config(self, api_config):
        def _convert_filter_shape(api_config):
            if api_config['data_format'] == "NCHW":
                num_channels = api_config['input_shape'][1]
            elif api_config['data_format'] == "NHWC":
                num_channels = api_config['input_shape'][4]

            return [
                api_config['filter_size'][0], api_config['filter_size'][1],
                num_channels, api_config['num_filters']
            ]

        def _convert_padding(api_config):
            if isinstance(api_config['padding'], str):
                return api_config['padding']

            assert isinstance(api_config['padding'], list)
            pad_top = api_config['padding'][0] if len(api_config[
                'padding']) == 2 else api_config['padding'][0]
            pad_bottom = api_config['padding'][0] if len(api_config[
                'padding']) == 2 else api_config['padding'][1]
            pad_left = api_config['padding'][1] if len(api_config[
                'padding']) == 2 else api_config['padding'][2]
            pad_right = api_config['padding'][1] if len(api_config[
                'padding']) == 2 else api_config['padding'][3]

            if api_config['data_format'] == "NCHW":
                return [[0, 0], [0, 0], [pad_top, pad_bottom],
                        [pad_left, pad_right]]
            elif api_config['data_format'] == "NHWC":
                return [[0, 0], [pad_top, pad_bottom], [pad_left, pad_right],
                        [0, 0]]

        api_config['filter_size'] = _convert_filter_shape(api_config)
        api_config['padding'] = _convert_padding(api_config)
        return api_config

    def build_graph(self, backward=False, dtype=None):
        import tensorflow as tf

        self.name = "conv2d"
        self.allow_growth = True

        input = tf.placeholder(
            name='input',
            shape=self.api_config['input_shape'],
            dtype=tf.float32)
        filter = tf.placeholder(
            name='filter',
            shape=self.api_config['filter_size'],
            dtype=tf.float32)
        result = tf.nn.conv2d(
            input=input,
            filter=filter,
            strides=self.api_config['stride'],
            padding=self.api_config['padding'],
            data_format=self.api_config['data_format'],
            dilations=self.api_config['dilation'],
            use_cudnn_on_gpu=self.api_config['use_cudnn'])

        self.feed_list = [input, filter]
        self.fetch_list = [result]
        if backward:
            self.append_gradients(result, [input])


if __name__ == '__main__':
    feed_spec = [
        {
            "range": [0, 1]
        },  # input
        {
            "permute": [2, 3, 1, 0]
        }  # filters
    ]
    test_main(PDConv2d(), TFConv2d(), feed_spec=feed_spec)

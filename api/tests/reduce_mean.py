#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
from common import api_param


class PDReduceMean(paddle_api.PaddleAPIBenchmarkBase):
    def build_program(self, config):
        import paddle.fluid as fluid

        self.name = "reduce_mean"
        with fluid.program_guard(self.main_program, self.startup_program):
            input = fluid.data(
                name='input',
                shape=config.input_shape,
                dtype=config.input_dtype,
                lod_level=0)
            input.stop_gradient = False
            result = fluid.layers.reduce_mean(
                input=input, dim=config.dim, keep_dim=config.keep_dim)

            self.feed_vars = [input]
            self.fetch_vars = [result]
            if config.backward:
                self.append_gradients(result, [input])


class TFReduceMean(tensorflow_api.TensorflowAPIBenchmarkBase):
    def build_graph(self, config):
        import tensorflow as tf

        self.name = "reduce_mean"
        self.allow_growth = True

        input = tf.placeholder(
            name='input',
            shape=config.input_shape,
            dtype=tf.as_dtype(config.input_dtype))
        result = tf.reduce_mean(
            input_tensor=input, axis=config.dim, keepdims=config.keep_dim)

        self.feed_list = [input]
        self.fetch_list = [result]
        if config.backward:
            self.append_gradients(result, [input])


if __name__ == '__main__':
    test_main(
        PDReduceMean(),
        TFReduceMean(),
        config=api_param.APIConfig("reduce_mean", ""))

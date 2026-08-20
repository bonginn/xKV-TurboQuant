################################################################################
#
# Copyright 2024 ByteDance Ltd. and/or its affiliates. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
################################################################################

def choose_model_class(model_name):
    if 'llama' in model_name.lower():
        from .llama import Llama
        return Llama
    elif 'glm' in model_name.lower():
        from .glm import GLM
        return GLM
    elif 'yi' in model_name.lower():
        from .llama import Llama
        return Llama
    elif 'qwen' in model_name.lower():
        from .qwen import Qwen2
        return Qwen2
    elif 'phi' in model_name.lower():
        from .phi3 import Phi3
        return Phi3
    else:
        raise ValueError(f"Model {model_name} not found")

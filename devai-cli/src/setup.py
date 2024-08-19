# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from setuptools import setup, find_packages

setup(
    name='devai-cli',
    version='0.0.0',
    packages=find_packages(),
    py_modules=['devai'],
    install_requires=[
        'click==8.1.7',
        'GitPython==3.1.43',
        'google-cloud-aiplatform==1.62.0',
        'google-cloud-secret-manager==2.20.2',
        'json-repair==0.23.1',
        'langchain-community==0.2.1',
        'langchain-google-community==1.0.6',
        'langchain-google-vertexai==1.0.4',
        'rich==13.7.1'
    ],
    entry_points={
        'console_scripts': [
            'devai = devai.cli:devai',
        ],
    },
)

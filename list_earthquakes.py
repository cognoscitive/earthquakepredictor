# Copyright 2020 Cognoscitive Automata Incorporated
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

"""
Earthquake Predictor Neural Network Experiment
List Earthquakes
"""
import numpy as np
import csv

def list_earthquakes(earthquake_csv,filename):
    print('Creating list...')
    values = []
    with open(earthquake_csv) as csvfile:
        reader = csv.DictReader(csvfile, fieldnames=['latitude','longitude','magnitude'])
        for row in reader:
            values.append((row['latitude'], row['longitude'], row['magnitude']))
    dtype = [('latitude', float), ('longitude', float), ('magnitude', float)]
    unsorted = np.array(values, dtype=dtype)
    sorted = np.sort(unsorted, order='magnitude')
    reversed = sorted[::-1]
    print('Done list creation!')
    print('Creating CSV...')
    with open(filename, 'w') as f:
        writer = csv.writer(f)
        writer.writerows(reversed)
    print('CSV created!')

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
import os
import numpy as np
import csv
from datetime import datetime, timedelta

def list_earthquakes(earthquake_csv, filename):
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

def add_deltas(model_name, current_date, previous_date=None):
    print('Adding Deltas...')
    if previous_date:
        deltas = {}
        lines = []
        with open('prediction/' + model_name + '/' + 'list_' + current_date + '.csv', 'r') as f:
            lines_1 = f.readlines()
        for line in lines_1:
            data = line.split(',')
            if len(data) > 1:
                deltas[(data[0], data[1])] = float(data[2])
        with open('prediction/' + model_name + '/' + 'list_' + previous_date + '.csv', 'r') as f:
            lines_2 = f.readlines()
        for line in lines_2:
            data = line.split(',')
            if len(data) > 1 and (data[0], data[1]) in deltas:
                deltas[(data[0], data[1])] = float(deltas[(data[0], data[1])]) - float(data[2])
        for i, line in enumerate(lines_1):
            data = line.split(',')
            if len(data) > 1:
                lines.append(lines_1[i].strip() + ',' + str(deltas[(data[0], data[1])]) + '\n')
        with open('prediction/' + model_name + '/' + 'list_' + current_date + '.csv', 'w') as f:
            for line in lines:
                f.write(line)
    else:
        lines = []
        with open('prediction/' + model_name + '/' + 'list_' + current_date + '.csv', 'r') as f:
            lines_1 = f.readlines()
        for line in lines_1:
            data = line.split(',')
            if len(data) > 1:
                lines.append(line)
        with open('prediction/' + model_name + '/' + 'list_' + current_date + '.csv', 'w') as f:
            for line in lines:
                f.write(line.strip() + ',0.0\n')
    print('Deltas added!')

def main():
    # Script to update prediction data with deltas for backtesting purposes
    for model_name in ['high-recall', 'high-precision']:
        date = datetime.strptime('2018-03-01', '%Y-%m-%d')
        end = datetime.strptime('2023-02-11', '%Y-%m-%d')
        while date.year != end.year or date.month != end.month or date.day != end.day:
            previous_date = date - timedelta(days=1)
            date_str = date.strftime('%Y-%m-%d')
            previous_date_str = previous_date.strftime('%Y-%m-%d')
            if os.path.exists('prediction/' + model_name + '/' + 'list_' + date_str + '.csv') and os.path.exists('prediction/' + model_name + '/' + 'list_' + previous_date_str + '.csv'):
                add_deltas(model_name, date_str, previous_date_str)
            elif os.path.exists('prediction/' + model_name + '/' + 'list_' + date_str + '.csv'):
                add_deltas(model_name, date_str, None)
            date += timedelta(days=1)

if __name__ == '__main__':
    main()
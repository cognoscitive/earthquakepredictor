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
Stage 3:  Auto-updating Prediction Website
"""
import os
import sys
import time
import random
from datetime import datetime, timedelta
from ftplib import FTP
import earthquake_rnn
import create_map
import list_earthquakes
import config

def upload_prediction(date, model_names):
    uploading = True
    while(uploading):
        try:
            print('Uploading via FTP...')
            session = FTP(config.ftp_url) # connect to host, default port
            session.login(config.username, config.password)
            for model_name in model_names:
                file = open('prediction/' + model_name + '/' + 'map_' + date + '.png','rb') # file to send
                session.storbinary('STOR ' + config.web_dir + 'images/' + model_name + '/' + 'map_' + date + '.png', file) # send the file
                file.close() # close file and FTP

                file = open('prediction/' + model_name + '/' + 'list_' + date + '.csv','rb') # file to send
                session.storbinary('STOR ' + config.web_dir + 'data/' + model_name + '/' + 'list_' + date + '.csv', file) # send the file
                file.close() # close file and FTP

                file = open('prediction/' + model_name + '/' + 'date.txt','rb') # file to send
                session.storbinary('STOR ' + config.web_dir + model_name + '/' + 'date.txt', file) # send the file
                file.close() # close file and FTP

                file = open('prediction/' + model_name + '/' + 'archive.txt','rb') # file to send
                session.storbinary('STOR ' + config.web_dir + model_name + '/' + 'archive.txt', file) # send the file
                file.close() # close file and FTP

            session.quit()
            print('Done upload!')
            uploading = False
        except:
            print(sys.exc_info())
            time.sleep(random.randint(10, 30))

def update_prediction(date, model_names):
    print('Updating prediction...')
    for model_name in model_names:
        create_map.create_map('prediction/' + model_name + '/' + date + '.png', filename='prediction/' + model_name + '/' + 'map_' + date + '.png')
        list_earthquakes.list_earthquakes('prediction/' + model_name + '/' + date + '.csv', filename='prediction/' + model_name + '/' + 'list_' + date + '.csv')
        list_earthquakes.add_deltas(model_name, date, (datetime.strptime(date, '%Y-%m-%d') - timedelta(days=1)).strftime('%Y-%m-%d'))
        with open('prediction/' + model_name + '/' + 'date.txt', 'w') as f:
            f.write(date)
        with open('prediction/' + model_name + '/' + 'archive.txt', 'a') as f:
            f.write('\n' + date)
    print('Prediction updated successfully!')

def generate_prediction(today, model_names):
    print('Generating prediction...')
    yesterday = today - timedelta(days=1)
    yesterday_date = str(yesterday.year) + '-' + earthquake_rnn.pad_number(yesterday.month) + '-' + earthquake_rnn.pad_number(yesterday.day)
    #earthquake_rnn.generate_seedset(yesterday)
    old_data = earthquake_rnn.load_seedset()
    while not os.path.isfile(earthquake_rnn.DATA_PATH + yesterday_date + '.csv'):
        try:
            earthquake_rnn.download_day(yesterday.year, yesterday.month, yesterday.day)
        except:
            print(sys.exc_info())
            time.sleep(random.randint(10, 30))
    earthquake_rnn.generate_seedset(today)
    new_data = earthquake_rnn.load_seedset()
    earthquake_rnn.predict(old_data, new_data, models=model_names, today=today)

def main():
    with open('date.txt', 'r') as f:
        text = f.read()
    last_updated = datetime.strptime(text, '%Y-%m-%d')
    while(True):
        today = datetime.utcnow()
        today.replace(hour=0, minute=0, second=0, microsecond=0)
        if today.year != last_updated.year or today.month != last_updated.month or today.day != last_updated.day:
            last_updated += timedelta(days=1)
            date = last_updated.strftime('%Y-%m-%d')
            generate_prediction(last_updated, earthquake_rnn.MODEL_NAMES)
            update_prediction(date, earthquake_rnn.MODEL_NAMES)
            upload_prediction(date, earthquake_rnn.MODEL_NAMES)
            with open('date.txt', 'w') as f:
                f.write(date)
        time.sleep(300)
        #time.sleep(random.randint(3, 5))

if __name__ == '__main__':
    main()
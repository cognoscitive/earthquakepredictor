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
Prediction Map
"""
import numpy as np
from PIL import Image

def load_world_map(scale):
    print('Loading world map...')
    if scale==6:
        earth_map = np.array(Image.open('map/1080.jpg'))
    if scale==4:
        earth_map = np.array(Image.open('map/720.jpg'))
    if scale==2:
        earth_map = np.array(Image.open('map/360.jpg'))
    return earth_map

def merge_maps(background_image, earthquake_matrix, scale, min_mag):
    print('Merging earthquakes with world map...')
    earth_map = background_image
    quake_map = Image.new('RGBA',(181*scale,91*scale),(0,0,0,0))
    border_map = Image.new('RGBA',(181*scale,91*scale),(0,0,0,0))
    quake_map = np.array(quake_map)
    border_map = np.array(border_map)
    for y in range(91):
        for x in range(181):
            if earthquake_matrix[90-y,x] >= (min_mag + 1) / 11 and earthquake_matrix[90-y,x] > 0:
                for i in range(scale):
                    for j in range(scale):
                        quake_map[y*scale+i,x*scale+j] = (0,0,0,earthquake_matrix[90-y,x])
                        if scale > 4:
                            if i == 0 or i == scale-1 or j == 0 or j == scale-1:
                                border_map[y*scale+i,x*scale+j] = (0,0,0,0)
                            elif i == 1 or i == scale-2 or j == 1 or j == scale-2:
                                border_map[y*scale+i,x*scale+j] = (0,0,0,255)
                        elif scale > 2:
                            if i == 0 or i == scale-1 or j == 0 or j == scale-1:
                                border_map[y*scale+i,x*scale+j] = (0,0,0,255)
    earth_map = Image.fromarray(earth_map)
    yellow_map = Image.new('RGB',(181*scale,91*scale),(255,255,0))
    red_map = Image.new('RGB',(181*scale,91*scale),(255,0,0))
    border_map = Image.fromarray(border_map)
    quake_map = Image.fromarray(quake_map)
    earth_map = Image.composite(red_map, earth_map, quake_map)
    earth_map = Image.composite(yellow_map, earth_map, border_map)
    earth_map = np.array(earth_map)
    return earth_map

def create_legend(image, scale):
    x1 = 5
    x2 = 45
    y1 = 81
    y2 = 84
    legend = Image.new('RGBA',(181*scale,91*scale),(0,0,0,0))
    border = Image.new('RGBA',(181*scale,91*scale),(0,0,0,0))
    legend = np.array(legend)
    border = np.array(border)
    for y in range(91):
        for x in range(181):
            if x >= x1 and x <= x2 and y >= y1 and y <= y2:
                for i in range(scale):
                    for j in range(scale):
                        legend[y*scale+i,x*scale+j] = (0,0,0, (float(x-x1)/float(x2-x1))*255)
                        if x == x1 and j == 0 or x == x2 and j == scale-1 or y == y1 and i == 0 or y == y2 and i == scale-1:
                            border[y*scale+i,x*scale+j] = (0,0,0,255)
                        elif j == (float(x-x1)/float(x2-x1))*(scale-1) and (i == 1 or i == scale-2):
                            border[y*scale+i,x*scale+j] = (0,0,0,255)
                        elif (float(x-x1) % (float(x2-x1) / 10)) == 0 and (y == y1 and i == 1 or y == y2 and i == scale-2):
                            border[y*scale+i,x*scale+j] = (0,0,0,255)
    red = Image.new('RGB',(181*scale,91*scale),(255,0,0))
    yellow = Image.new('RGB',(181*scale,91*scale),(255,255,0))
    legend = Image.fromarray(legend)
    border = Image.fromarray(border)
    image = Image.fromarray(image)
    image = Image.composite(red,image,legend)
    image = Image.composite(yellow,image,border)
    image = np.array(image)
    return image

def create_map(earthquake_img, scale=6, min_mag=-1.0,filename='prediction/prediction_map.png'):
    print('Creating map of earthquakes...')
    bg = load_world_map(scale)
    eq = np.array(Image.open(earthquake_img))
    earthquake_map = merge_maps(bg, eq, scale, min_mag)
    earthquake_map = create_legend(earthquake_map, scale)
    Image.fromarray(earthquake_map).save(filename)
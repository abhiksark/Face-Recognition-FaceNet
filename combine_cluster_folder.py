#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
###################################################################
# File Name: comvine_cluster_folder.py
# Author: Abhik Sarkar
# mail: abhiksark@gmail.com
# Created Time: Thu Oct  5 02:27:35 2017 IST
###################################################################
"""

import os

folder_numbers = 10

for i in range(folder_numbers):
    for j in os.listdir("./faces/group_photos/" + str(i)):
        os.rename("./faces/group_photos/" + str(i)+"/" +
                  str(j), "./faces/group_photos/"+str(j))
    os.rmdir("./faces/group_photos/" + str(i))

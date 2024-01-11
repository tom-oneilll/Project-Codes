#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  3 23:04:02 2023

@author: toneill
"""

import ROOT
import uproot
import numpy as np
import pandas

file = uproot.open("trackana_p234_2004.root")


print(file.keys())

protonTree = file["testbeamtrackana/protonTree"]

print(protonTree.keys())
"""
x= protonTree["prong_x"].array()

proton_xprong= np.asarray(x)

proton_yprong = np.asarray(file["protonTree"]["prong_y"])
proton_zprong = np.asarray(file["protonTree"]["prong_y"])

proton_prongs = file["protonTree"].arrays(["period_n","prong_length","prong_energy","prong_nhit","prong_mintime","prong_x","prong_y","prong_z","prong_n","prong_dr"], library ="pd")

proton = file["protonTree"].arrays(library="pd")
pimu = file["pimuTree"].arrays(library="pd")
electron = file["electronTree"].arrays(library="pd")
kaon = file["kaonTree"].arrays(library="pd")
other = file["otherTree"].arrays(library="pd")
"""





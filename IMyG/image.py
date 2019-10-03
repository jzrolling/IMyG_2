__author__ = "jz-rolling"

import nd2reader as nd2
import tifffile
import os
import re
import numpy as np
from IMyG.helper_func import *
from IMyG import microcolony
import IMyG.config as conf
from IMyG.remove_background import Rolling_ball_bg_subtraction

class image():
    def __init__(self,input_file,image_type = "nd2"):
        self.pixel_microns = 1
        self.channels = None
        self.shape = None
        self.fl_img = {}
        self.ph_img = None
        self.ph_filtered = None
        self.ph_binary = None
        self.shape_indexed = None
        #self.shape_idx_binary = None deprecated
        self.shape_indexed_smoothed = None
        self.microcolony_labels = None
        self.microcolony_info = None
        self.sobel = None
        self.celllike = []
        self.cells = []
        self.microcolonies = []
        self.shape = None
        self.global_fl_background = []

        if image_type == "nd2":
            img = nd2.Nd2(input_file)
            self.pixel_microns = img.pixel_microns
            self.channels = img.channels
            self.shape = (img.width, img.height)
            self.min_length = int(round(conf.min_length / self.pixel_microns))
            self.max_length = int(round(conf.max_length / self.pixel_microns))
            for channel in self.channels:
                if re.search("ph",channel,re.IGNORECASE):
                    self.ph_img = np.array(img[img.channels.index(channel)]).astype(np.uint16)
                else:
                    self.fl_img[channel] = np.array(img[img.channels.index(channel)]).astype(np.uint16)
        #tif reader function not awailable at the moment 091519

    def preprocess_ph(self,normalize = True,\
                      adjust_gamma = True,gamma = 1.0,\
                      ph_high_pass = conf.ph_high_pass,\
                      ph_low_pass = conf.ph_low_pass):
        ph_fft = fft(self.ph_img.copy(),subtract_mean=True)
        ph_fft_filters = bandpass_filter(pixel_microns = self.pixel_microns,\
                                         img_width=self.shape[0],img_height=self.shape[1],\
                                         high_pass_width=ph_high_pass,low_pass_width=ph_low_pass)
        ph_fft_reconstructed = fft_reconstruction(ph_fft,ph_fft_filters)
        if normalize:
            ph_fft_reconstructed = normalize_img(ph_fft_reconstructed,adjust_gamma=adjust_gamma,gamma=gamma)
        self.ph_filtered = ph_fft_reconstructed
        sobel = filters.gaussian(filters.sobel(self.ph_filtered),sigma = 1)
        self.sobel = (sobel - sobel.min() + 1) / (sobel.max() + 1)
        del ph_fft,ph_fft_filters,ph_fft_reconstructed


    def preprocess_fl(self,normalize = False,adjust_gamma = False,gamma = 1.0):
        #remove background fluroescence using a gausian filter
        for channel,data in self.fl_img.items():
            #deprecated, use rolling ball method instead
            #fl_bg = (filters.gaussian(img,sigma = bg_sigma,preserve_range=True)).astype(int)
            #fl_bg_subtracted[fl_bg_subtracted<=0] = 0
            fl_bg_subtracted = Rolling_ball_bg_subtraction(data)
            if normalize:
                fl_bg_subtracted = normalize_img(normalize_img,adjust_gamma=adjust_gamma,gamma=gamma)
            self.fl_img[channel] = fl_bg_subtracted
            #del fl_bg_subtracted

    def raw_segmentation(self):
        self.ph_binary, \
        self.shape_indexed, \
        self.microcolony_labels, \
        self.microcolony_info = init_Segmentation(self.ph_filtered)
        self.shape_indexed_smoothed = filters.median(self.shape_indexed,morphology.disk(1)).astype(np.uint8)
        #self.shape_indexed_smoothed = filters.median(self.shape_indexed)
        for channel in self.channels:
            self.global_fl_background[channel] = round(np.average(self.fl_img[channel][self.ph_binary == 0]),1)


    def process_microcolonies(self):
        warnings.filterwarnings("ignore")
        for m in self.microcolony_info:
            colony = microcolony.microcolony(self,m)
            colony.create_seeds(self, apply_median_filter=True)
            if conf.advanced:
                colony.remove_false_boundries_advanced()
            else:
                colony.remove_false_boundries_fast()
            colony.extract_celllike_particles(self)

    def optimize_single_cells(self):
        warnings.filterwarnings("ignore")
        for cell in self.cells:
            try:
                cell.skeleton_optimization()
            except:
                print("Error found while optimizing skeleton!")
                cell.is_good_baby = False

    def measure(self):
        warnings.filterwarnings("ignore")
        for cell in self.cells:
            try:
                cell.measure(self)
            except:
                print("Error found!")
                cell.is_good_baby = False

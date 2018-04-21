# -*- coding:utf-8 -*-
"""  
#====#====#====#====
# Project Name:     seek 
# File Name:        mitigation 
# Date:             3/17/18 9:10 PM 
# Using IDE:        PyCharm Community Edition  
# From HomePage:    https://github.com/DuFanXin/seek
# Author:           DuFanXin 
# BlogPage:         http://blog.csdn.net/qq_30239975  
# E-mail:           18672969179@163.com
# Copyright (c) 2018, All Rights Reserved.
#====#====#====#==== 
"""
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import matplotlib
from seek.mitigation import sum_threshold
import h5py
TEST_DIRECTORY = '../Res-Unet/data_set/test'
TEST_RESULT_DIRECTORY = '../Res-Unet/data_set/test_result'
GROUND_TRUTH_DIRECTORY = '../Res-Unet/data_set/label'
# file_path = '/home/dufanxin/PycharmProjects/hide&seek/bgs_example_data/seek_cache/' \
#             'HIMap_RSG7M_A1_24_MP_PXX_Z0_C0-M9703A_DPUA_20160321_000000.h5'
# file_path = '/home/dufanxin/PycharmProjects/tf_unet/data_set.h5' \
# rfi_mask = sum_threshold.get_rfi_mask(tod=np.ma.array(data),
#                                           chi_1=200000, sm_kwargs=sum_threshold.get_sm_kwargs(40, 20, 15, 7.5),
#                                           di_kwargs=sum_threshold.get_di_kwrags(3, 7)
#                                           )


def do_sum_threshold():
    with h5py.File('./sum_threshold.h5', 'w') as file_to_write:
        with h5py.File('../Res-Unet/data_set/data_set.h5', 'r') as file_to_read:
            for i in range(0, 30):
                tod = file_to_read['%04d/tod' % i].value
                rfi_mask_threshold = sum_threshold.get_rfi_mask(tod=np.ma.array(tod),
                                                                chi_1=1e10,
                                                                sm_kwargs=sum_threshold.get_sm_kwargs(20, 20, 15, 7.5),
                                                                di_kwargs=sum_threshold.get_di_kwrags(2, 2),
                                                                plotting=False
                                                                )
                rfi_mask_threshold = np.asarray(a=rfi_mask_threshold, dtype=np.uint8)
                file_to_write['%04d/prediction' % i] = rfi_mask_threshold
                file_to_write['%04d/ground_truth' % i] = file_to_read['%04d/rfi_mask' % i].value
                file_to_write['%04d/tod' % i] = file_to_read['%04d/tod' % i].value


def show():
    with h5py.File("../hide/2016/03/21/TEST_MP_PXX_20160321_001500.h5", "r") as fp:
        # tod = cv2.imread(os.path.join(TEST_DIRECTORY, '*.tif'))
        # ground_truth = cv2.imread(os.path.join(GROUND_TRUTH_DIRECTORY, '*.tif'))
        tod = fp["P/Phase1"].value
        rfi = fp["RFI/Phase0"].value
        time = fp["TIME"].value

        plt.subplot(231)
        plt.title(s='TOD')
        plt.imshow(tod, aspect="auto",
                   extent=(time[0], time[-1], 990, 1260),
                   cmap="gist_earth",  norm=matplotlib.colors.LogNorm()
                   )

        plt.subplot(232)
        plt.title(s='RFI')
        plt.imshow(rfi, aspect="auto",
                   extent=(time[0], time[-1], 990, 1260),
                   cmap="gist_earth",  norm=matplotlib.colors.LogNorm()
                   )

        plt.subplot(233)
        plt.title(s='real_data')
        plt.imshow(tod - rfi, aspect="auto",
                   extent=(time[0], time[-1], 990, 1260),
                   cmap="gist_earth", norm=matplotlib.colors.LogNorm()
                   )

        rfi_mask_threshold = sum_threshold.get_rfi_mask(tod=np.ma.array(tod),
                                                        chi_1=1e10,
                                                        sm_kwargs=sum_threshold.get_sm_kwargs(20, 20, 15, 7.5),
                                                        di_kwargs=sum_threshold.get_di_kwrags(2, 2),
                                                        plotting=False
                                                        )
        rfi_mask_threshold = np.asarray(a=rfi_mask_threshold, dtype=np.uint8)
        plt.subplot(234)
        plt.title(s='sum_threshold')
        plt.imshow(rfi_mask_threshold, aspect="auto",
                   extent=(time[0], time[-1], 990, 1260),
                   cmap="gist_earth"
                   )

        rfi_mask = np.abs(rfi) >= 500
        # print(rfi_mask)
        plt.subplot(235)
        plt.title(s='rfi_mask')
        plt.imshow(rfi_mask, aspect="auto",
                   extent=(time[0], time[-1], 990, 1260),
                   cmap="gist_earth"
                   )

        plt.subplot(235)
        plt.title(s='res_unet_rfi_mask')

        # plt.colorbar()
        plt.show()
    # # cv2.imshow('p', tod)
    # # cv2.waitKey(0)


def show_result():
    with h5py.File('./sum_threshold.h5', 'r') as fp:
        i = 0
        # tod = cv2.imread(os.path.join(TEST_DIRECTORY, '*.tif'))
        # ground_truth = cv2.imread(os.path.join(GROUND_TRUTH_DIRECTORY, '*.tif'))
        tod = fp['%04d/tod' % i].value
        ground_truth = fp['%04d/ground_truth' % i].value
        prediction = fp['%04d/prediction' % i].value
        # rfi = fp["RFI/Phase0"].value
        # time = fp["TIME"].value

        plt.subplot(231)
        plt.title(s='TOD')
        plt.imshow(tod, aspect="auto",
                   extent=(0, 255, 990, 1260),
                   cmap="gist_earth",  norm=matplotlib.colors.LogNorm()
                   )

        plt.subplot(232)
        plt.title(s='ground_truth')
        plt.imshow(ground_truth, aspect="auto",
                   extent=(0, 255, 990, 1260),
                   cmap="gist_earth"
                   )
        #
        plt.subplot(233)
        plt.title(s='prediction')
        plt.imshow(prediction, aspect="auto",
                   extent=(0, 255, 990, 1260),
                   cmap="gist_earth"
                   )

        plt.show()


def socre():
    with h5py.File(os.path.join(TEST_RESULT_DIRECTORY, 'SumThreshold_Score.h5'), "w") as fp:
        for i in range(54):
            tod = cv2.imread(os.path.join(TEST_DIRECTORY, '%d.tif' % i), flags=0)
            ground_truth = cv2.imread(os.path.join(GROUND_TRUTH_DIRECTORY, '%d.tif' % i), flags=0)
            # tod = tod[:, 4 * 256:4 * 256 + 256]
            # rfi = fp["RFI/Phase0"].value
            # time = fp["time"].value
            data = tod
            print(tod.shape)
            rfi_mask_sumthreshold = sum_threshold.get_rfi_mask(tod=np.ma.array(data),
                                                               chi_1=20,
                                                               sm_kwargs=sum_threshold.get_sm_kwargs(10, 10, 15, 7.5),
                                                               di_kwargs=sum_threshold.get_di_kwrags(1, 1),
                                                               plotting=False
                                                               )
            fp['%d/predict' % i] = np.asarray(a=rfi_mask_sumthreshold, dtype=np.uint8)
            fp['%d/ground_truth' % i] = np.asarray(a=ground_truth, dtype=np.uint8)


if __name__ == '__main__':
    # do_sum_threshold()
    # show_result()
    show()
    # socre()

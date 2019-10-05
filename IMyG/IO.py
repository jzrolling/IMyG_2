__author__ = "jz-rolling"

import tifffile
import os,glob
import multiprocessing as mp
import re
import numpy as np
from IMyG.helper_func import *
from IMyG import image
import pickle as pk

def data_dump_txt(image,output_path,header):
    with open(output_path+header+"_data.txt",'w') as data:
        for cell in image.cells:
            if cell.is_good_baby:
                data.write("File header: {}, Colony label: {}, cell label: {}\n".format(header,\
                                                                                        cell.colony_label,\
                                                                                        cell.cell_label))
                data.write("Bounding box: {},{},{},{}\n".format(cell.bbox[0],\
                                                                cell.bbox[1],\
                                                                cell.bbox[2],\
                                                                cell.bbox[3]))
                data.write("Cell length: {} um, perimeter: {} um\n".format(round(cell.length,2),round(cell.perimeter_precise,2)))
                data.write("Cell eccentricity: {}, solidity: {}, sphericity: {}, compactness: {}, sinuosity: {}\n".format(round(cell.eccentricity,2),\
                                                                                                                          round(cell.solidity,2),\
                                                                                                                          round(cell.sphericity,2),\
                                                                                                                          round(cell.compactness,2),\
                                                                                                                          round(cell.sinuosity,2)))
                if len(cell.measure_along_midline) != 0:
                    midline = "Cell midline x-axis coordinates: \n{}\nCell midline y-axis coordinates: \n{}\n"
                    midline_formated = midline.format(np.array2string(cell.midline.T[0],precision=2,separator=',',max_line_width = 20000)[1:-1],\
                                                      np.array2string(cell.midline.T[1],precision=2,separator=',',max_line_width = 20000)[1:-1])
                    data.write(midline_formated)
                    contour = "Cell contour x-axis coordinates: \n{}\nCell contour y-axis coordinates: \n{}\n"
                    contour_formated = contour.format(np.array2string(cell.optimized_contour[0].T[0], precision=2, separator=',', max_line_width=20000)[1:-1],\
                                                      np.array2string(cell.optimized_contour[0].T[1], precision=2, separator=',', max_line_width=20000)[1:-1])
                    data.write(contour_formated)
                    phase = "Phase contrast image: \n{}\n".format(np.array2string(cell.measure_along_midline["Phase"],\
                                                                                  precision=1,separator=',',\
                                                                                  max_line_width = 20000,suppress_small=True)[1:-1])
                    data.write(phase)
                    shape_idx = "Shape indexed image: \n{}\n".format(np.array2string(cell.measure_along_midline["Shape_indexed"],\
                                                                                     precision=1,separator=',',\
                                                                                     max_line_width = 20000,suppress_small=True)[1:-1])
                    data.write(shape_idx)
                    for channel,val in image.fl_img.items():
                        fl = "{} channel: \n{}\n".format(channel,np.array2string(cell.measure_along_midline[channel],\
                                                                                 precision=1,separator=',',\
                                                                                 max_line_width = 20000,\
                                                                                 suppress_small=True)[1:-1])
                        data.write(fl)
                        if len(cell.measure_along_contour) != 0:
                            fl_contour = "{} channel signal measured along contour: \n{}\n".format(channel,np.array2string(cell.measure_along_contour[channel],\
                                                                                                                         precision=1, separator=',',\
                                                                                                                         max_line_width=20000,\
                                                                                                                         suppress_small=True)[1:-1])
                            data.write(fl_contour)
                        if len(cell.fl_straighten) != 0:
                            data.write("{} channel signal of a normalized cell:\n".format(channel))
                            (w_average, w_std, h_average, h_std) = cell.fl_straighten[channel]
                            data.write("Lateral axis average: \n{}\n".format(np.array2string(w_average, \
                                                                                             precision=1, separator=',',\
                                                                                             max_line_width=20000,\
                                                                                             suppress_small=True)[1:-1]))
                            data.write("Lateral axis standard deviation: \n{}\n".format(np.array2string(w_std, \
                                                                                                        precision=1,
                                                                                                        separator=',',\
                                                                                                        max_line_width=20000,\
                                                                                                        suppress_small=True)[1:-1]))
                            data.write("Axial axis average: \n{}\n".format(np.array2string(h_average,\
                                                                                           precision=1,
                                                                                           separator=',',\
                                                                                           max_line_width=20000,\
                                                                                           suppress_small=True)[1:-1]))
                            data.write("Axial axis standard deviation: \n{}\n".format(np.array2string(h_std,\
                                                                                                      precision=1,
                                                                                                      separator=',',\
                                                                                                      max_line_width=20000,\
                                                                                                      suppress_small=True)[1:-1]))
    data.close()

def data_dump_image(image,output_path,header):
    tifffile.imsave(output_path+header+"_phase_bg_filtered.tif",\
                    image.ph_filtered.astype(np.uint16),imagej = True)
    tifffile.imsave(output_path + header + "_phase_mask.tif", \
                    (image.ph_binary*1*254).astype(np.uint8), imagej=True)
    for channel,fl_image in image.fl_img.items():
        tifffile.imsave(output_path + header +"_"+ channel+"_bg_filtered.tif",\
                        fl_image.astype(np.uint16), imagej=True)

def data_dump_sample_cells(image,output_path,header,max_number = 5):
    idx_list = np.arange(len(image.cells))
    np.random.shuffle(idx_list)
    counter = 0
    for i in idx_list:
        cell = image.cells[i]
        if counter >= max_number:
            break
        if cell.cell_label == 0:
            length = cell.length
            if conf.sample_cell_min_length < length < conf.sample_cell_max_length:
                cell.plot_advanced(image,savefig=True,output_path=output_path,header=header)
                counter += 1

def process_nd2(nd2file,output_path,header,sample_cells = True):
    nd = image.image(nd2file)
    nd.preprocess_ph()
    nd.preprocess_fl()
    nd.raw_segmentation()
    nd.process_microcolonies()
    nd.optimize_single_cells()
    nd.measure()
    # print function for later
    data_dump_txt(nd,output_path,header)
    data_dump_image(nd,output_path,header)
    if sample_cells:
        data_dump_sample_cells(nd,output_path,header)

def process_nd2_no_output(nd2file,plate_idx=0):
    nd = image.image(nd2file,plate_index=plate_idx)
    nd.preprocess_ph()
    nd.preprocess_fl()
    nd.raw_segmentation()
    nd.process_microcolonies()
    nd.optimize_single_cells()
    nd.measure()
    return(nd)

def pickel_dump(nd2file, output_path, header, plate_idx, sample_cells = True):
    nd = process_nd2_no_output(nd2file,plate_idx = plate_idx)
    # print function for later
    data_dump_image(nd,output_path,header)
    output = "{}{}_sample_{}_cells.pk".format(output_path,header,plate_idx)
    pk.dump(nd.cells, open(output, "wb"))
    if sample_cells:
        data_dump_sample_cells(nd,output_path,header)


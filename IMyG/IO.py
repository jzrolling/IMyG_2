__author__ = "jz-rolling"

import tifffile
import os,glob
import multiprocessing as mp
import re
import numpy as np
from IMyG.helper_func import *
from IMyG.metrics import *
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
                cell.plot_advanced(savefig=True,output_path=output_path,header=header)
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
    data_dump_image(nd,output_path,"{}_{}".format(header,plate_idx))
    output = "{}{}_sample_{}_cells.pk".format(output_path,header,plate_idx)
    pk.dump(nd.cells, open(output, "wb"))
    if sample_cells:
        data_dump_sample_cells(nd,output_path,"{}_{}".format(header,plate_idx))

def read_from_folder(input_path,output_path,header,n_cores = 4):
    files = glob.glob(input_path+"*.nd2")
    if len(files) == 0:
        print("Empty folder!")
    else:
        print("%d .nd2 files found in input directory."%(len(files)))
    if not os.path.isdir(output_path):
        os.mkdir(output_path)
    count = 0
    for i in range(0,len(files),n_cores):
        processes = []
        for j in range(0,min(n_cores,len(files)-i)):
            file = files[i+j]
            processes.append(mp.Process(target=pickel_dump,args=(file,output_path,header,count)))
            count += 1
        for p in processes:
            p.start()
        for p in processes:
            p.join()
    print("{} nd2 files processed!".format(len(files)))

def merge_pks(input_path):
    files = glob.glob(input_path + "*.pk")
    if len(files) == 0:
        raise ValueError("No pickled files found in {}".format(input_path))
    else:
        pks = []
        for file in files:
            pks += pk.load(open(file, "rb"))
    return pks

def pad_data(array,length,max_len = 15,max_pixel=512,normalize = True,base = 50):
    pixelated_length = int(round(length*max_pixel*0.5/max_len))*2
    pad_length = int((max_pixel-pixelated_length)/2)
    interpolated = np.interp(np.linspace(0,1,pixelated_length),\
                             np.linspace(0,1,len(array)),array)
    if normalize:
        interpolated = (interpolated-interpolated.min()+base)/(interpolated.max()-interpolated.min()+base)
    padded = np.pad(interpolated,(pad_length,pad_length),'constant', constant_values=(0,0))
    return padded

def create_metrics_relative_MIDX(pooled_data,output_path,header,channel):
    padded_fl_axial = []
    fl_lateral=[]
    data_output = []
    temp = True
    counter = 0
    column_name = ["Index","Image","Colony","Cell","Curvature","Length [um]","Cell width mean [um]",\
                   "Cell width std [um]","Perimeter [um]","Fluorescent intensity mean","Fluorescent intensity std",\
                   "SNR","Membrane index","Centrifugality","Signal variability","Signal variability on membrane",\
                   "Body symmetry","Pole symmetry"]
    for cell in pooled_data:
        try:
        #if temp:
            #Cell length:
            cell_length = np.round(cell.length,2)
            #Cell width:
            cell_width = cell.width*cell.pixel_microns
            #Ignore cell caps when measuring the mean of cell width,
            cell_width_mean = round(np.mean(cell_width[5:-5]),2)
            cell_width_std = round(np.std(cell_width),2)
            #Perimeter:
            cell_perimeter = round(cell.perimeter_precise,2)
            #Basic fluroescence metrics
            cell_fluorescent_intensity_mean = round(cell.mean_fl_intensity[channel],1)
            cell_fluorescent_intensity_std = round(np.std(cell.fl_img[channel][cell.mask>0]),1)
            cell_fluorescent_SNR = round(cell.SNR[channel],2)
            #Measure along midline
            midline_measure = straighten_cell(cell.fl_img[channel],cell.mask,cell.midline,half_width_by_pixel=3)
            midline_average = np.average(midline_measure,axis=0)
            half_cell = int(0.5*len(midline_average))
            if midline_average[:half_cell].mean() < midline_average[-half_cell:].mean():
                midline_average = np.flip(midline_average)
            #Measure along lateral axis
            fl_straighten_lateral = cell.fl_straighten[channel][0]
            #Measure along contour
            fl_contour = cell.measure_along_contour[channel]
            #Membrane idx
            _r,membrane_idx = measure_membrane_idx(fl_straighten_lateral)
            membrane_idx = membrane_idx/cell_width_mean
            #Segmentate cells based on weighted centers
            l1,l2,r1,r2,h1,h2,pole,center = segmented_mean(midline_average)
            #Measure centrifugality along cell axis
            centrifugality = measure_centrifugality(pole,center)
            variability_axial = measure_variability(midline_average)
            variability_contour = measure_variability(fl_contour)
            #Measure symmetry
            body_symmetry = measure_symmetry(h1,h2)
            pole_symmetry = measure_symmetry(l1,r2)
            #Create info_table
            data = [counter,cell.plate_idx,cell.colony_label,cell.cell_label,cell.curvature,\
                    cell_length,cell_width_mean,cell_width_std,cell_perimeter,cell_fluorescent_intensity_mean,\
                    cell_fluorescent_intensity_std,cell_fluorescent_SNR,membrane_idx,centrifugality,\
                    variability_axial,variability_contour,body_symmetry,pole_symmetry]
            padded_fl_axial.append(pad_data(midline_average,cell.length,normalize=True))
            fl_lateral.append(np.interp(np.linspace(0,1,100),\
                                        np.linspace(0,1,len(fl_straighten_lateral)),\
                                        fl_straighten_lateral))
            data_output.append(data)
            counter += 1
        except:
            continue
    df = pd.DataFrame(data_output,columns=column_name)
    df.to_excel(output_path+"{}_{}_summary_normalized.xls".format(header,channel))
    np.save(output_path+"{}_{}_padded_demograph.npy".format(header,channel),np.array(padded_fl_axial),allow_pickle=True)
    np.save(output_path+"{}_{}__mean_lateral.npy".format(header,channel),np.array(fl_lateral),allow_pickle=True)
    return df,np.array(padded_fl_axial),np.array(fl_lateral)

def initiate_projection():
    from skimage import measure
    width = 75
    heights = [250, 350, 450, 550, 650]
    pad = np.zeros((50, width))
    gap = np.zeros((750, 10))
    padded = [gap]
    xt_list, yt_list, nxt_list, nyt_list = [], [], [], []
    for i in range(len(heights)):
        a = create_canvas(width=width, height=heights[i])
        half_pad = np.tile(pad, (len(heights) - i, 1))
        m_pad = np.concatenate([half_pad, a, half_pad], axis=0)
        m_pad = np.concatenate([m_pad, gap], axis=1)
        xt, yt, norm_xt, norm_yt = create_mesh(m_pad)
        xt_list.append(xt)
        yt_list.append(yt)
        nxt_list.append(norm_xt)
        nyt_list.append(norm_yt)
        padded.append(m_pad)
    contours = measure.find_contours(np.concatenate(padded, axis=1), level=0)
    optimized_outline = []
    for contour in contours:
        optimized_outline.append(spline_approximation(contour, n=2 * len(contour)))
    return padded,xt_list, yt_list, nxt_list, nyt_list,optimized_outline

def project_by_length(folder,pks,header,channel,all_cells = True,re_align = True,max_th=0.5,min_cells=5):
    padded, xt_list, yt_list, nxt_list, nyt_list,optimized_outline = initiate_projection()
    well_paints=padded.copy()
    for m in well_paints:
        m*=0
    group_count = [0,0,0,0,0]
    for cell in pks:
        if not all_cells:
            if cell.cell_label == 0:
                accept = True
            else:
                accept = False
        else:
            accept = True
        if cell.is_good_baby*accept:
            if cell.SNR[channel] > 3:
                try:
                    group = group_by_length(cell.length)
                    group_count[group-1] += 1
                    canvas = padded[group].copy()
                    data = straighten_cell_normalize_width(cell.fl_img[channel],\
                                                       cell.width,cell.midline,\
                                                       remove_cap=0,pad=1)
                    normalized_data = normalize_data(data,re_align=re_align,max_th=max_th)
                    #coords = feature.peak_local_max(normalized_data,min_distance=4)
                    #peaks = np.zeros(normalized_data.shape)
                    #peaks[coords] += 1
                    well_paints[group]  += project_image(xt_list[group-1],\
                                                         yt_list[group-1],\
                                                         nxt_list[group-1],\
                                                         nyt_list[group-1],\
                                                         canvas,normalized_data.T)
                except:
                    continue
    for i in range(len(well_paints)):
        graph = well_paints[i]
        if graph.max()>0:
            min_val = graph[graph>0].min()
            normalized = (graph-min_val)/(graph.max()-min_val)
            #normalized = subtract_alignment_bias(normalized)
            normalized[graph==0] = 0
            if (i > 0) and (group_count[i-1] < min_cells):
                normalized *= 0
            well_paints[i] = normalized
    painttttt = np.concatenate(well_paints,axis = 1)
    fig = plt.figure(figsize=(8,8))
    plt.imshow(painttttt,cmap = "viridis")
    for outline in optimized_outline:
        plt.plot(outline.T[1],outline.T[0],c="salmon",alpha = 0.5)
    fig.savefig(folder+"projection_{}.png".format(header),\
                bbox_inches = "tight")
    plt.close

def project_by_percentile(folder,pks,header,channel,all_cells = True,re_align = True,max_th=0.5,min_cells = 5):
    padded, xt_list, yt_list, nxt_list, nyt_list,optimized_outline = initiate_projection()
    well_paints=padded.copy()
    for m in well_paints:
        m*=0
    group_count = [0,0,0,0,0]
    group_val = []
    filtered_pks = []
    filtered_length = []
    for cell in pks:
        if not all_cells:
            if cell.cell_label == 0:
                accept = True
            else:
                accept = False
        else:
            accept = True
        if cell.is_good_baby*accept:
            if cell.SNR[channel] > 2:
                filtered_pks.append(cell)
                filtered_length.append(cell.length)
    print("{} cells accounted".format(len(filtered_length)))
    for i in range(1,5):
        group_val.append(np.percentile(np.array(filtered_length),20*i))

    for cell in filtered_pks:
        try:
            group = group_by_length(cell.length,groups=group_val)
            group_count[group-1] += 1
            canvas = padded[group].copy()
            data = straighten_cell_normalize_width(cell.fl_img[channel],\
                                                   cell.width,cell.midline,\
                                                   remove_cap=0,pad=1)
            normalized_data = normalize_data(data,re_align=re_align,max_th=max_th)
            well_paints[group]  += project_image(xt_list[group-1],\
                                                 yt_list[group-1],\
                                                 nxt_list[group-1],\
                                                 nyt_list[group-1],\
                                                 canvas,normalized_data.T)
        except:
            continue
    for i in range(len(well_paints)):
        graph = well_paints[i]
        if graph.max()>0:
            min_val = graph[graph>0].min()
            normalized = (graph-min_val)/(graph.max()-min_val)
            normalized[graph==0] = 0
            if (i > 0) and (group_count[i-1] < min_cells):
                normalized *= 0
            well_paints[i] = normalized
    painttttt = np.concatenate(well_paints,axis = 1)
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    plt.imshow(painttttt,cmap = "viridis")
    for outline in optimized_outline:
        ax.plot(outline.T[1],outline.T[0],c="salmon",alpha = 0.5)
    fig.savefig(folder+"projection_{}.png".format(header),\
                bbox_inches = "tight")
    plt.close



def create_metrics_folder(input_path,output_path,header,channel,\
                          all_cells = False,\
                          measure = False,\
                          project = True,\
                          re_align = True,max_th=0.5):
    pooled_data = merge_pks(input_path)
    #output_path = input_path + "output/"
    if measure:
        create_metrics_relative_MIDX(pooled_data,output_path,header,channel)
    if project:
        project_by_length(output_path,pooled_data,header,channel,all_cells=all_cells,re_align=re_align,max_th=max_th)
    else:
        project_by_percentile(output_path,pooled_data,header,channel,all_cells=all_cells,re_align=re_align,max_th=max_th)


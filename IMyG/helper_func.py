__author__ = "jz-rolling"

# IMyG helper
# 09/15/19

import numpy as np
import pandas as pd
import cmath
from math import pi
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from scipy import ndimage as ndi
from scipy import fftpack
from skimage import morphology, feature, measure, filters, segmentation, util, exposure
from scipy.interpolate import splprep, splev, RectBivariateSpline
import warnings
import IMyG.config as conf
from itertools import combinations


def plot_spectrum(im_fft):
    # A logarithmic colormap
    plt.figure()
    plt.imshow(np.abs(im_fft), norm=LogNorm(vmin=5), cmap="Greys")
    plt.colorbar()


def fft(img, subtract_mean=True):
    warnings.filterwarnings("ignore")
    if subtract_mean:
        img = img - np.mean(img)
    return (fftpack.fftshift(fftpack.fft2(img)))


def fft_reconstruction(fft_img, filters):
    # reconstruct image after FFT band pass filtering.
    warnings.filterwarnings("ignore")
    if len(filters) > 0:
        for filter in filters:
            try:
                fft_img *= filter
            except:
                raise ValueError("Illegal input filter found, shape doesn't match?")
    return (fftpack.ifft2(fftpack.ifftshift(fft_img)).real)


def bandpass_filter(pixel_microns, img_width=2048, img_height=2048, high_pass_width=0.2, low_pass_width=20):
    # bandpass filter on frequency space
    u_max = round(1 / pixel_microns, 3) / 2
    v_max = round(1 / pixel_microns, 3) / 2
    u_axis_vec = np.linspace(-u_max / 2, u_max / 2, img_width)
    v_axis_vec = np.linspace(-v_max / 2, v_max / 2, img_height)
    u_mat, v_mat = np.meshgrid(u_axis_vec, v_axis_vec)
    centered_mesh = np.sqrt(u_mat ** 2 + v_mat ** 2)
    if high_pass_width == 0:
        high_pass_filter = np.ones((img_width, img_height)).astype(np.int)
    else:
        high_pass_filter = np.e ** (-(centered_mesh * high_pass_width) ** 2)
    if low_pass_width == 0:
        low_pass_filter = np.ones((2048, 2048)).astype(np.int)
    else:
        low_pass_filter = 1 - np.e ** (-(centered_mesh * low_pass_width) ** 2)
    return (high_pass_filter, low_pass_filter)


def is_integer(x):
    try:
        isinstance(x, (int))
        return True
    except:
        return False


def normalize_img(img, dtype=16, adjust_gamma=True, gamma=1):
    from skimage import exposure
    if is_integer(dtype) & (dtype > 2):
        n_range = (0, 2 ** dtype - 1)
    else:
        print("Illegal input found where an integer no less than 2 was expected.")
    outimg = exposure.rescale_intensity(img, out_range=n_range)
    if adjust_gamma:
        outimg = exposure.adjust_gamma(outimg, gamma=gamma)
    return outimg


def Shape_indexing_normalization(img, shape_indexing_sigma=2):
    surface = feature.shape_index(img, sigma=shape_indexing_sigma)
    surface = np.nan_to_num(surface, copy=True)
    surface = (exposure.equalize_hist(surface) * 255).astype(np.uint8)
    return (surface)


def init_Segmentation(phase_contrast_img, shape_indexing_sigma=2):
    #
    shape_indexed = Shape_indexing_normalization(phase_contrast_img, \
                                                 shape_indexing_sigma=shape_indexing_sigma)
    #
    ph_gau = filters.gaussian(phase_contrast_img, sigma=1)
    ph_binary_glob = (ph_gau < filters.threshold_isodata(ph_gau))*1
    ph_binary_local = (ph_gau < filters.threshold_local(ph_gau, method="mean", block_size=15))
    ph_binary_local[morphology.dilation(ph_binary_glob, morphology.disk(5)) == 0] = 0
    ph_binary_local.dtype = bool
    ph_binary_local = morphology.remove_small_objects(ph_binary_local, min_size=80)
    ph_binary_local = morphology.remove_small_holes(ph_binary_local, area_threshold=80)*1
    ph_binary_local = morphology.opening(ph_binary_local, morphology.disk(1))*1
    # deprecated
    #if use_shape_indexed_binary:
    #    shape_indexed_binary = util.invert(shape_indexed > filters.threshold_local(shape_indexed, 27))
    #    shape_indexed_binary[ph_binary_local == 0] = 0
    #    shape_indexed_binary = (filters.median(shape_indexed_binary, morphology.disk(3)))*1
    #    shape_indexed_binary = (morphology.remove_small_holes(shape_indexed_binary, area_threshold=100))*1
    #
    microcolony_labels = measure.label(ph_binary_local, connectivity=1)
    region_info = measure.regionprops(microcolony_labels,coordinates='xy')
    #return ph_binary_local, shape_indexed, shape_indexed_binary, microcolony_labels, region_info
    return ph_binary_local, shape_indexed, microcolony_labels, region_info


def find_direct_contact(labeled_mask, boundary):
    x, y = np.nonzero(boundary)
    shape = labeled_mask.shape
    label_set = np.zeros((1)).astype(int)
    if len(x) == 1:
        label_set = np.unique(labeled_mask[max(0, x[0] - 1):min(x[0] + 2, shape[0] - 1),
                              max(0, y[0] - 1):min(y[0] + 2, shape[1] - 1)]).astype(int)
        if label_set[0] == 0:
            return (label_set[1:], len(x))
    else:
        for i in range(len(x)):
            xi, yi = x[i], y[i]
            if (xi in range(1, shape[0] - 1)) & (yi in range(1, shape[1] - 1)):
                neighbors = labeled_mask[xi - 1:xi + 2, yi - 1:yi + 2][morphology.disk(1) == 1]
                label_set = np.concatenate((label_set, neighbors))
        label_set = np.unique(label_set).astype(int)
        if label_set[0] == 0:
            return (label_set[1:], len(x))


def between_two_particles(labeled_mask, boundary):
    label_set, boundary_length = find_direct_contact(labeled_mask, boundary)
    if len(label_set) == 2:
        return True, label_set, boundary_length
    else:
        return False, label_set, boundary_length


def distance(v1, v2):
    #Euclidean distance of two points
    return np.sqrt(np.sum((np.array(v1) - np.array(v2)) ** 2,axis=1))


def region_label_to_index(input_region_props):
    #
    index = 0
    out_dict = {}
    for item in input_region_props:
        out_dict[item.label] = index
        index += 1
    return out_dict


def list_branch_points():
    one_d_list = [0, 1, 2, 3, 5, 6, 7, 8]
    three_branches, four_branches, five_branches = [], [], []
    for i in range(len(one_d_list)):
        for j in range(i + 1, len(one_d_list)):
            for k in range(j + 1, len(one_d_list)):
                cube = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])
                cube[int(i / 3), i % 3] = 1
                cube[int(j / 3), j % 3] = 1
                cube[int(k / 3), k % 3] = 1
                three_branches.append(cube.copy())
                for m in range(k + 1, len(one_d_list)):
                    cube[int(m / 3), m % 3] = 1
                    four_branches.append(cube.copy())
                    for n in range(m + 1, len(one_d_list)):
                        cube[int(n / 3), n % 3] = 1
                        five_branches.append(cube.copy())
    return (three_branches, four_branches, five_branches)


def list_terminal_points():
    one_d_list = [0, 1, 2, 3, 5, 6, 7, 8]
    cube = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])
    terminal_points = []
    for i in one_d_list:
        temp_cube = cube.copy()
        temp_cube[int(i / 3), i % 3] += 1
        terminal_points.append(temp_cube)
    return (terminal_points)

def subdivide_polygon_closed(input_image,tolerance=conf.polygon_subdivision_tolerance,simplify = False):
    contours_list = measure.find_contours(input_image, 0.2)
    coords = []
    for contour in contours_list:
        p = measure.subdivide_polygon(contour, degree=conf.b_spline_degree, preserve_ends=True)
        if simplify:
            coords.append(measure.approximate_polygon(p, tolerance=tolerance))
        else:
            coords.append(p)
    return contours_list,coords

def angle(p1,p2,p3):
    '''get the angle between the vectors p2p1,p2p3  with p a point
        reference needed here
    '''
    z1=complex(p1[0],p1[1])
    z2=complex(p2[0],p2[1])
    z3=complex(p3[0],p3[1])
    v12=z1-z2
    v32=z3-z2
    #print v12,v32
    angle=(180/pi)*cmath.phase(v32/v12)
    return angle

def bilinear_interpolate_numpy(im, x, y):
    x0 = x.astype(int)
    x1 = x0 + 1
    y0 = y.astype(int)
    y1 = y0 + 1
    Ia = im[x0,y0]
    Ib = im[x0,y1]
    Ic = im[x1,y0]
    Id = im[x1,y1]
    wa = (x1-x) * (y1-y)
    wb = (x1-x) * (y-y0)
    wc = (x-x0) * (y1-y)
    wd = (x-x0) * (y-y0)
    return np.round((Ia*wa) + (Ib*wb) + (Ic*wc) + (Id*wd),2)

def optimize_bbox(img_shape,bbox,edge_width = 10):
    (rows,columns) = img_shape
    (x1,y1,x2,y2) = bbox
    return(max(0,x1-edge_width),max(0,y1-edge_width),\
           min(rows-1,x2+edge_width),min(columns-1,y2+edge_width))

def touching_edge(img_shape,optimized_bbox):
    (rows, columns) = img_shape
    (x1, y1, x2, y2) = optimized_bbox
    if min(x1,y1,rows-x2-1,columns-y2-1) <= 0:
        return True
    else:
        return False

def spline_approximation(init_contour,n = 200, smooth_factor = 5):
    warnings.filterwarnings("ignore")
    tck, u = splprep(init_contour.T, u=None,s=smooth_factor,per=1)
    u_new = np.linspace(u.min(), u.max(), n)
    x_new, y_new = splev(u_new, tck, der=0)
    return np.array([x_new,y_new]).T

def interpolate_2Dmesh(data_array,s = 1):
    warnings.filterwarnings("ignore")
    x_mesh = np.linspace(0,data_array.shape[0]-1,data_array.shape[0]).astype(int)
    y_mesh = np.linspace(0,data_array.shape[1]-1,data_array.shape[1]).astype(int)
    return RectBivariateSpline(x_mesh,y_mesh,data_array,s = s)

def optimize_boundry(image_sobel,init_contour,n_points,\
                     sigma = conf.boundary_optimization_sobel_sigma,\
                     tolerance = conf.boundary_optimization_tolerance,\
                     max_iteration = conf.boundary_optimization_max_iteration,\
                     step = conf.boundary_optimization_evolve_step,\
                     pre_smooth = conf.boundary_optimization_pre_smooth,\
                     post_smooth = conf.boundary_optimization_post_soomth):
    if sigma == 0:
        sobel = image_sobel.copy()
    else:
        sobel = filters.gaussian(image_sobel,sigma = sigma)
    sobel = (sobel-sobel.min()+1)/(sobel.max()+1)
    data = spline_approximation(init_contour,n=n_points,smooth_factor = pre_smooth)
    gradient = np.gradient(sobel)
    g_x,g_y = gradient[0],gradient[1]
    fg_x = interpolate_2Dmesh(g_x)
    fg_y = interpolate_2Dmesh(g_y)
    converged_list = np.zeros(len(data))
    converged = False
    while max_iteration > 0:
        for i in range(len(data)):
            if converged_list[i] == 0:
                x,y = data[i][0],data[i][1]
                dx = fg_x(x,y)*step
                dy = fg_y(x,y)*step
                if min(abs(dx),abs(dy)) >= tolerance:
                    data[i][0] = x+dx
                    data[i][1] = y+dy
                else:
                    converged_list[i] = 1
        if converged_list.sum() == len(data):
            converged = True
            max_iteration = -1
        max_iteration -= 1
    return spline_approximation(data,n=n_points,smooth_factor = post_smooth)

def touching_edge_microcolony(colony_bbox,particle_bbox,n_rows,n_columns,min_edge_width = 5):
    (x1,y1,x2,y2) = colony_bbox
    (x3,y3,x4,y4) = particle_bbox
    a = x1+x3
    b = y1+y3
    c = a + x4
    d = b + y4
    if  min(a,b,n_rows-c,n_columns-d) <= min_edge_width:
        return True
    else:
        return False

def bending_energy_pixelwise(contour,step = 3):
    p1 = np.concatenate((contour[-step:],contour[:-step]))
    p2 = contour.copy()
    p3 = np.concatenate((contour[step:],contour[0:step]))
    p4 = p3-(2*p2-p1)
    bending_energy = np.square(np.sum(np.abs(p4),axis = 1)).sum()
    return bending_energy/len(contour)

"""
removed
def simplify_contour(mask,tolerance = 0.2,simplify = True):
    self.init_contour,\
    self.init_polygon = subdivide_polygon_closed(input_image=mask,\
                                                 tolerance = tolerance,\
                                               simplify=simplify)
"""

def find_concave_points(data,window = 3,cutoff = 1):
    output = []
    length = len(data)
    p1 = np.concatenate((data[-window:],data[:-window])).T
    p2 = data.copy().T
    p3 = np.concatenate((data[window:],data[0:window])).T
    p1p2 = p1[0]*1+p1[1]*1j - (p2[0]*1+p2[1]*1j)
    p3p2 = p3[0]*1+p3[1]*1j - (p2[0]*1+p2[1]*1j)
    return (np.angle(p3p2/p1p2,deg = True))

def larget_curvature(data,window = 5,cutoff = 1):
    output = []
    length = len(data)
    p1 = np.concatenate((data[-window:],data[:-window])).T
    p2 = data.copy().T
    p3 = np.concatenate((data[window:],data[0:window])).T
    p1p2 = p1[0]*1+p1[1]*1j - (p2[0]*1+p2[1]*1j)
    p3p2 = p3[0]*1+p3[1]*1j - (p2[0]*1+p2[1]*1j)
    return (np.angle(p3p2/p1p2,deg = True))

def Skeleton_analysis(skeleton,max_length = 500,min_length = 15):
    endpoints = []
    skeleton_path = np.where(skeleton>0)
    skeleton_length = len(skeleton_path[0])
    branched = False
    abnormal_length = 0
    # need optimization here
    # exclude particles with total length >= 20um or total length <= 1um
    if skeleton_length >= max_length:
        abnormal_length = 2
    elif skeleton_length <= min_length:
        abnormal_length = 1
    else:
        for i in range(skeleton_length):
            x = skeleton_path[0][i]
            y = skeleton_path[1][i]
            cube = skeleton[x-1:x+2,y-1:y+2]
            if cube.sum() == 2:
                endpoints.append([x,y])
            if cube.sum() > 3:
                branched = True
            if len(endpoints) > 2:
                branched = True
    if len(endpoints) != 2:
        branched = True
    return branched,abnormal_length,endpoints

def Find_neighbor(skeleton, branched, abnormal_length, cellpoles):
    Abandon_cell = False
    output = np.zeros((skeleton.sum(), 2)).astype(int)
    if branched or len(cellpoles) != 2:
        if conf.remove_branched_cells:
            Abandon_cell = True
    elif abnormal_length > 0:
        Abandon_cell = True
    else:
        input_map = skeleton.copy()
        if cellpoles[0][1] < cellpoles[1][1]:
            y_index = 0
        else:
            y_index = 1
        x, y = cellpoles[y_index][0], cellpoles[y_index][1]
        output[0][0], output[0][1] = x, y
        end_reached = False
        n = 1
        while not end_reached:
            cube = input_map[x - 1:x + 2, y - 1:y + 2]
            if cube.sum() == 1:
                cube[1, 1] = 0
                input_map[x, y] = 0
                end_reached = True
            elif cube.sum() == 2:
                cube[1, 1] = 0
                input_map[x, y] = 0
                neighbor = np.where(cube > 0)
                x += (neighbor[0][0] - 1)
                y += (neighbor[1][0] - 1)
                output[n][0], output[n][1] = x, y
                n += 1
    return Abandon_cell, output

def Find_opposites(x1,y1,x2,y2):
    output = np.zeros((3,2))
    dx = x2-x1
    dy = y2-y1
    x3 = x2+dx
    y3 = y2+dy
    output[1][0],output[1][1] = x3,y3
    if abs(dx) == 0:
        output[0][0],output[0][1] = x3-1,y3
        output[2][0],output[2][1] = x3+1,y3
    elif abs(dy) == 0:
        output[0][0],output[0][1] = x3,y3-1
        output[2][0],output[2][1] = x3,y3+1
    else:
        output[0][0],output[0][1] = x3+dx,y3
        output[2][0],output[2][1] = x3,y3+dy
    return(x3,y3,output)

def polygon_optimize(polygon,\
                     min_dist=conf.polygon_optimization_min_dist):
    data_1= polygon[:-1]
    data_2= polygon[1:]
    dxy = data_2-data_1
    outlist = []
    n_gap = (np.sqrt(np.sum(np.square(dxy),axis = 1))/min_dist).astype(int)+1
    ddxy = np.divide(dxy,np.array([n_gap,n_gap]).T)
    for i in range(len(n_gap)):
        n = n_gap[i]
        if n==1:
            outlist.append(data_1[i])
        else:
            outlist.append(data_1[i])
            for j in range(1,n):
                outlist.append(data_1[i]+ddxy[i]*j)
    outlist.append(data_2[-1])
    return(np.array(outlist))

def is_single_cell(cell):
    by_length = in_range(cell.robust_length_micron,conf.min_length,conf.max_length)
    by_eccentricity = in_range(cell.eccentricity,conf.eccentricity[0],conf.eccentricity[1])
    by_sinuosity = in_range(cell.sinuosity,conf.sinuosity[0],conf.sinuosity[1])
    by_solidity = in_range(cell.solidity,conf.solidity[0],conf.solidity[1])
    by_curvature = in_range(cell.curvature,conf.curvature[0],conf.curvature[1])
    by_area = in_range(cell.area,conf.area[0],conf.area[1])
    by_sphericity = in_range(cell.sphericity,conf.sphericity[0],conf.sphericity[1])
    output = np.prod(np.array([by_length,\
                               by_eccentricity,\
                               by_sinuosity,\
                               by_solidity,\
                               by_curvature,\
                               by_area,\
                               by_sphericity]))
    return output

def in_range(val,min_val,max_val):
    if (val < min_val) or (val > max_val):
        return(False)
    else:
        return(True)

def larget_curvature(data,max_window = conf.curvature_estimation_window):
    window = min(max_window,int((len(data)-1)/2))
    output = []
    length = len(data)
    p1 = data[:-2*window].T
    p2 = data[window:-window].T
    p3 = data[2*window:].T
    p2p1 = p2[0]*1+p2[1]*1j - (p1[0]*1+p1[1]*1j)
    p2p3 = p2[0]*1+p2[1]*1j - (p3[0]*1+p3[1]*1j)
    return (180-np.absolute(np.angle(p2p3/p2p1,deg = True))).max()

def sinuosity(cell):
    if not cell.branched:
        if cell.robust_length > 2:
            dist = distance(cell.skeleton_path[0], cell.skeleton_path[-1])
            if dist <= 1:
                cell.sinuosity = 0
            else:
                cell.sinuosity = np.sqrt(2) * (cell.robust_length / distance(cell.skeleton_path[0],cell.skeleton_path[-1]))
        else:
            cell.sinuosity = 1
    elif len(cell.endpoints) <= 1:
        cell.sinuosity = 0
    else:
        dist = 0
        for pairs in combinations(cell.endpoints,2):
            newdist = distance(pairs[0],pairs[1])
            if newdist > dist:
                dist = newdist
        cell.sinuosity = np.sqrt(2) * (cell.robust_length/dist)

def distance(v1, v2):
    #Euclidean distance of two points
    return np.sqrt(np.sum((np.array(v1) - np.array(v2)) ** 2))

def line_intersect(a1, a2, b1, b2):
    #@Hamish Grubijan
    T = np.array([[0, -1], [1, 0]])
    da = np.atleast_2d(a2 - a1)
    db = np.atleast_2d(b2 - b1)
    dp = np.atleast_2d(a1 - b1)
    dap = np.dot(da, T)
    denom = np.sum(dap * db, axis=1)
    num = np.sum(dap * dp, axis=1)
    return np.atleast_2d(num / denom).T * db + b1

def vector_line_intersect(v1,v2,l1,l2):
    warnings.filterwarnings("ignore")
    xy = line_intersect(v1,v2,l1,l2)
    dxy_v1 = xy-v1
    dxy_v2 = xy-v2
    dxy = dxy_v1*dxy_v2
    intersection_points = xy[np.where(np.logical_and(dxy[:,0]<0,dxy[:,1]<0))]
    if len(intersection_points)>2:
        sub_l1 = l1[np.where(np.logical_and(dxy[:,0]<0,dxy[:,1]<0))]
        dist = np.sum(np.square(sub_l1-intersection_points),axis = 1)
        intersection_points = intersection_points[np.argsort(dist)[0:2]]
    return(intersection_points)

def perpendicular_line_initiation(skeleton):
    warnings.filterwarnings("ignore")
    d = skeleton.copy()
    dxy = d[1:]-d[:-1]
    dxy_perp = np.vstack([np.ones(len(dxy)),-(dxy.T[0]/dxy.T[1])]).T
    dxy_perp = np.vstack([dxy_perp[0],dxy_perp])
    d_perp = d+dxy_perp
    return(d,d_perp)

def find_midpoints(smoothed_skeleton,smoothed_contour):
    warnings.filterwarnings("ignore")
    dd,d_perp = perpendicular_line_initiation(smoothed_skeleton)
    p2,p1 = smoothed_contour[1:],smoothed_contour[:-1]
    updated_skeleton = np.zeros(dd.shape)
    width = np.zeros(dd.shape[0])
    for i in range(len(dd)):
        l1 = np.tile(dd[i],(len(p2),1))
        l2 = np.tile(d_perp[i],(len(p2),1))
        xy = vector_line_intersect(p1,p2,l1,l2)
        updated_skeleton[i] = np.average(xy,axis=0)
        width[i] = distance(xy[0],xy[1])
    return(updated_skeleton,width)

def appoximate_midline(smoothed_skeleton,smoothed_contour,\
                       tolerance = 0.01,\
                       max_iteration = 10):
    warnings.filterwarnings("ignore")
    midline = smoothed_skeleton.copy()
    n=0
    converged = False
    while n<max_iteration:
        updated_midline,width = find_midpoints(smoothed_skeleton,smoothed_contour)
        dxy = updated_midline-midline
        midline = updated_midline
        if np.max(dxy) < tolerance:
            converged = True
            break
        n += 1
    return midline.astype(np.float),converged,width.astype(np.float)

def find_poles(midline,smoothed_contour):
    warnings.filterwarnings("ignore")
    #find endpoints and their nearest neighbors on a midline
    length = len(midline)
    p2,p1 = smoothed_contour[1:],smoothed_contour[:-1]
    for i in range(5):
        pole1 = midline[i]
        pole1_neighbor = midline[i+1]
        #vectorize contour
        l1 = np.tile(pole1,(len(p2),1))
        l2 = np.tile(pole1_neighbor,(len(p2),1))
        #find the two intersection points between the vectorized contour and line through pole1
        intersection_points_pole1 = vector_line_intersect(p1,p2,l1,l2)
        dxy_1 = l1[0:2]-l2[0:2]
        ddxy_intersection_1 = (intersection_points_pole1-l1[0:2])*dxy_1
        index_1 = np.where(np.logical_and(ddxy_intersection_1[:,0]>0,ddxy_intersection_1[:,1]>0))[0]
        if len(index_1) > 0:
            extended_pole1 = intersection_points_pole1[index_1][0]
            break
    for j in range(5):
        pole2 = midline[-1-j]
        pole2_neighbor = midline[-2-j]
        #vectorize contour
        l3 = np.tile(pole2,(len(p2),1))
        l4 = np.tile(pole2_neighbor,(len(p2),1))
        #find the two intersection points between the vectorized contour and line through pole2
        intersection_points_pole2 = vector_line_intersect(p1,p2,l3,l4)
        dxy_2 = l3[0:2]-l4[0:2]
        ddxy_intersection_2 = (intersection_points_pole2-l3[0:2])*dxy_2
        index_2 = np.where(np.logical_and(ddxy_intersection_2[:,0]>0,ddxy_intersection_2[:,1]>0))[0]
        if len(index_2) > 0:
            extended_pole2 = intersection_points_pole2[index_2][0]
            break
    trimmed_midline = midline[i:length-j]
    return(extended_pole1,extended_pole2,trimmed_midline)

def extend_skeleton(smoothed_skeleton,smoothed_contour,step = 1):
    warnings.filterwarnings("ignore")
    new_pole1,new_pole2,smoothed_skeleton = find_poles(smoothed_skeleton,smoothed_contour)
    pole1,pole2 = smoothed_skeleton[0],smoothed_skeleton[-1]
    dist1,dist2 = distance(pole1,new_pole1),distance(pole2,new_pole2)
    n_segments_1,n_segments_2 = int(round(dist1/step)),int(round(dist2/step))
    if n_segments_1 > 1:
        unit_v1 = (pole1 - new_pole1)/n_segments_1
        pole1_extention = vector_segments(new_pole1,unit_v1,n_segments_1)[:-1]
    else:
        pole1_extention = np.array([new_pole1])
    if n_segments_2 > 1:
        unit_v2 = (new_pole2 - pole2)/n_segments_2
        pole2_extention = np.vstack([vector_segments(pole2,unit_v2,\
                                                     n_segments_2)[1:-1],new_pole2])
    else:
        pole2_extention = np.array([new_pole2])
    return(np.vstack([pole1_extention[1:],smoothed_skeleton,pole2_extention[:-1]]).astype(np.float))


def spline_approximation_line(init_contour,n = 200, smooth_factor = 8):
    warnings.filterwarnings("ignore")
    tck, u = splprep(init_contour.T, u=None,s=smooth_factor)
    u_new = np.linspace(u.min(), u.max(), n)
    x_new, y_new = splev(u_new, tck, der=0)
    return np.array([x_new,y_new]).T

def measure_along_line(line,data):
    warnings.filterwarnings("ignore")
    return(bilinear_interpolate_numpy(data,line.T[0],line.T[1]).astype(float))

def measure_length(data,pixel_microns):
    v1,v2 = data[:-1],data[1:]
    length = np.sqrt(np.sum((np.array(v1) - np.array(v2)) ** 2, axis=1)).sum()*pixel_microns
    return(length)

def vector_segments(start_point,vector,n_segments):
    m = np.linspace(0,n_segments,n_segments+1)
    x = (np.repeat(vector[0],n_segments+1) * m) + start_point[0]
    y = (np.repeat(vector[1],n_segments+1) * m) + start_point[1]
    return(np.array([x,y]).T)

def unit_perpendicular_vector(data):
    d = data.copy()
    dxy = d[1:]-d[:-1]
    dxy_perp = np.vstack([np.ones(len(dxy)),-(dxy.T[0]/dxy.T[1])]).T
    dxy_perp = np.vstack([dxy_perp[0],dxy_perp])
    vector_length = np.sqrt(np.sum(np.square(dxy_perp),axis=1))
    dxy_unit = dxy_perp/np.vstack([vector_length,vector_length]).T
    return(dxy_unit)

def straighten_cell(img,mask,midline,\
                       subpixel = 0.5,\
                       half_width_by_pixel = 8,\
                       remove_background = False):
    unit_dxy = unit_perpendicular_vector(midline)*subpixel
    data = bilinear_interpolate_numpy(img,midline.T[0],midline.T[1])
    copied_img = img.copy()
    if remove_background:
        copied_img[mask == 0] = 0
    for i in range(1,int(half_width_by_pixel/subpixel)):
        dxy = unit_dxy*i
        v1 = midline+dxy
        v2 = midline-dxy
        p1 = bilinear_interpolate_numpy(copied_img,v1.T[0],v1.T[1])
        p2 = bilinear_interpolate_numpy(copied_img,v2.T[0],v2.T[1])
        data = np.vstack([p1,data,p2])
    return (data)

def straighten_cell_normalize_width(img,width,midline,\
                                    subpixel = 1/conf.sample_density,\
                                    remove_cap = 5,\
                                    pad = 3):
    decapped_midline = midline[remove_cap+1:-remove_cap-1]
    decapped_width = width[remove_cap:-remove_cap]
    unit_dxy = unit_perpendicular_vector(decapped_midline)*subpixel
    normalization_factor = decapped_width/decapped_width.mean()
    width_normalized_dxy = unit_dxy*(np.vstack([normalization_factor,normalization_factor]).T)
    data = bilinear_interpolate_numpy(img,decapped_midline.T[0],decapped_midline.T[1])
    copied_img = img.copy()
    for i in range(1,int(decapped_width.mean()*0.5/subpixel)+pad):
        dxy = width_normalized_dxy*i
        v1 = decapped_midline+dxy
        v2 = decapped_midline-dxy
        p1 = bilinear_interpolate_numpy(copied_img,v1.T[0],v1.T[1])
        p2 = bilinear_interpolate_numpy(copied_img,v2.T[0],v2.T[1])
        data = np.vstack([p1,data,p2])
    return (data)


def generate_96_well_plate():
    rows = ["A","B","C","D","E","F","G","G"]
    columns = np.linspace(1,12,12).astype(int)
    plate = pd.DataFrame(0,index=rows,columns=columns)
    return(plate)
from IMyG.helper_func import *



def moving_window_average(data,window_size = 5):
    if len(data) <= window_size+2:
        raise ValueError("Input array does not have enough values")
    else:
        cum_sum = np.cumsum(np.insert(data,0,0))
        return ((cum_sum[window_size:]-cum_sum[:-window_size])/float(window_size))


def measure_variability(data,cut_off = 10,percentile = 90):
    val = np.log10(data[data >= np.percentile(data,percentile)].mean()/data.mean())
    if val > 1:
        val = 1
    return round(val,3)

def normalize_array(data,min_val,max_val,base=0):
    return((data-min_val+base)/(max_val-min_val+base))


def weighted_segmentation(data,quantile_filter = 0.05,weighted_center = True):
    if quantile_filter:
        min_threshould = np.quantile(data,quantile_filter)
    else:
        min_threshould = 0
    filtered_data = data.copy()
    normalized_data = normalize_array(filtered_data,min_threshould,filtered_data.max())
    normalized_data[filtered_data<=min_threshould] = 0.0
    xcor = np.linspace(0,1,len(normalized_data))
    if weighted_center:
        com_0 = center_of_mass(normalized_data,xcor)
    else:
        com_0 = 0.5
    adjusted_center_dist_1 = int(com_0*len(normalized_data))
    adjusted_center_dist_2 = len(normalized_data) - adjusted_center_dist_1
    first_half = normalized_data[:adjusted_center_dist_1]
    xcor_first = xcor[:adjusted_center_dist_1]
    second_half = normalized_data[-adjusted_center_dist_2:]
    xcor_second = xcor[-adjusted_center_dist_2:]
    com_1 = center_of_mass(first_half,xcor_first)
    com_2 = center_of_mass(second_half,xcor_second)
    com_dist = com_2-com_1
    return [com_1,com_0,com_2],round(com_dist,3)


def segmented_mean(data,quantile_filter = 0.05,weighted_center = True):
    smoothed = moving_window_average(data)
    interpolated = np.interp(np.linspace(0,1,200),np.linspace(0,1,len(data)),data)
    smoothed = moving_window_average(interpolated)
    [c1,c2,c3],c_dist = weighted_segmentation(smoothed,\
                                              quantile_filter=quantile_filter,\
                                              weighted_center=weighted_center)
    n_points = len(smoothed)
    c1,c2,c3 = int(c1*n_points),int(c2*n_points),int(c3*n_points)
    l1,l2,r1,r2 = smoothed[:c1].mean(),smoothed[c1:c2].mean(),smoothed[c2:c3].mean(),smoothed[c3:].mean()
    h1,h2 = smoothed[:c2].mean(),smoothed[c2:].mean()
    pole,center = np.concatenate([smoothed[:c1],smoothed[c3:]]).mean(),\
                  smoothed[c1:c3].mean()
    return(l1,l2,r1,r2,h1,h2,pole,center)


def measure_symmetry(p1,p2):
    if p1>p2:
        return round(p2/p1,3)
    else:
        return round(p1/p2,3)


def measure_centrifugality(pole,center):
    return(round(np.log2(pole/center),3))


def measure_membrane_idx(data,quantile_filter = 0.2,weighted_center = True,window = 2):
    interpolated = np.interp(np.linspace(0,1,100),np.linspace(0,1,len(data)),data)
    smoothed = moving_window_average(interpolated)
    [c1,c2,c3],c_dist = weighted_segmentation(smoothed,\
                                              quantile_filter=quantile_filter,\
                                              weighted_center=weighted_center)
    n_points = len(smoothed)
    w = window
    c1,c2,c3 = int(c1*n_points),int(c2*n_points),int(c3*n_points)
    #print(c1,c2,c3)
    v1 = smoothed[max(0,c1-window):min(len(smoothed),c1+window)].mean()
    v2 = smoothed[max(0,c2-window):min(len(smoothed),c2+window)].mean()
    v3 = smoothed[max(0,c3-window):min(len(smoothed),c3+window)].mean()
    ratio = round(np.average([v1,v3])/v2,3)
    return ratio,c_dist






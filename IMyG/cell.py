__author__ = "jz-rolling"

from IMyG.helper_func import *
import IMyG.config as conf

class cell:
    def __init__(self,image,\
                 colony_label,\
                 cell_label,\
                 labeled_binary,bbox):
        self.regionprop = measure.regionprops(label_image=labeled_binary,coordinates='xy')
        self.pixel_microns = image.pixel_microns
        self.bbox = bbox
        (x1, y1, x2, y2) = self.bbox
        self.phase = image.ph_filtered[x1:x2, y1:y2].copy()
        self.sobel = image.sobel[x1:x2, y1:y2].copy()
        self.shape_indexed = image.shape_indexed_smoothed[x1:x2, y1:y2].copy()
        self.mask = labeled_binary
        self.init_contour = None
        self.init_polygon = None
        self.colony_label = colony_label
        self.cell_label = cell_label
        self.optimized_contour = None
        self.is_good_baby = False
        #self.skeleton = None
        #self.skeleton_path = None
        #self.branched = False
        #self.abnormal_length =0
        #self.abandoned = False
        #self.rough_length = 0
        self.max_length = image.max_length
        self.min_length = image.min_length
        self.skeleton_robust()
        self.perimeter = self.regionprop[0].perimeter
        self.solidity = self.regionprop[0].solidity
        self.eccentricity = self.regionprop[0].eccentricity
        self.robust_length =  self.skeleton.sum()
        self.robust_length_micron = self.robust_length * image.pixel_microns
        self.sphericity = 2*self.area / (pi* (self.robust_length ** 2))
        self.curvature = 0
        self.compactness = (4*pi*self.area)/(self.perimeter**2)
        self.midline = np.array([])
        self.measure_along_midline = {}
        self.measure_along_contour = {}
        self.mean_fl_intensity = {}
        self.phase_straighten = []
        self.fl_straighten = {}
        if not self.branched:
            if self.robust_length > 2:
                dist = distance(self.skeleton_path[0], self.skeleton_path[-1])
                if dist <= 1:
                    self.sinuosity = 0
                else:
                    self.sinuosity = np.sqrt(2)*(self.robust_length/ distance(self.skeleton_path[0], self.skeleton_path[-1]))
                    self.curvature = larget_curvature(self.skeleton_path)
        else:
            self.sinuosity = 1

    def simplify_contour(self,tolerance = 0.2,simplify = True):
        self.init_contour,\
        self.init_polygon = subdivide_polygon_closed(input_image=self.mask,\
                                                     tolerance = tolerance, \
                                                     simplify=simplify)

    def split(self):
        #future function
        return None

    def contour_optimization(self,polygon = True):
        warnings.filterwarnings("ignore")
        output = []
        if polygon:
            contours = self.init_polygon
        else:
            contours = self.init_contour
        for i in range(len(contours)):
            length = len(self.init_contour[i])
            contour = contours[i]
            if polygon:
                contour = polygon_optimize(contour)
            if len(contour) >= 5:
                optimized = optimize_boundry(self.sobel,contour[:-1],n_points=int(length*0.8))
                output.append(optimized)
        self.optimized_contour = output


    def skeleton_robust(self):
        warnings.filterwarnings("ignore")
        self.skeleton = morphology.skeletonize(self.mask)
        self.branched, self.abnormal_length, self.endpoints = Skeleton_analysis(self.skeleton,\
                                                                                max_length=self.max_length,\
                                                                                min_length=self.min_length)
        self.abandoned,self.skeleton_path = Find_neighbor(self.skeleton,\
                                                          self.branched,\
                                                          self.abnormal_length,\
                                                          self.endpoints)
        self.rough_length = len(self.skeleton_path)
        self.area = self.regionprop[0].area
        return (True)

    def skeleton_optimization(self):
        warnings.filterwarnings("ignore")
        if self.optimized_contour == None:
            self.contour_optimization(polygon=True)
        if len(self.init_contour) != 1:
            print("Object with hole(s) detected!")
        else:
            smoothed_skel = spline_approximation_line(self.skeleton_path,\
                                                      n=int(len(self.skeleton_path) * conf.skeleton_density_factor),\
                                                      smooth_factor=20)
            extended_skel = extend_skeleton(smoothed_skel, self.optimized_contour[0])
            midline, _converged, _width = appoximate_midline(extended_skel, self.optimized_contour[0])
            pole1,pole2,midline = find_poles(midline,self.optimized_contour[0])
            midline = np.vstack([pole1,midline,pole2])
            self.length = measure_length(midline, self.pixel_microns)
            pixel_length = int(round(self.length/self.pixel_microns))
            self.midline = spline_approximation_line(midline,n=conf.sample_density*pixel_length,smooth_factor=1)
            _midline, self.width = find_midpoints(self.midline[1:-1], self.optimized_contour[0])

    def plot(self,plot_skeleton = True,savefig = False,output_path = None):
        if self.is_good_baby:
            if self.optimized_contour == None:
                self.skeleton_optimization()
            contour = self.optimized_contour[0][:-1]
            dcontour = self.optimized_contour[0][1:] - self.optimized_contour[0][:-1]
            skeleton = self.midline[:-1]
            dskeleton = self.midline[1:] - self.midline[:-1]
            fig = plt.figure(figsize=(4,4))
            plt.imshow(self.phase,cmap = "gist_gray")
            for i in range(len(contour)):
                x,y = contour[i][0],contour[i][1]
                dx,dy = dcontour[i][0],dcontour[i][1]
                plt.arrow(y,x,dy,dx,color = "lightcoral",alpha = 0.5)
            if plot_skeleton:
                for j in range(len(skeleton)):
                    m,n = skeleton[j][0],skeleton[j][1]
                    dm,dn = dskeleton[j][0],dskeleton[j][1]
                    plt.arrow(n,m,dn,dm,color="lightcoral",alpha=0.5)
            if savefig:
                if output_path == None:
                    print("No output directory assigned.")
                else:
                    try:
                        fig.savefig(output_path+"%d_%d.png"%(self.colony_label,self.cell_label))
                        plt.close()
                    except:
                        print("Could not find output director: %s"(output_path))
                        plt.close()
        else:
            print("I'm sorry that our current plot function does not support drawing cells with apparent abnormalties.")

    def measure(self,image,midline = True,contour = True,straighten = True):
        x1,y1,x2,y2 = self.bbox
        if midline:
            self.measure_along_midline["Shape_indexed"] = measure_along_line(self.midline,self.shape_indexed)
            self.measure_along_midline["Phase"] = measure_along_line(self.midline,self.phase)
            if len(image.fl_img) != 0:
                for channel,data in image.fl_img.items():
                    croped_data = data[x1:x2,y1:y2]
                    self.measure_along_midline[channel] = measure_along_line(self.midline,croped_data)
                    self.mean_fl_intensity[channel] = croped_data[self.mask > 0].mean()
                    if contour:
                        self.measure_along_contour[channel] = measure_along_line(self.optimized_contour[0], croped_data)
                    if straighten:
                        width_normalized_data = straighten_cell_normalize_width(croped_data,self.width,self.midline)
                        w_average,w_std = np.average(width_normalized_data,axis = 1),\
                                          np.std(width_normalized_data,axis = 1)
                        h_average,h_std = np.average(width_normalized_data,axis = 0),\
                                          np.std(width_normalized_data,axis = 0)
                        self.fl_straighten[channel] = [w_average,w_std,h_average,h_std]
        self.perimeter_precise = measure_length(self.optimized_contour[0],self.pixel_microns)


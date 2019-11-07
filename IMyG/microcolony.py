__author__ = "jz-rolling"

#from IMyG.helper_func import *
from IMyG.cell import *
import IMyG.config as conf


class microcolony:
    """

    """
    def __init__(self, image, microcolony_regionprop):
        self.bbox = optimize_bbox(image.shape,microcolony_regionprop.bbox)
        (x1, y1, x2, y2) = self.bbox
        self.label = microcolony_regionprop.label
        self.phase = image.ph_filtered[x1:x2, y1:y2].copy()
        self.mask = (image.microcolony_labels[x1:x2, y1:y2] == self.label)*1
        self.sobel = image.sobel[x1:x2, y1:y2].copy()
        self.particle_counts = 0
        self.pixel_microns = image.pixel_microns

    def create_seeds(self, image,\
                     low_bound=conf.shape_index_threshould_low,\
                     high_bound=conf.shape_index_threshould_high,\
                     apply_median_filter = True):
        x1, y1, x2, y2 = self.bbox
        if apply_median_filter:
            # use smoothed shape-indexed image
            self.shape_indexed = image.shape_indexed_smoothed[x1:x2, y1:y2].copy()* (self.mask > 0)
        else:
            self.shape_indexed = image.shape_indexed[x1:x2, y1:y2].copy()* (self.mask > 0)
        #generate watershedding seeds, remove the ones that are too small
        init_seeds = morphology.opening((self.shape_indexed < high_bound) & (self.shape_indexed > low_bound),\
                                        morphology.disk(2)).astype(bool)
        init_seeds = morphology.remove_small_objects(init_seeds,\
                                                     min_size=conf.min_seed_size)
        self.labeled_seeds = measure.label(init_seeds, connectivity=1)
        labels = len(np.unique(self.labeled_seeds))
        if labels > 1:
            #remove subimages whose seeds that are too small
            if labels == 2:
                #single particl subimage
                if not touching_edge(image.shape,self.bbox):
                    #remove particles that are too close to the edge
                    if conf.smooth_individual_mask:
                        self.mask = filters.median(self.mask)
                    contour, polygon = subdivide_polygon_closed(self.mask, \
                                                                simplify=True, \
                                                                tolerance=conf.polygon_subdivision_tolerance)
                    normalized_bending_energy = bending_energy_pixelwise(contour[0],\
                                                                         step=conf.bending_energy_stepsize)
                    if normalized_bending_energy <= conf.bending_energy_cutoff:
                        init_cell = cell(image, self.label, 0, self.mask, self.bbox)
                        init_cell.init_contour = contour
                        init_cell.init_polygon = polygon
                        #init_cell.contour_optimization(polygon=False)
                        #init_cell.skeleton_robust()
                        if is_single_cell(init_cell) and (not init_cell.abandoned):
                            init_cell.is_good_baby = True
                            image.cells.append(init_cell)
                        else:
                            image.celllike.append(init_cell)
            else:
                self.watersheded = segmentation.watershed(self.shape_indexed, self.labeled_seeds, mask=self.mask, \
                                                          connectivity=1, \
                                                          compactness=0, \
                                                          watershed_line=True)
                self.boundries = ((self.mask > 0) * 1) - ((self.watersheded > 0) * 1)
                self.labeled_boundary = measure.label(self.boundries, connectivity=2)
        self.particle_counts = labels

    def remove_false_boundries_fast(self,cutoff = conf.boundary_curvature_cutoff):
        if self.particle_counts > 2:
            copy_watersheded = self.watersheded.copy()
            for i in range(1,self.labeled_boundary.max()):
                if np.average(self.shape_indexed[self.labeled_boundary == i]) <= cutoff:
                    copy_watersheded[self.labeled_boundary==i] = i+1
            self.watersheded = measure.label(copy_watersheded > 0, connectivity=1)

    def remove_false_boundries_advanced(self, cutoff=conf.boundary_curvature_cutoff_robust):
        if self.particle_counts > 2:
            regionprops = measure.regionprops(self.watersheded,coordinates='xy')
            copy_watersheded = self.watersheded.copy()
            for i in range(1, self.labeled_boundary.max()):
                boundary = self.labeled_boundary == i
                if np.average(self.shape_indexed[boundary]) < cutoff:
                    sandwiched, neighbors, boundary_length = between_two_particles(self.watersheded, boundary)
                    if (sandwiched <= 3) and boundary_length in range(3, 8):
                        for pair in combinations(neighbors, 2):
                            img1 = (copy_watersheded == pair[0]) * 1
                            img2 = (copy_watersheded == pair[1]) * 1
                            img3 = img1 + img2 + boundary * 1
                            prop1 = regionprops[pair[0] - 1]
                            prop2 = regionprops[pair[1] - 1]
                            prop3 = measure.regionprops(img3,coordinates='xy')[0]
                            if prop3.eccentricity > max(prop1.eccentricity, prop2.eccentricity):
                                if prop3.major_axis_length < (0.8*conf.max_length/self.pixel_microns):
                                    copy_watersheded[boundary] = i + 1
            self.watersheded = measure.label(copy_watersheded > 0, connectivity=1)

    def extract_celllike_particles(self,image):
        if self.particle_counts > 2:
            self.particle_info = measure.regionprops(self.watersheded)
            n = 1
            for info in self.particle_info:
                (x1,y1,x2,y2) = optimize_bbox(self.mask.shape,info.bbox)
                label = info.label
                roi = (self.watersheded==label)[x1:x2,y1:y2]*1
                global_bbox = (x1+self.bbox[0],y1+self.bbox[1],x2+self.bbox[0],y2+self.bbox[1])
                if not touching_edge(image.shape,global_bbox):
                    if conf.smooth_individual_mask:
                        roi = filters.median(roi)
                    contour, polygon = subdivide_polygon_closed(roi,\
                                                                simplify=True,\
                                                                tolerance=conf.polygon_subdivision_tolerance)
                    normalized_bending_energy = bending_energy_pixelwise(contour[0],\
                                                                         step = conf.bending_energy_stepsize)
                    if normalized_bending_energy < conf.bending_energy_cutoff:
                        init_cell = cell(image,self.label,n,roi,global_bbox)
                        init_cell.init_contour = contour
                        init_cell.init_polygon = polygon
                        #init_cell.contour_optimization(polygon=False)
                        #init_cell.skeleton_robust()
                        if is_single_cell(init_cell) and (not init_cell.abandoned):
                            init_cell.is_good_baby = True
                            image.cells.append(init_cell)
                        else:
                            image.celllike.append(init_cell)
                        n+=1


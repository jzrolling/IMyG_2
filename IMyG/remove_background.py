from numba import jit
import numpy as np
from skimage import filters,transform


class rolling_ball:
    def __init__(self,radius = 50):
        self.width = 0
        if radius <= 10:
            self.shrink_factor = 1
            self.arc_trim_per = 24
        elif radius <= 20:
            self.shrink_factor = 2
            self.arc_trim_per = 24
        elif radius <= 100:
            self.shrink_factor = 4
            self.arc_trim_per = 32
        else:
            self.shrink_factor = 8
            self.arc_trim_per = 40
        self.radius = radius/self.shrink_factor
        self.build()
        
    def build(self):
        x_trim = int(self.arc_trim_per*self.radius/100)
        half_width = int(self.radius-x_trim)
        self.width = int(2*half_width+1)
        r_squre = np.ones((self.width,self.width))*(self.radius**2)
        squared = np.square(np.linspace(0,self.width-1,self.width)-half_width)
        x_val = np.tile(squared,(self.width,1))
        y_val = x_val.T
        self.ball = np.sqrt(r_squre-x_val-y_val)


def Rolling_ball_bg_subtraction(data,radius = 40):
    output_data = data.copy()
    smoothed = filters.gaussian(data,sigma=1)*65535
    ball = rolling_ball(radius = radius)
    shrinked_img = shrink_img_local_min(smoothed,shrink_factor = ball.shrink_factor)
    #shrinked_img = transform.rescale(data,scale = 1/ball.shrink_factor,\
                                     #anti_aliasing=True, \
                                     #mode="reflect", \
                                     #multichannel=False)
    bg = rolling_ball_bg(shrinked_img,ball.ball)
    bg_rescaled = transform.rescale(bg,scale=ball.shrink_factor,\
                                    anti_aliasing=True,\
                                    mode="reflect",\
                                    multichannel=False)
    output_data = output_data-bg_rescaled
    output_data[output_data<=0] = 0
    return(output_data.astype(np.uint16))
    

@jit(nopython=True)
def shrink_img_local_min(image, shrink_factor=4):
    s = shrink_factor
    r, c = image.shape[0], image.shape[0]
    r_s, c_s = int(r / s), int(c / s)
    shrk_img = np.ones((r_s, c_s))
    for x in range(r_s):
        for y in range(c_s):
            shrk_img[x, y] = image[x * s:x * s + s, y * s:y * s + s].min()
    return shrk_img


@jit(nopython=True)
def rolling_ball_bg(image, ball):
    width = ball.shape[0]
    radius = int(width / 2)
    peak = ball.max()
    r, c = image.shape[0], image.shape[1]
    bg = np.ones((r, c)).astype(np.float32)
    # ignore edges to begin with
    for x in range(r):
        for y in range(c):
            x1, x2, y1, y2 = max(x - radius, 0), min(x + radius + 1, r), max(y - radius, 0), min(y + radius + 1, c)
            cube = image[x1:x2, y1:y2]
            cropped_ball = ball[radius - (x - x1):radius + (x2 - x), radius - (y - y1):radius + (y2 - y)]
            bg_cropped = bg[x1:x2, y1:y2]
            bg_mask = ((cube - cropped_ball).min()) + cropped_ball
            bg[x1:x2, y1:y2] = bg_cropped * (bg_cropped >= bg_mask) + bg_mask * (bg_cropped < bg_mask)
    return (bg)


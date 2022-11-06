from pathlib import Path

import cv2
import numpy as np
from skimage.filters import threshold_otsu
from skin_lesion_cad.utils.hair_removal import HairRemoval


class Segment:

    def __init__(self) -> None:

        # TODO: Move all segmentation hyperparameters to config file
        # and load them in the constructor
        self.hair_rem = HairRemoval()

    @staticmethod
    def find_point(x1, y1, x2, y2, x, y):
        """
        find_point returns True if the point (x,y) is with in the rectangle (x1,y1) and (x2,y2)

        """
        if (x > x1 and x < x2 and
                y > y1 and y < y2):
            return True
        else:
            return False
    @staticmethod
    def check_corners(x, y, img, percentage=0.15):
        """
        check_corners check if point (x,y) lies in the border of the image img

        _extended_summary_

        Parameters
        ----------
        x : int
        y : int
        img : np.ndarray
        percentage : float, optional
            the proportion of the image to consider as a corner, by default 0.15

        Returns
        -------
        bool
            True if the point is in the corner, False otherwise
        """
        height = img.shape[0]
        width = img.shape[1]

        corner_tl = (0, 0), (int(height*percentage), int(width*percentage))
        corner_tr = (int(height*(1-percentage)),
                     0), (int(width), int(height*percentage))
        corner_bl = (0, int((1-percentage)*height)
                     ), (int(width*percentage), int(height))
        corner_br = (int((1-percentage)*width),
                     int((1-percentage)*height)), (int(width), int(height))
        return Segment.find_point(corner_tl[0][0], corner_tl[0][1], corner_bl[1][0], corner_bl[1][1], x=x, y=y) |\
               Segment.find_point(corner_tl[0][0], corner_tl[0][1], corner_tr[1][0], corner_tr[1][1], x=x, y=y) |\
               Segment.find_point(corner_tr[0][0], corner_tr[0][1], corner_br[1][0], corner_br[1][1], x=x, y=y) |\
               Segment.find_point(corner_bl[0][0], corner_bl[0][1],
                           corner_br[1][0], corner_br[1][1], x=x, y=y)

    @staticmethod
    def asf(img, kernel_size):
        """
        asf Alternate Sequential Filtering

        Parameters
        ----------
        img : np.ndarray
            Image to perform the operation on
        kernel_size : int
            size of the kernel to be used
        Returns
        -------
        np.ndarray
            Image after ASF.
        """
        closing = img
        for i in range(1, kernel_size):
            opening = cv2.morphologyEx(
                closing, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (i, i)))
            closing = cv2.morphologyEx(
                opening, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (i, i)))
        return closing

    @staticmethod
    def fill_holes(mask):
        """
        Fill_holes fill holes in the mask

        Parameters
        ----------
        mask : np.ndarray(np.uint8)
            binary mask to fill holes in

        Returns
        -------
        np.ndarray
            mask with holes filled
        """
        contour, _ = cv2.findContours(mask,
                                      cv2.RETR_CCOMP,
                                      cv2.CHAIN_APPROX_SIMPLE)
        # need to copy the image because drawContours modifies the
        # original image by reference (C++ style)
        mask_filled = mask.copy()
        for cnt in contour:
            cv2.drawContours(mask_filled, [cnt], -1, 255, -1)
        return mask_filled

    @staticmethod
    def fov_mask(image):
        """
        Create a mask of the field of view

        Parameters
        ----------
        image : np.ndarray
            image to create the mask from (RGB)

        Returns
        -------
        np.ndarray
            mask of the field of view
        """
        image_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        image_hsv = cv2.medianBlur(image_hsv, 7)

        # black range for HSV to get a mask of black-ish colors
        black_mask = cv2.inRange(image_hsv, (0, 0, 0, 0), (180, 255, 80, 0))
        retval, labels, stats, centroids = cv2.connectedComponentsWithStatsWithAlgorithm(black_mask, 4, cv2.CV_32S, cv2.CCL_WU)

        for lab in np.unique(labels):

            # ignore background label
            if lab == 0:
                continue

            x, y, w, h, area = stats[lab]
            # check if cc bbox is in the corner of the image
            # and keep  cc if it is
            if (x == 0 or x + w == image.shape[1]) and (y == 0 or y + h == image.shape[0]) and area > 1000:
                black_mask[labels == lab] = 255
            else:
                black_mask[labels == lab] = 0

        # dilate the mask to include neighboring black pixels
        black_mask = cv2.dilate(black_mask, cv2.getStructuringElement(cv2.MORPH_CROSS,
                                                                      (10, 10)),
                                iterations=8)
        return black_mask

    @staticmethod
    def fov_artifact_removal(img_segm_sk, fov_thr= 0.5):

        src = np.zeros(img_segm_sk.shape, dtype=np.uint8)

        circles = cv2.HoughCircles(img_segm_sk, cv2.HOUGH_GRADIENT, 1, 50,
                                   param1=100, param2=30,
                                   minRadius=200, maxRadius=500)
        # print(circles)
        if circles is None:
            return src
        else:
            circles = circles[0]

        # sort circles in ascending order by their radius
        # and select biggest one
        circles = circles[circles[:, 2].argsort()][-1:,:]


        # remove FOV circle only if it seems like it based on its size
        if 2*circles[0, -1]/max(img_segm_sk.shape) > fov_thr:
            if circles is not None:
                circles = np.uint16(np.around(circles))
                for i in circles:
                    center = (i[0], i[1])
                    # circle center
                    cv2.circle(src, center, 1, (0, 100, 100), 3)
                    # circle outline
                    radius = i[2]
                    cv2.circle(src, center, radius, (255, 0, 255), 3)
            src = cv2.dilate(src, np.ones((5,5), np.uint8), iterations=5)
        return src

    @staticmethod
    def remove_hair_like_structures(image, max_hair_diam=17):

        # remove structures smaller than max_hair_diam
        se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                       (max_hair_diam, max_hair_diam))
        # also smoothes the contours of segmentation
        return cv2.morphologyEx(image, cv2.MORPH_OPEN, se)

    def segment(self, img, img_name, save=False, resize=1, overrite=False):
        """
        Segment segment the lesion image

        Parameters
        ----------
        img : np.ndarray
            Image to be segmented
        img_name : Path
            Filename of the Image
        save : bool, optional
            Flag, whether or not to save the masks, by default False
        resize: float optional
            If not None will resize the image by given scale factor, None by default.
        overrite: bool optional
            If True will overwrite existing images, False by default.
        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            output, img - segmentation mask and inpainted image
        """
        # load segmentation if images already exist
        save_dir = Path(str(img_name.parent).replace("raw", "processed"))
        save_dir.mkdir(parents=True, exist_ok=True)
        scale_str = 1 if resize is None else str(resize).replace('.', '_')
        save_path = save_dir/Path(img_name.stem+f"_mask_{scale_str}.png")
        save_path_inp = save_dir/Path(img_name.stem+f"_inpaint_{scale_str}.png")
        if not overrite and save_path.exists() and save_path_inp.exists():
            return cv2.imread(str(save_path), cv2.IMREAD_GRAYSCALE),\
                   cv2.imread(str(save_path_inp), cv2.IMREAD_GRAYSCALE)

        # extract FOV mask to ignore it for segmentation
        fov = Segment.fov_mask(img)

        # inpaint hair
        hr_channel_gray = cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8)).apply(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY))
        hair_mask = self.hair_rem.get_hair_mask(hr_channel_gray)
        img = cv2.inpaint(img, hair_mask, 3, cv2.INPAINT_TELEA)

        gray_img = img[:, :, 2]
        clahe = cv2.createCLAHE(clipLimit=0.8, tileGridSize=(8, 8))
        gray_img_enh = clahe.apply(gray_img)

        gray_img_enh_asf = self.asf(
            gray_img_enh, kernel_size=5)
        blur = cv2.GaussianBlur(gray_img_enh_asf, (7, 7), 0)

        # binarize image
        blur_bins = np.histogram(blur[fov == 0], bins=256, range=(0, 255))
        th3_thr = threshold_otsu(None, hist=blur_bins)
        th3 = 255*(blur < th3_thr).astype(np.uint8)

        kernel = np.ones((5, 5), np.uint8)

        closing = cv2.morphologyEx(th3, cv2.MORPH_CLOSE, kernel)

        analysis = cv2.connectedComponentsWithStats(closing, 4,
                                                    cv2.CV_32S)
        (totalLabels, label_ids, values, centroids) = analysis

        output = np.zeros(gray_img.shape, dtype="uint8")

        for i in range(1, totalLabels):
            area = values[i, cv2.CC_STAT_AREA]

            if (area > 1000) and not self.check_corners(centroids[i][0],
                                                        centroids[i][1], img):
                componentMask = (label_ids == i).astype("uint8") * 255
                # Creating the Final output mask
                output = cv2.bitwise_or(output, componentMask)

        # fill fov with zeros
        output[fov != 0] = 0
        fov_artifact_mask = Segment.fov_artifact_removal(output)
        output[fov_artifact_mask !=0 ] = 0

        # fill holes
        output = Segment.fill_holes(output)

        # remove hair-like structures and smooth and obtain final segmentation mask
        output = Segment.remove_hair_like_structures(output)

        if save:
            save_dir = Path(str(img_name.parent).replace(
                "raw", "processed"))
            save_dir.mkdir(parents=True, exist_ok=True)
            scale_str = 1 if resize is None else str(resize).replace('.', '_')
            save_path = save_dir/Path(img_name.stem+f"_mask_{scale_str}.png")
            save_path_inp = save_dir/Path(img_name.stem+f"_inpaint_{scale_str}.png")
            save_path_fov = save_dir/Path(img_name.stem+f"_fov_{scale_str}.png")

            cv2.imwrite(str(save_path), cv2.resize(output, (0, 0),
                                                   fx=resize, fy=resize))
            cv2.imwrite(str(save_path_inp), cv2.resize(img, (0, 0),
                                                       fx=resize, fy=resize))
            cv2.imwrite(str(save_path_fov), cv2.resize(fov, (0, 0),
                                                       fx=resize, fy=resize))
        return output, img

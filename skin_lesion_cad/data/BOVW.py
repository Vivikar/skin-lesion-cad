from skimage.feature import local_binary_pattern
from sklearn.preprocessing import StandardScaler
from skin_lesion_cad.features.colour import ColorFeaturesDescriptor
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfTransformer


from sklearn.utils import check_random_state
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import random
import cv2
import cpuinfo
from scipy.stats import truncnorm
from sklearn.impute import KNNImputer

if "Intel" in cpuinfo.get_cpu_info()['brand_raw']:
    from sklearnex import patch_sklearn
    patch_sklearn()


class DescriptorsTransformer(BaseEstimator, TransformerMixin):
    
    def __init__(self, imputation='knn'):
        self.imp = KNNImputer(missing_values=np.nan, n_neighbors=10, weights='distance', add_indicator=False)
        self.scaler = StandardScaler()
        self.imp_type = imputation
    
    def fit(self, X, y=None):
        if self.imp_type != 'knn':
            X[np.isnan(X)] = 0
            X[np.isinf(X)] = 0
        else:
            X[np.isnan(X)] = np.nan
            X[np.isinf(X)] = np.nan
            X = self.imp.fit_transform(X)
        return self.scaler.fit(X)
    
    def transform(self, X, y=None):
        if self.imp_type != 'knn':
            X[np.isnan(X)] = 0
            X[np.isinf(X)] = 0
        else:
            X[np.isnan(X)] = np.nan
            X[np.isinf(X)] = np.nan
            X = self.imp.transform(X)
        return self.scaler.transform(X)
    
    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X)
        return self.transform(X, y)

class DenseDescriptor:
    """Generate Descriptors with dense sampling"""

    def __init__(self, descriptor, min_keypoints=100, max_keypoints=500, kp_size=10, sample_method="random"):
        """
        __init__ Constructor for DenseDescriptor class


        Parameters
        ----------
        descriptor : object
            Descriptor object with detect and compute methods
        min_keypoints : int, optional
            Min keypoints to be sampled from an image, by default 100
        max_keypoints : int, optional
            Max keypoints to be sampled from an image, by default 500
        kp_size : int, optional
            Radius of the keypoint for dense sampling, by default 10
        sample_method: str, optional
            Method to sample the keypoints, by default "random".
            If "random" then keypoints are sampled first using BRISK and
            then the rest is sampled randomly from the image.
            If 'gaussian' then keypoints are sampled using gaussian distribution
            centres in the middle of the image. No mask needed in this case.
        """

        self.descriptor = descriptor
        self.min_keypoints = min_keypoints
        self.max_keypoints = max_keypoints
        self.kp_size = kp_size
        self.sample_method = sample_method
        
    def _sample_keypoints(self, img,  num):
        """
        _sample_keypoints sample the "num" number of keypoints 
        randomly from the entire image


        Parameters
        ----------
        img : np.ndarray
        num : int
            number of keypoints to return

        Returns
        -------
        list
            list of keypoint objects
        """
        if self.sample_method == "random":

            x, y  = img.shape[0], img.shape[1]
            additional_kp = [cv2.KeyPoint(
                random.randint(0, y-1), random.randint(0, x-1), size=self.kp_size) for i in range(num)]
        elif self.sample_method == "gaussian":
            scale = img.shape[0]//8
            range_x = img.shape[0]//2
            range_y = img.shape[1]//2
            
            xs = truncnorm(a=-range_x/scale, b=+range_x/scale, scale=scale).rvs(size=num).round().astype(np.float32)
            ys = truncnorm(a=-range_y/scale, b=+range_y/scale, scale=scale).rvs(size=num).round().astype(np.float32)

            additional_kp = [cv2.KeyPoint(range_y + ys[i], range_x + xs[i], size=self.kp_size) for i in range(len(xs))]
        else:
            raise ValueError("Unknown sampling method. Should be either 'random' or 'gaussian'")
        return additional_kp

    def _sample_keypoints_with_mask(self, mask,  num):
        """
        _sample_keypoints_with_mask sample the "num" number of keypoints 
        with in the provided mask 

        Parameters
        ----------
        mask : np.ndarray
            Region to sample the keypoints from.
        num : int
            Required number of keypoints

        Returns
        -------
        list
            list of keypoint objects
        """
        all_mask_points = np.argwhere(mask)
        num_available = len(all_mask_points)

        if num <= num_available:
            if self.sample_method == "random":
                additional_kp = [cv2.KeyPoint(int(i[1]), int(
                    i[0]), size=self.kp_size) for i in random.sample(list(all_mask_points), num)]
            elif self.sample_method == "gaussian":
                scale = mask.shape[0]//8
                range_x = mask.shape[0]//2
                range_y = mask.shape[1]//2
                
                xs = truncnorm(a=-range_x/scale, b=+range_x/scale, scale=scale).rvs(size=num).round().astype(np.float32)
                ys = truncnorm(a=-range_y/scale, b=+range_y/scale, scale=scale).rvs(size=num).round().astype(np.float32)

                additional_kp = [cv2.KeyPoint(range_y + ys[i], range_x + xs[i], size=self.kp_size) for i in range(len(xs)) if mask[int(range_x + xs[i]), int(range_y + ys[i])] !=0]
        else:
            additional_kp = self._sample_keypoints(mask, num)

        return additional_kp

    def detect(self, img, mask=None):
        """
        detect Detect the keypoints from input image

        Parameters
        ----------
        img : np.ndarray
            Image to detect keypoints on
        mask : np.ndarray, optional
            Region to sample the keypoints from, by default None

        Returns
        -------
        tuple
            tuple of keypoint objects
        """
        # if using 'gaussian' sampling method, no BRISK detection needed
        if self.sample_method != "gaussian":
            desc_keypoints = self.descriptor.detect(img, None)
        else:
            desc_keypoints = tuple()

        if len(desc_keypoints) < self.min_keypoints:
            if mask is not None:
                additional_kp = self._sample_keypoints_with_mask(mask,
                                                                 num=self.min_keypoints-len(desc_keypoints))
            else:
                additional_kp = self._sample_keypoints(img,
                                                       num=self.min_keypoints-len(desc_keypoints))
            desc_keypoints = desc_keypoints + tuple(additional_kp)

        if len(desc_keypoints) > self.max_keypoints:
            desc_keypoints = random.sample(desc_keypoints, self.max_keypoints)

        return desc_keypoints

    def compute(self, img, keypoints):
        """
        compute Compute the descriptors from given keypoints.

        Parameters
        ----------
        img : np.ndarray
        keypoints : tuple
            tuple with keypoints

        Returns
        -------
        _type_
            _description_
        """
        des = self.descriptor.compute(img, keypoints=keypoints)
        return des

    def detectAndCompute(self, img, mask=None):
        kp = self.detect(img, mask=mask)
        des = self.compute(img, kp)
        return kp, des[1]


class BagofWords(TransformerMixin, BaseEstimator):
    """
    A generic BagofWords for any input descriptors
    """

    def __init__(self, n_words, batch_size=1024, n_jobs=None, random_state=None, transformer=DescriptorsTransformer(imputation='knn')):
        """
        __init__ Constructor for BagofWords class

        _extended_summary_

        Parameters
        ----------
        n_words : int
            Number of words for the visual features to be used. 
            Also, the number of classes to cluster into.
        batch_size : int, optional
            Batch size for the MiniBatch Kmeans, by default 1024
        n_jobs : int, optional
            The maximum number of concurrently running jobs, by default None
        random_state : int, RandomState instance or None, optional
            Controls the pseudo random number generation for, by default None
        """
        self.n_words = n_words
        self.batch_size = batch_size
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.transformer = transformer

    def _descriptors_to_histogram(self, descriptors, get_tfidf=True):


        descriptors = np.array(descriptors).astype(np.float32)

        descriptors = self.transformer.transform(descriptors)

        use_density = False if get_tfidf else True
        return np.histogram(
            self.dictionary.predict(descriptors),
            bins=range(self.dictionary.n_clusters+1),
            density=get_tfidf            
        )[0]

    def fit_transform(self, X, y=None, get_tfidf=True):
        """
        fit_transform 

        Parameters
        ----------
        X : list
            list of generated descriptors for each image

        Returns
        -------
        np.ndarray
            tfidf values of Bag of Words to be used as image features 
        """
        random_state = check_random_state(self.random_state)

        descriptors = np.vstack(X).astype(np.float32)

        # learn and save scaling for color features
        descriptors = self.transformer.fit_transform(descriptors)

        self.dictionary = KMeans(n_clusters=self.n_words, random_state=random_state,
                                 max_iter=100).fit(descriptors)

        X_trans = [self._descriptors_to_histogram(X[i], get_tfidf) for i in range(len(X))]

        frequency_vectors = np.stack(X_trans)
        
        if get_tfidf:
            # df is the number of images that a visual word appears in
            self.tfidftransformer = TfidfTransformer(smooth_idf=False)
            tfidf = self.tfidftransformer.fit_transform(
                frequency_vectors)

            return tfidf
        else:
            return frequency_vectors

    def transform(self, X, y=None, get_tfidf=True):

        X_trans = [self._descriptors_to_histogram(X[i], get_tfidf) for i in range(len(X))]
        frequency_vectors = np.stack(X_trans)
        
        if get_tfidf:
            tfidf = self.tfidftransformer.transform(frequency_vectors)
            return tfidf
        else:
            return frequency_vectors


class ColorDescriptor(DenseDescriptor):
    def __init__(self, descriptor,  color_spaces: dict, meanshift=None, min_keypoints=100, max_keypoints=500, kp_size=25, sample_method='random'):
        super().__init__(descriptor, min_keypoints, max_keypoints, kp_size, sample_method)
        self.fe = ColorFeaturesDescriptor(color_spaces, meanshift, kp_size)

    def compute(self, img, keypoints):
        """
        compute Compute the descriptors from given keypoints.

        Parameters
        ----------
        img : np.ndarray
        keypoints : tuple
            tuple with keypoints

        Returns
        -------
        _type_
            _description_
        """

        return self.fe.extract_masked(img, keypoints)

    def detectAndCompute(self, img, mask=None):
        kp = self.detect(img, mask=mask)
        des = self.compute(img, kp)
        return kp, des


class LBPDescriptor(DenseDescriptor):
    def __init__(self, descriptor, n_points_radius=[(24, 8), (8, 3), (12, 3), (8, 2), (8, 1)], kp_size=25, min_keypoints=100, max_keypoints=500, method="uniform") -> None:
        super().__init__(descriptor, min_keypoints, max_keypoints, kp_size)
        self.n_points_radius = n_points_radius
        self.method = method

    def lbp_hist(self, img, kp, eps=1e-7):
        # slice a patch around the keypoint
        x1 = max(0, int(kp.pt[1] - self.kp_size))
        x2 = min(img.shape[0], int(kp.pt[1] + self.kp_size))
        y1 = max(0, int(kp.pt[0] - self.kp_size))
        y2 = min(img.shape[1], int(kp.pt[0] + self.kp_size))
        patch = img[x1:x2, y1:y2]

        hist_concat = np.array([])
        for (n_points, radius) in self.n_points_radius:
            lbp = local_binary_pattern(
                patch, n_points, radius, self.method)
            (hist, _) = np.histogram(lbp.ravel(),
                                     bins=np.arange(0, n_points + 3),
                                     range=(0, n_points + 2))
            # normalize the histogram
            hist = hist.astype("float")
            hist /= (hist.sum() + eps)
            hist_concat = np.append(hist_concat, hist)

        return hist_concat

    def compute(self, img, keypoints):

        kp_features = []

        for kp_idx in range(len(keypoints)):
            kp_features.append(self.lbp_hist(
                img, keypoints[kp_idx]))
        return np.array(kp_features).astype(np.float32)

    def detectAndCompute(self, img, mask=None):
        kp = self.detect(img, mask=mask)
        des = self.compute(img, kp)
        return kp, des


from dataclasses import dataclass
from typing import List, Sequence, Tuple

from vital.data.camus.config import CamusTags
import numpy as np

from contour_uncertainty.utils.contour import reconstruction
from vital.data.config import LabelEnum


@dataclass(frozen=True)
class ContourTags(CamusTags):
    contour: str = "contour"  # Contour GT (K,2)
    # contour_mu: str = "contour_pred"  # Contour prediction (K,2)
    # contour_cov: str = "contour_sigma"  # Contour prediction covariance matrices (K,2,2)
    # contour_alpha: str = "contour_alpha"  # Contour prediction alpha vector (skew) (K,2)
    #
    # # For a view of N (2 for ES and ED) samples
    # view_metrics: str = 'view_metrics'  # Dict of arrays of shape (1,)
    # instant_metrics: str = 'instant_metrics'  # Dict of arrays of shape (N,)
    #
    # view_uncertainty: str = 'view_u'  # Dict of arrays of shape (1,)
    # instant_uncertainty: str = 'instant_u'  # Dict of arrays of shape (N,)
    # point_uncertainty: str = 'point_u'  # Dict of arrays of shape (N, K)
    #
    # # Samples
    # contour_samples: str = 'contour_samples'
    # mask_samples: str = 'mask_samples'
    # uncertainty_map: str = 'uncertainty_map'
    # sample_entropy_map: str = 'sample_entropy_map'
    #
    # # Validity
    # contour_validity: str = 'contour_val'  # Validity of prediction contour (contour_mu)
    # sample_validity: str = 'sample_val'  # Validity of sampled contours.


@dataclass
class BatchResult:
    id: str
    img: np.ndarray  # [N (C), H, W]
    gt: np.ndarray  # [N, H, W]
    pred: np.ndarray  # [N, H, W]
    labels: Sequence[LabelEnum]
    # gt_full_res: np.ndarray

    uncertainty_map: np.ndarray  # [N, H, W]

    instants: dict = None
    voxelspacing: Tuple = None  # TODO

    contour: np.ndarray = None  # GT contour
    mu: np.ndarray = None  # [N, K, 2]
    mode: np.ndarray = None  # [N, K, 2]
    cov: np.ndarray = None  # [N, K, 2, 2]
    alpha: np.ndarray = None  # [N, K, 2]
    pca_cov: np.ndarray = None # [N, K, 2]
    post_mu: np.ndarray = None # [N, K, 2]
    post_cov: np.ndarray = None # [N, K, 2, 2]

    # Samples
    contour_samples: np.ndarray = None  # [N, T, K, 2]
    pred_samples: np.ndarray = None  # [N, T, H, W]
    entropy_map: np.ndarray = None   # [N, H, W]
    sample_weights: np.ndarray = None

    # For a view of N (2 for ES and ED) samples
    view_metrics: dict = None  # Dict of arrays of shape (1,)
    instant_metrics: dict = None  # Dict of arrays of shape (N,)

    view_uncertainty: dict = None  # Dict of arrays of shape (1,)
    instant_uncertainty: dict = None  # Dict of arrays of shape (N,)
    point_uncertainty: dict = None  # Dict of arrays of shape (N, K)

    # Validity
    contour_validity: np.ndarray = None  # Validity of prediction contour (contour_mu)
    sample_validity: np.ndarray = None  # Validity of sampled contours.

    def __post_init__(self):

        assert self.img.ndim in [3, 4]  # May or may not contain channel axis
        n = self.img.shape[0]
        w = self.img.shape[-1]
        h = self.img.shape[-2]

        assert self.gt.shape == (n, h, w)
        assert self.pred.shape == (n, h, w), f'Invalid shape for pred: got {self.pred.shape}'
        assert self.uncertainty_map.shape == (n, h, w), f'Invalid shape for u-map: got {self.uncertainty_map.shape}'

        if self.entropy_map is not None:
            assert self.entropy_map.shape == (n, h, w), f'Invalid shape for entropy: got {self.entropy_map.shape}'

        if self.instant_uncertainty is not None:
            for key, item in self.instant_uncertainty.items():
                assert item.ndim == 1 and len(item) == n, f'Invalid shape for instant_metrics {key}: {item.shape}'

        if self.mu is not None:
            assert self.mu.ndim == 3 and self.mu.shape[0] == n and self.mu.shape[-1] == 2
            k = self.mu.shape[1]

            assert self.cov.shape == (n, k, 2, 2)
            assert self.mode.shape == (n, k, 2)
            assert self.alpha is None or self.alpha.shape == (n, k, 2)

        # if self.pred_samples is not None:
        #     assert self.pred_samples.ndim == 4 and self.pred_samples.shape[0] == n, f'Invalid shape for pred samples: got {self.pred_samples.shape}'
        #     assert self.pred_samples.shape[-1] == w and self.pred_samples.shape[-2] == h



LV_example_shape = np.array([[114., 205.],
                          [105., 185.],
                          [102., 167.],
                          [102., 151.],
                          [101., 134.],
                          [98., 116.],
                          [98., 100.],
                          [99., 83.],
                          [100., 67.],
                          [106., 51.],
                          [123., 35.],
                          [139., 40.],
                          [153., 55.],
                          [168., 74.],
                          [174., 92.],
                          [179., 111.],
                          [182., 128.],
                          [184., 145.],
                          [182., 163.],
                          [179., 180.],
                          [173., 200.]])
LV_MYO_example_shape = np.array([[106.25, 203.75],
 [ 92.5 , 187.5 ],
 [ 88.75, 172.5 ],
 [ 88.75, 157.5 ],
 [ 90.  , 142.5 ],
 [ 93.75, 127.5 ],
 [ 93.75, 112.5 ],
 [ 93.75,  97.5 ],
 [ 93.75,  82.5 ],
 [ 96.25,  67.5 ],
 [108.75,  52.5 ],
 [123.75,  56.25],
 [138.75,  68.75],
 [152.5 ,  82.5 ],
 [160.  ,  97.5 ],
 [165.  , 112.5 ],
 [170.  , 127.5 ],
 [172.5 , 142.5 ],
 [173.75, 157.5 ],
 [171.25, 172.5 ],
 [162.5 , 191.25],
 [ 72.5 , 210.  ],
 [ 68.75, 191.25],
 [ 67.5 , 172.5 ],
 [ 66.25, 155.  ],
 [ 66.25, 136.25],
 [ 66.25, 117.5 ],
 [ 66.25, 100.  ],
 [ 67.5 ,  81.25],
 [ 70.  ,  63.75],
 [ 78.75,  45.  ],
 [ 98.75,  30.  ],
 [118.75,  28.75],
 [138.75,  30.  ],
 [158.75,  40.  ],
 [176.25,  60.  ],
 [185.  ,  80.  ],
 [191.25, 100.  ],
 [196.25, 121.25],
 [198.75, 141.25],
 [198.75, 161.25],
 [191.25, 185.  ],])

XRAY_example_shape = np.array([[ 79.07769775,  15.61120224],
 [ 69.05825806,  19.01955414],
 [ 59.02706146,  26.83041   ],
 [ 48.99978638,  33.064888  ],
 [ 43.0824585 ,  43.05845261],
 [ 38.29551697,  53.05545044],
 [ 35.84365082,  63.05884552],
 [ 34.56684113,  73.06519318],
 [ 32.54629517,  83.06970215],
 [ 29.69275665,  93.07203674],
 [ 26.67584991, 103.07357788],
 [ 25.39902115, 113.0802002 ],
 [ 22.69446564, 123.08255005],
 [ 20.97131348, 133.08798218],
 [ 19.69446182, 143.09458923],
 [ 19.04232025, 153.1026001 ],
 [ 17.21515274, 163.10772705],
 [ 14.92694092, 173.11151123],
 [ 13.6501236 , 183.11781311],
 [ 14.24735641, 193.12965393],
 [ 14.21992874, 203.13934326],
 [ 13.88014221, 213.14852905],
 [ 19.68549728, 203.32550049],
 [ 27.04916   , 194.75610352],
 [ 36.8974762 , 191.34735107],
 [ 46.73892975, 190.43734741],
 [ 56.58035278, 189.52709961],
 [ 66.41494751, 191.11593628],
 [ 76.24396515, 194.73486328],
 [ 86.07044983, 199.290802  ],
 [ 78.92302704, 185.9442749 ],
 [ 78.02253723, 172.61514282],
 [ 79.62082672, 159.29263306],
 [ 84.55084991, 145.97924805],
 [ 85.31611633, 132.6546936 ],
 [ 88.47609711, 119.33638   ],
 [ 94.03085327, 106.02474213],
 [101.04295349,  92.71727753],
 [105.45236206,  79.40247345],
 [107.05064392,  66.07990265],
 [108.64891052,  52.75762939],
 [107.12368774,  39.42654037],
 [102.78744507,  26.0877037 ],
 [ 92.40454865,  15.64774513],
 [173.10301208,  12.43318558],
 [183.52957153,  17.90568542],
 [193.94973755,  25.56427383],
 [203.0236969 ,  36.03062057],
 [209.24211121,  46.48883057],
 [212.02441406,  56.93795013],
 [215.61004639,  67.38896179],
 [218.52651978,  77.83840942],
 [222.06765747,  88.28956604],
 [224.09143066,  98.73625946],
 [226.33859253, 109.18392944],
 [228.09472656, 119.62986755],
 [230.25250244, 130.07727051],
 [232.41033936, 140.52438354],
 [234.56808472, 150.97169495],
 [236.85986328, 161.41946411],
 [239.50836182, 171.8678894 ],
 [239.79211426, 182.31011963],
 [239.76339722, 192.75123596],
 [240.04718018, 203.19352722],
 [238.59091187, 213.63069153],
 [233.7428894 , 224.05889893],
 [231.46490479, 219.57568359],
 [228.56164551, 215.09060669],
 [224.72094727, 210.91558838],
 [220.25468445, 207.05101013],
 [215.7855072 , 204.12332153],
 [211.31588745, 201.50827026],
 [213.84107971, 191.8324585 ],
 [214.8046875 , 182.15226746],
 [211.39547729, 172.4602356 ],
 [206.42442322, 162.7638092 ],
 [202.07821655, 153.0690918 ],
 [192.41851807, 144.60925293],
 [182.75384521, 138.0234375 ],
 [174.02700806, 131.12786865],
 [170.30548096, 121.43487549],
 [167.83319092, 111.74529266],
 [163.17460632, 102.04975891],
 [160.07780457,  92.35845947],
 [156.668396  ,  82.66633606],
 [154.81767273,  74.14978027],
 [150.62437439,  65.62682343],
 [142.75692749,  58.5774231 ],
 [134.25149536,  56.36769104],
 [133.65454102,  46.26689911],
 [138.05502319,  36.17951965],
 [143.39257812,  26.09501266],
 [152.89299011,  16.646492  ],
 [163.00485229,  12.093153  ],
 [ 84.20066071, 197.72393799],
 [ 78.91957092, 187.21456909],
 [ 77.69894409, 176.71632385],
 [ 78.3523941 , 166.22328186],
 [ 80.69251251, 155.73483276],
 [ 85.5938797 , 145.25332642],
 [ 95.48802185, 133.23292542],
 [108.49554443, 124.96887207],
 [121.49259949, 120.45332336],
 [134.48188782, 118.83782959],
 [147.46554565, 119.18580627],
 [160.43836975, 123.59425354],
 [173.39984131, 132.06315613],
 [185.7681427 , 139.90577698],
 [198.12995911, 150.03881836],
 [206.73846436, 162.13989258],
 [212.63919067, 174.54602051],
 [215.72859192, 186.94406128],
 [211.32171631, 199.32188416],
 [195.41418457, 205.48614502],
 [179.50552368, 212.04092407],
 [163.60044861, 217.30726624],
 [147.6995697 , 221.01173401],
 [131.80981445, 220.65579224],
 [115.92955017, 216.86419678],
 [100.06219482, 208.34822083],])

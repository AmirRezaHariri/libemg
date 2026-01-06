from libemg._datasets.dataset import Dataset
from libemg.data_handler import OfflineDataHandler
import pickle
import numpy as np
from libemg.feature_extractor import FeatureExtractor
from libemg.utils import *

class EMGEPN100(Dataset):
    def __init__(self, dataset_file: str = 'DATASET_85', cross_user: bool = True):
        split = '30 Reps x 12 Gestures x 43 Users (Train), 15 Reps x 12 Gestures x 42 Users (Test) --> Cross User Split'
        if not cross_user:
            split = '15 Reps x 12 Gestures (Train), 15 Reps x 12 Gestures (Test) from the 43 Test Users --> User Dependent Split'

        Dataset.__init__(self, 
                         500, 
                         8, 
                         ('Myo', 'gForce'), 
                         85, 
                         {  "relax": 0,       # Matches EPN-612 static classes IDs
                            "fist": 1,
                            "wave in": 2,
                            "wave out": 3,
                            "open": 4,
                            "pinch": 5,
                            "up": 6,
                            "down": 7,
                            "left": 8,
                            "right": 9,
                            "forward": 10,
                            "backward": 11,}, 
                         split,
                         "EMG dataset for 12 different hand gesture categories using the Myo armband and the G-force armband.", 
                         'https://doi.org/10.3390/s22249613')
        self.resolution_bit = (8, 12)
        self.url = "https://nextcloud.epn.edu.ec/index.php/s/wmPWpFWSo4XxLLA/download"
        self.dataset_name = dataset_file

    def get_odh(self, subjects=None, feature_list = None, window_size = None, window_inc = None, feature_dic = None):
        pass


        
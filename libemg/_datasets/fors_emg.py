from libemg._datasets.dataset import Dataset
from libemg.data_handler import OfflineDataHandler, RegexFilter
import os
import scipy.io
import numpy as np

class FORSEMG(Dataset):
    def __init__(self):
        Dataset.__init__(self, 
                         985, 
                         8, 
                         'Experimental Device', 
                         19, 
                         {0: 'Thump Up', 1: 'Index', 2: 'Right Angle', 3: 'Peace', 4: 'Index Little', 5: 'Thumb Little', 6: 'Hand Close', 7: 'Hand Open', 8: 'Wrist Flexion', 9: 'Wrist Extension', 10: 'Ulnar Deviation', 11: 'Radial Deviation'}, 
                         '5 Train, 10 Test (2 Forarm Orientations x 5 Reps)', 
                         "FORS-EMG: Twelve gestures elicited in three forearm orientations (neutral, pronation, and supination).", 
                         'https://arxiv.org/abs/2409.07484t')
        self.dataset_name = 'FORS-EMG'
        self.dataset_folder = os.path.join(self.save_dir , self.dataset_name)

    def prepare_data(self):
        print('\nPlease cite: ' + self.citation+'\n')
        if (not self.check_exists(self.dataset_folder)):
            print("Please download the pickled dataset from: https://www.kaggle.com/datasets/ummerummanchaity/fors-emg-a-novel-semg-dataset?resource=download")
            return 
        
        odh = OfflineDataHandler()
        odh.subjects = []
        odh.classes = []
        odh.reps = []
        odh.orientation = []
        odh.extra_attributes = ['subjects', 'classes', 'reps', 'orientation']

        for s in range(1, 20):
            for g_i, g in enumerate(['Thumb_UP', 'Index', 'Right_Angle', 'Peace', 'Index_Little', 'Thumb_Little', 'Hand_Close', 'Hand_Open', 'Wrist_Flexion', 'Wrist_Extension', 'Radial_Deviation']):
                for r in [1,2,3,4,5]:
                    for o_i, o in enumerate(['rest', 'pronation', 'supination']):
                        mat = scipy.io.loadmat('FORS-EMG/Subject' + str(s) + '/' + o + '/' + g + '-' + str(r) + '.mat')
                        odh.data.append(mat['value'].T)
                        odh.classes.append(np.ones((len(odh.data[-1]), 1)) * g_i)
                        odh.subjects.append(np.ones((len(odh.data[-1]), 1)) * s-1)
                        odh.reps.append(np.ones((len(odh.data[-1]), 1)) * r-1)
                        odh.orientation.append(np.ones((len(odh.data[-1]), 1)) * o_i)
        return {'All': odh, 'Train': odh.isolate_data('orientation', [0]), 'Test': odh.isolate_data('orientation', [1,2])}
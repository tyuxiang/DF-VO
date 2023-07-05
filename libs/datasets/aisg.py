''''''
'''
@Author: Tan Yu Xiang
@Date: 2023-07-05
@Description: Dataset loaders for AISG Driving Sequence
'''

from glob import glob
import os

from .dataset import Dataset
from libs.general.utils import *
import json
import datetime


class AISG(Dataset):
    """Base class of dataset loaders for AISG Driving Sequence
    """
    
    def __init__(self, *args, **kwargs):
        super(AISG, self).__init__(*args, **kwargs)
        return

    def synchronize_timestamps(self):
        """Synchronize RGB, Depth, and Pose timestamps to form pairs
        
        Returns:
            a dictionary containing
                - **rgb_timestamp** : {'depth': depth_timestamp, 'pose': pose_timestamp}
        """
        # Load timestamps
        timestamp_txt = os.path.join(self.cfg.directory.img_seq_dir,
                                    self.cfg.seq,
                                    "train"+str(self.cfg.seq[-1])+".time")
                                    # "stereo",
                                    # "times-single-absolute.txt")
        timestamps = np.loadtxt(timestamp_txt)*1e6
        self.timestamps = timestamps
        self.rgb_d_pose_pair = {}
        len_seq = len(glob(os.path.join(self.data_dir['img'], "*.{}".format(self.cfg.image.ext))))
        for i in range(len_seq):
            self.rgb_d_pose_pair[i] = {}
            self.rgb_d_pose_pair[i]['depth'] = i
            self.rgb_d_pose_pair[i]['pose'] = i
    
    def get_timestamp(self, img_id):
        """Get timestamp for the query img_id

        Args:
            img_id (int): query image id

        Returns:
            timestamp (int): timestamp for query image
        """
        return img_id
    
    def save_result_traj(self, traj_txt, poses):
        """Save trajectory (absolute poses) as KITTI odometry file format

        Args:
            txt (str): pose text file path
            poses (dict): poses, each pose is a [4x4] array
        """
        global_poses_arr = convert_SE3_to_arr(poses)
        save_traj(traj_txt, global_poses_arr, format='robotcar', timestamps=self.timestamps[self.cfg.start_frame:])

    def get_intrinsics_param(self):
        """Read intrinsics parameters for each dataset

        Returns:
            intrinsics_param (list): [cx, cy, fx, fy]
        """
        img_seq_dir = self.cfg.directory.config_dir
        f = open(os.path.join(img_seq_dir, "intrinsic_parameters.json"))
        intrinsics_param = json.load(f)
        param = [intrinsics_param['Cx'], intrinsics_param['Cy'], intrinsics_param['fx'], intrinsics_param['fy']]
        f.close()
        return param
    
    def get_data_dir(self):
        """Get data directory

        Returns:
            a dictionary containing
                - **img** (str) : image data directory
                - (optional) **depth** (str) : depth data direcotry or None
                - (optional) **depth_src** (str) : depth data type [gt/None]
        """
        data_dir = {}

        # get image data directory
        img_seq_dir = os.path.join(
                            self.cfg.directory.img_seq_dir,
                            self.cfg.seq
                            )
        data_dir['img'] = os.path.join(img_seq_dir, "train_images-"+self.cfg.seq[-1])

        # get depth data directory
        data_dir['depth_src'] = self.cfg.depth.depth_src

        if data_dir['depth_src'] == "gt":
            data_dir['depth'] = "{}/gt/{}/".format(
                                self.cfg.directory.depth_dir, self.cfg.seq
                            )
        elif data_dir['depth_src'] is None:
            data_dir['depth'] = None
        else:
            assert False, "Wrong depth src [{}] is given.".format(data_dir['depth_src'])
 
        return data_dir
    
    def convert_filename(self, unix_ts):
        out = datetime.datetime.fromtimestamp(unix_ts, datetime.timezone(datetime.timedelta(hours=8))).strftime('%Y%m%d_%I%M%S')
        out = out[2:] + str(int(datetime.datetime.fromtimestamp(unix_ts, datetime.timezone(datetime.timedelta(hours=8))).strftime('%f')))
        return out
    
    def get_image(self, timestamp):
        """Get image data given the image timestamp

        Args:
            timestamp (int): timestamp for the image
            
        Returns:
            img (array, [CxHxW]): image data
        """
        # file = self.convert_filename(timestamp)
        img_path = sorted(glob(os.path.join(self.data_dir['img'], '*.'+self.cfg.image.ext)))[timestamp]
        # img_path = os.path.join(self.data_dir['img'], 
        #                     file+"."+self.cfg.image.ext
        #                     )
        img = read_image(img_path, self.cfg.image.height, self.cfg.image.width)
        return img
     
    def get_depth(self, timestamp):
        raise NotImplementedError
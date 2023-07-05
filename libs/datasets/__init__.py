from .kitti import KittiOdom, KittiRaw
from .tum import TUM
from .adelaide import Adelaide
from .kinect import Kinect
from .oxford_robotcar_custom import OxfordRobotCar 
from .aisg import AISG 
# from .fourseasons import FourSeasons 
# from .dso import Singapore
# marcelprasetyo: modified oxford_robotcar.py to oxford_robotcar_custom.py therefore accordingly changed the above

datasets = {
            "kitti_odom": KittiOdom,
            "kitti_raw": KittiRaw,
            "tum-1": TUM,
            "tum-2": TUM,
            "tum-3": TUM,
            "adelaide1": Adelaide,
            "adelaide2": Adelaide,
            "kinect": Kinect,
            'robotcar': OxfordRobotCar,
            "aisg": AISG,
            # "4seasons": FourSeasons,
            # "singapore": Singapore
        }
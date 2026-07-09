import av2
import av2.utils.io as io_utils
import numpy as np
import pandas as pd

from pypcd import pypcd, pdutil
from pathlib import Path

def read_pcd(path, sensor_type="invs"):
    pc = pypcd.PointCloud.from_path(path)
    points = np.vstack((
        pc.pc_data['x'],
        pc.pc_data['y'],
        pc.pc_data['z'],
        pc.pc_data['intensity'],
        pc.pc_data['label']
    )).T
    if sensor_type == "rbszv":
        points = points[(points[:, 4] == 26) | (
            points[:, 4] == 27) | (points[:, 4] == 29)]
        points = points[:, :4]
    elif sensor_type in ["rbs", "invs"]:
        points = points[points[:, 4] == 26]
        points = points[:, :4]
    elif sensor_type == "zv":
        points = points[(points[:, 4] == 27) | (points[:, 4] == 29)]
        points = points[:, :4]
    else:
        raise NotImplementedError
    points = points.astype(np.float32)
    return points
def write_pcd(path, df):
    """"""
    pc_data = {
        'x': df['x'].to_numpy(),
        'y': df['y'].to_numpy(),
        'z': df['z'].to_numpy(),
        'intensity': df['intensity'].to_numpy().astype(float)/255,
    }
    pc_df = pd.DataFrame(pc_data)
    pc = pdutil.data_frame_to_point_cloud(pc_df)
    pc.save_pcd(path, 'binary_compressed')
    return

if __name__ == '__main__':
    
    path = r"data/av2/sensor/train/00a6ffc1-6ce9-3bc3-a060-6006e9893a1a/sensors/lidar/315967376859506000.feather"
    path = Path(path)
    df = io_utils.read_feather(path)

    print(f"--- display point cloud info: ")
    print(df.shape)
    print(df.head(3))

    # convert to pcd
    #----------------------------------------------
    pcd_path = f"./{path.stem}.pcd"

    df = df.astype(dict([(name, float) for name in df.columns]))
    pc = pdutil.data_frame_to_point_cloud(df)
    pc.save_pcd(pcd_path, 'binary_compressed')

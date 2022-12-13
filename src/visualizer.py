import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation as R
import cv2
import io


def get_img_from_fig(fig, dpi=100):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi)
    buf.seek(0)
    img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    buf.close()
    img = cv2.imdecode(img_arr, 1)
    return img


class Visualization(object):
    def __init__(self, K, d, input_wh):
        self.bev_img_h = 480
        self.bev_img_w = 480

        ipm_height_max_meter = 52
        ipm_height_min_meter = 10
        ipm_width_meter = 8
        self.camera_height = 2
        self.ipm_region = np.float32([[-ipm_width_meter, ipm_height_max_meter],
                                      [ipm_width_meter, ipm_height_max_meter],
                                      [-ipm_width_meter,
                                       (ipm_height_max_meter+ipm_height_min_meter)/2],
                                      [ipm_width_meter, (ipm_height_max_meter+ipm_height_min_meter)/2]])
        self.H_target_pts = np.float32([[0, 0],
                                        [self.bev_img_w-1, 0],
                                        [0, self.bev_img_h/2],
                                        [self.bev_img_w-1, self.bev_img_h/2]])

        self.front2vcs = np.asarray([[0, -1, 0, 0],
                                     [1, 0, 0, -20],
                                     [0, 0, 1, 0.7],
                                     [0, 0, 0, 1]], dtype=np.float32)

        self.K = K
        self.d = d
        self.input_wh = input_wh
        self.output_wh = (480, 256)
        # resize K by input and output shape
        self.output_K = self.K.copy()
        self.output_K[0, :] *= self.output_wh[0] / self.input_wh[0]
        self.output_K[1, :] *= self.output_wh[1] / self.input_wh[1]

        # storage for plots
        self.plot_history_length = 50
        self.compensation_rotations = []

    def get_vanishing_line(self, dynamic_R=np.eye(3, dtype=np.float32)):
        R_mat = dynamic_R
        R_mat = R_mat.astype(np.float32)
        t = np.zeros((3, 1), dtype=np.float32)
        Rt = np.concatenate((R_mat, t), axis=1)
        P = self.output_K @ Rt
        p0 = (P[0, 2]/P[2, 2], P[1, 2]/P[2, 2])
        p1 = (0, p0[1])
        p2 = (self.output_wh[0], p0[1])
        return np.asarray(p1).astype(int), np.asarray(p2).astype(int)

    def paint_vanishing_line(self, img, dynamic_R):
        paint_img = img.copy()
        p1, p2 = self.get_vanishing_line()
        cv2.line(paint_img, p1, p2, (0, 0, 255), thickness=1)
        p1, p2 = self.get_vanishing_line(dynamic_R)
        cv2.line(paint_img, p1, p2, (0, 255, 0), thickness=1)
        return paint_img

    def get_bev(self, img, dynamic_R=np.eye(3, dtype=np.float32)):
        ground_pts_2d = self.ipm_region.copy()
        y_axis = (self.camera_height * np.ones((4, 1), dtype=np.float32)).copy()
        ground_pts_vcs_3d = np.concatenate(
            (ground_pts_2d[:, :1], y_axis, ground_pts_2d[:, 1:]), axis=1).T
        img_plane = self.K @ dynamic_R @ ground_pts_vcs_3d
        img_plane /= img_plane[2, :]
        H_source_pts = (img_plane[:2, :].T).astype(np.float32)
        H = cv2.getPerspectiveTransform(H_source_pts, self.H_target_pts)
        ipm_img = cv2.warpPerspective(img, H, (self.bev_img_w, self.bev_img_h))
        return ipm_img

    def get_plot(self):
        idx = len(self.compensation_rotations)
        start_idx = max(0, idx - self.plot_history_length)
        timestamps = np.arange(start_idx, idx, 1)
        pitch_list = []
        for j in range(start_idx, idx):
            rotation = self.compensation_rotations[j]
            pitch = R.from_matrix(rotation).as_euler('zxy', degrees=True)[1]
            pitch_list.append(pitch)
        fig, ax1 = plt.subplots(figsize=(4.8, 2.56), dpi=100)
        axes = plt.gca()
        axes.set_ylim([-2, 2])
        ax1.set_xlabel('frame')
        ax1.set_ylabel('pitch (degree)')
        ax1.plot((timestamps, timestamps), ([0] * len(timestamps), pitch_list), c='black')
        ax1.scatter(timestamps, pitch_list, s=10, c='red')
        plt.tight_layout()
        cv_plot = get_img_from_fig(fig)
        plt.close(fig)
        return cv_plot

    def get_frame(self, image, dynamic_R):
        undistort_image = cv2.undistort(image, self.K, self.d)
        resized_image = cv2.resize(undistort_image, self.output_wh)
        self.compensation_rotations.append(dynamic_R)
        vanilla_bev = self.get_bev(undistort_image)
        calib_bev = self.get_bev(undistort_image, dynamic_R)
        vanishing_line = self.paint_vanishing_line(resized_image, dynamic_R)
        plot = self.get_plot()
        top = np.concatenate((vanilla_bev, calib_bev), axis=1)
        bottom = np.concatenate((plot, vanishing_line), axis=1)
        combined = np.concatenate((top, bottom), axis=0)
        return combined

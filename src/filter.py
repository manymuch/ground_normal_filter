import numpy as np
import inekf


class So3Process(inekf.ProcessModel[inekf.SO3, "Vec3"]):
    def __init__(self, Q):
        super().__init__()
        self.Q = Q

    def f(self, u, dt, state):
        return state

    def makePhi(self, u, dt, state, error):
        return np.eye(3, dtype=float)


class GroundNormalFilterIEKF:
    def __init__(self):
        cov_init = np.eye(3, dtype=np.float32)
        process_cov = np.eye(3, dtype=np.float32) * 1e-2
        b = np.asarray([0., 0., 1.])
        measure_cov = np.eye(3)
        measure = inekf.MeasureModel[inekf.SO3](b, measure_cov, inekf.ERROR.LEFT)
        process = So3Process(process_cov)
        x0 = inekf.SO3(0, 0, 0, cov_init)
        iekf = inekf.InEKF(process, x0, inekf.ERROR.LEFT)
        iekf.addMeasureModel("measure", measure)
        self.iekf = iekf
        self.u = inekf.SO3(0., 0., 0.).log
        self.cumulative_rotation = np.eye(3, dtype=np.float32)

    def update(self, relative_se3):
        relative_so3 = relative_se3[:3, :3]
        self.cumulative_rotation = self.cumulative_rotation @ relative_so3
        state = self.iekf.predict(self.u)
        predict_rotation = state.mat.copy()
        measure_vector = self.cumulative_rotation[:, 2]
        self.iekf.update("measure", measure_vector)
        compensation_se3 = np.eye(4, dtype=np.float32)
        compensation_so3 = predict_rotation.T @ self.cumulative_rotation
        compensation_se3[:3, :3] = compensation_so3
        return compensation_se3

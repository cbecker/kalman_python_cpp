import numpy as np

# offsets of each variable in the state vector
iX = 0
iV = 1
NUMVARS = iV + 1


class KF:
    def __init__(self, initial_x: float, 
                       initial_v: float,
                       accel_variance: float) -> None:
        # mean of state GRV
        self._x = np.zeros(NUMVARS)

        self._x[iX] = initial_x
        self._x[iV] = initial_v
        
        self._accel_variance = accel_variance

        # covariance of state GRV
        self._P = np.eye(NUMVARS)

    def predict(self, dt: float) -> None:
        # x = F x
        # P = F P Ft + G Gt a
        F = np.eye(NUMVARS)
        F[iX, iV] = dt
        new_x = F.dot(self._x)

        G = np.zeros((2, 1))
        G[iX] = 0.5 * dt**2
        G[iV] = dt
        new_P = F.dot(self._P).dot(F.T) + G.dot(G.T) * self._accel_variance

        self._P = new_P
        self._x = new_x

    def update(self, meas_value: float, meas_variance: float):
        # y = z - H x
        # S = H P Ht + R
        # K = P Ht S^-1
        # x = x + K y
        # P = (I - K H) * P

        H = np.zeros((1, NUMVARS))
        H[0, iX] = 1

        z = np.array([meas_value])
        R = np.array([meas_variance])

        y = z - H.dot(self._x)
        S = H.dot(self._P).dot(H.T) + R

        K = self._P.dot(H.T).dot(np.linalg.inv(S))

        new_x = self._x + K.dot(y)
        new_P = (np.eye(2) - K.dot(H)).dot(self._P)

        self._P = new_P
        self._x = new_x

    @property
    def cov(self) -> np.array:
        return self._P

    @property
    def mean(self) -> np.array:
        return self._x

    @property
    def pos(self) -> float:
        return self._x[iX]

    @property
    def vel(self) -> float:
        return self._x[iV]

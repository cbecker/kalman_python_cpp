import numpy as np

# offsets of each variable in the state vector
iX = 0
iV = 1
NUMVARS = iV + 1

"""
Creating a class named KF for Kalman Filter that initializes itself
three arguments initial position,velocity, and the acceleration variance.
"""
class KF:
    def __init__(self, initial_x: float, 
                       initial_v: float,
                       accel_variance: float) -> None:

        """
        mean of state GRV
        This outputs a new array given the shape of the new array.
        So for example if NUMWARS is 5 then X will be
        X = [0,0,0,0,0]
        This will serve as out state matrix
        """
        self._x = np.zeros(NUMVARS)

        """
        two class variables that are arrays have been created that have 
        the Initial Values. Lastly an accelerate variance has also been 
        added from our inital accerlation variance.
        """

        self._x[iX] = initial_x
        self._x[iV] = initial_v
        self._accel_variance = accel_variance

        self._Measured = 0;
        """
        This P will serve as the Initial Process of Covariance Matrix 
        """
        self._P = np.eye(NUMVARS)

    """
    This function will run the predicted State Matrix value and
    Predicted Process of Covariance matrix. The general equations are:
        X_kp = AX_kp-1 + BU_k + W_k
        P_kp = AP_kp-1A^T + Q_K
        
        Notice that in our Predicted State Matrix we do not have a control variable 
        or a predicted noise of the state for simplicity. 
        
    """
    def predict(self, dt: float) -> None:
        # x = A x_kp
        # P = A P_kp A^T + Q_K

        """
        'A' will be our state transition matrix and it simply
        a conversion matrix. We set the matrix oh to have data
        that matches the sampling time. Lastly the dot product is
        done between A and our State Matrix to get the Predicted
        New State. Here is were we can Predicted noise if wanted or
        needed and also our control variable with its control transition matrix
        """
        A = np.eye(NUMVARS)
        A[iX, iV] = dt
        new_x = A.dot(self._x)

        """
        'Q_K' will be our Process Noise Covariance Matrix which is a size
        of 2 x 1 matrix for this example. 
        Q_K = | 0 |
              | 0 |
        We thing initialize  the Process noise with 1/2 of the rate of change in time squared
        and the deltT for velocity. Lastly we calculated the predicted covariance matrix
        by the simple equation below.
                new_P = A.dot(self._P).dot(A.T) + Q_K.dot(Q_K.T) * self._accel_variance
        """
        Q_K = np.zeros((2, 1))
        Q_K[iX] = 0.5 * dt**2
        Q_K[iV] = dt
        new_P = A.dot(self._P).dot(A.T) + Q_K.dot(Q_K.T) * self._accel_variance

        """
        We update our state or state vector and processed covariance matrix with the new predicted state
        """
        self._P = new_P
        self._x = new_x

    """
    This function takes care of calculating the Kalman Gain and
    updating the new state based on the predicted value and the measurement 
    input.
    Y = CX_km + Z_k
    x = x + K(y - HX_kp)
    K = P_kp H/ (H P Ht + R)
    """
    def update(self, meas_value: float, meas_variance: float):
        # y = H - Z_k
        # K = P_kp H/ (H P Ht + R)
        # x = x + K(y - HX_kp)
        # P = (I - K H) * P_kp

        """
        In this function the measurement Noise matrix is being
        created with the number of rows of the array being
        the index of the filter.
        """
        Z_k = np.zeros((1, NUMVARS))
        Z_k[0, iX] = 1

        """
        Measured values are adding into the measurement matrix
        and finally the final Measurement vector was created
        """
        X_km = np.array([meas_value])
        y = X_km - Z_k.dot(self._x)
        self._Measured = y
        """
        Next the Kalman Gain will be calculated in this project
        which is typically  modeled as:
        K = P_kp H/ (H P_kp H^T + R )
        
        First the Sensor Noise Covariance Matrix or 
        Measurement Error which is denoted as R.
        
        The Bottom half of the Kalman Gain is then calculated
        based on the error measurement and process of covariance
        matrix. '_P' and the Observation matrix is also shown.
        Finally the Kalman Gain is called as shown below.
        """
        R = np.array([meas_variance])
        S = Z_k.dot(self._P).dot(Z_k.T) + R
        K = self._P.dot(Z_k.T).dot(np.linalg.inv(S))

        """
        Next the new-state vector and process covariance matrix
        are Calculated 
        """
        new_x = self._x + K.dot(y)

        new_P = (np.eye(2) - K.dot(Z_k)).dot(self._P)

        self._P = new_P
        self._x = new_x

    """
    Function for Covariance
    """
    @property
    def cov(self) -> np.array:
        return self._P

    """
     Function for Mean
     """
    @property
    def mean(self) -> np.array:
        return self._x

    """
     Function for Position
    """
    @property
    def pos(self) -> float:
        return self._x[iX]

    """
     Function for Velocity
    """
    @property
    def vel(self) -> float:
        return self._x[iV]
    """
     Function for Velocity
    """
    @property
    def Measured(self) -> float:
        return self._Measured[iX]

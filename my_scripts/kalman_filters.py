import numpy as np
from math import ceil
from copy import deepcopy
from square_bounding_box import *


crop_alpha = 0.95

class Kalman:

    def __init__(self, num_of_joints=15):
        self.dim = num_of_joints*3
        self.x = np.zeros((self.dim, 1))          # state
        self.P = np.eye(self.dim)                 # uncertainty covariance
        self.Q = np.eye(self.dim)                 # process uncertainty
        self.F = np.eye(self.dim)                 # state transition matrix
        self.H = np.eye(self.dim, self.dim)  # Measurement function
        self.R = np.eye(self.dim)                 # state uncertainty
        self.M = np.zeros((self.dim, self.dim))           # process-measurement cross correlation
        self.z = np.array([[None]*self.dim]).T
        self._I = np.eye(self.dim)


        # gain and residual are computed during the innovation step. We
        # save them so that in case you want to inspect them for various
        # purposes
        self.K = np.zeros((self.dim, self.dim)) # kalman gain
        self.y = np.zeros((self.dim, 1))
        self.S = np.zeros((self.dim, self.dim)) # system uncertainty
        self.SI = np.zeros((self.dim, self.dim)) # inverse system uncertainty

        # these will always be a copy of x,P after predict() is called
        self.x_prior = self.x.copy()
        self.P_prior = self.P.copy()

        # these will always be a copy of x,P after update() is called
        self.x_post = self.x.copy()
        self.P_post = self.P.copy()

    def init_state(self, state):
        self.x = state.copy()

    def init_process_noise(self, KALMAN_PROCESS_NOISE_AMOUNT):
        self.Q = np.eye(self.dim)*KALMAN_PROCESS_NOISE_AMOUNT

    def predict(self):
        """
        Predict next state (prior) using the Kalman filter state propagation
        equations.
        """
        #self.x = np.dot(self.F, self.x)
        self.P = self.P + self.Q
        # P = FPF' + Q
        #self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q

        # save prior
        self.x_prior = self.x.copy()
        self.P_prior = self.P.copy()

    def update(self, z, noise_cov=None, H=None):
        """
        Add a new measurement (z) to the Kalman filter.
        """

        # set to None to force recompute
        self._log_likelihood = None
        self._likelihood = None
        self._mahalanobis = None

        if z is None:
            self.z = np.array([[None]*self.dim]).T
            self.x_post = self.x.copy()
            self.P_post = self.P.copy()
            self.y = np.zeros((self.dim, 1))
            return

        #z = reshape_z(z, self.dim, self.x.ndim)

        R = np.zeros([self.dim, self.dim])
        for index in range(0, int(self.dim/3)):
            R[index*3:(index+1)*3,index*3:(index+1)*3] = noise_cov

        if H is None:
            H = self.H

        # y = z - Hx
        # error (residual) between measurement and prediction
        self.y = z - np.dot(H, self.x)

        # common subexpression for speed
        PHT = np.dot(self.P, H.T)

        # S = HPH' + R
        # project system uncertainty into measurement space
        self.S = np.dot(H, PHT) + R
        self.SI = np.linalg.inv(self.S)
        # K = PH'inv(S)
        # map system uncertainty into kalman gain
        self.K = np.dot(PHT, self.SI)

        # x = x + Ky
        # predict new x with residual scaled by the kalman gain
        self.x = self.x + np.dot(self.K, self.y)

        # P = (I-KH)P(I-KH)' + KRK'
        # This is more numerically stable
        # and works for non-optimal K vs the equation
        # P = (I-KH)P usually seen in the literature.

        I_KH = self._I - np.dot(self.K, H)
        self.P = np.dot(np.dot(I_KH, self.P), I_KH.T) + np.dot(np.dot(self.K, R), self.K.T)

        # save measurement and posterior state
        self.z = deepcopy(z)
        self.x_post = self.x.copy()
        self.P_post = self.P.copy()


class ExtendedKalman:

    def __init__(self, num_of_joints=15):
        self.dim = num_of_joints*3
        self.x = np.zeros((self.dim, 1))          # state
        self.P = np.eye(self.dim)                 # uncertainty covariance
        self.Q = np.eye(self.dim)                 # process uncertainty
        self.F = np.eye(self.dim)                 # state transition matrix
        self.H = np.eye(self.dim, self.dim)  # Measurement function
        self.R = np.eye(self.dim)                 # state uncertainty
        self.M = np.zeros((self.dim, self.dim))           # process-measurement cross correlation
        self.z = np.array([[None]*self.dim]).T
        self._I = np.eye(self.dim)


        # gain and residual are computed during the innovation step. We
        # save them so that in case you want to inspect them for various
        # purposes
        self.K = np.zeros((self.dim, self.dim)) # kalman gain
        self.y = np.zeros((self.dim, 1))
        self.S = np.zeros((self.dim, self.dim)) # system uncertainty
        self.SI = np.zeros((self.dim, self.dim)) # inverse system uncertainty

        # these will always be a copy of x,P after predict() is called
        self.x_prior = self.x.copy()
        self.P_prior = self.P.copy()

        # these will always be a copy of x,P after update() is called
        self.x_post = self.x.copy()
        self.P_post = self.P.copy()

    def init_state(self, state):
        self.x = state.copy()

    def init_process_noise(self, KALMAN_PROCESS_NOISE_AMOUNT):
        self.Q = np.eye(self.dim)*KALMAN_PROCESS_NOISE_AMOUNT

    def predict(self):
        """
        Predict next state (prior) using the Kalman filter state propagation
        equations.
        """
        #self.x = np.dot(self.F, self.x)
        self.P = self.P + self.Q
        # P = FPF' + Q
        #self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q

        # save prior
        self.x_prior = self.x.copy()
        self.P_prior = self.P.copy()

    def update(self, z, noise_cov=None, H=None):
        """
        Add a new measurement (z) to the Kalman filter.
        """

        # set to None to force recompute
        self._log_likelihood = None
        self._likelihood = None
        self._mahalanobis = None

        if z is None:
            self.z = np.array([[None]*self.dim]).T
            self.x_post = self.x.copy()
            self.P_post = self.P.copy()
            self.y = np.zeros((self.dim, 1))
            return

        #z = reshape_z(z, self.dim, self.x.ndim)

        R = np.zeros([self.dim, self.dim])
        for index in range(0, int(self.dim/3)):
            R[index*3:(index+1)*3,index*3:(index+1)*3] = noise_cov

        if H is None:
            H = self.H

        # y = z - Hx
        # error (residual) between measurement and prediction
        self.y = z - np.dot(H, self.x)

        # common subexpression for speed
        PHT = np.dot(self.P, H.T)

        # S = HPH' + R
        # project system uncertainty into measurement space
        self.S = np.dot(H, PHT) + R
        self.SI = np.linalg.inv(self.S)
        # K = PH'inv(S)
        # map system uncertainty into kalman gain
        self.K = np.dot(PHT, self.SI)

        # x = x + Ky
        # predict new x with residual scaled by the kalman gain
        self.x = self.x + np.dot(self.K, self.y)

        # P = (I-KH)P(I-KH)' + KRK'
        # This is more numerically stable
        # and works for non-optimal K vs the equation
        # P = (I-KH)P usually seen in the literature.

        I_KH = self._I - np.dot(self.K, H)
        self.P = np.dot(np.dot(I_KH, self.P), I_KH.T) + np.dot(np.dot(self.K, R), self.K.T)

        # save measurement and posterior state
        self.z = deepcopy(z)
        self.x_post = self.x.copy()
        self.P_post = self.P.copy()

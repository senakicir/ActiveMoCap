import numpy as np
from math import ceil
from square_bounding_box import *


crop_alpha = 0.95

class Kalman:

    def __init__(self, num_of_joints=15):
        self.num_of_joints = num_of_joints
        self.x = np.zeros((num_of_joints, 1))          # state
        self.P = np.eye(num_of_joints)                 # uncertainty covariance
        self.Q = np.eye(num_of_joints)                 # process uncertainty
        self.F = np.eye(num_of_joints)                 # state transition matrix
        self.H = np.eye(num_of_joints, num_of_joints)  # Measurement function
        self.R = np.eye(num_of_joints)                 # state uncertainty
        self.M = np.zeros((num_of_joints, num_of_joints))           # process-measurement cross correlation
        self.z = np.array([[None]*self.num_of_joints]).T

        # gain and residual are computed during the innovation step. We
        # save them so that in case you want to inspect them for various
        # purposes
        self.K = np.zeros((num_of_joints, num_of_joints)) # kalman gain
        self.y = np.zeros((num_of_joints, 1))
        self.S = np.zeros((num_of_joints, num_of_joints)) # system uncertainty
        self.SI = np.zeros((num_of_joints, num_of_joints)) # inverse system uncertainty

        # these will always be a copy of x,P after predict() is called
        self.x_prior = self.x.copy()
        self.P_prior = self.P.copy()

        # these will always be a copy of x,P after update() is called
        self.x_post = self.x.copy()
        self.P_post = self.P.copy()


    def predict(self):
        """
        Predict next state (prior) using the Kalman filter state propagation
        equations.
        """
        self.x = np.dot(F, self.x)

        # P = FPF' + Q
        self.P = np.dot(dot(F, self.P), F.T) + Q

        # save prior
        self.x_prior = self.x.copy()
        self.P_prior = self.P.copy()

    def update(self, z, R=None, H=None):
        """
        Add a new measurement (z) to the Kalman filter.

        If z is None, nothing is computed. However, x_post and P_post are
        updated with the prior (x_prior, P_prior), and self.z is set to None.

        Parameters
        ----------
        z : (dim_z, 1): array_like
            measurement for this update. z can be a scalar if dim_z is 1,
            otherwise it must be convertible to a column vector.

        R : np.array, scalar, or None
            Optionally provide R to override the measurement noise for this
            one call, otherwise  self.R will be used.

        H : np.array, or None
            Optionally provide H to override the measurement function for this
            one call, otherwise self.H will be used.
        """

        # set to None to force recompute
        self._log_likelihood = None
        self._likelihood = None
        self._mahalanobis = None

        if z is None:
            self.z = np.array([[None]*self.num_of_joints]).T
            self.x_post = self.x.copy()
            self.P_post = self.P.copy()
            self.y = np.zeros((self.num_of_joints, 1))
            return

        z = reshape_z(z, self.num_of_joints, self.x.ndim)

        if R is None:
            R = self.R
        elif isscalar(R):
            R = np.eye(self.num_of_joints) * R

        if H is None:
            H = self.H

        # y = z - Hx
        # error (residual) between measurement and prediction
        self.y = z - dot(H, self.x)

        # common subexpression for speed
        PHT = dot(self.P, H.T)

        # S = HPH' + R
        # project system uncertainty into measurement space
        self.S = dot(H, PHT) + R
        self.SI = self.inv(self.S)
        # K = PH'inv(S)
        # map system uncertainty into kalman gain
        self.K = dot(PHT, self.SI)

        # x = x + Ky
        # predict new x with residual scaled by the kalman gain
        self.x = self.x + dot(self.K, self.y)

        # P = (I-KH)P(I-KH)' + KRK'
        # This is more numerically stable
        # and works for non-optimal K vs the equation
        # P = (I-KH)P usually seen in the literature.

        I_KH = self._I - dot(self.K, H)
        self.P = dot(dot(I_KH, self.P), I_KH.T) + dot(dot(self.K, R), self.K.T)

        # save measurement and posterior state
        self.z = deepcopy(z)
        self.x_post = self.x.copy()
        self.P_post = self.P.copy()


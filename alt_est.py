import numpy as np
import matplotlib.pyplot as plt
from math import sin
import random
tau = 0.5


def z(t):
    return 10 - 10 * np.exp( -tau * t)

def v(t):
    return -10 * -tau * np.exp( -tau * t)


def a(t):
    return -10 * -tau * -tau * np.exp(-tau * t)


def bbaro(t):
    walk_noise = 0.0
    random_noise = 0.0
    offset = 0.0
    bbaro.bias += walk_noise * (random.random() - 0.5)
    return offset + bbaro.bias + random_noise * (random.random() - 0.5)
bbaro.bias = 0


def alpha_accel(t):
    walk_noise = 0.00
    random_noise = 0.00
    offset = 0.01
    alpha_accel.alpha += walk_noise * (random.random() - 0.5)
    return offset + alpha_accel.alpha + random_noise * (random.random() - 0.5)
alpha_accel.alpha = 0


class Estimator:
    def __init__(self):
        self.alpha = np.zeros([3,1])
        self.qhat = np.zeros([4,1])
        self.qhat[0] = 1

        self.khat = np.array([[0, 0, 1]]).T

        self.g = np.array([[0, 0, 9.88065]]).T

        self.vzI = v(0)
        self.azI = a(0)

        self.bb = 0

        self.zhat = 0
        self.vcomp = 0

        self.kbI = 0.0
        self.kbP = 10.0
        self.ksP = 10.0
        self.kaI = 0.001

        self.vb_prev = 0.0
        self.v_err_prev = 0.0

        self.zb = 0.0

    def propagate(self, dt, acc, baro, sonar, truth):

        got_baro = baro > 0
        got_sonar = sonar > 0
        # Get inertial component of acceleration (rotate to body frame, remove gravity and add bias)
        # Eq 5.12
        azI_t = self.khat.T.dot(acc + self.alpha - self.g)[0,0]

        # Integrate acceleration to velocity
        # Eq 5.12
        self.vzI += dt*(self.azI + azI_t)/2.0
        self.azI = azI_t

        # figure out if sonar is valid
        # Eq 5.13
        zs = sonar

        # calculate barometer measurement
        # Eq 5.14
        zb = baro + self.bb

        # Correct estimated velocity with barometer and sonar data
        # Eq. 5.16
        if got_baro:
            vbaro = (zb - self.zb) / dt
            self.zb = zb
            vb = self.kbP * (self.zhat - self.zb) * dt

            # Integrate Accel bias
            self.alpha += self.kaI * vb
        else:
            vb = 0

        if got_sonar:
            vs = self.ksP * (zs - self.zhat)

            # Integrate barometer bias
            # Eq. 5.15
            self.bb += self.kbI * (zs - self.zb)
            self.alpha += self.kaI * vs
        else:
            vs = 0

        vcomp_t = self.vzI + vb + vs

        # Integrate Velocity
        # Eq. 5.17
        self.zhat += dt * (self.vcomp + vcomp_t) / 2.0
        self.vcomp = vcomp_t

        return self.zhat, self.alpha, self.bb

dt = 0.001

time = np.arange(0, 25, dt)

estimator = Estimator()

zhat = [0 for t in time]
bb = [0 for t in time]
alphahat = np.zeros([3, len(time)])

alpha = np.array([[0.0, 0.0, 0.0]]).T

baro_bias = [bbaro(t) for t in time]
acc_bias = [alpha_accel(t) for t in time]

for i, t in enumerate(time):
    acc = a(t)
    vel = v(t)
    alt = z(t)

    imu = np.array([[0, 0, acc]]).T + estimator.g - acc_bias[i]

    baro = 0
    if i % int(1000 / 160) == 0:
        baro = alt - baro_bias[i]

    sonar = 0
    if i % int(1000 / 40) == 0:
        sonar = alt

    estimate, estimated_alpha, estimated_bb = estimator.propagate(dt, imu, baro, sonar, z(t))
    zhat[i] = estimate
    bb[i] = estimated_bb
    alphahat[:, i] = estimated_alpha.T

plt.figure(1)
ax1 = plt.subplot(3, 1, 1)
plt.plot(time, [z(t) for t in time], label="z")
plt.plot(time, zhat, label="zhat")
plt.legend()
plt.subplot(3, 1, 2, sharex=ax1)
plt.plot(time, acc_bias, label="alpha")
plt.plot(time, alphahat[2, :], label="alphahat")
plt.legend()
plt.subplot(3, 1, 3, sharex=ax1)
plt.plot(time, baro_bias, label="b_baro")
plt.plot(time, bb, label="b_baro_hat")
plt.legend()
plt.show()

debug = 1





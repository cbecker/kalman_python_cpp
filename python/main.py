import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output
from kf import KF

plt.ion()
plt.figure()

real_x = 0.0
meas_variance = 0.8 ** 2
real_v = 0.9

kf = KF(initial_x=10.0, initial_v=0.0, accel_variance=0.1)

DT = 0.3
NUM_STEPS = 600
MEAS_EVERY_STEPS = 20

mus = []
covs = []
real_xs = []
real_vs = []
measured_values = []

"""
Plotting Data and running the system
"""
for step in range(NUM_STEPS):
    if step > 300:
        real_v *= 0.99
        
    covs.append(kf.cov)
    mus.append(kf.mean)

    real_x = real_x + DT * real_v

    kf.predict(dt=DT)
    if step != 0 and step % MEAS_EVERY_STEPS == 0:
        kf.update(meas_value=real_x + np.random.randn() * np.sqrt(meas_variance),
                  meas_variance=meas_variance)

    real_xs.append(real_x)
    real_vs.append(real_v)


"""
Plotting of the results 
"""
plt.subplot(2, 1, 1)
plt.title('Position')
plt.plot([mu[0] for mu in mus], 'r')
plt.plot(real_xs, 'b')
plt.plot([mu[0] - 2*np.sqrt(cov[0,0]) for mu, cov in zip(mus,covs)], 'r--')
plt.plot([mu[0] + 2*np.sqrt(cov[0,0]) for mu, cov in zip(mus,covs)], 'r--')


plt.subplot(2, 1, 2)
plt.title('Velocity')
plt.plot(real_vs, 'b')
plt.plot([mu[1] for mu in mus], 'r')
plt.plot([mu[1] - 2*np.sqrt(cov[1,1]) for mu, cov in zip(mus,covs)], 'r--')
plt.plot([mu[1] + 2*np.sqrt(cov[1,1]) for mu, cov in zip(mus,covs)], 'r--')

plt.show()
print("HI")
plt.ginput(1)


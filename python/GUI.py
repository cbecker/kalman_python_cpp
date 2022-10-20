import functools
import numpy as np
import pylab
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from kf import KF

"""
This function is responsible of of plotting the X and Y variable on the plot. 
"""


def draw_function(p, v, mus, covs, measured_p):
    Pxlabel = "Time in Seconds",
    Pylabel = "Position (Meters)"
    Vylabel = "Velocity (m/s)"
    figure, (ax1, ax2) = plt.subplots(2)
    plt.subplots_adjust(left=0.11, bottom=0.33)

    ax1.grid
    lines1 = ax1.plot(p,"-b")
    lines1a = ax1.plot([mu[0] for mu in mus], 'r')
    line1Cov1 = ax1.plot([mu[0] - 2 * np.sqrt(cov[0, 0]) for mu, cov in zip(mus, covs)], 'r--')
    line1Cov2 = ax1.plot([mu[0] + 2 * np.sqrt(cov[0, 0]) for mu, cov in zip(mus, covs)], 'r--')
    measuredline = ax1.plot(measured_p, 'g', label="MEASURED")
    # ax1.set_xlabel(Pxlabel)
    ax1.set_ylabel(Pylabel)
    ax1.set_ylim(-5, max(p) * 1.5)

    ax2.grid
    lines2 = ax2.plot(v, "-b", label='TRUE')
    lines2a = ax2.plot([mu[1] for mu in mus], 'r', label='PREDICTED')
    line2Cov1 = ax2.plot([mu[1] - 2 * np.sqrt(cov[1, 1]) for mu, cov in zip(mus, covs)], 'r--', label='UPPER ERROR')
    line2Cov2 = ax2.plot([mu[1] + 2 * np.sqrt(cov[1, 1]) for mu, cov in zip(mus, covs)], 'r--', label='LOWER ERROR')
    ax2.set_xlabel(Pxlabel)
    ax2.set_ylabel(Vylabel)
    ax2.set_ylim(max(v) * -15, max(v) * 15)
    plt.gcf().legend()


    return figure, ax1, lines1, lines1a, line1Cov1, line1Cov2, ax2, lines2, lines2a, line2Cov1, line2Cov2, measuredline


"""
This function is set to control what the follows are set to be. 
We will have 5 Sliders on our GUI which can help the student to see the efforts 
of variance error and initial measurements.
"""


def draw_interactive_controls(E_mea, Var, dt, init_X, init_V):
    # Placing Sliders onto the Plot GUI
    axK = plt.axes([0.1, 0.01, 0.75, 0.03])
    axB = plt.axes([0.1, 0.06, 0.75, 0.03])
    axN = plt.axes([0.1, 0.11, 0.75, 0.03])
    axX = plt.axes([0.1, 0.16, 0.75, 0.03])
    axV = plt.axes([0.1, 0.21, 0.75, 0.03])

    Nslider = Slider(axN, "E_Mea", 0.0, 4, valinit=E_mea, valfmt='%1.3f')
    Bslider = Slider(axB, "Var(A)", 0, 0.25, valinit=Var, valfmt='%1.3f')
    Kslider = Slider(axK, "dT", 0.00, 0.65, valinit=dt, valfmt='%1.3f')
    Xslider = Slider(axX, "Init_Pos.", 0, 20, valinit=init_X, valfmt='%1.3f')
    Vslider = Slider(axV, "Init_Vel.", 0.00, 2, valinit=init_V, valfmt='%1.3f')
    return Nslider, Bslider, Kslider, Xslider, Vslider

"""
This function updates the plot of the figure 
"""


def update_plot(val, line1=None, line2=None, line3=None, line4=None, measuredLine=None, ax1=None,
                     line5=None, line6=None, line7=None, line8=None, ax2=None,
                Nslider=None, Bslider=None, Kslider=None, Xslider=None, Vslider=None):
    # getting the slider values from the system
    measured_variance = Nslider.val
    accel_variance = Bslider.val
    DT = Kslider.val
    inital_x = Xslider.val
    inital_v = Vslider.val

    # print("Measured Variance:", measured_variance)
    # print("Initial X:", inital_x)
    # run the kalman filter and get the updated position and velocity
    position, velocity, mus, covs, measured_P = RunKalmanFilter(init_x=inital_x, init_v=inital_v,
                                         accel_var=accel_variance, mea_var=measured_variance,
                                         ddt=DT)

    # Now that we have the updated position and velocity
    # go ahead and redraw the plots on the graph
    # print(position)
    line1[0].set_ydata(position)  # Plotting of True value
    line2[0].set_ydata([mu[0] for mu in mus])  # Plotting Predicted Values
    line3[0].set_ydata([mu[0] - 2 * np.sqrt(cov[0, 0]) for mu, cov in zip(mus, covs)])  # Upper Uncertainty Bound
    line4[0].set_ydata([mu[0] + 2 * np.sqrt(cov[0, 0]) for mu, cov in zip(mus, covs)])  # Lower Uncertainty Bound
    measuredLine[0].set_ydata(measured_P) # Measured position
    ax1.set_ylim(-5, max(position) * 1.5)

    line5[0].set_ydata(velocity)  # Plotting of True value
    line6[0].set_ydata([mu[1] for mu in mus])  # Plotting Predicted Values
    line7[0].set_ydata([mu[1] - 2 * np.sqrt(cov[1, 1]) for mu, cov in zip(mus, covs)])  # Upper Uncertainty Bound
    line8[0].set_ydata([mu[1] + 2 * np.sqrt(cov[1, 1]) for mu, cov in zip(mus, covs)])  # Lower Uncertainty Bound
    ax2.set_ylim(max(velocity) * -15, max(velocity) * 15)
    pylab.draw()

"""
This function runs the Kalman Filter and outputs the basic it takes the inital position, inital velocity, sample time,
and variance in the measured and acceleration.
Returns: Estimated Position, Velocity, Measured Value, and Covariance
"""
def RunKalmanFilter(init_x=None, init_v=None, accel_var=None, mea_var=None, ddt=None):
    # initial values for the number of steps and Measurements for each steps
    NUM_STEPS = 600
    MEAS_EVERY_STEPS = 20
    real_x = 0.8 ** 2
    real_v = 0.1

    mus = []
    covs = []
    real_xs = []
    measured_xs = []
    real_vs = []

    kf = KF(initial_x=init_x, initial_v=init_v, accel_variance=accel_var)

    """
    Running the system 
    """
    for step in range(NUM_STEPS):
        if step > 300:
            real_v *= 0.99

        covs.append(kf.cov)
        mus.append(kf.mean)

        real_x = real_x + ddt * real_v
        measured_x = real_x + np.random.randn() * np.sqrt(mea_var)
        kf.predict(dt=DT)
        if step != 0 and step % MEAS_EVERY_STEPS == 0:
            kf.update(meas_value=measured_x,
                      meas_variance=mea_var)

        measured_xs.append(measured_x)
        real_xs.append(real_x)
        real_vs.append(real_v)

    return real_xs, real_vs, mus, covs, measured_xs


if __name__ == "__main__":
    # Values that can be controlled
    measured_variance = 0.8 ** 2
    inital_x = 9.0
    inital_v = 0.1
    accel_variance = 0.1
    DT = 0.3

    # Running the Kalman Filter
    real_xs, real_vs, mus, covs, measured_xs = RunKalmanFilter(init_x=inital_x, init_v=inital_v,
                                       accel_var=accel_variance, mea_var=measured_variance, ddt=DT)

    """
    Plotting of the results 
    
    plt.subplot(2, 1, 1)
    plt.title('Position')
    plt.plot([mu[0] for mu in mus], 'r')
    plt.plot(real_xs, 'b')
    plt.plot([mu[0] - 2 * np.sqrt(cov[0, 0]) for mu, cov in zip(mus, covs)], 'r--')
    plt.plot([mu[0] + 2 * np.sqrt(cov[0, 0]) for mu, cov in zip(mus, covs)], 'r--')

    plt.subplot(2, 1, 2)
    plt.title('Velocity')
    plt.plot(real_vs, 'b')
    plt.plot([mu[1] for mu in mus], 'r')
    plt.plot([mu[1] - 2 * np.sqrt(cov[1, 1]) for mu, cov in zip(mus, covs)], 'r--')
    plt.plot([mu[1] + 2 * np.sqrt(cov[1, 1]) for mu, cov in zip(mus, covs)], 'r--')

    plt.show()
    print("HI")
    plt.ginput(1)
    """

     # setup initial graph and control settings
    fig, ax1, Pos, PosPredicted, PosCov1, PosCov2, \
    ax2, VelMea, VelPredicted, VelCov1, VelCov2, PosMeasured = draw_function(real_xs, real_vs, mus, covs, measured_xs)

    # generated the needed sliders
    E_Measlider, E_Acelslider, DTslider, inital_xslider, inital_vslider = draw_interactive_controls(measured_variance,
                                                                                                    accel_variance, DT,
                                                                                                    inital_x, inital_v)

    # specify updating function for interactive controls
    updatefxn = functools.partial(update_plot, line1=Pos, line2= PosPredicted, line3=PosCov1, line4=PosCov2,
                                  line5=VelMea, line6=VelPredicted, line7=VelCov1, line8=VelCov2,
                                  measuredLine=PosMeasured, ax1=ax1, ax2=ax2,
                                  Nslider=E_Measlider, Bslider=E_Acelslider, Kslider=DTslider,
                                  Xslider=inital_xslider, Vslider=inital_vslider)


    #  update fxn function when the slider value gets changed
    E_Measlider.on_changed(updatefxn)
    E_Acelslider.on_changed(updatefxn)
    DTslider.on_changed(updatefxn)
    inital_xslider.on_changed(updatefxn)
    inital_vslider.on_changed(updatefxn)

    # show the gui screen
    fig.suptitle("Kalman Filter Simplified")
    pylab.show()

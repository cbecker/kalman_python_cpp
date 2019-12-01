#pragma once

#include <Eigen/Dense>

class KF
{
public:
    static const int NUM_VARS = 2;
    static const int iX = 0;
    static const int iV = 1;

    using Vector = Eigen::Matrix<double, NUM_VARS, 1>;
    using Matrix = Eigen::Matrix<double, NUM_VARS, NUM_VARS>;

    KF(double initialX, double initialV, double accelVariance) : m_accelVariance(accelVariance)
    {
        m_mean(iX) = initialX;
        m_mean(iV) = initialV;

        m_cov.setIdentity();
    }

    void predict(double dt)
    {
        Matrix F;
        F.setIdentity();
        F(iX, iV) = dt;

        const Vector newX = F * m_mean;

        Vector G;
        G(iX) = 0.5 * dt * dt;
        G(iV) = dt;

        const Matrix newP = F * m_cov * F.transpose() + G * G.transpose() * m_accelVariance;

        m_cov = newP;
        m_mean = newX;
    }

    void update(double measValue, double measVariance)
    {
        Eigen::Matrix<double, 1, NUM_VARS> H;
        H.setZero();
        H(0, iX) = 1;

        const double y = measValue - H * m_mean;
        const double S = H * m_cov * H.transpose() + measVariance;

        const Vector K = m_cov * H.transpose() * 1.0 / S;

        Vector newX = m_mean + K * y;
        Matrix newP = (Matrix::Identity() - K * H) * m_cov;

        m_cov = newP;
        m_mean = newX;
    }

    Matrix cov() const
    {
        return m_cov;
    }

    Vector mean() const
    {
        return m_mean;
    }

    double pos() const
    {
        return m_mean(iX);
    }

    double vel() const
    {
        return m_mean(iV);
    }

private:
    Vector m_mean;
    Matrix m_cov;

    const double m_accelVariance;
};

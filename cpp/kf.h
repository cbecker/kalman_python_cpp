#pragma once

#include <Eigen/Dense>

class KF
{
public:
    static const int NUM_VARS = 2;

    using Vector = Eigen::Matrix<double, NUM_VARS, 1>;
    using Matrix = Eigen::Matrix<double, NUM_VARS, NUM_VARS>;

    KF(double initialX, double initialV, double accelVariance)
    {
    }

    void predict(double dt)
    {
    }

    void update(double measValue, double measVariance)
    {
    }

    Matrix cov() const
    {
        return Matrix::Identity();
    }

    Vector mean() const
    {
        return Vector::Zero();
    }

    double pos() const
    {
        return 0.0;
    }

    double vel() const
    {
        return 0.0;
    }
};

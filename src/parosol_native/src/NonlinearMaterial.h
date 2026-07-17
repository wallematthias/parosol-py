#ifndef NONLINEARMATERIAL_H
#define NONLINEARMATERIAL_H

#include <Eigen/Core>

struct PlasticUpdate {
    Eigen::Matrix<double, 6, 1> stress;
    Eigen::Matrix<double, 6, 1> plastic_strain;
    double von_mises;
    double yield_function;
    bool yielded;
};

class VonMisesMaterial {
public:
    VonMisesMaterial(double youngs_mpa, double poisson_ratio, double yield_mpa);

    PlasticUpdate Update(
        const Eigen::Matrix<double, 6, 1>& total_strain,
        const Eigen::Matrix<double, 6, 1>& old_plastic_strain) const;

    Eigen::Matrix<double, 6, 6> ElasticMatrix() const;
    double YoungsModulus() const { return E; }
    double PoissonRatio() const { return nu; }

private:
    double E;
    double nu;
    double Y;
};

struct AsymmetricMaterialProperties {
    double E;
    double nu;
    double sigma_t;
    double sigma_c;
    double plateau;
};

class AsymmetricPerfectPlasticMaterial {
public:
    PlasticUpdate Update(
        const Eigen::Matrix<double, 6, 1>& total_strain,
        const Eigen::Matrix<double, 6, 1>& old_plastic_strain,
        const AsymmetricMaterialProperties& properties) const;

    Eigen::Matrix<double, 6, 6> ElasticMatrix(
        const AsymmetricMaterialProperties& properties) const;
};

#endif

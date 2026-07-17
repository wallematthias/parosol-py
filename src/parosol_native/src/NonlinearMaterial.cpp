#include "NonlinearMaterial.h"

#include <cmath>
#include <algorithm>
#include <Eigen/Eigenvalues>

VonMisesMaterial::VonMisesMaterial(double youngs_mpa, double poisson_ratio, double yield_mpa)
    : E(youngs_mpa), nu(poisson_ratio), Y(yield_mpa) {}

Eigen::Matrix<double, 6, 6> VonMisesMaterial::ElasticMatrix() const {
    Eigen::Matrix<double, 6, 6> D;
    D.setZero();
    const double s = nu / (1.0 - nu);
    const double t = (1.0 - 2.0 * nu) / (2.0 * (1.0 - nu));
    D(0,0) = 1.0; D(0,1) = s;   D(0,2) = s;
    D(1,0) = s;   D(1,1) = 1.0; D(1,2) = s;
    D(2,0) = s;   D(2,1) = s;   D(2,2) = 1.0;
    D(3,3) = t; D(4,4) = t; D(5,5) = t;
    D *= E * (1.0 - nu) / ((1.0 + nu) * (1.0 - 2.0 * nu));
    return D;
}

PlasticUpdate VonMisesMaterial::Update(
    const Eigen::Matrix<double, 6, 1>& total_strain,
    const Eigen::Matrix<double, 6, 1>& old_plastic_strain) const {
    const Eigen::Matrix<double, 6, 6> D = ElasticMatrix();
    Eigen::Matrix<double, 6, 1> trial_stress = D * (total_strain - old_plastic_strain);
    const double mean = (trial_stress(0) + trial_stress(1) + trial_stress(2)) / 3.0;
    Eigen::Matrix<double, 6, 1> dev = trial_stress;
    dev(0) -= mean;
    dev(1) -= mean;
    dev(2) -= mean;
    const double seq = std::sqrt(
        1.5 * (dev(0)*dev(0) + dev(1)*dev(1) + dev(2)*dev(2)
             + 2.0 * (dev(3)*dev(3) + dev(4)*dev(4) + dev(5)*dev(5))));

    PlasticUpdate out;
    out.von_mises = seq;
    out.yield_function = seq - Y;
    out.yielded = out.yield_function > 0.0;
    if (!out.yielded || seq <= 0.0) {
        out.stress = trial_stress;
        out.plastic_strain = old_plastic_strain;
        return out;
    }

    const double G = E / (2.0 * (1.0 + nu));
    const double delta_gamma = (seq - Y) / (3.0 * G);
    Eigen::Matrix<double, 6, 1> flow;
    flow.setZero();
    flow(0) = 1.5 * dev(0) / seq;
    flow(1) = 1.5 * dev(1) / seq;
    flow(2) = 1.5 * dev(2) / seq;
    flow(3) = 3.0 * dev(3) / seq;
    flow(4) = 3.0 * dev(4) / seq;
    flow(5) = 3.0 * dev(5) / seq;
    out.plastic_strain = old_plastic_strain + delta_gamma * flow;
    out.stress = D * (total_strain - out.plastic_strain);

    Eigen::Matrix<double, 6, 1> corrected_dev = out.stress;
    const double corrected_mean = (out.stress(0) + out.stress(1) + out.stress(2)) / 3.0;
    corrected_dev(0) -= corrected_mean;
    corrected_dev(1) -= corrected_mean;
    corrected_dev(2) -= corrected_mean;
    out.von_mises = std::sqrt(
        1.5 * (corrected_dev(0)*corrected_dev(0) + corrected_dev(1)*corrected_dev(1) + corrected_dev(2)*corrected_dev(2)
             + 2.0 * (corrected_dev(3)*corrected_dev(3) + corrected_dev(4)*corrected_dev(4) + corrected_dev(5)*corrected_dev(5))));
    out.yield_function = out.von_mises - Y;
    return out;
}

Eigen::Matrix<double, 6, 6> AsymmetricPerfectPlasticMaterial::ElasticMatrix(
    const AsymmetricMaterialProperties& properties) const {
    Eigen::Matrix<double, 6, 6> D;
    D.setZero();
    const double s = properties.nu / (1.0 - properties.nu);
    const double t = (1.0 - 2.0 * properties.nu) / (2.0 * (1.0 - properties.nu));
    D(0,0) = 1.0; D(0,1) = s;   D(0,2) = s;
    D(1,0) = s;   D(1,1) = 1.0; D(1,2) = s;
    D(2,0) = s;   D(2,1) = s;   D(2,2) = 1.0;
    D(3,3) = t; D(4,4) = t; D(5,5) = t;
    D *= properties.E * (1.0 - properties.nu)
        / ((1.0 + properties.nu) * (1.0 - 2.0 * properties.nu));
    return D;
}

PlasticUpdate AsymmetricPerfectPlasticMaterial::Update(
    const Eigen::Matrix<double, 6, 1>& total_strain,
    const Eigen::Matrix<double, 6, 1>& old_plastic_strain,
    const AsymmetricMaterialProperties& properties) const {
    const Eigen::Matrix<double, 6, 6> D = ElasticMatrix(properties);
    const Eigen::Matrix<double, 6, 1> trial_stress =
        D * (total_strain - old_plastic_strain);

    Eigen::Matrix3d trial_tensor;
    trial_tensor << trial_stress(0), trial_stress(3), trial_stress(4),
                    trial_stress(3), trial_stress(1), trial_stress(5),
                    trial_stress(4), trial_stress(5), trial_stress(2);
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(trial_tensor);
    Eigen::Vector3d principal = solver.eigenvalues();

    const double max_principal = principal.maxCoeff();
    const double min_principal = principal.minCoeff();
    const double tensile_yield = max_principal - properties.sigma_t;
    const double compressive_yield = -min_principal - properties.sigma_c;

    PlasticUpdate out;
    out.yield_function = std::max(tensile_yield, compressive_yield);
    out.yielded = out.yield_function > 0.0;
    if (!out.yielded) {
        out.stress = trial_stress;
        out.plastic_strain = old_plastic_strain;
    } else {
        Eigen::Vector3d capped_principal = principal;
        for (int i = 0; i < 3; ++i) {
            if (capped_principal(i) > properties.sigma_t) {
                capped_principal(i) = properties.sigma_t;
            } else if (-capped_principal(i) > properties.sigma_c) {
                capped_principal(i) = -properties.plateau;
            }
        }

        Eigen::Matrix3d capped_tensor =
            solver.eigenvectors() * capped_principal.asDiagonal() * solver.eigenvectors().transpose();
        out.stress(0) = capped_tensor(0, 0);
        out.stress(1) = capped_tensor(1, 1);
        out.stress(2) = capped_tensor(2, 2);
        out.stress(3) = capped_tensor(0, 1);
        out.stress(4) = capped_tensor(0, 2);
        out.stress(5) = capped_tensor(1, 2);
        out.plastic_strain = total_strain - D.ldlt().solve(out.stress);
    }

    const double mean = (out.stress(0) + out.stress(1) + out.stress(2)) / 3.0;
    Eigen::Matrix<double, 6, 1> dev = out.stress;
    dev(0) -= mean;
    dev(1) -= mean;
    dev(2) -= mean;
    out.von_mises = std::sqrt(
        1.5 * (dev(0)*dev(0) + dev(1)*dev(1) + dev(2)*dev(2)
             + 2.0 * (dev(3)*dev(3) + dev(4)*dev(4) + dev(5)*dev(5))));
    return out;
}

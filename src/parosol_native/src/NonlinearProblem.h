#ifndef NONLINEARPROBLEM_H
#define NONLINEARPROBLEM_H

#include "GenericMatrix.h"
#include "NonlinearMaterial.h"
#include "PCGSolver.h"
#include "Problem.h"
#include "Toolbox.h"
#include "fem.h"

#include <Eigen/Core>
#include <algorithm>
#include <cmath>
#include <memory>
#include <mpi.h>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <vector>

struct NonlinearIterationSummary {
    int plastic_iterations;
    int yielded_last;
    double plastic_convergence_last;
    std::tuple<int, double, double> final_inner_solve;
};

template <class Grid>
class NonlinearProblem {
public:
    NonlinearProblem(Grid& grid, GenericMatrix<Grid>& matrix, const VonMisesMaterial& material)
        : _grid(grid), _matrix(matrix), _material_mode(MaterialMode::VonMises),
          _von_mises_material(new VonMisesMaterial(material)),
          _plastic_strain(grid.GetNrElem() * 6) {
        InitializePlasticState();
    }

    NonlinearProblem(
        Grid& grid,
        GenericMatrix<Grid>& matrix,
        const double* youngs_mpa,
        const double* poisson_ratio,
        const double* sigma_c_mpa,
        const double* sigma_t_mpa,
        const double* plateau_mpa,
        const unsigned short* material_id)
        : _grid(grid), _matrix(matrix), _material_mode(MaterialMode::AsymmetricMap),
          _asymmetric_material(new AsymmetricPerfectPlasticMaterial()),
          _asymmetric_properties(BuildAsymmetricProperties(
              grid,
              youngs_mpa,
              poisson_ratio,
              sigma_c_mpa,
              sigma_t_mpa,
              plateau_mpa,
              material_id)),
          _plastic_strain(grid.GetNrElem() * 6) {
        InitializePlasticState();
    }

    void InitializePlasticState() {
        _plastic_strain.setZero();
        _plastic_gauss.resize(_grid.GetNrElem() * 8);
        for (size_t i = 0; i < _plastic_gauss.size(); ++i) {
            _plastic_gauss[i].setZero();
        }
    }

    Eigen::VectorXd BuildPlasticRHS() {
        const int dimension = 3;
        const int material_properties = 2;
        const int nodes_per_element = 8;
        const int dofs_per_element = 24;
        const int gauss_points = 8;
        const int stress_strain_size = 6;

        Eigen::VectorXd rhs(_matrix.GetNrDofs());
        rhs.setZero();
        _grid.Recv_export_Ghost();

        double grid_dimensions[3];
        _grid.GetRes(grid_dimensions);
        double coordinates[dimension * nodes_per_element];
        setcoord(grid_dimensions, coordinates);

        // Match GenericMatrix: local stiffness uses E=1000 and element weights
        // carry the image modulus scaling.
        double material[material_properties] = {1000.0, 0.3};
        Eigen::Matrix<double, dofs_per_element, 1> element_rhs;
        std::vector<double> plastic_fem_order(stress_strain_size * gauss_points);
        t_index element_index = 0;

        for (_grid.initIterateOverElements(); _grid.TestIterateOverElements(); _grid.IncIterateOverElements()) {
            material[1] = ElementPoissonRatio(element_index);
            for (int gauss_point = 0; gauss_point < gauss_points; ++gauss_point) {
                const Eigen::Matrix<double, 6, 1>& plastic =
                    _plastic_gauss[static_cast<size_t>(element_index) * gauss_points + gauss_point];
                const int offset = gauss_point * stress_strain_size;
                plastic_fem_order[offset + 0] = plastic(0);
                plastic_fem_order[offset + 1] = plastic(1);
                plastic_fem_order[offset + 2] = plastic(2);
                plastic_fem_order[offset + 3] = plastic(3);
                plastic_fem_order[offset + 4] = plastic(5);
                plastic_fem_order[offset + 5] = plastic(4);
            }

            Initial_Strain_Load(
                material, material_properties,
                nodes_per_element, dofs_per_element,
                dimension, gauss_points, stress_strain_size,
                coordinates, plastic_fem_order.data(), element_rhs.data());
            _grid.SearchIndexes();
            _grid.SumInToNodalDisplacementsOfElement(rhs, element_rhs, _grid.GetElementWeight());
            ++element_index;
        }

        _grid.Send_export_Ghost(rhs);
        _grid.WaitAndCopy_export_Ghost(rhs);
        return rhs;
    }

    NonlinearIterationSummary Solve(
        Problem<Grid>& problem,
        PCGSolver& solver,
        int maximum_plastic_iterations,
        double plastic_tolerance) {
        NonlinearIterationSummary summary = {0, 0, 0.0, std::make_tuple(0, 0.0, 0.0)};
        for (int iteration = 1; iteration <= maximum_plastic_iterations; ++iteration) {
            Eigen::VectorXd plastic_rhs = BuildPlasticRHS();
            problem.Impose(0);
            problem.AddToRHS(plastic_rhs);
            problem.SetSolver(solver);
            summary.final_inner_solve = problem.Solve(0, 0);

            summary.plastic_iterations = iteration;
            UpdatePlasticState(
                problem.GetSol(),
                summary.yielded_last,
                summary.plastic_convergence_last);
            if (summary.plastic_convergence_last <= plastic_tolerance) {
                break;
            }
        }
        return summary;
    }

    const Eigen::VectorXd& PlasticStrain() const {
        return _plastic_strain;
    }

    const std::vector<Eigen::Matrix<double, 6, 1> >& PlasticGauss() const {
        return _plastic_gauss;
    }

private:
    enum class MaterialMode {
        VonMises,
        AsymmetricMap
    };

    static std::vector<AsymmetricMaterialProperties> BuildAsymmetricProperties(
        Grid& grid,
        const double* youngs_mpa,
        const double* poisson_ratio,
        const double* sigma_c_mpa,
        const double* sigma_t_mpa,
        const double* plateau_mpa,
        const unsigned short* material_id) {
        if (youngs_mpa == 0 || poisson_ratio == 0 || sigma_c_mpa == 0
            || sigma_t_mpa == 0 || plateau_mpa == 0 || material_id == 0) {
            throw std::runtime_error("missing asymmetric nonlinear material map data");
        }

        const long ldim_x = grid.ldim[0];
        const long ldim_y = grid.ldim[1];
        const long slice = ldim_x * ldim_y;

        std::vector<AsymmetricMaterialProperties> properties;
        properties.reserve(grid.GetNrElem());
        for (grid.initIterateOverElements(); grid.TestIterateOverElements(); grid.IncIterateOverElements()) {
            t_coord x = 0;
            t_coord y = 0;
            t_coord z = 0;
            DecodeMortonKey(grid._GridIterator->key, x, y, z);
            const long local_x = static_cast<long>(x - grid.corner[0]);
            const long local_y = static_cast<long>(y - grid.corner[1]);
            const long local_z = static_cast<long>(z - grid.corner[2]);
            const long dense_index = local_z * slice + local_y * ldim_x + local_x;
            AsymmetricMaterialProperties element_properties;
            (void) material_id[dense_index];
            element_properties.E = youngs_mpa[dense_index];
            element_properties.nu = poisson_ratio[dense_index];
            element_properties.sigma_c = sigma_c_mpa[dense_index];
            element_properties.sigma_t = sigma_t_mpa[dense_index];
            element_properties.plateau = plateau_mpa[dense_index];
            element_properties.plasticity_enabled = material_id[dense_index] == 1;
            properties.push_back(element_properties);
        }
        if (properties.size() != static_cast<size_t>(grid.GetNrElem())) {
            throw std::runtime_error("asymmetric material map active element count mismatch");
        }
        return properties;
    }

    static void DecodeMortonKey(t_octree_key key, t_coord& x, t_coord& y, t_coord& z) {
        x = 0;
        y = 0;
        z = 0;
        for (int bit = 0; bit < 16; ++bit) {
            x |= static_cast<t_coord>(((key >> (3 * bit)) & 1) << bit);
            y |= static_cast<t_coord>(((key >> (3 * bit + 1)) & 1) << bit);
            z |= static_cast<t_coord>(((key >> (3 * bit + 2)) & 1) << bit);
        }
    }

    double ElementPoissonRatio(t_index element_index) const {
        if (_material_mode == MaterialMode::VonMises) {
            return _von_mises_material->PoissonRatio();
        }
        return _asymmetric_properties[static_cast<size_t>(element_index)].nu;
    }

    PlasticUpdate UpdateMaterialPoint(
        t_index element_index,
        const Eigen::Matrix<double, 6, 1>& total_strain,
        const Eigen::Matrix<double, 6, 1>& old_plastic) const {
        if (_material_mode == MaterialMode::VonMises) {
            return _von_mises_material->Update(total_strain, old_plastic);
        }
        const AsymmetricMaterialProperties& properties =
            _asymmetric_properties[static_cast<size_t>(element_index)];
        if (!properties.plasticity_enabled) {
            PlasticUpdate update;
            const Eigen::Matrix<double, 6, 6> D =
                _asymmetric_material->ElasticMatrix(properties);
            update.stress = D * total_strain;
            update.plastic_strain = old_plastic;
            const double mean =
                (update.stress(0) + update.stress(1) + update.stress(2)) / 3.0;
            Eigen::Matrix<double, 6, 1> dev = update.stress;
            dev(0) -= mean;
            dev(1) -= mean;
            dev(2) -= mean;
            update.von_mises = std::sqrt(
                1.5 * (dev(0)*dev(0) + dev(1)*dev(1) + dev(2)*dev(2)
                     + 2.0 * (dev(3)*dev(3) + dev(4)*dev(4) + dev(5)*dev(5))));
            update.yield_function = 0.0;
            update.yielded = false;
            return update;
        }
        return _asymmetric_material->Update(total_strain, old_plastic, properties);
    }

    void UpdatePlasticState(
        Eigen::VectorXd& displacement,
        int& yielded_last,
        double& plastic_convergence_last) {
        const int dimension = 3;
        const int material_properties = 2;
        const int nodes_per_element = 8;
        const int dofs_per_element = 24;
        const int gauss_points = 8;
        const int stress_strain_size = 6;

        double grid_dimensions[3];
        _grid.GetRes(grid_dimensions);
        double coordinates[dimension * nodes_per_element];
        setcoord(grid_dimensions, coordinates);

        // Element_Stress needs material data to populate its stress workspace,
        // although this update consumes only the kinematic strain values.
        double material[material_properties] = {1000.0, 0.3};

        Eigen::Matrix<double, 24, 1> element_displacements;
        std::vector<double> strain((stress_strain_size + 1) * gauss_points);
        std::vector<double> stress((stress_strain_size + 1) * gauss_points);
        double local_max_change = 0.0;
        int local_yielded = 0;
        t_index element_index = 0;

        _grid.Recv_import_Ghost(displacement);
        _grid.Send_import_Ghost(displacement);
        _grid.Wait_import_Ghost();

        for (_grid.initIterateOverElements(); _grid.TestIterateOverElements(); _grid.IncIterateOverElements()) {
            material[1] = ElementPoissonRatio(element_index);
            _grid.GetNodalDisplacementsOfElement(displacement, element_displacements);
            Element_Stress(
                material, material_properties,
                nodes_per_element, dofs_per_element,
                dimension, gauss_points, stress_strain_size,
                coordinates, element_displacements.data(),
                strain.data(), stress.data(),
                0, 0, 0, 0,
                0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0,
                0, 0,
                0, 0, 0,
                0, 0, 0,
                false, false, false, false);

            Eigen::Matrix<double, 6, 1> averaged_plastic;
            averaged_plastic.setZero();
            bool element_yielded = false;

            for (int gauss_point = 0; gauss_point < gauss_points; ++gauss_point) {
                const int offset = gauss_point * (stress_strain_size + 1);
                Eigen::Matrix<double, 6, 1> total_strain;
                total_strain(0) = strain[offset + 0];
                total_strain(1) = strain[offset + 1];
                total_strain(2) = strain[offset + 2];
                total_strain(3) = strain[offset + 3];
                total_strain(4) = strain[offset + 5];
                total_strain(5) = strain[offset + 4];

                const size_t state_index =
                    static_cast<size_t>(element_index) * gauss_points + gauss_point;
                const Eigen::Matrix<double, 6, 1> old_plastic = _plastic_gauss[state_index];
                const PlasticUpdate update = UpdateMaterialPoint(
                    element_index,
                    total_strain,
                    old_plastic);
                _plastic_gauss[state_index] = update.plastic_strain;
                averaged_plastic += update.plastic_strain;
                element_yielded = element_yielded || update.yielded;
                for (int component = 0; component < 6; ++component) {
                    local_max_change = std::max(
                        local_max_change,
                        std::abs(update.plastic_strain(component) - old_plastic(component)));
                }
            }
            averaged_plastic /= static_cast<double>(gauss_points);

            for (int component = 0; component < 6; ++component) {
                _plastic_strain[element_index * 6 + component] = averaged_plastic(component);
            }
            if (element_yielded) {
                ++local_yielded;
            }
            ++element_index;
        }

        MPI_Allreduce(&local_yielded, &yielded_last, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(
            &local_max_change,
            &plastic_convergence_last,
            1,
            MPI_DOUBLE,
            MPI_MAX,
            MPI_COMM_WORLD);
    }

    Grid& _grid;
    GenericMatrix<Grid>& _matrix;
    MaterialMode _material_mode;
    std::unique_ptr<VonMisesMaterial> _von_mises_material;
    std::unique_ptr<AsymmetricPerfectPlasticMaterial> _asymmetric_material;
    std::vector<AsymmetricMaterialProperties> _asymmetric_properties;
    Eigen::VectorXd _plastic_strain;
    std::vector<Eigen::Matrix<double, 6, 1> > _plastic_gauss;
};

#endif

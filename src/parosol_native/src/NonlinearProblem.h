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
#include <mpi.h>
#include <tuple>
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
        : _grid(grid), _matrix(matrix), _material(material),
          _plastic_strain(grid.GetNrElem() * 6) {
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
        double material[material_properties] = {1000.0, _material.PoissonRatio()};
        Eigen::Matrix<double, dofs_per_element, 1> element_rhs;
        std::vector<double> plastic_fem_order(stress_strain_size * gauss_points);
        t_index element_index = 0;

        for (_grid.initIterateOverElements(); _grid.TestIterateOverElements(); _grid.IncIterateOverElements()) {
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
                const PlasticUpdate update = _material.Update(total_strain, old_plastic);
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
    VonMisesMaterial _material;
    Eigen::VectorXd _plastic_strain;
    std::vector<Eigen::Matrix<double, 6, 1> > _plastic_gauss;
};

#endif

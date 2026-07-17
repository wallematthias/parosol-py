#ifndef NONLINEARPROBLEM_H
#define NONLINEARPROBLEM_H

#include "GenericMatrix.h"
#include "NonlinearMaterial.h"
#include "Problem.h"

#include <Eigen/Core>
#include <vector>

template <class Grid>
class NonlinearProblem {
public:
    NonlinearProblem(Grid& grid, GenericMatrix<Grid>& matrix, const VonMisesMaterial& material)
        : _grid(grid), _matrix(matrix), _material(material) {}

    Eigen::VectorXd BuildPlasticRHS() const {
        Eigen::VectorXd rhs(_matrix.GetNrDofs());
        rhs.setZero();
        return rhs;
    }

private:
    Grid& _grid;
    GenericMatrix<Grid>& _matrix;
    VonMisesMaterial _material;
};

#endif

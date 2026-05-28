/*
 * ParOSol: a parallel FE solver for trabecular bone modeling
 * Copyright (C) 2011, Cyril Flaig
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation; either version 2
 * of the License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#ifndef PROBLEM_H
#define PROBLEM_H

#include "GenericMatrix.h"
#include "BoundaryCondition.h"
#include "Solver.h"

//! This class holds the linear system

/*! Problem stores the Matrix, RHS and LHS. It also computes the the
 *  LHS out of the boundary conditions
 */

template <class Grid>
class Problem
{
  public:
	//!Problem constructor
    Problem(GenericMatrix<Grid> &mat, std::string file, const char *sv_format) : 
        _ldofs(mat.GetNrDofs()), 
        _mat(mat), 
        _bcond(mat.GetBC()), 
        _file(file),
        _startvectorSet(false)
	{
		//allocate the LHS and RHS. Value are set through the BC
		_x = new Eigen::VectorXd(_ldofs);
        _x0 = new Eigen::VectorXd(_ldofs);
        _b = new Eigen::VectorXd(_ldofs);
        
        // Startvector filename
        MPI_Comm_rank(MPI_COMM_WORLD, &MyPID);
        sprintf(_startVectorFile, sv_format, _file.c_str(), MyPID);
    }
    std::string _file;
    char _startVectorFile[256];
    bool _startvectorSet;

	//! Problem destructor
	~Problem()
	{
		delete _x;
        delete _x0;
		delete _b;
	}

	//! Impose boundary condition
	int Impose(int startvector_flag = 1);

	/** Set the solver
		 * @param s Implemented Solver
		 */
	void SetSolver(PCGSolver &s)
	{
        // Because zero-startvecotr need to be "injected",
        // only PCGSolver can be used, not generic Solver
        solver = &s;
	}

	//! Start the solver
	std::tuple<int, double, double> Solve(int startvector_flag = 0, int extrapolation = 0)
	{
        // Set zero-startvector
        if(startvector_flag && _startvectorSet)
            solver->SetX0(*_x0);

        // Solve
        Eigen::VectorXd tmp = *_b; // The solver may change b
        std::tuple<int, double, double> res = solver->Solve_res(tmp, *_x);

        // Store startvector
        MPI_File myfile; 
        if(startvector_flag && extrapolation && MPI_File_open(MPI_COMM_SELF, _startVectorFile, MPI_MODE_RDONLY, MPI_INFO_NULL, &myfile) == 0)
        {
            // Read last startvector (Separate file for each thread)
            MPI_File_read(myfile, _x0->data(), _ldofs, MPI_DOUBLE, MPI_STATUS_IGNORE);
            MPI_File_close(&myfile);

            // Arbenz: Extrapolate new startvector
            // - Linear:    x0_i+1 = 2x_i - x_i-1
            // - Quadratic: x0_i+1 = 3x_i - 3x_i-1 + x_i-2
            *_x0 = 2*(*_x) - *_x0;

            // Store startvector to file (Separate file for each thread)
            MPI_File_open(MPI_COMM_SELF, _startVectorFile, MPI_MODE_WRONLY | MPI_MODE_CREATE, MPI_INFO_NULL, &myfile);
            MPI_File_set_size(myfile, 0);
            MPI_File_write(myfile, _x0->data(), _ldofs, MPI_DOUBLE, MPI_STATUS_IGNORE); 
            MPI_File_close(&myfile);

            PCOUT(MyPID, "Startvector: Linear extrapolation" << std::endl)
            PCOUT(MyPID, "Startvector stored (" << _startVectorFile << ")" << std::endl);
        }
        else if(startvector_flag)
        {
            // Store startvector to file (Separate file for each thread)
            MPI_File_open(MPI_COMM_SELF, _startVectorFile, MPI_MODE_WRONLY | MPI_MODE_CREATE, MPI_INFO_NULL, &myfile); 
            MPI_File_set_size(myfile, 0);
            MPI_File_write(myfile, _x->data(), _ldofs, MPI_DOUBLE, MPI_STATUS_IGNORE); 
            MPI_File_close(&myfile);

            PCOUT(MyPID, "Startvector stored (" << _startVectorFile << ")" << std::endl);
        }        

		return res;
	}

	/** Gets the solution
		 * @return a refence to the solution vector
		 */
	Eigen::VectorXd &GetSol()
	{
		return *_x;
	}

	/** Gets the residual vector
		 * @return a refence to a the residual vector
		 */
	Eigen::VectorXd &GetRes()
	{
		Eigen::VectorXd *tmp = new Eigen::VectorXd(*_x);
		_mat.Apply(*_x, *tmp);
		*tmp = *_b - *tmp;
		return *tmp;
	}

	/** computes the true residuum
		 * @return the residuum
		 */
	double Res()
	{
		Eigen::VectorXd tmp = *_x;
		_mat.Apply(*_x, tmp);
		tmp = *_b - tmp;
		return sqrt(_mat.dot(tmp, tmp));
	}

	/** @name Some Debug function */
	//@{
	/** Compares a vector with the solution
		 * @param x vector that is compared to the solution of the problem
		 * @return the norm of the difference
		 */
	double CompareSol(Eigen::VectorXd &x)
	{
		Eigen::VectorXd tmp = x - *_x;
		return sqrt(_mat.dot(tmp, tmp));
    }
    
	//! Print solution vector
	void PrintSol()
	{
		std::cout << "\nSolution\n";
		_mat.GetGrid().PrintVector(*_x);
	}

	//! Print the RHS
	void PrintRHS()
	{
		std::cout << "\n RHS \n";
		_mat.GetGrid().PrintVector(*_b);
	} //@}

    long _ldofs;

  private:
	//! Helper function which set BC on the RHS
	int SetBoundaryConditions(Eigen::VectorXd &rhs);
	GenericMatrix<Grid> &_mat;
	BoundaryCondition &_bcond;
	Eigen::VectorXd *_x, *_x0, *_b;
    PCGSolver *solver;
    int MyPID;
};

template <class Grid>
int Problem<Grid>::Impose(int startvector_flag)
{
    //renaming
    Eigen::VectorXd &b = *_b;
    Eigen::VectorXd &x = *_x;
    Eigen::VectorXd &x0 = *_x0;
	  
	//Store Loads in B
	//At the moment none load supported
	b.setZero(_ldofs);
    x.setZero(_ldofs);

	//apply fixed nodes on Matrix
	long num_ind = _bcond.FixedNodes_Ind.size();

	//write displacement in x
	//corr = A*fixed_disp -> load which is cause by the fixed nodes
	for (long i = 0; i < num_ind; ++i)
	{
		x[_bcond.FixedNodes_Ind[i]] = _bcond.FixedNodes[i];
	}

	//b = b - A*fixed
	_mat.Apply_NoResetBoundaries(x, b);
	b = -1 * b;

	//add the loaded nodes
	num_ind = _bcond.LoadedNodes_Ind.size();
	for (long i = 0; i < num_ind; ++i)
	{
		b[_bcond.LoadedNodes_Ind[i]] += _bcond.LoadedNodes[i];
	}

	SetBoundaryConditions(b);
	x.setZero(_ldofs);
    x0.setZero(_ldofs);

    // Read startvector (Separate file per thread)
    MPI_File myfile; 
    if(startvector_flag && MPI_File_open(MPI_COMM_SELF, _startVectorFile, MPI_MODE_RDONLY, MPI_INFO_NULL, &myfile) == 0)
    {
        MPI_File_read(myfile, x.data(), _ldofs, MPI_DOUBLE, MPI_STATUS_IGNORE); 
        MPI_File_close(&myfile);
        _startvectorSet = true;
    }
     
    // Boundary conditions
	num_ind = _bcond.FixedNodes_Ind.size();
	for (long i = 0; i < num_ind; ++i)
    {
        x[_bcond.FixedNodes_Ind[i]] = _bcond.FixedNodes[i];
        x0[_bcond.FixedNodes_Ind[i]] = _bcond.FixedNodes[i];
    }
        
	return 0;
}

template <class Grid>
int Problem<Grid>::SetBoundaryConditions(Eigen::VectorXd &rhs)
{
	long num_ind = _bcond.FixedNodes_Ind.size();

	// write the disp in the vector
	for (long i = 0; i < num_ind; ++i)
		rhs[_bcond.FixedNodes_Ind[i]] = _bcond.FixedNodes[i];
	return 0;
}

#endif /* PROBLEM_H */

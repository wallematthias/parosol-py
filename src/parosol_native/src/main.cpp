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

#include <iostream>
#include <iomanip>
#include <memory>

#include <mpi.h>
#include <Eigen/Core>
#include <Eigen/Eigenvalues>

#include "Timing.h"
#include "Config.h"
#include "NonlinearMaterial.h"
#include "NonlinearProblem.h"

//is chosen by the templates
#include "KeyGenerator.h"

#include "HDF5Image.h"
#include "OctreeGrid.h"
#include "GenericMatrix.h"
#include "PCGSolver.h"
#include "Problem.h"
#include "MlCycle.h"

#include "VTKPrinter.h"
#include "HDF5Printer.h"

#define EXIT(X)     \
	MPI_Finalize(); \
	exit(X);

/*
//measure mflops with a matrix
template <class T>
void mflops(GenericMatrix<T> &matr, int MyPID);
*/

template <class T>
void print(GenericMatrix<T> &matr, Problem<T> &problem, std::string file, int MyPID, int SED_flag, int EFF_flag, int VonMises_flag, int e_dev_flag, int e_vol_flag, int strain_flag, int stress_flag, int DP_s_flag, int DP_e_flag, const Eigen::VectorXd* plastic_strain, const std::vector<Eigen::Matrix<double, 6, 1> >* plastic_gauss, const NonlinearIterationSummary* nonlinear_summary);

static void nonlinear_material_compile_check() {
    VonMisesMaterial material(1000.0, 0.3, 25.0);
    Eigen::Matrix<double, 6, 1> strain;
    strain.setZero();
    Eigen::Matrix<double, 6, 1> plastic;
    plastic.setZero();
    (void) material.Update(strain, plastic);
}

int main(int argc, char *argv[])
{
    int MyPID, psize;
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &MyPID);
	MPI_Comm_size(MPI_COMM_WORLD, &psize);

	if (argc < 2)
	{
        std::string usage = std::string("usage: ") + argv[0]
            + std::string(" [--level arg (6)]")
            + std::string(" [--tol arg (1e-6)]")
            + std::string(" [--out_args (e.g. --SED, --strain,...)]")
            + std::string(" [--startvector --extrapolation]")
            + std::string(" filename\n");

		PCOUT(MyPID, usage)
		MPI_Finalize();
		return -1;
    }
    
    std::string file;
	int level = 6, degree = 10;
	double tol = 1e-6;

    int SED_flag = 0, EFF_flag = 0, VonMises_flag = 0, e_dev_flag = 0, e_vol_flag = 0, strain_flag = 0, stress_flag = 0, DP_s_flag = 0, DP_e_flag = 0, startvector_flag = 0, extrapolation_flag = 0;
    
	std::string param;
	for (int i = 1; i < argc; i++)
	{
		param = std::string(argv[i]);
		if (param.compare("--level") == 0)
		{
			i++;
			level = atoi(argv[i]);
		}
		else if (param.compare("--tol") == 0)
		{
			i++;
			tol = atof(argv[i]);
        }
		else if (param.compare("--SED") == 0)
		{
			SED_flag = 1;
		}
		else if (param.compare("--EFF") == 0)
		{
			EFF_flag = 1;
		}
		else if (param.compare("--VonMises") == 0)
		{
			VonMises_flag = 1;
		}
		else if (param.compare("--e_dev") == 0)
		{
			e_dev_flag = 1;
		}
		else if (param.compare("--e_vol") == 0)
		{
			e_vol_flag = 1;
		}
		else if (param.compare("--strain") == 0)
		{
			strain_flag = 1;
		}
		else if (param.compare("--stress") == 0)
		{
			stress_flag = 1;
		}
		else if (param.compare("--DP_s") == 0)
		{
			DP_s_flag = 1;
		}
		else if (param.compare("--DP_e") == 0)
		{
			DP_e_flag = 1;
        }
        else if (param.compare("--startvector") == 0)
		{
			startvector_flag = 1;
        }
        else if (param.compare("--extrapolation") == 0)
		{
			extrapolation_flag = 1;
        }
		else
		{
			file = param;
		}
	}

	if ((SED_flag == 0) && (EFF_flag == 0) && (VonMises_flag == 0) && (e_dev_flag == 0) && (e_vol_flag == 0) && (strain_flag == 0) && (stress_flag == 0) && (DP_s_flag == 0) && (DP_e_flag == 0))
	{
		SED_flag = 1;
		EFF_flag = 1;
		VonMises_flag = 1;
	}

	PCOUT(MyPID, "Parameters:\n   file: " << file << std::endl << "   tolerance: " << tol << std::endl << "   max. num. level: " << level << std::endl);
	if (file.compare("") == 0)
	{
		PCOUT(MyPID, "No file given\n");
		MPI_Finalize();
		return -10;
	}

    // Timing
	Timer timer(MPI_COMM_WORLD);
    timer.Start("Overall");
    timer.Start("Constant");
	t_timing elapsed_time;

	// Read in the Problem
    timer.Start("Read");
    CPULayout layout;
	HDF5Image ir(file, layout);
	typedef OctreeGrid<OctreeKey_Lookup> t_Ogrid;
	t_Ogrid grid(ir);
	timer.Stop("Read");
	PCOUT(MyPID, "Nr. of global elements: " << grid.GetNrElemGlobal() << " global nodes: " << grid.GetNrNodesGlobal() << "\n");

	// Generate the BC
	timer.Start("BC");
	grid.GenerateBC();
	timer.Stop("BC");
	PCOUT(MyPID, grid);

	// Construct the matrix
	timer.Start("Mat");
	GenericMatrix<t_Ogrid> matr(grid);
	timer.Stop("Mat");
	PCOUT(MyPID, matr);

    std::cout << std::setprecision(7);
	MPI_Barrier(MPI_COMM_WORLD);

	// Construct the preconditioner
	timer.Start("Prec");
	MlCycle<OctreeKey_Lookup> prec(grid, matr, degree, level - 1, 0, 16, 0);
	PCOUT(MyPID, "Preconditioning done\n");
	timer.Stop("Prec");

	// Setting up the problem
	timer.Start("Setup");
    Problem<t_Ogrid> problem(matr, file, "%s.sv_%d");                       // Startvector naming scheme: [filename].sv_[thread_id]
    if (problem.Impose(startvector_flag))                                   // READ STARTVECTOR  
    {                
        std::cout << "Error in impose\n";
    }
    if (problem._startvectorSet)
    {
        PCOUT(MyPID, "Startvector: Startvector read (" << problem._startVectorFile << ")" << std::endl);
    }
    PCGSolver solver(matr, prec, tol, 1000, true);
    problem.SetSolver(solver);
	timer.Stop("Setup");

    // Solving the system
    timer.Stop("Constant");
	timer.Start("Solve");
	std::tuple<int, double, double> output;
    std::unique_ptr<NonlinearProblem<t_Ogrid> > nonlinear_problem;
    NonlinearIterationSummary nonlinear_summary = {
        0, 0, 0.0, std::make_tuple(0, 0.0, 0.0)};
    if (ir.nonlinear_enabled) {
        if (ir.nonlinear_material_type != "VonMisesIsotropic") {
            PCOUT(MyPID, "ERROR: only VonMisesIsotropic nonlinear material is currently supported\n");
            MPI_Finalize();
            return 2;
        }
        VonMisesMaterial material(ir.nonlinear_E_mpa, ir.nonlinear_nu, ir.nonlinear_Y_mpa);
        nonlinear_problem.reset(new NonlinearProblem<t_Ogrid>(grid, matr, material));
        nonlinear_summary = nonlinear_problem->Solve(
            problem,
            solver,
            ir.nonlinear_maximum_plastic_iterations,
            ir.nonlinear_convergence_tolerance);
        output = nonlinear_summary.final_inner_solve;
    } else {
        output = problem.Solve(startvector_flag, extrapolation_flag);   // STORE STARTVECTOR
    }
	int iterations = std::get<0>(output);
	double rel_res = std::get<1>(output);
	double abs_res = std::get<2>(output);
    timer.Stop("Solve");
    timer.Restart("Constant");

    // Print out the solution / Write to file
    timer.Start("Print");
	print(matr, problem, file, MyPID, SED_flag, EFF_flag, VonMises_flag, e_dev_flag, e_vol_flag, strain_flag, stress_flag, DP_s_flag, DP_e_flag,
        nonlinear_problem ? &nonlinear_problem->PlasticStrain() : 0,
        nonlinear_problem ? &nonlinear_problem->PlasticGauss() : 0,
        nonlinear_problem ? &nonlinear_summary : 0);
    timer.Stop("Print");

    // Timings
    timer.Stop("Constant");
    timer.Stop("Overall");
    PCOUT(MyPID, "### Timing, averages [s]\n");
    PCOUT(MyPID, "#  Nr of It: " << iterations << "\n");
    PCOUT(MyPID, "#  Relative residuum: " << rel_res << "\n");
    PCOUT(MyPID, "#  Absolute residuum: " << abs_res << "\n");
    PCOUT(MyPID, "#  MPI Size: " << psize << "\n");
	elapsed_time = timer.ElapsedTime("Overall");
    PCOUT(MyPID, "#  Overall:  " << elapsed_time.avg << "\n");
    t_timing constant_time = timer.ElapsedTime("Constant");
    PCOUT(MyPID, "#  Constant: " << constant_time.avg << "\n");
    t_timing solving_time = timer.ElapsedTime("Solve");
    PCOUT(MyPID, "#  Solving:  " << solving_time.avg << "\n");
    PCOUT(MyPID, "#  Solve/It: " << solving_time.avg/iterations << "\n");

    // Additional timings
    /*
    elapsed_time = timer.ElapsedTime("Read");
    PCOUT(MyPID, "Time for Read in: " << COUTTIME(elapsed_time.avg) << "s\n");
    elapsed_time = timer.ElapsedTime("BC");
    PCOUT(MyPID, "Time for Construction of Boundary Condition: " << COUTTIME(elapsed_time.avg) << "\n");
    elapsed_time = timer.ElapsedTime("Mat");
	PCOUT(MyPID, "Time for Construction of Matrix: " << COUTTIME(elapsed_time.avg) << "\n");
    elapsed_time = timer.ElapsedTime("Prec");
    PCOUT(MyPID, "Time for Building Preconditioner: " << COUTTIME(elapsed_time.avg) << "\n");
    elapsed_time = timer.ElapsedTime("Setup");
    PCOUT(MyPID, "Time for Setting up the Problem: " << COUTTIME(elapsed_time.avg) << "\n");
    */
    
    //mflops(matr, MyPID);
    
	MPI_Finalize();
	
	return 0;
}

template <class T>
void print(GenericMatrix<T> &matr, Problem<T> &problem, std::string file, int MyPID, int SED_flag, int EFF_flag, int VonMises_flag, int e_dev_flag, int e_vol_flag, int strain_flag, int stress_flag, int DP_s_flag, int DP_e_flag, const Eigen::VectorXd* plastic_strain, const std::vector<Eigen::Matrix<double, 6, 1> >* plastic_gauss, const NonlinearIterationSummary* nonlinear_summary)
{
	//VTKPrinter<OctreeKey_Lookup> print("out", matr.GetGrid());
	HDF5Printer<OctreeKey_Lookup> print(file, matr.GetGrid());
	//matr.PrintTimings();
	int dofs = matr.GetGrid().GetNrDofs();
	Eigen::VectorXd force(dofs);
	matr.Apply_NoResetBoundaries(problem.GetSol(), force);
	print.PrintAll(problem.GetSol(), force, problem.GetRes(), SED_flag, EFF_flag, VonMises_flag, e_dev_flag, e_vol_flag, strain_flag, stress_flag, DP_s_flag, DP_e_flag);
	if (plastic_strain != 0 && plastic_gauss != 0 && nonlinear_summary != 0) {
		print.PrintPlasticStrain(*plastic_strain);
		print.PrintGaussPointPlasticStrain(*plastic_gauss);
		print.PrintNonlinearResults(
			nonlinear_summary->plastic_iterations,
			nonlinear_summary->yielded_last,
			nonlinear_summary->plastic_convergence_last);
	}
}

/*
template <class T>
void mflops(GenericMatrix<T> &matr, int MyPID) {
	Timer timer(MPI_COMM_WORLD);
	int dofs = matr.GetGrid().GetNrDofs();
    Eigen::VectorXd a,b;
	a.setRandom(dofs);
	b.setZero(dofs);
	t_octree_key globalelem = matr.GetGrid().GetNrElemGlobal();
	int size = matr.GetGrid().GetNrCPU();
	int loops = 16e6/globalelem*size+1;
	double flop = globalelem*(24*(24+24)+48)*loops; //+48 the factor

	PCOUT(MyPID, "Meassure with " << loops << "loops" << std::endl);
	MPI_Barrier(MPI_COMM_WORLD);
	timer.Start("gflop");
	for(int i = 0; i <loops; i++)
		matr.Apply_NoResetBoundaries(a, b);
	timer.Stop("gflop");
	t_timing ela = timer.ElapsedTime("gflop");
	double elaps = ela.max;
	PCOUT(MyPID, "time needed: " << elaps << std::endl)
	PCOUT(MyPID, "MFlop/s: " << flop/elaps/1e6<< std::endl)
}
*/

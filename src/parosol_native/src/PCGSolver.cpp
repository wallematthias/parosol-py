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

#include "PCGSolver.h"

PCGSolver::PCGSolver(StiffnessMatrix &M, Solver &precon, double tol, int maxIter, bool verbose) : 
    _mat(M),
    _prec(precon),
    _maxIter(maxIter),
    _tol(tol),
    _verbose(verbose),
    _startvectorSet(false)
{
}

PCGSolver::~PCGSolver()
{
}

int PCGSolver::Solve(Eigen::VectorXd &r, Eigen::VectorXd &x)
{
    // This solver is used in:
    // - problem.h
    // - MlCycle.h : _solver = MlLevelCG (Wrapper)


    return std::get<0>(Solve_res(r, x));
}

std::tuple<int, double, double> PCGSolver::Solve_res(Eigen::VectorXd &r, Eigen::VectorXd &x)
{
    // This solver is used in:
    // - problem.h
    // - MlCycle.h : _solver = MlLevelCG (Wrapper)

    int MyPID = _mat.GetPID();
    double alpha, res0, resreal0, delta_new, delta_old, beta, norm;
    
    // Cout flags
    std::ios_base::fmtflags origflag;
    if (_verbose)
    {
        origflag = std::cout.flags();
        std::cout.setf(std::ios::scientific);
    }

    Eigen::VectorXd s = r;
    Eigen::VectorXd d(_mat.GetNrDofs());

    if(_startvectorSet)
    {
        // Residuum based on zero-vector; used for having the same resreal0 w/ or w/o startvector
        // This part is only executed from main.cpp:problem.Solve():solver->Solve(tmp, *_x)

        Eigen::VectorXd r0 = r;
        Eigen::VectorXd d0 = d;

        _mat.Apply(x0, d0);                     // d = A*x
        r0 = s - d0;                            // r = s - d = s - A*x = b - A*x, s is used as b initially (RHS)
        _prec.Solve(r0, d0);                    // d = C^-1 * r
        double ignored_delta = 0.0;
        _mat.dot_pair(r0, r0, r0, d0, resreal0, ignored_delta);
        resreal0 = sqrt(resreal0);              // resreal0 = ||r^T*r|| => non-preconditioned

        if (_verbose) 
            PCOUT(MyPID, "   Zero-based residuum: " << resreal0 << std::endl);
    }  

    // Start solving, startvector-based if given
    _mat.Apply(x, d);                           // d = A*x
    r = s - d;                                  // r = s - d = s - A*x = b - A*x, s is used as b initially (RHS)
    _prec.Solve(r, d);                          // d = inv(C) * r
    _mat.dot_pair(r, r, r, d, norm, delta_new);
    norm = sqrt(norm);                          // norm = ||r^T*r||

    if (_startvectorSet && _verbose) 
        PCOUT(MyPID, "   Startvector-based residuum: " << norm << std::endl);

    if(false && _startvectorSet && norm > resreal0)
    {
        // Startvector worse than zero-vector
        x = x0;

        // Repeat calculations with zero-vector
        _mat.Apply(x, d);                           // d = A*x
        r = s - d;                                  // r = s - d = s - A*x = b - A*x, s is used as b initially (RHS)
        _prec.Solve(r, d);                          // d = inv(C) * r
        _mat.dot_pair(r, r, r, d, norm, delta_new);
        resreal0 = norm = sqrt(norm);               // norm = ||r^T*r||

        if (_verbose) 
            PCOUT(MyPID, "   Bad startvector discarded, using zero-vector instead" << std::endl);
    }
    else if(!_startvectorSet)
    {
        // No startvector given
        resreal0 = norm;
    }

    // Arbenz:
    // res = sqrt( r^T inv(C) r);
    // res0 = sqrt(b^T inv(C) b);
    res0 = sqrt(delta_new);                     // res0 = sqrt(delta_new) = sqrt(r^T * d) = sqrt(r^T * inv(C) * r)
    
    if (_verbose)
    {
        PCOUT(MyPID, "   It: " << std::setw(3) << 0 << "\tPrec rel res: " << std::setw(13) << sqrt(delta_new) / res0);   // res/res0
        PCOUT(MyPID, "\tRel res: " << std::setw(13)  << norm / resreal0);
        PCOUT(MyPID, "\tAbs res: " << std::setw(13)  << norm << std::endl);
    }

    // Solve loop
    int i = 0;
    while (i < _maxIter && norm / resreal0 > _tol)
    {
        // PCG-Algorithm
        // _mat = A
        // _prec = C

        // Arbenz:
        // res = sqrt( r^T inv(C) r);
        // res0 = sqrt(b^T inv(C) b);

        _mat.Apply(d, s);                       // s = A*d
        double dTs = 0.0;
        _mat.dot_pair(d, s, r, r, dTs, norm);
        alpha = delta_new / dTs;                // alpha = delta_new / (d^T*s)
        x = x + alpha * d;                      // x = x + alpha*d
        //r = r - alpha*q;
        r = r - alpha * s;                      // r = r - alpha*s
        _prec.Solve(r, s);                      // s = inv(C) * r
        delta_old = delta_new;                  // delta_old = delta_new
        _mat.dot_pair(r, s, r, r, delta_new, norm);
        beta = delta_new / delta_old;           // beta = delta_new/delta_old
        d = s + beta * d;                       // d = s + beta*d
        // res = sqrt(delta_new);               // res = sqrt(delta_new) = sqrt(r^T * s) = sqrt(r^T * inv(C) * r)
        norm = sqrt(norm);                      // norm = ||r*r||

        i++;

        if (_verbose)
        {
            PCOUT(MyPID, "   It: " << std::setw(3) << i << "\tPrec rel res: " << std::setw(13)  << sqrt(delta_new) / res0);   // res/res0
            PCOUT(MyPID, "\tRel res: " << std::setw(13)  << norm / resreal0);
            PCOUT(MyPID, "\tAbs res: " << std::setw(13)  << norm << std::endl);
        }
    }

    std::cout.flags(origflag);
    return std::tuple<int,double, double>(i, norm / resreal0, norm);
}

void PCGSolver::SetX0(Eigen::VectorXd x)
{
    x0 = x;
    _startvectorSet = true;
}

const std::string PCGSolver::Label() const
{
    return ("Preconditioned CG-Solver\n");
}

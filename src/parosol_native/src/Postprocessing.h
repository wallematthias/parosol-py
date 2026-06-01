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

#ifndef POSTPROCESSING_H
#define POSTPROCESSING_H

#include "Config.h"
#include "StiffnessMatrix.h"
#include "Toolbox.h"
#include "fem.h"


/*! The PostProcess Class computes the von mises stress and the strain energy density
 *  from the resulting displacement of the computation.
 */


template <class Grid>
class PostProcess
{
	public:
		PostProcess(Grid &grid) :_grid(grid)
	{
	}

		//! ElementByElementMatrix destructor
		~PostProcess() { }

		void ComputeStressAndStrain(Eigen::VectorXd &disp, Eigen::VectorXd &VonMises, Eigen::VectorXd &SED, Eigen::VectorXd &eff, 
									Eigen::VectorXd &e_dev, Eigen::VectorXd &e_vol, Eigen::VectorXd &e_xx, Eigen::VectorXd &e_yy, 
									Eigen::VectorXd &e_zz, Eigen::VectorXd &e_xy, Eigen::VectorXd &e_yz, Eigen::VectorXd &e_xz, 
									Eigen::VectorXd &s_xx, Eigen::VectorXd &s_yy, Eigen::VectorXd &s_zz, Eigen::VectorXd &s_xy, 
									Eigen::VectorXd &s_yz, Eigen::VectorXd &s_xz, Eigen::VectorXd &DP_s,Eigen::VectorXd &DP_e, 
									Eigen::VectorXd &e1, Eigen::VectorXd &e2, Eigen::VectorXd &e3, 
									Eigen::VectorXd &s1, Eigen::VectorXd &s2, Eigen::VectorXd &s3) { // add stress, strain, distortioanl strain, volumetric strain
			//fetch the nodes of the neighbours
			_grid.Recv_import_Ghost(disp);
			_grid.Send_import_Ghost(disp);

            // Set the reference element up
			double GridDim[3];
			int Dimension = 3;
			const int NumMaterialProps = 2;
			double _matprop[2];
			_matprop[0] = 1000; //reference value Emodule is linear
			_matprop[1] = _grid.poisons_ratio;
			int NumNodesPerElement = 8;
			int NumDofsPerElement = 24;
			int NumGaussPoints = 1;
			int SSMatrixSize = 6;
			double *coord = new double[Dimension * NumNodesPerElement];
			_grid.GetRes(GridDim);
			setcoord(GridDim,coord);

			double* strainbuf = new double[(SSMatrixSize +1) * NumGaussPoints ];
			double* stressbuf = new double[(SSMatrixSize +1) * NumGaussPoints ];
			double temp_e_dev;
			double temp_e_vol;
			double temp_e_xx;
			double temp_e_yy;
			double temp_e_zz;
			double temp_e_xy;
			double temp_e_yz;
			double temp_e_xz;
			double temp_s_xx;
			double temp_s_yy;
			double temp_s_zz;
			double temp_s_xy;
			double temp_s_yz;
			double temp_s_xz;
			double temp_DP_s;
			double temp_DP_e;
			double temp_e1, temp_e2, temp_e3;
			double temp_s1, temp_s2, temp_s3;
			double sigma, theta;
			t_index nr_elem =_grid.GetNrElem();
			VonMises.setZero(nr_elem);
			SED.setZero(nr_elem);
			eff.setZero(nr_elem);
			e_dev.setZero(nr_elem);
			e_vol.setZero(nr_elem);
			e_xx.setZero(nr_elem);
			e_yy.setZero(nr_elem);
			e_zz.setZero(nr_elem);
			e_xy.setZero(nr_elem);
			e_yz.setZero(nr_elem);
			e_xz.setZero(nr_elem);
			s_xx.setZero(nr_elem);
			s_yy.setZero(nr_elem);
			s_zz.setZero(nr_elem);
			s_xy.setZero(nr_elem);
			s_yz.setZero(nr_elem);
			s_xz.setZero(nr_elem);
			DP_s.setZero(nr_elem);
			DP_e.setZero(nr_elem);
			e1.setZero(nr_elem);
			e2.setZero(nr_elem);
			e3.setZero(nr_elem);
			s1.setZero(nr_elem);
			s2.setZero(nr_elem);
			s3.setZero(nr_elem);
			//fetch the 24 values in pref and store store
			// res = K_e * xpref
            Eigen::Matrix<double,24,1> xpref;
			int nr_ele = 0;

			_grid.Wait_import_Ghost();
			MPI_Barrier(MPI_COMM_WORLD);

			for(_grid.initIterateOverElements(); _grid.TestIterateOverElements(); _grid.IncIterateOverElements()){

				_grid.GetNodalDisplacementsOfElement(disp, xpref);

				_matprop[0] = 1000*_grid.GetElementWeight();
				_matprop[1] = _grid.GetElementPoissonRatio();
                double emoduli = 1000*_grid.GetElementWeight();
				Element_Stress(_matprop, NumMaterialProps,
						NumNodesPerElement, NumDofsPerElement,
	                    Dimension, NumGaussPoints, SSMatrixSize,
						coord, &xpref[0],
						strainbuf, stressbuf,
						&temp_e_vol, &temp_e_dev, &sigma, &theta,
						&temp_e_xx,&temp_e_yy,&temp_e_zz,
						&temp_e_xy,&temp_e_yz,&temp_e_xz,
						&temp_s_xx,&temp_s_yy,&temp_s_zz,
						&temp_s_xy,&temp_s_yz,&temp_s_xz,
						&temp_DP_s,&temp_DP_e,
						&temp_e1, &temp_e2, &temp_e3,
						&temp_s1, &temp_s2, &temp_s3); // add extra structures
				VonMises[nr_ele] = stressbuf[6];
				SED[nr_ele] = strainbuf[6];
                eff[nr_ele] = sqrt(2*SED[nr_ele]/emoduli);
                e_dev[nr_ele] = temp_e_dev;
                e_vol[nr_ele] = temp_e_vol;
                e_xx[nr_ele] = temp_e_xx;
                e_yy[nr_ele] = temp_e_yy;
                e_zz[nr_ele] = temp_e_zz;
                e_xy[nr_ele] = temp_e_xy;
                e_yz[nr_ele] = temp_e_yz;
                e_xz[nr_ele] = temp_e_xz;
                s_xx[nr_ele] = temp_s_xx;
                s_yy[nr_ele] = temp_s_yy;
                s_zz[nr_ele] = temp_s_zz;
                s_xy[nr_ele] = temp_s_xy;
                s_yz[nr_ele] = temp_s_yz;
                s_xz[nr_ele] = temp_s_xz;
				e1[nr_ele] = temp_e1;
				e2[nr_ele] = temp_e2;
				e3[nr_ele] = temp_e3;
				s1[nr_ele] = temp_s1;
				s2[nr_ele] = temp_s2;
				s3[nr_ele] = temp_s3;

		DP_s[nr_ele] = temp_DP_s;
		DP_e[nr_ele] = temp_DP_e;
				nr_ele++;

			}
			delete[] strainbuf;
			delete[] stressbuf;
		}

	private:
		Grid& _grid;
};

#endif /* POSTPROCESSING_H */

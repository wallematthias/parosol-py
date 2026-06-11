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


#ifndef HDF5PRINTER_H
#define HDF5PRINTER_H

#include "Config.h"
#include "OctreeGrid.h"
#include "Postprocessing.h"
#include "GWriter.hpp"

#include <Eigen/Core>

//! This class prints the grid and result into a HDF5 file.

/*! The printer writes the whole into a single HDF5-file. This file can be read by ParaView.
 *  To read it with ParaView a XML-file is used. This file can be generate with a helper script.
 */

template <class T>
class HDF5Printer {
	public:
		HDF5Printer(std::string filename, OctreeGrid<T> &grid):_MyPID(grid.GetPID()), _Size(grid.GetNrCPU()),_filename(filename), _grid(grid)
		{
		  Writer = new HDF5_GWriter(filename, MPI_COMM_WORLD);
		}
		int _MyPID;
		int _Size;
		~HDF5Printer()
		{
		  Writer->Close();
		  delete Writer;
		}

		void OctKey_to_Coord(long key, int &x, int &y, int &z)
		{
			x = y = z = 0;
			for(int i = 0; i < 20; i++) {
				x += (key & 1) << i;
				key = key >> 1;
				y += (key & 1) << i;
				key = key >> 1;
				z += (key & 1) << i;
				key = key >> 1;
			}
		}


		void PrintGrid() {
			PrintCoord("Coordinates");
			PrintElems("Elements");
            PrintEmoduli();
		}

		void PrintCoord(std::string dset) {
		  Writer->Select("/Mesh");


			t_octree_key k;
			std::vector<OctreeNode> &grid = _grid.GetOctGrid();
			std::vector<OctreeNode>::iterator iter;
			int x =0,y=0,z=0;
			double res[3];
			_grid.GetRes(res);
            Eigen::VectorXf coord(_grid.GetNrPrivateNodes()*3);
			long i=0;
			T keys;
			t_octree_key tmp;
			for(iter = grid.begin(); iter != _grid._GridIteratorEnd; ++iter ) {
				k = iter->key;
				OctKey_to_Coord(k, x, y, z);
				tmp = keys(x,y,z);
				if (tmp != k) {
				  PCOUT(_MyPID, "ERROR not the same key\n")
				  break;
				}
				coord[i++] = x*res[0];
				coord[i++] = y*res[1];
				coord[i++] = z*res[2];
			}
			Writer->Write(dset, coord.data(), _grid.GetNrNodesGlobal(),_grid.GetNrPrivateNodes(), 3, _grid.GetNodeOffset());
		}

		void PrintElems(std::string dset) {
			Writer->Select("/Mesh");

			//print element to node
			t_octree_key *elems = new t_octree_key[_grid.GetNrElem()*8];

			t_index local_nodes[8];

			//Quick an dirty hack:
			//compute the offset with double
            Eigen::VectorXd ind(_grid.GetNrDofs());
			_grid.Recv_import_Ghost(ind);
			ind.setZero(_grid.GetNrDofs());
			t_octree_key offset = _grid.GetNodeOffset();
			MPI_Barrier(MPI_COMM_WORLD);
			for (t_index i = 0; i < _grid.GetNrNodes(); i++) {
				ind[3*i] = i + offset;
			}
			MPI_Barrier(MPI_COMM_WORLD);
			_grid.Send_import_Ghost(ind);
			_grid.Wait_import_Ghost();
			MPI_Barrier(MPI_COMM_WORLD);

			t_index e =0;
			for(_grid.initIterateOverElements(); _grid.TestIterateOverElements(); _grid.IncIterateOverElements()){
				_grid.SearchIndexes(local_nodes);
				for(int i =0; i <8; i++)
					elems[e*8+i] = (long) ind[3*local_nodes[i]];
				e++;
			}
			MPI_Barrier(MPI_COMM_WORLD);

			Writer->Write(dset, elems, _grid.GetNrElemGlobal(),_grid.GetNrElem(), 8, _grid.GetElemOffset());
			delete[] elems;
		}

        void PrintEmoduli() {
            Writer->Select("/Mesh");
            Eigen::VectorXf emoduli(_grid.GetNrElem());
			t_index i =0;
			for(_grid.initIterateOverElements(); _grid.TestIterateOverElements(); _grid.IncIterateOverElements()){
				emoduli[i]=_grid.GetElementWeight()*1000;
				i++;
			}
            Writer->Write("Emoduli", emoduli.data(), _grid.GetNrElemGlobal(),_grid.GetNrElem(), 1, _grid.GetElemOffset());
        }

        //x displacement, res residuum
		void PrintAll(Eigen::VectorXd &x, Eigen::VectorXd &force, Eigen::VectorXd &res,int SED_flag, int EFF_flag, int VonMises_flag,
				  int e_dev_flag, int e_vol_flag, int strain_flag, int stress_flag, int DP_s_flag, int DP_e_flag) {
			PrintGrid();

			PostProcess<OctreeGrid<T> > post(_grid);
            typename PostProcess<OctreeGrid<T> >::FieldSelection selection;
            selection.von_mises = VonMises_flag == 1;
            selection.sed = SED_flag == 1;
            selection.eff = EFF_flag == 1;
            selection.e_dev = e_dev_flag == 1;
            selection.e_vol = e_vol_flag == 1;
            selection.strain = strain_flag == 1;
            selection.stress = stress_flag == 1;
            selection.dp_s = DP_s_flag == 1;
            selection.dp_e = DP_e_flag == 1;
            selection.principal = strain_flag == 1 || stress_flag == 1;
			Eigen::VectorXd m, s, eff, e_dev, e_vol, e_xx, e_yy, e_zz, e_xy, e_yz, e_xz, s_xx, s_yy, s_zz, s_xy, s_yz, s_xz, dp_s, dp_e, e1, e2, e3, s1, s2, s3;
	    post.ComputeStressAndStrain(x,m,s,eff,e_dev,e_vol,e_xx,e_yy,e_zz,e_xy,e_yz,e_xz,s_xx,s_yy,s_zz,s_xy,s_yz,s_xz,dp_s,dp_e,e1,e2,e3,s1,s2,s3,selection);

			Writer->Select("/Solution");
			Writer->Write("disp", x.data(), _grid.GetNrNodesGlobal(),_grid.GetNrPrivateNodes(), 3, _grid.GetNodeOffset());
			Writer->Write("force", force.data(), _grid.GetNrNodesGlobal(),_grid.GetNrPrivateNodes(), 3, _grid.GetNodeOffset());
			if(VonMises_flag==1){
			Writer->Write("VonMises", m.data(), _grid.GetNrElemGlobal(),_grid.GetNrElem(), 1, _grid.GetElemOffset());
			}
			if(SED_flag==1){
			Writer->Write("SED", s.data(), _grid.GetNrElemGlobal(),_grid.GetNrElem(), 1, _grid.GetElemOffset());
			}
			if(EFF_flag==1){
			Writer->Write("EFF", eff.data(), _grid.GetNrElemGlobal(),_grid.GetNrElem(), 1, _grid.GetElemOffset());
			}
			if(e_dev_flag==1){
			Writer->Write("e_dev", e_dev.data(), _grid.GetNrElemGlobal(),_grid.GetNrElem(), 1, _grid.GetElemOffset());
			}
			if(e_vol_flag==1){
			Writer->Write("e_vol", e_vol.data(), _grid.GetNrElemGlobal(),_grid.GetNrElem(), 1, _grid.GetElemOffset());
			}
			if(strain_flag==1){
			Writer->Write("e_xx", e_xx.data(), _grid.GetNrElemGlobal(),_grid.GetNrElem(), 1, _grid.GetElemOffset());
			Writer->Write("e_yy", e_yy.data(), _grid.GetNrElemGlobal(),_grid.GetNrElem(), 1, _grid.GetElemOffset());
			Writer->Write("e_zz", e_zz.data(), _grid.GetNrElemGlobal(),_grid.GetNrElem(), 1, _grid.GetElemOffset());
			Writer->Write("e_xy", e_xy.data(), _grid.GetNrElemGlobal(),_grid.GetNrElem(), 1, _grid.GetElemOffset());
			Writer->Write("e_yz", e_yz.data(), _grid.GetNrElemGlobal(),_grid.GetNrElem(), 1, _grid.GetElemOffset());
			Writer->Write("e_xz", e_xz.data(), _grid.GetNrElemGlobal(),_grid.GetNrElem(), 1, _grid.GetElemOffset());
			}
			if(stress_flag==1){
			Writer->Write("s_xx", s_xx.data(), _grid.GetNrElemGlobal(),_grid.GetNrElem(), 1, _grid.GetElemOffset());
			Writer->Write("s_yy", s_yy.data(), _grid.GetNrElemGlobal(),_grid.GetNrElem(), 1, _grid.GetElemOffset());
			Writer->Write("s_zz", s_zz.data(), _grid.GetNrElemGlobal(),_grid.GetNrElem(), 1, _grid.GetElemOffset());
			Writer->Write("s_xy", s_xy.data(), _grid.GetNrElemGlobal(),_grid.GetNrElem(), 1, _grid.GetElemOffset());
			Writer->Write("s_yz", s_yz.data(), _grid.GetNrElemGlobal(),_grid.GetNrElem(), 1, _grid.GetElemOffset());
			Writer->Write("s_xz", s_xz.data(), _grid.GetNrElemGlobal(),_grid.GetNrElem(), 1, _grid.GetElemOffset());		   		     
			}
			if(DP_s_flag==1){
			Writer->Write("DP_s", dp_s.data(), _grid.GetNrElemGlobal(),_grid.GetNrElem(), 1, _grid.GetElemOffset());
			}
			if(DP_e_flag==1){
			Writer->Write("DP_e", dp_e.data(), _grid.GetNrElemGlobal(),_grid.GetNrElem(), 1, _grid.GetElemOffset());
			}
			
            if(selection.principal){
			    Writer->Write("e1", e1.data(), _grid.GetNrElemGlobal(),_grid.GetNrElem(), 1, _grid.GetElemOffset());
			    Writer->Write("e2", e2.data(), _grid.GetNrElemGlobal(),_grid.GetNrElem(), 1, _grid.GetElemOffset());
			    Writer->Write("e3", e3.data(), _grid.GetNrElemGlobal(),_grid.GetNrElem(), 1, _grid.GetElemOffset());

			    Writer->Write("s1", s1.data(), _grid.GetNrElemGlobal(),_grid.GetNrElem(), 1, _grid.GetElemOffset());
			    Writer->Write("s2", s2.data(), _grid.GetNrElemGlobal(),_grid.GetNrElem(), 1, _grid.GetElemOffset());
			    Writer->Write("s3", s3.data(), _grid.GetNrElemGlobal(),_grid.GetNrElem(), 1, _grid.GetElemOffset());
            }
			
		}

		void PrintPartition(std::string dset) {
          Eigen::VectorXi part(_grid.GetNrElem());
		  part.setConstant(_grid.GetNrElem(), _MyPID);
		  Writer->Write(dset, part.data(), _grid.GetNrElemGlobal(),_grid.GetNrElem(), 1, _grid.GetElemOffset());
		}






		std::string _filename;

		OctreeGrid<T> &_grid;

		HDF5_GWriter *Writer;

};
#endif /* HDF5PRINTER_H */

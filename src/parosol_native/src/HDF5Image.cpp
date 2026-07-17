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


#include "Config.h"
#include "HDF5Image.h"
#include <mpi.h>
#include <hdf5.h>
#include <string>
#include <iostream>
#include <map>
#include <cstdlib>

#include "Timing.h"

void HDF5Image::ReadBC(HDF5_GReader &reader, std::string s,std::vector<unsigned short> & coordinates, std::vector<float> & values)
{
  hsize_t global_dims_of_hdf5[3];
  if (reader.GetSizeOfDataset((s+"_Values").c_str(),global_dims_of_hdf5, 1) > 0)
  {
    std::map<bcitem, double> bcmap;
    double bc_per_cpu = ((double) global_dims_of_hdf5[0])/mpi_size;
    hsize_t my_bc_offset = MyPID*bc_per_cpu;
    hsize_t my_bc_upper_offset =(MyPID+1)*bc_per_cpu; //last element is not included. eg 2  then 0,1 belongs to it
    hsize_t my_bc_count =my_bc_upper_offset - my_bc_offset;
    if (MyPID == mpi_size -1) {
        my_bc_upper_offset = global_dims_of_hdf5[0];
        my_bc_count =my_bc_upper_offset - my_bc_offset;
    }

    float *bcdisp = new float[my_bc_count];
    reader.Read((s+"_Values").c_str(), bcdisp, &my_bc_offset, &my_bc_count, 1);

    short *bccoords = new short[my_bc_count*4];
    reader.Read((s+"_Coordinates").c_str(), bccoords, global_dims_of_hdf5[0], my_bc_count, 4, my_bc_offset);
    bcitem tmp;
    unsigned int d =0;
    double disp;
    for(unsigned int i=0; i < my_bc_count*4;) {
        tmp.z = bccoords[i++]; 
        tmp.y = bccoords[i++];
        tmp.x = bccoords[i++];
        tmp.d = bccoords[i++];
        disp = bcdisp[d++];
        if (disp ==0)
            disp = 1e-16;
        bcmap[tmp] = disp;
    }

    for(std::map<bcitem, double>::iterator it =  bcmap.begin(); it != bcmap.end(); ++it) {
        coordinates.push_back(it->first.x);
        coordinates.push_back(it->first.y);
        coordinates.push_back(it->first.z);
        coordinates.push_back(it->first.d);
        values.push_back(it->second);
    }
    delete[] bcdisp;
    delete[] bccoords;
  }
}

HDF5Image::HDF5Image(std::string &fi, CPULayout &layout): _file(fi),_layout(layout), MyPID(0)
{
}

HDF5Image::~HDF5Image()
{
}

int HDF5Image::Scan(BaseGrid* grid)
{
  //Timer timer(MPI_COMM_WORLD);
  int mpi_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
  MyPID = mpi_rank;
  PCOUT(MyPID, "Generating mesh... \n")

  //setting left corner
  grid->corner_x = 0;
  grid->corner_y = 0;
  grid->corner_z = 0;

  HDF5_GReader reader(_file);

  hsize_t global_dims_of_hdf5[3];
  if (!reader.Select("Image_Data")) {
      PCOUT(MyPID, "Error Selecting Image!!!\n")
      MPI_Finalize();
      exit(-1);
  }
      
  reader.GetSizeOfDataset("Image",global_dims_of_hdf5, 3);
  
  PCOUT(MyPID, "Image has following dimension: " << global_dims_of_hdf5[0] << " " << global_dims_of_hdf5[1] << " " << global_dims_of_hdf5[2] << std::endl;)
  
  for(int i=0; i<3; i++) {
    grid->gdim[i] = global_dims_of_hdf5[2-i];
    grid->ldim[i] = global_dims_of_hdf5[2-i];
  }
  const char* shape_aware = std::getenv("PAROSOL_LAYOUT_SHAPE_AWARE");
  if (shape_aware != 0 && std::string(shape_aware) == "1") {
    _layout.ComputeGridForDimensions(grid->gdim[0], grid->gdim[1], grid->gdim[2]);
  }
  _procx = _layout.CPUGrid()[0];
  _procy = _layout.CPUGrid()[1];
  _procz = _layout.CPUGrid()[2];

  PCOUT(MyPID, "Computing dimension, using cores in x: "<< _procx << ", y: " << _procy << ", z: " << _procz << "\n")
   
  grid->_grid =0;
  
  
  //Compute the local dimension
  double dims_per_cpu[3];
  for(int i=0; i<3; i++) {
    dims_per_cpu[2-i] = ((double) grid->gdim[i])/_layout.CPUGrid()[i];
  }
  hsize_t my_offset[3] = {};
  hsize_t my_upper_offset[3] ={}; //last element is not included. eg 2  then 0,1 belongs to it
  hsize_t my_count[3] ={};
  
  for(int i=0;i <3; i++) {
    my_offset[2-i]     = _layout.CPUCoord()[i]*dims_per_cpu[2-i];
    my_upper_offset[2-i]  = (_layout.CPUCoord()[i]+1)*dims_per_cpu[2-i];
    my_count[2-i] = my_upper_offset[2-i] - my_offset[2-i];
  }
  
  for(int i=0; i <3; i++) {
    if (_layout.CPUCoord()[i] == _layout.CPUGrid()[i] -1) {
      my_upper_offset[2-i] = grid->gdim[i];
      my_count[2-i] = my_upper_offset[2-i] - my_offset[2-i];
    }
  }
  
  //CPU are xmajor  HDF is zmajor
  for(int i=0; i <3; i++) {
    grid->corner[i] = my_offset[2-i];
    grid->ldim[i] = my_count[2-i];
  }
  
    long imagesize = grid->ldim[0]*grid->ldim[1]*grid->ldim[2];
    //timer.Start("Image");

    grid->_grid = new double[imagesize];
    reader.Read("Image", grid->_grid, my_offset, my_count, 3);

    //timer.Stop("Image");
    //t_timing elapsed_time = timer.ElapsedTime("Image");
    //PCOUT(MyPID, "Time for Reading the Image: " << COUTTIME(elapsed_time) << "s\n");

    hsize_t poisson_dims_of_hdf5[3];
    std::string poisson_dataset = "";
    if (reader.GetSizeOfDataset("Poissons_ratio_Image", poisson_dims_of_hdf5, 3) > 0) {
      poisson_dataset = "Poissons_ratio_Image";
    } else if (reader.GetSizeOfDataset("Poisons_ratio_Image", poisson_dims_of_hdf5, 3) > 0) {
      poisson_dataset = "Poisons_ratio_Image";
    }
    if (!poisson_dataset.empty()) {
      for(int d=0; d<3; d++) {
        if (poisson_dims_of_hdf5[d] != global_dims_of_hdf5[d]) {
          PCOUT(MyPID, "Error: Poisson ratio image dimensions do not match Image dimensions.\n")
          MPI_Finalize();
          exit(-1);
        }
      }
      grid->_poisson_grid = new double[imagesize];
      reader.Read(poisson_dataset.c_str(), grid->_poisson_grid, my_offset, my_count, 3);
    }

    //timer.Start("BC");

    //timer.Stop("BC");
    ReadBC(reader, std::string("Fixed_Displacement"), grid->fixed_nodes_coordinates, grid->fixed_nodes_values);
    ReadBC(reader, std::string("Loaded_Nodes"), grid->loaded_nodes_coordinates, grid->loaded_nodes_values);
    //elapsed_time = timer.ElapsedTime("BC");
    //PCOUT(MyPID, "Time for Reading BC: " << COUTTIME(elapsed_time) << "s\n");
    double res;
    reader.Read("Voxelsize", res);
    for(int i=0; i<3; i++) {
      grid->res[i] = res;
    }
    reader.Read("Poisons_ratio", grid->poisons_ratio);

    if (reader.GroupExists("/Nonlinear")) {
      reader.Select("/Nonlinear");
      nonlinear_enabled = true;
      reader.ReadAttribute("material_type", nonlinear_material_type);
      reader.ReadAttribute("youngs_modulus_mpa", nonlinear_E_mpa);
      reader.ReadAttribute("poisson_ratio", nonlinear_nu);
      reader.ReadAttribute("yield_strength_mpa", nonlinear_Y_mpa);
      reader.ReadAttribute("convergence_tolerance", nonlinear_convergence_tolerance);
      reader.ReadAttribute("maximum_plastic_iterations", nonlinear_maximum_plastic_iterations);
      reader.ReadAttribute("plastic_convergence_window", nonlinear_plastic_convergence_window);
    }

   PCOUT(MyPID, "HDF5 ImageReader: \n")
   PCOUT(MyPID, "   global Dimension: " << grid->gdim[0] << " " << grid->gdim[1] << " " << grid->gdim[2] << "\n")
   PCOUT(MyPID, "   local Dimension: " << grid->ldim[0] << " " << grid->ldim[1] << " " << grid->ldim[2] << "\n")
   PCOUT(MyPID, "   Resolution: " << grid->res[0] << "\n")
   PCOUT(MyPID, "   Poison's ratio: " << grid->poisons_ratio << "\n")
   if (grid->_poisson_grid != 0) {
     PCOUT(MyPID, "   Per-element Poison's ratio image: " << poisson_dataset << "\n")
   }
   
   long num_gl_bc[2], num_loc_bc[2];
   num_loc_bc[0]=grid->fixed_nodes_values.size();
   num_loc_bc[1]=grid->loaded_nodes_values.size();
   MPI_Reduce(&num_loc_bc, &num_gl_bc, 2, MPI_LONG, MPI_SUM, 0, MPI_COMM_WORLD );
   PCOUT(MyPID, "   BC: Fixednodesize: " << num_gl_bc[0] << ", Loadednodesize " << num_gl_bc[1] << "\n")
      return 0;
  }

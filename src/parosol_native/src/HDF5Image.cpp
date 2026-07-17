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
#include <cmath>
#include <sstream>

#include "Timing.h"

static int GetDatasetRankAndDims(const std::string& filename, const std::string& dataset_path, hsize_t* dims)
{
  int mpi_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
  int rank = -1;
  if (mpi_rank == 0) {
    hid_t file = H5Fopen(filename.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    if (file >= 0) {
      hid_t dataset = H5Dopen(file, dataset_path.c_str(), H5P_DEFAULT);
      if (dataset >= 0) {
        hid_t dataspace = H5Dget_space(dataset);
        if (dataspace >= 0) {
          rank = H5Sget_simple_extent_ndims(dataspace);
          if (rank > 0 && rank <= 3) {
            hsize_t actual_dims[3] = {};
            H5Sget_simple_extent_dims(dataspace, actual_dims, 0);
            for (int d=0; d<rank; d++) {
              dims[d] = actual_dims[d];
            }
          }
          H5Sclose(dataspace);
        }
        H5Dclose(dataset);
      }
      H5Fclose(file);
    }
  }
  MPI_Bcast(&rank, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(dims, 3*sizeof(hsize_t), MPI_BYTE, 0, MPI_COMM_WORLD);
  return rank;
}

template<typename T>
static bool ReadRequiredMapDataset(HDF5_GReader& reader, const char* dataset_name, T* target, hsize_t* my_offset, hsize_t* my_count, std::ostringstream& error)
{
  if (reader.Read(dataset_name, target, my_offset, my_count, 3) <= 0) {
    error << " failed to read " << dataset_name << ";";
    return false;
  }
  return true;
}

static bool ReadRequiredFloatMapDatasetAsDouble(
  HDF5_GReader& reader,
  const char* dataset_name,
  double* target,
  long imagesize,
  hsize_t* my_offset,
  hsize_t* my_count,
  std::ostringstream& error)
{
  float* buffer = new float[imagesize];
  bool ok = ReadRequiredMapDataset(reader, dataset_name, buffer, my_offset, my_count, error);
  if (ok) {
    for (long i = 0; i < imagesize; ++i) {
      target[i] = static_cast<double>(buffer[i]);
    }
  }
  delete[] buffer;
  return ok;
}

static bool ReadRequiredMaterialIdDataset(
  HDF5_GReader& reader,
  unsigned short* target,
  long imagesize,
  hsize_t* my_offset,
  hsize_t* my_count,
  std::ostringstream& error)
{
  short* buffer = new short[imagesize];
  bool ok = ReadRequiredMapDataset(reader, "MaterialID", buffer, my_offset, my_count, error);
  if (ok) {
    for (long i = 0; i < imagesize; ++i) {
      target[i] = static_cast<unsigned short>(buffer[i]);
    }
  }
  delete[] buffer;
  return ok;
}

static void ValidateAsymmetricMaterialMap(
  BaseGrid* grid,
  const double* youngs_mpa,
  const double* poisson_ratio,
  const double* sigma_c_mpa,
  const double* sigma_t_mpa,
  const double* plateau_mpa,
  const unsigned short* material_id,
  long imagesize,
  std::ostringstream& error)
{
  bool invalid_material_id = false;
  bool unsupported_material_id = false;
  bool invalid_youngs = false;
  bool mismatched_youngs = false;
  bool invalid_poisson = false;
  bool invalid_sigma_c = false;
  bool invalid_sigma_t = false;
  bool invalid_plateau = false;
  const double stiffness_tolerance = 1.0e-5;

  for (long i = 0; i < imagesize; ++i) {
    const bool active_image_voxel = grid->_grid[i] > 0.0;
    const bool active_material_voxel = material_id[i] != 0;
    if (!active_image_voxel && !active_material_voxel) {
      continue;
    }

    if (active_image_voxel && material_id[i] == 0) {
      invalid_material_id = true;
    }
    if (active_material_voxel && material_id[i] != 1 && material_id[i] != 2) {
      unsupported_material_id = true;
    }
    if (!std::isfinite(youngs_mpa[i]) || youngs_mpa[i] <= 0.0) {
      invalid_youngs = true;
    } else {
      const double expected_stiffness_gpa = youngs_mpa[i] / 1000.0;
      const double image_stiffness_gpa = grid->_grid[i];
      const double tolerance = stiffness_tolerance
        * std::max(1.0, std::fabs(image_stiffness_gpa));
      if (std::fabs(expected_stiffness_gpa - image_stiffness_gpa) > tolerance) {
        mismatched_youngs = true;
      }
    }
    if (!std::isfinite(poisson_ratio[i])
        || poisson_ratio[i] <= -1.0
        || poisson_ratio[i] >= 0.5) {
      invalid_poisson = true;
    }
    if (material_id[i] == 1) {
      if (!std::isfinite(sigma_c_mpa[i]) || sigma_c_mpa[i] <= 0.0) {
        invalid_sigma_c = true;
      }
      if (!std::isfinite(sigma_t_mpa[i]) || sigma_t_mpa[i] <= 0.0) {
        invalid_sigma_t = true;
      }
      if (!std::isfinite(plateau_mpa[i]) || plateau_mpa[i] <= 0.0) {
        invalid_plateau = true;
      }
    } else if (material_id[i] == 2) {
      if (!std::isfinite(sigma_c_mpa[i])) {
        invalid_sigma_c = true;
      }
      if (!std::isfinite(sigma_t_mpa[i])) {
        invalid_sigma_t = true;
      }
      if (!std::isfinite(plateau_mpa[i])) {
        invalid_plateau = true;
      }
    }
  }

  if (invalid_material_id) {
    error << " MaterialID must be positive for active Image voxels;";
  }
  if (unsupported_material_id) {
    error << " MaterialID values must be 1 for nonlinear bone or 2 for elastic fixture voxels;";
  }
  if (invalid_youngs) {
    error << " YoungsModulusMPa values must be finite and positive;";
  }
  if (mismatched_youngs) {
    error << " YoungsModulusMPa must match Image stiffness;";
  }
  if (invalid_poisson) {
    error << " PoissonRatio values must satisfy -1 < nu < 0.5;";
  }
  if (invalid_sigma_c) {
    error << " CompressiveYieldStressMPa values must be finite and positive for nonlinear bone voxels;";
  }
  if (invalid_sigma_t) {
    error << " TensileYieldStressMPa values must be finite and positive for nonlinear bone voxels;";
  }
  if (invalid_plateau) {
    error << " PlateauStressMPa values must be finite and positive for nonlinear bone voxels;";
  }
}

void HDF5Image::ReadBC(HDF5_GReader &reader, std::string s,std::vector<unsigned short> & coordinates, std::vector<float> & values)
{
  hsize_t global_dims_of_hdf5[3] = {};
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
  delete[] nonlinear_map_E_mpa;
  delete[] nonlinear_map_nu;
  delete[] nonlinear_map_sigma_c_mpa;
  delete[] nonlinear_map_sigma_t_mpa;
  delete[] nonlinear_map_plateau_mpa;
  delete[] nonlinear_map_material_id;
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

  hsize_t global_dims_of_hdf5[3] = {};
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

    hsize_t poisson_dims_of_hdf5[3] = {};
    std::string poisson_dataset = "";
    if (reader.GetSizeOfDataset("Poissons_ratio_Image", poisson_dims_of_hdf5, 3) > 0) {
      poisson_dataset = "Poissons_ratio_Image";
    } else {
      for(int d=0; d<3; d++) {
        poisson_dims_of_hdf5[d] = 0;
      }
      if (reader.GetSizeOfDataset("Poisons_ratio_Image", poisson_dims_of_hdf5, 3) > 0) {
        poisson_dataset = "Poisons_ratio_Image";
      }
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
      int enabled = 0;
      if (reader.AttributeExists("enabled")) {
        reader.ReadAttribute("enabled", enabled);
      }
      nonlinear_enabled = enabled != 0;
      if (nonlinear_enabled) {
        std::ostringstream error;
        if (reader.AttributeExists("material_type")) {
          reader.ReadAttribute("material_type", nonlinear_material_type);
        } else {
          error << " missing material_type;";
        }
        if (reader.AttributeExists("convergence_tolerance")) {
          reader.ReadAttribute("convergence_tolerance", nonlinear_convergence_tolerance);
        }
        if (reader.AttributeExists("maximum_plastic_iterations")) {
          reader.ReadAttribute("maximum_plastic_iterations", nonlinear_maximum_plastic_iterations);
        }
        if (reader.AttributeExists("plastic_convergence_window")) {
          reader.ReadAttribute("plastic_convergence_window", nonlinear_plastic_convergence_window);
        }
        if (!std::isfinite(nonlinear_convergence_tolerance) || nonlinear_convergence_tolerance <= 0.0) {
          error << " convergence_tolerance must be finite and positive;";
        }
        if (nonlinear_maximum_plastic_iterations <= 0) {
          error << " maximum_plastic_iterations must be positive;";
        }
        if (nonlinear_plastic_convergence_window <= 0) {
          error << " plastic_convergence_window must be positive;";
        }
        if (nonlinear_material_type == "AsymmetricPerfectPlasticDensityMap") {
          const char* dataset_names[] = {
            "YoungsModulusMPa",
            "PoissonRatio",
            "CompressiveYieldStressMPa",
            "TensileYieldStressMPa",
            "PlateauStressMPa",
            "MaterialID"
          };
          bool map_datasets_valid = true;
          for (int dataset_index = 0; dataset_index < 6; dataset_index++) {
            hsize_t dataset_dims[3] = {};
            const char* dataset_name = dataset_names[dataset_index];
            int dataset_rank = GetDatasetRankAndDims(_file, std::string("/Nonlinear/") + dataset_name, dataset_dims);
            if (dataset_rank < 0) {
              error << " missing " << dataset_name << ";";
              map_datasets_valid = false;
              continue;
            }
            if (dataset_rank != 3) {
              error << " " << dataset_name << " rank must be 3;";
              map_datasets_valid = false;
              continue;
            }
            for (int d=0; d<3; d++) {
              if (dataset_dims[d] != global_dims_of_hdf5[d]) {
                error << " " << dataset_name << " dimensions do not match Image dimensions;";
                map_datasets_valid = false;
                break;
              }
            }
          }
          if (map_datasets_valid) {
            nonlinear_map_E_mpa = new double[imagesize];
            nonlinear_map_nu = new double[imagesize];
            nonlinear_map_sigma_c_mpa = new double[imagesize];
            nonlinear_map_sigma_t_mpa = new double[imagesize];
            nonlinear_map_plateau_mpa = new double[imagesize];
            nonlinear_map_material_id = new unsigned short[imagesize];
            bool map_reads_valid = true;
            map_reads_valid = ReadRequiredFloatMapDatasetAsDouble(reader, "YoungsModulusMPa", nonlinear_map_E_mpa, imagesize, my_offset, my_count, error) && map_reads_valid;
            map_reads_valid = ReadRequiredFloatMapDatasetAsDouble(reader, "PoissonRatio", nonlinear_map_nu, imagesize, my_offset, my_count, error) && map_reads_valid;
            map_reads_valid = ReadRequiredFloatMapDatasetAsDouble(reader, "CompressiveYieldStressMPa", nonlinear_map_sigma_c_mpa, imagesize, my_offset, my_count, error) && map_reads_valid;
            map_reads_valid = ReadRequiredFloatMapDatasetAsDouble(reader, "TensileYieldStressMPa", nonlinear_map_sigma_t_mpa, imagesize, my_offset, my_count, error) && map_reads_valid;
            map_reads_valid = ReadRequiredFloatMapDatasetAsDouble(reader, "PlateauStressMPa", nonlinear_map_plateau_mpa, imagesize, my_offset, my_count, error) && map_reads_valid;
            map_reads_valid = ReadRequiredMaterialIdDataset(reader, nonlinear_map_material_id, imagesize, my_offset, my_count, error) && map_reads_valid;
            if (map_reads_valid) {
              ValidateAsymmetricMaterialMap(
                grid,
                nonlinear_map_E_mpa,
                nonlinear_map_nu,
                nonlinear_map_sigma_c_mpa,
                nonlinear_map_sigma_t_mpa,
                nonlinear_map_plateau_mpa,
                nonlinear_map_material_id,
                imagesize,
                error);
            }
          }
        } else {
          if (reader.AttributeExists("youngs_modulus_mpa")) {
            reader.ReadAttribute("youngs_modulus_mpa", nonlinear_E_mpa);
          } else {
            error << " missing youngs_modulus_mpa;";
          }
          if (reader.AttributeExists("poisson_ratio")) {
            reader.ReadAttribute("poisson_ratio", nonlinear_nu);
          } else {
            error << " missing poisson_ratio;";
          }
          if (reader.AttributeExists("yield_strength_mpa")) {
            reader.ReadAttribute("yield_strength_mpa", nonlinear_Y_mpa);
          } else {
            error << " missing yield_strength_mpa;";
          }
          if (!std::isfinite(nonlinear_E_mpa) || nonlinear_E_mpa <= 0.0) {
            error << " youngs_modulus_mpa must be finite and positive;";
          }
          if (!std::isfinite(nonlinear_nu) || nonlinear_nu <= -1.0 || nonlinear_nu >= 0.5) {
            error << " poisson_ratio must satisfy -1 < nu < 0.5;";
          }
          if (!std::isfinite(nonlinear_Y_mpa) || nonlinear_Y_mpa <= 0.0) {
            error << " yield_strength_mpa must be finite and positive;";
          }
        }
        nonlinear_config_error = error.str();
      }
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

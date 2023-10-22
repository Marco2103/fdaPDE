#include <cstddef>
#include <gtest/gtest.h> // testing framework
#include <unsupported/Eigen/SparseExtra>

#include "../fdaPDE/core/utils/Symbols.h"
#include "../fdaPDE/core/utils/IO/CSVReader.h"
#include "../fdaPDE/core/FEM/PDE.h"
using fdaPDE::core::FEM::PDE;
#include "../fdaPDE/core/FEM/operators/SpaceVaryingFunctors.h"
using fdaPDE::core::FEM::SpaceVaryingDiffusion;
using fdaPDE::core::FEM::SpaceVaryingAdvection;
#include "core/MESH/Mesh.h"
#include "../fdaPDE/models/regression/MSQRPDE.h"
using fdaPDE::models::MSQRPDE;
#include "../fdaPDE/models/SamplingDesign.h"
#include "../../fdaPDE/models/regression/Distributions.h"

#include "../utils/MeshLoader.h"
using fdaPDE::testing::MeshLoader;
#include "../utils/Constants.h"
using fdaPDE::testing::DOUBLE_TOLERANCE;
#include "../utils/Utils.h"
using fdaPDE::testing::almost_equal;

#include<fstream>
#include<iostream>


/* test 1
   domain:       unit square [0,1] x [0,1]
   sampling:     locations = nodes
   penalization: simple laplacian
   covariates:   no
   BC:           no
   order FE:     1
 */
TEST(SQRPDE, Test1_Laplacian_NonParametric_GeostatisticalAtNodes) {

  // path test  
  std::string path = "/mnt/c/Users/marco/PACS/Project/Code/Cpp/fdaPDE-fork/test/data/models/MSQRPDE/2D_test1"; 
  // define domain and regularizing PDE
  MeshLoader<Mesh2D<>> domain("unit_square_coarse");
  auto L = Laplacian();
  DMatrix<double> u = DMatrix<double>::Zero(domain.mesh.elements()*3, 1);
  PDE problem(domain.mesh, L, u);  // definition of regularizing PDE

  std::cout << "test: here 1" << std::endl; 

  // define statistical model
  const unsigned int h = 3; 
  // use optimal lambda to avoid possible numerical issues
  double lambda = 1.778279*std::pow(0.1, 4);
  std::vector<double> alphas = {0.1, 0.5, 0.9}; 
  MSQRPDE<decltype(problem), fdaPDE::models::GeoStatMeshNodes> model(problem, alphas);
  std::cout << "test: here 1.2" << std::endl; 
  model.setLambdaS(lambda);

  std::cout << "test: here 2" << std::endl; 
  
  // load data from .csv files
  CSVReader<double> reader{};
  CSVFile<double> yFile; // observation file
  yFile = reader.parseFile(path + "/z.csv");
  DMatrix<double> y = yFile.toEigen();

  std::cout << "test: here 3" << std::endl; 

  // set model data
  BlockFrame<double, int> df;
  df.insert(OBSERVATIONS_BLK, y);
  model.setData(df);

  std::cout << "test: here 4" << std::endl; 

  // solve smoothing problem
  model.init();
  std::cout << "test: here 5" << std::endl; 
  model.solve();
  std::cout << "test: here 6" << std::endl; 

  // Save solution
  DMatrix<double> computedF = model.f();
  const static Eigen::IOFormat CSVFormatf(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n");
  std::ofstream filef(path + "/f.csv");
  if (filef.is_open()){
    filef << computedF.format(CSVFormatf);
    filef.close();
  }


}

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
#include "../fdaPDE/models/regression/SQRPDE.h"
using fdaPDE::models::SQRPDE;
#include "../fdaPDE/models/SamplingDesign.h"
using fdaPDE::models::Sampling;
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
   domain:       unit square
   sampling:     locations = nodes
   penalization: simple laplacian
   covariates:   no
   BC:           no
   order FE:     1
   Data generation: as described in data/models/SRPDE/test1/README.md 

 */

TEST(SQRPDE, Test1_Laplacian_NonParametric_GeostatisticalAtNodes) {
  // define domain and regularizing PDE
  MeshLoader<Mesh2D<>> domain("unit_square");
  auto L = Laplacian();
  DMatrix<double> u = DMatrix<double>::Zero(domain.mesh.elements()*3, 1);
  PDE problem(domain.mesh, L, u); // definition of regularizing PDE

  double alpha = 0.1; 
  const std::string alpha_string = "10"; 
  const std::string TestNumber = "1"; 
  // use optimal lambda to avoid possible numerical issues
  double lambda = 0.001;     // from R code 

  SQRPDE<decltype(problem), Sampling::GeoStatMeshNodes> model(problem, alpha);
  model.setLambdaS(lambda);

  // load data from .csv files
  CSVReader<double> reader{};
  CSVFile<double> yFile; // observation file
  yFile = reader.parseFile("data/models/SQRPDE/2D_test" + TestNumber + "/z.csv");
  DMatrix<double> y = yFile.toEigen();

  // set model data
  BlockFrame<double, int> df;
  df.insert(OBSERVATIONS_BLK,  y);
  // df.insert(SPACE_LOCATIONS_BLK, loc);
  model.setData(df);

  // solve smoothing problem
  model.init();     
  model.solve();

  /*   **  test correctness of computed results  **   */
  
  // \Psi matrix (sensible to locations != nodes)
  SpMatrix<double> expectedPsi;
  Eigen::loadMarket(expectedPsi, "data/models/SQRPDE/2D_test" + TestNumber + "/psiR_" + alpha_string + ".mtx");
  SpMatrix<double> computedPsi = model.Psi();
  Eigen::saveMarket(computedPsi, "data/models/SQRPDE/2D_test" + TestNumber + "/psiCpp_" + alpha_string + ".mtx");
  EXPECT_TRUE( almost_equal(expectedPsi, computedPsi) );
  // => OK 
  // Nota: stiamo salvando Psi come symmetric nel formato mtx, anche se in generale non lo è 
          // (in questo caso sì perchè è diagonale)     

  // R0 matrix (discretization of identity operator)
  SpMatrix<double> expectedR0;
  Eigen::loadMarket(expectedR0, "data/models/SQRPDE/2D_test" + TestNumber + "/R0R_" + alpha_string + ".mtx");
  SpMatrix<double> computedR0 = model.R0();
  // EXPECT_TRUE( almost_equal(expectedR0, computedR0) );
  Eigen::saveMarket(computedR0, "data/models/SQRPDE/2D_test" + TestNumber + "/R0Cpp_" + alpha_string + ".mtx");  

  // R1 matrix (discretization of differential operator)
  SpMatrix<double> expectedR1;
  Eigen::loadMarket(expectedR1,  "data/models/SQRPDE/2D_test1/R1R_10.mtx");
  SpMatrix<double> computedR1 = model.R1();
  // EXPECT_TRUE( almost_equal(expectedR1, computedR1) );
  Eigen::saveMarket(computedR1, "data/models/SQRPDE/2D_test" + TestNumber + "/R1Cpp_" + alpha_string + ".mtx");

  // Non-parametric system matrix 
  SpMatrix<double> computedA = model.A(); 
  Eigen::saveMarket(computedA, "data/models/SQRPDE/2D_test" + TestNumber + "/ACpp_" + alpha_string + ".mtx");

  // Pseudo-observations  
  DVector<double> computedPseudo = model.py();
  const static Eigen::IOFormat CSVFormatPseudo(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n");
  std::ofstream filePseudo("data/models/SQRPDE/2D_test1/PseudoCpp_" + alpha_string + ".csv");
  if (filePseudo.is_open()){
    filePseudo << computedPseudo.format(CSVFormatPseudo);
    filePseudo.close();
  }

  // // estimate of spatial field \hat f
  // SpMatrix<double> expectedSolution;
  // Eigen::loadMarket(expectedSolution, "data/models/SQRPDE/2D_test1/fR_10.mtx");
 
  // DMatrix<double> computedF = model.f();
  // std::size_t N = computedF.rows();

  // std::cout << "Almost equal " << std::endl ; 
  // EXPECT_TRUE( almost_equal(DMatrix<double>(expectedSolution).topRows(N), computedF) );



  // Save C++ solution 
  DMatrix<double> computedF = model.f();
  const static Eigen::IOFormat CSVFormatf(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n");
  std::ofstream filef("data/models/SQRPDE/2D_test1/fCpp_" + alpha_string + ".csv");
  if (filef.is_open()){
    filef << computedF.format(CSVFormatf);
    filef.close();
  }





}


// /* test 2
//    domain:       c-shaped
//    sampling:     locations != nodes
//    penalization: simple laplacian
//    covariates:   yes
//    BC:           no
//    order FE:     1
//  */


// TEST(SQRPDE, Test2_Laplacian_SemiParametric_GeostatisticalAtLocations) {
//   // define domain and regularizing PDE
//   MeshLoader<Mesh2D<>> domain("c_shaped");
//   auto L = Laplacian();
//   DMatrix<double> u = DMatrix<double>::Zero(domain.mesh.elements()*3, 1);
//   PDE problem(domain.mesh, L, u); // definition of regularizing PDE

//   // define statistical model
//   CSVReader<double> reader{};
//   // load locations where data are sampled
//   CSVFile<double> locFile;
//   locFile = reader.parseFile("data/models/SQRPDE/2D_test2/locs.csv");
//   DMatrix<double> loc = locFile.toEigen();

//   double alpha = 0.1;
//   // use optimal lambda to avoid possible numerical issues
//   double lambda = 0.1 ; 
//   std::string alpha_string = "01" ; 
//   SQRPDE<decltype(problem), Sampling::GeoStatLocations> model(problem, alpha);
//   model.setLambdaS(lambda);

  
//   // load data from .csv files
//   CSVFile<double> yFile; // observation file
//   yFile = reader.parseFile("data/models/SQRPDE/2D_test2/z.csv");
//   DMatrix<double> y = yFile.toEigen();
//   CSVFile<double> XFile; // design matrix
//   XFile = reader.parseFile("data/models/SQRPDE/2D_test2/X.csv");
//   DMatrix<double> X = XFile.toEigen();

//   // set model data
//   BlockFrame<double, int> df;
//   df.insert(OBSERVATIONS_BLK,  y);
//   df.insert(DESIGN_MATRIX_BLK, X);
//   df.insert(SPACE_LOCATIONS_BLK, loc);
//   model.setData(df);
  
//   // solve smoothing problem
//   model.init();
//   model.solve();

//   /*   **  test correctness of computed results  **   */


//   DMatrix<double> computedF = model.f();
//   std::size_t N = computedF.rows();


//   const static Eigen::IOFormat CSVFormat(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n");
//   std::ofstream file("data/models/SQRPDE/2D_test2/solCpp_" + alpha_string + ".csv");
//   if (file.is_open()){
//     file << computedF.format(CSVFormat);
//     file.close();
//   }

//   DVector<double> computedBeta = model.beta();
//   const static Eigen::IOFormat CSVFormat_beta(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n");
//   std::ofstream file_beta("data/models/SQRPDE/2D_test2/betaCpp_" + alpha_string + ".csv");
//   if (file_beta.is_open()){
//     file_beta << computedBeta.format(CSVFormat_beta);
//     file_beta.close();
//   }


  
// }; 




// /* test 3
//    domain:       unit square [0,1] x [0,1]
//    sampling:     locations = nodes
//    penalization: costant coefficients PDE
//    covariates:   no
//    BC:           no
//    order FE:     1
//  */


// TEST(SQRPDE, Test3_CostantCoefficientsPDE_NonParametric_GeostatisticalAtNodes) {
//   // define domain and regularizing PDE
//   MeshLoader<Mesh2D<>> domain("unit_square");

//   // non unitary diffusion tensor
//   SMatrix<2> K;
//   K << 1,0,0,4;
//   auto L = Laplacian(K); // anisotropic diffusion
  
//   DMatrix<double> u = DMatrix<double>::Zero(domain.mesh.elements()*3, 1);
//   PDE problem(domain.mesh, L, u); // definition of regularizing PDE

//   // define statistical model
//   double lambda = 0.001 ;
//   double alpha = 0.1;
//   std::string alpha_string = "01"; 
//   SQRPDE<decltype(problem), Sampling::GeoStatMeshNodes> model(problem, alpha);
//   model.setLambdaS(lambda);
  
//   // load data from .csv files
//   CSVReader<double> reader{};
//   CSVFile<double> yFile; // observation file
//   yFile = reader.parseFile("data/models/SQRPDE/2D_test3/z.csv");
//   DMatrix<double> y = yFile.toEigen();

//   // set model data
//   BlockFrame<double, int> df;
//   df.insert(OBSERVATIONS_BLK, y);
//   model.setData(df);

//   // solve smoothing problem
//   model.init();
//   model.solve();

//   /*   **  test correctness of computed results  **   */
  
//   // // \Psi matrix
//   // SpMatrix<double> expectedPsi;
//   // Eigen::loadMarket(expectedPsi, "data/models/SRPDE/2D_test3/Psi.mtx");
//   // SpMatrix<double> computedPsi = model.Psi();
//   // EXPECT_TRUE( almost_equal(expectedPsi, computedPsi) );

//   // // R0 matrix (discretization of identity operator)
//   // SpMatrix<double> expectedR0;
//   // Eigen::loadMarket(expectedR0,  "data/models/SRPDE/2D_test3/R0.mtx");
//   // SpMatrix<double> computedR0 = model.R0();
//   // EXPECT_TRUE( almost_equal(expectedR0, computedR0) );
  
//   // // R1 matrix (discretization of differential operator)
//   // SpMatrix<double> expectedR1;
//   // Eigen::loadMarket(expectedR1,  "data/models/SRPDE/2D_test3/R1.mtx");
//   // SpMatrix<double> computedR1 = model.R1();
//   // EXPECT_TRUE( almost_equal(expectedR1, computedR1) );
    
//   // // estimate of spatial field \hat f
//   // SpMatrix<double> expectedSolution;
//   // Eigen::loadMarket(expectedSolution, "data/models/SRPDE/2D_test3/sol.mtx");
//   // DMatrix<double> computedF = model.f();
//   // std::size_t N = computedF.rows();
//   // EXPECT_TRUE( almost_equal(DMatrix<double>(expectedSolution).topRows(N), computedF) );ù


//   DMatrix<double> computedF = model.f();
//   std::size_t N = computedF.rows();
//   const static Eigen::IOFormat CSVFormat(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n");
//   std::ofstream file("data/models/SQRPDE/2D_test3/solCpp_" + alpha_string + ".csv");
//   if (file.is_open()){
//     file << computedF.format(CSVFormat);
//     file.close();
//   }

// }




// TEST 5

// TEST(SQRPDE, Test5_Laplacian_NonParametric_GeostatisticalAtNodes) {
//   // define domain and regularizing PDE
//   MeshLoader<Mesh2D<>> domain("unit_square");
//   auto L = Laplacian();
//   DMatrix<double> u = DMatrix<double>::Zero(domain.mesh.elements()*3, 1);
//   PDE problem(domain.mesh, L, u); // definition of regularizing PDE

//   double alpha = 0.1; 
//   // use optimal lambda to avoid possible numerical issues
//   double lambda = 0.001;
//   std::string alpha_string = "01" ; 
//   SQRPDE<decltype(problem), Sampling::GeoStatMeshNodes> model(problem, alpha);
//   model.setLambdaS(lambda);

//   // load data from .csv files
//   CSVReader<double> reader{};
//   CSVFile<double> yFile; // observation file
//   yFile = reader.parseFile("data/models/SQRPDE/2D_test5/z.csv");
//   DMatrix<double> y = yFile.toEigen();

//   // set model data
//   BlockFrame<double, int> df;
//   df.insert(OBSERVATIONS_BLK,  y);
//   // df.insert(SPACE_LOCATIONS_BLK, loc);
//   model.setData(df);

//   // solve smoothing problem
//   model.init();     
//   model.solve();

//   /*   **  test correctness of computed results  **   */
  
//   // // \Psi matrix (sensible to locations != nodes)
//   // SpMatrix<double> expectedPsi;
//   // Eigen::loadMarket(expectedPsi, "data/models/SQRPDE/2D_test3/Psi.mtx");
//   // SpMatrix<double> computedPsi = model.Psi();
//   // EXPECT_TRUE( almost_equal(expectedPsi, computedPsi) );

//   // // R0 matrix (discretization of identity operator)
//   // SpMatrix<double> expectedR0;
//   // Eigen::loadMarket(expectedR0,  "data/models/SQRPDE/2D_test3/R0.mtx");
//   // SpMatrix<double> computedR0 = model.R0();
//   // EXPECT_TRUE( almost_equal(expectedR0, computedR0) );

//   // // R1 matrix (discretization of differential operator)
//   // SpMatrix<double> expectedR1;
//   // Eigen::loadMarket(expectedR1,  "data/models/SQRPDE/2D_test3/R1.mtx");
//   // SpMatrix<double> computedR1 = model.R1();
//   // EXPECT_TRUE( almost_equal(expectedR1, computedR1) );

//   // estimate of spatial field \hat f

//   // std::cout << "leggo sol " << std::endl ; 
//   // SpMatrix<double> expectedSolution;
//   // Eigen::loadMarket(expectedSolution, "data/models/SQRPDE/2D_test1/sol.mtx");
 
//   DMatrix<double> computedF = model.f();
//   std::size_t N = computedF.rows();

//   const static Eigen::IOFormat CSVFormat(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n");
//   std::ofstream file("data/models/SQRPDE/2D_test5/solCpp_" + alpha_string + ".csv");
//   if (file.is_open()){
//     file << computedF.format(CSVFormat);
//     file.close();
//   }

//   // std::cout << "Save temp " << std::endl ; 
//   // DMatrix<double> temp = DMatrix<double>(expectedSolution).topRows(N) ; 
//   // std::cout << "Almost equal " << std::endl ; 
//   // EXPECT_TRUE( almost_equal(DMatrix<double>(expectedSolution).topRows(N), computedF) );

// }



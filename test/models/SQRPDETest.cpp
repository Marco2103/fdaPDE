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
#include "../../fdaPDE/models/regression/Distributions.h"

#include "../utils/MeshLoader.h"
using fdaPDE::testing::MeshLoader;
#include "../utils/Constants.h"
using fdaPDE::testing::DOUBLE_TOLERANCE;
#include "../utils/Utils.h"
using fdaPDE::testing::almost_equal;

#include "../../fdaPDE/preprocess/InitialConditionEstimator.h"
using fdaPDE::preprocess::InitialConditionEstimator;

#include<fstream>
#include<iostream>


// /* test 1
//    domain:       unit square [0,1] x [0,1]
//    sampling:     locations = nodes
//    penalization: simple laplacian
//    covariates:   no
//    BC:           no
//    order FE:     1
//  */
// TEST(SQRPDE, Test1_Laplacian_NonParametric_GeostatisticalAtNodes) {
//   // define domain and regularizing PDE
//   MeshLoader<Mesh2D<>> domain("unit_square_coarse");
//   auto L = Laplacian();
//   DMatrix<double> u = DMatrix<double>::Zero(domain.mesh.elements()*3, 1);
//   PDE problem(domain.mesh, L, u);  // definition of regularizing PDE

//   // define statistical model
//   // use optimal lambda to avoid possible numerical issues
//   double lambda = 1.778279*std::pow(0.1, 4);
//   double alpha = 0.1; 
//   SQRPDE<decltype(problem), fdaPDE::models::GeoStatMeshNodes> model(problem, alpha);
//   model.setLambdaS(lambda);
  
//   // load data from .csv files
//   CSVReader<double> reader{};
//   CSVFile<double> yFile; // observation file
//   yFile = reader.parseFile("data/models/SQRPDE/2D_test1/z.csv");
//   DMatrix<double> y = yFile.toEigen();

//   // set model data
//   BlockFrame<double, int> df;
//   df.insert(OBSERVATIONS_BLK, y);
//   model.setData(df);

//   // solve smoothing problem
//   model.init();
//   model.solve();

//   /*   **  test correctness of computed results  **   */
//   // estimate of spatial field \hat f
//   SpMatrix<double> expectedSolution;
//   Eigen::loadMarket(expectedSolution,   "data/models/SQRPDE/2D_test1/sol.mtx");
//   DMatrix<double> computedF = model.f();
//   std::size_t N = computedF.rows();
  
//   EXPECT_TRUE( almost_equal(DMatrix<double>(expectedSolution).topRows(N), computedF) );

//   // Eigen::saveMarket(computedF, "data/models/SQRPDE/2D_test1/sol.mtx"); 

// }

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

//   // use optimal lambda to avoid possible numerical issues
//   double alpha = 0.9;
//   double lambda = 3.162277660168379*std::pow(0.1, 4);
//   SQRPDE<decltype(problem), fdaPDE::models::GeoStatLocations> model(problem, alpha);
//   model.setLambdaS(lambda);
//   model.set_spatial_locations(loc);
  
//   // load data from .csv files
//   CSVFile<double> yFile; // observation file
//   yFile = reader.parseFile("data/models/SQRPDE/2D_test2/z.csv");
//   DMatrix<double> y = yFile.toEigen();
//   CSVFile<double> XFile; // design matrix
//   XFile = reader.parseFile("data/models/SQRPDE/2D_test2/X.csv");
//   DMatrix<double> X = XFile.toEigen();

//   // set data
//   BlockFrame<double, int> df;
//   df.insert(OBSERVATIONS_BLK,  y);
//   df.insert(DESIGN_MATRIX_BLK, X);
//   model.setData(df);
  
//   // solve smoothing problem
//   model.init();
//   model.solve();

//   /*   **  test correctness of computed results  **   */

//   // estimate of spatial field \hat f
//   SpMatrix<double> expectedSolution;
//   Eigen::loadMarket(expectedSolution, "data/models/SQRPDE/2D_test2/sol.mtx");
//   DMatrix<double> computedF = model.f();
//   std::size_t N = computedF.rows();
//   EXPECT_TRUE( almost_equal(DMatrix<double>(expectedSolution).topRows(N), computedF) );


//   // estimate of coefficient vector \hat \beta
//   SpMatrix<double> expectedBeta;
//   Eigen::loadMarket(expectedBeta, "data/models/SQRPDE/2D_test2/beta.mtx");
//   DVector<double> computedBeta = model.beta();
//   EXPECT_TRUE( almost_equal(DMatrix<double>(expectedBeta), computedBeta) );

// }


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
//   MeshLoader<Mesh2D<>> domain("unit_square_coarse");

//   // non unitary diffusion tensor
//   SMatrix<2> K;
//   K << 1,0,0,4;
//   auto L = Laplacian(K); // anisotropic diffusion
  
//   DMatrix<double> u = DMatrix<double>::Zero(domain.mesh.elements()*3, 1);
//   PDE problem(domain.mesh, L, u); // definition of regularizing PDE

//   // define statistical model
//   double lambda = 5.623413251903491*pow(0.1,4);
//   double alpha = 0.1; 
//   SQRPDE<decltype(problem), fdaPDE::models::GeoStatMeshNodes> model(problem, alpha);
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
//   // estimate of spatial field \hat f
//   SpMatrix<double> expectedSolution;
//   Eigen::loadMarket(expectedSolution,   "data/models/SQRPDE/2D_test3/sol.mtx");
//   DMatrix<double> computedF = model.f();
//   std::size_t N = computedF.rows();
  
//   EXPECT_TRUE( almost_equal(DMatrix<double>(expectedSolution).topRows(N), computedF) );

// }


// /* test 4
//    domain:       c-shaped
//    sampling:     areal
//    penalization: simple laplacian
//    covariates:   yes
//    BC:           no
//    order FE:     1
//  */
// TEST(SQRPDE, Test4_Laplacian_SemiParametric_Areal) {
//   // define domain and regularizing PDE
//   MeshLoader<Mesh2D<>> domain("c_shaped_areal");
//   auto L = Laplacian();
//   DMatrix<double> u = DMatrix<double>::Zero(domain.mesh.elements()*3, 1);
//   PDE problem(domain.mesh, L, u); // definition of regularizing PDE

//   // define statistical model
//   CSVReader<int> int_reader{};
//   CSVFile<int> arealFile; // incidence matrix for specification of subdomains
//   arealFile = int_reader.parseFile("data/models/SQRPDE/2D_test4/incidence_matrix.csv");
//   DMatrix<int> areal = arealFile.toEigen();

// // use optimal lambda to avoid possible numerical issues
//   double alpha = 0.5;
//   double lambda = 5.623413251903491*std::pow(0.1, 3);
//   SQRPDE<decltype(problem), fdaPDE::models::Areal> model(problem, alpha);
//   model.setLambdaS(lambda);
//   model.set_spatial_locations(areal);

//   // load data from .csv files
//   CSVReader<double> reader{};
//   CSVFile<double> yFile; // observation file
//   yFile = reader.parseFile("data/models/SQRPDE/2D_test4/z.csv");

//   DMatrix<double> y = yFile.toEigen();

//   CSVFile<double> XFile; // design matrix
//   XFile = reader.parseFile("data/models/SQRPDE/2D_test4/X.csv");
//   DMatrix<double> X = XFile.toEigen();

//   // set data
//   BlockFrame<double, int> df;
//   df.insert(OBSERVATIONS_BLK,  y);
//   df.insert(DESIGN_MATRIX_BLK, X);
//   model.setData(df);
  
//   // solve smoothing problem
//   model.init();
//   model.solve();


//   /*   **  test correctness of computed results  **   */

//   // estimate of spatial field \hat f
//   SpMatrix<double> expectedSolution;
//   Eigen::loadMarket(expectedSolution, "data/models/SQRPDE/2D_test4/sol.mtx");
//   DMatrix<double> computedF = model.f();
//   std::size_t N = computedF.rows();
//   EXPECT_TRUE( almost_equal(DMatrix<double>(expectedSolution).topRows(N), computedF) );


//   // estimate of coefficient vector \hat \beta
//   SpMatrix<double> expectedBeta;
//   Eigen::loadMarket(expectedBeta, "data/models/SQRPDE/2D_test4/beta.mtx");
//   DVector<double> computedBeta = model.beta();
//   EXPECT_TRUE( almost_equal(DMatrix<double>(expectedBeta), computedBeta) ); 

// }













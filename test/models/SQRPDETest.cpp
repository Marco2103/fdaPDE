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



/* test 1
   domain:       unit square [1,1] x [1,1]
   sampling:     locations = nodes
   penalization: simple laplacian
   covariates:   no
   BC:           no
   order FE:     1
   time penalization: separable (mass penalization)
 */
TEST(SQTRPDE, Test1_Laplacian_NonParametric_GeostatisticalAtNodes_Separable_Monolithic) {
  // define time domain
  DVector<double> time_mesh;
  time_mesh.resize(11);
  std::size_t i = 0;
  for(double x = 0; x <= 2; x+=0.2, ++i) time_mesh[i] = x;
  
  // define spatial domain and regularizing PDE
  MeshLoader<Mesh2D<>> domain("unit_square_coarse");
  auto L = Laplacian();
  DMatrix<double> u = DMatrix<double>::Zero(domain.mesh.elements()*3*time_mesh.rows(), 1);
  PDE problem(domain.mesh, L, u); // definition of regularizing PDE
  
  // define statistical model
  double alpha = 0.5;
  double lambdaS = 0.01; // smoothing in space
  double lambdaT = 0.01; // smoothing in time
  SQRPDE<decltype(problem), fdaPDE::models::SpaceTimeSeparable, fdaPDE::models::GeoStatMeshNodes,
	 fdaPDE::models::MonolithicSolver> model(problem, time_mesh, alpha);
  model.setLambdaS(lambdaS);
  model.setLambdaT(lambdaT);

  // load data from .csv files
  CSVReader<double> reader{};
  CSVFile<double> yFile; // observation file
  yFile = reader.parseFile("data/models/STRPDE/2D_test1/y.csv");
  DMatrix<double> y = yFile.toEigen();

  // set model data
  BlockFrame<double, int> df;
  df.stack(OBSERVATIONS_BLK, y);
  model.setData(df);
  
  // // solve smoothing problem
  model.init();
  model.solve();

  // //    **  test correctness of computed results  **   
  
  // // // \Psi matrix
  // SpMatrix<double> expectedPsi;
  // Eigen::loadMarket(expectedPsi, "data/models/STRPDE/2D_test1/Psi.mtx");
  // SpMatrix<double> computedPsi = model.Psi();
  // EXPECT_TRUE( almost_equal(expectedPsi, computedPsi) );

  // // R0 matrix (discretization of identity operator)
  // SpMatrix<double> expectedR0;
  // Eigen::loadMarket(expectedR0,  "data/models/STRPDE/2D_test1/R0.mtx");
  // SpMatrix<double> computedR0 = model.R0();
  // EXPECT_TRUE( almost_equal(expectedR0, computedR0) );
  
  // // R1 matrix (discretization of differential operator)
  // SpMatrix<double> expectedR1;
  // Eigen::loadMarket(expectedR1,  "data/models/STRPDE/2D_test1/R1.mtx");
  // SpMatrix<double> computedR1 = model.R1();
  // EXPECT_TRUE( almost_equal(expectedR1, computedR1) );
  
  // // estimate of spatial field \hat f
  // SpMatrix<double> expectedSolution;
  // Eigen::loadMarket(expectedSolution,   "data/models/STRPDE/2D_test1/sol.mtx");
  // DMatrix<double> computedF = model.f();
  // std::size_t N = computedF.rows();
  // EXPECT_TRUE( almost_equal(DMatrix<double>(expectedSolution).topRows(N), computedF) );
}

// con la nostra inizializzazione specifica fa 23 iter
// restituendo i dati ne fa 13



/* test 2
   domain:       c-shaped
   sampling:     locations != nodes
   penalization: simple laplacian
   covariates:   yes
   BC:           no
   order FE:     1
   time penalization: separable (mass penalization)
 */
TEST(STQRPDE, Test2_Laplacian_SemiParametric_GeostatisticalAtLocations_Separable_Monolithic) {
  // define time domain
  DVector<double> time_mesh;
  time_mesh.resize(5);
  for(std::size_t i = 0; i < 5; ++i)
    time_mesh[i] = (fdaPDE::testing::pi/4)*i;

  // define domain and regularizing PDE
  MeshLoader<Mesh2D<>> domain("c_shaped");
  auto L = Laplacian();
  DMatrix<double> u = DMatrix<double>::Zero(domain.mesh.elements()*3, 1);
  PDE problem(domain.mesh, L, u); // definition of regularizing PDE
  // define statistical model
  double lambdaS = 0.01; // smoothing in space
  double lambdaT = 0.01; // smoothing in time
  // load sample position
  CSVReader<double> reader{};
  CSVFile<double> locFile; // locations file
  locFile = reader.parseFile("data/models/STRPDE/2D_test2/locs.csv");
  DMatrix<double> loc = locFile.toEigen();

  // Define model 
  double alpha = 0.5;
  STRPDE<decltype(problem), fdaPDE::models::SpaceTimeSeparable, fdaPDE::models::GeoStatLocations,
	 fdaPDE::models::MonolithicSolver> model(problem, time_mesh, alpha);
  model.setLambdaS(lambdaS);
  model.setLambdaT(lambdaT);
  model.set_spatial_locations(loc);
  
  // load data from .csv files
  CSVFile<double> yFile; // observation file
  yFile = reader.parseFile  ("data/models/STRPDE/2D_test2/y.csv");
  DMatrix<double> y = yFile.toEigen();
  CSVFile<double> XFile; // design matrix
  XFile = reader.parseFile  ("data/models/STRPDE/2D_test2/X.csv");
  DMatrix<double> X = XFile.toEigen();

  // set model data
  BlockFrame<double, int> df;
  df.stack (OBSERVATIONS_BLK,  y);
  df.stack (DESIGN_MATRIX_BLK, X);
  model.setData(df);
  
  // solve smoothing problem
  model.init();
  model.solve();

  //   **  test correctness of computed results  **   
  
  // // \Psi matrix (sensible to locations != nodes)
  // SpMatrix<double> expectedPsi;
  // Eigen::loadMarket(expectedPsi, "data/models/STRPDE/2D_test2/Psi.mtx");
  // SpMatrix<double> computedPsi = model.Psi();
  // EXPECT_TRUE( almost_equal(expectedPsi, computedPsi) );

  // // R0 matrix (discretization of identity operator)
  // SpMatrix<double> expectedR0;
  // Eigen::loadMarket(expectedR0,  "data/models/STRPDE/2D_test2/R0.mtx");
  // SpMatrix<double> computedR0 = model.R0();
  // EXPECT_TRUE( almost_equal(expectedR0, computedR0) );
  
  // // R1 matrix (discretization of differential operator)
  // SpMatrix<double> expectedR1;
  // Eigen::loadMarket(expectedR1,  "data/models/STRPDE/2D_test2/R1.mtx");
  // SpMatrix<double> computedR1 = model.R1();
  // EXPECT_TRUE( almost_equal(expectedR1, computedR1) );

  // // estimate of spatial field \hat f
  // SpMatrix<double> expectedSolution;
  // Eigen::loadMarket(expectedSolution, "data/models/STRPDE/2D_test2/sol.mtx");
  // DMatrix<double> computedF = model.f();
  // std::size_t N = computedF.rows();
  // EXPECT_TRUE( almost_equal(DMatrix<double>(expectedSolution), computedF) );

  // // estimate of coefficient vector \hat \beta
  // SpMatrix<double> expectedBeta;
  // Eigen::loadMarket(expectedBeta, "data/models/STRPDE/2D_test2/beta.mtx");
  DVector<double> computedBeta = model.beta();
  // EXPECT_TRUE( almost_equal(DMatrix<double>(expectedBeta), computedBeta) );
  
  for(std::size_t i = 0; i < computedBeta.size(); ++i)
    std::cout << computedBeta[i] << std::endl ;   

}
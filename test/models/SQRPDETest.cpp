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


/* test 2
   domain:       c-shaped
   sampling:     locations != nodes
   penalization: simple laplacian
   covariates:   yes
   BC:           no
   order FE:     1
   alpha = 0.25 -> lambda_opt = 1
 */
TEST(SQRPDE, Test2_Laplacian_NonParametric_GeostatisticalAtLocations) {
  // define domain and regularizing PDE
  MeshLoader<Mesh2D<>> domain("c_shaped");
  auto L = Laplacian();
  DMatrix<double> u = DMatrix<double>::Zero(domain.mesh.elements()*3, 1);
  PDE problem(domain.mesh, L, u); // definition of regularizing PDE

  std::cout << "domain fatto" << std::endl ; 

  // define statistical model
  CSVReader<double> reader{};
  // load locations where data are sampled
  CSVFile<double> locFile;
  locFile = reader.parseFile("data/models/SQRPDE/2D_test3/locs.csv");
  DMatrix<double> loc = locFile.toEigen();


  std::cout << "reader fatto" << std::endl ; 

  // use optimal lambda to avoid possible numerical issues
  double lambda = 0.2201047 ;
  double alpha = 0.25 ; 
  SQRPDE<decltype(problem), Sampling::GeoStatLocations> model(problem, alpha);
  model.setLambdaS(lambda);

  std::cout << "lambda fatto" << std::endl ; 
  
  // load data from .csv files
  CSVFile<double> yFile; // observation file
  yFile = reader.parseFile  ("data/models/SQRPDE/2D_test3/z.csv");
  DMatrix<double> y = yFile.toEigen();
  CSVFile<double> XFile; // design matrix
  XFile = reader.parseFile  ("data/models/SQRPDE/2D_test3/X.csv");
  DMatrix<double> X = XFile.toEigen();

std::cout << "dati fatto" << std::endl ; 

  // set model data
  BlockFrame<double, int> df;
  df.insert(OBSERVATIONS_BLK,  y);
  df.insert(DESIGN_MATRIX_BLK, X);
  df.insert(SPACE_LOCATIONS_BLK, loc);
  model.setData(df);
  
  std::cout << "risolvi" << std::endl ; 

  // solve smoothing problem
  model.init();     // PROBLEMA

  std::cout << "init fatto" << std::endl ; 

  model.solve();

  std::cout << "risolto" << std::endl ; 

  /*   **  test correctness of computed results  **   */
  
  // // \Psi matrix (sensible to locations != nodes)
  // SpMatrix<double> expectedPsi;
  // Eigen::loadMarket(expectedPsi, "data/models/SQRPDE/2D_test3/Psi.mtx");
  // SpMatrix<double> computedPsi = model.Psi();
  // EXPECT_TRUE( almost_equal(expectedPsi, computedPsi) );

  // // R0 matrix (discretization of identity operator)
  // SpMatrix<double> expectedR0;
  // Eigen::loadMarket(expectedR0,  "data/models/SQRPDE/2D_test3/R0.mtx");
  // SpMatrix<double> computedR0 = model.R0();
  // EXPECT_TRUE( almost_equal(expectedR0, computedR0) );

  // // R1 matrix (discretization of differential operator)
  // SpMatrix<double> expectedR1;
  // Eigen::loadMarket(expectedR1,  "data/models/SQRPDE/2D_test3/R1.mtx");
  // SpMatrix<double> computedR1 = model.R1();
  // EXPECT_TRUE( almost_equal(expectedR1, computedR1) );

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
}

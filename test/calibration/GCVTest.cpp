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
#include "../fdaPDE/models/regression/SRPDE.h"
using fdaPDE::models::SRPDE;
#include "../fdaPDE/models/regression/SQRPDE.h"
using fdaPDE::models::SQRPDE;

#include "../fdaPDE/models/regression/GSRPDE.h"
using fdaPDE::models::GSRPDE;

#include "../fdaPDE/models/SamplingDesign.h"
#include "../fdaPDE/calibration/GCV.h"
using fdaPDE::calibration::GCV;
using fdaPDE::calibration::ExactGCV;
using fdaPDE::calibration::ExactEDF;
using fdaPDE::calibration::StochasticEDF;
#include "../fdaPDE/core/OPT/optimizers/GridOptimizer.h"
using fdaPDE::core::OPT::GridOptimizer;
#include "../fdaPDE/core/OPT/optimizers/Newton.h"
using fdaPDE::core::OPT::NewtonOptimizer;

#include "../utils/MeshLoader.h"
using fdaPDE::testing::MeshLoader;
#include "../utils/Constants.h"
using fdaPDE::testing::DOUBLE_TOLERANCE;
#include "../utils/Utils.h"
using fdaPDE::testing::almost_equal;

#include "../../fdaPDE/preprocess/InitialConditionEstimator.h"
using fdaPDE::preprocess::InitialConditionEstimator;

// for time and memory performances
#include <chrono>
#include <iomanip>
using namespace std::chrono;
#include <unistd.h>
#include <fstream>


// /* test 1
//    domain:       unit square [0,1] x [0,1] (coarse)
//    sampling:     locations = nodes
//    penalization: simple laplacian
//    covariates:   no
//    BC:           no
//    order FE:     1
//    GCV optimization: grid exact
//  */
// TEST(GCV_SRPDE, Test1_Laplacian_NonParametric_GeostatisticalAtNodes_GridExact) {
//   // define domain and regularizing PDE
//   MeshLoader<Mesh2D<>> domain("unit_square_coarse");
//   auto L = Laplacian();
//   DMatrix<double> u = DMatrix<double>::Zero(domain.mesh.elements()*3, 1);
//   PDE problem(domain.mesh, L, u); // definition of regularizing PDE

//   // define statistical model
//   SRPDE<decltype(problem), fdaPDE::models::GeoStatMeshNodes>  model(problem);
  
//   // load data from .csv files
//   CSVReader<double> reader{};
//   CSVFile<double> yFile; // observation file
//   yFile = reader.parseFile("data/models/SRPDE/2D_test6/y.csv");
//   DMatrix<double> y = yFile.toEigen();

//   // set model data
//   BlockFrame<double, int> df;
//   df.insert(OBSERVATIONS_BLK, y);
//   model.setData(df);
//   model.init(); // init model

//   // define grid of lambda values
//   std::vector<SVector<1>> lambdas;
//   for(double x = -6.0; x <= -3.0; x +=0.25) lambdas.push_back(SVector<1>(std::pow(10,x)));
  
//   // define GCV function and optimize
//   GCV<decltype(model), ExactEDF<decltype(model)>> GCV(model);
//   GridOptimizer<1> opt;

//   ScalarField<1, decltype(GCV)> obj(GCV);
//   opt.optimize(obj, lambdas); // optimize gcv field
//   SVector<1> best_lambda = opt.optimum();
  
//   // expected values for q + Tr[S]
//   std::vector<double> expected_edfs = {
//     432.0526271699909557, 425.5738291798880368, 414.9490914902157783, 398.3650445980014752, 374.2000509470916541,
//     341.8926575588438936, 302.6569434589166576, 259.4124363611769581, 215.8693404067796564, 175.3273544830321384,
//     139.8641263839342344, 110.2252857831315538, 86.2049347912456341
//   };
//   for(std::size_t i = 0; i < expected_edfs.size(); ++i)
//     EXPECT_TRUE( almost_equal(expected_edfs[i], GCV.edfs()[i]) );
  
//   // expected value of gcv(\lambda)
//   std::vector<double> expected_gcvs = {
//     0.0400234253534814, 0.0397604316008810, 0.0394198736081135, 0.0391060717299170, 0.0391008105529502,
//     0.0398771699638851, 0.0419425162273298, 0.0456080838211829, 0.0509765872233825, 0.0581903455597975,
//     0.0675560664911411, 0.0793326714839587, 0.0934793416959190
//   };
//   for(std::size_t i = 0; i < expected_edfs.size(); ++i)
//     EXPECT_TRUE( almost_equal(expected_gcvs[i], GCV.values()[i]) );

//   // check optimal lambda
//   EXPECT_TRUE( almost_equal(best_lambda[0], lambdas[4][0]) );
// }

// /* test 2
//    domain:       unit square [0,1] x [0,1] (coarse)
//    sampling:     locations = nodes
//    penalization: simple laplacian
//    covariates:   no
//    BC:           no
//    order FE:     1
//    GCV optimization: grid stochastic
//  */
// TEST(GCV_SRPDE, Test2_Laplacian_NonParametric_GeostatisticalAtNodes_GridStochastic) {
//   // define domain and regularizing PDE
//   MeshLoader<Mesh2D<>> domain("unit_square_coarse");
//   auto L = Laplacian();
//   DMatrix<double> u = DMatrix<double>::Zero(domain.mesh.elements()*3, 1);
//   PDE problem(domain.mesh, L, u); // definition of regularizing PDE

//   // define statistical model
//   SRPDE<decltype(problem), fdaPDE::models::GeoStatMeshNodes>  model(problem);
  
//   // load data from .csv files
//   CSVReader<double> reader{};
//   CSVFile<double> yFile; // observation file
//   yFile = reader.parseFile("data/models/SRPDE/2D_test6/y.csv");
//   DMatrix<double> y = yFile.toEigen();

//   // set model data
//   BlockFrame<double, int> df;
//   df.insert(OBSERVATIONS_BLK, y);
//   model.setData(df);
//   model.init(); // init model

//   // define grid of lambda values
//   std::vector<SVector<1>> lambdas;
//   for(double x = -6.0; x <= -3.0; x +=0.25) lambdas.push_back(SVector<1>(std::pow(10,x)));
  
//   // define GCV calibrator
//   std::size_t seed = 476813;
//   GCV<decltype(model), StochasticEDF<decltype(model)>> GCV(model, 1000, seed);
//   GridOptimizer<1> opt;

//   ScalarField<1, decltype(GCV)> obj(GCV);
//   opt.optimize(obj, lambdas); // optimize gcv field
//   SVector<1> best_lambda = opt.optimum();

//   // expected values for q + Tr[S]
//   std::vector<double> expected_edfs = {
//     432.0405848764650045, 425.5529143800780503, 414.9133694755933561, 398.3056506911505608, 374.1052725793284139,
//     341.7500896866504831, 302.4588492694787192, 259.1628602033034667, 215.5870958975179690, 175.0398481074188624,
//     139.5961352594555080, 109.9912391645740399, 86.0073046035169710 
//   };
//   for(std::size_t i = 0; i < expected_edfs.size(); ++i)
//     EXPECT_TRUE( almost_equal(expected_edfs[i], GCV.edfs()[i]) );  

//   // expected value of gcv(\lambda)
//   std::vector<double> expected_gcvs = {
//     0.0399159071906597, 0.0396528360942592, 0.0393119874506983, 0.0389973432053000, 0.0389900907451345,
//     0.0397626888527567, 0.0418226582836935, 0.0454829731994149, 0.0508490093083462, 0.0580646045230030,
//     0.0674359858974278, 0.0792205235204336, 0.0933752877387227
//   };
//   for(std::size_t i = 0; i < expected_edfs.size(); ++i)
//     EXPECT_TRUE( almost_equal(expected_gcvs[i], GCV.values()[i]) );

//   // check optimal lambda
//   EXPECT_TRUE( almost_equal(best_lambda[0], lambdas[4][0]) );
// }

// /* test 3
//    domain:       c-shaped
//    sampling:     locations != nodes
//    penalization: simple laplacian
//    covariates:   yes
//    BC:           no
//    order FE:     1
//    GCV optimization: grid exact
//  */
// TEST(GCV_SRPDE, Test3_Laplacian_SemiParametric_GeostatisticalAtLocations_GridExact) {
//   // define domain and regularizing PDE
//   MeshLoader<Mesh2D<>> domain("c_shaped");
//   auto L = Laplacian();
//   DMatrix<double> u = DMatrix<double>::Zero(domain.mesh.elements()*3, 1);
//   PDE problem(domain.mesh, L, u); // definition of regularizing PDE

//   // define statistical model
//   CSVReader<double> reader{};
//   // load locations where data are sampled
//   CSVFile<double> locFile;
//   locFile = reader.parseFile("data/models/SRPDE/2D_test2/locs.csv");
//   DMatrix<double> loc = locFile.toEigen();

//   // define statistical model
//   SRPDE<decltype(problem), fdaPDE::models::GeoStatLocations> model(problem);
//   model.set_spatial_locations(loc);
  
//   // load data from .csv files
//   CSVFile<double> yFile; // observation file
//   yFile = reader.parseFile  ("data/models/SRPDE/2D_test2/z.csv");
//   DMatrix<double> y = yFile.toEigen();
//   CSVFile<double> XFile; // design matrix
//   XFile = reader.parseFile  ("data/models/SRPDE/2D_test2/X.csv");
//   DMatrix<double> X = XFile.toEigen();

//   // set model data
//   BlockFrame<double, int> df;
//   df.insert(OBSERVATIONS_BLK,  y);
//   df.insert(DESIGN_MATRIX_BLK, X);
//   model.setData(df);
//   model.init(); // init model

//   // define grid of lambda values
//   std::vector<SVector<1>> lambdas;
//   for(double x = -3.0; x <= 3.0; x +=0.25) lambdas.push_back(SVector<1>(std::pow(10,x)));
  
//   // define GCV calibrator
//   GCV<decltype(model), ExactEDF<decltype(model)>> GCV(model);
//   GridOptimizer<1> opt;
  
//   ScalarField<1, decltype(GCV)> obj(GCV);
//   opt.optimize(obj, lambdas); // optimize gcv field
//   SVector<1> best_lambda = opt.optimum();

//   // expected values for q + Tr[S]
//   std::vector<double> expected_edfs = {
//     109.0637499040006588, 93.2229476278395026, 78.0169165468049499, 64.2577092364702054, 52.3413662612859625,
//      42.3404295978456844, 34.1086127469056137, 27.3954638966633190, 21.9573643522161781, 17.6119343172742404,
//      14.2191625526008725, 11.6385106137848773,  9.7178138350592107,  8.3098853216415982,  7.2842171925382466,
//       6.5281267228208399,  5.9498450377698173,  5.4843127845976953,  5.0921215492618579,  4.7522541472780739,
//       4.4551261509828262,  4.1946327397326151,  3.9615593551368464,  3.7465168923635441,  3.5489031263616644
//   };
//   for(std::size_t i = 0; i < expected_edfs.size(); ++i)
//     EXPECT_TRUE( almost_equal(expected_edfs[i], GCV.edfs()[i]) );  

//   // expected value of gcv(\lambda)
//   std::vector<double> expected_gcvs = {
//     2.1771484290002601, 2.0318749168846373, 1.9229411153254741, 1.8376099983427667, 1.7683487774209516,
//     1.7103300463087854, 1.6607101851375488, 1.6184324970362900, 1.5835055354976433, 1.5560348751711195,
//     1.5355021206088371, 1.5209371474466595, 1.5116339095670082, 1.5071872569869649, 1.5067140204400502,
//     1.5086445845177732, 1.5116269898476760, 1.5153441560837564, 1.5210973148544640, 1.5339264639588630,
//     1.5682200504833339, 1.6582436157224418, 1.8695409528491944, 2.2884361440774796, 2.9586790292370440
//   };
//   for(std::size_t i = 0; i < expected_edfs.size(); ++i)
//     EXPECT_TRUE( almost_equal(expected_gcvs[i], GCV.values()[i]) );

//   // check optimal lambda
//   EXPECT_TRUE( almost_equal(best_lambda[0], lambdas[14][0]) );
// }

// /* test 4
//    domain:       c-shaped
//    sampling:     locations != nodes
//    penalization: simple laplacian
//    covariates:   yes
//    BC:           no
//    order FE:     1
//    GCV optimization: grid stochastic
//  */
// TEST(GCV_SRPDE, Test4_Laplacian_SemiParametric_GeostatisticalAtLocations_GridStochastic) {
//   // define domain and regularizing PDE
//   MeshLoader<Mesh2D<>> domain("c_shaped");
//   auto L = Laplacian();
//   DMatrix<double> u = DMatrix<double>::Zero(domain.mesh.elements()*3, 1);
//   PDE problem(domain.mesh, L, u); // definition of regularizing PDE

//   // define statistical model
//   CSVReader<double> reader{};
//   // load locations where data are sampled
//   CSVFile<double> locFile;
//   locFile = reader.parseFile("data/models/SRPDE/2D_test2/locs.csv");
//   DMatrix<double> loc = locFile.toEigen();

//   // define statistical model
//   SRPDE<decltype(problem), fdaPDE::models::GeoStatLocations> model(problem);
//   model.set_spatial_locations(loc);
  
//   // load data from .csv files
//   CSVFile<double> yFile; // observation file
//   yFile = reader.parseFile  ("data/models/SRPDE/2D_test2/z.csv");
//   DMatrix<double> y = yFile.toEigen();
//   CSVFile<double> XFile; // design matrix
//   XFile = reader.parseFile  ("data/models/SRPDE/2D_test2/X.csv");
//   DMatrix<double> X = XFile.toEigen();

//   // set model data
//   BlockFrame<double, int> df;
//   df.insert(OBSERVATIONS_BLK,  y);
//   df.insert(DESIGN_MATRIX_BLK, X);
//   model.setData(df);
//   model.init(); // init model

//   // define grid of lambda values
//   std::vector<SVector<1>> lambdas;
//   for(double x = -3.0; x <= 3.0; x +=0.25) lambdas.push_back(SVector<1>(std::pow(10, x)));
  
//   // define GCV calibrator
//   std::size_t seed = 66546513;
//   GCV<decltype(model), StochasticEDF<decltype(model)>> GCV(model, 100, seed);
//   GridOptimizer<1> opt;

//   ScalarField<1, decltype(GCV)> obj(GCV);
//   opt.optimize(obj, lambdas); // optimize gcv field
//   SVector<1> best_lambda = opt.optimum();

//   // expected values for q + Tr[S] (approximated)
//   std::vector<double> expected_edfs = {
//     106.2709775815650062, 91.2781878154271595, 76.8216314734006431, 63.6368232668402314, 52.0874830058305562,
//     42.2619783483174274,  34.0612617815139203, 27.2926608538180062, 21.7650509877889817, 17.3309390327612363,
//     13.8629820650085431,  11.2197970258596094,  9.2470064712368139,  7.7975062657152465,  6.7408531721908309,
//     5.9632650796519151,    5.3718179354249518,  4.9012488716219842,  4.5124258626382030,  4.1837705547537087,
//     3.9038225985944632,    3.6643245426561473,  3.4547903696710578,  3.2650834389510122,  3.0930648940657273
//   };
//   for(std::size_t i = 0; i < expected_edfs.size(); ++i)
//     EXPECT_TRUE( almost_equal(expected_edfs[i], GCV.edfs()[i]) );  

//   // expected value of gcv(\lambda)
//   std::vector<double> expected_gcvs = {
//     1.9933325455157656, 1.9339516163142583, 1.8744400183178187, 1.8164174958658492, 1.7608058393404040,
//     1.7082461763764853, 1.6595618959678919, 1.6161177888901777, 1.5794269592455561, 1.5503494275186216,
//     1.5285490034180831, 1.5129761785234856, 1.5028470225931159, 1.4977383948589298, 1.4967621840119401,
//     1.4983351571813046, 1.5010945226728243, 1.5047240675196598, 1.5105230912061263, 1.5234895155928254,
//     1.5578890056928336, 1.6477498970873763, 1.8582485673557088, 2.2753184385488714, 2.9426362338294938
//   };  
//   for(std::size_t i = 0; i < expected_edfs.size(); ++i)
//     EXPECT_TRUE( almost_equal(expected_gcvs[i], GCV.values()[i]) );

//   // check optimal lambda
//   EXPECT_TRUE( almost_equal(best_lambda[0], lambdas[14][0]) );
// }



// /* test 5
//    domain:       unit square [0,1] x [0,1]
//    sampling:     locations = nodes
//    penalization: costant coefficients PDE
//    covariates:   no
//    BC:           no
//    order FE:     1
//    GCV optimization: grid exact
//  */
// TEST(GCV_SRPDE, Test5_CostantCoefficientsPDE_NonParametric_GeostatisticalAtNodes_GridExact) {
//   // define domain and regularizing PDE
//   MeshLoader<Mesh2D<>> domain("unit_square_coarse");

//   // non unitary diffusion tensor
//   SMatrix<2> K;
//   K << 1,0,0,4;
//   auto L = Laplacian(K); // anisotropic diffusion
  
//   DMatrix<double> u = DMatrix<double>::Zero(domain.mesh.elements()*3, 1);
//   PDE problem(domain.mesh, L, u); // definition of regularizing PDE

//   // define statistical model
//   SRPDE<decltype(problem), fdaPDE::models::GeoStatMeshNodes> model(problem);
  
//   // load data from .csv files
//   CSVReader<double> reader{};
//   CSVFile<double> yFile; // observation file
//   yFile = reader.parseFile("data/models/SRPDE/2D_test5/y.csv");
//   DMatrix<double> y = yFile.toEigen();

//   // set model data
//   BlockFrame<double, int> df;
//   df.insert(OBSERVATIONS_BLK, y);
//   model.setData(df);
//   model.init(); // init model

//   // define grid of lambda values
//   std::vector<SVector<1>> lambdas;
//   for(double x = -6.0; x <= -3.0; x +=0.25) lambdas.push_back(SVector<1>(std::pow(10,x)));
  
//   // define GCV calibrator
//   GCV<decltype(model), ExactEDF<decltype(model)>> GCV(model);
//   GridOptimizer<1> opt;

//   ScalarField<1, decltype(GCV)> obj(GCV);
//   opt.optimize(obj, lambdas); // optimize gcv field
//   SVector<1> best_lambda = opt.optimum();

//   // expected values for q + Tr[S]
//   std::vector<double> expected_edfs = {
//     392.6223660935990551, 366.9539394059336246, 334.2897566350799252, 296.6918970050114126, 257.1195078756908288,
//     218.2528178022757857, 181.8976825480112609, 149.0838748306272237, 120.3659604356742108,  95.9678116102015224, 
//      75.7957057139429935,  59.4825236929897869,  46.5013015724528103
//   };
//   for(std::size_t i = 0; i < expected_edfs.size(); ++i)
//     EXPECT_TRUE( almost_equal(expected_edfs[i], GCV.edfs()[i]) );  

//   // expected value of gcv(\lambda)  
//   std::vector<double> expected_gcvs = {
//     0.4401244042041300, 0.4129157240364682, 0.3816562739955814, 0.3494124806447785, 0.3189922357544511,
//     0.2922854246010580, 0.2703666651130021, 0.2539506267229450, 0.2437506624632450, 0.2405042252882551,
//     0.2449930879669074, 0.2586192765896194, 0.2850382613491984
//   };
//   for(std::size_t i = 0; i < expected_edfs.size(); ++i)
//     EXPECT_TRUE( almost_equal(expected_gcvs[i], GCV.values()[i]) );  

//   // check optimal lambda
//   EXPECT_TRUE( almost_equal(best_lambda[0], lambdas[9][0]) );  
// }

// /* test 6
//    domain:       unit square [0,1] x [0,1]
//    sampling:     locations = nodes
//    penalization: costant coefficients PDE
//    covariates:   no
//    BC:           no
//    order FE:     1
//    GCV optimization: grid stochastic
//  */
// TEST(GCV_SRPDE, Test6_CostantCoefficientsPDE_NonParametric_GeostatisticalAtNodes_GridStochastic) {
//   // define domain and regularizing PDE
//   MeshLoader<Mesh2D<>> domain("unit_square_coarse");

//   // non unitary diffusion tensor
//   SMatrix<2> K;
//   K << 1,0,0,4;
//   auto L = Laplacian(K); // anisotropic diffusion
  
//   DMatrix<double> u = DMatrix<double>::Zero(domain.mesh.elements()*3, 1);
//   PDE problem(domain.mesh, L, u); // definition of regularizing PDE

//   // define statistical model
//   SRPDE<decltype(problem), fdaPDE::models::GeoStatMeshNodes> model(problem);
  
//   // load data from .csv files
//   CSVReader<double> reader{};
//   CSVFile<double> yFile; // observation file
//   yFile = reader.parseFile("data/models/SRPDE/2D_test5/y.csv"); // load file for unit square coarse!
//   DMatrix<double> y = yFile.toEigen();

//   // set model data
//   BlockFrame<double, int> df;
//   df.insert(OBSERVATIONS_BLK, y);
//   model.setData(df);
//   model.init(); // init model

//   // define grid of lambda values
//   std::vector<SVector<1>> lambdas;
//   for(double x = -6.0; x <= -3.0; x +=0.25) lambdas.push_back(SVector<1>(std::pow(10,x)));
  
//   // define GCV calibrator
//   std::size_t seed = 4564168;
//   GCV<decltype(model), StochasticEDF<decltype(model)>> GCV(model, 100, seed);
//   GridOptimizer<1> opt;

//   ScalarField<1, decltype(GCV)> obj(GCV);
//   opt.optimize(obj, lambdas); // optimize gcv field
//   SVector<1> best_lambda = opt.optimum();
  
//   // expected values for q + Tr[S]
//   std::vector<double> expected_edfs = {
//     392.6398736612755442, 367.0053593585058707, 334.3960328473687582, 296.8543902422412657, 257.2935036770556962,
//     218.3463104599331643, 181.8107241424921483, 148.7611248424874759, 119.8187842113097616,  95.2531545500184080,
//      74.9790711890389048,  58.6159568240670694,  45.6216944905244262
//   };
//   for(std::size_t i = 0; i < expected_edfs.size(); ++i)
//     EXPECT_TRUE( almost_equal(expected_edfs[i], GCV.edfs()[i]) );  

//   // expected value of gcv(\lambda)  
//   std::vector<double> expected_gcvs = {
//     0.4404431338340472, 0.4134898057270393, 0.3824176190452916, 0.3502006998245064, 0.3195967826322993, 
//     0.2925309384160517, 0.2701852786717965, 0.2533900079360062, 0.2429208445928721, 0.2395110106902127, 
//     0.2439010923516004, 0.2574484277620766, 0.2837714099478065
//   };
//   for(std::size_t i = 0; i < expected_edfs.size(); ++i)
//     EXPECT_TRUE( almost_equal(expected_gcvs[i], GCV.values()[i]) );  

//   // check optimal lambda
//   EXPECT_TRUE( almost_equal(best_lambda[0], lambdas[9][0]) );  
// }

// /* test 7
//    domain:       quasicircular domain
//    sampling:     areal
//    penalization: non-costant coefficients PDE
//    covariates:   no
//    BC:           yes
//    order FE:     1
//    GCV optimization: grid exact
//  */
// TEST(GCV_SRPDE, Test7_NonCostantCoefficientsPDE_NonParametric_Areal_GridExact) {
//   // define domain and regularizing PDE
//   MeshLoader<Mesh2D<>> domain("quasi_circle");

//   // load PDE coefficients data
//   CSVReader<double> reader{};
//   CSVFile<double> diffFile; // diffusion tensor
//   diffFile = reader.parseFile("data/models/SRPDE/2D_test4/K.csv");
//   DMatrix<double> diffData = diffFile.toEigen();
//   CSVFile<double> adveFile; // transport vector
//   adveFile = reader.parseFile("data/models/SRPDE/2D_test4/b.csv");
//   DMatrix<double> adveData = adveFile.toEigen();
  
//   // define non-constant coefficients
//   SpaceVaryingDiffusion<2> diffCoeff;
//   diffCoeff.setData(diffData);
//   SpaceVaryingAdvection<2> adveCoeff;
//   adveCoeff.setData(adveData);

//   auto L = Laplacian(diffCoeff.asParameter()) + Gradient(adveCoeff.asParameter());
  
//   // load non-zero forcing term
//   CSVFile<double> forceFile; // transport vector
//   forceFile = reader.parseFile("data/models/SRPDE/2D_test4/force.csv");
//   DMatrix<double> u = forceFile.toEigen();
  
//   PDE problem(domain.mesh, L, u); // definition of regularizing PDE
  
//   // define statistical model
//   CSVReader<int> int_reader{};
//   CSVFile<int> arealFile; // incidence matrix for specification of subdomains
//   arealFile = int_reader.parseFile("data/models/SRPDE/2D_test4/incidence_matrix.csv");
//   DMatrix<int> areal = arealFile.toEigen();

//   double lambda = std::pow(0.1, 3);
//   SRPDE<decltype(problem), fdaPDE::models::Areal> model(problem);
//   model.set_spatial_locations(areal);
  
//   // load data from .csv files
//   CSVFile<double> yFile; // observation file
//   yFile = reader.parseFile("data/models/SRPDE/2D_test4/z.csv");
//   DMatrix<double> y = yFile.toEigen();
  
//   // set model data
//   BlockFrame<double, int> df;
//   df.insert(OBSERVATIONS_BLK, y);
//   model.setData(df);
//   model.init(); // init model

//   // define grid of lambda values
//   std::vector<SVector<1>> lambdas;
//   for(double x = -6.0; x <= -3.0; x +=0.25) lambdas.push_back(SVector<1>(std::pow(10,x)));
  
//   // define GCV calibrator
//   GCV<decltype(model), ExactEDF<decltype(model)>> GCV(model);
//   GridOptimizer<1> opt;

//   ScalarField<1, decltype(GCV)> obj(GCV);
//   opt.optimize(obj, lambdas); // optimize gcv field
//   SVector<1> best_lambda = opt.optimum();
  
//   // expected values for q + Tr[S]
//   std::vector<double> expected_edfs = {
//     6.8256470417592974, 6.7027363460970051, 6.5065235219756685, 6.2115932502007984, 5.8013394965992671,
//     5.2785614057558456, 4.6679680479060641, 4.0086692884299344, 3.3453578380021134, 2.7250862599643653,
//     2.1926047404607063, 1.7772541551903058, 1.4821993609506263
//   };
//   for(std::size_t i = 0; i < expected_edfs.size(); ++i)
//     EXPECT_TRUE( almost_equal(expected_edfs[i], GCV.edfs()[i]) );  

//   // expected value of gcv(\lambda)  
//   std::vector<double> expected_gcvs = {
//     24.1288371239571298, 23.3331732915000316, 22.1169607470995366, 20.4274688418074035, 18.3836687947684716,
//     16.3298382592033455, 14.7125701729663341, 13.8561091376951602, 13.8455543587040868, 14.5546522388024950,
//     15.7134846701210957, 16.9881160092813985, 18.1037530114315466
//   };
//   for(std::size_t i = 0; i < expected_gcvs.size(); ++i)
//     EXPECT_TRUE( almost_equal(expected_gcvs[i], GCV.values()[i]) );  
  
//   EXPECT_TRUE( almost_equal(best_lambda[0], lambdas[8][0]) );
// }

// /* test 8
//    domain:       quasicircular domain
//    sampling:     areal
//    penalization: non-costant coefficients PDE
//    covariates:   no
//    BC:           yes
//    order FE:     1
//    GCV optimization: grid stochastic
//  */
// TEST(GCV_SRPDE, Test8_NonCostantCoefficientsPDE_NonParametric_Areal_GridStochastic) {
//   // define domain and regularizing PDE
//   MeshLoader<Mesh2D<>> domain("quasi_circle");

//   // load PDE coefficients data
//   CSVReader<double> reader{};
//   CSVFile<double> diffFile; // diffusion tensor
//   diffFile = reader.parseFile("data/models/SRPDE/2D_test4/K.csv");
//   DMatrix<double> diffData = diffFile.toEigen();
//   CSVFile<double> adveFile; // transport vector
//   adveFile = reader.parseFile("data/models/SRPDE/2D_test4/b.csv");
//   DMatrix<double> adveData = adveFile.toEigen();
  
//   // define non-constant coefficients
//   SpaceVaryingDiffusion<2> diffCoeff;
//   diffCoeff.setData(diffData);
//   SpaceVaryingAdvection<2> adveCoeff;
//   adveCoeff.setData(adveData);

//   auto L = Laplacian(diffCoeff.asParameter()) + Gradient(adveCoeff.asParameter());
  
//   // load non-zero forcing term
//   CSVFile<double> forceFile; // transport vector
//   forceFile = reader.parseFile("data/models/SRPDE/2D_test4/force.csv");
//   DMatrix<double> u = forceFile.toEigen();
  
//   PDE problem(domain.mesh, L, u); // definition of regularizing PDE
  
//   // define statistical model
//   CSVReader<int> int_reader{};
//   CSVFile<int> arealFile; // incidence matrix for specification of subdomains
//   arealFile = int_reader.parseFile("data/models/SRPDE/2D_test4/incidence_matrix.csv");
//   DMatrix<int> areal = arealFile.toEigen();

//   double lambda = std::pow(0.1, 3);
//   SRPDE<decltype(problem), fdaPDE::models::Areal> model(problem);
//   model.set_spatial_locations(areal);
  
//   // load data from .csv files
//   CSVFile<double> yFile; // observation file
//   yFile = reader.parseFile("data/models/SRPDE/2D_test4/z.csv");
//   DMatrix<double> y = yFile.toEigen();
  
//   // set model data
//   BlockFrame<double, int> df;
//   df.insert(OBSERVATIONS_BLK, y);
//   model.setData(df);
//   model.init(); // init model

//   // define grid of lambda values
//   std::vector<SVector<1>> lambdas;
//   for(double x = -6.0; x <= -3.0; x +=0.25) lambdas.push_back(SVector<1>(std::pow(10,x)));
  
//   // define GCV calibrator
//   std::size_t seed = 438172;
//   GCV<decltype(model), StochasticEDF<decltype(model)>> GCV(model, 100, seed);
//   GridOptimizer<1> opt;

//   ScalarField<1, decltype(GCV)> obj(GCV);
//   opt.optimize(obj, lambdas); // optimize gcv field
//   SVector<1> best_lambda = opt.optimum();

//   std::vector<double> expected_edfs = {
//     6.831047684097598, 6.712363557176861, 6.523569437275610, 6.241165650575311, 5.850412256000765,
//     5.354051019972657, 4.772559035126907, 4.136996797446094, 3.484589099615463, 2.860439790874401,
//     2.313681605547809, 1.880681758930778, 1.569948378164815
//   };
//   for(std::size_t i = 0; i < expected_edfs.size(); ++i)
//     EXPECT_TRUE( almost_equal(expected_edfs[i], GCV.edfs()[i]) );  

//   // expected value of gcv(\lambda)  
//   std::vector<double> expected_gcvs = {
//     25.6960716321667775, 24.9212379291962769, 23.7278907064885978, 22.0506431657932183, 19.9866647378768612,
//     17.8620876450079429, 16.1266862014873773, 15.1260832893288590, 14.9640061955584951, 15.5220163816125858,
//     16.5359284471161168, 17.6814853738732261, 18.6935898436080734
//   };
//   for(std::size_t i = 0; i < expected_edfs.size(); ++i)
//     EXPECT_TRUE( almost_equal(expected_gcvs[i], GCV.values()[i]) );  

//   // check optimal lambda is correct
//   EXPECT_TRUE( almost_equal(best_lambda[0], lambdas[8][0]) );  
// }


// // GCV test for SQRPDE
// // -------------------

// /* test 9
//    domain:       unit square [0,1] x [0,1] (coarse)
//    sampling:     locations = nodes
//    penalization: simple laplacian
//    covariates:   no
//    BC:           no
//    order FE:     1
//    GCV optimization: grid exact
//  */
// TEST(GCV_SQRPDE, Test9_Laplacian_NonParametric_GeostatisticalAtNodes_GridExact) {
//   // define domain and regularizing PDE
//   MeshLoader<Mesh2D<>> domain("unit_square_coarse");
//   auto L = Laplacian();
//   DMatrix<double> u = DMatrix<double>::Zero(domain.mesh.elements()*3, 1);
//   PDE problem(domain.mesh, L, u); // definition of regularizing PDE

//   // define statistical model
//   SRPDE<decltype(problem), fdaPDE::models::GeoStatMeshNodes>  model(problem);
  
//   // load data from .csv files
//   CSVReader<double> reader{};
//   CSVFile<double> yFile; // observation file
//   yFile = reader.parseFile("data/models/SQRPDE/2D_test1/z.csv");
//   DMatrix<double> y = yFile.toEigen();

//   // set model data
//   BlockFrame<double, int> df;
//   df.insert(OBSERVATIONS_BLK, y);
//   model.setData(df);
//   model.init(); // init model

//   // define grid of lambda values
//   std::vector<SVector<1>> lambdas;
//   for(double x = -6.0; x <= -3.0; x +=0.25) lambdas.push_back(SVector<1>(std::pow(10,x)));
  
//   // define GCV function and optimize
//   GCV<decltype(model), ExactEDF<decltype(model)>> GCV(model);
//   GridOptimizer<1> opt;

//   ScalarField<1, decltype(GCV)> obj(GCV);
//   opt.optimize(obj, lambdas); // optimize gcv field
//   SVector<1> best_lambda = opt.optimum();
  
//   // expected values for q + Tr[S]
//   std::vector<double> expected_edfs = {
//     432.052627169991, 425.573829179888, 414.9490914902158, 398.3650445980015, 
//     374.2000509470916, 341.8926575588438, 302.6569434589166, 259.4124363611769,
//     215.8693404067796, 175.3273544830321, 139.8641263839342, 110.2252857831316,
//     86.20493479124558
//   };
//   for(std::size_t i = 0; i < expected_edfs.size(); ++i)
//     EXPECT_TRUE( almost_equal(expected_edfs[i], GCV.edfs()[i]) );
  
//   // expected value of gcv(\lambda)
//   std::vector<double> expected_gcvs = {
//     0.2889341077416062, 0.2854077925496184, 0.2798584776184149, 0.2717592794338346, 0.2611294135474913,
//     0.2489481972514338, 0.2370314032770059, 0.2273015740624733, 0.2211327405250494, 0.2192751843189231,
//     0.2221581616314486, 0.2302753758381074, 0.244551719201662
//   };
//   for(std::size_t i = 0; i < expected_edfs.size(); ++i)
//     EXPECT_TRUE( almost_equal(expected_gcvs[i], std::sqrt(GCV.values()[i])) );

//   // check optimal lambda
//   EXPECT_TRUE( almost_equal(best_lambda[0], lambdas[9][0]) );
     
// }


// /* test 10
//    domain:       unit square [0,1] x [0,1] (coarse)
//    sampling:     locations = nodes
//    penalization: simple laplacian
//    covariates:   no
//    BC:           no
//    order FE:     1
//    GCV optimization: grid stochastic
//  */
// TEST(GCV_SQRPDE, Test10_Laplacian_NonParametric_GeostatisticalAtNodes_GridStochastic) {
//   // define domain and regularizing PDE
//   MeshLoader<Mesh2D<>> domain("unit_square_coarse");
//   auto L = Laplacian();
//   DMatrix<double> u = DMatrix<double>::Zero(domain.mesh.elements()*3, 1);
//   PDE problem(domain.mesh, L, u); // definition of regularizing PDE

//   // define statistical model
//   SRPDE<decltype(problem), fdaPDE::models::GeoStatMeshNodes>  model(problem);
  
//   // load data from .csv files
//   CSVReader<double> reader{};
//   CSVFile<double> yFile; // observation file
//   yFile = reader.parseFile("data/models/SQRPDE/2D_test1/z.csv");
//   DMatrix<double> y = yFile.toEigen();

//   // set model data
//   BlockFrame<double, int> df;
//   df.insert(OBSERVATIONS_BLK, y);
//   model.setData(df);
//   model.init(); // init model

//   // define grid of lambda values
//   std::vector<SVector<1>> lambdas;
//   for(double x = -6.0; x <= -3.0; x +=0.25) lambdas.push_back(SVector<1>(std::pow(10,x)));
  
//   // define GCV calibrator
//   std::size_t seed = 438172;
//   GCV<decltype(model), StochasticEDF<decltype(model)>> GCV(model, 1000, seed);
//   GridOptimizer<1> opt;

//   ScalarField<1, decltype(GCV)> obj(GCV);
//   opt.optimize(obj, lambdas); // optimize gcv field
//   SVector<1> best_lambda = opt.optimum();
  
//   // expected values for q + Tr[S]
//   std::vector<double> expected_edfs = {
//     432.0698631231691, 425.6028136550763, 414.9960871141209, 398.4372916256007, 374.3037338038849,
//     342.0307832172483, 302.8294646252283, 259.6178491516016, 216.1035329093824, 175.5801638935219,
//     140.1198645690421, 110.4687739069256, 86.4267002235808
//   };
//   for(std::size_t i = 0; i < expected_edfs.size(); ++i)
//     EXPECT_TRUE( almost_equal(expected_edfs[i], GCV.edfs()[i]) );
  
//   // expected value of gcv(\lambda)
//   std::vector<double> expected_gcvs = {
//     0.289491776097791, 0.2859450592226583, 0.2803642524154052, 0.2722205705713464, 0.2615353529670946,
//     0.2492956399691977, 0.2373273631503242, 0.2275589899679398, 0.2213630137284524, 0.2194840414672656,
//     0.2223469887369042, 0.2304450097933664, 0.2447046723929239
//   };
//   for(std::size_t i = 0; i < expected_edfs.size(); ++i)
//     EXPECT_TRUE( almost_equal(expected_gcvs[i], std::sqrt(GCV.values()[i])) );

//   // check optimal lambda
//   EXPECT_TRUE( almost_equal(best_lambda[0], lambdas[9][0]) );
     
// }


// /* test 11
//    domain:       c-shaped
//    sampling:     locations != nodes
//    penalization: simple laplacian
//    covariates:   yes
//    BC:           no
//    order FE:     1
//    GCV optimization: grid exact
//  */
// TEST(GCV_SQRPDE, Test11_Laplacian_SemiParametric_GeostatisticalAtLocations_GridExact) {
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

//   // define statistical model
//   double alpha = 0.9;
//   SQRPDE<decltype(problem), fdaPDE::models::GeoStatLocations> model(problem, alpha);
//   model.set_spatial_locations(loc);
  
//   // load data from .csv files
//   CSVFile<double> yFile; // observation file
//   yFile = reader.parseFile  ("data/models/SQRPDE/2D_test2/z.csv");
//   DMatrix<double> y = yFile.toEigen();
//   CSVFile<double> XFile; // design matrix
//   XFile = reader.parseFile  ("data/models/SQRPDE/2D_test2/X.csv");
//   DMatrix<double> X = XFile.toEigen();

//   // set model data
//   BlockFrame<double, int> df;
//   df.insert(OBSERVATIONS_BLK,  y);
//   df.insert(DESIGN_MATRIX_BLK, X);
//   model.setData(df);
//   model.init(); // init model

//   // define grid of lambda values
//   std::vector<SVector<1>> lambdas;
//   for(double x = -6.0; x <= -2.0; x +=0.25) lambdas.push_back(SVector<1>(std::pow(10,x)));
  
//   // define GCV calibrator
//   GCV<decltype(model), ExactEDF<decltype(model)>> GCV(model);
//   GridOptimizer<1> opt;
  
//   ScalarField<1, decltype(GCV)> obj(GCV);
//   opt.optimize(obj, lambdas); // optimize gcv field
//   SVector<1> best_lambda = opt.optimum();

//   // expected values for q + Tr[S]
//   std::vector<double> expected_edfs = {
//     187.7253599931921, 179.2755565628839, 167.6420689172974, 151.536953215778, 
//     132.9609167899539, 115.6188521061802, 96.98364501070189, 80.85716309709031, 
//     67.0846022368506, 55.63929214088965, 45.06689308847258, 36.37777081975081, 
//     30.23149565775936, 25.17175285716757, 21.42956032669428, 17.93979438721523, 13.65791375051949
//   };
//   for(std::size_t i = 0; i < expected_edfs.size(); ++i)
//     EXPECT_TRUE( almost_equal(expected_edfs[i], GCV.edfs()[i]) );  

//   // expected value of gcv(\lambda)
//   std::vector<double> expected_gcvs = {
//     0.207113480685508, 0.1619999542307615, 0.1349081332158126, 0.1175998787554854, 
//     0.1048199410568254, 0.09740880446646295, 0.09481460961310464, 0.09467290640026461, 
//     0.0934387481846385, 0.0921480812228711, 0.0907388945712925, 0.09097376907415111, 
//     0.09186989268419447, 0.09213936903056989, 0.09267690820632558, 0.09285169875977703, 0.09299131594187338
//   };
//   for(std::size_t i = 0; i < expected_edfs.size(); ++i)
//     EXPECT_TRUE( almost_equal(expected_gcvs[i], std::sqrt(GCV.values()[i])) );

//   // check optimal lambda
//   EXPECT_TRUE( almost_equal(best_lambda[0], lambdas[10][0]) );
 
// }

// /* test 12
//    domain:       c-shaped
//    sampling:     locations != nodes
//    penalization: simple laplacian
//    covariates:   yes
//    BC:           no
//    order FE:     1
//    GCV optimization: grid stochastic
//  */
// TEST(GCV_SQRPDE, Test12_Laplacian_SemiParametric_GeostatisticalAtLocations_GridStochastic) {
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

//   // define statistical model
//   double alpha = 0.9;
//   SQRPDE<decltype(problem), fdaPDE::models::GeoStatLocations> model(problem, alpha);
//   model.set_spatial_locations(loc);
  
//   // load data from .csv files
//   CSVFile<double> yFile; // observation file
//   yFile = reader.parseFile  ("data/models/SQRPDE/2D_test2/z.csv");
//   DMatrix<double> y = yFile.toEigen();
//   CSVFile<double> XFile; // design matrix
//   XFile = reader.parseFile  ("data/models/SQRPDE/2D_test2/X.csv");
//   DMatrix<double> X = XFile.toEigen();

//   // set model data
//   BlockFrame<double, int> df;
//   df.insert(OBSERVATIONS_BLK,  y);
//   df.insert(DESIGN_MATRIX_BLK, X);
//   model.setData(df);
//   model.init(); // init model

//   // define grid of lambda values
//   std::vector<SVector<1>> lambdas;
//   for(double x = -6.0; x <= -2.0; x +=0.25) lambdas.push_back(SVector<1>(std::pow(10, x)));
  
//   // define GCV calibrator
//   std::size_t seed = 66546513;
//   GCV<decltype(model), StochasticEDF<decltype(model)>> GCV(model, 1000, seed);
//   GridOptimizer<1> opt;

//   ScalarField<1, decltype(GCV)> obj(GCV);
//   opt.optimize(obj, lambdas); // optimize gcv field
//   SVector<1> best_lambda = opt.optimum();

//   // expected values for q + Tr[S]
//   std::vector<double> expected_edfs = {
//     187.6952072153096, 179.215083383998, 167.6087832370042, 151.4695023652123, 
//     132.8436017261902, 115.4993621499878, 96.88454935643702, 80.68461366031806, 
//     66.80508876832693, 55.33985331850447, 44.76154075634351, 36.23318157143425, 
//     30.27186405859233, 25.21371158992702, 21.37865765199833, 17.78354313825074, 13.32681446752576
//   };
//   for(std::size_t i = 0; i < expected_edfs.size(); ++i)
//     EXPECT_TRUE( almost_equal(expected_edfs[i], GCV.edfs()[i]) );  

//   // expected value of gcv(\lambda)
//   std::vector<double> expected_gcvs = {
//     0.2065074474629024, 0.1614784372918274, 0.1347603764140301, 0.1174294053616538, 
//     0.1046312111946083, 0.0972677222387157, 0.09472168893962025, 0.09453365990254047, 
//     0.09323967500143035, 0.09195466551730594, 0.09055808269672189, 0.09089245555748103, 
//     0.09189200370708005, 0.09216174378725546, 0.09265019850948182, 0.09277119449200742, 0.09282459245170223
//   };
//   for(std::size_t i = 0; i < expected_edfs.size(); ++i)
//     EXPECT_TRUE( almost_equal(expected_gcvs[i], std::sqrt(GCV.values()[i])) );

//   // check optimal lambda
//   EXPECT_TRUE( almost_equal(best_lambda[0], lambdas[10][0]) );

// }


// /* test 13
//    domain:       unit square [0,1] x [0,1]
//    sampling:     locations = nodes
//    penalization: costant coefficients PDE
//    covariates:   no
//    BC:           no
//    order FE:     1
//    GCV optimization: grid exact
//  */
// TEST(GCV_SQRPDE, Test13_ConstantCoefficientsPDE_NonParametric_GeostatisticalAtNodes_GridExact) {
//   // define domain and regularizing PDE
//   MeshLoader<Mesh2D<>> domain("unit_square_coarse");

//   // non unitary diffusion tensor
//   SMatrix<2> K;
//   K << 1,0,0,4;
//   auto L = Laplacian(K); // anisotropic diffusion
  
//   DMatrix<double> u = DMatrix<double>::Zero(domain.mesh.elements()*3, 1);
//   PDE problem(domain.mesh, L, u); // definition of regularizing PDE

//   // define statistical model
//   SRPDE<decltype(problem), fdaPDE::models::GeoStatMeshNodes> model(problem);
  
//   // load data from .csv files
//   CSVReader<double> reader{};
//   CSVFile<double> yFile; // observation file
//   yFile = reader.parseFile("data/models/SQRPDE/2D_test3/z.csv");
//   DMatrix<double> y = yFile.toEigen();

//   // set model data
//   BlockFrame<double, int> df;
//   df.insert(OBSERVATIONS_BLK, y);
//   model.setData(df);
//   model.init(); // init model

//   // define grid of lambda values
//   std::vector<SVector<1>> lambdas;
//   for(double x = -5.0; x <= -2.0; x +=0.25) lambdas.push_back(SVector<1>(std::pow(10,x)));
  
//   // define GCV calibrator
//   GCV<decltype(model), ExactEDF<decltype(model)>> GCV(model);
//   GridOptimizer<1> opt;

//   ScalarField<1, decltype(GCV)> obj(GCV);
//   opt.optimize(obj, lambdas); // optimize gcv field
//   SVector<1> best_lambda = opt.optimum();

//   // expected values for q + Tr[S]
//   std::vector<double> expected_edfs = {
//     257.1195078756906, 218.2528178022757, 181.8976825480112,
//     149.0838748306272, 120.3659604356742, 95.96781161020145,
//     75.79570571394295, 59.48252369298979, 46.50130157245282,
//     36.28422736255408, 28.30032513213107, 22.08993878058206, 17.27217825537496
//   };
//   for(std::size_t i = 0; i < expected_edfs.size(); ++i)
//     EXPECT_TRUE( almost_equal(expected_edfs[i], GCV.edfs()[i]) );  

//   // expected value of gcv(\lambda)  
//   std::vector<double> expected_gcvs = {
//     0.3617697736910833, 0.3530773702466216, 0.3465155454930449,
//     0.342036051955046, 0.3390389023648164, 0.3368769368239065,
//     0.3354060682628043, 0.3353428882992653, 0.3388896270899528,
//     0.3514774903223452, 0.384994822686168, 0.4601618283038805, 0.602603367160696
//   };
//   for(std::size_t i = 0; i < expected_edfs.size(); ++i)
//     EXPECT_TRUE( almost_equal(expected_gcvs[i], std::sqrt(GCV.values()[i])) );  

//   // check optimal lambda
//   EXPECT_TRUE( almost_equal(best_lambda[0], lambdas[7][0]) );  

// }


// /* test 14
//    domain:       unit square [0,1] x [0,1]
//    sampling:     locations = nodes
//    penalization: costant coefficients PDE
//    covariates:   no
//    BC:           no
//    order FE:     1
//    GCV optimization: grid stochastic
//  */
// TEST(GCV_SQRPDE, Test14_ConstantCoefficientsPDE_NonParametric_GeostatisticalAtNodes_GridStochastic) {
//   // define domain and regularizing PDE
//   MeshLoader<Mesh2D<>> domain("unit_square_coarse");

//   // non unitary diffusion tensor
//   SMatrix<2> K;
//   K << 1,0,0,4;
//   auto L = Laplacian(K); // anisotropic diffusion
  
//   DMatrix<double> u = DMatrix<double>::Zero(domain.mesh.elements()*3, 1);
//   PDE problem(domain.mesh, L, u); // definition of regularizing PDE

//   // define statistical model
//   SRPDE<decltype(problem), fdaPDE::models::GeoStatMeshNodes> model(problem);
  
//   // load data from .csv files
//   CSVReader<double> reader{};
//   CSVFile<double> yFile; // observation file
//   yFile = reader.parseFile("data/models/SQRPDE/2D_test3/z.csv");
//   DMatrix<double> y = yFile.toEigen();

//   // set model data
//   BlockFrame<double, int> df;
//   df.insert(OBSERVATIONS_BLK, y);
//   model.setData(df);
//   model.init(); // init model

//   // define grid of lambda values
//   std::vector<SVector<1>> lambdas;
//   for(double x = -5.0; x <= -2.0; x +=0.25) lambdas.push_back(SVector<1>(std::pow(10,x)));
  
//   // define GCV calibrator
//   std::size_t seed = 438172;
//   GCV<decltype(model), StochasticEDF<decltype(model)>> GCV(model, 1000, seed);
//   GridOptimizer<1> opt;
  
//   ScalarField<1, decltype(GCV)> obj(GCV);
//   opt.optimize(obj, lambdas); // optimize gcv field
//   SVector<1> best_lambda = opt.optimum();

//   // expected values for q + Tr[S]
//   std::vector<double> expected_edfs = {
//     257.4909456984228, 218.6404154907185, 182.2674333064996,
//     149.4091206601632, 120.6366321614015, 96.18961500118381,
//     75.98125382333421, 59.6433782941554, 46.64535157474903,
//     36.41611304077789, 28.42181838856386, 22.19978875573882, 17.36673238404154
//   };
//   for(std::size_t i = 0; i < expected_edfs.size(); ++i)
//     EXPECT_TRUE( almost_equal(expected_edfs[i], GCV.edfs()[i]) );  

//   // expected value of gcv(\lambda)  
//   std::vector<double> expected_gcvs = {
//     0.3625020262634784, 0.3536928236926742, 0.3470107455655037,
//     0.3424175652578667, 0.3393253528582386, 0.3370936369297677,
//     0.3355765635113262, 0.3354843345033716, 0.3390134168100827,
//     0.3515920644531638, 0.385108193380021, 0.460282527300801, 0.6027378670705686
//   };
//   for(std::size_t i = 0; i < expected_edfs.size(); ++i)
//     EXPECT_TRUE( almost_equal(expected_gcvs[i], std::sqrt(GCV.values()[i])) );  

//   // check optimal lambda
//   EXPECT_TRUE( almost_equal(best_lambda[0], lambdas[7][0]) );  

// }


// /* test 15
//    domain:       c-shaped
//    sampling:     areal
//    penalization: simple laplacian
//    covariates:   yes
//    BC:           no
//    order FE:     1
//    GCV optimization: grid exact
//  */
// TEST(GCV_SQRPDE, Test15_Laplacian_SemiParametric_Areal_GridExact) {
//   // define domain and regularizing PDE
//   MeshLoader<Mesh2D<>> domain("c_shaped_areal");
//   auto L = Laplacian();
//   DMatrix<double> u = DMatrix<double>::Zero(domain.mesh.elements()*3, 1);
//   PDE problem(domain.mesh, L, u); // definition of regularizing PDE

//   // define statistical model
//   double alpha = 0.5;

//   CSVReader<int> int_reader{};
//   CSVFile<int> arealFile; // incidence matrix for specification of subdomains
//   arealFile = int_reader.parseFile("data/models/SQRPDE/2D_test4/incidence_matrix.csv");
//   DMatrix<int> areal = arealFile.toEigen();

//   SQRPDE<decltype(problem), fdaPDE::models::Areal> model(problem, alpha);
//   model.set_spatial_locations(areal);
  
//   // load data from .csv files
//   CSVReader<double> reader{};
//   CSVFile<double> yFile; // observation file
//   yFile = reader.parseFile  ("data/models/SQRPDE/2D_test4/z.csv");
//   DMatrix<double> y = yFile.toEigen();
//   CSVFile<double> XFile; // design matrix
//   XFile = reader.parseFile  ("data/models/SQRPDE/2D_test4/X.csv");
//   DMatrix<double> X = XFile.toEigen();

//   // set model data
//   BlockFrame<double, int> df;
//   df.insert(OBSERVATIONS_BLK,  y);
//   df.insert(DESIGN_MATRIX_BLK, X);
//   model.setData(df);
//   model.init(); // init model

//   // define grid of lambda values
//   std::vector<SVector<1>> lambdas;
//   for(double x = -4.0; x <= -1.0; x +=0.25) lambdas.push_back(SVector<1>(std::pow(10,x)));
  
//   // define GCV calibrator
//   GCV<decltype(model), ExactEDF<decltype(model)>> GCV(model);
//   GridOptimizer<1> opt;
  
//   ScalarField<1, decltype(GCV)> obj(GCV);
//   opt.optimize(obj, lambdas); // optimize gcv field
//   SVector<1> best_lambda = opt.optimum();

//   // expected values for q + Tr[S]
//   std::vector<double> expected_edfs = {
//     15.5665805247258, 14.57206645250153, 12.07731267732268, 10.81888046946576, 
//     9.132700712396884, 8.905677185957018, 7.543159447321961, 6.728320474488777, 
//     6.102418202565106, 4.577356496044276, 4.056257902018482, 3.456352438552989, 2.524958204290085
//   };
//   for(std::size_t i = 0; i < expected_edfs.size(); ++i)
//     EXPECT_TRUE( almost_equal(expected_edfs[i], GCV.edfs()[i]) );  

//   // expected value of gcv(\lambda)
//   std::vector<double> expected_gcvs = {
//     0.2700544903239684, 0.2501916455113311, 0.2062436415274343, 0.1959783033624303, 
//     0.1773834460441135, 0.178788524640947, 0.1658405913836493, 0.162394199833477, 
//     0.1757835636865077, 0.2174451166320289, 0.3091663357975384, 0.5086198705658231, 0.7375918496256139
//   };
//   for(std::size_t i = 0; i < expected_edfs.size(); ++i)
//     EXPECT_TRUE( almost_equal(expected_gcvs[i], std::sqrt(GCV.values()[i])) );

//   // check optimal lambda
//   EXPECT_TRUE( almost_equal(best_lambda[0], lambdas[7][0]) );

// }

// /* test 16
//    domain:       c-shaped
//    sampling:     areal
//    penalization: simple laplacian
//    covariates:   yes
//    BC:           no
//    order FE:     1
//    GCV optimization: grid stochastic
//  */
// TEST(GCV_SQRPDE, Test16_Laplacian_SemiParametric_Areal_GridStochastic) {
//   // define domain and regularizing PDE
//   MeshLoader<Mesh2D<>> domain("c_shaped_areal");
//   auto L = Laplacian();
//   DMatrix<double> u = DMatrix<double>::Zero(domain.mesh.elements()*3, 1);
//   PDE problem(domain.mesh, L, u); // definition of regularizing PDE

//   // define statistical model
//   double alpha = 0.5;

//   CSVReader<int> int_reader{};
//   CSVFile<int> arealFile; // incidence matrix for specification of subdomains
//   arealFile = int_reader.parseFile("data/models/SQRPDE/2D_test4/incidence_matrix.csv");
//   DMatrix<int> areal = arealFile.toEigen();

//   SQRPDE<decltype(problem), fdaPDE::models::Areal> model(problem, alpha);
//   model.set_spatial_locations(areal);
  
//   // load data from .csv files
//   CSVReader<double> reader{};
//   CSVFile<double> yFile; // observation file
//   yFile = reader.parseFile  ("data/models/SQRPDE/2D_test4/z.csv");
//   DMatrix<double> y = yFile.toEigen();
//   CSVFile<double> XFile; // design matrix
//   XFile = reader.parseFile  ("data/models/SQRPDE/2D_test4/X.csv");
//   DMatrix<double> X = XFile.toEigen();

//   // set model data
//   BlockFrame<double, int> df;
//   df.insert(OBSERVATIONS_BLK,  y);
//   df.insert(DESIGN_MATRIX_BLK, X);
//   model.setData(df);
//   model.init(); // init model

//   // define grid of lambda values
//   std::vector<SVector<1>> lambdas;
//   for(double x = -4.0; x <= -1.0; x +=0.25) lambdas.push_back(SVector<1>(std::pow(10,x)));
  
//   // define GCV calibrator
//   std::size_t seed = 438172;
//   GCV<decltype(model), StochasticEDF<decltype(model)>> GCV(model, 100, seed);
//   GridOptimizer<1> opt;

//   ScalarField<1, decltype(GCV)> obj(GCV);
//   opt.optimize(obj, lambdas); // optimize gcv field
//   SVector<1> best_lambda = opt.optimum();

//   // expected values for q + Tr[S]
//   std::vector<double> expected_edfs = {
//     15.67597974463673, 14.72866022659128, 12.33514848654242, 11.15275933010177, 
//     9.444495374763706, 9.149456717766643, 7.443647450043284, 7.096969631945466, 
//     6.638288840960764, 5.167892242081829, 4.0219954504947, 4.698181394281461, 3.348548069419884
//   };
//   for(std::size_t i = 0; i < expected_edfs.size(); ++i)
//     EXPECT_TRUE( almost_equal(expected_edfs[i], GCV.edfs()[i]) );  

//   // expected value of gcv(\lambda)
//   std::vector<double> expected_gcvs = {
//     0.2768869630762044, 0.2576239977596116, 0.2131814140487006, 0.203374170060016, 
//     0.182623101904484, 0.1828053726176995, 0.1645262663507382, 0.1670339226921229, 
//     0.182833353146178, 0.22610262615585, 0.308503374627373, 0.5498972441252941, 0.7740735435035143
//   };
//   for(std::size_t i = 0; i < expected_edfs.size(); ++i)
//     EXPECT_TRUE( almost_equal(expected_gcvs[i], std::sqrt(GCV.values()[i])) );

//   // check optimal lambda
//   EXPECT_TRUE( almost_equal(best_lambda[0], lambdas[6][0]) );

// }




// // GCV test for STQRPDE   ---> non va 
// // -------------------

// /* test 17
//    domain:       unit square 
//    sampling:     locations != nodes 
//    penalization: laplacian 
//    covariates:   yes
//    BC:           no
//    order FE:     1
//    time penalization: parabolic (monolithic solution)
//    GCV optimization: grid exact
//  */
// TEST(GCV_SQRPDE, Test17_Laplacian_SemiParametric_GeoStatLocations_GridExact) {
//   // define time domain, we skip the first time instant because we are going to use the first block of data
//   // for the estimation of the initial condition
//   DVector<double> time_mesh;
//   time_mesh.resize(11);
//   for(std::size_t i = 0; i < 10; ++i) time_mesh[i] = 0.4*i;

//   // define domain and regularizing PDE
//   MeshLoader<Mesh2D<>> domain("unit_square_coarse");
  
//   // parabolic PDE
//   auto L = dT() + Laplacian();
  
//   DMatrix<double> u = DMatrix<double>::Zero(domain.mesh.elements()*3, time_mesh.rows()); 
//   PDE problem(domain.mesh, L, u); // definition of regularizing PDE
//   // define statistical model
//   double lambdaT = std::pow(0.1, 6); // smoothing in time
//   double alpha = 0.5; 
//   SQRPDE<decltype(problem), fdaPDE::models::SpaceTimeParabolic, fdaPDE::models::GeoStatLocations,
//         fdaPDE::models::MonolithicSolver> 
//       model(problem, time_mesh, alpha);
//   model.setLambdaT(lambdaT);

//   // load sample position
//   CSVReader<double> reader{};
//   CSVFile<double> locFile; // locations file
//   locFile = reader.parseFile("data/models/STQRPDE/2D_test4/locs.csv");
//   DMatrix<double> loc = locFile.toEigen();
//   model.set_spatial_locations(loc);

//   // load data from .csv files
//   CSVFile<double> yFile; // observation file
//   yFile = reader.parseFile  ("data/models/STQRPDE/2D_test4/y.csv");
//   DMatrix<double> y = yFile.toEigen();

//   CSVFile<double> XFile; // design matrix
//   XFile = reader.parseFile("data/models/STQRPDE/2D_test4/X.csv");
//   DMatrix<double> X = XFile.toEigen();

//   // set model data
//   BlockFrame<double, int> df;
//   df.stack(OBSERVATIONS_BLK, y);
//   df.stack(DESIGN_MATRIX_BLK, X);
//   model.setData(df);

//   model.init(); // init model
  
//   // define initial condition estimator over grid of lambdas
//   InitialConditionEstimator ICestimator(model);
//   std::vector<SVector<1>> lambdas_IC;
//   for(double x = -9; x <= 3; x += 0.1) lambdas_IC.push_back(SVector<1>(std::pow(10,x))); 
//   // compute estimate
//   std::cout << "Computing IC..." << std::endl; 
//   ICestimator.apply(lambdas_IC);
//   DMatrix<double> ICestimate = ICestimator.get();
//   std::cout << "End IC computation..." << std::endl; 

//   // set estimated initial condition
//   model.setInitialCondition(ICestimate);
//   model.shift_time(1); // shift data one time instant forward


//   // define grid of lambda values
//   std::vector<SVector<1>> lambdas;
//   for(double x = -8.0; x <= -2.0; x +=1) lambdas.push_back(SVector<1>(std::pow(10,x)));
  
//   // define GCV calibrator
//   GCV<decltype(model), ExactEDF<decltype(model)>> GCV(model);
//   GridOptimizer<1> opt;
  
//   ScalarField<1, decltype(GCV)> obj(GCV);
//   opt.optimize(obj, lambdas); // optimize gcv field
//   SVector<1> best_lambda = opt.optimum();

//   const std::string path_test = "/mnt/c/Users/marco/PACS/Project/Code/Cpp/fdaPDE-fork/test/data/models/STQRPDE/2D_test4";
//   // Save best lambda 
//   const static Eigen::IOFormat CSVFormatl(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n");
//   std::ofstream filel(path_test + "/lambda_exact.csv");
//   if(filel.is_open()){
//     filel << best_lambda.format(CSVFormatl);
//     filel.close();
//   }

// }



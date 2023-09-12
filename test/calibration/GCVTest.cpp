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
#include "../fdaPDE/models/SamplingDesign.h"
#include "../fdaPDE/calibration/GCV.h"
using fdaPDE::calibration::GCV;
using fdaPDE::calibration::ExactGCV;
using fdaPDE::calibration::ExactEDF;
using fdaPDE::calibration::StochasticEDF;
using fdaPDE::calibration::StochasticEDFMethod;
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

// M: for time and memory performances
#include <chrono>
#include <iomanip>
using namespace std::chrono;
#include <unistd.h>
#include <fstream>

// credits to https://www.tutorialspoint.com/how-to-get-memory-usage-at-runtime-using-cplusplus
void mem_usage(double& vm_usage, double& resident_set){
  vm_usage = 0.0;
  resident_set = 0.0;
  std::ifstream stat_stream("/proc/self/stat", std::ios_base::in); //get info from proc directory
  //create some variables to get info
  std::string pid, comm, state, ppid, pgrp, session, tty_nr;
  std::string tpgid, flags, minflt, cminflt, majflt, cmajflt;
  std::string utime, stime, cutime, cstime, priority, nice;
  std::string O, itrealvalue, starttime;
  unsigned long vsize;
  long rss;
  stat_stream >> pid >> comm >> state >> ppid >> pgrp >> session >> tty_nr
  >> tpgid >> flags >> minflt >> cminflt >> majflt >> cmajflt
  >> utime >> stime >> cutime >> cstime >> priority >> nice
  >> O >> itrealvalue >> starttime >> vsize >> rss; // don't care  about the rest
  stat_stream.close();
  long page_size_kb = sysconf(_SC_PAGE_SIZE) / 1024; // for x86-64 is configured to use 2MB pages
  vm_usage = vsize / 1024.0;
  resident_set = rss * page_size_kb;
}


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


/* test 4   (MODIFIED)
   domain:       c-shaped
   sampling:     locations != nodes
   penalization: simple laplacian
   covariates:   yes
   BC:           no
   order FE:     1
   GCV optimization: grid stochastic
 */
//TEST(GCV_SRPDE, Test4_Laplacian_SemiParametric_GeostatisticalAtLocations_GridStochastic) {

//   // // TEST (Wood,Wood) VS (Wood,Chol): accuracy 

//   // const std::string TestNumber = "2"; 

//   // // Marco
//   // std::string R_path = "/mnt/c/Users/marco/OneDrive - Politecnico di Milano/Corsi/Magistrale/Anno_II_Semestre_II/PACS_project_shared"; 
  
//   // // Ilenia 
//   // // std::string R_path = "/mnt/c/Users/ileni/OneDrive - Politecnico di Milano/PACS_project_shared";

//   // std::string GCV_type = "Stochastic";  

//   // auto gcv_enum = fdaPDE::calibration::StochasticEDFMethod::CholeskyGCV;  
//   // std::string GCV_lin_sys_solver;      
//   // if(gcv_enum) 
//   //   GCV_lin_sys_solver = "Cholesky"; 
//   // else
//   //   GCV_lin_sys_solver = "Woodbury";  

//   // const std::string nnodes = "";  
  
//   // std::string path_solutions = R_path + "/R/Our/data/SRPDE/Test_" + TestNumber + "/accuracy"; 
//   // std::string path_GCV = path_solutions + "/GCV/" + GCV_type + "/" + GCV_lin_sys_solver; 

//   // // define domain and regularizing PDE
//   // MeshLoader<Mesh2D<>> domain("c_shaped");
//   // auto L = Laplacian();
//   // DMatrix<double> u = DMatrix<double>::Zero(domain.mesh.elements()*3, 1);
//   // PDE problem(domain.mesh, L, u); // definition of regularizing PDE

//   // // define statistical model
//   // CSVReader<double> reader{};
//   // // load locations where data are sampled
//   // CSVFile<double> locFile;
//   // locFile = reader.parseFile("data/models/SRPDE/2D_test2/locs.csv");
//   // DMatrix<double> loc = locFile.toEigen();

//   // // define statistical model
//   // SRPDE<decltype(problem), fdaPDE::models::GeoStatLocations> model(problem);
//   // model.set_spatial_locations(loc);
  
//   // // load data from .csv files
//   // CSVFile<double> yFile; // observation file
//   // yFile = reader.parseFile("data/models/SRPDE/2D_test2/z.csv");
//   // DMatrix<double> y = yFile.toEigen();
//   // CSVFile<double> XFile; // design matrix
//   // XFile = reader.parseFile("data/models/SRPDE/2D_test2/X.csv");
//   // DMatrix<double> X = XFile.toEigen();

//   // // set model data
//   // BlockFrame<double, int> df;
//   // df.insert(OBSERVATIONS_BLK,  y);
//   // df.insert(DESIGN_MATRIX_BLK, X);
//   // model.setData(df);
//   // model.init(); // init model

//   // // define grid of lambda values
//   // std::vector<SVector<1>> lambdas;
//   // for(double x = -3.0; x <= 3.0; x +=0.25) lambdas.push_back(SVector<1>(std::pow(10, x)));
  
//   // // define GCV calibrator
//   // std::size_t seed = 66546513;
//   // GCV<decltype(model), StochasticEDF<decltype(model)>> GCV(model, 100, seed, gcv_enum);
//   // GridOptimizer<1> opt;

//   // ScalarField<1, decltype(GCV)> obj(GCV);
//   // opt.optimize(obj, lambdas);     // optimize gcv field
//   // SVector<1> best_lambda = opt.optimum();

//   // // Save results 

//   // // Lambda opt
//   // std::ofstream fileLambdaopt(path_solutions + "/LambdaCpp_" + GCV_lin_sys_solver + ".csv");
//   // if(fileLambdaopt.is_open()){
//   //   fileLambdaopt << std::setprecision(16) << best_lambda[0];
//   //   fileLambdaopt.close();
//   // }

//   // // GCV scores
//   // std::ofstream fileGCV_scores(path_GCV + "/GCV_scoresCpp.csv");
//   // if(fileGCV_scores.is_open()){
//   //   for(std::size_t i = 0; i < GCV.values().size(); ++i) 
//   //     fileGCV_scores << std::setprecision(16) << std::sqrt(GCV.values()[i]) << "\n";

//   //   fileGCV_scores.close();  
//   // }

//   // // expected values for q + Tr[S] (approximated)
//   // std::vector<double> expected_edfs = {
//   //   106.2709775815650062, 91.2781878154271595, 76.8216314734006431, 63.6368232668402314, 52.0874830058305562,
//   //   42.2619783483174274,  34.0612617815139203, 27.2926608538180062, 21.7650509877889817, 17.3309390327612363,
//   //   13.8629820650085431,  11.2197970258596094,  9.2470064712368139,  7.7975062657152465,  6.7408531721908309,
//   //   5.9632650796519151,    5.3718179354249518,  4.9012488716219842,  4.5124258626382030,  4.1837705547537087,
//   //   3.9038225985944632,    3.6643245426561473,  3.4547903696710578,  3.2650834389510122,  3.0930648940657273
//   // };
//   // for(std::size_t i = 0; i < expected_edfs.size(); ++i)
//   //   EXPECT_TRUE( almost_equal(expected_edfs[i], GCV.edfs()[i]) );  

//   // // expected value of gcv(\lambda)
//   // std::vector<double> expected_gcvs = {
//   //   1.9933325455157656, 1.9339516163142583, 1.8744400183178187, 1.8164174958658492, 1.7608058393404040,
//   //   1.7082461763764853, 1.6595618959678919, 1.6161177888901777, 1.5794269592455561, 1.5503494275186216,
//   //   1.5285490034180831, 1.5129761785234856, 1.5028470225931159, 1.4977383948589298, 1.4967621840119401,
//   //   1.4983351571813046, 1.5010945226728243, 1.5047240675196598, 1.5105230912061263, 1.5234895155928254,
//   //   1.5578890056928336, 1.6477498970873763, 1.8582485673557088, 2.2753184385488714, 2.9426362338294938
//   // };  
//   // for(std::size_t i = 0; i < expected_edfs.size(); ++i)
//   //   EXPECT_TRUE( almost_equal(expected_gcvs[i], GCV.values()[i]) );

//   // // check optimal lambda
//   // EXPECT_TRUE( almost_equal(best_lambda[0], lambdas[14][0]) );



  // TEST (Wood,Wood) VS (Wood,Chol): performance

//   const std::string TestNumber = "2"; 

//   // Marco
//   std::string R_path = "/mnt/c/Users/marco/OneDrive - Politecnico di Milano/Corsi/Magistrale/Anno_II_Semestre_II/PACS_project_shared";  
//   // Ilenia 
//   // std::string R_path = "/mnt/c/Users/ileni/OneDrive - Politecnico di Milano/PACS_project_shared"; 
//   std::string GCV_type = "Stochastic"; 



//   const std::string nnodes = "11011";      // 742 1452 2789 5502 11011 22086 44136                  
//   unsigned int launch_sim = 5; 
//   auto gcv_enum = fdaPDE::calibration::StochasticEDFMethod::CholeskyGCV;  // WoodburyGCV CholeskyGCV 
  


//   unsigned int MC_nsim = 1000;          // number of Monte Carlo simulations
//   std::string GCV_lin_sys_solver;      
//   if(gcv_enum) 
//     GCV_lin_sys_solver = "Cholesky"; 
//   else
//     GCV_lin_sys_solver = "Woodbury"; 

    
//   for(int nsim = launch_sim; nsim <= launch_sim; ++nsim){

//     std::string path_solutions = R_path + "/R/Our/data/SRPDE/Test_" + 
//       TestNumber + "/performance/MC_nsim_1000/mesh_" + nnodes + "/sim_" + std::to_string(nsim); 

//     std::string path_GCV = path_solutions + "/GCV/" + GCV_type + "/" + GCV_lin_sys_solver; 

//     // define domain and regularizing PDE
//     MeshLoader<Mesh2D<>> domain("c_shaped_" + nnodes);
//     auto L = Laplacian();
//     DMatrix<double> u = DMatrix<double>::Zero(domain.mesh.elements()*3, 1);
//     PDE problem(domain.mesh, L, u); // definition of regularizing PDE

//     // define statistical model
//     CSVReader<double> reader{};
//     // load locations where data are sampled
//     CSVFile<double> locFile;
//     locFile = reader.parseFile(path_solutions + "/locs.csv");
//     DMatrix<double> loc = locFile.toEigen();

//     // define statistical model
//     SRPDE<decltype(problem), fdaPDE::models::GeoStatLocations> model(problem);
//     model.set_spatial_locations(loc);
    
//     // load data from .csv files
//     CSVFile<double> yFile; // observation file
//     yFile = reader.parseFile(path_solutions + "/z.csv");
//     DMatrix<double> y = yFile.toEigen();
//     CSVFile<double> XFile; // design matrix
//     XFile = reader.parseFile(path_solutions + "/X.csv");
//     DMatrix<double> X = XFile.toEigen();

//     // set model data
//     BlockFrame<double, int> df;
//     df.insert(OBSERVATIONS_BLK, y);
//     df.insert(DESIGN_MATRIX_BLK, X);
//     model.setData(df);
//     model.init(); // init model

//     // define grid of lambda values
//     std::vector<SVector<1>> lambdas;
//     for(double x = -3.0; x <= 3.0; x +=0.25) lambdas.push_back(SVector<1>(std::pow(10, x)));
    
//     // define GCV calibrator
//     std::size_t seed = 66546513;
//     GCV<decltype(model), StochasticEDF<decltype(model)>> GCV(model, MC_nsim, seed, gcv_enum);
//     GridOptimizer<1> opt;

//     ScalarField<1, decltype(GCV)> obj(GCV);
//     std::chrono::duration<double> delta_time; 
//     double vm_init, rss_init, vm_final, rss_final, delta_vm, delta_rss;

//     mem_usage(vm_init, rss_init);
//     auto t0 = high_resolution_clock::now();
//     opt.optimize(obj, lambdas);     // optimize gcv field
//     auto t1 = high_resolution_clock::now();
//     mem_usage(vm_final, rss_final);

//     delta_vm = vm_final - vm_init; 
//     delta_rss = rss_final - rss_init; 
//     delta_time = t1 - t0;
//     std::cout << "Duration: " << delta_time.count() << " seconds" << std::endl;
//     std::cout << "rss used: " << delta_rss*1e-3 << " Mb" << std::endl;

//     // Save results 

//     // Lambda opt
//     SVector<1> best_lambda = opt.optimum();
//     std::ofstream fileLambdaopt(path_solutions + "/LambdaCpp_" + GCV_lin_sys_solver + ".csv");
//     if(fileLambdaopt.is_open()){
//       fileLambdaopt << std::setprecision(16) << best_lambda[0];
//       fileLambdaopt.close();
//     }

//     // GCV scores
//     std::ofstream fileGCV_scores(path_GCV + "/GCV_scoresCpp.csv");
//     if(fileGCV_scores.is_open()){
//       for(std::size_t i = 0; i < GCV.values().size(); ++i) 
//         fileGCV_scores << std::setprecision(16) << GCV.values()[i] << "\n";

//       fileGCV_scores.close();  
//     }

//     // Duration 
//     std::ofstream myfileTime(path_GCV +  "/Time_Cpp.csv");
//     myfileTime << std::setprecision(16) << delta_time.count() << "\n";

//     // Memory rss (Resident set size)
//     std::ofstream myfileRSS(path_GCV +  "/rss_Cpp.csv");
//     myfileRSS << std::setprecision(16) << delta_rss << "\n";

//     // Memory vm (Virtual Memory)
//     std::ofstream myfileVM(path_GCV +  "/vm_Cpp.csv");
//     myfileVM << std::setprecision(16) << delta_vm << "\n"; 
//   }
  
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








// GCV Test for SQRPDE
// -------------------


/* test 9
   domain:       unit square [0,1] x [0,1] (coarse)
   sampling:     locations = nodes
   penalization: simple laplacian
   covariates:   no
   BC:           no
   order FE:     1
   GCV optimization: grid exact
   Correspondent in R: Test_1 
 */
TEST(GCV_SQRPDE, Test9_Laplacian_NonParametric_GeostatisticalAtNodes_GridExact) {

  const std::string TestNumber = "1"; 

  // parameters 
  double alpha = 0.5; 
  unsigned int alpha_int = alpha*100; 
  const std::string alpha_string = std::to_string(alpha_int); 

    // Marco
  std::string R_path = "/mnt/c/Users/marco/OneDrive - Politecnico di Milano/Corsi/Magistrale/Anno_II_Semestre_II/PACS_project_shared"; 
  
  // Ilenia 
  // std::string R_path = "/mnt/c/Users/ileni/OneDrive - Politecnico di Milano/PACS_project_shared"; 

  std::string data_macro_strategy_type = "matern_data"; 
  std::string data_strategy_type = "F"; 
  double tol_weights = 0.000001; 
  double tol_FPIRLS = 0.000001;

  bool massLumping_system = false;
  bool massLumping_GCV = false; 

  unsigned int M = 11;   // number of simulations

  // define domain and regularizing PDE
  MeshLoader<Mesh2D<>> domain("unit_square"); 
  auto L = Laplacian();
  DMatrix<double> u = DMatrix<double>::Zero(domain.mesh.elements()*3, 1);
  PDE problem(domain.mesh, L, u, massLumping_system); // definition of regularizing PDE

  // define the statistical model 
  SQRPDE<decltype(problem), fdaPDE::models::GeoStatMeshNodes> model(problem, alpha);

  // define grid of lambda values
  std::vector<SVector<1>> lambdas;
  if(almost_equal(alpha, 0.5))
    for(double x = -7.0; x <= -4.9; x +=0.2) lambdas.push_back(SVector<1>(std::pow(10,x)));
  if(almost_equal(alpha, 0.1))
    for(double x = -8.0; x <= -5.9; x +=0.2) lambdas.push_back(SVector<1>(std::pow(10,x)));
   for(auto l : lambdas) 
    std::cout << "lambdas " << l << std::endl; 

  for(unsigned int m = 1; m <= M; ++m){

    std::cout << "Simulation " << m << std::endl; 

    std::string solution_path = R_path + "/R/Our/data/Test_" + 
                    TestNumber + "/alpha_" + alpha_string + "/" + data_macro_strategy_type + "/strategy_"  + 
                    data_strategy_type + "/our/tol_weights_1e-06/tol_FPIRLS_1e-06/LU/sim_" + std::to_string(m);

    std::string GCV_path = solution_path +  "/GCV/Exact";              

    // load data from .csv files
    CSVReader<double> reader{};
    CSVFile<double> yFile; // observation file
    yFile = reader.parseFile(solution_path + "/z.csv");             
    DMatrix<double> y = yFile.toEigen();
    
    // set model data
    BlockFrame<double, int> df;
    df.insert(OBSERVATIONS_BLK, y);
    model.setData(df);

    model.setTolerances(tol_weights, tol_FPIRLS); 
    model.setMassLumpingGCV(massLumping_GCV); 

    model.init(); // init model

    // define GCV function and optimize
    GCV<decltype(model), ExactEDF<decltype(model)>> GCV(model);
    GridOptimizer<1> opt;

    ScalarField<1, decltype(GCV)> obj(GCV);
    opt.optimize(obj, lambdas); // optimize gcv field
    SVector<1> best_lambda = opt.optimum();
    
    // Lambda opt
    std::cout << "Lambda optimal is: " << best_lambda[0] << std::endl;
    std::ofstream fileLambdaopt(solution_path + "/LambdaCpp.csv");
    if (fileLambdaopt.is_open()){
      fileLambdaopt << std::setprecision(16) << best_lambda[0];
      fileLambdaopt.close();
    }

    // GCV scores
    std::ofstream fileGCV_scores(GCV_path + "/GCV_scoresCpp.csv");
    for(std::size_t i = 0; i < GCV.values().size(); ++i) 
      fileGCV_scores << std::setprecision(16) << std::sqrt(GCV.values()[i]) << "\n" ; 
    fileGCV_scores.close(); 

    // // Degub
    // DVector<double> invR0_Cpp = model.lumped_invR0().diagonal();
    // const static Eigen::IOFormat CSVFormat(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n");
    // std::ofstream fileLumped(R_path + "/R/Our/data/Test_" + 
    //           TestNumber + "/alpha_" + alpha_string + "/" + data_macro_strategy_type + "/strategy_"  + data_strategy_type + 
    //           "/mass" + mass_type + "/" + stopping_type + "/tol_weights_" + tol_weights_string + "/tol_FPIRLS_" + tol_FPIRLS_string + 
    //           "/" + lin_sys_solver + "/sim_" + std::to_string(m) + "/lumped_invR0_Cpp.csv");
    // if (fileLumped.is_open()){
    //   fileLumped << invR0_Cpp.format(CSVFormat) << '\n';
    //   fileLumped.close();
    // }
  }
 

}


/* test 10
   domain:       unit square [0,1] x [0,1] (coarse)
   sampling:     locations = nodes
   penalization: simple laplacian
   covariates:   no
   BC:           no
   order FE:     1
   GCV optimization: grid stochastic
   Correspondent in R: Test_1
 */
// TEST(GCV_SQRPDE, Test10_Laplacian_NonParametric_GeostatisticalAtNodes_GridStochastic) {
//   // define domain and regularizing PDE
//   MeshLoader<Mesh2D<>> domain("unit_square"); 
//   auto L = Laplacian();
//   DMatrix<double> u = DMatrix<double>::Zero(domain.mesh.elements()*3, 1);
//   PDE problem(domain.mesh, L, u); // definition of regularizing PDE

//   // define statistical model
//   double alpha = 0.5; 
//   const std::string alpha_string = "50";
//   const std::string TestNumber = "1"; 
//   SQRPDE<decltype(problem), fdaPDE::models::GeoStatMeshNodes> model(problem, alpha);

//   // Marco
//   // std::string R_path = "/mnt/c/Users/marco/OneDrive - Politecnico di Milano/Corsi/Magistrale/Anno_II_Semestre_II/PACS_project_shared"; 
  
//   // Ilenia 
//   std::string R_path = "/mnt/c/Users/ileni/OneDrive - Politecnico di Milano/PACS_project_shared"; 
  
//   // load data from .csv files
//   CSVReader<double> reader{};
//   CSVFile<double> yFile; // observation file
//   std::string data_macro_strategy_type = "skewed_data"; 
//   std::string data_strategy_type = "B"; 
//   yFile = reader.parseFile(R_path + "/R/Our/data/Test_" + 
//                   TestNumber + "/alpha_" + alpha_string + "/" + data_macro_strategy_type + "/strategy_"  + data_strategy_type + 
//                   "/z.csv");             
//   DMatrix<double> y = yFile.toEigen();
  
//   // set model data
//   BlockFrame<double, int> df;
//   df.insert(OBSERVATIONS_BLK, y);
//   model.setData(df);
//   model.init(); // init model

//   // define grid of lambda values
//   std::vector<SVector<1>> lambdas;
//   for(double x = -4.0; x <= -1.0; x +=0.25) lambdas.push_back(SVector<1>(std::pow(10,x)));

//   std::vector<double> seq_tol_weights = {0.00000001, 0.000001}; 
//   std::vector<std::string> seq_tol_weights_string = {"1e-08",  "1e-06"}; 

//   std::vector<double> seq_tol_FPIRLS = {0.000000001, 0.00000001, 0.0000001, 0.000001};
//   std::vector<std::string> seq_tol_FPIRLS_string = {"1e-09", "1e-08", "1e-07", "1e-06"}; 

//   std::string lin_sys_solver = "LU";    // depends on the "symmetry" option in R 

//   std::string GCV_type = "Stochastic";
  
// for(int i = 0; i < seq_tol_weights.size(); ++i ){
//     for(int j = 0; j < seq_tol_FPIRLS.size(); ++j){

//       model.setTolerances(seq_tol_weights[i], seq_tol_FPIRLS[j]); 
  
//       // define GCV function and optimize
//       std::size_t seed = 476813;  
//       GCV<decltype(model), StochasticEDF<decltype(model)>> GCV(model, 1000, seed) ; // semi-param: StochasticEDFMethod::Cholesky);
//       GridOptimizer<1> opt;

//       ScalarField<1, decltype(GCV)> obj(GCV);
//       opt.optimize(obj, lambdas); // optimize gcv field
//       SVector<1> best_lambda = opt.optimum();
      
//       std::cout << "Lambda optimal is: " << best_lambda[0] << std::endl ; 
//       // check optimal lambda
//       // EXPECT_TRUE( almost_equal(best_lambda[0], lambdas[4][0]) );


//       // // Lambda vector
//       // std::ofstream fileGCV_lambda("data/models/SQRPDE/2D_test1_GCV/Eaxct/GCV_lambdasCpp_" + alpha_string + ".csv");
//       // for(std::size_t i = 0; i < lambdas.size(); ++i) 
//       //   fileGCV_lambda << std::setprecision(16) << lambdas[i] << "\n" ; 

//       // fileGCV_lambda.close(); 

//       // GCV scores
//       std::ofstream fileGCV_scores(R_path + "/R/Our/data/Test_" + TestNumber + "/alpha_" + alpha_string + 
//                   "/" + data_macro_strategy_type + "/strategy_"  + data_strategy_type + 
//                   "/block/tol_weights_" + seq_tol_weights_string[i] + "/tol_FPIRLS_" + seq_tol_FPIRLS_string[j] + 
//                   "/" + lin_sys_solver +
//                   "/GCV/" + GCV_type + "/GCV_scoresCpp_" + alpha_string + ".csv");
//       for(std::size_t i = 0; i < GCV.values().size(); ++i) 
//         fileGCV_scores << std::setprecision(16) << std::sqrt(GCV.values()[i]) << "\n" ; 

//       fileGCV_scores.close(); 


//       // Edf 
//       std::ofstream fileGCV_edf(R_path + "/R/Our/data/Test_" + TestNumber + "/alpha_" + alpha_string + 
//                   "/" + data_macro_strategy_type + "/strategy_"  + data_strategy_type + 
//                   "/block/tol_weights_" + seq_tol_weights_string[i] + "/tol_FPIRLS_" + seq_tol_FPIRLS_string[j] + 
//                   "/" + lin_sys_solver + 
//                   "/GCV/" + GCV_type + "/GCV_edfCpp_" + alpha_string + ".csv");
//       for(std::size_t i = 0; i < GCV.edfs().size(); ++i) 
//         fileGCV_edf << std::setprecision(16) << GCV.edfs()[i] << "\n" ; 

//       fileGCV_edf.close(); 


//     }
// }


// }


/* test 11
   domain:       horseshoe_medium
   sampling:     locations != nodes
   penalization: simple laplacian
   covariates:   no
   BC:           no
   order FE:     1
   GCV optimization: grid exact
   Correspondent in R: Test_6
 */
// TEST(GCV_SQRPDE, Test11_Laplacian_NonParametric_GeostatisticalAtLocations_GridExact) {
//   // define domain and regularizing PDE
//   MeshLoader<Mesh2D<>> domain("c_shaped_medium"); 
//   auto L = Laplacian();
//   DMatrix<double> u = DMatrix<double>::Zero(domain.mesh.elements()*3, 1);
//   PDE problem(domain.mesh, L, u); // definition of regularizing PDE

//   // define statistical model
//   double alpha = 0.1; 
//   const std::string alpha_string = "10";
//   const std::string TestNumber = "6"; 
//   SQRPDE<decltype(problem), fdaPDE::models::GeoStatLocations> model(problem, alpha);

//   // Marco
//   // std::string R_path = "/mnt/c/Users/marco/OneDrive - Politecnico di Milano/Corsi/Magistrale/Anno_II_Semestre_II/PACS_project_shared"; 
  
//   // Ilenia 
//   std::string R_path = "/mnt/c/Users/ileni/OneDrive - Politecnico di Milano/PACS_project_shared"; 
  
//   // load data from .csv files
//   CSVReader<double> reader{};
//   CSVFile<double> yFile; // observation file
//   yFile = reader.parseFile(R_path + "/R/Our/data/Test_" + 
//                   TestNumber + "/alpha_" + alpha_string + "/z.csv");             
//   DMatrix<double> y = yFile.toEigen();

//   // load locations where data are sampled
//   CSVFile<double> locFile;
//   locFile = reader.parseFile(R_path + "/R/Our/data/Test_" + TestNumber + "/alpha_" + alpha_string + "/locs.csv");
//   DMatrix<double> loc = locFile.toEigen();

//   model.set_spatial_locations(loc);

  
//   // set model data
//   BlockFrame<double, int> df;
//   df.insert(OBSERVATIONS_BLK, y);
//   model.setData(df);
//   model.init(); // init model

//   // define grid of lambda values
//   std::vector<SVector<1>> lambdas;
//   for(double x = -3.0; x <= 3.0; x +=0.25) lambdas.push_back(SVector<1>(std::pow(10,x)));

//   std::vector<double> seq_tol_weights = {0.00000001, 0.000001}; 
//   std::vector<std::string> seq_tol_weights_string = {"1e-08",  "1e-06"}; 

//   std::vector<double> seq_tol_FPIRLS = {0.000000001, 0.00000001, 0.0000001, 0.000001};
//   std::vector<std::string> seq_tol_FPIRLS_string = {"1e-09", "1e-08", "1e-07", "1e-06"}; 

  
// for(int i = 0; i < seq_tol_weights.size(); ++i ){
//     for(int j = 0; j < seq_tol_FPIRLS.size(); ++j){

//       model.setTolerances(seq_tol_weights[i], seq_tol_FPIRLS[j]); 
  
//       // define GCV function and optimize
//       std::string GCV_type = "Exact"
//       GCV<decltype(model), ExactEDF<decltype(model)>> GCV(model);
//       GridOptimizer<1> opt;


//       ScalarField<1, decltype(GCV)> obj(GCV);
//       opt.optimize(obj, lambdas); // optimize gcv field
//       SVector<1> best_lambda = opt.optimum();
      
//       std::cout << "Lambda optimal is: " << best_lambda[0] << std::endl ; 
//       // check optimal lambda
//       // EXPECT_TRUE( almost_equal(best_lambda[0], lambdas[4][0]) );


//       // // Lambda vector
//       // std::ofstream fileGCV_lambda("data/models/SQRPDE/2D_test1_GCV/" + GCV_type + "/GCV_lambdasCpp_" + alpha_string + ".csv");
//       // for(std::size_t i = 0; i < lambdas.size(); ++i) 
//       //   fileGCV_lambda << std::setprecision(16) << lambdas[i] << "\n" ; 

//       // fileGCV_lambda.close(); 

//       // GCV scores
//       std::ofstream fileGCV_scores(R_path + "/R/Our/data/Test_" 
//                   + TestNumber + "/alpha_" + alpha_string + "/" + 
//                   "/tol_weights_" + seq_tol_weights_string[i] + "/tol_FPIRLS_" + seq_tol_FPIRLS_string[j] + 
//                   "/GCV/" + GCV_type + "/GCV_scoresCpp_" + alpha_string + ".csv");
//       for(std::size_t i = 0; i < GCV.values().size(); ++i) 
//         fileGCV_scores << std::setprecision(16) << std::sqrt(GCV.values()[i]) << "\n" ; 

//       fileGCV_scores.close(); 


//       // Edf
//       std::ofstream fileGCV_edf(R_path + "/R/Our/data/Test_" 
//                   + TestNumber + "/alpha_" + alpha_string + 
//                   "/tol_weights_" + seq_tol_weights_string[i] + "/tol_FPIRLS_" + seq_tol_FPIRLS_string[j] + 
//                   "/GCV/" + GCV_type + "/GCV_edfCpp_" + alpha_string + ".csv");
//       for(std::size_t i = 0; i < GCV.edfs().size(); ++i) 
//         fileGCV_edf << std::setprecision(16) << GCV.edfs()[i] << "\n" ; 

//       fileGCV_edf.close(); 
//     }
// }


// }


/* test 12
   domain:       horseshoe_medium
   sampling:     locations != nodes
   penalization: simple laplacian
   covariates:   no
   BC:           no
   order FE:     1
   GCV optimization: grid stochastic
   Correspondent in R: Test_6
 */
// TEST(GCV_SQRPDE, Test12_Laplacian_NonParametric_GeostatisticalAtLocations_GridStochastic) {
//   // define domain and regularizing PDE
//   MeshLoader<Mesh2D<>> domain("c_shaped_medium"); 
//   auto L = Laplacian();
//   DMatrix<double> u = DMatrix<double>::Zero(domain.mesh.elements()*3, 1);
//   PDE problem(domain.mesh, L, u); // definition of regularizing PDE

//   // define statistical model
//   double alpha = 0.1; 
//   const std::string alpha_string = "10";
//   const std::string TestNumber = "6"; 
//   SQRPDE<decltype(problem), fdaPDE::models::GeoStatLocations> model(problem, alpha);

//   // Marco
//   // std::string R_path = "/mnt/c/Users/marco/OneDrive - Politecnico di Milano/Corsi/Magistrale/Anno_II_Semestre_II/PACS_project_shared"; 
  
//   // Ilenia 
//   std::string R_path = "/mnt/c/Users/ileni/OneDrive - Politecnico di Milano/PACS_project_shared"; 
  
//   // load data from .csv files
//   CSVReader<double> reader{};
//   CSVFile<double> yFile; // observation file
//   yFile = reader.parseFile(R_path + "/R/Our/data/Test_" + 
//                   TestNumber + "/alpha_" + alpha_string + "/z.csv");             
//   DMatrix<double> y = yFile.toEigen();

//   // load locations where data are sampled
//   CSVFile<double> locFile;
//   locFile = reader.parseFile(R_path + "/R/Our/data/Test_" + TestNumber + "/alpha_" + alpha_string + "/locs.csv");
//   DMatrix<double> loc = locFile.toEigen();

//   model.set_spatial_locations(loc);

  
//   // set model data
//   BlockFrame<double, int> df;
//   df.insert(OBSERVATIONS_BLK, y);
//   model.setData(df);
//   model.init(); // init model

//   // define grid of lambda values
//   std::vector<SVector<1>> lambdas;
//   for(double x = -3.0; x <= 3.0; x +=0.25) lambdas.push_back(SVector<1>(std::pow(10,x)));

//   std::vector<double> seq_tol_weights = {0.00000001 , 0.000001}; 
//   std::vector<std::string> seq_tol_weights_string = {"1e-08",  "1e-06"}; 

//   std::vector<double> seq_tol_FPIRLS = {0.000000001, 0.00000001, 0.0000001, 0.000001};
//   std::vector<std::string> seq_tol_FPIRLS_string = {"1e-09", "1e-08", "1e-07", "1e-06"}; 


//   std::string GCV_type = "Stochastic";
  
// for(int i = 0; i < seq_tol_weights.size(); ++i ){
//     for(int j = 0; j < seq_tol_FPIRLS.size(); ++j){

//       model.setTolerances(seq_tol_weights[i], seq_tol_FPIRLS[j]); 
  
//       // define GCV function and optimize
//       std::size_t seed = 476813;  
//       GCV<decltype(model), StochasticEDF<decltype(model)>> GCV(model, 1000, seed) ; // semi-param: StochasticEDFMethod::Cholesky);
//       GridOptimizer<1> opt;
      

//       ScalarField<1, decltype(GCV)> obj(GCV);
//       opt.optimize(obj, lambdas); // optimize gcv field
//       SVector<1> best_lambda = opt.optimum();
      
      
//       std::cout << "Lambda optimal is: " << best_lambda[0] << std::endl ; 
//       // check optimal lambda
//       // EXPECT_TRUE( almost_equal(best_lambda[0], lambdas[4][0]) );


//       // // Lambda vector
//       // std::ofstream fileGCV_lambda("data/models/SQRPDE/2D_test1_GCV/" + GCV_type + "/GCV_lambdasCpp_" + alpha_string + ".csv");
//       // for(std::size_t i = 0; i < lambdas.size(); ++i) 
//       //   fileGCV_lambda << std::setprecision(16) << lambdas[i] << "\n" ; 

//       // fileGCV_lambda.close(); 

//       // GCV scores
//       std::ofstream fileGCV_scores(R_path + "/R/Our/data/Test_" 
//                   + TestNumber + "/alpha_" + alpha_string + "/" + 
//                   "/tol_weights_" + seq_tol_weights_string[i] + "/tol_FPIRLS_" + seq_tol_FPIRLS_string[j] + 
//                   "/GCV/" + GCV_type + "/GCV_scoresCpp_" + alpha_string + ".csv");
//       for(std::size_t i = 0; i < GCV.values().size(); ++i) 
//         fileGCV_scores << std::setprecision(16) << std::sqrt(GCV.values()[i]) << "\n" ; 

//       fileGCV_scores.close(); 


//       // Edf 
//       std::ofstream fileGCV_edf(R_path + "/R/Our/data/Test_" 
//                   + TestNumber + "/alpha_" + alpha_string + 
//                   "/tol_weights_" + seq_tol_weights_string[i] + "/tol_FPIRLS_" + seq_tol_FPIRLS_string[j] + 
//                   "/GCV/" + GCV_type + "/GCV_edfCpp_" + alpha_string + ".csv");
//       for(std::size_t i = 0; i < GCV.edfs().size(); ++i) 
//         fileGCV_edf << std::setprecision(16) << GCV.edfs()[i] << "\n" ; 

//       fileGCV_edf.close(); 
//     }
// }


// }




/* test 13
   domain:       unit square [0,1] x [0,1] 
   sampling:     locations != nodes
   penalization: simple laplacian
   covariates:   yes
   BC:           no
   order FE:     1
   GCV optimization: grid exact
   Correspondent in R: Test_5
 */
// TEST(GCV_SQRPDE, Test13_Laplacian_SemiParametric_GeostatLocations_GridExact) {

//   // define statistical model
//   double alpha = 0.5; 
//   unsigned int alpha_int = alpha*100; 
//   const std::string alpha_string = std::to_string(alpha_int);
//   const std::string TestNumber = "5"; 

//   // Marco
//   std::string R_path = "/mnt/c/Users/marco/OneDrive - Politecnico di Milano/Corsi/Magistrale/Anno_II_Semestre_II/PACS_project_shared"; 
  
//   // Ilenia 
//   // std::string R_path = "/mnt/c/Users/ileni/OneDrive - Politecnico di Milano/PACS_project_shared"; 
  
//   // load data from .csv files
//   CSVReader<double> reader{};
//   CSVFile<double> yFile; // observation file
//   CSVFile<double> XFile; // covariates file
//   CSVFile<double> locFile; // locations file

//   std::string data_macro_strategy_type = "skewed_data"; 
//   std::string data_strategy_type = "E"; 

//   std::vector<std::string> seq_n_string = {"3969"};  

//   double tol_weights = 0.000001; 
//   double tol_FPIRLS = 0.000001;    
 
//   std::string GCV_type = "Exact"; 

//   bool massLumping_system = true;
//   bool massLumping_GCV = true; 
//   std::string mass_type; 
//   if(!massLumping_system & !massLumping_GCV)
//     mass_type = "FF";
//   if(!massLumping_system & massLumping_GCV)
//     mass_type = "FT"; 
//   if(massLumping_system & massLumping_GCV)
//     mass_type = "TT";

//   // define grid of lambda values
//   std::vector<SVector<1>> lambdas;
//   for(double x = -4.0; x <= -0.9; x +=0.2) lambdas.push_back(SVector<1>(std::pow(10,x))); 

//   DMatrix<double> loc; 
//   DMatrix<double> X; 
//   DMatrix<double> y; 

//   std::string lin_sys_solver = "Cholesky";  // Cholesky Woodbury
//   std::string lin_sys_solver_abbrv = "Chol";  // Chol Wood 


//   unsigned int launch_sim = 1;

//   for(int nsim = launch_sim; nsim <= launch_sim; ++nsim){

//     // define domain and regularizing PDE
//     MeshLoader<Mesh2D<>> domain("unit_square_71");  
//     auto L = Laplacian();
//     DMatrix<double> u = DMatrix<double>::Zero(domain.mesh.elements()*3, 1);
//     PDE problem(domain.mesh, L, u, massLumping_system); // definition of regularizing PDE
//     SQRPDE<decltype(problem), fdaPDE::models::GeoStatLocations> model(problem, alpha);
//     GCV<decltype(model), ExactEDF<decltype(model)>> GCV(model);


//     // Read data
//     std::string path_solutions = R_path + "/R/Our/data/Test_" + 
//           TestNumber + "/alpha_" + alpha_string + "/" + data_macro_strategy_type + 
//           "/for_slides_lumping&solvers/" +
//           "system_solver_" + lin_sys_solver_abbrv + "/lump" + mass_type + "/sim_" + std::to_string(nsim); 

//     std::string path_GCV = path_solutions + "/GCV" + "/" + GCV_type; 

//     yFile = reader.parseFile(path_solutions + "/z.csv");             
//     y = yFile.toEigen();

//     XFile = reader.parseFile(path_solutions + "/X.csv");             
//     X = XFile.toEigen();

//     locFile = reader.parseFile(path_solutions + "/locs.csv");
//     loc = locFile.toEigen();
//     model.set_spatial_locations(loc); 

//     // set model data
//     BlockFrame<double, int> df;
//     df.insert(OBSERVATIONS_BLK, y);
//     df.insert(DESIGN_MATRIX_BLK, X);
//     model.setData(df);
//     model.setTolerances(tol_weights, tol_FPIRLS); 
//     model.setMassLumpingGCV(massLumping_GCV); 
//     model.setLinearSystemType(lin_sys_solver); 

//     model.init(); // init model


//     // define GCV function and optimize  
//     GridOptimizer<1> opt;
//     ScalarField<1, decltype(GCV)> obj(GCV);

//     opt.optimize(obj, lambdas);     // optimize gcv field
   
//     // Lambda opt
//     SVector<1> best_lambda = opt.optimum();
//     std::ofstream fileLambdaopt(path_solutions + "/LambdaCpp.csv");
//     if(fileLambdaopt.is_open()){
//       fileLambdaopt << std::setprecision(16) << best_lambda[0];
//       fileLambdaopt.close();
//     }

//     // GCV scores
//     std::ofstream fileGCV_scores(path_GCV + "/GCV_scoresCpp.csv");
//     if(fileGCV_scores.is_open()){
//       for(std::size_t i = 0; i < GCV.values().size(); ++i) 
//         fileGCV_scores << std::setprecision(16) << std::sqrt(GCV.values()[i]) << "\n";

//       fileGCV_scores.close();  
//     }

//   }


// }


/* test 14
   domain:       unit square [0,1] x [0,1] 
   sampling:     locations != nodes
   penalization: simple laplacian
   covariates:   yes
   BC:           no
   order FE:     1
   GCV optimization: grid stochastic
   Correspondent in R: Test_5
 */
// TEST(GCV_SQRPDE, Test14_Laplacian_SemiParametric_GeostatisticalAtLocations_GridStochastic) {

//   // Parameters 
//   const std::string TestNumber = "5"; 
//   double alpha = 0.5; 
//   unsigned int alpha_int = alpha*100; 
//   const std::string alpha_string = std::to_string(alpha_int); 
  
//   std::string data_macro_strategy_type = "skewed_data"; 
//   std::string data_strategy_type = "E"; 
//   std::string GCV_type = "Stochastic"; 

//   // Marco
//   std::string R_path = "/mnt/c/Users/marco/OneDrive - Politecnico di Milano/Corsi/Magistrale/Anno_II_Semestre_II/PACS_project_shared"; 
  
//   // Ilenia 
//   // std::string R_path = "/mnt/c/Users/ileni/OneDrive - Politecnico di Milano/PACS_project_shared"; 
  
//   // load data from .csv files
//   CSVReader<double> reader{};
//   CSVFile<double> yFile; // observation file
//   CSVFile<double> XFile; // covariates file
//   CSVFile<double> locFile; // locations file

//   DMatrix<double> loc; 
//   DMatrix<double> X; 
//   DMatrix<double> y; 

//   // define grid of lambda values
//   std::vector<SVector<1>> lambdas;
//   for(double x = -4.0; x <= -0.9; x +=0.2) lambdas.push_back(SVector<1>(std::pow(10,x)));  

//   double tol_weights = 0.000001; 
//   double tol_FPIRLS = 0.000001;

//   bool massLumping_system = false;
//   bool massLumping_GCV = false; 
//   std::string mass_type; 
//   if(!massLumping_system & !massLumping_GCV)
//     mass_type = "FF";
//   if(!massLumping_system & massLumping_GCV)
//     mass_type = "FT"; 
//   if(massLumping_system & massLumping_GCV)
//     mass_type = "TT";

   
//   std::string lin_sys_solver = "Woodbury";  // Woodbury Cholesky  
//   std::string lin_sys_solver_abbrv = "Wood";   // Wood Chol



//   std::string launch_sqrtN = "71";  // 24 35 50 71 100 142 174 200 283

//   unsigned int launch_sim = 1;      // 1 2 3 4 5

//   auto gcv_enum = fdaPDE::calibration::StochasticEDFMethod::CholeskyGCV;  // WoodburyGCV CholeskyGCV 
  


//   std::string GCV_lin_sys_solver;       // Woodbury Cholesky 
//   if(gcv_enum) 
//     GCV_lin_sys_solver = "Cholesky"; 
//   else
//     GCV_lin_sys_solver = "Woodbury";  
    
//   for(int nsim = launch_sim; nsim <= launch_sim; ++nsim){

//     // define domain and regularizing PDE
//     MeshLoader<Mesh2D<>> domain("unit_square_" + launch_sqrtN);  
//     auto L = Laplacian();
//     DMatrix<double> u = DMatrix<double>::Zero(domain.mesh.elements()*3, 1);
//     PDE problem(domain.mesh, L, u, massLumping_system); // definition of regularizing PDE
//     SQRPDE<decltype(problem), fdaPDE::models::GeoStatLocations> model(problem, alpha);
//     std::size_t seed = 476813;
//     GCV<decltype(model), StochasticEDF<decltype(model)>> GCV(model, 1000, seed, gcv_enum);

//     // Read data
//     std::string path_solutions = R_path + "/R/Our/data/Test_" + 
//           TestNumber + "/alpha_" + alpha_string + "/" + data_macro_strategy_type + 
//           "/for_slides_lumping&solvers/" +
//           "system_solver_" + lin_sys_solver_abbrv + "/incr_N/sqrtN_" + launch_sqrtN + "/sim_" + std::to_string(nsim); 

//     std::string path_GCV = path_solutions + "/GCV/" + GCV_type + "/" + GCV_lin_sys_solver; 

//     yFile = reader.parseFile(path_solutions + "/z.csv");             
//     y = yFile.toEigen();

//     XFile = reader.parseFile(path_solutions + "/X.csv");             
//     X = XFile.toEigen();

//     locFile = reader.parseFile(path_solutions + "/locs.csv");
//     loc = locFile.toEigen();
//     model.set_spatial_locations(loc); 

//     // set model data
//     BlockFrame<double, int> df;
//     df.insert(OBSERVATIONS_BLK, y);
//     df.insert(DESIGN_MATRIX_BLK, X);
//     model.setData(df);
//     model.setTolerances(tol_weights, tol_FPIRLS); 
//     model.setMassLumpingGCV(massLumping_GCV); 
//     model.setLinearSystemType(lin_sys_solver); 

//     model.init(); // init model


//     // define GCV function and optimize  
//     GridOptimizer<1> opt;
//     std::chrono::duration<double> delta_time; 
//     double vm_init, rss_init, vm_final, rss_final, delta_vm, delta_rss;

//     ScalarField<1, decltype(GCV)> obj(GCV);

//     mem_usage(vm_init, rss_init);
//     auto t0 = high_resolution_clock::now();
//     opt.optimize(obj, lambdas);     // optimize gcv field
//     auto t1 = high_resolution_clock::now();
//     mem_usage(vm_final, rss_final);

//     delta_vm = vm_final - vm_init; 
//     delta_rss = rss_final - rss_init; 
//     delta_time = t1 - t0;
//     std::cout << "Duration: " << delta_time.count() << "seconds" << std::endl;
//     std::cout << "rss used: " << delta_rss*1e-3 << "Mb" << std::endl;
      
    // // Lambda opt
    // SVector<1> best_lambda = opt.optimum();
    // std::ofstream fileLambdaopt(path_solutions + "/LambdaCpp_" + GCV_lin_sys_solver + ".csv");
    // if(fileLambdaopt.is_open()){
    //   fileLambdaopt << std::setprecision(16) << best_lambda[0];
    //   fileLambdaopt.close();
    // }

    // // GCV scores
    // std::ofstream fileGCV_scores(path_GCV + "/GCV_scoresCpp.csv");
    // if(fileGCV_scores.is_open()){
    //   for(std::size_t i = 0; i < GCV.values().size(); ++i) 
    //     fileGCV_scores << std::setprecision(16) << std::sqrt(GCV.values()[i]) << "\n";

    //   fileGCV_scores.close();  
    // }

    // // Duration 
    // std::ofstream myfileTime(path_GCV +  "/Time_Cpp.csv");
    // myfileTime << std::setprecision(16) << delta_time.count() << "\n";

    // // Memory rss (Resident set size)
    // std::ofstream myfileRSS(path_GCV +  "/rss_Cpp.csv");
    // myfileRSS << std::setprecision(16) << delta_rss << "\n";

    // // Memory vm (Virtual Memory)
    // std::ofstream myfileVM(path_GCV +  "/vm_Cpp.csv");
    // myfileVM << std::setprecision(16) << delta_vm << "\n"; 


//   }

// }


/* test 15
   domain:       horseshoe
   sampling:     locations = nodes
   penalization: simple laplacian
   covariates:   yes
   BC:           no
   order FE:     1
   GCV optimization: grid exact
   Correspondent in R: Test_2
 */
// TEST(GCV_SQRPDE, Test15_Laplacian_SemiParametric_GeostatisticalAtNodes_GridExact) {

//   const std::string TestNumber = "2";
//   // Marco
//   std::string R_path = "/mnt/c/Users/marco/OneDrive - Politecnico di Milano/Corsi/Magistrale/Anno_II_Semestre_II/PACS_project_shared"; 

//   // Ilenia 
//   // std::string R_path = "/mnt/c/Users/ileni/OneDrive - Politecnico di Milano/PACS_project_shared"; 

//   double alpha = 0.1; 
//   std::string alpha_string = "10";

//   double tol_weights = 1e-6;
//   std::string tol_weights_string = "1e-6";
//   double tol_FPIRLS = 1e-6;
//   std::string tol_FPIRLS_string = "1e-6";
   
//   //std::vector<int> seq_n = {4729, 5502};   // {250, 497, 997, 2003, 3983} ; 
//   //std::vector<std::string> seq_n_string = {"4729", "5502"};  // {"250", "497", "997", "2003", "3983"} ;

//   std::vector<int> seq_N = {597, 1020, 1520, 2003, 2563, 3040, 3503, 4012};  
//   std::vector<std::string> seq_N_string = {"597", "1020", "1520", "2003", "2563", "3040", "3503", "4012"};

//   unsigned int M = 1;
  
//   DMatrix<double> lambda; 
//   CSVFile<double> yFile; 
//   CSVFile<double> XFile;
//   DMatrix<double> y;  
//   DMatrix<double> X;

//   // define grid of lambda values
//   std::vector<SVector<1>> lambdas;
//   for(double x = -4.0; x <= -0.9; x +=0.20) lambdas.push_back(SVector<1>(std::pow(10,x)));

//   for(int m = 1; m <= M ; ++m){

//     std::cout << "Simulation : " << m << std::endl; 

//     CSVReader<double> reader{};
//     CSVFile<double> locFile;

//     //for(int n = 0; n < seq_n.size(); ++n){
//         for (int k = 0; k < seq_N_string.size(); ++k){

//           // define domain and regularizing PDE
//           MeshLoader<Mesh2D<>> domain("multiple_c_shaped/mesh_" +  seq_N_string[k]);  
//           auto L = Laplacian();
//           DMatrix<double> u = DMatrix<double>::Zero(domain.mesh.elements()*3, 1);
//           PDE problem(domain.mesh, L, u); // definition of regularizing PDE

//           // define statistical model
//           SQRPDE<decltype(problem), fdaPDE::models::GeoStatLocations> model(problem, alpha);


//           // Reading 

//           // load locations where data are sampled
//           locFile = reader.parseFile(R_path + "/R/Our/data/Test_" 
//                           + TestNumber + "/alpha_" + alpha_string  + "/tol_weights_" + tol_weights_string + 
//                           "/compare_nodes" + "/locs.csv");  
//                           //  "/sim_M/n_" + seq_n_string[n] + "/sim_" + std::to_string(m) +"/locs.csv");
//           DMatrix<double> loc = locFile.toEigen();
          
//           // load data from .csv files
//           yFile = reader.parseFile(R_path + "/R/Our/data/Test_" 
//                           + TestNumber + "/alpha_" + alpha_string  + "/tol_weights_" + tol_weights_string + 
//                           "/compare_nodes" + "/z.csv"); 
//           y = yFile.toEigen();
//           XFile = reader.parseFile(R_path + "/R/Our/data/Test_" 
//                           + TestNumber + "/alpha_" + alpha_string  + "/tol_weights_" + tol_weights_string + 
//                           "/compare_nodes" + "/X.csv"); 
//           X = XFile.toEigen();

//           // set model data
//           model.set_spatial_locations(loc);
//           BlockFrame<double, int> df;
//           df.insert(OBSERVATIONS_BLK, y);
//           df.insert(DESIGN_MATRIX_BLK, X);
          
//           model.setData(df);
//           model.setTolerances(tol_weights, tol_FPIRLS); 

//           // solve smoothing problem
//           model.init();
  
//           // define GCV function and optimize     
//           GCV<decltype(model), ExactEDF<decltype(model)>> GCV(model);
//           GridOptimizer<1> opt;


//           ScalarField<1, decltype(GCV)> obj(GCV);
//           opt.optimize(obj, lambdas); // optimize gcv field

//           SVector<1> best_lambda = opt.optimum();
          
//           // Lambda opt
//           std::cout << "Lambda optimal is: " << best_lambda[0] << std::endl ; 
//           std::ofstream fileLambdaopt(R_path + "/R/Our/data/Test_" 
//                           + TestNumber + "/alpha_" + alpha_string  + "/tol_weights_" + tol_weights_string + 
//                           "/compare_nodes" + "/N_" + seq_N_string[k] + "/LambdaCpp_" + alpha_string + ".csv");
//           if (fileLambdaopt.is_open()){
//             fileLambdaopt << std::setprecision(16) << best_lambda[0];
//             fileLambdaopt.close();
//           }

//           // Lambda vector
//           std::ofstream fileGCV_lambda(R_path + "/R/Our/data/Test_" 
//                           + TestNumber + "/alpha_" + alpha_string  + "/tol_weights_" + tol_weights_string + 
//                           "/compare_nodes" + "/N_" + seq_N_string[k] + 
//                           "/GCV/Exact/GCV_lambdasCpp_" + alpha_string + ".csv");
//           for(std::size_t i = 0; i < lambdas.size(); ++i) 
//             fileGCV_lambda << std::setprecision(16) << lambdas[i] << "\n" ; 

//           fileGCV_lambda.close(); 

//           // GCV scores
//           std::ofstream fileGCV_scores(R_path + "/R/Our/data/Test_" 
//                           + TestNumber + "/alpha_" + alpha_string  + "/tol_weights_" + tol_weights_string + 
//                           "/compare_nodes" + "/N_" + seq_N_string[k] +
//                           "/GCV/Exact/GCV_scoresCpp_" + alpha_string + ".csv");
//           for(std::size_t i = 0; i < GCV.values().size(); ++i) 
//             fileGCV_scores << std::setprecision(16) << std::sqrt(GCV.values()[i]) << "\n" ; 

//           fileGCV_scores.close(); 


//           // // Edf
//           // std::ofstream fileGCV_edf(R_path + "/R/Our/data/Test_" + 
//           //             TestNumber + "/alpha_" + alpha_string + "/tol_weights_" + tol_weights_string  + "/n_" + seq_n_string[n] + 
//           //             "/GCV/Exact/GCV_edfCpp_" + alpha_string + ".csv");
//           // for(std::size_t i = 0; i < GCV.edfs().size(); ++i) 
//           //   fileGCV_edf << std::setprecision(16) << GCV.edfs()[i] << "\n" ; 

//           //fileGCV_edf.close(); 

//         }
        
//       //}
//   }

  

// }



/* test 16
   domain:       horseshoe
   sampling:     locations = nodes
   penalization: simple laplacian
   covariates:   yes
   BC:           no
   order FE:     1
   GCV optimization: grid stochastic
   Correspondent in R: Test_2
 */
// TEST(GCV_SQRPDE, Test16_Laplacian_SemiParametric_GeostatisticalAtNodes_GridStochastic) {

//   double alpha = 0.5;  
//   double tol_weights = 1e-6;
//   std::string tol_weights_string = "1e-6";
//   double tol_FPIRLS = 1e-6;
//   std::string alpha_string = "50" ;
//   const std::string TestNumber = "2"; 
  
//   // Marco
//   std::string R_path = "/mnt/c/Users/marco/OneDrive - Politecnico di Milano/Corsi/Magistrale/Anno_II_Semestre_II/PACS_project_shared"; 
  
//   // Ilenia 
//   // std::string R_path = "/mnt/c/Users/ileni/OneDrive - Politecnico di Milano/PACS_project_shared"; 
   
//   std::vector<int> seq_n = {997};  // {5502, 5254, 5038, 4614, 3671, 2247} ; 
//   std::vector<std::string> seq_n_string = {"997"};   // {"5502" , "5254", "5038", "4614", "3671", "2247"} ; 
  
//   unsigned int M = 1;

//   DMatrix<double> lambda; 
//   CSVFile<double> yFile; 
//   CSVFile<double> XFile;
//   DMatrix<double> y;  
//   DMatrix<double> X;

//   MeshLoader<Mesh2D<>> domain("c_shaped_fine");         
//   auto L = Laplacian();
//   DMatrix<double> u = DMatrix<double>::Zero(domain.mesh.elements()*3, 1);
//   PDE problem(domain.mesh, L, u); // definition of regularizing PDE
//   // define statistical model
//   SQRPDE<decltype(problem), fdaPDE::models::GeoStatLocations> model(problem, alpha);


//   for(int n = 0; n < seq_n.size(); ++n ){
//     std::ofstream myfile(R_path + "/R/Our/data/Test_" 
//                       + TestNumber + "/alpha_" + alpha_string  + "/tol_weights_" + tol_weights_string + "/n_" + seq_n_string[n] + "/GCV/DurationTimeWood.csv");

//     for (int m = 1; m <= M ; ++m){

//       std::cout << "Simulation m: " << m << std::endl ; 
    
//       CSVReader<double> reader{};
//       // load locations where data are sampled
//       CSVFile<double> locFile;
//       locFile = reader.parseFile(R_path + "/R/Our/data/Test_" 
//                       + TestNumber + "/alpha_" + alpha_string  + "/tol_weights_" + tol_weights_string + "/n_" + seq_n_string[n] + "/locs.csv");
//       DMatrix<double> loc = locFile.toEigen();


//       model.set_spatial_locations(loc);

      
//       // load data from .csv files
//       yFile = reader.parseFile(R_path + "/R/Our/data/Test_" + 
//                       TestNumber + "/alpha_" + alpha_string  + "/tol_weights_" + tol_weights_string + "/n_" + seq_n_string[n] + "/z.csv"); 
//       y = yFile.toEigen();
//       XFile = reader.parseFile(R_path + "/R/Our/data/Test_" + 
//                       TestNumber + "/alpha_" + alpha_string  + "/tol_weights_" + tol_weights_string + "/n_" + seq_n_string[n] + "/X.csv"); 
//       X = XFile.toEigen();

//       // set model data
//       BlockFrame<double, int> df;
//       df.insert(OBSERVATIONS_BLK,  y);
//       df.insert(DESIGN_MATRIX_BLK, X);
      
//       model.setData(df);
//       model.setTolerances(tol_weights, tol_FPIRLS); 
//       // solve smoothing problem
//       model.init();


//       // define grid of lambda values
//       std::vector<SVector<1>> lambdas;
//       for(double x = -5.0; x <= -0.9; x +=0.20) lambdas.push_back(SVector<1>(std::pow(10,x)));  

//       // define GCV function and optimize
//       std::string stoch_type = "Wood_" ; 
//       std::size_t seed = 21;  
//       GCV<decltype(model), StochasticEDF<decltype(model)>> GCV(model, 1000, seed, StochasticEDFMethod::Woodbury);
//       GridOptimizer<1> opt;


//       ScalarField<1, decltype(GCV)> obj(GCV);

//       auto start = high_resolution_clock::now();
//       opt.optimize(obj, lambdas); // optimize gcv field
//       auto stop = high_resolution_clock::now();
//       // auto duration = duration_cast<microseconds>(stop - start);
 
//       // auto int_ms = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);

//       std::chrono::duration<double, std::milli> fp_ms = stop - start;


//       std::cout << "Duration Wood: "  << fp_ms.count() << "milliseconds " << std::endl;
//       myfile << std::setprecision(16) << fp_ms.count() << "\n" ;
 

//       SVector<1> best_lambda = opt.optimum();


//       std::cout << "Lambda optimal is: " << best_lambda[0] << std::endl ; 
//       // check optimal lambda
//       // EXPECT_TRUE( almost_equal(best_lambda[0], lambdas[4][0]) );

//       // Lambda opt
//       std::ofstream fileLambdaopt(R_path + "/R/Our/data/Test_" + 
//                   TestNumber + "/alpha_" + alpha_string + "/tol_weights_" + tol_weights_string  + "/n_" + seq_n_string[n] +
//                   "/GCV/Stochastic/LambdaCpp_" + stoch_type + alpha_string + ".csv");
//       if (fileLambdaopt.is_open()){
//         fileLambdaopt << std::setprecision(16) << best_lambda[0];
//         fileLambdaopt.close();
//       }

//       // Lambda vector
//       std::ofstream fileGCV_lambda(R_path + "/R/Our/data/Test_" + 
//                   TestNumber + "/alpha_" + alpha_string + "/tol_weights_" + tol_weights_string  + "/n_" + seq_n_string[n] + 
//                   "/GCV/Stochastic/GCV_lambdasCpp_" + stoch_type + alpha_string + ".csv");
//       for(std::size_t i = 0; i < lambdas.size(); ++i) 
//         fileGCV_lambda << std::setprecision(16) << lambdas[i] << "\n" ; 

//       fileGCV_lambda.close(); 

//       // GCV scores
//       std::ofstream fileGCV_scores(R_path + "/R/Our/data/Test_" + 
//                   TestNumber + "/alpha_" + alpha_string + "/tol_weights_" + tol_weights_string  + "/n_" + seq_n_string[n] + 
//                   "/GCV/Stochastic/GCV_scoresCpp_" + stoch_type + alpha_string + ".csv");
//       for(std::size_t i = 0; i < GCV.values().size(); ++i) 
//         fileGCV_scores << std::setprecision(16) << std::sqrt(GCV.values()[i]) << "\n" ; 

//       fileGCV_scores.close(); 


//       // Edf
//       std::ofstream fileGCV_edf(R_path + "/R/Our/data/Test_" + 
//                   TestNumber + "/alpha_" + alpha_string + "/tol_weights_" + tol_weights_string  + "/n_" + seq_n_string[n] + 
//                   "/GCV/Stochastic/GCV_edfCpp_" + stoch_type  + alpha_string + ".csv");
//       for(std::size_t i = 0; i < GCV.edfs().size(); ++i) 
//         fileGCV_edf << std::setprecision(16) << GCV.edfs()[i] << "\n" ; 

//       fileGCV_edf.close(); 
      
//     }
  
//     myfile.close();
//   }

//   // // Durations
//   // std::ofstream fileDuration(R_path + "/R/Our/data/Test_" + 
//   //             TestNumber + "/alpha_" + alpha_string + "/tol_weights_" + tol_weights_string  + "/n_" + seq_n_string[n] + "/DurationWood.csv");
//   // for(std::size_t i = 0; i < M; ++i) 
//   //     fileDuration << std::setprecision(16) << durations[i] << "\n" ; 

//   // fileDuration.close(); 
    
// }



/* test 17
   domain:       horseshoe
   sampling:     locations = nodes
   penalization: simple laplacian
   covariates:   yes
   BC:           no
   order FE:     1
   GCV optimization: grid stochastic
   Correspondent in R: Test_2
 */
// TEST(GCV_SQRPDE, Test17_Laplacian_SemiParametric_GeostatisticalAtNodes_GridStochastic) {

//   double alpha = 0.5;  
//   double tol_weights = 1e-6;
//   std::string tol_weights_string = "1e-6";
//   double tol_FPIRLS = 1e-6;
//   std::string alpha_string = "50" ;
//   const std::string TestNumber = "2"; 
  
//   // Marco
//   std::string R_path = "/mnt/c/Users/marco/OneDrive - Politecnico di Milano/Corsi/Magistrale/Anno_II_Semestre_II/PACS_project_shared"; 
  
//   // Ilenia 
//   // std::string R_path = "/mnt/c/Users/ileni/OneDrive - Politecnico di Milano/PACS_project_shared"; 

//   std::vector<int> seq_n = {997};  // {5502, 5254, 5038, 4614, 3671, 2247} ; 
//   std::vector<std::string> seq_n_string = {"997"};   // {"5502" , "5254", "5038", "4614", "3671", "2247"} ; 
  
//   unsigned int M = 1;

//   DMatrix<double> lambda; 
//   CSVFile<double> yFile; 
//   CSVFile<double> XFile;
//   DMatrix<double> y;  
//   DMatrix<double> X;

//   MeshLoader<Mesh2D<>> domain("c_shaped_fine");         
//   auto L = Laplacian();
//   DMatrix<double> u = DMatrix<double>::Zero(domain.mesh.elements()*3, 1);
//   PDE problem(domain.mesh, L, u); // definition of regularizing PDE
//   // define statistical model
//   SQRPDE<decltype(problem), fdaPDE::models::GeoStatLocations> model(problem, alpha);


//   for(int n = 0; n < seq_n.size(); ++n ){
//     std::ofstream myfile(R_path + "/R/Our/data/Test_" 
//                       + TestNumber + "/alpha_" + alpha_string  + "/tol_weights_" + tol_weights_string + "/n_" + seq_n_string[n] + "/GCV/DurationTimeChol.csv");

//     for (int m = 1; m <= M ; ++m){

//       std::cout << "Simulation m: " << m << std::endl; 
    
//       CSVReader<double> reader{};
//       // load locations where data are sampled
//       CSVFile<double> locFile;
//       locFile = reader.parseFile(R_path + "/R/Our/data/Test_" 
//                       + TestNumber + "/alpha_" + alpha_string  + "/tol_weights_" + tol_weights_string + "/n_" + seq_n_string[n] + "/locs.csv");
//       DMatrix<double> loc = locFile.toEigen();


//       model.set_spatial_locations(loc);

      
//       // load data from .csv files
//       yFile = reader.parseFile(R_path + "/R/Our/data/Test_" + 
//                       TestNumber + "/alpha_" + alpha_string  + "/tol_weights_" + tol_weights_string + "/n_" + seq_n_string[n] + "/z.csv"); 
//       y = yFile.toEigen();
//       XFile = reader.parseFile(R_path + "/R/Our/data/Test_" + 
//                       TestNumber + "/alpha_" + alpha_string  + "/tol_weights_" + tol_weights_string + "/n_" + seq_n_string[n] + "/X.csv"); 
//       X = XFile.toEigen();

//       // set model data
//       BlockFrame<double, int> df;
//       df.insert(OBSERVATIONS_BLK,  y);
//       df.insert(DESIGN_MATRIX_BLK, X);
      
//       model.setData(df);
//       model.setTolerances(tol_weights, tol_FPIRLS); 
//       // solve smoothing problem
//       model.init();


//       // define grid of lambda values
//       std::vector<SVector<1>> lambdas;
//       for(double x = -5.0; x <= -0.9; x +=0.20) lambdas.push_back(SVector<1>(std::pow(10,x)));  

//       // define GCV function and optimize
//       std::string stoch_type = "Chol_" ; 
//       std::size_t seed = 21;  
//       GCV<decltype(model), StochasticEDF<decltype(model)>> GCV(model, 1000, seed, StochasticEDFMethod::Cholesky);
//       GridOptimizer<1> opt;


//       ScalarField<1, decltype(GCV)> obj(GCV);

//       auto start = high_resolution_clock::now();
//       opt.optimize(obj, lambdas); // optimize gcv field
//       auto stop = high_resolution_clock::now();
//       // auto duration = duration_cast<microseconds>(stop - start);
 
//       // auto int_ms = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);

//       std::chrono::duration<double, std::milli> fp_ms = stop - start;


//       std::cout << "Duration Wood: "  << fp_ms.count() << "milliseconds " << std::endl;
//       myfile << std::setprecision(16) << fp_ms.count() << "\n" ;
 

//       SVector<1> best_lambda = opt.optimum();


//       std::cout << "Lambda optimal is: " << best_lambda[0] << std::endl ; 
//       // check optimal lambda
//       // EXPECT_TRUE( almost_equal(best_lambda[0], lambdas[4][0]) );

//       // Lambda opt
//       std::ofstream fileLambdaopt(R_path + "/R/Our/data/Test_" + 
//                   TestNumber + "/alpha_" + alpha_string + "/tol_weights_" + tol_weights_string  + "/n_" + seq_n_string[n] +
//                   "/GCV/Stochastic/LambdaCpp_" + stoch_type + alpha_string + ".csv");
//       if (fileLambdaopt.is_open()){
//         fileLambdaopt << std::setprecision(16) << best_lambda[0];
//         fileLambdaopt.close();
//       }

//       // Lambda vector
//       std::ofstream fileGCV_lambda(R_path + "/R/Our/data/Test_" + 
//                   TestNumber + "/alpha_" + alpha_string + "/tol_weights_" + tol_weights_string  + "/n_" + seq_n_string[n] + 
//                   "/GCV/Stochastic/GCV_lambdasCpp_" + stoch_type + alpha_string + ".csv");
//       for(std::size_t i = 0; i < lambdas.size(); ++i) 
//         fileGCV_lambda << std::setprecision(16) << lambdas[i] << "\n" ; 

//       fileGCV_lambda.close(); 

//       // GCV scores
//       std::ofstream fileGCV_scores(R_path + "/R/Our/data/Test_" + 
//                   TestNumber + "/alpha_" + alpha_string + "/tol_weights_" + tol_weights_string  + "/n_" + seq_n_string[n] + 
//                   "/GCV/Stochastic/GCV_scoresCpp_" + stoch_type + alpha_string + ".csv");
//       for(std::size_t i = 0; i < GCV.values().size(); ++i) 
//         fileGCV_scores << std::setprecision(16) << std::sqrt(GCV.values()[i]) << "\n" ; 

//       fileGCV_scores.close(); 


//       // Edf
//       std::ofstream fileGCV_edf(R_path + "/R/Our/data/Test_" + 
//                   TestNumber + "/alpha_" + alpha_string + "/tol_weights_" + tol_weights_string  + "/n_" + seq_n_string[n] + 
//                   "/GCV/Stochastic/GCV_edfCpp_" + stoch_type  + alpha_string + ".csv");
//       for(std::size_t i = 0; i < GCV.edfs().size(); ++i) 
//         fileGCV_edf << std::setprecision(16) << GCV.edfs()[i] << "\n" ; 

//       fileGCV_edf.close(); 
      
//     }

//    myfile.close();
//   }


//   // // Durations
//   // std::ofstream fileDuration(R_path + "/R/Our/data/Test_" + 
//   //             TestNumber + "/alpha_" + alpha_string + "/tol_weights_" + tol_weights_string  + "/n_" + seq_n_string[n] + "/DurationWood.csv");
//   // for(std::size_t i = 0; i < M; ++i) 
//   //     fileDuration << std::setprecision(16) << durations[i] << "\n" ; 

//   // fileDuration.close(); 
    
// }



/* test 18
   domain:       unit square [0,1] x [0,1] 
   sampling:     locations = nodes
   penalization: costant coefficients PDE
   covariates:   no
   BC:           no
   order FE:     1
   GCV optimization: grid exact
   Correspondent in R: Test_3 
 */
// TEST(GCV_SQRPDE, Test18_Laplacian_NonParametric_GeostatisticalAtNodes_GridExact) {
//   // define domain and regularizing PDE
//   MeshLoader<Mesh2D<>> domain("unit_square");
//   // non unitary diffusion tensor
//   SMatrix<2> K;
//   K << 1, 0., 0., 4;
//   auto L = Laplacian(K); // anisotropic diffusion
//   std::cout << "Number mesh elements: " << domain.mesh.elements() << std::endl; 
//   DMatrix<double> u = DMatrix<double>::Zero(domain.mesh.elements()*3, 1);
//   PDE problem(domain.mesh, L, u); // definition of regularizing PDE

//   double alpha = 0.01; 
//   unsigned int alpha_int = alpha*100; 
//   const std::string alpha_string = std::to_string(alpha_int); 
//   const std::string TestNumber = "3"; 

//   SQRPDE<decltype(problem), fdaPDE::models::GeoStatMeshNodes> model(problem, alpha);

//   // load data from .csv files
//   CSVReader<double> reader{};
//   CSVFile<double> yFile; // observation file
//   std::string data_macro_strategy_type = "skewed_data"; 
//   std::string data_strategy_type = "E"; 

//   // Marco
//   std::string R_path = "/mnt/c/Users/marco/OneDrive - Politecnico di Milano/Corsi/Magistrale/Anno_II_Semestre_II/PACS_project_shared"; 
  
//   // // Ilenia 
//   // std::string R_path = "/mnt/c/Users/ileni/OneDrive - Politecnico di Milano/PACS_project_shared"; 
  
//   std::vector<double> seq_tol_weights = {0.000001}; 
//   std::vector<std::string> seq_tol_weights_string = {"1e-06"};

//   std::vector<double> seq_tol_FPIRLS = {0.000001}; 
//   std::vector<std::string> seq_tol_FPIRLS_string = {"1e-06"}; 

//   std::string lin_sys_solver = "Chol";    // depends on the "symmetry" option in R 
//   std::string stopping_type = "our";

//   // define grid of lambda values
//   std::vector<SVector<1>> lambdas;
//   for(double x = -9.0; x <= -6.9; x +=0.2) lambdas.push_back(SVector<1>(std::pow(10,x)));
//   std::string GCV_type = "Exact"; 
//   unsigned int M = 10;      // number of simulations

// for(int m = 10; m <= M; ++m){
//   for(int i = 0; i < seq_tol_weights.size(); ++i ){

//     std::string tol_weights_string = seq_tol_weights_string[i]; 
//     double tol_weights = seq_tol_weights[i]; 

//       for(int j = 0; j < seq_tol_FPIRLS.size(); ++j){

//         std::string tol_FPIRLS_string = seq_tol_FPIRLS_string[j]; 
//         double tol_FPIRLS = seq_tol_FPIRLS[j]; 

//         std::string path_solutions = R_path + "/R/Our/data/Test_" + 
//                         TestNumber + "/alpha_" + alpha_string + "/" + data_macro_strategy_type + "/strategy_"  + data_strategy_type + 
//                         "/" + stopping_type + "/tol_weights_" + tol_weights_string + "/tol_FPIRLS_" + tol_FPIRLS_string + 
//                         "/" + lin_sys_solver + "/sim_" + std::to_string(m); 

//         std::cout << path_solutions << std::endl; 

//         std::string path_GCV = path_solutions + "/GCV/" + GCV_type;  

//         yFile = reader.parseFile(path_solutions + "/z.csv");          
//         DMatrix<double> y = yFile.toEigen();
        
//         // set model data
//         BlockFrame<double, int> df;
//         df.insert(OBSERVATIONS_BLK, y);
//         model.setData(df);
//         model.init(); // init model

//         model.setTolerances(tol_weights, tol_FPIRLS); 

//         // define GCV function and optimize
//         GCV<decltype(model), ExactEDF<decltype(model)>> GCV(model);
//         GridOptimizer<1> opt;

//         ScalarField<1, decltype(GCV)> obj(GCV);
//         opt.optimize(obj, lambdas); // optimize gcv field
//         SVector<1> best_lambda = opt.optimum();
        
//         // Lambda opt
//         std::ofstream fileLambdaopt(path_solutions + "/LambdaCpp.csv");
//         if (fileLambdaopt.is_open()){
//           fileLambdaopt << std::setprecision(16) << best_lambda[0];
//           fileLambdaopt.close();
//         }


//         // // Lambda vector
//         // std::ofstream fileGCV_lambda("data/models/SQRPDE/2D_test1_GCV/Eaxct/GCV_lambdasCpp_" + alpha_string + ".csv");
//         // for(std::size_t i = 0; i < lambdas.size(); ++i) 
//         //   fileGCV_lambda << std::setprecision(16) << lambdas[i] << "\n"; 

//         // fileGCV_lambda.close(); 

//         // GCV scores
//         std::ofstream fileGCV_scores(path_GCV + "/GCV_scoresCpp.csv");
//         for(std::size_t i = 0; i < GCV.values().size(); ++i) 
//           fileGCV_scores << std::setprecision(16) << std::sqrt(GCV.values()[i]) << "\n" ; 
//         fileGCV_scores.close(); 


//       //   // Edf
//       //   std::ofstream fileGCV_edf(R_path + "/R/Our/data/Test_" + 
//       //               TestNumber + "/alpha_" + alpha_string + "/" + data_macro_strategy_type + "/strategy_"  + data_strategy_type + 
//       //               "/" + stopping_type + "/tol_weights_" + tol_weights_string + "/tol_FPIRLS_" + tol_FPIRLS_string + 
//       //               "/" + lin_sys_solver + "/GCV/Exact/GCV_edfCpp_" + alpha_string + ".csv");
//       //   for(std::size_t i = 0; i < GCV.edfs().size(); ++i) 
//       //     fileGCV_edf << std::setprecision(16) << GCV.edfs()[i] << "\n" ; 

//       //   fileGCV_edf.close(); 
//       }
//   }
// }


// }




/* test 19
   domain:       unit sphere 
   sampling:     locations != nodes
   penalization: laplacian
   covariates:   yes
   BC:           no
   order FE:     1
   GCV optimization: grid exact
   Correspondent in R: Test_7
 */
// TEST(GCV_SQRPDE, Test19_Laplacian_SemiParametric_GeostatisticalAtLocations_GridExact) {
//   // define domain and regularizing PDE
//   MeshLoader<Mesh3D<>> domain("unit_sphere");

//   auto L = Laplacian();
//   DMatrix<double> u = DMatrix<double>::Zero(domain.mesh.elements()*6, 1);
//   PDE problem(domain.mesh, L, u); // definition of regularizing PDE


//   double alpha = 0.1; 
//   const std::string alpha_string = "10"; 
//   const std::string TestNumber = "7"; 

//   SQRPDE<decltype(problem), fdaPDE::models::GeoStatLocations> model(problem, alpha);

//   // load data from .csv files
//   CSVReader<double> reader{};
//   CSVFile<double> yFile; 
//   CSVFile<double> XFile;
//   DMatrix<double> y;  
//   DMatrix<double> X;
//   DMatrix<double> loc;
//   CSVFile<double> locFile; // observation file

//   // Marco
//   std::string R_path = "/mnt/c/Users/marco/OneDrive - Politecnico di Milano/Corsi/Magistrale/Anno_II_Semestre_II/PACS_project_shared"; 
  
//   // Ilenia 
//   //std::string R_path = "/mnt/c/Users/ileni/OneDrive - Politecnico di Milano/PACS_project_shared"; 
  
//   double tol_weights = 0.000001; 
//   std::string tol_weights_string = "1e-06";

//   double tol_FPIRLS = 0.000001; 
//   std::string tol_FPIRLS_string = "1e-06"; 

//   // define grid of lambda values
//   std::vector<SVector<1>> lambdas;
//   for(double x = -5.0; x <= -1.9; x +=0.2) lambdas.push_back(SVector<1>(std::pow(10,x)));

//   unsigned int M = 30;

//   // load locations where data are sampled
//   locFile = reader.parseFile(R_path + "/R/Our/data/Test_" 
//                       + TestNumber + "/alpha_" + alpha_string + "/locs.csv");

//   loc = locFile.toEigen();
//   model.set_spatial_locations(loc); 

//   for(std::size_t m=1; m<=M; m++) {

//     yFile = reader.parseFile(R_path + "/R/Our/data/Test_" + 
//                     TestNumber + "/alpha_" + alpha_string + "/sim_" + std::to_string(m) + "/z.csv");             
//     y = yFile.toEigen();

//     XFile = reader.parseFile(R_path + "/R/Our/data/Test_" + 
//                     TestNumber + "/alpha_" + alpha_string + "/sim_" + std::to_string(m) + "/X.csv");             
//     X = XFile.toEigen();

//     // set model data
//     BlockFrame<double, int> df;
//     df.insert(OBSERVATIONS_BLK, y);
//     df.insert(DESIGN_MATRIX_BLK, X);

//     model.setData(df);
//     model.init(); // init model

//     model.setTolerances(tol_weights, tol_FPIRLS); 

//     // define GCV function and optimize
//     GCV<decltype(model), ExactEDF<decltype(model)>> GCV(model);
//     GridOptimizer<1> opt;

//     ScalarField<1, decltype(GCV)> obj(GCV);
//     opt.optimize(obj, lambdas); // optimize gcv field
//     SVector<1> best_lambda = opt.optimum();
    
//     std::cout << "Lambda optimal is: " << best_lambda[0] << std::endl; 
//     // Lambda opt
//     std::ofstream fileLambdaopt(R_path + "/R/Our/data/Test_" + 
//                     TestNumber + "/alpha_" + alpha_string + "/sim_" + std::to_string(m) + 
//                     "/LambdaCpp.csv");
//     if (fileLambdaopt.is_open()){
//       fileLambdaopt << std::setprecision(16) << best_lambda[0];
//       fileLambdaopt.close();
//     }


//     // Lambda vector
//     std::ofstream fileGCV_lambda(R_path + "/R/Our/data/Test_" + 
//                 TestNumber + "/alpha_" + alpha_string + "/sim_" + std::to_string(m) + "/GCV/GCV_lambdasCpp.csv");
//     for(std::size_t i = 0; i < lambdas.size(); ++i) 
//       fileGCV_lambda << std::setprecision(16) << lambdas[i] << "\n" ; 

//     fileGCV_lambda.close(); 

//     // GCV scores
//     std::ofstream fileGCV_scores(R_path + "/R/Our/data/Test_" + 
//                 TestNumber + "/alpha_" + alpha_string + "/sim_" + std::to_string(m) + "/GCV/GCV_scoresCpp.csv");
//     for(std::size_t i = 0; i < GCV.values().size(); ++i) 
//       fileGCV_scores << std::setprecision(16) << std::sqrt(GCV.values()[i]) << "\n" ; 

//     fileGCV_scores.close(); 

//   }

// }


/* test 20
   domain:       linear network 
   sampling:     locations = nodes
   penalization: laplacian
   covariates:   no
   BC:           no
   order FE:     1
   GCV optimization: grid exact
   Correspondent in R: Test_9
 */
// TEST(GCV_SQRPDE, Test20_Laplacian_NonParametric_GeostatisticalAtNodes_GridExact) {

//   const std::string TestNumber = "9"; 
  
//   // define domain and regularizing PDE
//   MeshLoader<NetworkMesh<>> domain("network");
//   auto L = Laplacian();
//   DMatrix<double> u = DMatrix<double>::Zero(domain.mesh.elements()*3, 1);
//   PDE problem(domain.mesh, L, u); // definition of regularizing PDE


//   double alpha = 0.9; 
//   unsigned int alpha_int = alpha*100; 
//   const std::string alpha_string = std::to_string(alpha_int);
//   SQRPDE<decltype(problem), fdaPDE::models::GeoStatMeshNodes> model(problem, alpha);
//   //GeoStatLocations

//   // load data from .csv files
//   CSVReader<double> reader{};
//   CSVFile<double> yFile; 
//   DMatrix<double> y;  

//   // Marco
//   std::string R_path = "/mnt/c/Users/marco/OneDrive - Politecnico di Milano/Corsi/Magistrale/Anno_II_Semestre_II/PACS_project_shared"; 
//   // Ilenia 
//   //std::string R_path = "/mnt/c/Users/ileni/OneDrive - Politecnico di Milano/PACS_project_shared"; 
  
//   double tol_weights = 0.000001; 
//   double tol_FPIRLS = 0.000001; 
//   model.setTolerances(tol_weights, tol_FPIRLS); 
   
//   std::string GCV_type = "Exact";
//   // define grid of lambda values
//   std::vector<SVector<1>> lambdas;
//   for(double x = -8.0; x <= -4.9; x +=0.2) lambdas.push_back(SVector<1>(std::pow(10,x)));

//   unsigned int M = 10;
//   for(std::size_t m = 1; m <= M; m++) {

//     std::string path_solutions = R_path + "/R/Our/data/Test_" + 
//                     TestNumber + "/alpha_" + alpha_string + "/c_network/sim_" + std::to_string(m); 
//     std::string path_GCV =  path_solutions + "/GCV/" + GCV_type;                

//     yFile = reader.parseFile(path_solutions + "/z.csv");             
//     y = yFile.toEigen();

//     // set model data
//     BlockFrame<double, int> df;
//     df.insert(OBSERVATIONS_BLK, y);

//     model.setData(df);
//     model.init(); // init model

//     model.setTolerances(tol_weights, tol_FPIRLS); 

//     // define GCV function and optimize
//     GCV<decltype(model), ExactEDF<decltype(model)>> GCV(model);
//     GridOptimizer<1> opt;

//     ScalarField<1, decltype(GCV)> obj(GCV);
//     opt.optimize(obj, lambdas); // optimize gcv field
//     SVector<1> best_lambda = opt.optimum();
    
//     std::cout << "Lambda optimal is: " << best_lambda[0] << std::endl; 
//     // Lambda opt
//     std::ofstream fileLambdaopt(path_solutions + 
//                     "/LambdaCpp.csv");
//     if (fileLambdaopt.is_open()){
//       fileLambdaopt << std::setprecision(16) << best_lambda[0];
//       fileLambdaopt.close();
//     }
//     // Lambda vector
//     std::ofstream fileGCV_lambda(path_GCV + "/Lambdas.csv");
//     for(std::size_t i = 0; i < lambdas.size(); ++i) 
//       fileGCV_lambda << std::setprecision(16) << lambdas[i] << "\n" ; 

//     fileGCV_lambda.close(); 
//     // GCV scores
//     std::ofstream fileGCV_scores(path_GCV + "/GCV_scoresCpp.csv");
//     for(std::size_t i = 0; i < GCV.values().size(); ++i) 
//       fileGCV_scores << std::setprecision(16) << std::sqrt(GCV.values()[i]) << "\n" ; 

//     fileGCV_scores.close(); 

//   }

// }


// /* test 22
//    domain:       quasicircular domain
//    sampling:     areal
//    penalization: laplacian 
//    covariates:   no
//    BC:           yes
//    order FE:     1
//    GCV optimization: grid exact
//  */
// TEST(GCV_SQRPDE, Test22_NonCostantCoefficientsPDE_NonParametric_Areal_GridExact) {

//   const std::string TestNumber = "4";

//   double alpha = 0.5; 
//   unsigned int alpha_int = alpha*100; 
//   const std::string alpha_string = std::to_string(alpha_int);

//   // define domain and regularizing PDE
//   MeshLoader<Mesh2D<>> domain("quasi_circle");

//   // load PDE coefficients data
//   auto L = Laplacian();
//   DMatrix<double> u = DMatrix<double>::Zero(domain.mesh.elements()*3, 1);
//   PDE problem(domain.mesh, L, u); // definition of regularizing PDE

//   std::string GCV_type = "Exact";

//   // Marco
//   std::string R_path = "/mnt/c/Users/marco/OneDrive - Politecnico di Milano/Corsi/Magistrale/Anno_II_Semestre_II/PACS_project_shared"; 
//   // Ilenia 
//   //std::string R_path = "/mnt/c/Users/ileni/OneDrive - Politecnico di Milano/PACS_project_shared"; 
//   std::string path_solutions = R_path + "/R/Our/data/Test_" + TestNumber + "/alpha_" + alpha_string; 
//   std::string path_GCV = path_solutions + "/GCV/" + GCV_type; 
  
//   // define statistical model
//   CSVReader<double> reader{};
//   CSVReader<int> int_reader{};
//   CSVFile<int> arealFile; // incidence matrix for specification of subdomains
//   arealFile = int_reader.parseFile(path_solutions + "/incidence_matrix.csv");
//   DMatrix<int> areal = arealFile.toEigen();

//   SQRPDE<decltype(problem), fdaPDE::models::Areal> model(problem, alpha);
//   model.set_spatial_locations(areal);
  
//   // load data from .csv files
//   CSVFile<double> yFile; // observation file
//   std::cout << "here 3" << std::endl;
//   yFile = reader.parseFile(path_solutions + "/z.csv");
//   std::cout << "here 4" << std::endl;
//   DMatrix<double> y = yFile.toEigen();
  
//   // set model data
//   BlockFrame<double, int> df;
//   df.insert(OBSERVATIONS_BLK, y);
//   model.setData(df);
//   model.init(); // init model

//   // define grid of lambda values
//   std::vector<SVector<1>> lambdas;
//   for(double x = -6.0; x <= -3.0; x +=0.2) lambdas.push_back(SVector<1>(std::pow(10,x)));
  
//   // define GCV calibrator
//   GCV<decltype(model), ExactEDF<decltype(model)>> GCV(model);
//   GridOptimizer<1> opt;

//   ScalarField<1, decltype(GCV)> obj(GCV);
//   opt.optimize(obj, lambdas); // optimize gcv field
//   SVector<1> best_lambda = opt.optimum();
//   std::cout << "Lambda optimal is: " << best_lambda[0] << std::endl; 

//   // Lambda opt
//   std::ofstream fileLambdaopt(path_solutions + 
//                   "/LambdaCpp.csv");
//   if (fileLambdaopt.is_open()){
//     fileLambdaopt << std::setprecision(16) << best_lambda[0];
//     fileLambdaopt.close();
//   }

//   // Lambda vector
//   std::ofstream fileGCV_lambda(path_GCV + "/Lambdas.csv");
//   for(std::size_t i = 0; i < lambdas.size(); ++i) 
//     fileGCV_lambda << std::setprecision(16) << lambdas[i] << "\n" ; 
//   fileGCV_lambda.close(); 

//   // GCV scores
//   std::ofstream fileGCV_scores(path_GCV + "/GCV_scoresCpp.csv");
//   for(std::size_t i = 0; i < GCV.values().size(); ++i) 
//     fileGCV_scores << std::setprecision(16) << std::sqrt(GCV.values()[i]) << "\n" ; 

//   fileGCV_scores.close(); 

// }

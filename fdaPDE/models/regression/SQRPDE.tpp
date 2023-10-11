// finds a solution to the SQR-PDE smoothing problem
template <typename PDE, typename RegularizationType, typename SamplingDesign, typename Solver>
void SQRPDE<PDE, RegularizationType, SamplingDesign, Solver>::solve() {
  // execute FPIRLS for minimization of the functional
  // \norm{V^{-1/2}(y - \mu)}^2 + \lambda \int_D (Lf - u)^2
   
  FPIRLS<decltype(*this)> fpirls(*this, tol_, max_iter_); // FPIRLS engine

  fpirls.compute();
  // fpirls converged: extract matrix P and solution estimates
  // W_ = fpirls.weights().asDiagonal();
  W_ = fpirls.solver().W();

  if(hasCovariates()) {
    XtWX_ = X().transpose()*W_*X(); 
    invXtWX_ = XtWX_.partialPivLu();
  }

  invA_ = fpirls.solver().invA();

  if(hasCovariates()) {
    U_ = fpirls.solver().U(); 
    V_ = fpirls.solver().V(); 
  }

  f_ = fpirls.solver().f();
  g_ = fpirls.solver().g(); 
  n_iter_ = fpirls.n_iter();   
 
  if(hasCovariates()) beta_ = fpirls.solver().beta();
  return;
}


// Non-parametric and semi-parametric cases coincide here, since beta^(0) = 0
template <typename PDE, typename RegularizationType, typename SamplingDesign, typename Solver>
DVector<double> 
SQRPDE<PDE, RegularizationType, SamplingDesign, Solver>::initialize_mu() {  // M tolto const per il caso time perchè assembla u

  DVector<double> f;
  f.resize(Base::n_temporal_locs() * Base::n_basis()); 

  // correction on the matrix if time is present
  if constexpr(std::is_same<RegularizationType, SpaceOnly>::value){
    std::cout << "I'm SpaceOnly" << std::endl ; 
    SparseBlockMatrix<double,2,2>
    A_init(PsiTD()*Psi()/n_obs(), 2*lambdaS()*R1().transpose(),
           lambdaS()*R1(),     -lambdaS()*R0()            );

    // cache non-parametric matrix and its factorization for reuse 
    fdaPDE::SparseLU<SpMatrix<double>> invA_init;
    invA_init.compute(A_init);
    DVector<double> b_init; 
    b_init.resize(A_init.rows());

    b_init.block(Base::n_basis(),0, Base::n_temporal_basis()*Base::n_basis(),1) = lambdaS()*u();  
    b_init.block(0,0, Base::n_basis(),1) = PsiTD()*y()/n_obs(); 
    BLOCK_FRAME_SANITY_CHECKS;
    f = (invA_init.solve(b_init)).head(Base::n_basis());         
  }
  else{
    if constexpr(std::is_same<RegularizationType, SpaceTimeSeparable>::value){
      std::cout << "I'm SpaceTimeSeparable" << std::endl; 
      SparseBlockMatrix<double,2,2>
        A_init(PsiTD()*Psi()/n_obs() - Base::lambdaT()*Kronecker(Base::Pt(), pde().R0()), 2*lambdaS()*R1().transpose(),
               lambdaS()*R1(),                                                -lambdaS()*R0()                           );
        
        // cache non-parametric matrix and its factorization for reuse 
        fdaPDE::SparseLU<SpMatrix<double>> invA_init;
        invA_init.compute(A_init);
        DVector<double> b_init; 
        b_init.resize(A_init.rows());

        b_init.block(Base::n_temporal_basis()*Base::n_basis(),0, Base::n_temporal_basis()*Base::n_basis(),1) = lambdaS()*u();  
        b_init.block(0,0, Base::n_temporal_basis()*Base::n_basis(),1) = PsiTD()*y()/n_obs(); 
        BLOCK_FRAME_SANITY_CHECKS;
        f = (invA_init.solve(b_init)).head(Base::n_temporal_basis()*Base::n_basis());  
    }

    if constexpr(std::is_same<RegularizationType, SpaceTimeParabolic>::value){
      std::cout << "I'm SpaceTimeParabolic" << std::endl;

      if constexpr(std::is_same<Solver, MonolithicSolver>::value){
        std::cout << "...monolithic solver" << std::endl;
        SparseBlockMatrix<double,2,2> 
           A_init(PsiTD()*Psi()/n_obs(),                                              2*lambdaS()*(R1() + Base::lambdaT()*Kronecker(Base::L(), pde().R0())).transpose(),
                 lambdaS()*(R1() + Base::lambdaT()*Kronecker(Base::L(), pde().R0())), -lambdaS()*R0());

        fdaPDE::SparseLU<SpMatrix<double>> invA_init;
        invA_init.compute(A_init);
        DVector<double> b_init; 
        b_init.resize(A_init.rows());         
        b_init.block(Base::n_temporal_locs()*Base::n_basis(),0, Base::n_temporal_locs()*Base::n_basis(),1) = lambdaS()*u();  
        b_init.block(0,0, Base::n_temporal_locs()*Base::n_basis(),1) = PsiTD()*y()/n_obs(); 
        BLOCK_FRAME_SANITY_CHECKS;
        f = (invA_init.solve(b_init)).head(Base::n_temporal_locs()*Base::n_basis());    

      } 
      else{

        std::cout << "...iterative solver" << std::endl;

        SparseBlockMatrix<double,2,2> 
           A_init(PsiTD()*Psi()/n_obs(), 2*lambdaS()*R1(),
                 lambdaS()*R1(),         -lambdaS()*R0());

        std::cout << "dim A_init: " << A_init.rows() << ", " << A_init.cols() << std::endl;          

        fdaPDE::SparseLU<SpMatrix<double>> invA_init;
        invA_init.compute(A_init);
        DVector<double> b_init; 
        b_init.resize(A_init.rows());        
        std::cout << "dim b_init: " << b_init.size() << std::endl;   
        std::cout << "dim y(): " << y().size() << std::endl;   // ---> da dove esce 500 ?? 
        std::cout << "n*m: " << Base::n_temporal_locs()*Base::n_basis() << std::endl; 
        // DVector<double> u_init; 
        // u_init.resize(Base::n_temporal_locs()Base::n_basis());
        // for(unsigned int i = 0; i < Base::n_temporal_locs(); ++i)
        //   u_init.block(0,0, model().n_basis(),1) = u(i); 

        b_init.block(Base::n_basis(),0, Base::n_basis(),1) = DVector<double>::Zero(Base::n_basis()); //lambdaS()*u();  
        b_init.block(0,0, Base::n_basis(),1) = PsiTD()*y()/n_obs(); 
        BLOCK_FRAME_SANITY_CHECKS;
        std::cout << "after sanity check " << std::endl; 
        f = (invA_init.solve(b_init)).head(Base::n_temporal_locs()*Base::n_basis()); 
        std::cout << "after f " << std::endl; 
                   
      } 
    }
  }

  DVector<double> fn =  Psi(not_nan())*f;
  return fn;
  
}

template <typename PDE, typename RegularizationType, typename SamplingDesign, typename Solver>
std::tuple<DVector<double>&, DVector<double>&>
SQRPDE<PDE, RegularizationType, SamplingDesign, Solver>::compute(const DVector<double>& mu){
  // compute weight matrix and pseudo-observation vector
  DVector<double> abs_res{};
  abs_res.resize(y().size()); 

  for(int i = 0; i < y().size(); ++i)
    abs_res(i) = std::abs(y()(i) - mu(i));   

  pW_.resize(n_obs());
  for(int i = 0; i < y().size(); ++i) {
    if (abs_res(i) < tol_weights_){
      pW_(i) = ( 1./(abs_res(i) + tol_weights_) )/(2.*n_obs());

    }    
    else
      pW_(i) = (1./abs_res(i))/(2.*n_obs()); 
  }
 
  py_ = y() - (1 - 2.*alpha_)*abs_res;

  return std::tie(pW_, py_);
}



template <typename PDE, typename RegularizationType, typename SamplingDesign, typename Solver>
double
SQRPDE<PDE, RegularizationType, SamplingDesign, Solver>::model_loss(const DVector<double>& mu) const{
  
  // compute value of functional J given mu: /(2*n) 
    return (pW_.cwiseSqrt().matrix().asDiagonal()*(py_ - mu)).squaredNorm();     
}

// required to support GCV based smoothing parameter selection
// in case of an SRPDE model we have T = \Psi^T*Q*\Psi + \lambda*(R1^T*R0^{-1}*R1)
template <typename PDE, typename RegularizationType, typename SamplingDesign, typename Solver>
const DMatrix<double>& SQRPDE<PDE, RegularizationType, SamplingDesign, Solver>::T() {
  // // compute value of R = R1^T*R0^{-1}*R1, cache for possible reuse
  // if(R_.size() == 0){
  //     invR0_.compute(R0());
  //     R_ = R1().transpose()*invR0_.solve(R1());
  // }
  // // compute and store matrix T for possible reuse
  // if(!hasCovariates()) // case without covariates, Q is the identity matrix
  //   T_ = PsiTD()*W()*Psi()   + lambdaS()*R_;
  // else // general case with covariates
  //   T_ = PsiTD()*lmbQ(Psi()) + lambdaS()*R_;

  if constexpr(!std::is_same<Solver, IterativeSolver>::value){   // M: iterative non ha pen()
    if(!hasCovariates()) // case without covariates, Q is the identity matrix
      T_ = PsiTD()*W()*Psi()  + Base::pen();
    else {// general case with covariates
      T_ = PsiTD()*lmbQ(Psi()) + Base::pen();
  }

    // fdaPDE::SparseLU<SpMatrix<double>> invR0_temp ;
    // invR0_temp.compute(pde().R0());
    // auto penS_temp = Kronecker(pde().R1().transpose()*invR0_temp.solve(pde().R1()), Base::Rt()); // (R1^T*R0^{-1}*R1) \kron Rt
    // auto penT_temp = Kronecker(pde().R0(), Base::Pt()); // (R0 \kron Pt)
    
    //   T_ = PsiTD()*lmbQ(Psi()) + lambdaS()*penS_temp + lambdaT()*penT_temp;
  }


  return T_;
}

// Q is computed on demand only when it is needed by GCV and cached for fast reacess (in general operations
// involving Q can be substituted with the more efficient routine lmbQ(), which is part of iRegressionModel interface)
template <typename PDE, typename RegularizationType, typename SamplingDesign, typename Solver>
const DMatrix<double>& SQRPDE<PDE, RegularizationType, SamplingDesign, Solver>::Q() {
  // compute Q = W(I - H) = W ( I - X*(X^T*W*X)^{-1}*X^T*W ) 
  Q_ = W()*(DMatrix<double>::Identity(n_obs(), n_obs()) - X()*invXtWX().solve(X().transpose()*W()));

  return Q_;
}


// returns the numerator of the GCV score 
template <typename PDE, typename RegularizationType, typename SamplingDesign, typename Solver>
double SQRPDE<PDE, RegularizationType, SamplingDesign, Solver>::norm
(const DMatrix<double>& fitted, const DMatrix<double>& obs) const {   
  double result = 0;
  for(std::size_t i = 0; i < obs.rows(); ++i)
    result += rho_alpha(obs.coeff(i,0) - fitted.coeff(i,0));
  return result*result / n_obs() ;
}

// returns the pinball loss at a specific x 
template <typename PDE, typename RegularizationType, typename SamplingDesign, typename Solver>
double SQRPDE<PDE, RegularizationType, SamplingDesign, Solver>::rho_alpha(const double& x) const{ 
  return 0.5*std::abs(x) + (alpha_ - 0.5)*x; 
}




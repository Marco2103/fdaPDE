// finds a solution to the SQR-PDE smoothing problem
template <typename PDE, typename SamplingDesign>
void SQRPDE<PDE, SamplingDesign>::solve() {
  // execute FPIRLS for minimization of the functional
  // \norm{V^{-1/2}(y - \mu)}^2 + \lambda \int_D (Lf - u)^2

  std::cout << std::endl;
  std::cout << "Lambda: " << lambdaS() << std::endl; 
  

  FPIRLS<decltype(*this)> fpirls(*this, tol_, max_iter_); // FPIRLS engine

  fpirls.compute();
  
  // fpirls converged: extract matrix P and solution estimates
  // W_ = fpirls.weights().asDiagonal();
  W_ = fpirls.solver().W();

  if(hasCovariates()) {
    XtWX_ = X().transpose()*W_*X(); 
    invXtWX_ = XtWX_.partialPivLu();
  }
  
  // A_ =    fpirls.solver().A(); 
  invA_ = fpirls.solver().invA();
  

  if(hasCovariates()) {
    U_ =    fpirls.solver().U(); 
    V_ =    fpirls.solver().V(); 

  }

  f_ = fpirls.solver().f();
  g_ = fpirls.solver().g();   // per completezza (non sappiamo se lo usa)
 
  Jfinal_sqrpde_ = fpirls.J_final();
  niter_sqrpde_ = fpirls.n_iter();

  if(hasCovariates()) beta_ = fpirls.solver().beta();
  return;
}


// Non-parametric and semi-parametric cases coincide here, since beta^(0) = 0
template <typename PDE, typename SamplingDesign>
DVector<double> 
SQRPDE<PDE, SamplingDesign>::initialize_mu() const {

  // assemble system matrix 
  SparseBlockMatrix<double,2,2>
    A_temp(PsiTD()*Psi()/n_obs(), 2*lambdaS()*R1().transpose(),
      lambdaS()*R1(),     -lambdaS()*R0()            );
  // cache non-parametric matrix and its factorization for reuse 
  fdaPDE::SparseLU<SpMatrix<double>> invA_temp;
  invA_temp.compute(A_temp);

  DVector<double> b_temp ; 
  b_temp.resize(A_temp.rows());
  b_temp.block(n_basis(),0, n_basis(),1) = 0.*u();
  b_temp.block(0,0, n_basis(),1) = PsiTD()*y()/n_obs() ; 
  BLOCK_FRAME_SANITY_CHECKS;
  DVector<double> f = (invA_temp.solve(b_temp)).head(n_basis());
  DVector<double> fn = Psi()*f ; 

  // implementazione diretta

  // fdaPDE::SparseLU<SpMatrix<double>> invR0_temp ; 
  // invR0_temp.compute(R0());

  // SpMatrix<double> A_temp{} ; 
  // // A_temp = PsiTD()*Psi()/n_obs() + 2*lambdaS()*R1().transpose()*invR0_temp.solve(R1()) ; 
  // A_temp = PsiTD()*Psi() + 2*n_obs()*lambdaS()*R1().transpose()*invR0_temp.solve(R1()) ; 
  // fdaPDE::SparseLU<SpMatrix<double>> invA_temp ;
  // // invA_temp.compute( A_temp.derived() );

  // invA_temp.compute( A_temp );
  
  // DVector<double> b_temp{} ; 
  // // b_temp = PsiTD()*y()/n_obs() ; 
  // b_temp = PsiTD()*y() ; 

  // DVector<double> f = (invA_temp.solve(b_temp)) ;
  // DVector<double> fn = Psi()*f ; 

  return fn ; 
}



template <typename PDE, typename SamplingDesign>
std::tuple<DVector<double>&, DVector<double>&>
SQRPDE<PDE, SamplingDesign>::compute(const DVector<double>& mu) {
  // compute weight matrix and pseudo-observation vector
  DVector<double> abs_res{} ;
  abs_res.resize(y().size()) ; 

  for(int i = 0; i < y().size(); ++i)
    abs_res(i) = std::abs(y()(i) - mu(i)) ;   

  pW_.resize(n_obs());

  for(int i = 0; i < y().size(); ++i) {
    if (abs_res(i) < tol_weights_){
      pW_(i) = ( 1./(abs_res(i) + tol_weights_ ) )/(2.*n_obs());

    }    
    else
      pW_(i) = (1./abs_res(i))/(2.*n_obs()); 
  }
 
  py_ = y() - (1 - 2.*alpha_)*abs_res;
  return std::tie(pW_, py_);
}



template <typename PDE, typename SamplingDesign>
double
SQRPDE<PDE, SamplingDesign>::model_loss(const DVector<double>& mu) {
  
  // compute value of functional J given mu: /(2*n) 
    return (pW_.cwiseSqrt().matrix().asDiagonal()*(py_ - mu)).squaredNorm() ;    // serve .matrix() dopo cwiseSqrt() ? 
    // differentemente da GSRPDE, in cui la sequenza dei calcoli è   
        // array() --> sqrt() --> .. --> matrix() ---> .asDiagonal() 
    // noi chiamiamo cwiseSqrt che opera component wise sul vettore (quindi è lo stesso) e poi chiamiamo la view asDiagonal   
}

// required to support GCV based smoothing parameter selection
// in case of an SRPDE model we have T = \Psi^T*Q*\Psi + \lambda*(R1^T*R0^{-1}*R1)
template <typename PDE, typename SamplingDesign>
const DMatrix<double>& SQRPDE<PDE, SamplingDesign>::T() {
  // compute value of R = R1^T*R0^{-1}*R1, cache for possible reuse
  if(R_.size() == 0){
    if(!massLumping()){
      std::cout << "In NON mass lumping" << std::endl; 
      invR0_.compute(R0());
      R_ = R1().transpose()*invR0_.solve(R1());
    } else{
        std::cout << "In mass lumping" << std::endl; 
        DVector<double> lumped_invR0;
        lumped_invR0.resize(R0().cols()); 
        for(std::size_t j = 0; j < R0().cols(); ++j)    // M: troppe chiamate al getter di R0?? 
          lumped_invR0[j] = 1 / R0().col(j).sum();  
        lumped_invR0_ = lumped_invR0.asDiagonal();
        R_ = R1().transpose()*lumped_invR0_*R1();
    }

  }
  // compute and store matrix T for possible reuse
  if(!hasCovariates()) // case without covariates, Q is the identity matrix
    T_ = PsiTD()*W()*Psi()   + lambdaS()*R_;
  else // general case with covariates
    T_ = PsiTD()*lmbQ(Psi()) + lambdaS()*R_;
  return T_;
}

// Q is computed on demand only when it is needed by GCV and cached for fast reacess (in general operations
// involving Q can be substituted with the more efficient routine lmbQ(), which is part of iRegressionModel interface)
template <typename PDE, typename SamplingDesign>
const DMatrix<double>& SQRPDE<PDE, SamplingDesign>::Q() {
  // if(Q_.size() == 0){ // Q is computed on request since not needed in general
    // compute Q = W(I - H) = W ( I - X*(X^T*W*X)^{-1}*X^T*W ) 
    Q_ = W()*(DMatrix<double>::Identity(n_obs(), n_obs()) - X()*invXtWX().solve(X().transpose()*W()));
    // W_inQ_ = W() ; 
  // }
  return Q_;
}


// returns the numerator of the GCV score 
template <typename PDE, typename SamplingDesign>
double SQRPDE<PDE, SamplingDesign>::norm
(const DMatrix<double>& fitted, const DMatrix<double>& obs) const {   // CONTROLLA ORDINE degli input 
  double result = 0;
  for(std::size_t i = 0; i < obs.rows(); ++i)
    result += rho_alpha(obs.coeff(i,0) - fitted.coeff(i,0));
  return result*result / n_obs() ;
}

// returns the pinball loss at a specific x 
template <typename PDE, typename SamplingDesign>
double SQRPDE<PDE, SamplingDesign>::rho_alpha(const double& x) const{  // , const double& eps = 0.0) const{
  // if(eps < std::numeric_limits<double>::epsilon){
  //   return 0.5*std::abs(x) + (alpha_ - 0.5)*x; 
  // } else{
  //   return (alpha_ - 1)*x + eps*std::log1p(std::exp(x / eps));  
  // }

  return 0.5*std::abs(x) + (alpha_ - 0.5)*x; 
  
}




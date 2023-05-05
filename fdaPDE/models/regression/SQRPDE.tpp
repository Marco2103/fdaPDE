
// finds a solution to the SQR-PDE smoothing problem
template <typename PDE, Sampling SamplingDesign>
void SQRPDE<PDE, SamplingDesign>::solve() {
  // execute FPIRLS for minimization of the functional
  // \norm{V^{-1/2}(y - \mu)}^2 + \lambda \int_D (Lf - u)^2

  std::cout << "inizio FPIRLS fatto" << std::endl ; 

  FPIRLS<decltype(*this)> fpirls(*this, tol_, max_iter_); // FPIRLS engine

  std::cout << "chiama compute" << std::endl ; 

  fpirls.compute();

  std::cout << "post compute " << std::endl ; 
  
  // fpirls converged: extract matrix P and solution estimates
  std::cout << "W " << std::endl ; 
  W_ =    fpirls.weights().asDiagonal();

  if(hasCovariates()) {
    std::cout << "XtWX " << std::endl ; 
    XtWX_ = X().transpose()*W_*X(); 
    std::cout << "invXtWX " << std::endl ; 
    invXtWX_ = XtWX_.partialPivLu();

  }
  
  std::cout << "invA " << std::endl ; 
  //A_ =    fpirls.solver().A(); 
  invA_ = fpirls.solver().invA();

  if(hasCovariates()) {
    std::cout << "U " << std::endl ; 
    U_ =    fpirls.solver().U(); 
    std::cout << "V " << std::endl ; 
    V_ =    fpirls.solver().V(); 

  }
  

  std::cout << "bha " << std::endl ; 

  f_ = fpirls.f();
  std::cout << "ultimo " << std::endl ; 

  if(hasCovariates()) beta_ = fpirls.beta();
  return;
}

template <typename PDE, Sampling SamplingDesign>
DVector<double> 
SQRPDE<PDE, SamplingDesign>::initialize_mu() const {

  std::cout << "Entra " << std::endl ; 
  
  std::cout << "Calcolo R0inv_temp " << std::endl ;
  fdaPDE::SparseLU<SpMatrix<double>> invR0_temp ; 
  invR0_temp.compute(R0());
  std::cout << "Fine Calcolo R0inv_temp " << std::endl ;
  SpMatrix<double> A_temp = PsiTD()*Psi() + 2*n_obs()*lambdaS()*R1().transpose()*invR0_temp.solve(R1()) ; // factor of 2 ? 

  std::cout << "A_temp ok " << std::endl ; 

  fdaPDE::SparseLU<SpMatrix<double>> invA_temp ; 
  invA_temp.compute(A_temp);

  std::cout << "invA_temp ok " << std::endl ; 

  // FullPivLU<DMatrix<double>> lu(A_temp) ; 
  DVector<double> b_temp = PsiTD()*y() ; 
  std::cout << "b_temp ok " << std::endl ; 
  DVector<double> f = invA_temp.solve(b_temp);
  std::cout << "f_temp ok " << std::endl ; 
  DVector<double> fn = Psi()*f ; 
    
  std::cout << "return  " << std::endl ; 
  return fn ; 

}

template <typename PDE, Sampling SamplingDesign>
std::tuple<DVector<double>&, DVector<double>&>
SQRPDE<PDE, SamplingDesign>::compute(const DVector<double>& mu) {
  // compute weight matrix and pseudo-observation vector

  std::cout << "entrato in compute  " << std::endl ; 
  DVector<double> abs_res{} ;
  abs_res.resize(y().size()) ; 
  std::cout << "fatto resize " << std::endl ; 

  double tol = 1e-6; 
  for(int i = 0; i < y().size(); ++i)
    abs_res(i) = std::abs(y()(i) - mu(i)) > tol ? std::abs( y()(i) - mu(i) ) : ( std::abs(y()(i) - mu(i)) + tol );   

  std::cout << "fatto for " << std::endl ; 
  pW_ = (2*n_obs()*abs_res).cwiseInverse();   
  std::cout << "fatto pW " << std::endl ; 
  py_ = y() - (1 - 2*alpha_)*abs_res;
  std::cout << "fatto py " << std::endl ; 
  return std::tie(pW_, py_);
}

template <typename PDE, Sampling SamplingDesign>
double
SQRPDE<PDE, SamplingDesign>::compute_J_unpenalized(const DVector<double>& mu) {

  std::cout << "entrato in compute_J_unpen  " << std::endl ; 

  
  // compute value of functional J given mu: /(2*n) 
    return (pW_.cwiseSqrt().matrix().asDiagonal()*(py_ - mu)).squaredNorm() ;    // serve .matrix() dopo cwiseSqrt() ? 
    // differentemente da GSRPDE, in cui la sequenza dei calcoli è   
        // array() --> sqrt() --> .. --> matrix() ---> .asDiagonal() 
    // noi chiamiamo cwiseSqrt che opera component wise sul vettore (quindi è lo stesso) e poi chiamiamo la view asDiagonal   
}

// required to support GCV based smoothing parameter selection
// in case of an SRPDE model we have T = \Psi^T*Q*\Psi + \lambda*(R1^T*R0^{-1}*R1)
template <typename PDE, Sampling SamplingDesign>
const DMatrix<double>& SQRPDE<PDE, SamplingDesign>::T() {
  // compute value of R = R1^T*R0^{-1}*R1, cache for possible reuse
  if(R_.size() == 0){
    invR0_.compute(R0());
    R_ = R1().transpose()*invR0_.solve(R1());
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
template <typename PDE, Sampling SamplingDesign>
const DMatrix<double>& SQRPDE<PDE, SamplingDesign>::Q() {
  if(Q_.size() == 0){ // Q is computed on request since not needed in general
    // compute Q = W(I - H) = W - W*X*(X*W*X^T)^{-1}*X^T*W
    Q_ = W()*(DMatrix<double>::Identity(n_obs(), n_obs()) - X()*invXtWX().solve(X().transpose()*W()));
  }
  return Q_;
}


// returns the numerator of the GCV score 
template <typename PDE, Sampling SamplingDesign>
double SQRPDE<PDE, SamplingDesign>::norm
(const DMatrix<double>& fitted, const DMatrix<double>& obs) const {   // CONTROLLA ORDINE degli input 
  double result = 0;
  for(std::size_t i = 0; i < obs.rows(); ++i)
    result += rho_alpha(obs.coeff(i,0) - fitted.coeff(i,0));
  return result*result / obs.rows() ;
}

// returns the pinball loss at a specific x 
template <typename PDE, Sampling SamplingDesign>
double SQRPDE<PDE, SamplingDesign>::rho_alpha(const double& x) const{
  return 0.5*std::abs(x) + (alpha_ - 0.5)*x; 
}




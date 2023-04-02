// finds a solution to the SQR-PDE smoothing problem
template <typename PDE, Sampling SamplingDesign, typename Distribution = fdaPDE::models::Quantile>
void SQRPDE<PDE, SamplingDesign, Distribution>::solve() {
  // execute FPIRLS for minimization of the functional
  // \norm{V^{-1/2}(y - \mu)}^2 + \lambda \int_D (Lf - u)^2
  FPIRLS<decltype(*this), Distribution> fpirls(*this, tol_, max_iter_); // FPIRLS engine
  fpirls.compute();
  
  // fpirls converged: extract matrix P and solution estimates
  W_ = fpirls.weights().asDiagonal();
  f_ = fpirls.f();
  if(hasCovariates()) beta_ = fpirls.beta();
  return;
}

template <typename PDE, Sampling SamplingDesign, typename Distribution>
std::tuple<DVector<double>&, DVector<double>&>
SQRPDE<PDE, SamplingDesign, Distribution>::compute(const DVector<double>& mu) {
  // compute weight matrix and pseudo-observation vector
  DVector<double> abs_res = std::abs( y() - mu ); 
  pW_ = 1 / (2*ModelBase::n_obs()*abs_res).matrix();    // controlla n_obs e std::abs 
  py_ = y() - (1 - 2*alpha_)*abs_res;
  return std::tie(pW_, py_);
}

template <typename PDE, Sampling SamplingDesign, typename Distribution>
double
SQRPDE<PDE, SamplingDesign, Distribution>::compute_J_unpenalized(const DVector<double>& mu) {
  
  // compute value of functional J given mu: /(2*n) 
    return (pW_.sqrt().asDiagonal()*(py_ - mu)).squaredNorm() ;
    // differentemente da GSPDE, in cui la sequenza dei calcoli è   
        // array() --> sqrt() --> .. --> matrix()
    // noi stiamo facendo 
        // matrix() --> sqrt() 
    // per riusare pW_ già calcolata. E' più efficiente così oppure ricalcolandosi pW_ 
    // facendo la radice sull'array?       
}


// required to support GCV based smoothing parameter selection
// in case of an SRPDE model we have T = \Psi^T*Q*\Psi + \lambda*(R1^T*R0^{-1}*R1)
template <typename PDE, Sampling SamplingDesign>
const DMatrix<double>& SRPDE<PDE, SamplingDesign>::T() {
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
const DMatrix<double>& SRPDE<PDE, SamplingDesign>::Q() {
  if(Q_.size() == 0){ // Q is computed on request since not needed in general
    // compute Q = W(I - H) = W - W*X*(X*W*X^T)^{-1}*X^T*W
    Q_ = W()*(DMatrix<double>::Identity(n_obs(), n_obs()) - X()*invXtWX().solve(X().transpose()*W()));
  }
  return Q_;
}

// returns the euclidean norm of op1 - op2
template <typename PDE, Sampling SamplingDesign>
double SRPDE<PDE, SamplingDesign>::norm(const DMatrix<double>& op1, const DMatrix<double>& op2) const {
  return (op1 - op2).squaredNorm();
}

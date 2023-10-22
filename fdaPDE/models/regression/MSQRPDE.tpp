// perform proper initialization and update of model. Computes quantites which can be reused
// across many calls to solve() and are **not affected by a change in the data**.
// It is implicitly called by ModelBase::init() as part of the initialization process.
// NB: a change in the smoothing parameter must trigger a re-initialization of the model
template <typename PDE, typename SamplingDesign>
void MSQRPDE<PDE, SamplingDesign>::init_model() {

    std::cout << "MSQRPDE init model: here 1" << std::endl;
    // Assemble matrices
    assemble_matrices();  
    std::cout << "MSQRPDE init model: here 2" << std::endl;

    // Definition of h SQRPDE models for initialization 
    for(std::size_t j = 0; j < h_; ++j){
        std::cout << "MSQRPDE init model: here 3." << j << std::endl;
        SQRPDE<PDE, fdaPDE::models::SpaceOnly, SamplingDesign, fdaPDE::models::MonolithicSolver> 
            model_j(pde(), alphas_[j]);

        // solver initialization
        model_j.data() = data();
        model_j.setLambdaS(lambdas_[j]);
        model_j.set_spatial_locations(this->locs());
        model_j.init_pde();
        model_j.init_regularization();
        model_j.init_sampling();    
        model_j.init_nan();
        model_j.init_model();
        model_j.solve();

        f_curr_.segment(j*n_basis(), (j+1)*n_basis()-1) = model_j.f();
        fn_curr_.segment(j*n_obs(), (j+1)*n_obs()-1) = Psi()*model_j.f();
        g_curr_.segment(j*n_basis(), (j+1)*n_basis()-1) = model_j.g();
        beta_curr_.segment(j*q(), (j+1)*q()-1) = model_j.beta();

    }
    std::cout << "MSQRPDE init model: here 4" << std::endl;
    
    return;
}

  
// finds a solution to the SR-PDE smoothing problem
template <typename PDE, typename SamplingDesign>
void MSQRPDE<PDE, SamplingDesign>::solve() {

    
    SpMatrix<double> W_bar(h_*n_obs(), h_*n_obs());    
    DVector<double> t{};    

    while(crossing_constraints()){

        // algorithm stops when an enought small difference between two consecutive values of the J is recordered
        double J_old = tolerance_+1; double J_new = 0;
        k_ = 0;

        while(k_ < max_iter_ && std::abs(J_new - J_old) > tolerance_){

            // assemble W, Delta, z     
            DVector<double> w_(h_*n_obs());
            DVector<double> delta_((h_-1)*n_obs());
            DVector<double> z_(h_*n_obs());
            for(int j = 0; j < h_; ++j){

                DVector<double> w_j;
                DVector<double> z_j;
                DVector<double> delta_j; 

                if(!hasCovariates()){
                     w_j = 2*(y() - fn_prev_.segment(j*n_obs(), (j+1)*n_obs()-1)).cwiseAbs(); 

                    if(j < h_-1) {
                        delta_j = 2*(eps_*DVector<double>::Ones(n_obs()) - D_script_.row(j)*fn_prev_.segment(j*n_obs(), (j+1)*n_obs()-1)).cwiseAbs().cwiseInverse(); 
                    }
                    
                    z_j = y() + (0.5 - alphas_[j])*DiagMatrix<double>(w_j)*DVector<double>::Ones(n_obs()); 

                }
                else{
                    w_j = 2*(y()-X().transpose()*beta_prev_.segment(j*q(), (j+1)*q()-1) - fn_prev_.segment(j*n_obs(), (j+1)*n_obs()-1)).cwiseAbs(); 

                    if(j < h_-1) {
                        delta_j = 2*(eps_*DVector<double>::Ones(n_obs())-X().transpose()*D_.row(j)*beta_prev_.segment(j*q(), (j+1)*q()-1) - D_script_.row(j)*fn_prev_.segment(j*n_obs(), (j+1)*n_obs()-1)).cwiseAbs().cwiseInverse(); 
                    }
                    
                    z_j = y() + (0.5 - alphas_[j])*DiagMatrix<double>(w_j)*DVector<double>::Ones(n_obs()); 
                }

                w_ << w_ , w_j;
                delta_ << delta_ , delta_j ; 
                z_ << z_, z_j;                 
            }
            DiagMatrix<double> Delta_(delta_);  // non riusciamo a definirla fuori dal ciclo e poi riempirla qui
            W_bar.diagonal() = w_.cwiseInverse(); 
            W_multiple_ = W_bar + gamma_*D_script_.transpose()*Delta_*D_script_; 

            // assemble t 
            t = D_script_.transpose()*Delta_*D_script_*(eps_ + 0.5)*DVector<double>::Ones(h_*n_basis());  

            // assemble system matrix for nonparameteric part
            A_ = SparseBlockMatrix<double,2,2>
            (-Psi_multiple_.transpose()*W_multiple_*Psi_multiple_,    R1_multiple_.transpose(),
              R1_multiple_,                                           R0_multiple_            );
            
            // cache non-parametric matrix factorization for reuse
            invA_.compute(A_);

            // prepare rhs of linear system
            b_.resize(A_.rows());
            b_.block(h_*n_basis(),0, h_*n_basis(),1) = DVector<double>::Zero(h_*n_basis());

            XtWX_multiple_ = X_multiple_.transpose()*W_multiple_*X_multiple_;
            invXtWX_multiple_ = XtWX_multiple_.partialPivLu();

            DVector<double> sol; // room for problem' solution     
            if(!hasCovariates()){ // nonparametric case       
                // update rhs of SR-PDE linear system
                b_.block(0,0, n_basis(),1) = -Psi_multiple_.transpose()*(W_multiple_*z_ + gamma_*t);
                // solve linear system A_*x = b_
                sol = invA_.solve(b_);
                f_curr_ = sol.head(h_*n_basis());
                } else{ // parametric case
                // update rhs of SR-PDE linear system
                b_.block(0,0, n_basis(),1) = -Psi_multiple_.transpose()*(Q_multiple()*z_ + gamma_*(Ihn_ - H_multiple())*t);  
                
                // definition of matrices U and V  for application of woodbury formula
                U_multiple_ = DMatrix<double>::Zero(2*h_*n_basis(), h_*q());
                U_multiple_.block(0,0, h_*n_basis(), h_*q()) = Psi_multiple_.transpose()*W_multiple_*X_multiple_;
                V_multiple_ = DMatrix<double>::Zero(h_*q(), 2*h_*n_basis());
                V_multiple_.block(0,0, h_*q(), h_*n_basis()) = X_multiple_.transpose()*W_multiple_*Psi_multiple_;
                // solve system (A_ + U_*(X^T*W_*X)*V_)x = b using woodbury formula from NLA module
                sol = SMW<>().solve(invA_, U_multiple_, XtWX_multiple_, V_multiple_, b_); 
                // store result of smoothing 
                f_curr_    = sol.head(h_*n_basis());
                beta_curr_ = invXtWX_multiple_.solve((X_multiple_.transpose()*W_multiple_)*(z_ - Psi_multiple_*f_curr_) + 
                                                         gamma_*t);
                }
                // store PDE misfit
                g_curr_ = sol.tail(h_*n_basis());

                // update J 
                J_old = J_new; 
                J_new = model_loss() + g_curr_.dot(R0_multiple_*g_curr_);   // R0 multiple already contains lambdas!
        }

        gamma_ *= C_;  

    }

  return;
}


template <typename PDE, typename SamplingDesign>
const bool MSQRPDE<PDE, SamplingDesign>::crossing_constraints() const {

    bool crossed = false; 
    int i = 0; 
    int j = 0; 
    while(!crossed & i < n_obs()){
        while(!crossed & j < h_-1){
            if(X().row(i) * D_.row(j) * beta_curr_ + (D_script_.row(j) * f_curr_)(i) < eps_)
                crossed = true; 
            j++;   
        }
        i++; 
    }

    return crossed;
}

template <typename PDE, typename SamplingDesign>
DVector<double> MSQRPDE<PDE, SamplingDesign>::fitted() const{

    DVector<double> ret; 
    for(int j = 0; j < h_; ++j){
        ret << ret, fitted(j); 
    }
    return ret; 

}


template <typename PDE, typename SamplingDesign>
DVector<double> MSQRPDE<PDE, SamplingDesign>::fitted(unsigned int j) const{

    return X()*beta_curr_.segment(j*q(), (j+1)*q()-1) + f_curr_.segment(j*n_basis(), (j+1)*n_basis()-1); 

}

template <typename PDE, typename SamplingDesign>
double MSQRPDE<PDE, SamplingDesign>::model_loss() const{
  
    // compute value of the unpenalized unconstrained functional J: 
    double loss = 0.; 
    for(int j = 0; j < h_; ++j){
        loss += rho_alpha(alphas_[j], y() - fitted(j)).sum(); 
    }
    return loss; 

}

template <typename PDE, typename SamplingDesign>
const DMatrix<double>& MSQRPDE<PDE, SamplingDesign>::H_multiple() {
  // compute H = X*(X^T*W*X)^{-1}*X^T*W
  H_multiple_ = X_multiple_*invXtWX_multiple_.solve(X_multiple_.transpose()*W_multiple_);

  return H_multiple_;
}

template <typename PDE, typename SamplingDesign>
const DMatrix<double>& MSQRPDE<PDE, SamplingDesign>::Q_multiple() {
  // compute Q = W(I - H) = W ( I - X*(X^T*W*X)^{-1}*X^T*W ) 
  Q_multiple_ = W_multiple_*(DMatrix<double>::Identity(n_obs()*h_, n_obs()*h_) - H_multiple());

  return Q_multiple_;
}


// returns the pinball loss at a specific x 
template <typename PDE, typename Solver>
 DVector<double> MSQRPDE<PDE, Solver>::rho_alpha(const double& alpha, const DVector<double>& x) const{ 
  return 0.5*x.cwiseAbs() + (alpha - 0.5)*x; 
}

// template <typename PDE, typename Solver>
// const std::pair<unsigned int, unsigned int>  MSQRPDE<PDE, Solver>::block_indexes(unsinged int j, usigned int dim) const{
//     return std::make_pair<unsigned int, unsigned int>(j*dim, (j+1)*dim-1); 
// }
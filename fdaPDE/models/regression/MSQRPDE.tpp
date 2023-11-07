// perform proper initialization and update of model. Computes quantites which can be reused
// across many calls to solve() and are **not affected by a change in the data**.
// It is implicitly called by ModelBase::init() as part of the initialization process.
// NB: a change in the smoothing parameter must trigger a re-initialization of the model
template <typename PDE, typename SamplingDesign>
void MSQRPDE<PDE, SamplingDesign>::init_model() {

    // Assemble matrices
    assemble_matrices();  

    // Definition of h SQRPDE models for initialization 
    for(std::size_t j = 0; j < h_; ++j){
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

        f_curr_.block(j*n_basis(), 0, n_basis(), 1) = model_j.f();
        fn_curr_.block(j*n_obs(), 0, n_obs(), 1) = Psi()*model_j.f();
        g_curr_.block(j*n_basis(), 0, n_basis(), 1) = model_j.g();
        if(hasCovariates()){
            
            beta_curr_.block(j*q(), 0, q(), 1) = model_j.beta();
        }

    }

    // init = curr 
    f_init_ = f_curr_; 
    fn_init_ = fn_curr_; 
    g_init_ = g_curr_; 
    if(hasCovariates()){
        beta_init_ = beta_curr_;
        std::cout << "Beta_init_ : " << beta_init_ << std::endl ; 

    }

    std::cout << "Range f_init_ : " << f_init_.minCoeff() << " , " << f_init_.maxCoeff() << std::endl ; 
    std::cout << "Range g_init_ : " << g_init_.minCoeff() << " , " << g_init_.maxCoeff() << std::endl ; 
    
    std::cout << "Model loss of the initializazion: " << model_loss() << " + penalty = " << g_curr_.dot(R0_multiple_*g_curr_) << std::endl;  

    std::cout << "----- crossing global at init = " << crossing_penalty() << std::endl;
    return;
}

  
// finds a solution to the SR-PDE smoothing problem
template <typename PDE, typename SamplingDesign>
void MSQRPDE<PDE, SamplingDesign>::solve() {

    w_.resize(h_*n_obs()); 
    W_bar_.resize(h_*n_obs());    

    DiagMatrix<double> Delta_; 
    Delta_.resize((h_-1)*n_obs()); 
    // Delta_debug.resize((h_-1)*n_obs()); 

    z_.resize(h_*n_obs());

    DVector<double> t{};    
    t.resize(h_*n_obs()); 

    double crossing_penalty_init = crossing_penalty(); 

    bool go = true;
    unsigned int count_gamma = 0;
    bool first = true; 
    while(crossing_constraints() && iter_ < max_iter_global_){ 

        std::cout << "----------------Gamma = " << gamma0_ << std::endl; 
        count_gamma++;
        // algorithm stops when an enought small difference between two consecutive values of the J is recordered
        double J_old = tolerance_+1; double J_new = 0;
        k_ = 0;

        // restore initialization
        // se ko  ---> curr = init
        //std::cout << "#################### restore init " << std::endl; 
        // f_curr_ = f_init_; 
        // fn_curr_ = fn_init_; 
        // g_curr_ = g_init_; 
        // if(hasCovariates()){
        //     beta_curr_ = beta_init_;
        // }
            

        // std::cout << "---J before cycle over k: " << model_loss() << " + " << g_curr_.dot(R0_multiple_*g_curr_) << " = " << model_loss() + g_curr_.dot(R0_multiple_*g_curr_) << std::endl ; 

        while(k_ < max_iter_ && std::abs(J_new - J_old) > tolerance_){    // && crossing_constraints()

            std::cout << "--------------------------  k_ = " << k_ << std::endl ; 

            // assemble W, Delta, z 
            DVector<double> delta_((h_-1)*n_obs()); 

            for(int j = 0; j < h_; ++j){

                // DVector<double> w_j;
                DVector<double> abs_res_j;
                DVector<double> delta_j; 
                DVector<double> z_j;
                

                // w_j = 2*(y() - fitted(j)).cwiseAbs(); 
                abs_res_j = (y() - fitted(j)).cwiseAbs(); 

                if(j < h_-1) {
                    delta_j = (2*(eps_*DVector<double>::Ones(n_obs()) - D_script_.block(j*n_obs(), 0, n_obs(), h_*n_obs())*fitted())).cwiseAbs().cwiseInverse(); 
                    // equivalentemente
                    // delta_j = (2*(eps_*DVector<double>::Ones(n_obs()) - X()*D_.block(j*q(), 0, q(), h_*q())*beta_curr_ - D_script_.block(j*n_obs(), 0, n_obs(), h_*n_obs())*fn_curr_)).cwiseAbs().cwiseInverse(); 
                
                    // delta per imporreil vincolo solo su f
                    // delta_j = (2*(eps_*DVector<double>::Ones(n_obs()) - D_script_.block(j*n_obs(), 0, n_obs(), h_*n_obs())*fn_curr_)).cwiseAbs().cwiseInverse(); 
                
                }
                
                // if(k_ == 0 && j == 0)
                    // std::cout << "w_j: " << std::endl << std::setprecision(10) << w_j << std::endl; 
                z_j = y() - (1 - 2*alphas_[j])*abs_res_j; 

                // Adjust weights
                // std::cout << "-- Mean ( w_j ) = " << w_j.sum()/w_j.size() << std::endl ; 
                // abs_res_adj(w_j);

                abs_res_adj(abs_res_j);

                // w_.block(j*n_obs(), 0, n_obs(), 1) = w_j;

                w_.block(j*n_obs(), 0, n_obs(), 1) = 2*n_obs()*abs_res_j;

                DVector<double> w_j_debug = (2*n_obs()*abs_res_j).cwiseInverse();

                // std::cout << "range w_j = " << w_j_debug.minCoeff() << " , " << w_j_debug.maxCoeff() << std::endl; 

                // if(j < h_-1)
                //     std::cout << "range delta_j = " << delta_j.minCoeff() << " , " << delta_j.maxCoeff() << std::endl; 


                if(j < h_-1) 
                    delta_.block(j*n_obs(), 0, n_obs(), 1) = delta_j; 
                z_.block(j*n_obs(), 0, n_obs(), 1) = z_j;         
            }
            // DiagMatrix<double> Delta_(delta_);  // non riusciamo a definirla fuori dal ciclo e poi riempirla qui
            Delta_.diagonal() = delta_;
            if(first)
                Delta_debug.diagonal() = delta_;  
            W_bar_.diagonal() = w_.cwiseInverse(); 
            if(first)
                W_bar_debug.diagonal() = w_.cwiseInverse(); 
            W_multiple_ = SpMatrix<double>(W_bar_) + gamma0_*D_script_.transpose()*Delta_*D_script_; 
            if(first)
                W_multiple_debug = W_multiple_; 

            //std::cout << "Range W_bar = " << (W_bar_.diagonal()).minCoeff() << " , " << (W_bar_.diagonal()).maxCoeff() << std::endl ; 

            // std::cout << "Range W_altro = " << (gamma0_*D_script_.transpose()*Delta_*D_script_).minCoeff() << " , " << (gamma0_*D_script_.transpose()*Delta_*D_script_).maxCoeff() << std::endl ; 

            // assemble t 
            t = D_script_.transpose()*Delta_*eps_*DVector<double>::Ones((h_-1)*n_obs()) + 0.5*l_hn_; 

            // assemble system matrix for nonparameteric part
            A_ = SparseBlockMatrix<double,2,2>
            (-Psi_multiple_.transpose()*W_multiple_*Psi_multiple_,    R1_multiple_.transpose(),
              R1_multiple_,                                           R0_multiple_            );
            
            // cache non-parametric matrix factorization for reuse
            invA_.compute(A_);

            // prepare rhs of linear system
            b_.resize(A_.rows());
            b_.block(h_*n_basis(),0, h_*n_basis(),1) = DVector<double>::Zero(h_*n_basis());  // b_g = 0 

            DVector<double> sol; // room for problem' solution      
               
            if(!hasCovariates()){ // nonparametric case     

                // update rhs of SR-PDE linear system
                b_.block(0,0, h_*n_basis(),1) = -Psi_multiple_.transpose()*(W_bar_*z_ + gamma0_*t);

                // solve linear system A_*x = b_
                sol = invA_.solve(b_);

                f_curr_ = sol.head(h_*n_basis());
                fn_curr_ = Psi_multiple_*f_curr_; 

                } else{ // parametric case
 
                    XtWX_multiple_ = X_multiple_.transpose()*W_multiple_*X_multiple_;
                    if(first)
                        XtWX_multiple_debug = XtWX_multiple_;
                    invXtWX_multiple_ = XtWX_multiple_.partialPivLu(); 
                    // update rhs of SR-PDE linear system
                    if(first)
                        H_multiple_debug = H_multiple(); 

                    b_.block(0,0, h_*n_basis(),1) = -Psi_multiple_.transpose()*(Ihn_ - H_multiple().transpose())*(W_bar_*z_ + gamma0_*t);  

                    // definition of matrices U and V  for application of woodbury formula
                    U_multiple_ = DMatrix<double>::Zero(2*h_*n_basis(), h_*q());
                    U_multiple_.block(0,0, h_*n_basis(), h_*q()) = Psi_multiple_.transpose()*W_multiple_*X_multiple_;
                    V_multiple_ = DMatrix<double>::Zero(h_*q(), 2*h_*n_basis());
                    V_multiple_.block(0,0, h_*q(), h_*n_basis()) = X_multiple_.transpose()*W_multiple_*Psi_multiple_;
                    // solve system (A_ + U_*(X^T*W_*X)*V_)x = b using woodbury formula from NLA module
                    sol = SMW<>().solve(invA_, U_multiple_, XtWX_multiple_, V_multiple_, b_); 
                    // store result of smoothing 
                    f_curr_    = sol.head(h_*n_basis());
                    fn_curr_ = Psi_multiple_*f_curr_; 
                    beta_curr_ = invXtWX_multiple_.solve(X_multiple_.transpose()*(W_bar_*z_ - W_multiple_*fn_curr_ + gamma0_*t));
                    
                    std::cout << "Beta_curr: " << beta_curr_ << std::endl; 





                    // Sistema per il caso in cui imponiamo il vicncolo solo su f
                    // XtWX_multiple_ = X_multiple_.transpose()*W_bar_*X_multiple_;
                    // if(first)
                    //     XtWX_multiple_debug = XtWX_multiple_; 
                    // invXtWX_multiple_ = XtWX_multiple_.partialPivLu();
                    // // update rhs of SR-PDE linear system
                    // if(first)
                    //     H_multiple_debug = H_multiple(); 

                    // b_.block(0,0, h_*n_basis(),1) = -Psi_multiple_.transpose()*(Q_multiple()*z_ + gamma0_*t);  

                    // // definition of matrices U and V  for application of woodbury formula
                    // U_multiple_ = DMatrix<double>::Zero(2*h_*n_basis(), h_*q());
                    // U_multiple_.block(0,0, h_*n_basis(), h_*q()) = Psi_multiple_.transpose()*W_bar_*X_multiple_;
                    // V_multiple_ = DMatrix<double>::Zero(h_*q(), 2*h_*n_basis());
                    // V_multiple_.block(0,0, h_*q(), h_*n_basis()) = X_multiple_.transpose()*W_bar_*Psi_multiple_;
                    // // solve system (A_ + U_*(X^T*W_*X)*V_)x = b using woodbury formula from NLA module
                    // sol = SMW<>().solve(invA_, U_multiple_, XtWX_multiple_, V_multiple_, b_); 
                    // // store result of smoothing 
                    // f_curr_    = sol.head(h_*n_basis());

                    // std::cout << "max(abs(fn_curr(k) - fn_curr(k-1))) = " << (fn_curr_ - Psi_multiple_*f_curr_).cwiseAbs().maxCoeff() << std::endl;

                    // fn_curr_ = Psi_multiple_*f_curr_; 
                    // beta_curr_ = invXtWX_multiple_.solve(X_multiple_.transpose()*W_bar_*(z_ - fn_curr_));
                    // std::cout << "Beta_curr: " << beta_curr_ << std::endl; 


                }
                // store PDE misfit
                g_curr_ = sol.tail(h_*n_basis());
                
                std::cout << "Range g_curr: " << g_curr_.minCoeff() << " , " << g_curr_.maxCoeff() << std::endl;
                std::cout << "Range f_curr: " << f_curr_.minCoeff() << " , " << f_curr_.maxCoeff() << std::endl; 
                
                // update J 
                J_old = J_new; 
                J_new = model_loss() + g_curr_.dot(R0_multiple_*g_curr_) + gamma0_*crossing_penalty();   // R0 multiple already contains lambdas!
                
                // J per il caso con vincolo solo su f
                // J_new = model_loss() + g_curr_.dot(R0_multiple_*g_curr_) + gamma0_*crossing_penalty_f();   // R0 multiple already contains lambdas!
                

                std::cout << "----- crossing global at iter k = " << crossing_penalty() << std::endl;
                // std::cout << "----- crossing only f at iter k = " << crossing_penalty_f() << std::endl;
                // std::cout << "----- crossing Xbeta at iter k = " << crossing_penalty_param() << std::endl;
                std::cout << "----- J_old = " << J_old << std::endl;
                std::cout << "----- J_new = " << J_new << " = " << model_loss() << " + " << g_curr_.dot(R0_multiple_*g_curr_) << std::endl;  
                std::cout << "----- J_old - J_new = " << J_old - J_new << std::endl; 

                k_++;  
                first = false;    
        }

        double crossing_penalty_new = crossing_penalty(); 

        // per il caso con vincolo solo su f
        // double crossing_penalty_new = crossing_penalty_f(); 

        std::cout << "#################### cross new:  " << crossing_penalty_new << " , cross init: " << crossing_penalty_init << std::endl; 
        std::cout << "#################### cross new-init: " << std::setprecision(10) << (crossing_penalty_new - crossing_penalty_init) << std::endl; 

        // // check 
        // if(crossing_penalty_new - crossing_penalty_init > tolerance_){
        //     // se ko  ---> curr = init
        //     std::cout << "#################### restore init " << std::endl; 
        //     f_curr_ = f_init_; 
        //     fn_curr_ = fn_init_; 
        //     g_curr_ = g_init_; 

        //     if(hasCovariates()){
        //         beta_curr_ = beta_init_;
        //     }
             
        // }
         
        gamma0_ *= C_;  
        if(count_gamma > 0)
            go = false;

        iter_++;     
        
    }

  return;
}


// Utilities 

template <typename PDE, typename SamplingDesign>
DVector<double> MSQRPDE<PDE, SamplingDesign>::fitted() const{

    DVector<double> fit = fn_curr_; 
    if(hasCovariates())
        fit += X_multiple_*beta_curr_;
    return fit; 
}

template <typename PDE, typename SamplingDesign>
DVector<double> MSQRPDE<PDE, SamplingDesign>::fitted(unsigned int j) const{
    return fitted().block(j*n_obs(), 0, n_obs(), 1); 
}

template <typename PDE, typename SamplingDesign>
const bool MSQRPDE<PDE, SamplingDesign>::crossing_constraints() const {

    // Return true if the current estimate of quantiles is crossing, false otherwise 

    bool crossed = false; 
     
    unsigned int j = 0;
    while(!crossed && j < h_-1){ 
        DVector<double> crossing_value = fitted(j+1)-fitted(j); 
        unsigned int i = 0; 
        while(!crossed && i < n_obs()){          
            if(crossing_value(i) < eps_)
                crossed = true;  
            i++;
        }
        j++; 
    }

    if(crossed)
        std::cout << "crossed = true" << std::endl;
    else{
        std::cout << "crossed = false" << std::endl;
    }

    return crossed;
}


template <typename PDE, typename SamplingDesign>
double MSQRPDE<PDE, SamplingDesign>::model_loss() const{

    // fpirls-like 
    // return (w_.cwiseInverse().cwiseSqrt().matrix().asDiagonal()*(z_ - fitted())).squaredNorm(); 

    // // non-parametric version 
    // return ( fn_curr_.transpose()*(w_.cwiseInverse().matrix().asDiagonal())*(fn_curr_ - z_) ); 
    // // abbiamo tolto un *0.5 e messo su 

    // rho 
    double loss = 0.; 
    for(auto j = 0; j < h_; ++j)
        loss += (rho_alpha(alphas_[j], y() - fitted(j))).sum(); 
    return loss/n_obs();

}

template <typename PDE, typename SamplingDesign>
double MSQRPDE<PDE, SamplingDesign>::crossing_penalty() const{
  
    // compute value of the unpenalized unconstrained functional J: 
    double pen = 0.; 
    for(int j = 0; j < h_-1; ++j){
        //std::cout << "Range mu(j+1) - mu(j) = " << (fitted(j+1) - fitted(j)).minCoeff() << " , " << (fitted(j+1) - fitted(j)).maxCoeff() << std::endl; 
        pen += (eps_*DVector<double>::Ones(n_obs()) - (fitted(j+1) - fitted(j))).cwiseMax(0.).sum(); 

    }
    
    return pen; 

}

template <typename PDE, typename SamplingDesign>
double MSQRPDE<PDE, SamplingDesign>::crossing_penalty_f() const{
  
    // compute value of the unpenalized unconstrained functional J: 
    double pen = 0.; 
    for(int j = 0; j < h_-1; ++j){
        // std::cout << "Range f(j+1) - f(j) = " << (fn_curr_.block((j+1)*n_obs(), 0, n_obs(), 1) - fn_curr_.block(j*n_obs(), 0, n_obs(), 1)).minCoeff() << " , " << (fn_curr_.block((j+1)*n_obs(), 0, n_obs(), 1) - fn_curr_.block(j*n_obs(), 0, n_obs(), 1)).maxCoeff() << std::endl ; 
        pen += (eps_*DVector<double>::Ones(n_obs()) - (fn_curr_.block((j+1)*n_obs(), 0, n_obs(), 1) - fn_curr_.block(j*n_obs(), 0, n_obs(), 1))).cwiseMax(0.).sum(); 

    }
    
    return pen; 

}

template <typename PDE, typename SamplingDesign>
double MSQRPDE<PDE, SamplingDesign>::crossing_penalty_param() const{
  
    // compute value of the unpenalized unconstrained functional J: 
    double pen = 0.; 
    for(int j = 0; j < h_-1; ++j){
        std::cout << "X * (Beta j+1 - Beta j) = " << (X()*(beta_curr_.block((j+1)*q(), 0, q(), 1) - beta_curr_.block(j*q(), 0, q(), 1))).minCoeff() << " , " << (X()*(beta_curr_.block((j+1)*q(), 0, q(), 1) - beta_curr_.block(j*q(), 0, q(), 1))).maxCoeff() << std::endl;
        pen += (eps_*DVector<double>::Ones(n_obs()) - (X()*(beta_curr_.block((j+1)*q(), 0, q(), 1) - beta_curr_.block(j*q(), 0, q(), 1)))).cwiseMax(0.).sum(); 

    }
    
    return pen; 

}

template <typename PDE, typename SamplingDesign>
const DMatrix<double>& MSQRPDE<PDE, SamplingDesign>::H_multiple() {
  // compute H = X*(X^T*W*X)^{-1}*X^T*W

  H_multiple_ = X_multiple_*(invXtWX_multiple_.solve(X_multiple_.transpose()*W_multiple_));

  // H per il caso con vincolo solo su f
  // H_multiple_ = X_multiple_*(invXtWX_multiple_.solve(X_multiple_.transpose()*W_bar_));

  //std::cout << "(easy) done!"  << std::endl; 

  return H_multiple_;
}

template <typename PDE, typename SamplingDesign>
const DMatrix<double>& MSQRPDE<PDE, SamplingDesign>::Q_multiple() {
  // compute Q = W(I - H) = W ( I - X*(X^T*W*X)^{-1}*X^T*W ) 
  Q_multiple_ = W_multiple_*(DMatrix<double>::Identity(n_obs()*h_, n_obs()*h_) - H_multiple());

  // Q per il caso con vincolo solo su f
  // Q_multiple_ = W_bar_*(DMatrix<double>::Identity(n_obs()*h_, n_obs()*h_) - H_multiple());

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


template <typename PDE, typename Solver>
    void MSQRPDE<PDE, Solver>::abs_res_adj(DVector<double>& res) {
        unsigned int count_debug = 1; 
        for(int i = 0; i < res.size(); ++i) {
            if(res(i) < tol_weights_) {
                count_debug++; 
                res(i) += tol_weights_;  
            }            
        }
        //std::cout << "Quanti aggiustamenti di tol su 441? " << count_debug << "    -> range(abs_res_j) = " << res.minCoeff() << " , " << res.maxCoeff() << std::endl; 
    }


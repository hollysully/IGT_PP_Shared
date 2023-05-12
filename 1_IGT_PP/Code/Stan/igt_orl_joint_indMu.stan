data{
  int<lower=1> N; // Number of participants
  int<lower=1> S; // Number of sessions
  int<lower=1> T; // Total possile number of trials
  int card[N,T,S]; // Cards presented on each trial
  int Tsubj[N,S]; // Total number of trials presented to each subject on each session
  int choice[N,T,S]; // Choices on each trial
  real outcome[N,T,S]; // Outcomes received on each trial
  real sign[N,T,S]; // Signs of the outcome received on each trial
  int day[N,S]; // num days from day 0 predictor for linear model
}


// ---------------------------------------------------------------------------------------
parameters{
// Declare parameters
  // Hyper(group)-parameters
  matrix[2,5] mu_pr; // 2 (intercept and slope) x 5 (number of parameters) matrix of mus
  
  // Variance components of random intercepts (element 1) and random slopes (element 2)
  vector<lower=0>[2] sigma_Arew;
  vector<lower=0>[2] sigma_Apun;
  vector<lower=0>[2] sigma_K;
  vector<lower=0>[2] sigma_betaF;
  vector<lower=0>[2] sigma_betaP;

  // Subject-level "raw" parameters - i.e., independent/uncorrelated & normally distributed person-level (random-)effects
  matrix[2,N] Arew_pr;  // Untransformed Arews
  matrix[2,N] Apun_pr;  // Untransformed Apunss
  matrix[2,N] K_pr;     // Untransformed Ks
  matrix[2,N] betaF_pr;
  matrix[2,N] betaP_pr;
  
  // Correlation matrices for correlating between sessions
  cholesky_factor_corr[2] R_chol_Arew;
  cholesky_factor_corr[2] R_chol_Apun;
  cholesky_factor_corr[2] R_chol_K;
  cholesky_factor_corr[2] R_chol_betaF;
  cholesky_factor_corr[2] R_chol_betaP;
}

transformed parameters{
  // Declare transformed parameters
  matrix<lower=0,upper=1>[N,2] Arew;  // Transformed Arews
  matrix<lower=0,upper=1>[N,2] Apun;  // Transformed Apuns
  matrix<lower=0,upper=5>[N,2] K;     // Transformed Ks
  matrix[N,2] betaF;
  matrix[N,2] betaP;
  
  // Untransformed subject-level parameters incorporating correlation between sessions
  matrix[2,N] Arew_tilde;
  matrix[2,N] Apun_tilde;
  matrix[2,N] K_tilde;
  matrix[2,N] betaF_tilde;
  matrix[2,N] betaP_tilde;
  
  // Untransformed subject-level parameters incorporating correlation between sessions
  Arew_tilde = diag_pre_multiply(sigma_Arew, R_chol_Arew) * Arew_pr;  
  Apun_tilde = diag_pre_multiply(sigma_Apun, R_chol_Apun) * Apun_pr;  
  K_tilde = diag_pre_multiply(sigma_K, R_chol_K) * K_pr;  
  betaF_tilde = diag_pre_multiply(sigma_betaF, R_chol_betaF) * betaF_pr;  
  betaP_tilde = diag_pre_multiply(sigma_betaP, R_chol_betaP) * betaP_pr;
  
  for (s in 1:S) {
    for (i in 1:N) {
      // (intercept) + day * (slope)
      Arew[i,s] = Phi_approx((mu_pr[1,1] + Arew_tilde[1,i]) + day[i,s] * (mu_pr[2,1] + Arew_tilde[2,i]));
      Apun[i,s] = Phi_approx((mu_pr[1,2] + Apun_tilde[1,i]) + day[i,s] * (mu_pr[2,2] + Apun_tilde[2,i]));
      K[i,s] = Phi_approx((mu_pr[1,3] + K_tilde[1,i]) + day[i,s] * (mu_pr[2,3] + K_tilde[2,i]))*5;
      betaF[i,s] = (mu_pr[1,4] + betaF_tilde[1,i]) + day[i,s] * (mu_pr[2,4] + betaF_tilde[2,i]);
      betaP[i,s] = (mu_pr[1,5] + betaP_tilde[1,i]) + day[i,s] * (mu_pr[2,5] + betaP_tilde[2,i]);
      
    }
  }
}

model{
  // Declare variables to calculate utility after each trial: These 4 (number of cards) x 2 (playing vs. not playing) matrices
  matrix[4,2] ef;
  matrix[4,2] ev;
  matrix[4,2] pers;
  matrix[4,2] utility;
    
  real PEval;
  real PEfreq;
  real PEfreq_fic;
  real K_tr;
  
  // Priors
  // Hyperparameters for correlations
  R_chol_Arew ~ lkj_corr_cholesky(1);
  R_chol_Apun ~ lkj_corr_cholesky(1);
  R_chol_K ~ lkj_corr_cholesky(1);
  R_chol_betaF ~ lkj_corr_cholesky(1);
  R_chol_betaP ~ lkj_corr_cholesky(1);
  
  // Hyperparameters for RL learning algorithm
  to_vector(mu_pr) ~ normal(0, 1);
  
  // 'Variance' components
  sigma_Arew ~ normal(0, 0.2);
  sigma_Apun ~ normal(0, 0.2);
  sigma_K ~ normal(0, 0.2);
  sigma_betaF ~ cauchy(0, 1);
  sigma_betaP ~ cauchy(0, 1);
  
  // Subject-level parameters
  to_vector(Arew_pr) ~ normal(0, 1.0);
  to_vector(Apun_pr) ~ normal(0, 1.0);
  to_vector(K_pr) ~ normal(0, 1.0);
  to_vector(betaF_pr) ~ normal(0, 1.0);
  to_vector(betaP_pr) ~ normal(0, 1.0);
  
  for (i in 1:N) {
    for (s in 1:S) {
      if (Tsubj[i,s] > 0) {
        
        // Initialize starting values
        K_tr = pow(3, K[i,s]) - 1;
        for (p in 1:2) {  
          ev[:,p] = rep_vector(0,4);
          ef[:,p] = rep_vector(0,4);
          pers[:,p] = rep_vector(0,4);
          utility[:,p] = rep_vector(0,4);
        }
        
        for (t in 1:Tsubj[i,s]) {
          
        // Likelihood - predict choice as a function of utility
        choice[i,t,s] ~ categorical_logit(to_vector(utility[card[i,t,s], :]));
        
        // After choice, calculate prediction error
        PEval      = outcome[i,t,s] - ev[card[i,t,s], choice[i,t,s]];     // Value prediction error
        PEfreq     = sign[i,t,s] - ef[card[i,t,s], choice[i,t,s]];        // Win-frequency prediction error
        PEfreq_fic = -sign[i,t,s]/1 - ef[card[i,t,s], (3-choice[i,t,s])]; // Lose-frequency prediction error?
        
        if (outcome[i,t,s] >= 0){  // If participant DID NOT lose
          // Update expected win-frequency of what participant DID NOT chose to do
          ef[card[i,t,s], (3-choice[i,t,s])] = ef[card[i,t,s], (3-choice[i,t,s])] + Apun[i,s] * PEfreq_fic;
          // Update what participant chose
          ef[card[i,t,s], choice[i,t,s]] = ef[card[i,t,s], choice[i,t,s]] + Arew[i,s] * PEfreq;
          ev[card[i,t,s], choice[i,t,s]] = ev[card[i,t,s], choice[i,t,s]] + Arew[i,s] * PEval;
        } else { // If participant DID lose
          // Update expected win-frequency of what participant DID NOT choose to do
          ef[card[i,t,s], (3-choice[i,t,s])] = ef[card[i,t,s], (3-choice[i,t,s])] + Arew[i,s] * PEfreq_fic;
          // Update what participant chose
          ef[card[i,t,s], choice[i,t,s]] = ef[card[i,t,s], choice[i,t,s]] + Apun[i,s] * PEfreq;
          ev[card[i,t,s], choice[i,t,s]] = ev[card[i,t,s], choice[i,t,s]] + Apun[i,s] * PEval;
        }
        
        // Update perseverance
        pers[card[i,t,s], choice[i,t,s]] = 1;
        pers = pers / (1 + K_tr);
        
        // Calculate expected value of card
        utility = ev + ef * betaF[i,s] + pers * betaP[i,s];
        }
      }
    }
  }
}

generated quantities { 
  // test-retest correlations
  corr_matrix[2] R_Arew;
  corr_matrix[2] R_Apun;
  corr_matrix[2] R_K;
  corr_matrix[2] R_betaF;
  corr_matrix[2] R_betaP;
  
   // Hyper(group)-parameters - these are 5 (number of parameters) x 2 (intercept and slope)
  vector<lower=0,upper=1>[2] mu_Arew;
  vector<lower=0,upper=1>[2] mu_Apun;
  vector<lower=0,upper=5>[2] mu_K;
  vector[2] mu_betaF;
  vector[2] mu_betaP;
  
  // For posterior predictive check
  real choice_pred[N,T,S];
  
  // Reconstruct correlation matrix of intercepts and slopes
  R_Arew  = R_chol_Arew * R_chol_Arew';
  R_Apun  = R_chol_Apun * R_chol_Apun';
  R_K     = R_chol_K * R_chol_K';
  R_betaF = R_chol_betaF * R_chol_betaF';
  R_betaP = R_chol_betaP * R_chol_betaP';

  // Set all posterior predictions to -1 (avoids NULL values)
  for (i in 1:N) {
    for (s in 1:S){
      for (t in 1:T){
        choice_pred[i,t,s] = -1;
      }
    }
  }
  
  // Group-level means
  for (s in 1:S) {
    mu_Arew[s] = Phi_approx(mu_pr[s, 1]);
    mu_Apun[s] = Phi_approx(mu_pr[s, 2]);
    mu_K[s] = Phi_approx(mu_pr[s, 3]) * 5;
    mu_betaF[s] = mu_pr[s, 4];
    mu_betaP[s] = mu_pr[s, 5];
  }
  
  { // local section, this saves time and space
  
    // Declare variables to calculate utility after each trial: These 4 (number of cards) x 2 (playing vs. not playing) matrices
    matrix[4,2] ef;
    matrix[4,2] ev;
    matrix[4,2] pers;
    matrix[4,2] utility;
      
    real PEval;
    real PEfreq;
    real PEfreq_fic;
    real K_tr;
  
    // -------------------------------
    for (i in 1:N) {        
      for (s in 1:S) {      
        if (Tsubj[i,s] > 0) {
          
          // Initialize starting values
          K_tr = pow(3, K[i,s]) - 1;
          for (p in 1:2) { 
            ev[:,p] = rep_vector(0,4);
            ef[:,p] = rep_vector(0,4);
            pers[:,p] = rep_vector(0,4);
            utility[:,p] = rep_vector(0,4);
          }
          
          for (t in 1:Tsubj[i,s]) {
            // generate choice as a function of utility
            choice_pred[i,t,s] = categorical_rng(softmax(to_vector(utility[card[i,t,s], :])));
            
            // After choice, calculate prediction error
            PEval      = outcome[i,t,s] - ev[card[i,t,s], choice[i,t,s]];     // Value prediction error
            PEfreq     = sign[i,t,s] - ef[card[i,t,s], choice[i,t,s]];        // Win-frequency prediction error
            PEfreq_fic = -sign[i,t,s]/1 - ef[card[i,t,s], (3-choice[i,t,s])]; // Lose-frequency prediction error?
            
            if (outcome[i,t,s] >= 0){  // If participant DID NOT lose
              // Update expected win-frequency of what participant DID NOT chose to do
              ef[card[i,t,s], (3-choice[i,t,s])] = ef[card[i,t,s], (3-choice[i,t,s])] + Apun[i,s] * PEfreq_fic;
              // Update what participant chose
              ef[card[i,t,s], choice[i,t,s]] = ef[card[i,t,s], choice[i,t,s]] + Arew[i,s] * PEfreq;
              ev[card[i,t,s], choice[i,t,s]] = ev[card[i,t,s], choice[i,t,s]] + Arew[i,s] * PEval;
            } else { // If participant DID lose
              // Update expected win-frequency of what participant DID NOT choose to do
              ef[card[i,t,s], (3-choice[i,t,s])] = ef[card[i,t,s], (3-choice[i,t,s])] + Arew[i,s] * PEfreq_fic;
              // Update what participant chose
              ef[card[i,t,s], choice[i,t,s]] = ef[card[i,t,s], choice[i,t,s]] + Apun[i,s] * PEfreq;
              ev[card[i,t,s], choice[i,t,s]] = ev[card[i,t,s], choice[i,t,s]] + Apun[i,s] * PEval;
            }
            
            // Update perseverance
            pers[card[i,t,s], choice[i,t,s]] = 1;
            pers = pers / (1 + K_tr);
            
            // Calculate expected value of card
            utility = ev + ef * betaF[i,s] + pers * betaP[i,s];
          }
        }
      }
    }
  }
}

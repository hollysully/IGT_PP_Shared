

// ---------------------------------------------------------------------------------------
data{
  int<lower=1> N;                  // Number of observations
  int<lower=1> n_J;                // Number of subjects
  int<lower=1> n_S;                // Number of sessions
  int J[N];                        // Vector of subjects
  int S[N];                        // Vector of sessions
  int<lower=1> Trials[N];          // Vector of trials
  int C[N];                        // Vector of cards presented to subject
  int<lower=1,upper=2> choice[N];  // Vector of choices on each trial
  real outcome[N];                 // Vector of outcomes received on each trial
  real sign[N];                    // Vector of the signs of the outcome received on each trial
}


// ---------------------------------------------------------------------------------------
parameters{
// Declare parameters
  // Hyper(group)-parameters - these are  & sigmas, respectively, for each parameter
  matrix[n_S, 5] mu_p;   // S (number of sessions) x 5 (number of parameters) matrix of mus
  matrix[n_J, 5] ind_p;   // n_J (number of participants) x 5 (number of parameters) matrix of mus
  vector<lower=0>[2] sigma_Arew;
  vector<lower=0>[2] sigma_Apun;
  vector<lower=0>[2] sigma_K;
  vector<lower=0>[2] sigma_betaF;
  vector<lower=0>[2] sigma_betaP;

  // Subject-level "raw" parameters - i.e., independent/uncorrelated & normally distributed person-level (random-)effects
    // Note, these are S (number of sessions) x J (number of subjects) matrices
  matrix[n_S, n_J] Arew_pr;  // Untransformed Arews
  matrix[n_S, n_J] Apun_pr;  // Untransformed Apunss
  matrix[n_S, n_J] K_pr;     // Untransformed Ks
  matrix[n_S, n_J] betaF_pr;
  matrix[n_S, n_J] betaP_pr;
  
  // Correlation matrices for correlating between sessions
  cholesky_factor_corr[2] R_chol_Arew;
  cholesky_factor_corr[2] R_chol_Apun;
  cholesky_factor_corr[2] R_chol_K;
  cholesky_factor_corr[2] R_chol_betaF;
  cholesky_factor_corr[2] R_chol_betaP;
}


// ---------------------------------------------------------------------------------------
transformed parameters{
// -------------------------------
// Declare transformed parameters
  // Parameters to use in the reinforcement-learning algorithm
    // Note, for each of these, we include the mus and the subject-level parameters (see below), allowing for shrinkage and subject-to-subject variability
  matrix<lower=0,upper=1>[n_J, n_S] Arew;  // Transformed Arews
  matrix<lower=0,upper=1>[n_J, n_S] Apun;  // Transformed Apuns
  matrix<lower=0,upper=5>[n_J, n_S] K;     // Transformed Ks
  matrix[n_J, n_S] betaF;
  matrix[n_J, n_S] betaP;
  
  // Untransformed subject-level parameters incorporating correlation between sessions
  matrix[n_S, n_J] Arew_tilde;
  matrix[n_S, n_J] Apun_tilde;
  matrix[n_S, n_J] K_tilde;
  matrix[n_S, n_J] betaF_tilde;
  matrix[n_S, n_J] betaP_tilde;
  
// -------------------------------
// Calculate transformed parameters
  
  // Untransformed subject-level parameters incorporating correlation between sessions
  Arew_tilde  = diag_pre_multiply(sigma_Arew, R_chol_Arew) * Arew_pr;  
  Apun_tilde  = diag_pre_multiply(sigma_Apun, R_chol_Apun) * Apun_pr;  
  K_tilde     = diag_pre_multiply(sigma_K, R_chol_K) * K_pr;  
  betaF_tilde = diag_pre_multiply(sigma_betaF, R_chol_betaF) * betaF_pr;  
  betaP_tilde = diag_pre_multiply(sigma_betaP, R_chol_betaP) * betaP_pr;
  
  // Calculate & transform Arew, Apun, & K to use in RL algorithm
  for(s in 1:n_S){    // Loop over sessions
    for(j in 1:n_J){  // Loop over subjects - This is structured the same as the OG joint retest model such that Arew,
                        // Apun, & K are looped over for individual subjects whereas betaF & betaP are vectorized for
                        // individual subjects.
                        // Maybe to avoid nesting a function within a function - e.g., to_vector(Phi_approx... 
      Arew[j, s] = Phi_approx(mu_p[s, 1] + ind_p[j, 1] + Arew_tilde[s, j]);
      Apun[j, s] = Phi_approx(mu_p[s, 2] + ind_p[j, 2] + Apun_tilde[s, j]);
      K[j, s]    = Phi_approx(mu_p[s, 3] + ind_p[j, 3] + K_tilde[s, j]) * 5;
    
      // Calculate betaF & betaP to use in RL algorithm
      betaF[j, s] = mu_p[s, 4] + ind_p[j, 4] + betaF_tilde[s, j];
      betaP[j, s] = mu_p[s, 5] + ind_p[j, 5] + betaP_tilde[s, j];
    }
  }
}


// ---------------------------------------------------------------------------------------
model{
// -------------------------------
// Priors
  // Hyperparameters for correlations
  R_chol_Arew   ~ lkj_corr_cholesky(1);
  R_chol_Apun   ~ lkj_corr_cholesky(1);
  R_chol_K      ~ lkj_corr_cholesky(1);
  R_chol_betaF  ~ lkj_corr_cholesky(1);
  R_chol_betaP  ~ lkj_corr_cholesky(1);
  
  // Hyperparameters for RL learning algorithm
  mu_p[1,:]   ~ normal(0, 1);
  mu_p[2,:]   ~ normal(0, 1);
  sigma_Arew  ~ normal(0, 0.2);
  sigma_Apun  ~ normal(0, 0.2);
  sigma_K     ~ normal(0, 0.2);
  sigma_betaF ~ cauchy(0, 1);
  sigma_betaP ~ cauchy(0, 1);
  
  // Subject-level parameters
  to_vector(Arew_pr)  ~ normal(0, 1.0);
  to_vector(Apun_pr)  ~ normal(0, 1.0);
  to_vector(K_pr)     ~ normal(0, 1.0);
  to_vector(betaF_pr) ~ normal(0, 1.0);
  to_vector(betaP_pr) ~ normal(0, 1.0);
  
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
// Loop through RL algorithm
  for(i in 1:N){
    if(Trials[i] == 1){  // Initialize starting values for each participant on each session
       
      // Initialize starting values
      K_tr = pow(3, K[J[i], S[i]]) - 1;
      for(pp in 1:2){  // Looping over play/pass columns and assigning each 0 to each card
        ev[:,pp] = rep_vector(0,4);
        ef[:,pp] = rep_vector(0,4);
        pers[:,pp] = rep_vector(0,4);
        utility[:,pp] = rep_vector(0,4);
      }
    }
    
    // Likelihood - predict choice as a function of utility
    choice[i] ~ categorical_logit(to_vector(utility[C[i], :]));
    
    // After choice, calculate prediction error
    PEval      = outcome[i] - ev[C[i], choice[i]];     // Value prediction error
    PEfreq     = sign[i] - ef[C[i], choice[i]];        // Win-frequency prediction error
    PEfreq_fic = -sign[i]/1 - ef[C[i], (3-choice[i])]; // Lose-frequency prediction error?
    
    if(outcome[i] >= 0){  // If participant DID NOT lose
      // Update expected win-frequency of what participant DID NOT chose to do
      ef[C[i], (3-choice[i])] = ef[C[i], (3-choice[i])] + Apun[J[i], S[i]] * PEfreq_fic;
      // Update what participant chose
      ef[C[i], choice[i]] = ef[C[i], choice[i]] + Arew[J[i], S[i]] * PEfreq;
      ev[C[i], choice[i]] = ev[C[i], choice[i]] + Arew[J[i], S[i]] * PEval;
    } else { // If participant DID lose
      // Update expected win-frequency of what participant DID NOT choose to do
      ef[C[i], (3-choice[i])] = ef[C[i], (3-choice[i])] + Arew[J[i], S[i]] * PEfreq_fic;
      // Update what participant chose
      ef[C[i], choice[i]] = ef[C[i], choice[i]] + Apun[J[i], S[i]] * PEfreq;
      ev[C[i], choice[i]] = ev[C[i], choice[i]] + Apun[J[i], S[i]] * PEval;
    }
    
    // Update perseverance
    pers[C[i], choice[i]] = 1;
    pers = pers / (1 + K_tr);
    
    // Calculate expected value of card
    utility = ev + ef * betaF[J[i], S[i]] + pers * betaP[J[i], S[i]];
  }
}


// ---------------------------------------------------------------------------------------
generated quantities { // STILL NEED TO FIGURE OUT HOW EXACTLY THIS BLOCK WORKS, PARTICULARLY WITH THE PPC
// ------------------------------------
// test-retest correlations
  corr_matrix[2] R_Arew;
  corr_matrix[2] R_Apun;
  corr_matrix[2] R_K;
  corr_matrix[2] R_betaF;
  corr_matrix[2] R_betaP;
  
  // Reconstruct correlation matrix from cholesky factor
    // Note that we're multipling the cholesky factor by its transpose which gives us the correlation matrix
  R_Arew  = R_chol_Arew * R_chol_Arew';
  R_Apun  = R_chol_Apun * R_chol_Apun';
  R_K     = R_chol_K * R_chol_K';
  R_betaF = R_chol_betaF * R_chol_betaF';
  R_betaP = R_chol_betaP * R_chol_betaP';
  
// ------------------------------------
// For posterior predictive check?
  // Hyper(group)-parameters - these are 5 (number of parameters) x S (number of sessions) matrix of mus & sigmas, respectively, for each parameter
  vector<lower=0,upper=1>[2] mu_Arew;
  vector<lower=0,upper=1>[2] mu_Apun;
  vector<lower=0,upper=5>[2] mu_K;
  vector[2] mu_betaF;
  vector[2] mu_betaP;

  // For posterior predictive check
  real y_pred[N];

  // Set all posterior predictions to -1 (avoids NULL values)
  for (i in 1:N) {
      y_pred[i] = -1;
    }
  
  // Group-level means
  for(s in 1:n_S){
    mu_Arew[s] = Phi_approx(mu_p[s, 1]);
    mu_Apun[s] = Phi_approx(mu_p[s, 2]);
    mu_K[s] = Phi_approx(mu_p[s, 3]) * 5;
    mu_betaF[s] = mu_p[s, 4];
    mu_betaP[s] = mu_p[s, 5];
  }
  
  // ------------------------------------
  { // local section, this saves time and space
  // -------------------------------
  
    // Declare variables to calculate utility after each trial: These 4 (number of cards) x 2 (playing vs. not playing) matrices
    matrix[4,2] ef;
    matrix[4,2] ev;
    matrix[4,2] pers;
    matrix[4,2] utility;
      
    real PEval;
    real PEfreq;
    real PEfreq_fic;
    real K_tr;
  
  // Loop through RL algorithm
    for(i in 1:N){
      // (Re)Declare values for when trial == 1
        // Note, this re-initializes the variables for each participant on each session
      if(Trials[i] == 1){

        // Initialize starting values
        K_tr = pow(3, K[J[i], S[i]]) - 1;
        for(pp in 1:2){  // Looping over play/pass columns and assigning each 0 to each card
          ev[:,pp] = rep_vector(0,4);
          ef[:,pp] = rep_vector(0,4);
          pers[:,pp] = rep_vector(0,4);
          utility[:,pp] = rep_vector(0,4);
        }
      }
      
      // Likelihood - predict choice as a function of utility
      y_pred[i] = categorical_rng(softmax(to_vector(utility[C[i], :])));
      
      // After choice, calculate prediction error
      PEval      = outcome[i] - ev[C[i], choice[i]];     // Value prediction error
      PEfreq     = sign[i] - ef[C[i], choice[i]];        // Win-frequency prediction error
      PEfreq_fic = -sign[i]/1 - ef[C[i], (3-choice[i])]; // Lose-frequency prediction error?
      
      if(outcome[i] >= 0){  // If participant DID NOT lose
        // Update expected win-frequency of what participant DID NOT chose to do
        ef[C[i], (3-choice[i])] = ef[C[i], (3-choice[i])] + Apun[J[i], S[i]] * PEfreq_fic;
        // Update what participant chose
        ef[C[i], choice[i]] = ef[C[i], choice[i]] + Arew[J[i], S[i]] * PEfreq;
        ev[C[i], choice[i]] = ev[C[i], choice[i]] + Arew[J[i], S[i]] * PEval;
      } else { // If participant DID lose
        // Update expected win-frequency of what participant DID NOT choose to do
        ef[C[i], (3-choice[i])] = ef[C[i], (3-choice[i])] + Arew[J[i], S[i]] * PEfreq_fic;
        // Update what participant chose
        ef[C[i], choice[i]] = ef[C[i], choice[i]] + Apun[J[i], S[i]] * PEfreq;
        ev[C[i], choice[i]] = ev[C[i], choice[i]] + Apun[J[i], S[i]] * PEval;
      }
      
      // Update perseverance
      pers[C[i], choice[i]] = 1;
      pers = pers / (1 + K_tr);
      
      // Calculate expected value of card
      utility = ev + ef * betaF[J[i], S[i]] + pers * betaP[J[i], S[i]];
    }
  }
}












































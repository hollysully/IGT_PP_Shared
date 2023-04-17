

// ---------------------------------------------------------------------------------------
data{
  int<lower=1> N;                      // Number of participants =49
  int<lower=1> S;                      // Number of sessions = 2
  int<lower=1> T;                      // Total possile number of trials = 120
  int card[N,T,S];                     // Cards presented on each trial
  int Tsubj[N,S];                      // Total number of trials presented to each subject on each session
  int choice[N,T,S];                   // Choices on each trial
  real outcome[N,T,S];                 // Outcomes received on each trial
  real sign[N,T,S];                    // Signs of the outcome received on each trial
}


// ---------------------------------------------------------------------------------------
parameters{
// Declare parameters
  // Hyper(group)-parameters - these are  & sigmas, respectively, for each parameter
  matrix[S, 5] mu_p;   // S (number of sessions) x 5 (number of parameters) matrix of mus
  
  // Variance components of random intercepts (element 1) and random slopes (element 2)
  vector<lower=0>[2] sigma_Arew;
  vector<lower=0>[2] sigma_Apun;
  vector<lower=0>[2] sigma_K;
  vector<lower=0>[2] sigma_betaF;
  vector<lower=0>[2] sigma_betaP;

  // Subject-level "raw" parameters - i.e., independent/uncorrelated & normally distributed person-level (random-)effects
    // Note, these are 2 (1 RI + 1 RS) x J (number of subjects) matrices
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


// ---------------------------------------------------------------------------------------
transformed parameters{
// -------------------------------
// Declare transformed parameters
  // Parameters to use in the reinforcement-learning algorithm
    // Note, for each of these, we include the mus and the subject-level parameters (see below), allowing for shrinkage and subject-to-subject variability
  matrix<lower=0,upper=1>[N,S] Arew;  // Transformed Arews
  matrix<lower=0,upper=1>[N,S] Apun;  // Transformed Apuns
  matrix<lower=0,upper=5>[N,S] K;     // Transformed Ks
  matrix[N,S] betaF;
  matrix[N,S] betaP;
  
  // Untransformed subject-level parameters incorporating correlation between sessions
  matrix[2,N] Arew_tilde;
  matrix[2,N] Apun_tilde;
  matrix[2,N] K_tilde;
  matrix[2,N] betaF_tilde;
  matrix[2,N] betaP_tilde;
  
  // Untransformed linear component that includes group-means + subject-level means & session-offsets
  matrix[2,N] linear_Arew;
  matrix[2,N] linear_Apun;
  matrix[2,N] linear_K;
  matrix[2,N] linear_betaF;
  matrix[2,N] linear_betaP;
  
// -------------------------------
// Calculate transformed parameters
  
  // Untransformed subject-level parameters incorporating correlation between sessions
  Arew_tilde  = diag_pre_multiply(sigma_Arew, R_chol_Arew) * Arew_pr;  
  Apun_tilde  = diag_pre_multiply(sigma_Apun, R_chol_Apun) * Apun_pr;  
  K_tilde     = diag_pre_multiply(sigma_K, R_chol_K) * K_pr;  
  betaF_tilde = diag_pre_multiply(sigma_betaF, R_chol_betaF) * betaF_pr;  
  betaP_tilde = diag_pre_multiply(sigma_betaP, R_chol_betaP) * betaP_pr;
  
  // Here, I'm constructing the linear model of the parameters before any transformations
    // This could be embedded in the next loop, but for now, this helps to understand how the random
    // effects get incorporated in the model.
  for(s in 1:S){
    for(i in 1:N){
      // For each linear model, I'm adding the group-level mu (mu_p) for session s, a random-intercept term which is
        // equal to the individual-subjects' mu on session 1, and a random 'slope' effect which captures the deviation
        // of session 2 from session 1 at the person-level (note that s-1 allows this to represent the deviation from
        // session 1). I think the last term could be directly extended as an actual slope if s > 2. In addition, if we
        // effect-code sessions as -1 for session 1 and +1 for session 2 and use that in place of (s-1), then the first
        // element of the tilde parameter should be the overall individual-subject mus which I believe would also
        // be equivalent to taking the average of session 1 and session 2 parameters for individual subjects
      linear_Arew[s,i]  = mu_p[s,1] + Arew_tilde[1,i] + Arew_tilde[2,i]*(s-1);
      linear_Apun[s,i]  = mu_p[s,2] + Apun_tilde[1,i] + Apun_tilde[2,i]*(s-1);
      linear_K[s,i]     = mu_p[s,3] + K_tilde[1,i] + K_tilde[2,i]*(s-1);
      linear_betaF[s,i] = mu_p[s,4] + betaF_tilde[1,i] + betaF_tilde[2,i]*(s-1);
      linear_betaP[s,i] = mu_p[s,5] + betaP_tilde[1,i] + betaP_tilde[2,i]*(s-1);
      
    }
  }
  
  // Calculate & transform Arew, Apun, & K to use in RL algorithm
  for(s in 1:S){    // Loop over sessions
    for(i in 1:N){  // Loop over subjects - This is structured the same as the OG joint retest model such that Arew,
                        // Apun, & K are looped over for individual subjects whereas betaF & betaP are vectorized for
                        // individual subjects.
                        // Maybe to avoid nesting a function within a function - e.g., to_vector(Phi_approx... 
      Arew[i,s]  = Phi_approx(linear_Arew[s,i]);
      Apun[i,s]  = Phi_approx(linear_Apun[s,i]);
      K[i,s]     = Phi_approx(linear_K[s,i]) * 5;
      betaF[i,s] = linear_betaF[s,i];
      betaP[i,s] = linear_betaP[s,i];
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
  
  // 'Variance' components
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
  for(i in 1:N){         // Loop through individual participants
    for(s in 1:S){       // Loop though sessions for participant i
      if(Tsubj[i,s] > 0){    // If we have data for participant i on session s, run through RL algorithm
        
        // Initialize starting values
        K_tr = pow(3, K[i,s]) - 1;
        for(pp in 1:2){  // Looping over play/pass columns and assigning each 0 to each card
          ev[:,pp] = rep_vector(0,4);
          ef[:,pp] = rep_vector(0,4);
          pers[:,pp] = rep_vector(0,4);
          utility[:,pp] = rep_vector(0,4);
        }
        
        for(t in 1:Tsubj[i,s]){ // Run through RL algorithm trial-by-trial
          
        // Likelihood - predict choice as a function of utility
        choice[i,t,s] ~ categorical_logit(to_vector(utility[card[i,t,s], :]));
        
        // After choice, calculate prediction error
        PEval      = outcome[i,t,s] - ev[card[i,t,s], choice[i,t,s]];     // Value prediction error
        PEfreq     = sign[i,t,s] - ef[card[i,t,s], choice[i,t,s]];        // Win-frequency prediction error
        PEfreq_fic = -sign[i,t,s]/1 - ef[card[i,t,s], (3-choice[i,t,s])]; // Lose-frequency prediction error?
        
        if(outcome[i,t,s] >= 0){  // If participant DID NOT lose
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
  real choice_pred[N,T,S];

  // Set all posterior predictions to -1 (avoids NULL values)
  for(i in 1:N) {
    for(s in 1:S){
      for(t in 1:T){
        choice_pred[i,t,s] = -1;
      }
    }
  }
  
  // Group-level means
  for(s in 1:S){
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
  
    // -------------------------------
    for(i in 1:N){         // Loop through individual participants
      for(s in 1:S){       // Loop though sessions for participant i
        if(Tsubj[i,s] > 0){    // If we have data for participant i on session s, run through RL algorithm
          
          // Initialize starting values
          K_tr = pow(3, K[i,s]) - 1;
          for(pp in 1:2){  // Looping over play/pass columns and assigning each 0 to each card
            ev[:,pp] = rep_vector(0,4);
            ef[:,pp] = rep_vector(0,4);
            pers[:,pp] = rep_vector(0,4);
            utility[:,pp] = rep_vector(0,4);
          }
          
          for(t in 1:Tsubj[i,s]){ // Run through RL algorithm trial-by-trial
            
          // Likelihood - predict choice as a function of utility
          choice_pred[i,t,s] = categorical_rng(softmax(to_vector(utility[card[i,t,s], :])));
          
          // After choice, calculate prediction error
          PEval      = outcome[i,t,s] - ev[card[i,t,s], choice[i,t,s]];     // Value prediction error
          PEfreq     = sign[i,t,s] - ef[card[i,t,s], choice[i,t,s]];        // Win-frequency prediction error
          PEfreq_fic = -sign[i,t,s]/1 - ef[card[i,t,s], (3-choice[i,t,s])]; // Lose-frequency prediction error?
          
          if(outcome[i,t,s] >= 0){  // If participant DID NOT lose
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












































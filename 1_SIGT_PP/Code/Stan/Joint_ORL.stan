functions {
  // to compute empirical correlation matrix given input matrix
  matrix empirical_correlation(matrix X) {
    int N = rows(X);             
    int M = cols(X);             
    matrix[N, M] centered;
    vector[M] std_devs;
    matrix[M, M] Sigma;
    matrix[M, M] R;
    
    // calculate covariance matrix
    for (j in 1:M) {
      centered[,j] = X[,j] - mean(X[,j]);
    }
    Sigma = (centered' * centered) / (N - 1);
    
    // Calculate correlation matrix
    std_devs = sqrt(diagonal(Sigma));
    for (i in 1:M) {
      for (j in 1:M) {
        R[i,j] = Sigma[i,j] / (std_devs[i] * std_devs[j]);
      }
    }
    
    return R;
  }
  
  real Phi_approx_group_mean_rng(real mu, real sigma, int n_samples) {
    vector[n_samples] par;
    for (i in 1:n_samples) {
      par[i] = Phi_approx(normal_rng(mu, sigma));
    }
    return mean(par);
  }
  
  matrix Phi_approx_corr_rng(vector mu, vector sigma, matrix R, int n_samples) {
    int N = rows(R);
    int M = cols(R);
    matrix[N,M] Sigma = quad_form_diag(R, sigma);
    matrix[n_samples, M] X_sim;
    for (i in 1:n_samples) {
      X_sim[i,] = Phi_approx(multi_normal_rng(mu, Sigma)');
    }
    return empirical_correlation(X_sim);
  }
}

data {
  int<lower=1> N;            // Number of participants
  int<lower=1> S;            // Number of sessions
  int<lower=1> T;            // Total possile number of trials
  array[N,T,S] int card;     // Cards presented on each trial
  array[N,S] int Tsubj;      // Total number of trials presented to each subject on each session
  array[N,T,S] int choice;   // Choices on each trial
  array[N,T,S] real outcome; // Outcomes received on each trial
  array[N,T,S] real sign;    // Signs of the outcome received on each trial
}


parameters {
// Declare parameters
  // Hyper(group)-parameters
  matrix[S, 4] mu_p;   // S (number of sessions) x 5 (number of parameters) matrix of mus
  vector<lower=0>[S] sigma_Arew;
  vector<lower=0>[S] sigma_Apun;
  vector<lower=0>[S] sigma_betaF;
  vector<lower=0>[S] sigma_betaB;

  // Subject-level "raw" parameters - i.e., independent/uncorrelated & normally distributed person-level (random-)effects
    // Note, these are S (number of sessions) x N (number of subjects) matrices
  matrix[S,N] Arew_pr;
  matrix[S,N] Apun_pr;
  matrix[S,N] betaF_pr;
  matrix[S,N] betaB_pr;
  
  // Correlation matrices for correlating between sessions
  cholesky_factor_corr[S] R_chol_Arew;
  cholesky_factor_corr[S] R_chol_Apun;
  cholesky_factor_corr[S] R_chol_betaF;
  cholesky_factor_corr[S] R_chol_betaB;
}

transformed parameters {
// Declare transformed parameters
  // Parameters to use in the reinforcement-learning algorithm
    // Note, for each of these, we include the mus and the subject-level parameters (see below), allowing for shrinkage and subject-to-subject variability
  matrix<lower=0,upper=1>[N,S] Arew;
  matrix<lower=0,upper=1>[N,S] Apun;
  matrix[N,S] betaF;
  matrix[N,S] betaB;
  
  // Untransformed subject-level parameters incorporating correlation between sessions
  matrix[S,N] Arew_tilde;
  matrix[S,N] Apun_tilde;
  matrix[S,N] betaF_tilde;
  matrix[S,N] betaB_tilde;
  
  // Calculate transformed parameters
  
  // Untransformed subject-level parameters incorporating correlation between sessions
  Arew_tilde  = diag_pre_multiply(sigma_Arew, R_chol_Arew) * Arew_pr;
  Apun_tilde  = diag_pre_multiply(sigma_Apun, R_chol_Apun) * Apun_pr;
  betaF_tilde = diag_pre_multiply(sigma_betaF, R_chol_betaF) * betaF_pr;
  betaB_tilde = diag_pre_multiply(sigma_betaB, R_chol_betaB) * betaB_pr;
  
  // Calculate & transform Arew, Apun, & K to use in RL algorithm
  for(s in 1:S){    // Loop over sessions
    for(i in 1:N){  // Loop over subjects - This is structured the same as the OG joint retest model such that Arew,
                        // Apun, & K are looped over for individual subjects whereas betaF & betaB are vectorized for
                        // individual subjects.
                        // Maybe to avoid nesting a function within a function - e.g., to_vector(Phi_approx... 
      Arew[i,s] = Phi_approx(mu_p[s,1] + Arew_tilde[s,i]);
      Apun[i,s] = Phi_approx(mu_p[s,2] + Apun_tilde[s,i]);
    }
    
  // Calculate betaF & betaB to use in RL algorithm
  betaF[:,s] = to_vector(mu_p[s,3] + betaF_tilde[s,:]);
  betaB[:,s] = to_vector(mu_p[s,4] + betaB_tilde[s,:]);
  }
}

model {
  // Declare variables to calculate utility after each trial: These 4 (number of cards) x 2 (playing vs. not playing) matrices
  vector[4] ef;
  vector[4] ev;
  vector[4] utility;
  
  real ef_chosen;  
  real PEval;
  real PEfreq;
  vector[4] PEfreq_fic;
  
  // Priors
  // Hyperparameters for correlations
  R_chol_Arew   ~ lkj_corr_cholesky(1);
  R_chol_Apun   ~ lkj_corr_cholesky(1);
  R_chol_betaF  ~ lkj_corr_cholesky(1);
  R_chol_betaB  ~ lkj_corr_cholesky(1);
  
  // Hyperparameters for RL learning algorithm
  for(s in 1:S){
    mu_p[s,:]   ~ normal(0, 1);
  }
  sigma_Arew  ~ normal(0, 0.2);
  sigma_Apun  ~ normal(0, 0.2);
  sigma_betaF ~ cauchy(0, 1);
  sigma_betaB ~ cauchy(0, 1);
  
  // Subject-level parameters
  to_vector(Arew_pr)  ~ normal(0, 1.0);
  to_vector(Apun_pr)  ~ normal(0, 1.0);
  to_vector(betaF_pr) ~ normal(0, 1.0);
  to_vector(betaB_pr) ~ normal(0, 1.0);
  
  for (i in 1:N) {         // Loop through individual participants
    for (s in 1:S) {       // Loop though sessions for participant i
      if (Tsubj[i,s] > 0) {    // If we have data for participant i on session s, run through RL algorithm
        
        // Initialize starting values
        ev = rep_vector(0,4);
        ef = rep_vector(0,4);
        utility = rep_vector(0,4);
        
        for (t in 1:Tsubj[i,s]) { // Run through RL algorithm trial-by-trial
          // Calculate expected value of card
          utility = ev + ef * betaF[i,s] + betaB[i,s];
          
          // Likelihood - predict choice as a function of utility
          choice[i,t,s] ~ categorical_logit(to_vector({utility[card[i,t,s]], 0}));
          // this could equivalently be specified as: 
          // choice[i,t,s] ~ bernoulli_logit(utility[card[i,t,s]]);
          // if choice = 0 for pass and 1 for play. Keeping categorical_*
          // is consistent with other models
          
          if(choice[i,t,s]==1){
            // After choice, calculate prediction error
            PEval      = outcome[i,t,s] - ev[card[i,t,s]];     // Value prediction error
            PEfreq     = sign[i,t,s] - ef[card[i,t,s]];        // Win-frequency prediction error
            PEfreq_fic = -sign[i,t,s]/3 - ef;
            ef_chosen  = ef[card[i,t,s]];
            
            if (outcome[i,t,s] >= 0) {  // If participant DID NOT lose
              // Update expected win-frequency of what participant DID NOT chose to do
              ef = ef + Apun[i,s] * PEfreq_fic;
              // Update what participant chose
              ef[card[i,t,s]] = ef_chosen + Arew[i,s] * PEfreq;
              ev[card[i,t,s]] = ev[card[i,t,s]] + Arew[i,s] * PEval;
            } else { // If participant DID lose
              // Update expected win-frequency of what participant DID NOT choose to do
              ef = ef + Arew[i,s] * PEfreq_fic;
              // Update what participant chose
              ef[card[i,t,s]] = ef_chosen + Apun[i,s] * PEfreq;
              ev[card[i,t,s]] = ev[card[i,t,s]] + Apun[i,s] * PEval;
            }
          }
        }
      }
    }
  }
}

generated quantities {
  // Hyper(group)-parameters - these are 5 (number of parameters) x S (number of sessions) matrix of mus & sigmas, respectively, for each parameter
  vector<lower=0,upper=1>[S] mu_Arew;
  vector<lower=0,upper=1>[S] mu_Apun;
  vector[S] mu_betaF;
  vector[S] mu_betaB;
  array[N] real log_lik;

  // For posterior predictive check
  array[N,T,S] real choice_pred;

  // test-retest correlations
  corr_matrix[S] R_Arew;
  corr_matrix[S] R_Apun;
  corr_matrix[S] R_betaF;
  corr_matrix[S] R_betaB;
  
  // Reconstruct correlation matrix from cholesky factor
    // Note that we're multipling the cholesky factor by its transpose which gives us the correlation matrix
  R_Arew  = Phi_approx_corr_rng(mu_p[,1], sigma_Arew, R_chol_Arew * R_chol_Arew', 10000);
  R_Apun  = Phi_approx_corr_rng(mu_p[,2], sigma_Apun, R_chol_Apun * R_chol_Apun', 10000);
  R_betaF = R_chol_betaF * R_chol_betaF';
  R_betaB = R_chol_betaB * R_chol_betaB';
  
  // Set all posterior predictions to -1 (avoids NULL values)
  for (i in 1:N) {
    for (s in 1:S) {
      for (t in 1:T) {
        choice_pred[i,t,s] = -1;
      }
    }
  }
  
  // Compute group-level means
  for (s in 1:S) {
    mu_Arew[s] = Phi_approx_group_mean_rng(mu_p[s, 1], sigma_Arew[s], 10000);
    mu_Apun[s] = Phi_approx_group_mean_rng(mu_p[s, 2], sigma_Apun[s], 10000);
    mu_betaF[s] = mu_p[s, 3];
    mu_betaB[s] = mu_p[s, 4];
  }
  
  { // local section, this saves time and space
    // Declare variables to calculate utility after each trial: These 4 (number of cards) x 2 (playing vs. not playing) matrices
    vector[4] ef;
    vector[4] ev;
    vector[4] utility;
    
    real ef_chosen;
    real PEval;
    real PEfreq;
    vector[4] PEfreq_fic;
  
    for (i in 1:N) {         // Loop through individual participants
      log_lik[i] = 0;        // Initialize log_lik
      
      for (s in 1:S) {       // Loop though sessions for participant i
        if (Tsubj[i,s] > 0) {    // If we have data for participant i on session s, run through RL algorithm
          
          // Initialize starting values
          ev = rep_vector(0,4);
          ef = rep_vector(0,4);
          utility = rep_vector(0,4);
          
          for (t in 1:Tsubj[i,s]) { // Run through RL algorithm trial-by-trial
            // Calculate expected value of card
            utility = ev + ef * betaF[i,s] + betaB[i,s];
            
            // softmax choice
            log_lik[i] += categorical_logit_lpmf(choice[i,t,s]|to_vector({utility[card[i,t,s]], 0}));
            
            // Likelihood - predict choice as a function of utility
            choice_pred[i,t,s] = categorical_rng(softmax(to_vector({utility[card[i,t,s]], 0})));
            
            if(choice[i,t,s]==1) {
              // After choice, calculate prediction error
              PEval      = outcome[i,t,s] - ev[card[i,t,s]];     // Value prediction error
              PEfreq     = sign[i,t,s] - ef[card[i,t,s]];        // Win-frequency prediction error
              PEfreq_fic = -sign[i,t,s]/3 - ef;
              ef_chosen = ef[card[i,t,s]];
              
              if (outcome[i,t,s] >= 0) {  // If participant DID NOT lose
                // Update expected win-frequency of what participant DID NOT chose to do
                ef = ef + Apun[i,s] * PEfreq_fic;
                // Update what participant chose
                ef[card[i,t,s]] = ef_chosen + Arew[i,s] * PEfreq;
                ev[card[i,t,s]] = ev[card[i,t,s]] + Arew[i,s] * PEval;
              } else { // If participant DID lose
                // Update expected win-frequency of what participant DID NOT choose to do
                ef = ef + Arew[i,s] * PEfreq_fic;
                // Update what participant chose
                ef[card[i,t,s]] = ef_chosen + Apun[i,s] * PEfreq;
                ev[card[i,t,s]] = ev[card[i,t,s]] + Apun[i,s] * PEval;
              }
            }
          }
        }
      }
    }
  }
}

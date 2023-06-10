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
  int<lower=1> N;      // Number of participants
  int<lower=1> S;      // Number of sessions
  
  // IGT-SPECIFIC DATA
  int<lower=1> T;      // Total possile number of trials
  int card[N,T,S];     // Cards presented on each trial
  int Tsubj[N,S];      // Total number of trials presented to each subject on each session
  int choice[N,T,S];   // Choices on each trial
  real outcome[N,T,S]; // Outcomes received on each trial
  real sign[N,T,S];    // Signs of the outcome received on each trial
  
  // SELF-REPORT-SPECIFIC DATA
  int missing[N,S];    // Matrix of missingness for current scale
  int score[N,S];      // Matrix of scores for current scale
  int M;               // Max score for current scale
}


parameters {
  // Hyper(group)-parameters
  vector[6] mu_p;         // Vector of 5 ORL mus and 1 binomial Mu
  real<lower=0>[6] sigma; // Vector of 5 ORL sigmas and 1 binomial sigma
  
  // Untransformed & non-correlated subject-level parameters
  vector[N] Arew_pr;  
  vector[N] Apun_pr;  
  vector[N] K_pr;   
  vector[N] betaF_pr;
  vector[N] betaP_pr;
  vector[N] theta_pr;
  
  // Correlation matrices for correlating between parameters
  cholesky_factor_corr[6] R_chol;
}

transformed parameters {
  // Transformed & correlated subject-level parameters
  vector<lower=0,upper=1>[N] Arew; 
  vector<lower=0,upper=1>[N] Apun; 
  vector<lower=0,upper=5>[N] K;    
  vector[N] betaF;
  vector[N] betaP;
  vector<lower=0, upper=1>[N] theta;
  
  // Untransformed but correlated subject-level parameters
  vector[N] Arew_tilde;
  vector[N] Apun_tilde;
  vector[N] K_tilde;
  vector[N] betaF_tilde;
  vector[N] betaP_tilde;
  vector[N] theta_tilde;
  
  // Calculate transformed parameters
  
  // Untransformed subject-level parameters incorporating correlation between parameters
  Arew_tilde  = diag_pre_multiply(sigma, R_chol) * Arew_pr;  
  Apun_tilde  = diag_pre_multiply(sigma, R_chol) * Apun_pr;  
  K_tilde     = diag_pre_multiply(sigma, R_chol) * K_pr;  
  betaF_tilde = diag_pre_multiply(sigma, R_chol) * betaF_pr;  
  betaP_tilde = diag_pre_multiply(sigma, R_chol) * betaP_pr;
  theta_tilde = diag_pre_multiply(sigma, R_chol) * theta_pr;
  
  // Transform parameters to use in likelihoods
  for(i in 1:N){  // Loop over participants
    Arew[i] = Phi_approx(mu_p[1] + Arew_tilde[i]);
    Apun[i] = Phi_approx(mu_p[2] + Apun_tilde[i]);
    K[i]    = Phi_approx(mu_p[3] + K_tilde[i]) * 5;
    betaF[i] = mu_p[4] + betaF_tilde[i];
    betaP[i] = mu_p[5] + betaP_tilde[i];
    theta[i] = Phi_approx(mu_p[6] + theta_tilde[i]);
  }
}

model {
  // Declare variables to calculate utility after each trial for ORL: These 4 (number of cards) x 2 (playing vs. not playing) matrices
  matrix[4,2] ef;
  matrix[4,2] ev;
  matrix[4,2] pers;
  matrix[4,2] utility;
    
  real PEval;
  real PEfreq;
  real PEfreq_fic;
  real K_tr;
  
  // Hyperpriors
  mu_p     ~ normal(0, 1);
  sigma[1] ~ normal(0, 0.2); // Arew
  sigma[2] ~ normal(0, 0.2); // Apun
  sigma[3] ~ normal(0, 0.2); // K
  sigma[4] ~ cauchy(0, 1);   // betaF
  sigma[5] ~ cauchy(0, 1);   // betaP
  sigma[6] ~ cauchy(0, 1);   // theta
  
  R_chol   ~ lkj_corr_cholesky(1); // Correlation matrix
  
  // Priors for untransformed and non-correated subject-level parameters
  Arew_pr  ~ normal(0, 1.0);
  Apun_pr  ~ normal(0, 1.0);
  K_pr     ~ normal(0, 1.0);
  betaF_pr ~ normal(0, 1.0);
  betaP_pr ~ normal(0, 1.0);
  theta_pr ~ normal(0, 1.0);
  
  // Loop through both sets of data
  for (i in 1:N) {         // Loop through individual participants
    for (s in 1:S) {       // Loop though sessions for participant i
    
      // SELF-REPORT LIKELIHOOD - predict number of "endorsements" out of max
      if(missing[i,s] == 0){
        score[i,s] ~ binomial(M, theta[i]);
      }
      
      if (Tsubj[i,s] > 0) {    // If we have data for participant i on session s, run through RL algorithm
        
        // Initialize starting values
        K_tr = pow(3, K[i]) - 1;
        for (pp in 1:2) {  // Looping over play/pass columns and assigning each 0 to each card
          ev[:,pp] = rep_vector(0,4);
          ef[:,pp] = rep_vector(0,4);
          pers[:,pp] = rep_vector(0,4);
          utility[:,pp] = rep_vector(0,4);
        }
        
        for (t in 1:Tsubj[i,s]) { // Run through RL algorithm trial-by-trial
          
        // ORL LIKELIHOOD - predict choice as a function of utility
        choice[i,t,s] ~ categorical_logit(to_vector(utility[card[i,t,s], :]));
        
        // After choice, calculate prediction error
        PEval      = outcome[i,t,s] - ev[card[i,t,s], choice[i,t,s]];     // Value prediction error
        PEfreq     = sign[i,t,s] - ef[card[i,t,s], choice[i,t,s]];        // Win-frequency prediction error
        PEfreq_fic = -sign[i,t,s]/1 - ef[card[i,t,s], (3-choice[i,t,s])]; // Lose-frequency prediction error?
        
        if (outcome[i,t,s] >= 0) {  // If participant DID NOT lose
          // Update expected win-frequency of what participant DID NOT chose to do
          ef[card[i,t,s], (3-choice[i,t,s])] = ef[card[i,t,s], (3-choice[i,t,s])] + Apun[i] * PEfreq_fic;
          // Update what participant chose
          ef[card[i,t,s], choice[i,t,s]] = ef[card[i,t,s], choice[i,t,s]] + Arew[i] * PEfreq;
          ev[card[i,t,s], choice[i,t,s]] = ev[card[i,t,s], choice[i,t,s]] + Arew[i] * PEval;
        } else { // If participant DID lose
          // Update expected win-frequency of what participant DID NOT choose to do
          ef[card[i,t,s], (3-choice[i,t,s])] = ef[card[i,t,s], (3-choice[i,t,s])] + Arew[i] * PEfreq_fic;
          // Update what participant chose
          ef[card[i,t,s], choice[i,t,s]] = ef[card[i,t,s], choice[i,t,s]] + Apun[i] * PEfreq;
          ev[card[i,t,s], choice[i,t,s]] = ev[card[i,t,s], choice[i,t,s]] + Apun[i] * PEval;
        }
        
        // Update perseverance
        pers[card[i,t,s], choice[i,t,s]] = 1;
        pers = pers / (1 + K_tr);
        
        // Calculate expected value of card
        utility = ev + ef * betaF[i] + pers * betaP[i];
        }
      }
    }
  }
}

generated quantities {
  // Hyper(group)-parameters 
  real<lower=0,upper=1> mu_Arew;
  real<lower=0,upper=1> mu_Apun;
  real<lower=0,upper=5> mu_K;
  real mu_betaF;
  real mu_betaP;
  real<lower=0,upper=M> mu_theta;

  // For posterior predictive check
  real choice_pred[N,T,S];
  real score_pred[N,S];

  // test-retest correlations
  corr_matrix[6] R;
  
  // Reconstruct correlation matrix from cholesky factor
  R = Phi_approx_corr_rng(mu_p, sigma, R_chol * R_chol', 10000);
  
  // Set all posterior predictions to -1 (avoids NULL values)
  for (i in 1:N) {
    for (s in 1:S) {
      score_pred[i,s] = theta*M;
      for (t in 1:T) {
        choice_pred[i,t,s] = -1;
      }
    }
  }
  
  // Compute group-level means
  mu_Arew = Phi_approx_group_mean_rng(mu_p[1], sigma, 10000);
  mu_Apun = Phi_approx_group_mean_rng(mu_p[2], sigma, 10000);
  mu_K = Phi_approx_group_mean_rng(mu_p[3], sigma, 10000) * 5; 
  mu_betaF = mu_p[4];
  mu_betaP = mu_p[5];
  mu_theta = Phi_approx_group_mean_rng(mu_p[6], sigma, 10000);
  
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
  
    for (i in 1:N) {    // Loop through individual participants
      for (s in 1:S) {    // Loop though sessions for participant i
    
        // SELF-REPORT LIKELIHOOD - predict number of "endorsements" out of max
        score_pred[i,s] ~ binomial_rng(M, theta[i]);
        
        if (Tsubj[i,s] > 0) {    // If we have data for participant i on session s, run through RL algorithm
          
          // Initialize starting values
          K_tr = pow(3, K[i]) - 1;
          for (pp in 1:2) {  // Looping over play/pass columns and assigning each 0 to each card
            ev[:,pp] = rep_vector(0,4);
            ef[:,pp] = rep_vector(0,4);
            pers[:,pp] = rep_vector(0,4);
            utility[:,pp] = rep_vector(0,4);
          }
          
          for (t in 1:Tsubj[i,s]) { // Run through RL algorithm trial-by-trial
            // ORL LIKELIHOOD - predict choice as a function of utility
            choice_pred[i,t,s] = categorical_rng(softmax(to_vector(utility[card[i,t,s], :])));
            
            // After choice, calculate prediction error
            PEval      = outcome[i,t,s] - ev[card[i,t,s], choice[i,t,s]];     // Value prediction error
            PEfreq     = sign[i,t,s] - ef[card[i,t,s], choice[i,t,s]];        // Win-frequency prediction error
            PEfreq_fic = -sign[i,t,s]/1 - ef[card[i,t,s], (3-choice[i,t,s])]; // Lose-frequency prediction error?
            
            if (outcome[i,t,s] >= 0) {  // If participant DID NOT lose
              // Update expected win-frequency of what participant DID NOT chose to do
              ef[card[i,t,s], (3-choice[i,t,s])] = ef[card[i,t,s], (3-choice[i,t,s])] + Apun[i] * PEfreq_fic;
              // Update what participant chose
              ef[card[i,t,s], choice[i,t,s]] = ef[card[i,t,s], choice[i,t,s]] + Arew[i] * PEfreq;
              ev[card[i,t,s], choice[i,t,s]] = ev[card[i,t,s], choice[i,t,s]] + Arew[i] * PEval;
            } else { // If participant DID lose
              // Update expected win-frequency of what participant DID NOT choose to do
              ef[card[i,t,s], (3-choice[i,t,s])] = ef[card[i,t,s], (3-choice[i,t,s])] + Arew[i] * PEfreq_fic;
              // Update what participant chose
              ef[card[i,t,s], choice[i,t,s]] = ef[card[i,t,s], choice[i,t,s]] + Apun[i] * PEfreq;
              ev[card[i,t,s], choice[i,t,s]] = ev[card[i,t,s], choice[i,t,s]] + Apun[i] * PEval;
            }
            
            // Update perseverance
            pers[card[i,t,s], choice[i,t,s]] = 1;
            pers = pers / (1 + K_tr);
            
            // Calculate expected value of card
            utility = ev + ef * betaF[i] + pers * betaP[i];
          }
        }
      }
    }
  }
}












































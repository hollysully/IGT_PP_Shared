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
  int<lower=1> N;                      // Number of participants
  int<lower=1> S;                      // Number of sessions
  int<lower=1> T;                      // Total possile number of trials
  int card[N,T,S];                     // Cards presented on each trial
  int Tsubj[N,S];                      // Total number of trials presented to each subject on each session
  int choice[N,T,S];                   // Choices on each trial
  real outcome[N,T,S];                 // Outcomes received on each trial
  real sign[N,T,S];                    // Signs of the outcome received on each trial
}


parameters {
// Declare parameters
  // Hyper(group)-parameters
  vector[5] mu_p;   // S (number of sessions) x 5 (number of parameters) matrix of mus
  real<lower=0> sigma_Arew;
  real<lower=0> sigma_Apun;
  real<lower=0> sigma_K;
  real<lower=0> sigma_betaF;
  real<lower=0> sigma_betaP;

  // Subject-level "raw" parameters - i.e., independent/uncorrelated & normally distributed person-level (random-)effects
    // Note, these are S (number of sessions) x N (number of subjects) matrices
  vector[N] Arew_pr;  
  vector[N] Apun_pr;  
  vector[N] K_pr;   
  vector[N] betaF_pr;
  vector[N] betaP_pr;
}

transformed parameters {
// Declare transformed parameters
  // Parameters to use in the reinforcement-learning algorithm
    // Note, for each of these, we include the mus and the subject-level parameters (see below), allowing for shrinkage and subject-to-subject variability
  vector<lower=0,upper=1>[N] Arew; 
  vector<lower=0,upper=1>[N] Apun; 
  vector<lower=0,upper=5>[N] K;    
  vector[N] betaF;
  vector[N] betaP;
  
  // Untransformed subject-level parameters
  vector[N] Arew_tilde;
  vector[N] Apun_tilde;
  vector[N] K_tilde;
  vector[N] betaF_tilde;
  vector[N] betaP_tilde;
  
  // Calculate transformed parameters
  
  // Untransformed subject-level parameters incorporating correlation between sessions
  Arew_tilde  = Arew_pr*sigma_Arew;  
  Apun_tilde  = Apun_pr*sigma_Apun;  
  K_tilde     = K_pr*sigma_K;  
  betaF_tilde = betaF_pr*sigma_betaF;  
  betaP_tilde =  betaP_pr*sigma_betaP;
  
  // Calculate & transform Arew, Apun, & K to use in RL algorithm
  for(i in 1:N){  // Loop over participants
    Arew[i] = Phi_approx(mu_p[1] + Arew_tilde[i]);
    Apun[i] = Phi_approx(mu_p[2] + Apun_tilde[i]);
    K[i]    = Phi_approx(mu_p[3] + K_tilde[i]) * 5;
    betaF[i] = mu_p[4] + betaF_tilde[i];
    betaP[i] = mu_p[5] + betaP_tilde[i];
  }
}

model {
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
  // Hyperparameters for RL learning algorithm
  mu_p        ~ normal(0, 1);
  sigma_Arew  ~ normal(0, 0.2);
  sigma_Apun  ~ normal(0, 0.2);
  sigma_K     ~ normal(0, 0.2);
  sigma_betaF ~ cauchy(0, 1);
  sigma_betaP ~ cauchy(0, 1);
  
  // Subject-level parameters
  Arew_pr  ~ normal(0, 1.0);
  Apun_pr  ~ normal(0, 1.0);
  K_pr     ~ normal(0, 1.0);
  betaF_pr ~ normal(0, 1.0);
  betaP_pr ~ normal(0, 1.0);
  
  for (i in 1:N) {         // Loop through individual participants
    for (s in 1:S) {       // Loop though sessions for participant i
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
          
        // Likelihood - predict choice as a function of utility
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
  // Hyper(group)-parameters - these are 5 (number of parameters) x S (number of sessions) matrix of mus & sigmas, respectively, for each parameter
  real<lower=0,upper=1> mu_Arew;
  real<lower=0,upper=1> mu_Apun;
  real<lower=0,upper=5> mu_K;
  real mu_betaF;
  real mu_betaP;

  // For posterior predictive check
  real choice_pred[N,T,S];
  
  // Set all posterior predictions to -1 (avoids NULL values)
  for (i in 1:N) {
    for (s in 1:S) {
      for (t in 1:T) {
        choice_pred[i,t,s] = -1;
      }
    }
  }
  
  // Compute group-level means
  mu_Arew = Phi_approx_group_mean_rng(mu_p[1], sigma_Arew, 10000);
  mu_Apun = Phi_approx_group_mean_rng(mu_p[2], sigma_Apun, 10000);
  mu_K = Phi_approx_group_mean_rng(mu_p[3], sigma_K, 10000) * 5; 
  mu_betaF = mu_p[4];
  mu_betaP = mu_p[5];
  
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
  
    for (i in 1:N) {         // Loop through individual participants
      for (s in 1:S) {       // Loop though sessions for participant i
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
            // Likelihood - predict choice as a function of utility
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












































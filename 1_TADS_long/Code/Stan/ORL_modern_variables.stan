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
  array[N,T,S] int card;               // Cards presented on each trial
  array[N,S] int Tsubj;                // Total number of trials presented to each subject on each session
  array[N,T,S] int choice;             // Choices on each trial
  array[N,T,S] real outcome;           // Outcomes received on each trial
  array[N,T,S] real sign;              // Signs of the outcome received on each trial
}


parameters {
// Declare parameters
  // Hyper(group)-parameters
  matrix[S, 5] mu_p;   // S (number of sessions) x 5 (number of parameters) matrix of mus
  vector<lower=0>[S] sigma_Arew;
  vector<lower=0>[S] sigma_Apun;
  vector<lower=0>[S] sigma_K;
  vector<lower=0>[S] sigma_betaF;
  vector<lower=0>[S] sigma_betaP;

  // Subject-level "raw" parameters - i.e., independent/uncorrelated & normally distributed person-level (random-)effects
    // Note, these are S (number of sessions) x N (number of subjects) matrices
  matrix[S,N] Arew_pr;  
  matrix[S,N] Apun_pr;  
  matrix[S,N] K_pr;   
  matrix[S,N] betaF_pr;
  matrix[S,N] betaP_pr;
}

transformed parameters {
// Declare transformed parameters
  // Parameters to use in the reinforcement-learning algorithm
    // Note, for each of these, we include the mus and the subject-level parameters (see below), allowing for shrinkage and subject-to-subject variability
  matrix<lower=0,upper=1>[N,S] Arew; 
  matrix<lower=0,upper=1>[N,S] Apun; 
  matrix<lower=0,upper=5>[N,S] K;    
  matrix[N,S] betaF;
  matrix[N,S] betaP;
  
  // Calculate & transform Arew, Apun, & K to use in RL algorithm
  for(s in 1:S){    // Loop over sessions
    for(i in 1:N){  // Loop over subjects - This is structured the same as the OG joint retest model such that Arew,
                        // Apun, & K are looped over for individual subjects whereas betaF & betaP are vectorized for
                        // individual subjects.
                        // Maybe to avoid nesting a function within a function - e.g., to_vector(Phi_approx... 
      Arew[i,s] = Phi_approx(mu_p[s,1] + sigma_Arew[s] * Arew_pr[s,i]);
      Apun[i,s] = Phi_approx(mu_p[s,2] + sigma_Apun[s] * Apun_pr[s,i]);
      K[i,s]    = Phi_approx(mu_p[s,3] + sigma_K[s] * K_pr[s,i]) * 5;
      betaF[i,s] = mu_p[s,4] + sigma_betaF[s] * betaF_pr[s,i];
      betaP[i,s] = mu_p[s,5] + sigma_betaP[s] * betaP_pr[s,i];
    }
  }
}

model {
  // Declare variables to calculate utility after each trial: These 4 (number of cards) x 2 (playing vs. not playing) matrices
  vector[4] ef;
  vector[4] ev;
  vector[4] pers;
  vector[4] utility;
  
  real ef_chosen;  
  real PEval;
  real PEfreq;
  vector[4] PEfreq_fic;
  real K_tr;
  
  // Priors
  for(s in 1:S){
    mu_p[s,:]   ~ normal(0, 1);
    mu_p[s,:]   ~ normal(0, 1);
    // Subject-level parameters
    Arew_pr[s,:]  ~ normal(0, 1.0);
    Apun_pr[s,:]  ~ normal(0, 1.0);
    K_pr[s,:]     ~ normal(0, 1.0);
    betaF_pr[s,:] ~ normal(0, 1.0);
    betaP_pr[s,:] ~ normal(0, 1.0);
  }
  sigma_Arew  ~ normal(0, 0.2);
  sigma_Apun  ~ normal(0, 0.2);
  sigma_K     ~ normal(0, 0.2);
  sigma_betaF ~ cauchy(0, 1);
  sigma_betaP ~ cauchy(0, 1);
  
  
  for (i in 1:N) {         // Loop through individual participants
    for (s in 1:S) {       // Loop though sessions for participant i
      if (Tsubj[i,s] > 0) {    // If we have data for participant i on session s, run through RL algorithm
        
        // Initialize starting values
        K_tr = pow(3, K[i,s]) - 1;
        ev = rep_vector(0,4);
        ef = rep_vector(0,4);
        pers = rep_vector(0,4);
        utility = rep_vector(0,4);
        
        for (t in 1:Tsubj[i,s]) { // Run through RL algorithm trial-by-trial
          
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
            
            // Update perseverance
            pers[card[i,t,s]] = 1;
          }
          // perseverance decay
          pers = pers / (1 + K_tr);
            
          // Calculate expected value of card
          utility = ev + ef * betaF[i,s] + pers * betaP[i,s];
        }
      }
    }
  }
}

generated quantities {
  // Hyper(group)-parameters - these are 5 (number of parameters) x S (number of sessions) matrix of mus & sigmas, respectively, for each parameter
  vector<lower=0,upper=1>[S] mu_Arew;
  vector<lower=0,upper=1>[S] mu_Apun;
  vector<lower=0,upper=5>[S] mu_K;
  vector[S] mu_betaF;
  vector[S] mu_betaP;
  vector[N] log_lik;

  // For posterior predictive check
  array[N,T,S] real choice_pred;
  
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
    mu_K[s] = Phi_approx_group_mean_rng(mu_p[s, 3], sigma_K[s], 10000) * 5; 
    mu_betaF[s] = mu_p[s, 4];
    mu_betaP[s] = mu_p[s, 5];
  }
  
  { // local section, this saves time and space
    // Declare variables to calculate utility after each trial: These 4 (number of cards) x 2 (playing vs. not playing) matrices
    vector[4] ef;
    vector[4] ev;
    vector[4] pers;
    vector[4] utility;
    
    real ef_chosen;
    real PEval;
    real PEfreq;
    vector[4] PEfreq_fic;
    real K_tr;
  
    for (i in 1:N) {         // Loop through individual participants
      log_lik[i] = 0;        // Initialize log_lik
      
      for (s in 1:S) {       // Loop though sessions for participant i
        if (Tsubj[i,s] > 0) {    // If we have data for participant i on session s, run through RL algorithm
          
          // Initialize starting values
          K_tr = pow(3, K[i,s]) - 1;
          ev = rep_vector(0,4);
          ef = rep_vector(0,4);
          pers = rep_vector(0,4);
          utility = rep_vector(0,4);
          
          for (t in 1:Tsubj[i,s]) { // Run through RL algorithm trial-by-trial
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
              
              // Update perseverance
              pers[card[i,t,s]] = 1;
            }
            // perseverance decay
            pers = pers / (1 + K_tr);
              
            // Calculate expected value of card
            utility = ev + ef * betaF[i,s] + pers * betaP[i,s];
          }
        }
      }
    }
  }
}

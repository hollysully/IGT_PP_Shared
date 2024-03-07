functions {
  real Phi_approx_group_mean_rng(real mu, real sigma, int n_samples) {
    vector[n_samples] par;
    for (i in 1:n_samples) {
      par[i] = Phi_approx(normal_rng(mu, sigma));
    }
    return mean(par);
  }
}


data {
  int<lower=1> N;          // Number of participants
  int<lower=1> G;          // Numer of groups for group-specific sigmas
  int<lower=1> T;          // Total possile number of trials
  int<lower=1> D;          // Number of person-level predictors
  array[N,T] int card;     // Cards presented on each trial
  array[N] int Tsubj;      // Total number of trials presented to each subject on each session
  array[N] int group;      // Groups to designate which sigma
  array[N,T] int choice;   // Choices on each trial
  array[N,T] real outcome; // Outcomes received on each trial
  array[N,T] real sign;    // Signs of the outcome received on each trial
  array[N,D] real X;       // person-level predictors
}


parameters {
// Declare parameters
  // Hyper(group)-parameters
  vector<lower=0>[G] sigma_Arew;
  vector<lower=0>[G] sigma_Apun;
  vector<lower=0>[G] sigma_K;
  vector<lower=0>[G] sigma_betaF;
  vector<lower=0>[G] sigma_betaP;
  
  vector[D] beta_Arew;
  vector[D] beta_Apun;
  vector[D] beta_K;
  vector[D] beta_betaF;
  vector[D] beta_betaP;

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
  
  // Calculate & transform Arew, Apun, & K to use in RL algorithm
  for(i in 1:N){  // Loop over subjects
  Arew[i] = Phi_approx(dot_product(beta_Arew, to_vector(X[i,])) + sigma_Arew[group[i]]*Arew_pr[i]);
  Apun[i] = Phi_approx(dot_product(beta_Apun, to_vector(X[i,])) + sigma_Apun[group[i]]*Apun_pr[i]);
  K[i]    = Phi_approx(dot_product(beta_K, to_vector(X[i,])) + sigma_K[group[i]]*K_pr[i]) * 5;
  
  // Calculate betaF & betaP to use in RL algorithm
  betaF = dot_product(beta_betaF, to_vector(X[i,])) + sigma_betaF[group[i]]*betaF_pr[i];
  betaP = dot_product(beta_betaP, to_vector(X[i,])) + sigma_betaP[group[i]]*betaP_pr[i];
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
  // Hyperparameters for correlations
  R_chol_Arew  ~ lkj_corr_cholesky(1);
  R_chol_Apun  ~ lkj_corr_cholesky(1);
  R_chol_K     ~ lkj_corr_cholesky(1);
  R_chol_betaF ~ lkj_corr_cholesky(1);
  R_chol_betaP ~ lkj_corr_cholesky(1);
  
  // Hyperparameters for RL learning algorithm
  beta_Arew   ~ normal(0, 1);
  beta_Apun   ~ normal(0, 1);
  beta_K      ~ normal(0, 1);
  beta_betaF  ~ normal(0, 1);
  beta_betaP  ~ normal(0, 1);
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
    if (Tsubj[i] > 0) {    // If we have data for participant i on session s, run through RL algorithm
      
      // Initialize starting values
      K_tr = pow(3, K[i]) - 1;
      ev = rep_vector(0,4);
      ef = rep_vector(0,4);
      pers = rep_vector(0,4);
      utility = rep_vector(0,4);
      
      for (t in 1:Tsubj[i]) { // Run through RL algorithm trial-by-trial
        
        // Likelihood - predict choice as a function of utility
        choice[i,t] ~ categorical_logit(to_vector({utility[card[i,t]], 0}));
        // this could equivalently be specified as: 
        // choice[i,t] ~ bernoulli_logit(utility[card[i,t]]);
        // if choice = 0 for pass and 1 for play. Keeping categorical_*
        // is consistent with other models
        
        if(choice[i,t]==1){
          // After choice, calculate prediction error
          PEval      = outcome[i,t] - ev[card[i,t]];     // Value prediction error
          PEfreq     = sign[i,t] - ef[card[i,t]];        // Win-frequency prediction error
          PEfreq_fic = -sign[i,t]/3 - ef;
          ef_chosen  = ef[card[i,t]];
          
          if (outcome[i,t] >= 0) {  // If participant DID NOT lose
            // Update expected win-frequency of what participant DID NOT chose to do
            ef = ef + Apun[i] * PEfreq_fic;
            // Update what participant chose
            ef[card[i,t]] = ef_chosen + Arew[i] * PEfreq;
            ev[card[i,t]] = ev[card[i,t]] + Arew[i] * PEval;
          } else { // If participant DID lose
            // Update expected win-frequency of what participant DID NOT choose to do
            ef = ef + Arew[i] * PEfreq_fic;
            // Update what participant chose
            ef[card[i,t]] = ef_chosen + Apun[i] * PEfreq;
            ev[card[i,t]] = ev[card[i,t]] + Apun[i] * PEval;
          }
          
          // Update perseverance
          pers[card[i,t]] = 1;
        }
        // perseverance decay
        pers = pers / (1 + K_tr);
          
        // Calculate expected value of card
        utility = ev + ef * betaF[i] + pers * betaP[i];
      }
    }
  }
}

generated quantities {
  // Hyper(group)-parameters - these are 5 (number of parameters) x S (number of sessions) matrix of mus & sigmas, respectively, for each parameter
  matrix<lower=0,upper=1>[D] mu_Arew;
  matrix<lower=0,upper=1>[D]  mu_Apun;
  matrix<lower=0,upper=5>[D]  mu_K;
  matrix[D]  mu_betaF;
  matrix[D]  mu_betaP;
  
  vector[N] log_lik;

  // For posterior predictive check
  array[N,T] real choice_pred;
  
  // Reconstruct correlation matrix from cholesky factor
    // Note that we're multipling the cholesky factor by its transpose which gives us the correlation matrix
  R_Arew  = R_chol_Arew * R_chol_Arew'; //Phi_approx_corr_rng(mu_p[,1], sigma_Arew, R_chol_Arew * R_chol_Arew', 10000);
  R_Apun  = R_chol_Apun * R_chol_Apun'; //Phi_approx_corr_rng(mu_p[,2], sigma_Apun, R_chol_Apun * R_chol_Apun', 10000);
  R_K     = R_chol_K * R_chol_K'; //Phi_approx_corr_rng(mu_p[,3], sigma_K, R_chol_K * R_chol_K', 10000);
  R_betaF = R_chol_betaF * R_chol_betaF';
  R_betaP = R_chol_betaP * R_chol_betaP';
  
  // Set all posterior predictions to -1 (avoids NULL values)
  for (i in 1:N) {
    for (t in 1:T) {
      choice_pred[i,t] = -1;
    }
  }
  
  // // Compute group-level means
    mu_Arew[1] = beta_Arew[1];
    mu_Apun[1] = beta_Apun[1];
    mu_K[1] = beta_K[1];
    mu_betaF[1] = beta_betaF[1];
    mu_betaP[1] = beta_betaP[1];
    
    for(d in 2:D){
      mu_Arew[d] = mu_Arew[1] + beta_Arew[d];
      mu_Arew[d] = Phi_approx_group_mean_rng(mu_Arew[d], sigma_Arew[2], 10000);
      mu_Apun[d] = mu_Apun[1] + beta_Apun[d];
      mu_Apun[d] = Phi_approx_group_mean_rng(mu_Apun[d], sigma_Apun[2], 10000);
      mu_K[d]    = mu_K[1] + beta_K[d];
      mu_K[d]    = Phi_approx_group_mean_rng(mu_K[d], sigma_K[2], 10000);
      mu_betaF[d] = mu_betaF[1] + beta_betaF[d];
      mu_betaP[d] = mu_betaP[1] + beta_betaP[d];
    }
    mu_Arew[1] = Phi_approx_group_mean_rng(mu_Arew[1], sigma_Arew[1], 10000);
    mu_Apun[1] = Phi_approx_group_mean_rng(mu_Apun[1], sigma_Apun[1], 10000);
    mu_K[1] = Phi_approx_group_mean_rng(mu_K[1], sigma_K[1], 10000);
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
      
      if (Tsubj[i] > 0) {    // If we have data for participant i on session s, run through RL algorithm
        
        // Initialize starting values
        K_tr = pow(3, K[i]) - 1;
        ev = rep_vector(0,4);
        ef = rep_vector(0,4);
        pers = rep_vector(0,4);
        utility = rep_vector(0,4);
        
        for (t in 1:Tsubj[i]) { // Run through RL algorithm trial-by-trial
          // softmax choice
          log_lik[i] += categorical_logit_lpmf(choice[i,t]|to_vector({utility[card[i,t]], 0}));
          
          // Likelihood - predict choice as a function of utility
          choice_pred[i,t] = categorical_rng(softmax(to_vector({utility[card[i,t]], 0})));
          
          if(choice[i,t]==1) {
            // After choice, calculate prediction error
            PEval      = outcome[i,t] - ev[card[i,t]];     // Value prediction error
            PEfreq     = sign[i,t] - ef[card[i,t]];        // Win-frequency prediction error
            PEfreq_fic = -sign[i,t]/3 - ef;
            ef_chosen = ef[card[i,t]];
            
            if (outcome[i,t] >= 0) {  // If participant DID NOT lose
              // Update expected win-frequency of what participant DID NOT chose to do
              ef = ef + Apun[i] * PEfreq_fic;
              // Update what participant chose
              ef[card[i,t]] = ef_chosen + Arew[i] * PEfreq;
              ev[card[i,t]] = ev[card[i,t]] + Arew[i] * PEval;
            } else { // If participant DID lose
              // Update expected win-frequency of what participant DID NOT choose to do
              ef = ef + Arew[i] * PEfreq_fic;
              // Update what participant chose
              ef[card[i,t]] = ef_chosen + Apun[i] * PEfreq;
              ev[card[i,t]] = ev[card[i,t]] + Apun[i] * PEval;
            }
            
            // Update perseverance
            pers[card[i,t]] = 1;
          }
          // perseverance decay
          pers = pers / (1 + K_tr);
            
          // Calculate expected value of card
          utility = ev + ef * betaF[i] + pers * betaP[i];
        }
      }
    }
  }
}

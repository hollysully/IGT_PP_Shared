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
  int<lower=1> T;          // Total possile number of trials
  array[N, T] int card;    // Cards presented on each trial
  array[N] int Tsubj;      // Total number of trials presented to each subject on each session
  array[N,T] int choice;   // Choices on each trial
  array[N,T] real outcome; // Outcomes received on each trial
  array[N,T] real sign;    // Signs of the outcome received on each trial
}


parameters {
// Declare parameters
  // Hyper(group)-parameters
  vector[4] mu_p;   // Untransformed parameters
  real<lower=0> sigma_Arew;
  real<lower=0> sigma_Apun;
  real<lower=0> sigma_betaF;
  real<lower=0> sigma_betaB;

  // Subject-level z-scored deviations
  vector[N] Arew_pr;
  vector[N] Apun_pr;
  vector[N] betaF_pr;
  vector[N] betaB_pr;
}

transformed parameters {
  // Transformed subject-level parameters
  vector<lower=0,upper=1>[N] Arew;
  vector<lower=0,upper=1>[N] Apun;
  vector[N] betaF;
  vector[N] betaB;
  
  // Subject-level deviations weighted by standard deviation of each parameter (for shrinkage)
  vector[N] Arew_tilde;
  vector[N] Apun_tilde;
  vector[N] betaF_tilde;
  vector[N] betaB_tilde;
  
  // Calculate weighted subject-level deviations
  Arew_tilde  = sigma_Arew * Arew_pr;
  Apun_tilde  = sigma_Apun * Apun_pr;
  betaF_tilde = sigma_betaF * betaF_pr;
  betaB_tilde = sigma_betaB * betaB_pr;
  
  // Calculate person-level parameters
    for(i in 1:N){  // Loop over subjects
      Arew[i] = Phi_approx(mu_p[1] + Arew_tilde[i]);
      Apun[i] = Phi_approx(mu_p[2] + Apun_tilde[i]);
    }
  betaF = mu_p[3] + betaF_tilde;
  betaB = mu_p[4] + betaB_tilde;
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
  // Hyperparameters
  mu_p        ~ normal(0, 1);
  sigma_Arew  ~ normal(0, 0.2);
  sigma_Apun  ~ normal(0, 0.2);
  sigma_betaF ~ cauchy(0, 1);
  sigma_betaB ~ cauchy(0, 1);
  
  // Subject-level parameters
  Arew_pr  ~ normal(0, 1.0);
  Apun_pr  ~ normal(0, 1.0);
  betaF_pr ~ normal(0, 1.0);
  betaB_pr ~ normal(0, 1.0);
  
  for (i in 1:N) {         // Loop through individual participants
    if (Tsubj[i] > 0) {    // If we have data for participant i on session s, run through RL algorithm
      
      // Initialize starting values
      ev = rep_vector(0,4);
      ef = rep_vector(0,4);
      utility = rep_vector(0,4);
      
      for (t in 1:Tsubj[i]) { // Run through RL algorithm trial-by-trial
        // Calculate expected value of card
        utility = ev + ef * betaF[i] + betaB[i];
        
        // Likelihood - predict choice as a function of utility
        choice[i,t] ~ categorical_logit(to_vector({utility[card[i,t]], 0}));
        
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
        }
      }
    }
  }
}

generated quantities {
  // Hyper(group)-parameters - these are 5 (number of parameters) x S (number of sessions) matrix of mus & sigmas, respectively, for each parameter
  real<lower=0,upper=1> mu_Arew;
  real<lower=0,upper=1> mu_Apun;
  real mu_betaF;
  real mu_betaB;
  array[N] real log_lik;

  // For posterior predictive check
  array[N,T] real choice_pred;
  
  // Set all posterior predictions to -1 (avoids NULL values)
  for (i in 1:N) {
    for (t in 1:T) {
      choice_pred[i,t] = -1;
    }
  }
  
  // Compute group-level means
    mu_Arew = Phi_approx_group_mean_rng(mu_p[1], sigma_Arew, 10000);
    mu_Apun = Phi_approx_group_mean_rng(mu_p[2], sigma_Apun, 10000);
    mu_betaF = mu_p[3];
    mu_betaB = mu_p[4];
  
  { // local section, this saves time and space
    vector[4] ef;
    vector[4] ev;
    vector[4] utility;
    
    real ef_chosen;
    real PEval;
    real PEfreq;
    vector[4] PEfreq_fic;
  
    for (i in 1:N) {         // Loop through individual participants
      log_lik[i] = 0;        // Initialize log_lik
      
      if (Tsubj[i] > 0) {    // If we have data for participant i on session s, run through RL algorithm
        
        // Initialize starting values
        ev = rep_vector(0,4);
        ef = rep_vector(0,4);
        utility = rep_vector(0,4);
        
        for (t in 1:Tsubj[i]) { // Run through RL algorithm trial-by-trial
          // Calculate expected value of card
          utility = ev + ef * betaF[i] + betaB[i];
          
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
          }
        }
      }
    }
  }
}

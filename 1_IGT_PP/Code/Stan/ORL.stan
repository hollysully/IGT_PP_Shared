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
  int<lower=1> N;                    // Number of participants
  int<lower=1> T;                    // Total possile number of trials
  int card[N,T];                     // Cards presented on each trial
  int Tsubj[N];                      // Total number of trials presented to each subject on each session
  int choice[N,T];                   // Choices on each trial
  real outcome[N,T];                 // Outcomes received on each trial
  real sign[N,T];                    // Signs of the outcome received on each trial
}


parameters {
// Declare parameters
  // Hyper(group)-parameters
  vector[5] mu_p;
  real<lower=0> sigma_Arew;
  real<lower=0> sigma_Apun;
  real<lower=0> sigma_K;
  real<lower=0> sigma_betaF;
  real<lower=0> sigma_betaP;

  // Subject-level "raw" parameters
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
    Arew[i] = Phi_approx(mu_p[1] + sigma_Arew * Arew_pr[i]);
    Apun[i] = Phi_approx(mu_p[2] + sigma_Apun * Apun_pr[i]);
    K[i]    = Phi_approx(mu_p[3] + sigma_K * K_pr[i]) * 5;
    betaF[i] = mu_p[4] + sigma_betaF * betaF_pr[i];
    betaP[i] = mu_p[5] + sigma_betaP * betaP_pr[i];
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
  // Hyperparameters for RL learning algorithm
  mu_p   ~ normal(0, 1);
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
    if (Tsubj[i] > 0) {    // If we have data for participant i, run through RL algorithm
      
      // Initialize starting values
      K_tr = pow(3, K[i]) - 1;
      ev = rep_vector(0,4);
      ef = rep_vector(0,4);
      pers = rep_vector(0,4);
      utility = rep_vector(0,4);
      
      for (t in 1:Tsubj[i]) { // Run through RL algorithm trial-by-trial
        
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
  real<lower=0,upper=1> mu_Arew;
  real<lower=0,upper=1> mu_Apun;
  real<lower=0,upper=5> mu_K;
  real mu_betaF;
  real mu_betaP;
  real log_lik[N];

  // For posterior predictive check
  real choice_pred[N,T];
  vector[4] utility_pred[N,T];
  
  // Set all posterior predictions to -1 (avoids NULL values)
  for (i in 1:N) {
    for (t in 1:T) {
      choice_pred[i,t] = -1;
      utility_pred[i,t] = rep_vector(0,4);
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
      
      if (Tsubj[i] > 0) {    // If we have data for participant i, run through RL algorithm
        
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
          utility_pred[i,t] = utility;
        }
      }
    }
  }
}

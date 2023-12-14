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
  int<lower=1> G;          // Total number of groups
  array[N] int group;      // Group that participant is assigned to (must be 1 or 2)
  array[N,T] int card;     // Cards presented on each trial
  array[N] int Tsubj;      // Total number of trials presented to each subject on each session
  array[N,T] int choice;   // Choices on each trial
  array[N,T] real outcome; // Outcomes received on each trial
  array[N,T] real sign;    // Signs of the outcome received on each trial
}


parameters {
// Declare parameters
  // Hyper(group)-parameters
  matrix[5,G] mu_p;
  vector<lower=0>[G] sigma_Arew;
  vector<lower=0>[G] sigma_Apun;
  vector<lower=0>[G] sigma_K;
  vector<lower=0>[G] sigma_betaF;
  vector<lower=0>[G] sigma_betaP;

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
    Arew[i] = Phi_approx(mu_p[1,group[i]] + sigma_Arew[group[i]] * Arew_pr[i]);
    Apun[i] = Phi_approx(mu_p[2,group[i]] + sigma_Apun[group[i]] * Apun_pr[i]);
    K[i]    = Phi_approx(mu_p[3,group[i]] + sigma_K[group[i]] * K_pr[i]) * 5;
    betaF[i] = mu_p[4,group[i]] + sigma_betaF[group[i]] * betaF_pr[i];
    betaP[i] = mu_p[5,group[i]] + sigma_betaP[group[i]] * betaP_pr[i];
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
  for(g in 1:G){
    mu_p[:,g]~ normal(0, 1);
  }
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
  vector<lower=0,upper=1>[G] mu_Arew;
  vector<lower=0,upper=1>[G] mu_Apun;
  vector<lower=0,upper=5>[G] mu_K;
  vector[G] mu_betaF;
  vector[G] mu_betaP;
  vector[N] log_lik;

  // For posterior predictive check
  array[N,T] real choice_pred;
  
  // Set all posterior predictions to -1 (avoids NULL values)
  for (i in 1:N) {
    for (t in 1:T) {
      choice_pred[i,t] = -1;
    }
  }
  
  // Compute group-level means
  for(g in 1:G){
    mu_Arew[g] = Phi_approx_group_mean_rng(mu_p[1,g], sigma_Arew[g], 10000);
    mu_Apun[g] = Phi_approx_group_mean_rng(mu_p[2,g], sigma_Apun[g], 10000);
    mu_K[g] = Phi_approx_group_mean_rng(mu_p[3,g], sigma_K[g], 10000) * 5; 
    mu_betaF[g] = mu_p[4,g];
    mu_betaP[g] = mu_p[5,g];
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
        }
      }
    }
  }
}

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
  int<lower=1> N;                // Number of participants
  int<lower=1> T;                // Total possile number of trials
  int Fp;                        // Number of focal predictors
  int Cp;                        // Number of covariate predictors
  array[N,T] int card;           // Cards presented on each trial
  array[N] int Tsubj;            // Total number of trials presented to each subject on each session
  array[N,T] int choice;         // Choices on each trial
  array[N,T] real outcome;       // Outcomes received on each trial
  array[N,T] real sign;          // Signs of the outcome received on each trial
  array[N] int focal_predictor;  // Indicator to reference which Xs are focal predictors 
  array[N,1+Fp+Cp] real X;       // Intercept (i.e., all 1s), focal predictors, and covariates
}


parameters {
// Declare parameters
  // Hyper(group)-parameters
  vector<lower=0>[1+Fp] sigma_Arew;  // group-level intercept + focal predictor
  vector<lower=0>[1+Fp] sigma_Apun;  // group-level intercept + focal predictor
  vector<lower=0>[1+Fp] sigma_betaF; // group-level intercept + focal predictor
  vector<lower=0>[1+Fp] sigma_betaB; // group-level intercept + focal predictor
  
  vector[1+Fp+Cp] beta_Arew;  // group-level intercept + focal predictor + covariates
  vector[1+Fp+Cp] beta_Apun;  // group-level intercept + focal predictor + covariates
  vector[1+Fp+Cp] beta_betaF; // group-level intercept + focal predictor + covariates
  vector[1+Fp+Cp] beta_betaB; // group-level intercept + focal predictor + covariates

  // Subject-level "raw" parameters
  vector[N] Arew_pr;  
  vector[N] Apun_pr;  
  vector[N] betaF_pr;
  vector[N] betaB_pr;
}

transformed parameters {
// Declare transformed parameters
  // Parameters to use in the reinforcement-learning algorithm
  vector<lower=0,upper=1>[N] Arew; 
  vector<lower=0,upper=1>[N] Apun;
  vector[N] betaF;
  vector[N] betaB;
  
  // Calculate person-level parameters
  for(i in 1:N){  // Loop over subjects
    Arew[i]  = Phi_approx(dot_product(beta_Arew, to_vector(X[i,])) + Arew_pr[i]*sigma_Arew[focal_predictor[i]]);
    Apun[i]  = Phi_approx(dot_product(beta_Apun, to_vector(X[i,])) + Apun_pr[i]*sigma_Apun[focal_predictor[i]]);
    betaF[i] = dot_product(beta_betaF, to_vector(X[i,])) + betaF_pr[i]*sigma_betaF[focal_predictor[i]];
    betaB[i] = dot_product(beta_betaB, to_vector(X[i,])) + betaB_pr[i]*sigma_betaB[focal_predictor[i]];
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
  // Hyperparameters for RL learning algorithm
  beta_Arew ~ normal(0, 1);
  beta_Apun ~ normal(0, 1);
  beta_betaF ~ normal(0, 1);
  beta_betaB ~ normal(0, 1);
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
  // Group-level parameters - Intercept + focal predictors
  vector<lower=0,upper=1>[1+Fp] mu_Arew;
  vector<lower=0,upper=1>[1+Fp]  mu_Apun;
  vector[1+Fp]  mu_betaF;
  vector[1+Fp]  mu_betaB;
  
  vector[N] log_lik;

  // For posterior predictive check
  array[N,T] real choice_pred;
  
  // Set all posterior predictions to -1 (avoids NULL values)
  for (i in 1:N) {
    for (t in 1:T) {
      choice_pred[i,t] = -1;
    }
  }
  
  // Calculate reference group (i.e., intercept) group-level means
  mu_Arew[1]  = Phi_approx_group_mean_rng(beta_Arew[1], sigma_Arew[1], 10000);
  mu_Apun[1]  = Phi_approx_group_mean_rng(beta_Apun[1], sigma_Apun[1], 10000);
  mu_betaF[1] = beta_betaF[1];
  mu_betaB[1] = beta_betaB[1];
  
  if(Fp>0){ // If there are focal predictors, then calculate their group-level means
    for(p in 2:(1+Fp)){
      mu_Arew[p]  = Phi_approx_group_mean_rng(beta_Arew[1] + beta_Arew[p], sigma_Arew[p], 10000);
      mu_Apun[p]  = Phi_approx_group_mean_rng(beta_Apun[1] + beta_Apun[p], sigma_Apun[p], 10000);
      mu_betaF[p] = beta_betaF[1] + beta_betaF[p];
      mu_betaB[p] = beta_betaB[1] + beta_betaB[p];
    }
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

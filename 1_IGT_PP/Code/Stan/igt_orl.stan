

// use an R script to preprocess the data into stan-ready list
data {
  int<lower=1> N; //numSubjs; the number of subjects
  int<lower=1> T; //maxTrials
  int<lower=1, upper=T> Tsubj[N];  // for each sub, a list of the trial nums, 1:T
  int stim[N, T];  // a value of 1-4, representing the deck presented on a given trial
  real rewlos[N, T]; // absolute value of outcome, 0 for pass & NA outcomes
  real Srewlos[N, T]; // the sign of the outcome: values of 1, 0, or -1
  int ydata[N, T]; // a value of 1 for play or 2 for pass
}
transformed data {
  real sign[N, T];
  
  for (i in 1:N) {
    for (t in 1:T) {
      if (Srewlos[i,t]>0) {
        sign[i,t] = 1;
      } else if (Srewlos[i,t]==0) {
        sign[i,t] = 0;
      } else {
        sign[i,t] = -1;
      }
    }
  }
}
parameters {
// Declare all parameters as vectors for vectorizing
  // Hyper(group)-parameters  
  vector[5] mu_p;  
  vector<lower=0>[5] sigma;
    
  // Subject-level raw parameters (for Matt trick)
  vector[N] Arew_pr;
  vector[N] Apun_pr;
  vector[N] K_pr;
  vector[N] betaF_pr;
  vector[N] betaP_pr;
}
transformed parameters {
  // Transform subject-level raw parameters
  vector<lower=0, upper=1>[N] Arew;
  vector<lower=0, upper=1>[N] Apun;
  vector<lower=0, upper=5>[N] K;
  vector[N]                   betaF;
  vector[N]                   betaP;

  for (i in 1:N) {
    Arew[i]  = Phi_approx( mu_p[1] + sigma[1] * Arew_pr[i] );
    Apun[i]  = Phi_approx( mu_p[2] + sigma[2] * Apun_pr[i] );
    K[i] = Phi_approx(mu_p[3] + sigma[3] * K_pr[i]) * 5;
  }
  betaF = mu_p[4] + sigma[4] * betaF_pr;
  betaP = mu_p[5] + sigma[5] * betaP_pr;
}
model {
  // Hyperparameters
  mu_p  ~ normal(0, 1);
  sigma[1:3] ~ normal(0, 0.2);
  sigma[4:5] ~ cauchy(0, 1);
  
  // individual parameters
  Arew_pr  ~ normal(0, 1.0);
  Apun_pr  ~ normal(0, 1.0);
  K_pr     ~ normal(0, 1.0);
  betaF_pr ~ normal(0, 1.0);
  betaP_pr ~ normal(0, 1.0);

  for (i in 1:N) {
    // Define values
    matrix[4,2] ef;
    matrix[4,2] ev;
    matrix[4,2] pers;
    matrix[4,2] util;
    
    real util_tr[2];

    real PEval;
    real PEfreq;
    real PEfreq_fic;
    real PEval_fic;
    real K_tr;
    
    // Initialize values
    K_tr = pow(3, K[i]) - 1;
    for (j in 1:2) {
      ef[:,j] = rep_vector(0,4);
      ev[:,j] = rep_vector(0,4);
      util[:,j] = rep_vector(0,4);
      pers[:,j] = rep_vector(0,4);
    }

    for (t in 1:Tsubj[i]) {
      // softmax choice
      ydata[i, t] ~ categorical_logit(to_vector(util[stim[i,t],:]));
      
      // Prediction error
      PEval  = rewlos[i,t] - ev[stim[i,t], ydata[i,t]];
      PEfreq = sign[i,t] - ef[stim[i,t], ydata[i,t]];
      PEfreq_fic = -sign[i,t]/1 - ef[stim[i,t], (3-ydata[i,t])];

      if (rewlos[i,t] >= 0) {
        // Update foregone deck
        ef[stim[i,t], (3-ydata[i,t])] = ef[stim[i,t], (3-ydata[i,t])] + Apun[i] * PEfreq_fic;
        // Update chosen deck
        ef[stim[i,t], ydata[i,t]] = ef[stim[i,t], ydata[i,t]] + Arew[i] * PEfreq;
        ev[stim[i,t], ydata[i,t]] = ev[stim[i,t], ydata[i,t]] + Arew[i] * PEval;
      } else {
        // Update ev for all decks 
        ef[stim[i,t], (3-ydata[i,t])] = ef[stim[i,t], (3-ydata[i,t])] + Arew[i] * PEfreq_fic;
        // Update chosendeck with stored value
        ef[stim[i,t], ydata[i,t]] = ef[stim[i,t], ydata[i,t]] + Apun[i] * PEfreq;
        ev[stim[i,t], ydata[i,t]] = ev[stim[i,t], ydata[i,t]] + Apun[i] * PEval;
      }
      
      // Perseverance updating
      pers[stim[i,t], ydata[i,t]] = 1;   
      pers = pers / (1 + K_tr); 
      
      // Utility of expected value and perseverance
      util = ev + ef * betaF[i] + pers * betaP[i];
    }
  }
}

generated quantities {
  // For group level parameters
  real<lower=0,upper=1> mu_Arew;
  real<lower=0,upper=1> mu_Apun;
  real<lower=0,upper=5> mu_K;
  real                  mu_betaF;
  real                  mu_betaP;
  real log_lik[N];
  
  // For log likelihood calculation
  int y_pred[N,T];

  mu_Arew   = Phi_approx(mu_p[1]);
  mu_Apun   = Phi_approx(mu_p[2]);
  mu_K      = Phi_approx(mu_p[3]) * 5;
  mu_betaF  = mu_p[4];
  mu_betaP  = mu_p[5];
  
  { // local section, this saves time and space
    for (i in 1:N) {
      log_lik[i] = 0;        // Initialize log_lik
      
      // Define values
      matrix[4,2] ef;
      matrix[4,2] ev;
      matrix[4,2] pers;
      matrix[4,2] util;
      
      real util_tr[2];
  
      real PEval;
      real PEfreq;
      real PEfreq_fic;
      real PEval_fic;
      real K_tr;
      
      // Initialize values
      K_tr = pow(3, K[i]) - 1;
      for (j in 1:2) {
        ef[:,j] = rep_vector(0,4);
        ev[:,j] = rep_vector(0,4);
        util[:,j] = rep_vector(0,4);
        pers[:,j] = rep_vector(0,4);
      }
      
      for (t in 1:Tsubj[i]) {
        // softmax choice
        log_lik[i] += categorical_logit_lpmf(ydata[i,t]|to_vector(util[stim[i,t], :]));
            
        // softmax choice
        y_pred[i,t] = categorical_rng(softmax(to_vector(util[stim[i,t],:])));
        
        // Prediction error
        PEval  = rewlos[i,t] - ev[stim[i,t], ydata[i,t]];
        PEfreq = sign[i,t] - ef[stim[i,t], ydata[i,t]];
        PEfreq_fic = -sign[i,t]/1 - ef[stim[i,t], (3-ydata[i,t])];
  
        if (rewlos[i,t] >= 0) {
          // Update foregone deck
          ef[stim[i,t], (3-ydata[i,t])] = ef[stim[i,t], (3-ydata[i,t])] + Apun[i] * PEfreq_fic;
          // Update chosen deck
          ef[stim[i,t], ydata[i,t]] = ef[stim[i,t], ydata[i,t]] + Arew[i] * PEfreq;
          ev[stim[i,t], ydata[i,t]] = ev[stim[i,t], ydata[i,t]] + Arew[i] * PEval;
        } else {
          // Update ev for all decks 
          ef[stim[i,t], (3-ydata[i,t])] = ef[stim[i,t], (3-ydata[i,t])] + Arew[i] * PEfreq_fic;
          // Update chosendeck with stored value
          ef[stim[i,t], ydata[i,t]] = ef[stim[i,t], ydata[i,t]] + Apun[i] * PEfreq;
          ev[stim[i,t], ydata[i,t]] = ev[stim[i,t], ydata[i,t]] + Apun[i] * PEval;
        }
        
        // Perseverance updating
        pers[stim[i,t], ydata[i,t]] = 1;   
        pers = pers / (1 + K_tr); 
        
        // Utility of expected value and perseverance
        util = ev + ef * betaF[i] + pers * betaP[i];
      }
    }
  }  
}

data {
  // THIS SHOULD BE ADJUSTED TO THE DATA FOR THE THEORETICAL MODEL OF INTEREST
  int<lower=1> N;           // number of participants
  int<lower=1> T;           // total possile number of trials
  int<lower=1> S;           // number of sessions
  
  array[N,T,S] int choice;  // choices made by participants
  array[N,T,S] int outcome; // outcomes received by participants after making choice
}


parameters {
  // THIS CAN BE USED FOR ANY SINGLE-PARAMETER MODEL
  real gamma00;    // group-level intercepts
  real gamma10;    // group-level slopes
  real<lower=0> R; // residual error
  
  cholesky_factor_corr[2] R_chol; // Cholesky correlation matrix for intercept-slope correlation
  vector<lower=0>[2] sigma_U;     // SD of intercepts (sigma_U[1]) and slopes (sigma_U[2])
  matrix[2,N] z_U;                // uncorrelated standardized person-specific deviations from group-level intercepts (z_U[1,]) and slopes (z_U[2,])
}


transformed parameters {
  // THIS CAN BE USED FOR ANY SINGLE-PARAMETER THEORETICAL MODEL
  vector[N] beta0;   // person-specific intercepts
  vector[N] beta1;   // person-specific slopes
  matrix[2,N] U;     // correlated person-specific deviations from group-level intercepts (z_U[1,]) and slopes (z_U[2,])
  matrix[S,N] theta; // unbounded theoretical model-parameter for each session
  
  U = diag_pre_multiply(sigma_U, R_chol) * z_U; // calculate person-specific deviations
  
  for(i in 1:N){ // loop through persons
    beta0[i] = gamma00 + U[1,i]; // calculate person-specific intercepts
    beta1[i] = gamma10 + U[2,i]; // calculate person-specific slopes
    
    for(s in 1:S){ // loop through sessions
      theta[s,i] = beta0[i] + beta1[i]*(s-1) + R; // combine intercepts & slopes into unbounded theoretical model-parameter
    }
  }
}
  

model {
  
  // PRIORS: THESE CAN BE USED FOR ANY SINGLE-PARAMETER THEORETICAL MODEL
  gamma00 ~ normal(0,1);    // group-level intercepts
  gamma10 ~ normal(0,0.05); // group-level slopes
  R       ~ normal(0,0.05); // residual error
  
  R_chol  ~ lkj_corr_cholesky(1);       // cholesky correlation
  to_vector(sigma_U)  ~ normal(0,0.05); // SD of intercepts and slopes
  to_vector(z_U[1,:]) ~ normal(0,1);    // uncorrelated standardized person-specific deviations
  to_vector(z_U[2,:]) ~ normal(0,1);    // uncorrelated standardized person-specific deviations
  
  // LIKELIHOOD: THIS SHOULD BE ADJUSTED FOR THE THEORETICAL MODEL OF INTEREST
  vector[2] utility; // declare utility variable
  matrix[S,N] A;     // declare bounded theoretical model-parameter
  
  for(s in 1:S){ // loop through sessions
    for(i in 1:N) { // loop through persons
      A[s,i] = Phi_approx(theta[s,i]); // transform theoretical model parameter
      utility = rep_vector(0,2); // set values of utility vector to 0 for starting the session
      
      for(t in 1:T) { // loop through trials
        choice[i,t,s] ~ categorical_logit(utility); // predict choice based on utility
        
        // RW learning rule
        utility[choice[i,t,s]] = utility[choice[i,t,s]] + A[s,i]*(outcome[i,t,s] - utility[choice[i,t,s]]);
      }        
    }
  }
}
  
  
generated quantities {
  // THIS SHOULD BE ADJUSTED TO THE DATA FOR THE THEORETICAL MODEL OF INTEREST
  vector[2] utility;      // declare utility variable
  real log_lik[N];        // log-likelihoods for calculating LOOIC
  int choice_pred[N,T,S]; // posterior predicted values
  corr_matrix[2] R_theta; // correlation matrix
  matrix[S,N] A;          // declare bounded theoretical model-parameter
  
  // calculate correlation matrix
  R_theta = R_chol * R_chol';
  
  // initialize log_lik to 0 and choice_pred = -1
  for (i in 1:N) {
    log_lik[i] = 0;
    
    for (s in 1:S) {
      for (t in 1:T) {
        choice_pred[i,t,s] = -1;
      }
    }
  }
  
  // LIKELIHOOD
  for(s in 1:S){ // loop through sessions
    for(i in 1:N) { // loop through persons
      A[s,i] = Phi_approx(theta[s,i]); // transform theoretical model parameter
      utility = rep_vector(0,2); // set values of utility vector to 0 for starting the session
      
      for(t in 1:T) { // loop through trials
        log_lik[i] += categorical_logit_lpmf(choice[i,t,s]|utility); // calculate log-likelihood for each participant
        choice_pred[i,t,s] = categorical_rng(softmax(utility)); // calculate posterior predicted value
        
        // RW learning rule
        utility[choice_pred[i,t,s]] = utility[choice_pred[i,t,s]] + A[s,i]*(outcome[i,t,s] - utility[choice_pred[i,t,s]]);
      }        
    }
  }
}
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
  // THIS SHOULD BE ADJUSTED TO THE DATA FOR THE THEORETICAL MODEL OF INTEREST
  int<lower=1> N;           // number of participants
  int<lower=1> T;           // total possile number of trials
  
  array[N,T] int choice;  // choices made by participants
  array[N,T] int outcome; // outcomes received by participants after making choice
}


parameters {
  // THIS CAN BE USED FOR ANY SINGLE-PARAMETER MODEL
  real mu_theta;         // group-level mean
  real<lower=0> sigma_U; // SD of unbounded theoretical parameter
  vector[N] z_U;         // standardized person-specific deviations from group-level mean
}


transformed parameters {
  vector[N] U;     // person-specific deviations (in SD units of unbounded theoretical parameter) from group-level mean
  vector[N] theta; // unbounded theoretical model-parameter for each participant
  
  U = sigma_U * z_U;    // calculate person-specific deviations
  theta = mu_theta + U; // calculate unbounded person-specific parameters
}
  

model {
  
  // PRIORS: THESE CAN BE USED FOR ANY SINGLE-PARAMETER THEORETICAL MODEL
  mu_theta ~ normal(0,1);  // group-level mean
  sigma_U  ~ normal(0,.2); // group-level SD
  z_U      ~ normal(0,1);  // standardized person-specific deviations
  
  // LIKELIHOOD: THIS SHOULD BE ADJUSTED FOR THE THEORETICAL MODEL OF INTEREST
  vector[2] utility; // declare utility variable
  vector[N] A;       // declare bounded theoretical model-parameter
  
  for(i in 1:N) { // loop through persons
    A[i] = Phi_approx(theta[i]); // transform theoretical model parameter
    utility = rep_vector(0,2); // set values of utility vector to 0 for starting the session
    
    for(t in 1:T) { // loop through trials
      choice[i,t] ~ categorical_logit(utility); // predict choice based on utility
      
      // RW learning rule
      utility[choice[i,t]] = utility[choice[i,t]] + A[i]*(outcome[i,t] - utility[choice[i,t]]);
    }
  }
}
  
  
generated quantities {
  // THIS SHOULD BE ADJUSTED TO THE DATA FOR THE THEORETICAL MODEL OF INTEREST
  vector[2] utility;    // declare utility variable
  real log_lik[N];      // log-likelihoods for calculating LOOIC
  int choice_pred[N,T]; // posterior predicted values
  vector[N] A;          // declare bounded theoretical model-parameter
  real mu_A;            // bounded group-level learning rate
  
  mu_A = Phi_approx_group_mean_rng(mu_theta, sigma_U, 10000); // calculate posterior of bounded group-level learning rates
  
  // initialize log_lik to 0 and choice_pred = -1
  for (i in 1:N) {
    log_lik[i] = 0;
    
    for (t in 1:T) {
      choice_pred[i,t] = -1;
    }
  }
  
  // LIKELIHOOD
  for(i in 1:N) { // loop through persons
    A[i] = Phi_approx(theta[i]); // transform theoretical model parameter
    utility = rep_vector(0,2);   // set values of utility vector to 0 for starting the session
    
    for(t in 1:T) { // loop through trials
      log_lik[i] += categorical_logit_lpmf(choice[i,t]|utility); // calculate log-likelihood for each participant
      choice_pred[i,t] = categorical_rng(softmax(utility));      // calculate posterior predicted value
      
      // RW learning rule
      utility[choice_pred[i,t]] = utility[choice_pred[i,t]] + A[i]*(outcome[i,t] - utility[choice_pred[i,t]]);
    }
  }
}

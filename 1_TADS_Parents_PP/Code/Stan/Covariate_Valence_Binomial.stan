

data {
  int<lower=1> N;       // number of participants
  int<lower=1> D;       // number of person-level predictors
  array[N,D] real X;    // person-level predictors
  array[N] int A_trials; // number of trials for deck A for each participant
  array[N] int A_plays; // number of plays for deck A for each participant
  array[N] int B_trials; // number of trials for deck B for each participant
  array[N] int B_plays; // number of plays for deck B for each participant
  array[N] int C_trials; // number of trials for deck C for each participant
  array[N] int C_plays; // number of plays for deck C for each participant
  array[N] int D_trials; // number of trials for deck D for each participant
  array[N] int D_plays; // number of plays for deck D for each participant
}


transformed data {
  array[N] int good_plays;
  array[N] int good_trials;
  array[N] int bad_plays;
  array[N] int bad_trials;
  
  for(i in 1:N){
    good_plays[i] = C_plays[i] + D_plays[i];
    good_trials[i] = C_trials[i] + D_trials[i];
    bad_plays[i] = A_plays[i] + B_plays[i];
    bad_trials[i] = A_trials[i] + B_trials[i];
  }
}


parameters {
  // group-level betas
  vector[D] mu_good;        // effect of person-level predictors for good decks
  vector[D] mu_bad;         // effect of person-level predictors for bad decks
  real<lower=0> sigma_good; // variance of subject-level betas for good decks
  real<lower=0> sigma_bad;  // variance of subject-level betas for bad decks
  
  // person-level log-odds, combining intercepts and slopes
  vector[N] tilde_good; // subject-level deviation from mean for good decks
  vector[N] tilde_bad;  // subject-level deviation from mean for bad decks
}


transformed parameters {
  vector[N] beta_good; // subject-level beta from mean for good decks
  vector[N] beta_bad;  // subject-level beta from mean for bad decks
  
  for(i in 1:N){
    beta_good[i] = dot_product(mu_good, to_vector(X[i,:])) + tilde_good[i]*sigma_good;
    beta_bad[i] = dot_product(mu_bad, to_vector(X[i,:])) + tilde_bad[i]*sigma_bad;
  }
}


model {
  // PRIORS
  // hyperpriors
  mu_good ~ normal(0,1);    // effect of person-level predictors for good decks
  mu_bad ~ normal(0,1);     // effect of person-level predictors for bad decks
  sigma_good ~ cauchy(0,1); // variance of subject-level betas for good decks
  sigma_bad ~ cauchy(0,1);  // variance of subject-level betas for bad decks
  
  // person-level priors
  tilde_good ~ normal(0,1); // subject-level deviation from mean for good decks
  tilde_bad ~ normal(0,1);  // subject-level deviation from mean for bad decks
  
  // LIKELIHOOD
  for(i in 1:N){
    if(X[i,1]>0){
      good_plays[i] ~ binomial_logit(good_trials[i], beta_good[i]);
      bad_plays[i] ~ binomial_logit(bad_trials[i], beta_bad[i]);
    }
  }
}


generated quantities {
  vector<lower=0,upper=1>[D] mu_good_theta;
  vector<lower=0,upper=1>[D] mu_bad_theta;
  vector<lower=0,upper=1>[N] good_theta;
  vector<lower=0,upper=1>[N] bad_theta;
  real log_lik[N];

  // For posterior predictive check
  real good_pred[N];
  real bad_pred[N];
  
  // calculate group-level probabilities
  mu_good_theta[1] = inv_logit(mu_good[1]);
  mu_bad_theta[1] = inv_logit(mu_bad[1]);
  for(d in 2:D){
    mu_good_theta[d] = inv_logit(mu_good[1] + mu_good[d]);
    mu_bad_theta[d] = inv_logit(mu_bad[1] + mu_bad[d]);
  }
  
  // calculate person-level probabilities
  good_theta = inv_logit(beta_good);
  bad_theta = inv_logit(beta_bad);
  
  { // LOCAL SECTION
    for(i in 1:N){
      // initialize values
      log_lik[i] = 0;
      good_pred[i] = -1;
      bad_pred[i] = -1;
      if(X[i,1]>0){
        log_lik[i] =  binomial_logit_lpmf(good_plays[i] | good_trials[i], beta_good[i]);
        log_lik[i] += binomial_logit_lpmf(bad_plays[i] | bad_trials[i], beta_bad[i]);
        
        good_pred[i] = binomial_rng(good_trials[i], good_theta[i]);
        bad_pred[i] = binomial_rng(bad_trials[i], bad_theta[i]);
      }
    }
  }
}











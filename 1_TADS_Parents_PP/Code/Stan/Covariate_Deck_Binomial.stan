

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


parameters {
  // group-level betas
  vector[D] mu_A;         // effect of person-level predictors for deck A
  vector[D] mu_B;         // effect of person-level predictors for deck B
  vector[D] mu_C;         // effect of person-level predictors for deck C
  vector[D] mu_D;         // effect of person-level predictors for deck D
  real<lower=0> sigma_A;  // variance of subject-level betas for A
  real<lower=0> sigma_B;  // variance of subject-level betas for B
  real<lower=0> sigma_C;  // variance of subject-level betas for C
  real<lower=0> sigma_D;  // variance of subject-level betas for D
  
  // person-level log-odds, combining intercepts and slopes
  vector[N] tilde_A; // subject-level deviation from mean for deck A
  vector[N] tilde_B; // subject-level deviation from mean for deck B
  vector[N] tilde_C; // subject-level deviation from mean for deck C
  vector[N] tilde_D; // subject-level deviation from mean for deck D
}


transformed parameters {
  vector[N] beta_A; // subject-level beta from mean for deck A
  vector[N] beta_B; // subject-level beta from mean for deck A
  vector[N] beta_C; // subject-level beta from mean for deck A
  vector[N] beta_D; // subject-level beta from mean for deck A
  
  for(i in 1:N){
    beta_A[i] = dot_product(mu_A, to_vector(X[i,:])) + tilde_A[i]*sigma_A;
    beta_B[i] = dot_product(mu_B, to_vector(X[i,:])) + tilde_B[i]*sigma_B;
    beta_C[i] = dot_product(mu_C, to_vector(X[i,:])) + tilde_C[i]*sigma_C;
    beta_D[i] = dot_product(mu_D, to_vector(X[i,:])) + tilde_D[i]*sigma_D;
  }
}


model {
  // PRIORS
  // hyperpriors
  mu_A ~ normal(0,1);    // effect of person-level predictors for deck A
  mu_B ~ normal(0,1);    // effect of person-level predictors for deck B
  mu_C ~ normal(0,1);    // effect of person-level predictors for deck C
  mu_D ~ normal(0,1);    // effect of person-level predictors for deck D
  sigma_A ~ normal(0,1); // variance of subject-level betas for A
  sigma_B ~ normal(0,1); // variance of subject-level betas for B
  sigma_C ~ normal(0,1); // variance of subject-level betas for C
  sigma_D ~ normal(0,1); // variance of subject-level betas for D
  
  // person-level priors
  tilde_A ~ normal(0,1); // subject-level deviation from mean for deck A
  tilde_B ~ normal(0,1); // subject-level deviation from mean for deck B
  tilde_C ~ normal(0,1); // subject-level deviation from mean for deck C
  tilde_D ~ normal(0,1); // subject-level deviation from mean for deck D
  
  // LIKELIHOOD
  for(i in 1:N){
    if(X[i,1]>0){
      A_plays[i] ~ binomial_logit(A_trials[i], beta_A[i]);
      B_plays[i] ~ binomial_logit(B_trials[i], beta_B[i]);
      C_plays[i] ~ binomial_logit(C_trials[i], beta_C[i]);
      D_plays[i] ~ binomial_logit(D_trials[i], beta_D[i]);
    }
  }
}


generated quantities {
  vector<lower=0,upper=1>[D] mu_A_theta;
  vector<lower=0,upper=1>[D] mu_B_theta;
  vector<lower=0,upper=1>[D] mu_C_theta;
  vector<lower=0,upper=1>[D] mu_D_theta;
  vector<lower=0,upper=1>[N] A_theta;
  vector<lower=0,upper=1>[N] B_theta;
  vector<lower=0,upper=1>[N] C_theta;
  vector<lower=0,upper=1>[N] D_theta;
  real log_lik[N];

  // For posterior predictive check
  real A_pred[N];
  real B_pred[N];
  real C_pred[N];
  real D_pred[N];
  
  // calculate group-level probabilities
  mu_A_theta[1] = inv_logit(mu_A[1]);
  mu_B_theta[1] = inv_logit(mu_B[1]);
  mu_C_theta[1] = inv_logit(mu_C[1]);
  mu_D_theta[1] = inv_logit(mu_D[1]);
  for(d in 2:D){
    mu_A_theta[d] = inv_logit(mu_A[1] + mu_A[d]);
    mu_B_theta[d] = inv_logit(mu_B[1] + mu_B[d]);
    mu_C_theta[d] = inv_logit(mu_C[1] + mu_C[d]);
    mu_D_theta[d] = inv_logit(mu_D[1] + mu_D[d]);
  }
  
  // calculate person-level probabilities
  A_theta = inv_logit(beta_A);
  B_theta = inv_logit(beta_B);
  C_theta = inv_logit(beta_C);
  D_theta = inv_logit(beta_D);
  
  { // LOCAL SECTION
    for(i in 1:N){
      // initialize values
      log_lik[i] = 0;
      A_pred[i] = -1;
      B_pred[i] = -1;
      C_pred[i] = -1;
      D_pred[i] = -1;
      if(X[i,1]>0){
        log_lik[i] =  binomial_logit_lpmf(A_plays[i] | A_trials[i], beta_A[i]);
        log_lik[i] += binomial_logit_lpmf(B_plays[i] | B_trials[i], beta_B[i]);
        log_lik[i] += binomial_logit_lpmf(C_plays[i] | C_trials[i], beta_C[i]);
        log_lik[i] += binomial_logit_lpmf(D_plays[i] | D_trials[i], beta_D[i]);
        
        A_pred[i] = binomial_rng(A_trials[i], A_theta[i]);
        B_pred[i] = binomial_rng(B_trials[i], B_theta[i]);
        C_pred[i] = binomial_rng(C_trials[i], C_theta[i]);
        D_pred[i] = binomial_rng(D_trials[i], D_theta[i]);
      }
    }
  }
}











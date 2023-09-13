

data {
  int<lower=1> N;
  int<lower=1> S;
  int<lower=1> T;
  int Tsubj[N,S];
  int good[N,T,S];
  int choice[N,T,S];
}


parameters {
  vector[S] mu_good;
  vector[S] mu_bad;
  vector<lower=0>[S] sigma_good;
  vector<lower=0>[S] sigma_bad;
  matrix[S,N] pl_good;
  matrix[S,N] pl_bad;
  
  cholesky_factor_corr[2] R_chol_good;
  cholesky_factor_corr[2] R_chol_bad;
}


transformed parameters {
  matrix[S,N] good_tilde;
  matrix[S,N] bad_tilde;
  matrix[N,S] good_alpha;
  matrix[N,S] bad_alpha;
  
  // Untransformed subject-level parameters incorporating correlation between sessions
  good_tilde  = diag_pre_multiply(sigma_good, R_chol_good) * pl_good;
  bad_tilde  = diag_pre_multiply(sigma_bad, R_chol_bad) * pl_bad;
  
  for(s in 1:S){
    for(i in 1:N){
      good_alpha[i,s] = mu_good[s] + good_tilde[s,i];
      bad_alpha[i,s] = mu_bad[s] + bad_tilde[s,i];
    }
  }
}


model {
  // PRIORS
  // hyperpriors
  mu_good     ~ normal(0,1);
  sigma_good  ~ normal(0,1);
  mu_bad      ~ normal(0,1);
  sigma_bad   ~ normal(0,1);
  R_chol_good ~ lkj_corr_cholesky(1);
  R_chol_bad  ~ lkj_corr_cholesky(1);
  
  // person-level priors
  to_vector(pl_good) ~ normal(0,1);
  to_vector(pl_bad)  ~ normal(0,1);
  
  // LIKELIHOOD
  for(s in 1:S){
    for(i in 1:N){
      if(Tsubj[i,s] > 0){
        for(t in 1:Tsubj[i,s]){
          if(good[i,t,s] == 1){
            choice[i,t,s] ~ bernoulli_logit(good_alpha[i,s]);
          }else{
            choice[i,t,s] ~ bernoulli_logit(bad_alpha[i,s]);
          }
        }
      }
    }
  }
}


generated quantities {
  vector<lower=0,upper=1>[S] mu_good_p;
  vector<lower=0,upper=1>[S] mu_bad_p;
  matrix<lower=0,upper=1>[S,N] pl_good_p;
  matrix<lower=0,upper=1>[S,N] pl_bad_p;
  real log_lik[N];

  // For posterior predictive check
  real y_pred[N,T,S];

  // test-retest correlations
  corr_matrix[2] R_good;
  corr_matrix[2] R_bad;
  
  // Reconstruct correlation matrix from cholesky factor
  R_good = R_chol_good * R_chol_good';
  R_bad = R_chol_bad * R_chol_bad';
  
  // Set all posterior predictions to -1 (avoids NULL values)
  for (i in 1:N) {
    for (s in 1:S) {
      for (t in 1:T) {
        y_pred[i,t,s] = -1;
      }
    }
  }
  
  mu_good_p[1] = inv_logit(mu_good[1]);
  mu_good_p[2] = inv_logit(mu_good[2]);
  mu_bad_p[1] = inv_logit(mu_bad[1]);
  mu_bad_p[2] = inv_logit(mu_bad[2]);
  
  for (i in 1:N) {
    for (s in 1:S) {
      pl_good_p[s,i] = inv_logit(mu_good[s] + good_alpha[i,s]);
      pl_bad_p[s,i] = inv_logit(mu_bad[s] + bad_alpha[i,s]);  
    }
  }
  
  { // LOCAL SECTION
    // initialize log_lik
    for(i in 1:N){
      log_lik[i] = 0;
    }
    
    for(s in 1:S){
      for(i in 1:N){
        if(Tsubj[i,s] > 0){
          for(t in 1:Tsubj[i,s]){
            if(good[i,t,s] == 1){
              log_lik[i] += bernoulli_logit_lpmf(choice[i,t,s] | good_alpha[i,s]);
              y_pred[i,t,s] = bernoulli_logit_rng(good_alpha[i,s]);
            }else{
              log_lik[i] += bernoulli_logit_lpmf(choice[i,t,s] | bad_alpha[i,s]);
              y_pred[i,t,s] = bernoulli_logit_rng(bad_alpha[i,s]);
            }
          }
        }
      }
    }
  }
}













data {
  int<lower=1> N;                      // Number of participants
  int<lower=1> S;                      // Number of sessions
  int<lower=1> T;                      // Total possile number of trials
  int<lower=1> D;                      // Number of person-level predictors
  array[N,T,S] int card;               // Cards presented on each trial
  array[N,S] int Tsubj;                // Total number of trials presented to each subject on each session
  array[N,T,S] int choice;             // Choices on each trial
  array[N,T,S] real outcome;           // Outcomes received on each trial
  array[N,T,S] real sign;              // Signs of the outcome received on each trial
  array[N,D,S] real X;                 // person-level predictors
}


transformed data {
  int B;            // number of blocks
  B = 3;
  
  int A_trials[N,B]; // number of trials for deck A for each participant
  int A_plays[N,B]; // number of plays for deck A for each participant
  int B_trials[N,B]; // number of trials for deck B for each participant
  int B_plays[N,B]; // number of plays for deck B for each participant
  int C_trials[N,B]; // number of trials for deck C for each participant
  int C_plays[N,B]; // number of plays for deck C for each participant
  int D_trials[N,B]; // number of trials for deck D for each participant
  int D_plays[N,B]; // number of plays for deck D for each participant
  int cur_block[4]; // varriable to keep track of blocks
  
  
  A_plays = rep_array(0,N,B);
  A_trials = rep_array(0,N,B);
  B_plays = rep_array(0,N,B);
  B_trials = rep_array(0,N,B);
  C_plays = rep_array(0,N,B);
  C_trials = rep_array(0,N,B);
  D_plays = rep_array(0,N,B);
  D_trials = rep_array(0,N,B);
  
  for(s in 1:S){
    for(i in 1:N) {
  
      cur_block = rep_array(1,4);
      
      for(t in 1:Tsubj[i,s]){
        
        if(card[i,t,s] == 1){
          A_trials[i,cur_block[1]] += 1;
          A_plays[i,cur_block[1]] += (choice[i,t,s]-2)*-1;
          if(A_trials[i,cur_block[1]] % (30/B) == 0){
            cur_block[1] += 1;
          }
        }
        
        if(card[i,t,s] == 2){
          B_trials[i,cur_block[2]] += 1;
          B_plays[i,cur_block[2]] += (choice[i,t,s]-2)*-1;
          if(B_trials[i,cur_block[2]] % (30/B) == 0){
            cur_block[2] += 1;
          }
        }
        
        if(card[i,t,s] == 3){
          C_trials[i,cur_block[3]] += 1;
          C_plays[i,cur_block[3]] += (choice[i,t,s]-2)*-1;
          if(C_trials[i,cur_block[3]] % (30/B) == 0){
            cur_block[3] += 1;
          }
        }
        
        if(card[i,t,s] == 4){
          D_trials[i,cur_block[4]] += 1;
          D_plays[i,cur_block[4]] += (choice[i,t,s]-2)*-1;
          if(D_trials[i,cur_block[4]] % (30/B) == 0){
            cur_block[4] += 1;
          }
        }
        
      }
    }
  }
}


parameters {
  // group-level betas
  matrix[D,B] mu_A;         // effect of person-level predictors for deck A
  matrix[D,B] mu_B;         // effect of person-level predictors for deck B
  matrix[D,B] mu_C;         // effect of person-level predictors for deck C
  matrix[D,B] mu_D;         // effect of person-level predictors for deck D
  vector<lower=0>[B] sigma_A; // variance of subject-level betas for A
  vector<lower=0>[B] sigma_B; // variance of subject-level betas for B
  vector<lower=0>[B] sigma_C; // variance of subject-level betas for C
  vector<lower=0>[B] sigma_D; // variance of subject-level betas for D
  
  // person-level log-odds, combining intercepts and slopes
  matrix[N,B] tilde_A; // subject-level deviation from mean for deck A
  matrix[N,B] tilde_B; // subject-level deviation from mean for deck B
  matrix[N,B] tilde_C; // subject-level deviation from mean for deck C
  matrix[N,B] tilde_D; // subject-level deviation from mean for deck D
}


transformed parameters {
  matrix[N,B] beta_A; // subject-level beta from mean for deck A
  matrix[N,B] beta_B; // subject-level beta from mean for deck A
  matrix[N,B] beta_C; // subject-level beta from mean for deck A
  matrix[N,B] beta_D; // subject-level beta from mean for deck A
  
  for(i in 1:N){
    for(b in 1:B){
      beta_A[i,b] = dot_product(mu_A[:,b], to_vector(X[i,:,1])) + tilde_A[i,b]*sigma_A[b];
      beta_B[i,b] = dot_product(mu_B[:,b], to_vector(X[i,:,1])) + tilde_B[i,b]*sigma_B[b];
      beta_C[i,b] = dot_product(mu_C[:,b], to_vector(X[i,:,1])) + tilde_C[i,b]*sigma_C[b];
      beta_D[i,b] = dot_product(mu_D[:,b], to_vector(X[i,:,1])) + tilde_D[i,b]*sigma_D[b];
    }
  }
}


model {
  // PRIORS
  // hyperpriors
  for(b in 1:B){
    to_vector(mu_A) ~ normal(0,1);    // effect of person-level predictors for deck A
    to_vector(mu_B) ~ normal(0,1);    // effect of person-level predictors for deck B
    to_vector(mu_C) ~ normal(0,1);    // effect of person-level predictors for deck C
    to_vector(mu_D) ~ normal(0,1);    // effect of person-level predictors for deck D
    to_vector(sigma_A) ~ normal(0,1); // variance of subject-level betas for A
    to_vector(sigma_B) ~ normal(0,1); // variance of subject-level betas for B
    to_vector(sigma_C) ~ normal(0,1); // variance of subject-level betas for C
    to_vector(sigma_D) ~ normal(0,1); // variance of subject-level betas for D
    
    // person-level priors
    to_vector(tilde_A) ~ normal(0,1); // subject-level deviation from mean for deck A
    to_vector(tilde_B) ~ normal(0,1); // subject-level deviation from mean for deck B
    to_vector(tilde_C) ~ normal(0,1); // subject-level deviation from mean for deck C
    to_vector(tilde_D) ~ normal(0,1); // subject-level deviation from mean for deck D
  }
  
  // LIKELIHOOD
  for(i in 1:N){
    if(X[i,1,1]>0){
      for(b in 1:B){
        A_plays[i,b] ~ binomial_logit(A_trials[i,b], beta_A[i,b]);
        B_plays[i,b] ~ binomial_logit(B_trials[i,b], beta_B[i,b]);
        C_plays[i,b] ~ binomial_logit(C_trials[i,b], beta_C[i,b]);
        D_plays[i,b] ~ binomial_logit(D_trials[i,b], beta_D[i,b]);
      }
    }
  }
}


generated quantities {
  // transformed data to be outputted
  int A_trials_data[N,B]; // number of trials for deck A for each participant
  int A_plays_data[N,B]; // number of plays for deck A for each participant
  int B_trials_data[N,B]; // number of trials for deck B for each participant
  int B_plays_data[N,B]; // number of plays for deck B for each participant
  int C_trials_data[N,B]; // number of trials for deck C for each participant
  int C_plays_data[N,B]; // number of plays for deck C for each participant
  int D_trials_data[N,B]; // number of trials for deck D for each participant
  int D_plays_data[N,B]; // number of plays for deck D for each participant
  // parameters to be outputted
  real<lower=0,upper=1> mu_A_theta[D,B];
  real<lower=0,upper=1> mu_B_theta[D,B];
  real<lower=0,upper=1> mu_C_theta[D,B];
  real<lower=0,upper=1> mu_D_theta[D,B];
  real<lower=0,upper=1> A_theta[N,B];
  real<lower=0,upper=1> B_theta[N,B];
  real<lower=0,upper=1> C_theta[N,B];
  real<lower=0,upper=1> D_theta[N,B];
  real log_lik[N];

  // For posterior predictive check
  real A_pred[N,B];
  real B_pred[N,B];
  real C_pred[N,B];
  real D_pred[N,B];
  
  // save transformed data
  A_trials_data = A_trials; // number of trials for deck A for each participant
  A_plays_data = A_plays; // number of plays for deck A for each participant
  B_trials_data = B_trials; // number of trials for deck B for each participant
  B_plays_data = B_plays; // number of plays for deck B for each participant
  C_trials_data = C_trials; // number of trials for deck C for each participant
  C_plays_data = C_plays; // number of plays for deck C for each participant
  D_trials_data = D_trials; // number of trials for deck D for each participant
  D_plays_data = D_plays; // number of plays for deck D for each participant
  
  // calculate group-level probabilities
  for(b in 1:B){
    mu_A_theta[1,b] = inv_logit(mu_A[1,b]);
    mu_B_theta[1,b] = inv_logit(mu_B[1,b]);
    mu_C_theta[1,b] = inv_logit(mu_C[1,b]);
    mu_D_theta[1,b] = inv_logit(mu_D[1,b]);
    for(d in 2:D){
      mu_A_theta[d,b] = inv_logit(mu_A[1,b] + mu_A[d,b]);
      mu_B_theta[d,b] = inv_logit(mu_B[1,b] + mu_B[d,b]);
      mu_C_theta[d,b] = inv_logit(mu_C[1,b] + mu_C[d,b]);
      mu_D_theta[d,b] = inv_logit(mu_D[1,b] + mu_D[d,b]);
    }
    // calculate person-level probabilities
    A_theta[:,b] = to_array_1d(inv_logit(beta_A[:,b]));
    B_theta[:,b] = to_array_1d(inv_logit(beta_B[:,b]));
    C_theta[:,b] = to_array_1d(inv_logit(beta_C[:,b]));
    D_theta[:,b] = to_array_1d(inv_logit(beta_D[:,b]));
  }
  
  { // LOCAL SECTION
    for(i in 1:N){
      // initialize values
      log_lik[i] = 0;
      A_pred[i,:] = rep_array(-1,B);
      B_pred[i,:] = rep_array(-1,B);
      C_pred[i,:] = rep_array(-1,B);
      D_pred[i,:] = rep_array(-1,B);
      if(X[i,1,1]>0){
        for(b in 1:B){
          log_lik[i] =  binomial_logit_lpmf(A_plays[i,b] | A_trials[i,b], beta_A[i,b]);
          log_lik[i] += binomial_logit_lpmf(B_plays[i,b] | B_trials[i,b], beta_B[i,b]);
          log_lik[i] += binomial_logit_lpmf(C_plays[i,b] | C_trials[i,b], beta_C[i,b]);
          log_lik[i] += binomial_logit_lpmf(D_plays[i,b] | D_trials[i,b], beta_D[i,b]);
          
          A_pred[i,b] = binomial_rng(A_trials[i,b], A_theta[i,b]);
          B_pred[i,b] = binomial_rng(B_trials[i,b], B_theta[i,b]);
          C_pred[i,b] = binomial_rng(C_trials[i,b], C_theta[i,b]);
          D_pred[i,b] = binomial_rng(D_trials[i,b], D_theta[i,b]);
        }
      }
    }
  }
}











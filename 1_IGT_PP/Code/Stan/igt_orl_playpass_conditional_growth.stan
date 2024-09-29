data {
  int<lower=1> N;                      // Number of participants
  int<lower=1> T;                      // Total possile number of trials
  int<lower=1> D;                      // Number of person-level predictors
  int<lower=1> S;                      // Number of sessions
  real time[N,S];
  int session_start[N,T];
  int card[N,T];                     // Cards presented on each trial
  int Tsubj[N];                      // Total number of trials presented to each subject on each session
  int choice[N,T];                   // Choices on each trial
  real outcome[N,T];                 // Outcomes received on each trial
  real sign[N,T];                    // Signs of the outcome received on each trial
  array[N,S,D] real X;                       // person-level predictors
  array[8] int D_start;
  array[8] int D_end;
  int prior_only;
}


parameters {
  vector[D] gamma0;
  vector[D] gamma1;
  
  array[4] matrix[2,N] beta_pr;  
  array[4] vector<lower=0>[2] sigma_beta;
  array[4] cholesky_factor_corr[2] R_chol;
  
  array[4] matrix[N,S] theta;
  array[4] real<lower=0> sigma_theta;
}

transformed parameters {
  array[4] matrix[2,N] beta_tilde;
  array[4,S] matrix[N,2] beta; 
  matrix[N,S] Arew;
  matrix[N,S] Apun;
  matrix[N,S] betaF;
  matrix[N,S] betaP;

  for (p in 1:4) {
    beta_tilde[p] = diag_pre_multiply(sigma_beta[p], R_chol[p]) * beta_pr[p];  
    for (s in 1:S) {
      beta[p,s][,1] = to_matrix(X[,s,D_start[p]:D_end[p]]) * gamma0[D_start[p]:D_end[p]] + to_vector(beta_tilde[p][1,]);
      beta[p,s][,2] = to_matrix(X[,s,D_start[p+4]:D_end[p+4]]) * gamma1[D_start[p+4]:D_end[p+4]] + to_vector(beta_tilde[p][2,]); 
    }
  }
  
  for (s in 1:S) {
    for (i in 1:N) {
      Arew[i,s] = Phi_approx(beta[1,s][i,1] + beta[1,s][i,2] * time[i,s] + sigma_theta[1] * theta[1][i,s]);
      Apun[i,s] = Phi_approx(beta[2,s][i,1] + beta[2,s][i,2] * time[i,s] + sigma_theta[2] * theta[2][i,s]);
      betaF[i,s] = beta[3,s][i,1] + beta[3,s][i,2] * time[i,s] + sigma_theta[3] * theta[3][i,s];
      betaP[i,s] = beta[4,s][i,1] + beta[4,s][i,2] * time[i,s] + sigma_theta[4] * theta[4][i,s];
    }  
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
  int session;
  
  // Priors
  for (p in 1:4) {
    R_chol[p] ~ lkj_corr_cholesky(1);  
    gamma0[p] ~ normal(-1,1);
    gamma1[p] ~ normal(0,.5);
    to_vector(beta_pr[p]) ~ std_normal();
    sigma_beta[p] ~ normal(0,.2);
    to_vector(theta[p]) ~ std_normal();
    sigma_theta[p] ~ normal(0,.2);
  }
  
  for (i in 1:N) {         
    session = 0;
    if (Tsubj[i] > 0) {    
      for (t in 1:Tsubj[i]) { 
        if (session_start[i,t] == 1) {
          session += 1;
          ev = rep_vector(0,4);
          ef = rep_vector(0,4);
          pers = rep_vector(1,4);
          utility = ev + ef * betaF[i,session] + pers * betaP[i,session];  
        }
        if (!prior_only)
          choice[i,t] ~ categorical_logit(to_vector({utility[card[i,t]], 0}));
          
        if (choice[i,t]==1) {
          PEval      = outcome[i,t] - ev[card[i,t]];
          PEfreq     = sign[i,t] - ef[card[i,t]];   
          PEfreq_fic = -sign[i,t]/3 - ef;
          ef_chosen  = ef[card[i,t]];
            
          if (outcome[i,t] >= 0) {
            ef = ef + Apun[i,session] * PEfreq_fic;
            ef[card[i,t]] = ef_chosen + Arew[i,session] * PEfreq;
            ev[card[i,t]] = ev[card[i,t]] + Arew[i,session] * PEval;
          } else {
            ef = ef + Arew[i,session] * PEfreq_fic;
            ef[card[i,t]] = ef_chosen + Apun[i,session] * PEfreq;
            ev[card[i,t]] = ev[card[i,t]] + Apun[i,session] * PEval;
          }
        }
        utility = ev + ef * betaF[i,session] + pers * betaP[i,session];
      }
    }
  }
}

generated quantities {
  int session;
  real log_lik[N,T];
  real choice_pred[N,T];

  // random effect intercept-slope correlations
  corr_matrix[S] R_Arew;
  corr_matrix[S] R_Apun;
  corr_matrix[S] R_betaF;
  corr_matrix[S] R_betaP;

  R_Arew  = R_chol[1] * R_chol[1]';
  R_Apun  = R_chol[2] * R_chol[2]';
  R_betaF = R_chol[3] * R_chol[3]';
  R_betaP = R_chol[4] * R_chol[4]';

  // Set all posterior predictions to -99 (avoids NULL values)
  for (i in 1:N) {
    for (s in 1:S) {
      for (t in 1:T) {
        choice_pred[i,t] = -99;
        log_lik[i,t] = -99;
      }
    }
  }

  { // local section, this saves time and space
    vector[4] ef;
    vector[4] ev;
    vector[4] pers;
    vector[4] utility;

    real ef_chosen;
    real PEval;
    real PEfreq;
    vector[4] PEfreq_fic;

    for (i in 1:N) {   
      session = 0;
      for (t in 1:Tsubj[i]) { 
        if (session_start[i,t] == 1) {
          session += 1;
          ev = rep_vector(0,4);
          ef = rep_vector(0,4);
          pers = rep_vector(1,4);
          utility = ev + ef * betaF[i,session] + pers * betaP[i,session];  
        }
        log_lik[i,t] = categorical_logit_lpmf(choice[i,t] | to_vector({utility[card[i,t]], 0}));   
        choice_pred[i,t] = categorical_rng(softmax(to_vector({utility[card[i,t]], 0})));   
  
        if (choice[i,t]==1) {
          PEval      = outcome[i,t] - ev[card[i,t]];
          PEfreq     = sign[i,t] - ef[card[i,t]];   
          PEfreq_fic = -sign[i,t]/3 - ef;
          ef_chosen  = ef[card[i,t]];
            
          if (outcome[i,t] >= 0) {
            ef = ef + Apun[i,session] * PEfreq_fic;
            ef[card[i,t]] = ef_chosen + Arew[i,session] * PEfreq;
            ev[card[i,t]] = ev[card[i,t]] + Arew[i,session] * PEval;
          } else {
            ef = ef + Arew[i,session] * PEfreq_fic;
            ef[card[i,t]] = ef_chosen + Apun[i,session] * PEfreq;
            ev[card[i,t]] = ev[card[i,t]] + Apun[i,session] * PEval;
          }
        }
        utility = ev + ef * betaF[i,session] + pers * betaP[i,session];
      }
    }
  }
}

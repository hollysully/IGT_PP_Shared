functions {
  real trunc_normal_rng(real mu, real sigma) {
    real u = uniform_rng(.5, 1);
    real z = inv_Phi(u);
    real y = mu + sigma * z;
    return y;
  }
}

data {
  int<lower=1> N;                      // Number of participants
  int<lower=1> T;                      // Total possile number of trials
  int<lower=1> D;                      // Number of person-level predictors
  int<lower=1> S;                      // Number of sessions
  real time[N,S];
  int card[T/2];                     // Cards presented on each trial                    // Total number of trials presented to each subject on each session
  real outcome[4,T/8];                 // Outcomes received on each trial
  array[N,T,D] real X;                       // person-level predictors
  array[8] int D_start;
  array[8] int D_end;
}

generated quantities {
  vector[D] gamma0;
  vector[D] gamma1;
  array[4] matrix[2,N] beta_pr;  
  array[4] vector<lower=0>[2] sigma_beta;
  array[4] cholesky_factor_corr[2] R_chol;
  array[4] matrix[N,S] theta;
  array[4] real<lower=0> sigma_theta;
  array[4] matrix[2,N] beta_tilde;
  array[4] matrix[N,2] beta; 
  matrix[N,S] Arew;
  matrix[N,S] Apun;
  matrix[N,S] betaF;
  matrix[N,S] betaP;
  
  int session;
  real log_lik[N,T];
  int choice[N,T];
  int card_sim[N,T];
  real outcome_sim[N,T];   

  // random effect intercept-slope correlations
  corr_matrix[S] R_Arew;
  corr_matrix[S] R_Apun;
  corr_matrix[S] R_betaF;
  corr_matrix[S] R_betaP;
  
  for (p in 1:4) {
    R_chol[p] = lkj_corr_cholesky_rng(2, 1);  
    gamma0[p] = normal_rng(-1,1);
    gamma1[p] = normal_rng(0,.5);
    for (t in 1:2) {
      sigma_beta[p][t] = trunc_normal_rng(0,.2);
      for (i in 1:N) {
        beta_pr[p][t,i] = normal_rng(0,1);
      }
    }
    sigma_theta[p] = trunc_normal_rng(0,.2);
    for (s in 1:S) {
      for (i in 1:N) {
        theta[p][i,s] = normal_rng(0,1);
      }
    }
    beta_tilde[p] = diag_pre_multiply(sigma_beta[p], R_chol[p]) * beta_pr[p];  
    beta[p][,1] = to_matrix(X[,1,D_start[p]:D_end[p]]) * gamma0[D_start[p]:D_end[p]] + to_vector(beta_tilde[p][1,]);
    beta[p][,2] = to_matrix(X[,1,D_start[p+4]:D_end[p+4]]) * gamma1[D_start[p+4]:D_end[p+4]] + to_vector(beta_tilde[p][2,]);
  }
  
  for (s in 1:S) {
    for (i in 1:N) {
      Arew[i,s] = Phi_approx(beta[1][i,1] + beta[1][i,2] * time[i,s] + sigma_theta[1] * theta[1][i,s]);
      Apun[i,s] = Phi_approx(beta[2][i,1] + beta[2][i,2] * time[i,s] + sigma_theta[2] * theta[2][i,s]);
      betaF[i,s] = beta[3][i,1] + beta[3][i,2] * time[i,s] + sigma_theta[3] * theta[3][i,s];
      betaP[i,s] = beta[4][i,1] + beta[4][i,2] * time[i,s] + sigma_theta[4] * theta[4][i,s];
    }  
  }

  R_Arew  = R_chol[1] * R_chol[1]';
  R_Apun  = R_chol[2] * R_chol[2]';
  R_betaF = R_chol[3] * R_chol[3]';
  R_betaP = R_chol[4] * R_chol[4]';

  { // local section, this saves time and space
    vector[4] ef;
    vector[4] ev;
    vector[4] pers;
    vector[4] utility;

    real ef_chosen;
    real PEval;
    real PEfreq;
    vector[4] PEfreq_fic;
    array[4] int choice_count;
    int card_count;
    int card_t;
    real outcome_t;
    real outcome_sign_t;

    for (i in 1:N) {   
      session = 0;
      for (t in 1:T) { 
        if (t == 1 || t == (T/2 + 1)) {
          card_count = 1;
          choice_count = {0,0,0,0};
          session += 1;
          ev = rep_vector(0,4);
          ef = rep_vector(0,4);
          pers = rep_vector(1,4);
          utility = ev + ef * betaF[i,session] + pers * betaP[i,session];  
        }
        card_t = card[card_count];
        card_sim[i,t] = card_t;
        choice[i,t] = categorical_rng(softmax(to_vector({utility[card[card_count]], 0})));   
        log_lik[i,t] = categorical_logit_lpmf(choice[i,t] | to_vector({utility[card[card_count]], 0}));   
        card_count +=1;
        
        if (choice[i,t]==1) {
          choice_count[card_t] += 1;
          outcome_t = outcome[card_t,choice_count[card_t]];
          outcome_sim[i,t] = outcome_t;
          if (outcome_t != 0) {
            outcome_sign_t = outcome_t > 0 ? 1 : -1;
          } else {
            outcome_sign_t = 0;
          }
          PEval      = outcome_t - ev[card_t];
          PEfreq     = outcome_sign_t - ef[card_t];   
          PEfreq_fic = -outcome_sign_t/3 - ef;
          ef_chosen  = ef[card_t];
            
          if (outcome_t >= 0) {
            ef = ef + Apun[i,session] * PEfreq_fic;
            ef[card_t] = ef_chosen + Arew[i,session] * PEfreq;
            ev[card_t] = ev[card_t] + Arew[i,session] * PEval;
          } else {
            ef = ef + Arew[i,session] * PEfreq_fic;
            ef[card_t] = ef_chosen + Apun[i,session] * PEfreq;
            ev[card_t] = ev[card_t] + Apun[i,session] * PEval;
          }
        } else {
          outcome_sim[i,t] = 0;
        }
        utility = ev + ef * betaF[i,session] + pers * betaP[i,session];
      }
    }
  }
}

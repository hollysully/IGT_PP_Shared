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
  array[N,T,D] real X;                       // person-level predictors
  array[4] int D_start;
  array[4] int D_end;
  
}


parameters {
  vector[D] gamma0;
  vector[D] gamma1;
  
  array[4] matrix[2,N] beta_pr;  
  array[4] vector<lower=0>[2] sigma_beta;
  array[4] cholesky_factor_corr[2] R_chol;
  
  array[4] vector[N] theta;
  array[4] real<lower=0> sigma_theta;
}

transformed parameters {
  array[4] matrix[2,N] beta_tilde;
  array[4] matrix[N,2] beta; 
  matrix[N,S] Arew;
  matrix[N,S] Apun;
  matrix[N,S] betaF;
  matrix[N,S] betaP;

  for (p in 1:4) {
    beta_tilde[p] = diag_pre_multiply(sigma_beta[p], R_chol[p]) * beta_pr[p];  
    beta[p][,1] = to_matrix(X[,1,D_start[p]:D_end[p]]) * gamma0[D_start[p]:D_end[p]] + to_vector(beta_tilde[p][1,]);
    beta[p][,2] = to_matrix(X[,1,D_start[p]:D_end[p]]) * gamma1[D_start[p]:D_end[p]] + to_vector(beta_tilde[p][2,]);
  }
  
  for (s in 1:S) {
    for (i in 1:N) {
      Arew[i,s] = Phi_approx(beta[1][i,1] + beta[1][i,2] * time[i,s] + sigma_theta[1] * theta[1][i]);
      Apun[i,s] = Phi_approx(beta[2][i,1] + beta[2][i,2] * time[i,s] + sigma_theta[2] * theta[2][i]);
      betaF[i,s] = beta[3][i,1] + beta[3][i,2] * time[i,s] + sigma_theta[3] * theta[3][i];
      betaP[i,s] = beta[4][i,1] + beta[4][i,2] * time[i,s] + sigma_theta[4] * theta[4][i];
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
    gamma0[p] ~ normal(0,1);
    gamma1[p] ~ normal(0,1);
    sigma_beta[p] ~ normal(0,1);
    theta[p] ~ normal(0,1);
    sigma_theta[p] ~ normal(0,1);
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
          }
        utility = ev + ef * betaF[i,session] + pers * betaP[i,session];  
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

// generated quantities {
//   // Hyper(group)-parameters - these are 5 (number of parameters) x S (number of sessions) matrix of mus & sigmas, respectively, for each parameter
//   // vector<lower=0,upper=1>[S] mu_Arew;
//   // vector<lower=0,upper=1>[S] mu_Apun;
//   // vector<lower=0,upper=5>[S] mu_K;
//   // vector[S] mu_betaF;
//   // vector[S] mu_betaP;
//   real log_lik[N];
// 
//   // For posterior predictive check
//   real choice_pred[N,T,S];
// 
//   // test-retest correlations
//   corr_matrix[S] R_Arew;
//   corr_matrix[S] R_Apun;
//   corr_matrix[S] R_betaF;
//   corr_matrix[S] R_betaP;
// 
//   // Reconstruct correlation matrix from cholesky factor
//     // Note that we're multipling the cholesky factor by its transpose which gives us the correlation matrix
//   R_Arew  = R_chol_Arew * R_chol_Arew'; //Phi_approx_corr_rng(mu_p[,1], sigma_Arew, R_chol_Arew * R_chol_Arew', 10000);
//   R_Apun  = R_chol_Apun * R_chol_Apun'; //Phi_approx_corr_rng(mu_p[,2], sigma_Apun, R_chol_Apun * R_chol_Apun', 10000);
//   R_betaF = R_chol_betaF * R_chol_betaF';
//   R_betaP = R_chol_betaP * R_chol_betaP';
// 
//   // Set all posterior predictions to -1 (avoids NULL values)
//   for (i in 1:N) {
//     for (s in 1:S) {
//       for (t in 1:T) {
//         choice_pred[i,t,s] = -1;
//       }
//     }
//   }
// 
//   // // Compute group-level means
//   // for (s in 1:S) {
//   //   mu_Arew[s] = Phi_approx_group_mean_rng(mu_p[s, 1], sigma_Arew[s], 10000);
//   //   mu_Apun[s] = Phi_approx_group_mean_rng(mu_p[s, 2], sigma_Apun[s], 10000);
//   //   mu_K[s] = Phi_approx_group_mean_rng(mu_p[s, 3], sigma_K[s], 10000) * 5;
//   //   mu_betaF[s] = mu_p[s, 4];
//   //   mu_betaP[s] = mu_p[s, 5];
//   // }
// 
//   { // local section, this saves time and space
//     // Declare variables to calculate utility after each trial: These 4 (number of cards) x 2 (playing vs. not playing) matrices
//     vector[4] ef;
//     vector[4] ev;
//     vector[4] pers;
//     vector[4] utility;
// 
//     real ef_chosen;
//     real PEval;
//     real PEfreq;
//     vector[4] PEfreq_fic;
// 
//     for (i in 1:N) {         // Loop through individual participants
//       log_lik[i] = 0;        // Initialize log_lik
// 
//       for (s in 1:S) {       // Loop though sessions for participant i
//         if (Tsubj[i,s] > 0) {    // If we have data for participant i on session s, run through RL algorithm
// 
//           // Initialize starting values
//           ev = rep_vector(0,4);
//           ef = rep_vector(0,4);
//           pers = rep_vector(1,4);
//           utility = ev + ef * betaF[i,s] + pers * betaP[i,s];
// 
//           for (t in 1:Tsubj[i,s]) { // Run through RL algorithm trial-by-trial
//             // softmax choice
//             log_lik[i] += categorical_logit_lpmf(choice[i,t,s]|to_vector({utility[card[i,t,s]], 0}));
// 
//             // Likelihood - predict choice as a function of utility
//             choice_pred[i,t,s] = categorical_rng(softmax(to_vector({utility[card[i,t,s]], 0})));
// 
//             if(choice[i,t,s]==1) {
//               // After choice, calculate prediction error
//               PEval      = outcome[i,t,s] - ev[card[i,t,s]];     // Value prediction error
//               PEfreq     = sign[i,t,s] - ef[card[i,t,s]];        // Win-frequency prediction error
//               PEfreq_fic = -sign[i,t,s]/3 - ef;
//               ef_chosen = ef[card[i,t,s]];
// 
//               if (outcome[i,t,s] >= 0) {  // If participant DID NOT lose
//                 // Update expected win-frequency of what participant DID NOT chose to do
//                 ef = ef + Apun[i,s] * PEfreq_fic;
//                 // Update what participant chose
//                 ef[card[i,t,s]] = ef_chosen + Arew[i,s] * PEfreq;
//                 ev[card[i,t,s]] = ev[card[i,t,s]] + Arew[i,s] * PEval;
//               } else { // If participant DID lose
//                 // Update expected win-frequency of what participant DID NOT choose to do
//                 ef = ef + Arew[i,s] * PEfreq_fic;
//                 // Update what participant chose
//                 ef[card[i,t,s]] = ef_chosen + Apun[i,s] * PEfreq;
//                 ev[card[i,t,s]] = ev[card[i,t,s]] + Apun[i,s] * PEval;
//               }
//             }
//             // Calculate expected value of card
//             utility = ev + ef * betaF[i,s] + pers * betaP[i,s];
//           }
//         }
//       }
//     }
//   }
// }

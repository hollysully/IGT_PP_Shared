functions {
  // to compute empirical correlation matrix given input matrix
  matrix empirical_correlation(matrix X) {
    int N = rows(X);             
    int M = cols(X);             
    matrix[N, M] centered;
    vector[M] std_devs;
    matrix[M, M] Sigma;
    matrix[M, M] R;
    
    // calculate covariance matrix
    for (j in 1:M) {
      centered[,j] = X[,j] - mean(X[,j]);
    }
    Sigma = (centered' * centered) / (N - 1);
    
    // Calculate correlation matrix
    std_devs = sqrt(diagonal(Sigma));
    for (i in 1:M) {
      for (j in 1:M) {
        R[i,j] = Sigma[i,j] / (std_devs[i] * std_devs[j]);
      }
    }
    
    return R;
  }
  
  real Phi_approx_group_mean_rng(real mu, real sigma, int n_samples) {
    vector[n_samples] par;
    for (i in 1:n_samples) {
      par[i] = Phi_approx(normal_rng(mu, sigma));
    }
    return mean(par);
  }
  
  matrix Phi_approx_corr_rng(vector mu, vector sigma, matrix R, int n_samples) {
    int N = rows(R);
    int M = cols(R);
    matrix[N,M] Sigma = quad_form_diag(R, sigma);
    matrix[n_samples, M] X_sim;
    for (i in 1:n_samples) {
      X_sim[i,] = Phi_approx(multi_normal_rng(mu, Sigma)');
  }
  return empirical_correlation(X_sim);
}
}

data {
  int<lower=1> N;                      // Number of participants
  int<lower=1> T;                      // Total possile number of trials
  int card[N,T];                       // Cards presented on each trial
  int choice[N,T];                     // Choices on each trial
  real outcome[N,T];                   // Outcomes received on each trial
  real sign[N,T];                      // Signs of the outcome received on each trial
}

parameters {
  // Declare parameters
  // Hyper(group)-parameters
  real mu_p[5];   // Vector of mus for the parameters
  
  real<lower=0> sigma_Arew;
  real<lower=0> sigma_Apun;
  real<lower=0> sigma_K;
  real<lower=0> sigma_betaF;
  real<lower=0> sigma_betaP;
  
  // Subject-level "raw" parameters - i.e., independent/uncorrelated & normally distributed person-level (random-)effects
  // Note, these are vectors for each parameter
  vector[N] Arew_pr;  
  vector[N] Apun_pr;  
  vector[N] K_pr;   
  vector[N] betaF_pr;
  vector[N] betaP_pr;
}

transformed parameters {
  // Declare transformed parameters
  // Parameters to use in the reinforcement-learning algorithm
  // Note, for each of these, we include the mus and the subject-level parameters (see below), allowing for shrinkage and subject-to-subject variability
  vector<lower=0,upper=1>[N] Arew; 
  vector<lower=0,upper=1>[N] Apun; 
  vector<lower=0,upper=5>[N] K;    
  vector[N] betaF;
  vector[N] betaP;
  
  // Calculate transformed parameters
  for (i in 1:N) {
    Arew[i] = Phi_approx(mu_p[1] + Arew_pr[i]);
    Apun[i] = Phi_approx(mu_p[2] + Apun_pr[i]);
    K[i]    = Phi_approx(mu_p[3] + K_pr[i]) * 5;
    betaF[i] = mu_p[4] + betaF_pr[i];
    betaP[i] = mu_p[5] + betaP_pr[i];
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
  real K_tr;
  
  // Priors
  // Hyperparameters for correlations
  sigma_Arew   ~ normal(0, 0.2);
  sigma_Apun   ~ normal(0, 0.2);
  sigma_K      ~ normal(0, 0.2);
  sigma_betaF  ~ cauchy(0, 1);
  sigma_betaP  ~ cauchy(0, 1);
  
  // Hyperparameters for RL learning algorithm
  mu_p[1]   ~ normal(0, 1);
  mu_p[2]   ~ normal(0, 1);
  mu_p[3]   ~ normal(0, 1);
  mu_p[4]   ~ normal(0, 1);
  mu_p[5]   ~ normal(0, 1);
  
  // Subject-level parameters
  Arew_pr  ~ normal(0, 1.0);
  Apun_pr  ~ normal(0, 1.0);
  K_pr     ~ normal(0, 1.0);
  betaF_pr ~ normal(0, 1.0);
  betaP_pr ~ normal(0, 1.0);
  
  for (i in 1:N) {         // Loop through individual participants
    // Initialize starting values
    K_tr = pow(3, K[i]) - 1;
    ev = rep_vector(0,4);
    ef = rep_vector(0,4);
    pers = rep_vector(0,4);
    utility = rep_vector(0,4);
    
    for (t in 1:T) { // Run through RL algorithm trial-by-trial
      
      // Likelihood - predict choice as a function of utility
      choice[i,t] ~ categorical_logit(to_vector({utility[card[i,t]], 0}));
      
      if(choice[i,t]==1){
        // After choice, calculate prediction error
        PEval      = outcome[i,t] - ev[card[i,t]];     // Value prediction error
        PEfreq     = sign[i,t] - ef[card[i,t]];        // Win-frequency prediction error
        PEfreq_fic = -sign[i,t]/3 - ef;
        ef_chosen  = ef[card[i,t]];
        
        if (outcome[i,t] >= 0) {  // If participant DID NOT lose
          // Update expected win-frequency of what participant DID NOT chose to do
          ef = ef + Apun[i] * PEfreq_fic;
          // Update what participant chose
          ef[card[i,t]] = ef_chosen + Arew[i] * PE
          
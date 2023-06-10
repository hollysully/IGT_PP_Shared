

data {
  int N;                // Number of participants for current scale
  int S;                // Number of sessions for current scale
  int missing[N,S]; // Matrix of missingness for current scale
  int score[N,S];       // Matrix of scores for current scale
  int M;                // Max score for current scale
}


parameters {
  real kappa;
  real<lower=0, upper=1> omega;
  vector<lower=0, upper=1>[N] theta; // specify theta as a vector with N values
}

model {
  // PRIORS
  // Hyperpriors
  kappa ~ gamma(2,.1);
  omega ~ beta(1,1);
  
  // Lower-level priors
  theta ~ beta(omega*kappa, (1-omega)*kappa);
  
  // LIKELIHOOD
  for(s in 1:S){
    for(i in 1:N){
      if(missing[i,s] == 0){
        score[i,s] ~ binomial(M, theta[i]);
      }
    }
  }
}















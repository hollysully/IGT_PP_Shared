library(dplyr)

make_stan_data <- function(task_data, survey_data, formula) {
  comb_data <- task_data %>% 
    left_join(survey_data, 
              by = c("ID", "session")) %>%
    arrange(ID, session, Trialorder)
  
  subj_list <- unique(comb_data$ID)
  n_subj <- length(subj_list)
  
  n_sessions <- length(unique(comb_data$session))
  t_subj <- array(0, c(n_subj, n_sessions)) 
  
  for (i in 1:n_subj)  {
    for (s in 1:n_sessions) {
      t_subj[i,s] <- sum(with(comb_data, ID==subj_list[i] & session==s))
    }
  }
  t_max <- max(t_subj) 
  
  # Behavioral data arrays
  choice <- outcome <- sign_outcome <- card <- array(-1, c(n_subj, t_max, n_sessions))
  X <- model.matrix(formula, comb_data)
  design_matrix <- array(0, c(n_subj, t_max, ncol(X), n_sessions))
  
  # Filling arrays with task and survey covariate data
  for (i in 1:n_subj) {
    subj_idx <- comb_data$Subject == subj_list[i]
    # sessions are "special" covariates because the model
    # initial conditions need reset each session start
    # regardless of the covariate model assumptions
    for (s in 1:n_sessions) {
      session_idx <- comb_data$session == s
      if (sum(subj_idx & session_idx) > 0) {
        design_matrix[i,,,s] <- X[subj_idx & session_idx,]
      }
      subj_dat <- comb_data %>% 
        filter(ID==subj_list[i] & session==s)
      if (nrow(subj_dat) > 0) {
        card[i,,s] <- subj_dat$card
        choice[i,,s] <- subj_dat$choice
        outcome[i,,s] <- subj_dat$outcome / 100
        sign_outcome[i,,s] <- sign(subj_dat$outcome)
      }
    }
  }  
  
  stan_list <- list(
    N = n_subj,
    T = t_max,
    S = n_sessions,
    D = ncol(X),
    Tsubj = t_subj,
    card = card,
    outcome = outcome,     
    sign = sign_outcome,
    choice = choice,
    X = design_matrix,
    subj_list = subj_list
  )
  return(stan_list)
}
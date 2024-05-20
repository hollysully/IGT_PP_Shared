library(dplyr)

PARAMETERS <- c("Arew", "Apun", "betaF", "betaP")

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
  
  # parsed list of formulas
  named_formulas <- parse_formula(formula)
  
  # Behavioral data arrays
  choice <- outcome <- sign_outcome <- card <- array(-1, c(n_subj, t_max, n_sessions))
  # create model matrix for each formula in list_formula
  X <- lapply(named_formulas, function(f) model.matrix(f, comb_data))
  design_matrix <- lapply(X, function(x) array(0, c(n_subj, t_max, ncol(x), n_sessions)))
  
  # Filling arrays with task and survey covariate data
  for (i in 1:n_subj) {
    subj_idx <- comb_data$Subject == subj_list[i]
    # sessions are "special" covariates because the model
    # initial conditions need reset each session start
    # regardless of the covariate model assumptions
    for (s in 1:n_sessions) {
      session_idx <- comb_data$session == s
      if (sum(subj_idx & session_idx) > 0) {
        for (par in PARAMETERS) {
          design_matrix[[par]][i,,,s] <- X[[par]][subj_idx & session_idx,]  
        }
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
    D = ncol(X[[1]]),
    Tsubj = t_subj,
    card = card,
    outcome = outcome,     
    sign = sign_outcome,
    choice = choice,
    X_Arew = design_matrix$Arew,
    X_Apun = design_matrix$Apun,
    X_betaF = design_matrix$betaF,
    X_betaP = design_matrix$betaP,
    subj_list = subj_list
  )
  return(stan_list)
}

make_stan_data_growth <- function(task_data, survey_data, formula, time_variable) {
  comb_data <- task_data %>% 
    left_join(survey_data, 
              by = c("ID", "session")) %>%
    arrange(ID, session, Trialorder)
  
  comb_data$SessionTime
  
  subj_list <- unique(comb_data$ID)
  
  # trials per subject
  n_subj <- length(subj_list)
  t_subj <- array(0, c(n_subj)) 
  for (i in 1:n_subj)  {
    t_subj[i] <- sum(comb_data$ID==subj_list[i])
  }
  t_max <- max(t_subj) 
    
  # new session start trial markers
  session_start <- array(0, c(n_subj, t_max)) 
  for (i in 1:n_subj) {
    subj_trials <- subset(comb_data, ID==subj_list[i])$Trialorder
    for (t in 1:t_subj[i]) {
      if (subj_trials[t] == 1) {
        session_start[i,t] <- 1
      }
    }
  }
  
  # time variable in model
  n_sessions <- length(unique(comb_data$session))
  time <- array(0, c(n_subj, n_sessions)) 
  for (i in 1:n_subj) {
    for (s in 1:n_sessions) {
      subj_session <- subset(comb_data, ID==subj_list[i] & session==s)
      if (nrow(subj_session) > 0) {
        time[i,s] <- as.integer(unique(subj_session[time_variable])[1])
        time[time[1:n_subj,s]==0,s] <- mean(time[time[1:n_subj,s]!=0,s])
      }
    }
  }
  time <- time - min(time)
  
  # parsed list of formulas
  named_formulas <- parse_formula(formula)
  
  # Behavioral data arrays
  choice <- outcome <- sign_outcome <- card <- array(-1, c(n_subj, t_max))
  # create model matrix for each formula in list_formula
  X <- lapply(named_formulas, function(f) model.matrix(f, comb_data))
  D_end <- cumsum(sapply(X, ncol))
  D <- D_end[length(D_end)]
  D_start <- c(1, D_end[2:length(D_end)]-1)
  names(D_start) <- names(D_end)
  design_matrix <- array(0, c(n_subj, t_max, D))
  
  # Filling arrays with task and survey covariate data
  for (i in 1:n_subj) {
    subj_idx <- comb_data$Subject == subj_list[i]
    # sessions are "special" covariates because the model
    # initial conditions need reset each session start
    # regardless of the covariate model assumptions
    if (sum(subj_idx) > 0) {
      for (par in PARAMETERS) {
        design_matrix[i,,D_start[par]:D_end[par]] <- X[[par]][subj_idx,1:ncol(X[[par]])]  
      }
      subj_dat <- comb_data %>% 
        filter(ID==subj_list[i])
      if (nrow(subj_dat) > 0) {
        card[i,1:t_subj[i]] <- subj_dat$card
        choice[i,1:t_subj[i]] <- subj_dat$choice
        outcome[i,1:t_subj[i]] <- subj_dat$outcome / 100
        sign_outcome[i,1:t_subj[i]] <- sign(subj_dat$outcome)
      }
    }
  }  
  
  stan_list <- list(
    N = n_subj,
    T = t_max,
    S = n_sessions,
    D = D,
    D_start = D_start,
    D_end = D_end,
    Tsubj = t_subj,
    session_start = session_start,
    time = time,
    card = card,
    outcome = outcome,     
    sign = sign_outcome,
    choice = choice,
    X = design_matrix,
    subj_list = subj_list
  )
  return(stan_list)
}

# parse text into a list of formulas
parse_formula <- function(text) {
  # clean up the text
  text <- gsub("\n", "", text)
  text <- gsub(" ", "", text)
  text <- gsub("~", " ~ ", text)
  text <- gsub(":", " : ", text)
  text <- gsub(";", " ; ", text)
  
  # rm empty strings
  list_formulas <- strsplit(text, ";")[[1]] %>% 
    .[. != ""] %>% 
    .[. != " "]
  
  # check that the formulas are well formed
  for (f in list_formulas) {
    if (length(strsplit(f, " ~ ")[[1]]) != 2) {
      stop("Formulas must be of the form 'lhs ~ rhs'")
    }
  }
  
  # check that the formula lhs is allowed 
  allowed_lhs <- PARAMETERS
  for (f in list_formulas) {
    lhs <- gsub(" ", "", strsplit(f, " ~ ")[[1]][1])
    if (!(lhs %in% allowed_lhs)) {
      stop(paste0("lhs '", lhs, "' not allowed"))
    }
  }
  # return named list where lhs is the name and rhs is the formula object
  formula_sides <- lapply(list_formulas, function(l) gsub(" ", "", strsplit(l, " ~ ")[[1]]))
  named_formulas <- list()
  for (f in formula_sides) {
    named_formulas[[f[1]]] <- as.formula(paste0(" ~ ", f[2]))
  }
  return(named_formulas)
}


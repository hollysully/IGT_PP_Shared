library(dplyr)
library(posterior)

PARAMETERS <- c("Arew", "Apun", "betaF", "betaP")

PARAMETERS_GROWTH <- c(
  "Arew_int", "Apun_int", "betaF_int", "betaP_int",
  "Arew_slope", "Apun_slope", "betaF_slope", "betaP_slope"
)

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
  named_formulas <- parse_formula(formula, PARAMETERS)
  
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



make_stan_data_growth <- function(task_data, survey_data, formula, time_variable, scale_covars=T) {
  comb_data <- task_data %>% 
    left_join(survey_data, 
              by = c("ID", "session")) %>%
    arrange(ID, session, Trialorder)
  
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
        # fill in missing times with mean of non-missing
        time[time[1:n_subj,s]==0,s] <- mean(time[time[1:n_subj,s]!=0,s])
      }
    }
  }
  time <- time - min(time)
  
  # parsed list of formulas
  named_formulas <- parse_formula(formula, PARAMETERS_GROWTH)
  
  # Behavioral data arrays
  choice <- outcome <- sign_outcome <- card <- array(-1, c(n_subj, t_max))
  # summarize data to get covariate values per ID, session
  covar_data <- comb_data %>%
    group_by(ID, session) %>% 
    summarize(across(where(is.numeric), mean))
  # create model matrix for each formula in list_formula
  X <- lapply(named_formulas, function(f) model.matrix(f, covar_data))
  D_end <- cumsum(sapply(X, ncol))
  D <- D_end[length(D_end)]
  D_start <- c(1, D_end[-length(D_end)] + 1)
  names(D_start) <- names(D_end)
  design_matrix <- array(-99, c(n_subj, n_sessions, D))
  
  # Filling arrays with task and survey covariate data
  for (i in 1:n_subj) {
    subj_dat <- comb_data %>% 
      filter(ID==subj_list[i])
    n_session_subj <- length(unique(subj_dat$session))
    
    if (nrow(subj_dat) > 0) {
      card[i,1:t_subj[i]] <- subj_dat$card
      choice[i,1:t_subj[i]] <- subj_dat$choice
      outcome[i,1:t_subj[i]] <- subj_dat$outcome / 100
      sign_outcome[i,1:t_subj[i]] <- sign(subj_dat$outcome)
      for (par in PARAMETERS_GROWTH) {
        for (s in 1:n_session_subj) {
          subj_covar_idx <- covar_data$ID==subj_list[i] & covar_data$session==s
          design_matrix[i,s,D_start[par]:D_end[par]] <- X[[par]][subj_covar_idx]
        }
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
parse_formula <- function(text, parameters) {
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
  for (f in list_formulas) {
    lhs <- gsub(" ", "", strsplit(f, " ~ ")[[1]][1])
    if (!(lhs %in% parameters)) {
      stop(paste0("lhs '", lhs, "' not allowed"))
    }
  }
  # return named list where lhs is the name and rhs is the formula object
  formula_sides <- lapply(list_formulas, function(l) gsub(" ", "", strsplit(l, " ~ ")[[1]]))
  named_formulas <- list()
  for (f in formula_sides) {
    named_formulas[[f[1]]] <- as.formula(paste0(" ~ ", f[2]))
  }
  sorted_formulas <- named_formulas[parameters] 
  if (!all(parameters %in% names(sorted_formulas))) {
    stop(
      paste0(
        "Must specify all of ", paste(parameters, collapse=", "), ". Only ", 
        paste(names(sorted_formulas), collapse=", "), " were specified."
      )
    )
  }
  return(sorted_formulas)
}

par_from_draws <- function(fit, par) {
  rvars_pars <- as_draws_rvars(
    fit$draws(
      c(par)
    )
  )
  return(lapply(rvars_pars, draws_of))
}

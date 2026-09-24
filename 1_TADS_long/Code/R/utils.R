library(dplyr)
library(posterior)

PARAMETERS_GROWTH <- c(
  "Arew_int", "Apun_int", "betaF_int", "betaP_int",
  "Arew_slope", "Apun_slope", "betaF_slope", "betaP_slope"
)

make_stan_data_growth <- function(task_data, survey_data, model_text, time_variable) {
  comb_data <- task_data %>% 
    left_join(survey_data, 
              by = c("ID", "session", "participant", "participant0ID")) %>%
    arrange(ID, session, task_trial)
  
  survey_columns <- setdiff(names(survey_data), names(task_data))
  
  subj_has_survey <- comb_data %>%
    filter(session == 1) %>%
    group_by(participant0ID) %>%
    summarize(survey = all(!is.na(across(all_of(survey_columns))))) %>%
    filter(survey) %>%
    {.$participant0ID}
  
  subj_has_task_sessions <- comb_data %>%
    group_by(participant0ID) %>%
    summarize(n_sessions = length(unique(session))) %>%
    ungroup() %>%
    filter(n_sessions >= 1) %>%
    {.$participant0ID}
  
  comb_data <- comb_data %>%
    filter(participant0ID %in% subj_has_survey) %>%
    filter(participant0ID %in% subj_has_task_sessions)
  
  subj_list <- unique(comb_data$participant0ID)
  
  # trials per subject
  n_subj <- length(subj_list)
  t_subj <- array(0, c(n_subj)) 
  for (i in 1:n_subj)  {
    t_subj[i] <- sum(comb_data$participant0ID==subj_list[i])
  }
  t_max <- max(t_subj) 
  
  # new session start trial markers
  session_start <- array(0, c(n_subj, t_max)) 
  for (i in 1:n_subj) {
    subj_trials <- subset(comb_data, participant0ID==subj_list[i])$task_trial
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
      subj_session <- subset(comb_data, participant0ID==subj_list[i] & session==s)
      if (nrow(subj_session) > 0) {
        time[i,s] <- as.integer(unique(subj_session[time_variable])[1])
        # fill in missing times with mean of non-missing
        time[time[1:n_subj,s]==0,s] <- mean(time[time[1:n_subj,s]!=0,s])
      }
    }
  }
  time <- time - min(time)
  
  # parsed list of formulas
  named_formulas <- parse_formula(model_text, PARAMETERS_GROWTH)
  
  # Behavioral data arrays
  choice <- outcome <- sign_outcome <- card <- array(-1, c(n_subj, t_max))
  # # summarize data to get covariate values per ID, session
  # covar_data <- comb_data %>%
  #   group_by(participant0ID, session) %>% 
  #   summarize(across(where(is.numeric), mean))
  # TODO: validate—carry forward last value if covar is missing for a session
  covar_data <- comb_data %>%
    group_by(participant0ID, session) %>%
    summarize(
      across(where(is.numeric), mean, na.rm = TRUE), 
      across(where(is.character), ~ first(na.omit(.))),
      .groups = "drop"
    ) %>%
    # Expand to include all participant0ID-session combinations, filling missing sessions with NA
    complete(participant0ID, session) %>%
    arrange(participant0ID, session) %>%
    # Apply last observation carried forward within each participant group
    group_by(participant0ID) %>%
    fill(everything(), .direction = "down") %>%
    ungroup()
  
  # create model matrix for each formula in list_formula
  X <- lapply(named_formulas, function(f) model.matrix.lm(f, covar_data, na.action="na.pass"))
  D_end <- cumsum(sapply(X, ncol))
  D <- D_end[length(D_end)]
  D_start <- c(1, D_end[-length(D_end)] + 1)
  names(D_start) <- names(D_end)
  design_matrix <- array(-99, c(n_subj, n_sessions, D))
  
  # Filling arrays with task and survey covariate data
  for (i in 1:n_subj) {
    subj_dat <- comb_data %>% 
      filter(participant0ID==subj_list[i])
    n_session_subj <- length(unique(subj_dat$session))
    
    if (nrow(subj_dat) > 0) {
      card[i,1:t_subj[i]] <- subj_dat$card
      choice[i,1:t_subj[i]] <- 2-subj_dat$play
      outcome[i,1:t_subj[i]] <- subj_dat$outcome / 100
      sign_outcome[i,1:t_subj[i]] <- sign(subj_dat$outcome)
      for (par in PARAMETERS_GROWTH) {
        for (s in 1:n_session_subj) {
          subj_covar_idx <- covar_data$participant0ID==subj_list[i] & covar_data$session==s
          if (any(subj_covar_idx)) {
            design_matrix[i,s,D_start[par]:D_end[par]] <- X[[par]][subj_covar_idx] 
          }
        }
      }
    }
  }  
  design_matrix[is.na(design_matrix)] <- -99
  
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

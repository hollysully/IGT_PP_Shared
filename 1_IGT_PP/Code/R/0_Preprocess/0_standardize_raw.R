library(dplyr)
library(tidyverse) 
library(haven)
library(readxl)
library(here)
library(foreach)

# dataset to get DOB for each subject
dob <- read_sav(here("1_IGT_PP", "Data", "0_Raw", "MergedQuest_3.21.16-Session1.sav")) %>%
  select(ID, DOB)

AB <- read_excel(here("1_IGT_PP", "Data", "0_Raw", "AB", "Merged_AB.xlsx")) 
BD <- read_excel(here("1_IGT_PP", "Data", "0_Raw", "BD", "Merged_BD.xlsx")) 

all_task_data <- rbind(AB, BD) %>%
  filter(Procedure!="Knowledge") %>%
  group_by(Subject) %>%
  mutate(
    ID = Subject,
    card = dplyr::recode(cardname, 'A' = 1, 'B' = 2, 'C' = 3, 'D' = 4), 
    choice = dplyr::recode(card.RESP, `1` = 1, `3` = 2, .missing = 2), 
    time = SessionDate,
    wave = as.integer(as.factor(SessionDate)),
    outcome = ifelse(is.na(absmoney1), 0, absmoney1)
  )

#Note: PlayPass IGT task subject numbers are 2049—2099

# import questionnaire data
Sess1_Q <- read_sav(here("1_IGT_PP", "Data", "0_Raw", "MergedQuest_3.21.16-Session1.sav"))
Sess2_Q <- read_sav(here("1_IGT_PP", "Data", "0_Raw", "MergedQuest_3.21.16-Session2.sav"))

all_survey_data <- Sess1_Q %>%
  full_join(Sess2_Q)


make_stan_data <- function(task_data, survey_data, covariates = c(), cumulative=TRUE) {
  subj_list <- unique(task_data$ID)
  n_subj <- length(subj_list)
  n_covariates <- length(covariates)
  
  n_sessions <- length(unique(task_data$wave))
  t_subj <- array(0, c(n_subj, n_sessions)) 
  
  for (i in 1:n_subj)  {
    for (s in 1:n_sessions) {
      Tsubj[i,s] <- sum(with(task_data, ID==subj_list[i] & wave==s))
    }
  }
  t_max <- max(t_subj) 
  
  # Behavioral data arrays
  choice <- outcome <- sign_outcome <- card <- array(-1, c(n_subj, t_max, n_sessions))
  design_matrix <- array(0, c(n_subj, n_covariates + n_sessions, n_sessions))
  
  # fill out design matrix for "session effect" covariates
  for (s in 1:n_sessions) {
    # first column always intercept
    if (s == 1) {
      design_matrix[,s,] <- 1
    } else {
      # cumulative or relative to intercept change
      if (cumulative) {
        design_matrix[,1:s,s] <- 1  
      } else {
        design_matrix[,s,s] <- 1  
      }
    }
  }
  
  # Filling arrays with task and survey covariate data
  for (i in 1:n_subj) {
    for (s in 1:n_sessions) {
      subj_task_dat <- task_data %>% 
        filter(ID==subj_list[i] & wave==s)
      if (nrow(subj_task_dat) > 0) {
        card[i,] <- subj_task_dat$card
        choice[i,] <- subj_task_dat$choice
        outcome[i,] <- subj_task_dat$outcome / 100
        sign_outcome[i,] <- sign(subj_task_dat$outcome) 
      }
    }
  }  
  
  stan_list <- list(
    N = n_subj,
    T = t_max,
    Tsubj = t_subj,
    card = card,
    outcome = outcome,     
    sign = sign_outcome,
    choice = choice,
    subj_list = subj_list
  )
  return(stan_list)
}

saveRDS(stan_dat, file = here("1_IGT_PP", "Data", "1_Preprocessed", "stan_ready_ORL_IGT.rds"))

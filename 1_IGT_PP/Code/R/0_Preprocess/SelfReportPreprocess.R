


library(dplyr)
library(tidyverse) 
library(haven)
library(lubridate)
library(readxl)
library(here)



# -----------------------------------------------
# MEASURES
# * Demographics
#   + ID, sex, race, Ethnicity
# 
# * Behavioral Activation Scale
#   + bastot    = ?
#     + basdrive  = Drive subscale: measures persistent pursuit of goals
#     + basfunsk  = Fun Seeking subscale: measures desire for and approach towards new rewards
#     + basrewres = Reward Responsiveness subscale: measures positive responses to the anticipation or consummation of rewards
#   
# * Behavioral Inhibition Scale: measures sensitivity to negatively-valenced events
#   + bis
#   
# * Positive Affect/Negative Affect Schedules: measures state-level mood
#   + panas_pa = Positive Affect
#   + panas_na = Negative Affect
#   
# * Mood and Anxiety Symptom Questionnaire (MASQ): assesses internalizing symptoms experienced in past week
#   + masqGDA = General Distress Anxiety subscale: measures general anxious mood
#   + masqAA  = Anxious Arousal subscale: measures somatic hyperarousal
#   + masqGDD = General Distress Depression subscale: measures general depressed mood
#   + masqAD  = Anhedonic Depression subscale: measures low positive affect
#   
# * Snaith-Hamilton Pleasure Scale: measures ability to experience pleasure in last few days
#   + shaps_tot
#   
# * Patient-Reported Outcomes Measurement Information System: assesses symptoms of depression experienced in past week
#   + prdep_tot

  
  
# -----------------------------------------------
## IMPORT DATA
session1 = read_sav(here("1_IGT_PP", "Data", "0_Raw", "MergedQuest_3.21.16-Session1.sav"),
                    col_select = c("ID", "sex", "race", "Ethnicity",
                                   "bastot", "basdrive", "basfunsk",
                                   "basrewres", "bis", "panas_pa",
                                   "panas_na", "masqGDA", "masqAA", "masqGDD",
                                   "masqAD", "shaps_tot", "prdep_tot")) %>% 
  mutate(session = 1) %>%                     # Create session variable
  filter(ID >= 2049,                          # Subset PP participants
         ID != 2059,                          # Remove participant that played old IFT
         ID != 2083)                          # Remove participant that had no task data

session2 = read_sav(here("1_IGT_PP", "Data", "0_Raw", "MergedQuest_3.21.16-Session2.sav"),
                    col_select = c("ID", "bastot", "basdrive", "basfunsk",
                                   "basrewres", "bis", "panas_pa",
                                   "panas_na", "shaps_tot")) %>% 
  mutate(session = 2) %>%                     # Create session variable
  filter(ID >= 2049,                          # Subset PP participants
         ID != 2059,                          # Remove participant that played old IFT
         ID != 2083,                          # Remove participant that had no task data
         !ID %in% c(2057, 2063, 2064,         # Remove participants that didn't do session 2
                    2067, 2086, 2090,
                    2093, 2094, 2096, 2098))

both_sessions = bind_rows(session1, session2) %>% 
  select(ID, session,
         # Session 1 only
         prdep_tot, masqGDA, masqAA, masqGDD, masqAD,
         # Both sessions
         bastot, basdrive, basfunsk, basrewres,
         bis, panas_pa, panas_na, shaps_tot,
         # Remove demographics
         -c(sex, race, Ethnicity)) %>% 
  pivot_longer(cols = -c("ID", "session"), values_to = "score", names_to = "scale")

  
  
# -----------------------------------------------
# PREP DATA
IDs = unique(both_sessions$ID)
N = length(IDs)

# Make dataframe describing each scale
scales = data.frame(scale = c(# Session 1 only
                              "prdep_tot", "masqGDA", "masqAA", "masqGDD", "masqAD",
                              # Both Sessions
                              "bastot", "basdrive", "basfunsk", "basrewres",
                              "bis", "panas_pa", "panas_na", "shaps_tot"),
                    n_items = c(28, 11, 17, 12, NA_real_, 13, 4, 4, 5, 7, 10, 10, 14),
                    min_item_score = c(0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1),
                    max_item_score = c(4, 5, 5, 5, NA_real_, 4, 4, 4, 4, 4, 5, 5, 4),
                    n_sessions = c(1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2)) %>% 
  mutate(min = n_items * min_item_score,
         max = n_items * max_item_score,
         scaled_max = max - min)

# Make scaled scores and check out which scores are below the minimums - for those scores, we'll truncate them to 0
full_data = both_sessions %>% 
  left_join(scales) %>% 
  mutate(scaled_score = case_when(score-min < 0 ~ 0, is.numeric(score) ~ score - min),
         confirm = score > max | score < min)


# Create lists to store data in for each scale
scores = list()
missingness = list()
maxes = list()

for(l in 1:length(scales$scale)){
  # Create matrices for the current scale in each list
  scores[[scales$scale[l]]] = matrix(data = -999, nrow = N, ncol = scales$n_sessions[l])
  missingness[[scales$scale[l]]] = matrix(data = 1, nrow = N, ncol = scales$n_sessions[l])
  
  maxes[[scales$scale[l]]] = filter(scales, scale == scales$scale[l])[,"scaled_max"] # Save max score for current scale
  
  cur_scale = filter(full_data, scale == scales$scale[l]) # Subset scores from scale of interest
  
  for(s in 1:scales$n_sessions[l]){
    cur_session = filter(cur_scale, session == s) # Subset scores from session of interest
    
    for(i in 1:N){
      cur_obs = filter(cur_session, ID == IDs[i]) # Subset score from person of interest
      
      if(length(cur_obs$scaled_score)){ # Check if data is missing, if not, then get data
        scores[[scales$scale[l]]][i,s] = cur_obs$scaled_score # Save score in element of matrix within the scale's score matrix
        missingness[[scales$scale[l]]][i,s] = 0 # Save data as not missing in element of matrix within the scale's missingness matrix
      }
    }
  }
}


stan_datas = list()

for(scale in unique(scales$scale)){
  stan_datas[[scale]] = list(N = nrow(scores[[scale]]),       # Number of participants for current scale
                             S = ncol(scores[[scale]]),       # Number of sessions for current scale
                             missing = missingness[[scale]],  # Matrix of missingness for current scale
                             score = scores[[scale]],         # Matrix of scores for current scale
                             M = maxes[[scale]])              # Max score for current scale
}


saveRDS(stan_datas, file = here("1_IGT_PP", "Data", "1_Preprocessed", "stan_ready_binomial_selfreport.rds"))
















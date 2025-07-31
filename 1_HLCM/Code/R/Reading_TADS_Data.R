library(here)
library(tidyverse)

TADS_12345_Full_IGT = readRDS(here("1_HLCM", "Data", "TADS_12345_Full_IGT.RDS"))
TADS_12345_BIS_BAS = readRDS(here("1_HLCM", "Data", "0_Raw_TADS", "Full_BIS_BAS.RDS")) %>% 
  select(-date) %>% 
  filter(participant != "child")


TADS_12345_Parent_IGT = filter(TADS_12345_Full_IGT, participant != "child") %>% 
  mutate(session = as.numeric(str_remove(time, "T")),
         participant0ID = case_when(participant == "dad" ~ as.numeric(paste0("20", id)),
                                    participant == "mom" ~ as.numeric(paste0("30", id))),
         date = as.Date(date, format = "%m-%d-%Y"),
         card_letter = str_remove(AB_card, "AB_"),
         card = match(card_letter, LETTERS[1:4]),
         outcome = case_when(is.na(outcome) ~ 0, T ~ outcome)) %>%
  group_by(participant0ID, date, session, card_letter) %>% 
  mutate(card_trial = rank(trial)) %>% 
  select(participant0ID, ID = id, session, date, participant, platform,
         card_letter, card, task_trial = trial, card_trial, 
         good_card = card_good, play = card_play, outcome)

saveRDS(TADS_12345_Parent_IGT, here("1_HLCM", "Data", "TADS_12345_IGT.RDS"))
saveRDS(TADS_12345_BIS_BAS, here("1_HLCM", "Data", "TADS_12345_BIS_BAS.RDS"))

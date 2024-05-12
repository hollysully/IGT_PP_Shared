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
    session = as.factor(as.integer(as.factor(SessionDate))),
    outcome = ifelse(is.na(absmoney1), 0, absmoney1)
  ) %>%
  filter(ID>=2049 & ID<=2099)

#Note: PlayPass IGT task subject numbers are 2049—2099

# import questionnaire data
Sess1_Q <- read_sav(here("1_IGT_PP", "Data", "0_Raw", "MergedQuest_3.21.16-Session1.sav"))
Sess2_Q <- read_sav(here("1_IGT_PP", "Data", "0_Raw", "MergedQuest_3.21.16-Session2.sav"))

all_survey_data <- Sess1_Q %>%
  mutate(session = 1) %>%
  full_join(Sess2_Q %>% mutate(session = 2)) %>%
  group_by(ID) %>%
  mutate(session = as.factor(session))
  

saveRDS(all_task_data, here("1_IGT_PP", "Data", "1_Preprocessed", "all_task_data.rds"))
saveRDS(all_survey_data, here("1_IGT_PP", "Data", "1_Preprocessed", "all_survey_data.rds"))

# test_data <- all_task_data %>%
#   filter(Subject == all_task_data$Subject[1]) %>%
#   left_join(all_survey_data %>%
#               filter(ID == all_task_data$Subject[1]),
#             by = c("session"))
# 
# 
# ff <- ~ session
# utils::str(m <- model.frame(ff, test_data))
# mat <- model.matrix(ff, m)
# 
# formula <- log(Volume) ~ log(Height) + log(Girth)
# model.matrix(formula, data.frame(Volume = c(1.0, 2.0, 3.0), Height = c(3.0, 4.0, 5.0), Girth = c(5.0, 4.0, 3.0)))
# 

# saveRDS(stan_dat, file = here("1_IGT_PP", "Data", "1_Preprocessed", "stan_ready_ORL_IGT.rds"))

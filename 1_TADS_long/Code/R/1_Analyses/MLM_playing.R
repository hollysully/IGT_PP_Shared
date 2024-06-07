library(lme4)
#install.packages("lme4")
#install.packages("psych")
library(psych)
#install.packages("lmerTest")
library("lmerTest")

# lme4
lmer(Arew ~  + (1 | id), data=TADS_child_long_IGT)

# lme4
Arew <- lmer(Arew_qnormed ~ age_centered + (age_centered | id), data=TADS_child_long_IGT)

summary(Arew)






# lme4
betaB <- lmer(betaB ~ age_centered + (age_centered | id), data=data)

summary(betaB)


describe(TADS_child_long_IGT)



1 unit change in age = .09 unit change in betaB
can include sex, family history and their interactions with age



# lme4
betaB_hx <- lmer(betaB ~ age_centered + MatHxDepress + MatHxDepress * age_centered + (age_centered | id), data=TADS_child_long_IGT)

summary(betaB_hx)



# lme4
Arew_qnormed_hx <- lmer(Arew_qnormed ~ age_centered + MatHxDepress + MatHxDepress * age_centered + (age_centered | id), data=TADS_child_long_IGT)

summary(Arew_qnormed_hx)



# lme4
Arew_qnormed_hx <- lmer(Arew_qnormed ~ age_centered + MatHxDepress + MatHxDepress * age_centered + (age_centered | id) + (1 | id), data=TADS_child_long_IGT)

summary(Arew_qnormed_hx)


# lme4
Apun_qnormed_hx <- lmer(Apun_qnormed ~ age_centered + MatHxDepress + MatHxDepress * age_centered + (age_centered | id), data=TADS_child_long_IGT)

summary(Apun_qnormed_hx)



# lme4
Apun_hx <- lmer(Apun ~ age_centered + MatHxDepress + MatHxDepress * age_centered + (age_centered | id), data=TADS_child_long_IGT)

summary(Apun_hx)



# lme4
betaF_hx <- lmer(betaF ~ age_centered + MatHxDepress + MatHxDepress * age_centered + (age_centered | id), data=TADS_child_long_IGT)

summary(betaF_hx)


# lme4
betaB_hx <- lmer(betaB ~ age_centered + MatHxDepress + MatHxDepress * age_centered + (age_centered | id), data=TADS_child_long_IGT)

summary(betaB_hx)


# lme4
betaF <- lmer(betaF ~ age_centered + (age_centered | id), data=TADS_child_long_IGT)

summary(betaF)


# lme4
betaF_hx <- lmer(betaF ~ age_centered + MatHxDepress + MatHxDepress * age_centered + (age_centered | id), data=TADS_child_long_IGT)

summary(betaF_hx)



betaB <- lmer(betaB ~ age_centered + child_sex + child_sex * age_centered + (age_centered | id), data=TADS_child_long_IGT)

summary(betaB)



betaF_hx <- lmer(betaF ~ age_centered*MatHxDepress + (1 | id ) + (0 + age_centered | id), data=TADS_child_long_IGT)
summary(betaF_hx)

Arew_qnormed_hx <- lmer(Arew_qnormed ~ age_centered*MatHxDepress + (1 | id ) + (0 + age_centered | id), data=TADS_child_long_IGT)
summary(Arew_qnormed_hx)


Apun_qnormed_hx <- lmer(Apun_qnormed ~ age_centered*MatHxDepress + (1 | id ) + (0 + age_centered | id), data=TADS_child_long_IGT)
summary(Apun_qnormed_hx)

betaF_hx <- lmer(betaF ~ age_centered*MatHxDepress + (1 | id ) + (0 + age_centered | id), data=TADS_child_long_IGT)
summary(betaF_hx)

betaB_hx <- lmer(betaB ~ age_centered*MatHxDepress + (1 | id ) + (0 + age_centered | id), data=TADS_child_long_IGT)
summary(betaB_hx)



make new variable 'baseline age' that is the T1 age 
include baseline age as a single predictor and as an interaction w/ age 



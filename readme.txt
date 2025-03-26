my thesis yuju

setwd("~/Documents/GitHub/Tesis/irace")
scenario <- library(irace)
scenario <- readScenario("scenario.txt")
irace_main(scenario=scenario)
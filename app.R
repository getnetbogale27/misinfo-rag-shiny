# Main Shiny entrypoint.
# Loads shared setup, UI definition, and server logic.

library(shiny)

source("global.R")
source("R/ui.R")
source("R/server.R")

shinyApp(ui = ui, server = server)

# Server logic for the misinformation detection MVP.

library(shiny)
source("R/services/rag_api.R")

server <- function(input, output, session) {
  result <- reactiveVal(NULL)

  observeEvent(input$analyze, {
    req(input$claim)

    # Call backend API and store result.
    api_result <- tryCatch(
      analyze_claim(input$claim),
      error = function(e) {
        list(
          verdict = "Error",
          confidence = NA,
          explanation = paste("Failed to reach API:", e$message),
          evidence = c("Make sure FastAPI is running on http://localhost:8000")
        )
      }
    )

    result(api_result)
  })

  output$verdict <- renderText({
    req(result())
    result()$verdict
  })

  output$confidence <- renderText({
    req(result())
    conf <- result()$confidence
    if (is.null(conf) || is.na(conf)) {
      "N/A"
    } else {
      sprintf("%.2f", as.numeric(conf))
    }
  })

  output$explanation <- renderText({
    req(result())
    result()$explanation
  })

  output$evidence <- renderUI({
    req(result())
    evidence_items <- result()$evidence
    if (is.null(evidence_items)) evidence_items <- character(0)
    tags$ul(lapply(evidence_items, tags$li))
  })
}

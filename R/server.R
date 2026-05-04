# Server logic for the misinformation detection MVP.

library(shiny)
source("R/services/rag_api.R")

server <- function(input, output, session) {
  result <- reactiveVal(NULL)

  evaluation_result <- reactiveVal(NULL)

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


  observeEvent(input$run_evaluation, {
    metrics <- tryCatch(
      run_model_evaluation("data/evaluation_dataset.json"),
      error = function(e) {
        list(error = paste("Evaluation failed:", e$message))
      }
    )
    evaluation_result(metrics)
  })

  output$language <- renderText({
    req(result())
    lang <- result()$language
    if (is.null(lang) || !nzchar(lang)) {
      "N/A"
    } else if (lang == "am") {
      "Amharic"
    } else {
      "English"
    }
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

  output$eval_accuracy <- renderText({
    req(evaluation_result())
    if (!is.null(evaluation_result()$error)) return(evaluation_result()$error)
    sprintf("Accuracy: %.2f", as.numeric(evaluation_result()$accuracy))
  })

  output$eval_f1 <- renderText({
    req(evaluation_result())
    if (!is.null(evaluation_result()$error)) return("")
    sprintf("F1 Score: %.2f", as.numeric(evaluation_result()$f1))
  })

  output$eval_retrieval <- renderText({
    req(evaluation_result())
    if (!is.null(evaluation_result()$error)) return("")
    sprintf("Retrieval Score: %.2f", as.numeric(evaluation_result()$retrieval_score))
  })

}

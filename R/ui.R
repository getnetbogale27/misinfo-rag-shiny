# UI for the misinformation detection MVP.

library(shiny)

ui <- fluidPage(
  tags$head(
    tags$link(rel = "stylesheet", type = "text/css", href = "styles.css"),
    tags$meta(name = "viewport", content = "width=device-width, initial-scale=1")
  ),

  div(
    class = "app-shell",
    div(
      class = "hero",
      h1("Misinformation Detection"),
      p("Analyze claims with retrieval-augmented evidence and confidence scoring.")
    ),

    div(
      class = "glass-card input-card",
      div(
        class = "control-grid",
        div(
          class = "control-block",
          selectInput(
            inputId = "language_mode",
            label = "Language",
            choices = c("Auto", "English", "Amharic"),
            selected = "Auto"
          )
        ),
        div(
          class = "control-block",
          textInput(
            inputId = "claim",
            label = "Enter a claim",
            placeholder = "Type a claim to analyze"
          )
        )
      ),
      div(
        class = "action-row",
        actionButton("analyze", "Analyze", class = "btn-primary-modern"),
        actionButton("run_evaluation", "Run Model Evaluation", class = "btn-secondary-modern")
      )
    ),

    div(
      class = "result-grid",
      div(
        class = "glass-card status-card",
        h3("Detected Language"),
        textOutput("language", container = span)
      ),
      div(
        class = "glass-card status-card",
        h3("Verdict"),
        textOutput("verdict", container = span)
      ),
      div(
        class = "glass-card status-card",
        h3("Confidence Score"),
        textOutput("confidence", container = span)
      )
    ),

    div(
      class = "glass-card",
      h3("Explanation"),
      textOutput("explanation")
    ),

    div(
      class = "glass-card",
      h3("Evidence"),
      uiOutput("evidence")
    ),

    div(
      class = "glass-card",
      h3("Evaluation Metrics"),
      div(
        class = "metric-stack",
        textOutput("eval_accuracy"),
        textOutput("eval_f1"),
        textOutput("eval_retrieval")
      )
    )
  )
)

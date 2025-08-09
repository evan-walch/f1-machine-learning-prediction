# F1 Next Race Predictor (VS Code Setup)

This project uses historical Formula 1 race data and machine learning to predict the outcome of upcoming races, including the likelihood of each driver failing to finish (DNF). The pipeline ingests raw F1 timing, weather, and driver performance data via the fastf1 library, processes and cleans it, then trains predictive models to forecast both race positions and DNF probabilities. The core prediction script retrieves the most recent season’s results, prepares structured features such as lap performance metrics, driver consistency, and track conditions, and feeds them into classification and regression models.

The output is a ranked table of all drivers in the next race, showing their predicted DNF probability and their expected finishing position if they complete the race. This allows fans, analysts, and bettors to assess both performance potential and risk. The system also tracks model performance, reporting metrics such as DNF AUC, position MAE, and prediction probability ranges to evaluate accuracy.

By combining live race data, statistical modeling, and predictive analytics, the project offers a data-driven lens into F1 race outcomes — turning historical trends into actionable insights for upcoming events.
